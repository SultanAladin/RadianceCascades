# Sparse Radiance Cascades 3D — Faithful Rewrite (handoff)

**Date:** 2026-06-07
**Status:** Builds clean (shaders + C++ link green). NOT yet visually verified — needs
you to run `RenderingSubsystem.exe -rcgi` and look at it (I can't see the GUI).

## What changed and why

The previous implementation was not radiance cascades — it was a world-space SH probe
hash using the cascade vocabulary. Three core invariants were violated (confirmed against
Sannikov's paper + m4xc fundamentals + IceKontroI sparse_rc_2d):

1. **Constant angular resolution** — every cascade traced the same ray count. Canonical RC
   needs dirs(c) = D0·4^c (angular res rises as spatial res falls).
2. **Non-disjoint intervals** — every cascade marched [0, 16m]. Canonical needs disjoint
   contiguous intervals [s0·(2^c−1), s0·(2^(c+1)−1)).
3. **No real merge** — compose picked ONE cascade by distance (the flicker source) and the
   miss path grabbed a single DC term. Canonical merges all cascades top-down into c0.

These are why per-buffer fixes couldn't fix the look, and why it read as "a flickering
grid, some parts brighter" rather than GI.

## New architecture (faithful to the paper)

- **Storage:** per-cascade sparse sub-tables in one key buffer. Cascade c:
  - probe pitch = s0·2^c (world-anchored — probes do NOT follow the camera; this kills
    the motion flicker)
  - slots(c) = BaseSlots>>2c, dirs(c) = D0·4^c (D0=16, octahedral res 4·2^c)
  - payload texel = vec4(rgb radiance, β) where β = transparency (1 escaped, 0 hit)
  - payload block = BaseSlots·D0 texels per cascade (constant — the paper's memory bound)
- **Passes (all one frame, barriers between):**
  1. `rc_insert` — screen depth + eye volume → insert c0 cell + 2×2×2 neighbourhood +
     parent chains at every cascade (bottom-up task generation, IceKontroI style).
  2. `rc_relight` — one dispatch per cascade, **top-down c=N−1…0**. Marches the disjoint
     interval; hit → sun·NdotL·vis·albedo (β=0); miss → merge parent's 2×2×2 trilinear
     probes, averaging the 2×2 oct texel block (β=1). Top cascade miss → sky.
  3. `rc_resolve` — integrate c0's merged directional radiance → RGB SH-L1 (per probe).
  4. `gi_compose` — trilinear gather of 8 c0 probes' SH, eval at pixel normal, add to
     LightHDR. Reads ONLY c0 (which now holds the full [0,∞) field).
- **No jitter, no temporal blend** — directions are fixed octahedral texels every frame,
  so output is deterministic and noiseless (RC's whole point). Per-frame buffers + the
  frame fence mean no cross-frame races and no payload clears needed.

## Files touched

- `Shaders/include/rc_hash.glsl` — rewritten: cascade geometry, interval math, sub-table
  offsets, octahedral encode/decode, merge-ready SH helpers.
- `Shaders/rc_insert.comp` — rewritten: parent-chain insertion.
- `Shaders/rc_relight.comp` — rewritten: per-cascade disjoint-interval trace + parent merge.
- `Shaders/rc_resolve.comp` — NEW: c0 dirs → SH-L1.
- `Shaders/gi_compose.comp` — rewritten: c0-only trilinear gather; debug views updated.
- `Source/GI/RadianceCascadeHash.{h,cpp}` — rewritten: per-cascade sub-tables, resolve
  buffer (binding 4), per-frame buffers, new 240-byte params.
- `Source/GI/RadianceCascadeGI.{h,cpp}` — rewritten frame protocol: insert → relight
  top-down loop → resolve; new FillParams; resolve pipeline; UI updated.

## Defaults

BaseLog2 = 14 (16384 c0 slots), Cascades = 4, D0 = 16, s0 = 1.0 m.
Payload ≈ 4 MB/cascade × 4 × 2 frames ≈ 32 MB. Tune `GISettings::HashLog2` (12–18) and
`CascadeCount` (apply on algo restart).

## What to check when you run `-rcgi`

1. **No flicker** — world-anchored probes + single-frame pipeline should be rock stable.
2. **Debug view 2 (Cascades)** — should show concentric interval shells around the camera,
   not random hashing.
3. **Debug view 4 (Irradiance only)** — smooth coloured bounce, brighter near lit SDF
   surfaces, dimmer in shadow. This is the real GI signal.
4. **Lit view** — indirect light filling shadowed areas, colour bleed from the SDF mesh
   albedo onto neighbours.
5. **Zero Vulkan validation errors** in the console / smoke logs.

## Known risk areas (where to look if it's wrong)

- **scaleLocalPerWorld / world↔local**: relight marches in SDF-local (cm) space; if the
  ShaderBall SDF mapping is off, every ray misses → only sky/ambient shows (no bounce).
  Check `MiscParams.y` and `WorldToLocal*` in FillParams.
- **Interval units**: t0/t1 are world metres × scaleLocalPerWorld → local. If bounce is
  absent but debug view 2 looks right, suspect the interval-to-local remap.
- **Parent merge addressing**: `RC_MergeFromParent` assumes parent oct res = 2× child.
  Holds for D0=16 (res 4,8,16,32…). If you change D0 off a power-of-4, revisit.
- **Insert eye-volume dispatch**: now 8×8×side groups (was side×side×side); the shader
  guards out-of-range, but if close-up GI has holes, bump the eye radius R.
