# RenderingSubsystem — Phase Progress

**Project:** `C:\Users\OS\Documents\TheGreatFoundation\RayArc\RenderingSubsystem`
**Plan:** `C:\Users\OS\.claude\plans\sharded-munching-bonbon.md`

## Completed

- ✓ Phase 1 Bootstrap
- ✓ Phase 2 Win32 + Vulkan bring-up
- ✓ Phase 3 ImGui bring-up
- ✓ Phase 4 Orbit camera + grid floor
- ✓ Phase 5 OBJ loader + ShaderBall
- ✓ Phase 6 GBuffer pass
- ✓ Phase 7 PBR direct lighting
- ✓ Phase 8 Sky + IBL
- ✓ Phase 9 NxN grid + per-submesh materials
- ✓ Phase 10 IShadowAlgorithm + PCFShadow + CSM
- ✓ Phase 11 PCSSShadow + VSMShadow
- ✓ Phase 11.5 Floor-as-GBuffer mesh
- ✓ Phase 12 MeshSDFBaker + GlobalSDF residency
- ✓ Phase 12.5 SDF Baker polish
- ✓ Phase 13 SDFConeShadow
- ✓ Phase 13.5 SDF-shadow coord-space fix
- ✓ Phase 14a IGIAlgorithm groundwork
- ✓ Phase 14b RC × Hash insert + payload + ProbeDebug
- ✓ Phase 14c Probe-count readback
- ✓ Phase 14.5 Sparse VDB storage scaffold
- ✓ Phase 15a Brick-native BVH baker
- ✓ Phase 15c Brick-aware sphere trace
- ✓ Phase 15d Sparse SDF in `lighting_sdfcone.comp`
- ✓ Phase 15e Sparse SDF in `rc_relight.comp`
- ✓ Phase 15f Sparse-aware SDFSlice debug view
- ✓ Phase 15g `.rsdfvdb` default + boot-load
- ✓ Phase 16 Tonemap (ACES) + PerfWidget

## Current status

Phase 16 built + smoke-tested green 2026-06-01 PM. Default/`-sdfcone`/`-rcgi` runs all clean, zero Vulkan validation errors.

## Deferred / parked

- **Phase 15b** — brick-native FSM baker (algorithmChoice=3). Perf-only, no visible result. Unblocks nothing.
- **Phase 17+ candidates** — multi-mesh global SDF compositor (unlocks floor shadows under SDFCone + GI bounces off non-ShaderBall geometry), aerial-perspective LUT consumer, RGB SH-L1 (if chroma bleed appears), frame-budget audit at 1920×1080.

## Known limitations

- Floor still unshadowed under SDFCone (single-mesh SDF). Fix lands with multi-mesh SDF compositor.
- RC × Hash GI bounces only off resident ShaderBall SDF. Same fix path.
- 15a brick-native baker file-size verification at 1024³ b16 still pending (handoff said skip).
