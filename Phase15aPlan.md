# Phase 15a — Brick-native BVH baker

## Context

`MeshSDFBakerBake` (`Source/SDF/MeshSDFBaker.cpp`) is **dense**: it allocates `res³ × int16` voxels (~2 GB at 1024³) before the panel optionally packs them sparse via `SDFCacheBuildSparse`. Phase 14.5 stood up the sparse `BakedSparseSDF` format end-to-end (storage, file, GPU residency), but bake-time RAM is still `O(res³)` — 1024³ OOMs an 8 GB box.

Phase 15a replaces this with a **brick-native** baker that produces `BakedSparseSDF` directly. Peak RAM becomes `O(occupiedBricks × brickSize³)`. The dense path stays — it's still load-bearing for legacy `.rsdf` files and the Phase 15f dense-vs-sparse debug toggle.

## Algorithm

1. Reuse the existing in-file `TriBVH` (lines 41–229). Add **one** new private walk `MinDistSqToBrick(brickMin, brickMax)` that prunes nodes by AABB-vs-AABB and leaves by `Tri::AABBMin/Max`-vs-brick — AABB-vs-AABB is conservative, only over-marks bricks as mixed, never misses a band brick.
2. Compute SDF bounds (5% pad, `MaxDist = 1.5 × diagonal`) — identical to dense path.
3. For each brick coord `(bx,by,bz)`:
   - Build the brick AABB (clamp the trailing brick to `AABBMax`).
   - If `sqrt(MinDistSqToBrick) > MaxDist − threshold` (where `threshold = MaxDist / 32767` — one R16_SNORM step, matching `SDFCacheBuildSparse` line 188): the brick is one-sided. Disambiguate sign by `bvh.ClosestPoint(brickCentre, …)` + face-normal dot. Write `kSparseBrickIndexAllOutside` or `kSparseBrickIndexAllInside` into `BrickIndex[flat]`.
   - Otherwise mixed: allocate `brickSize³` int16, run the existing per-voxel ClosestPoint + sign + R16_SNORM encode (lines 372–384 copied verbatim). Voxels with global coord ≥ res clamp to the last in-range coord (matches `SDFCacheBuildSparse` lines 239–241).
4. **Classify-gate correctness**: every voxel-centre inside the brick has distance ≥ `aabbDist` to the nearest tri. If `aabbDist > MaxDist − threshold`, every voxel saturates to ±32767 — exactly the band `SDFCacheBuildSparse` folds to sentinels. **No misclassification vs. dense-then-pack.**

## Concurrency

- Parallelise over bricks (replaces Z-slice parallelism). Atomic `nextBrick` ticket, same shape as the existing `nextZ` worker at lines 351–399. `min(hwc, 16)` workers.
- **Scheme (a)**: each worker accumulates mixed-brick voxels into a per-worker `std::vector<int16_t> localPool` + a `std::vector<std::pair<uint32_t,uint32_t>> localMap` (flatBrickIdx → localOffset). Sentinel bricks write `BrickIndex[flat]` directly (distinct slots, race-free).
- After join: sort all localMap entries by flatBrickIdx (cheap, matches `SDFCacheBuildSparse`'s (bz,by,bx) ordering for byte-parity verification), assign pool slots, `memcpy` into `out.BrickPool`. Set `out.OccupiedBrickCount`.
- Mirror `progress->TriangleCount` (after BVH build), `Cancel` checks per brick, `Fraction = doneBricks/totalBricks`, `DurationMs` at end, final `RS_LOG_INFO`.

## Implementation order

**Step A — Refactor only, no behaviour change.** In `Source/SDF/MeshSDFBaker.cpp`:
- Extract `BuildTris(verts, indices, &tris)` from lines 287–314.
- Extract `ComputeSdfBounds(mesh, res, &min, &max, &voxelSize, &maxDist)` from lines 330–342.
- Optionally extract `PrepareBake(ctx, mesh, res, progress, &tris, &bvh, &bounds)` calling those two + the existing `ReadMeshCpuData`.
- Rewire `MeshSDFBakerBake` to use them. Build + 64³ dense bake byte-identical vs. HEAD.

**Step B — New BVH walk.** Add `static AABBvsAABBDistSq(...)` (~6 LOC) + private `DescendBrickClassify(node, brickMin, brickMax, &outDistSq)` + public `float MinDistSqToBrick(brickMin, brickMax) const` on `TriBVH`. All in the anon namespace.

**Step C — New entry point.** Add `BakedSparseSDF MeshSDFBakerBakeSparse(ctx, mesh, resolution, brickSize, progress, algorithmChoice = 0)`. Reject `brickSize ∉ {4,8,16}` and `algorithmChoice != 0` (FSM is 15b) with `RS_LOG_WARN`. Body: validate → `PrepareBake` → init `BrickGrid = ceil(res/bs)`, `BrickIndex.assign(bg³, kSparseBrickIndexAllOutside)` → spawn workers → after-join sort+pack → set `OccupiedBrickCount` + `DurationMs` + log.

**Step D — Header.** Declare the new function in `Source/SDF/MeshSDFBaker.h` next to the existing one (line 33 area). `BakedSparseSDF` is already pulled in via `SDFCache.h` include (line 12).

**Step E — Panel wiring.** In `Source/UI/SdfBakerPanel.cpp`:
- Add `std::shared_ptr<BakedSparseSDF> PendingUploadSparse` to `SdfBakerState`.
- In `KickBake` (~line 397): branch on `s.UseSparse`. Sparse path: allocate `PendingUploadSparse`, capture `brickSize`, worker calls `MeshSDFBakerBakeSparse(...)`. Dense path unchanged.
- In `FinaliseBake` (~line 440): on sparse path, skip `GlobalSDFUploadBaked` + `SDFCacheSave` + `SDFCacheBuildSparse`; call `SDFCacheSaveSparse` + `GlobalSDFUploadSparse` directly with `*s.PendingUploadSparse`; populate the sparse stats rows.
- Gate `RegenerateSlice` (~line 369) on dense being present — slice preview will be sparse-aware in Phase 15f. Until then, sparse-only bakes simply leave the preview blank.

**Step F — Build script.** No change. No new TU.

## Critical files

- `Source/SDF/MeshSDFBaker.cpp` (main)
- `Source/SDF/MeshSDFBaker.h` (declare)
- `Source/UI/SdfBakerPanel.cpp` (panel wiring)
- (Read-only references) `Source/SDF/SDFCache.{h,cpp}` for `BakedSparseSDF`, sentinels, classify epsilon.

## Reused helpers / call sites

- `TriBVH::ClosestPoint` (MeshSDFBaker.cpp:58) — sign disambiguation per sentinel brick + every voxel of every mixed brick.
- `TriBVH::ClosestPointOnTriangle` (MeshSDFBaker.cpp:142) — already public.
- `ReadMeshCpuData` (MeshSDFBaker.cpp:234) — anon-namespace free function, already shared.
- `kSparseBrickIndexAllOutside` / `kSparseBrickIndexAllInside` (SDFCache.h:27/28).
- `SDFCacheSaveSparse` (SDFCache.cpp:298) and `GlobalSDFUploadSparse` (GlobalSDF.h:114) — existing sparse output sinks; sparse path calls them directly without going through `SDFCacheBuildSparse`.
- Progress atomics: `Fraction`, `Cancel`, `DurationMs`, `TriangleCount` (MeshSDFBaker.h:20–25).

## Verification

1. `Build.bat` exits 0 — no new TU, no script change.
2. Boot the exe; existing `Cache/SDF/shaderBall_64.rsdf` still auto-loads via the dense path. Smoke-clean (zero validation errors).
3. **Step A regression**: with `s.UseSparse=false`, bake ShaderBall at 64³; resulting `.rsdf` byte-identical to a HEAD bake.
4. **Sparse parity**: with `s.UseSparse=true`, brickSize=8, bake ShaderBall at 64³. Compare the new-baker `.rsdfvdb` to a dense-then-pack reference. Expected: `OccupiedBrickCount` matches exactly; `BrickIndex` matches exactly; `BrickPool` matches byte-for-byte.
5. **Mid-range**: bake at 256³, brickSize=8. Sparse-path peak RSS dominated by occupied bricks × 1 KB.
6. **Headline test — 1024³**: bake ShaderBall at 1024³, brickSize=16. Dense path would OOM at ~8 GB; new path completes with peak RSS in the single-digit GB at most.
7. **Cancel**: start a 1024³ bake, flip Cancel. Worker exits within one brick and returns an empty `BakedSparseSDF`.
8. **Non-divisible resolution**: bake 100³ brickSize=8. Trailing-brick voxels clamp; sentinel decisions match `SDFCacheBuildSparse`.
9. Boot smoke (no `-rcgi`, with `-sdfcone`): no Vulkan validation errors.

## Carry-forward into Phase 15b–g

- 15b: brick-native FSM (`algorithmChoice=3`).
- 15c: brick-aware sphere trace in `Shaders/include/sdf_sparse.glsl` — `AllOutside` bricks return analytic ray-AABB exit distance.
- 15d/e: wire sparse SSBOs into `lighting_sdfcone.comp` and `rc_relight.comp`.
- 15f: sparse-aware `SDFSlice` debug view + dense-vs-sparse toggle.
- 15g: flip default to `UseSparse=true`; boot prefers `.rsdfvdb`.
