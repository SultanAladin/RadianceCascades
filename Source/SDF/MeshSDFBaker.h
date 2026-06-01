// Source/SDF/MeshSDFBaker.h — Phase 12
// CPU brute-force mesh-to-SDF baker. BVH over triangles + point-to-triangle
// exact distance + face-normal sign at closest tri. Parallelised over Z slices.
//
// Output is a 64^3 (or other res) int16 voxel grid encoding signed distance
// scaled by MaxDist (R16_SNORM convention: int16 / 32767 ∈ [-1, 1] × MaxDist).
// The SDF AABB pads the mesh AABB by 5% so the gradient field has room to fan
// out past the surface.
#pragma once

#include "Core/VulkanContext.h"
#include "SDF/SDFCache.h"
#include "Scene/MeshRegistry.h"

#include <atomic>
#include <cstdint>

namespace RS {

struct BakeProgress {
    std::atomic<float> Fraction      { 0.0f };  // 0..1 over Z slices completed
    std::atomic<bool>  Cancel        { false };
    std::atomic<double> DurationMs   { 0.0 };   // populated on completion
    std::atomic<uint32_t> TriangleCount { 0 };  // populated once the BVH is built
};

// Synchronous brute-force bake. Caller may run this on a worker thread; pass a
// BakeProgress to surface a progress bar. Returns an empty BakedSDF on
// cancel/failure.
//
// Reads vertex + index data from the GpuMesh's host-visible buffers via
// vkMapMemory — no extra CPU copy is kept around.
BakedSDF MeshSDFBakerBake(const VulkanContext& ctx,
                          const GpuMesh&       mesh,
                          uint32_t             resolution,
                          BakeProgress*        progress,
                          int                  algorithmChoice = 0);

// Phase 15a: brick-native sparse bake. Produces BakedSparseSDF directly with
// peak RAM ≈ occupiedBricks × brickSize^3, NOT res^3 — unblocks 1024^3 bakes
// that the dense path can't fit. Output is bit-identical to
// SDFCacheBuildSparse(MeshSDFBakerBake(...), brickSize) (same classify
// threshold, same trailing-brick clamp). brickSize must be 4, 8, or 16.
// Currently only algorithmChoice = 0 (Exact BVH) is wired; FSM is Phase 15b.
BakedSparseSDF MeshSDFBakerBakeSparse(const VulkanContext& ctx,
                                      const GpuMesh&       mesh,
                                      uint32_t             resolution,
                                      uint32_t             brickSize,
                                      BakeProgress*        progress,
                                      int                  algorithmChoice = 0);

} // namespace RS
