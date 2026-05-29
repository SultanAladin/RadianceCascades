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
                          BakeProgress*        progress);

} // namespace RS
