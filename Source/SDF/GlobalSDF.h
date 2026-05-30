// Source/SDF/GlobalSDF.h — Phase 12
// Residency table keyed by MeshHandle. Owns one 3D R16_SNORM image per resident
// SDF + one shared linear/clamp sampler. Phase 13 (SDF cone shadows) and Phase
// 14 (SDFGI) consume this table.
//
// One mesh = one resident SDF. All instances of the same mesh share it (the
// world-space query path applies each instance's inverse transform). The Bake
// window's "per-cell" UX picks a cell, resolves its mesh handle, and re-bakes
// that mesh.
//
// Lifetime: synchronous bakes run on the caller's thread. The Bake window
// spawns a worker thread that runs MeshSDFBakerBake and then calls
// GlobalSDFUploadBaked on the main thread once the bake returns.
#pragma once

#include "Core/VulkanContext.h"
#include "SDF/SDFCache.h"
#include "RS/Scene.h"
#include "Scene/MeshRegistry.h"

#include <cstdint>
#include <unordered_map>

#include <glm/glm.hpp>

namespace RS {

struct ResidentSDF {
    VkImage        Image      = VK_NULL_HANDLE;
    VkDeviceMemory Memory     = VK_NULL_HANDLE;
    VkImageView    View       = VK_NULL_HANDLE;

    glm::vec3      AABBMin    = glm::vec3(0.0f);
    glm::vec3      AABBMax    = glm::vec3(0.0f);
    uint32_t       Resolution = 0;
    float          MaxDist    = 0.0f;

    uint64_t       Hash       = 0;       // FNV-1a over {meshHandle, res, AABB, triCount}
    bool           FromCache  = false;
};

// Sparse-brick residency. Two device-local SSBOs (BrickIndex + BrickPool),
// plus a small uniform parameter block that the sampler shader include reads.
struct ResidentSparseSDF {
    VkBuffer       IndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory IndexMemory = VK_NULL_HANDLE;
    VkDeviceSize   IndexBytes  = 0;

    VkBuffer       PoolBuffer  = VK_NULL_HANDLE;
    VkDeviceMemory PoolMemory  = VK_NULL_HANDLE;
    VkDeviceSize   PoolBytes   = 0;

    glm::vec3      AABBMin     = glm::vec3(0.0f);
    glm::vec3      AABBMax     = glm::vec3(0.0f);
    uint32_t       Resolution         = 0;
    uint32_t       BrickSize          = 0;
    uint32_t       BrickGrid          = 0;
    uint32_t       OccupiedBrickCount = 0;
    float          MaxDist            = 0.0f;

    bool           FromCache = false;
};

struct GlobalSDF {
    VkSampler Sampler = VK_NULL_HANDLE;

    // 1x1x1 dummy SDF so descriptor sets always have a valid 3D image to bind
    // when no real SDF is resident. Value = +1 in normalized space (always
    // "outside"), prevents accidental black holes if the slice mode is enabled
    // before any bake.
    VkImage        DummyImage  = VK_NULL_HANDLE;
    VkDeviceMemory DummyMemory = VK_NULL_HANDLE;
    VkImageView    DummyView   = VK_NULL_HANDLE;

    std::unordered_map<MeshHandle, ResidentSDF>       Resident;
    std::unordered_map<MeshHandle, ResidentSparseSDF> ResidentSparse;

    bool Initialized = false;
};

bool GlobalSDFInitialize(GlobalSDF& g, const VulkanContext& ctx);
void GlobalSDFTerminate (GlobalSDF& g, const VulkanContext& ctx);

// Look up a mesh's resident SDF. Returns nullptr if not resident. Phase 12 +
// downstream consumers read this every frame.
const ResidentSDF* GlobalSDFGet(const GlobalSDF& g, MeshHandle mesh);

// Evict a resident SDF (frees its image / memory).
void GlobalSDFEvict(GlobalSDF& g, const VulkanContext& ctx, MeshHandle mesh);

// Upload an already-baked SDF (from MeshSDFBakerBake or SDFCacheLoad). Replaces
// any existing resident SDF for the same handle. Returns the new residency
// record (or a zero record on failure).
ResidentSDF GlobalSDFUploadBaked(GlobalSDF& g, const VulkanContext& ctx,
                                 MeshHandle mesh, const BakedSDF& baked,
                                 bool fromCache);

// Convenience: try to load Cache/SDF/<sourceBase>_<res>.rsdf and upload it.
// Returns true if a residency was created. Used by Main.cpp's auto-load after
// InscribeMesh.
bool GlobalSDFTryLoadFromCache(GlobalSDF& g, const VulkanContext& ctx,
                               MeshHandle mesh, const char* sourcePath,
                               uint32_t resolution);

// ---- sparse VDB-style residency --------------------------------------------

const ResidentSparseSDF* GlobalSDFGetSparse(const GlobalSDF& g, MeshHandle mesh);

void GlobalSDFEvictSparse(GlobalSDF& g, const VulkanContext& ctx, MeshHandle mesh);

// Upload a sparse-baked SDF. Replaces any existing sparse residency for the
// mesh. The dense residency for the same mesh, if any, is untouched — they
// can coexist (consumer picks which to sample).
ResidentSparseSDF GlobalSDFUploadSparse(GlobalSDF& g, const VulkanContext& ctx,
                                       MeshHandle mesh,
                                       const BakedSparseSDF& baked,
                                       bool fromCache);

// Write the currently-resident SDF for `mesh` back to disk at the canonical
// Cache/SDF path. No-op if nothing is resident. Returns true on success.
bool GlobalSDFSaveResidentToCache(const GlobalSDF& g, MeshHandle mesh,
                                  const char* sourcePath);

} // namespace RS
