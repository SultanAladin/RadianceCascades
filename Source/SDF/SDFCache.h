// Source/SDF/SDFCache.h — Phase 12
// Raw binary .rsdf reader/writer. Header is 40 bytes, payload is res^3 int16
// voxels (R16_SNORM encoding of distance / maxDist).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace RS {

struct BakedSDF {
    glm::vec3            AABBMin    = glm::vec3(0.0f);
    glm::vec3            AABBMax    = glm::vec3(0.0f);
    uint32_t             Resolution = 0;
    float                MaxDist    = 0.0f;   // scale for R16_SNORM decode
    std::vector<int16_t> Voxels;              // size = Resolution^3
};

// Sparse "VDB-style" baked SDF (.rsdfvdb). 2-level structure:
//   - top-level dense index grid (one uint32 per brick coord)
//   - leaf brick pool (contiguous int16 voxels, only mixed bricks allocated)
//
// Empty bricks compress to sentinel index values (no payload):
constexpr uint32_t kSparseBrickIndexAllOutside = 0xFFFFFFFEu;  // implicit +MaxDist
constexpr uint32_t kSparseBrickIndexAllInside  = 0xFFFFFFFFu;  // implicit -MaxDist
// Any other value is a 0-based offset into the brick pool (in *bricks*, not voxels).

struct BakedSparseSDF {
    glm::vec3              AABBMin     = glm::vec3(0.0f);
    glm::vec3              AABBMax     = glm::vec3(0.0f);
    uint32_t               Resolution  = 0;   // dense voxel resolution (cube)
    uint32_t               BrickSize   = 8;   // 4, 8, or 16
    uint32_t               BrickGrid   = 0;   // Resolution / BrickSize (ceil-rounded)
    float                  MaxDist     = 0.0f;
    std::vector<uint32_t>  BrickIndex;        // size = BrickGrid^3
    std::vector<int16_t>   BrickPool;         // size = OccupiedBrickCount * BrickSize^3
    uint32_t               OccupiedBrickCount = 0;
};

// Derive a cache file path from a mesh's source filename.
//   "../ShaderBall-master/.../shaderBall.obj", 64 → "Cache/SDF/shaderBall_64.rsdf"
std::string SDFCacheDerivePath(const char* sourcePath, uint32_t resolution);

// Sparse variant: "Cache/SDF/shaderBall_64_b8.rsdfvdb"
std::string SDFCacheDeriveSparsePath(const char* sourcePath,
                                     uint32_t resolution,
                                     uint32_t brickSize);

// Make sure the Cache/SDF directory exists (best-effort; returns true if it
// existed or was created).
bool SDFCacheEnsureDirectory(const char* path);

// Read a .rsdf file. Returns true on success. Validates magic + version.
bool SDFCacheLoad(const char* path, BakedSDF& out);

// Write a .rsdf file. Returns true on success. Creates the parent dir.
bool SDFCacheSave(const char* path, const BakedSDF& in);

// Sparse .rsdfvdb load/save. Validates magic 'RSDV' + version.
bool SDFCacheLoadSparse(const char* path, BakedSparseSDF& out);
bool SDFCacheSaveSparse(const char* path, const BakedSparseSDF& in);

// Convert a fully-baked dense SDF into a sparse one. `brickSize` must be 4, 8,
// or 16. Bricks whose voxels all exceed +epsilon * MaxDist become AllOutside
// sentinels; all less than -epsilon * MaxDist become AllInside; everything else
// is allocated to the pool. `epsilon` defaults to 1/32767 (one R16_SNORM step).
BakedSparseSDF SDFCacheBuildSparse(const BakedSDF& dense,
                                   uint32_t brickSize,
                                   float epsilon = 1.0f / 32767.0f);

} // namespace RS
