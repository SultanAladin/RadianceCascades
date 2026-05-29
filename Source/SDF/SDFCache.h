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

// Derive a cache file path from a mesh's source filename.
//   "../ShaderBall-master/.../shaderBall.obj", 64 → "Cache/SDF/shaderBall_64.rsdf"
std::string SDFCacheDerivePath(const char* sourcePath, uint32_t resolution);

// Make sure the Cache/SDF directory exists (best-effort; returns true if it
// existed or was created).
bool SDFCacheEnsureDirectory(const char* path);

// Read a .rsdf file. Returns true on success. Validates magic + version.
bool SDFCacheLoad(const char* path, BakedSDF& out);

// Write a .rsdf file. Returns true on success. Creates the parent dir.
bool SDFCacheSave(const char* path, const BakedSDF& in);

} // namespace RS
