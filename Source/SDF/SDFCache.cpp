// Source/SDF/SDFCache.cpp — Phase 12
#include "SDF/SDFCache.h"
#include "Core/Logger.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <direct.h>     // _mkdir
#include <sys/stat.h>

namespace RS {

namespace {

constexpr char    kMagic[4]    = { 'R', 'S', 'D', 'F' };
constexpr uint32_t kVersion    = 1u;
constexpr uint32_t kFormatR16  = 0u;

constexpr char    kSparseMagic[4] = { 'R', 'S', 'D', 'V' };
constexpr uint32_t kSparseVersion = 1u;

#pragma pack(push, 1)
struct Header {
    char     Magic[4];
    uint32_t Version;
    uint32_t Resolution;
    float    AABBMin[3];
    float    AABBMax[3];
    uint32_t Format;
    float    MaxDist;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 44, "rsdf header must be 44 bytes");

#pragma pack(push, 1)
struct SparseHeader {
    char     Magic[4];           // 'RSDV'
    uint32_t Version;            // kSparseVersion
    uint32_t Resolution;         // dense voxel resolution (cube)
    uint32_t BrickSize;          // 4, 8, or 16
    uint32_t BrickGrid;          // ceil(Resolution / BrickSize)
    uint32_t OccupiedBrickCount; // entries in BrickPool
    uint32_t Format;             // 0 = R16_SNORM
    float    AABBMin[3];
    float    AABBMax[3];
    float    MaxDist;
    uint32_t Reserved[3];        // pad to 80 bytes for future use
};
#pragma pack(pop)
static_assert(sizeof(SparseHeader) == 68, "rsdfvdb header must be 68 bytes");

// Strip directory + extension from a path. Returns "shaderBall" for
// "../foo/bar/shaderBall.obj".
std::string DeriveBaseName(const char* sourcePath) {
    std::string s(sourcePath ? sourcePath : "");
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) s = s.substr(slash + 1);
    const size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
    if (s.empty()) s = "mesh";
    return s;
}

bool MakeDir(const char* path) {
    struct _stat st{};
    if (_stat(path, &st) == 0) return (st.st_mode & _S_IFDIR) != 0;
    return _mkdir(path) == 0;
}

} // namespace

std::string SDFCacheDerivePath(const char* sourcePath, uint32_t resolution) {
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%u.rsdf", resolution);
    return std::string("Cache/SDF/") + DeriveBaseName(sourcePath) + tail;
}

std::string SDFCacheDeriveSparsePath(const char* sourcePath,
                                     uint32_t resolution,
                                     uint32_t brickSize) {
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%u_b%u.rsdfvdb", resolution, brickSize);
    return std::string("Cache/SDF/") + DeriveBaseName(sourcePath) + tail;
}

bool SDFCacheEnsureDirectory(const char* /*pathHint*/) {
    // Always Cache/SDF in v1.
    MakeDir("Cache");
    return MakeDir("Cache/SDF");
}

bool SDFCacheLoad(const char* path, BakedSDF& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) { std::fclose(f); return false; }
    if (std::memcmp(h.Magic, kMagic, 4) != 0 || h.Version != kVersion ||
        h.Format != kFormatR16 || h.Resolution == 0 || h.Resolution > 512) {
        RS_LOG_ERROR("SDFCache: %s rejected (magic/version/format/res)", path);
        std::fclose(f);
        return false;
    }

    out.Resolution = h.Resolution;
    out.AABBMin    = glm::vec3(h.AABBMin[0], h.AABBMin[1], h.AABBMin[2]);
    out.AABBMax    = glm::vec3(h.AABBMax[0], h.AABBMax[1], h.AABBMax[2]);
    out.MaxDist    = h.MaxDist;

    const size_t voxelCount = static_cast<size_t>(h.Resolution) *
                              static_cast<size_t>(h.Resolution) *
                              static_cast<size_t>(h.Resolution);
    out.Voxels.resize(voxelCount);
    const size_t got = std::fread(out.Voxels.data(), sizeof(int16_t), voxelCount, f);
    std::fclose(f);

    if (got != voxelCount) {
        RS_LOG_ERROR("SDFCache: %s payload short (got %zu / %zu)", path, got, voxelCount);
        return false;
    }
    return true;
}

bool SDFCacheSave(const char* path, const BakedSDF& in) {
    if (in.Resolution == 0 || in.Voxels.empty() || in.MaxDist <= 0.0f) return false;

    SDFCacheEnsureDirectory(path);

    FILE* f = std::fopen(path, "wb");
    if (!f) {
        RS_LOG_ERROR("SDFCache: cannot create %s", path);
        return false;
    }

    Header h{};
    std::memcpy(h.Magic, kMagic, 4);
    h.Version    = kVersion;
    h.Resolution = in.Resolution;
    h.AABBMin[0] = in.AABBMin.x; h.AABBMin[1] = in.AABBMin.y; h.AABBMin[2] = in.AABBMin.z;
    h.AABBMax[0] = in.AABBMax.x; h.AABBMax[1] = in.AABBMax.y; h.AABBMax[2] = in.AABBMax.z;
    h.Format     = kFormatR16;
    h.MaxDist    = in.MaxDist;

    if (std::fwrite(&h, sizeof(h), 1, f) != 1) { std::fclose(f); return false; }
    const size_t put = std::fwrite(in.Voxels.data(), sizeof(int16_t),
                                   in.Voxels.size(), f);
    std::fclose(f);

    if (put != in.Voxels.size()) {
        RS_LOG_ERROR("SDFCache: %s payload short on write", path);
        return false;
    }
    RS_LOG_INFO("SDFCache: wrote %s (%u^3, %zu KB)", path, in.Resolution,
                (in.Voxels.size() * sizeof(int16_t)) / 1024);
    return true;
}

// ---- sparse VDB-style container --------------------------------------------

BakedSparseSDF SDFCacheBuildSparse(const BakedSDF& dense,
                                   uint32_t brickSize,
                                   float epsilon) {
    BakedSparseSDF out{};
    if (dense.Resolution == 0 || dense.Voxels.empty()) return out;
    if (brickSize != 4u && brickSize != 8u && brickSize != 16u) {
        RS_LOG_ERROR("SDFCache: BuildSparse rejects brickSize=%u (must be 4/8/16)", brickSize);
        return out;
    }

    out.AABBMin    = dense.AABBMin;
    out.AABBMax    = dense.AABBMax;
    out.Resolution = dense.Resolution;
    out.MaxDist    = dense.MaxDist;
    out.BrickSize  = brickSize;
    out.BrickGrid  = (dense.Resolution + brickSize - 1u) / brickSize;

    const uint32_t res        = dense.Resolution;
    const uint32_t bg         = out.BrickGrid;
    const uint32_t bs         = brickSize;
    const uint32_t bsCubed    = bs * bs * bs;
    const size_t   indexCount = static_cast<size_t>(bg) * bg * bg;
    out.BrickIndex.assign(indexCount, kSparseBrickIndexAllOutside);

    // R16_SNORM cutoff: anything outside [-epsilon, +epsilon] (in normalized
    // space) is treated as either certainly inside or certainly outside the
    // surface band. epsilon defaults to 1 SNORM step so we only fold bricks
    // that are *unambiguously* one-sided.
    const float threshold = std::max(0.0f, epsilon) * dense.MaxDist;

    // Visit each brick. For mixed bricks, allocate a slot in the pool.
    out.BrickPool.reserve(static_cast<size_t>(bg) * bg * bg / 8 * bsCubed); // optimistic
    uint32_t poolBricks = 0;

    auto denseIdx = [res](uint32_t x, uint32_t y, uint32_t z) -> size_t {
        return (static_cast<size_t>(z) * res + y) * res + x;
    };

    for (uint32_t bz = 0; bz < bg; ++bz) {
    for (uint32_t by = 0; by < bg; ++by) {
    for (uint32_t bx = 0; bx < bg; ++bx) {
        // Classify the brick by scanning its voxels (clamping at the dense
        // resolution edge — last brick may be partial).
        const uint32_t x0 = bx * bs, x1 = std::min(x0 + bs, res);
        const uint32_t y0 = by * bs, y1 = std::min(y0 + bs, res);
        const uint32_t z0 = bz * bs, z1 = std::min(z0 + bs, res);

        bool allOutside = true;
        bool allInside  = true;
        // Sample-decoded R16 → float in world units (signed distance).
        for (uint32_t z = z0; z < z1 && (allOutside || allInside); ++z) {
        for (uint32_t y = y0; y < y1 && (allOutside || allInside); ++y) {
        for (uint32_t x = x0; x < x1 && (allOutside || allInside); ++x) {
            const float d = static_cast<float>(dense.Voxels[denseIdx(x, y, z)])
                          * (dense.MaxDist / 32767.0f);
            if (d <  threshold) allOutside = false;
            if (d > -threshold) allInside  = false;
        }}}

        const size_t ii = (static_cast<size_t>(bz) * bg + by) * bg + bx;
        if (allOutside) {
            out.BrickIndex[ii] = kSparseBrickIndexAllOutside;
            continue;
        }
        if (allInside) {
            out.BrickIndex[ii] = kSparseBrickIndexAllInside;
            continue;
        }

        // Mixed — allocate a brick slot. Copy bs^3 voxels; out-of-bounds
        // voxels (when brick straddles the resolution edge) clamp to the last
        // in-range voxel along that axis so trilinear sampling stays smooth.
        out.BrickIndex[ii] = poolBricks;
        ++poolBricks;
        const size_t poolBase = out.BrickPool.size();
        out.BrickPool.resize(poolBase + bsCubed);
        for (uint32_t lz = 0; lz < bs; ++lz) {
        for (uint32_t ly = 0; ly < bs; ++ly) {
        for (uint32_t lx = 0; lx < bs; ++lx) {
            const uint32_t sx = std::min(x0 + lx, res - 1u);
            const uint32_t sy = std::min(y0 + ly, res - 1u);
            const uint32_t sz = std::min(z0 + lz, res - 1u);
            const size_t localIdx = (static_cast<size_t>(lz) * bs + ly) * bs + lx;
            out.BrickPool[poolBase + localIdx] = dense.Voxels[denseIdx(sx, sy, sz)];
        }}}
    }}}

    out.OccupiedBrickCount = poolBricks;
    return out;
}

bool SDFCacheLoadSparse(const char* path, BakedSparseSDF& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    SparseHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) { std::fclose(f); return false; }
    if (std::memcmp(h.Magic, kSparseMagic, 4) != 0 || h.Version != kSparseVersion ||
        h.Format != kFormatR16 || h.Resolution == 0 || h.Resolution > 1024 ||
        (h.BrickSize != 4u && h.BrickSize != 8u && h.BrickSize != 16u)) {
        RS_LOG_ERROR("SDFCache: %s rejected (magic/version/format/res/brick)", path);
        std::fclose(f);
        return false;
    }
    const uint32_t expectedGrid = (h.Resolution + h.BrickSize - 1u) / h.BrickSize;
    if (h.BrickGrid != expectedGrid) {
        RS_LOG_ERROR("SDFCache: %s BrickGrid mismatch (got %u, expected %u)",
                     path, h.BrickGrid, expectedGrid);
        std::fclose(f);
        return false;
    }

    out.Resolution         = h.Resolution;
    out.BrickSize          = h.BrickSize;
    out.BrickGrid          = h.BrickGrid;
    out.OccupiedBrickCount = h.OccupiedBrickCount;
    out.AABBMin            = glm::vec3(h.AABBMin[0], h.AABBMin[1], h.AABBMin[2]);
    out.AABBMax            = glm::vec3(h.AABBMax[0], h.AABBMax[1], h.AABBMax[2]);
    out.MaxDist            = h.MaxDist;

    const size_t indexCount = static_cast<size_t>(h.BrickGrid) * h.BrickGrid * h.BrickGrid;
    out.BrickIndex.resize(indexCount);
    if (std::fread(out.BrickIndex.data(), sizeof(uint32_t), indexCount, f) != indexCount) {
        std::fclose(f);
        return false;
    }
    const size_t poolVoxels = static_cast<size_t>(h.OccupiedBrickCount)
                            * h.BrickSize * h.BrickSize * h.BrickSize;
    out.BrickPool.resize(poolVoxels);
    const size_t got = std::fread(out.BrickPool.data(), sizeof(int16_t), poolVoxels, f);
    std::fclose(f);
    if (got != poolVoxels) {
        RS_LOG_ERROR("SDFCache: %s pool short (got %zu / %zu)", path, got, poolVoxels);
        return false;
    }
    return true;
}

bool SDFCacheSaveSparse(const char* path, const BakedSparseSDF& in) {
    if (in.Resolution == 0 || in.BrickIndex.empty() || in.MaxDist <= 0.0f) return false;
    if (in.BrickSize != 4u && in.BrickSize != 8u && in.BrickSize != 16u) return false;

    SDFCacheEnsureDirectory(path);

    FILE* f = std::fopen(path, "wb");
    if (!f) {
        RS_LOG_ERROR("SDFCache: cannot create %s", path);
        return false;
    }

    SparseHeader h{};
    std::memcpy(h.Magic, kSparseMagic, 4);
    h.Version            = kSparseVersion;
    h.Resolution         = in.Resolution;
    h.BrickSize          = in.BrickSize;
    h.BrickGrid          = in.BrickGrid;
    h.OccupiedBrickCount = in.OccupiedBrickCount;
    h.Format             = kFormatR16;
    h.AABBMin[0] = in.AABBMin.x; h.AABBMin[1] = in.AABBMin.y; h.AABBMin[2] = in.AABBMin.z;
    h.AABBMax[0] = in.AABBMax.x; h.AABBMax[1] = in.AABBMax.y; h.AABBMax[2] = in.AABBMax.z;
    h.MaxDist            = in.MaxDist;

    if (std::fwrite(&h, sizeof(h), 1, f) != 1) { std::fclose(f); return false; }
    if (std::fwrite(in.BrickIndex.data(), sizeof(uint32_t), in.BrickIndex.size(), f)
        != in.BrickIndex.size()) { std::fclose(f); return false; }
    if (std::fwrite(in.BrickPool.data(), sizeof(int16_t), in.BrickPool.size(), f)
        != in.BrickPool.size()) { std::fclose(f); return false; }
    std::fclose(f);

    const size_t totalBytes = sizeof(SparseHeader)
                            + in.BrickIndex.size() * sizeof(uint32_t)
                            + in.BrickPool.size() * sizeof(int16_t);
    RS_LOG_INFO("SDFCache: wrote %s (%u^3 @ b%u, %u/%u bricks occupied, %zu KB)",
                path, in.Resolution, in.BrickSize, in.OccupiedBrickCount,
                static_cast<uint32_t>(in.BrickIndex.size()),
                totalBytes / 1024);
    return true;
}

} // namespace RS
