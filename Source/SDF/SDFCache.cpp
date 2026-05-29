// Source/SDF/SDFCache.cpp — Phase 12
#include "SDF/SDFCache.h"
#include "Core/Logger.h"

#include <cstdio>
#include <cstring>
#include <direct.h>     // _mkdir
#include <sys/stat.h>

namespace RS {

namespace {

constexpr char    kMagic[4]    = { 'R', 'S', 'D', 'F' };
constexpr uint32_t kVersion    = 1u;
constexpr uint32_t kFormatR16  = 0u;

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

} // namespace RS
