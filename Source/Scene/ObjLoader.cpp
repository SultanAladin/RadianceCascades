// Source/Scene/ObjLoader.cpp — Phase 5
// User constraint (2026-05-28): NO tinyobjloader. This is a single-file OBJ
// parser tuned for ShaderBall and similar clean-room meshes. Limitations:
//   * Only handles 'v', 'vn', 'f', 'g'/'o' (and skips '#'/'mtllib'/'usemtl'/'s'/'vt').
//   * 'f' faces with N>3 verts are fan-triangulated.
//   * Negative (relative) indices ARE supported.
//   * Texture coords are parsed-and-dropped; v1 has no UV pipeline.
//   * If a face vertex lacks a normal index, the parser fails — flat shading
//     is required for the throwaway forward pass. ShaderBall has full normals.
#include "Scene/ObjLoader.h"
#include "Core/Logger.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace RS {

namespace {

inline const char* SkipWhitespace(const char* p) {
    while (*p == ' ' || *p == '\t') ++p;
    return p;
}

// Parse a non-negative integer; advances `*cur`. Returns 0 on failure (caller
// should check that *cur moved). Used inside face-vertex tuples only.
inline int ParseInt(const char*& cur) {
    char* end = nullptr;
    const long v = std::strtol(cur, &end, 10);
    cur = end;
    return static_cast<int>(v);
}

inline float ParseFloat(const char*& cur) {
    char* end = nullptr;
    const float v = std::strtof(cur, &end);
    cur = end;
    return v;
}

// Resolve a 1-based or negative OBJ index against the current array size.
// Returns -1 on out-of-range.
inline int ResolveIndex(int raw, int currentSize) {
    if (raw > 0) {
        const int idx = raw - 1;
        return (idx < currentSize) ? idx : -1;
    }
    if (raw < 0) {
        const int idx = currentSize + raw;
        return (idx >= 0 && idx < currentSize) ? idx : -1;
    }
    return -1;
}

} // namespace

bool ParseObjFile(const char* path, ParsedMesh& out, std::string* outError) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        if (outError) {
            *outError = std::string("cannot open OBJ file: ") + (path ? path : "(null)");
        }
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        std::fclose(f);
        if (outError) *outError = "OBJ file empty or unreadable";
        return false;
    }
    std::vector<char> buf(static_cast<size_t>(sz) + 1);
    const size_t got = std::fread(buf.data(), 1, static_cast<size_t>(sz), f);
    std::fclose(f);
    if (got != static_cast<size_t>(sz)) {
        if (outError) *outError = "OBJ read truncated";
        return false;
    }
    buf[sz] = '\0';

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    positions.reserve(1u << 15);
    normals  .reserve(1u << 15);

    // (positionIndex, normalIndex) → vertex index in `out.Vertices`.
    std::unordered_map<uint64_t, uint32_t> dedupe;
    dedupe.reserve(1u << 15);

    out.Vertices.clear();
    out.Indices .clear();
    out.Submeshes.clear();

    auto beginSubmesh = [&](std::string name) {
        // Close the previous range's index count first.
        if (!out.Submeshes.empty()) {
            ParsedSubmesh& prev = out.Submeshes.back();
            prev.IndexCount = static_cast<uint32_t>(out.Indices.size()) - prev.FirstIndex;
        }
        ParsedSubmesh sm{};
        sm.Name       = std::move(name);
        sm.FirstIndex = static_cast<uint32_t>(out.Indices.size());
        sm.IndexCount = 0;
        out.Submeshes.push_back(std::move(sm));
    };

    glm::vec3 aabbMin( std::numeric_limits<float>::max());
    glm::vec3 aabbMax(-std::numeric_limits<float>::max());

    // Temporary storage for a single face's resolved vertex indices.
    std::vector<uint32_t> faceVerts;
    faceVerts.reserve(8);

    const char* cur = buf.data();
    const char* end = buf.data() + sz;
    int  lineNo = 0;

    while (cur < end) {
        // Read a line.
        const char* lineBegin = cur;
        while (cur < end && *cur != '\n' && *cur != '\r') ++cur;
        const char* lineEnd = cur;
        // Swallow line terminators.
        while (cur < end && (*cur == '\n' || *cur == '\r')) ++cur;
        ++lineNo;

        const char* p = SkipWhitespace(lineBegin);
        if (p >= lineEnd || *p == '#') continue;

        // Dispatch on first 1-2 chars.
        if (p[0] == 'v' && p[1] == ' ') {
            p = SkipWhitespace(p + 2);
            const float x = ParseFloat(p);
            const float y = ParseFloat(p);
            const float z = ParseFloat(p);
            positions.emplace_back(x, y, z);
            aabbMin = glm::min(aabbMin, positions.back());
            aabbMax = glm::max(aabbMax, positions.back());
        } else if (p[0] == 'v' && p[1] == 'n' && p[2] == ' ') {
            p = SkipWhitespace(p + 3);
            const float x = ParseFloat(p);
            const float y = ParseFloat(p);
            const float z = ParseFloat(p);
            normals.emplace_back(x, y, z);
        } else if (p[0] == 'v' && p[1] == 't') {
            // UVs ignored in v1.
            continue;
        } else if (p[0] == 'f' && p[1] == ' ') {
            if (out.Submeshes.empty()) beginSubmesh("default");
            p = SkipWhitespace(p + 2);

            faceVerts.clear();
            while (p < lineEnd) {
                // Each face-vertex is `vi/vti/vni` (vti may be empty: `vi//vni`).
                const int rawV = ParseInt(p);
                int rawVt = 0, rawVn = 0;
                if (*p == '/') {
                    ++p;
                    if (*p != '/') rawVt = ParseInt(p);   // skip vt
                    if (*p == '/') { ++p; rawVn = ParseInt(p); }
                }
                if (rawV == 0 || rawVn == 0) {
                    if (outError) {
                        char tmp[160];
                        std::snprintf(tmp, sizeof(tmp),
                                      "OBJ line %d: face vertex missing position or normal", lineNo);
                        *outError = tmp;
                    }
                    return false;
                }
                const int posIdx = ResolveIndex(rawV,  static_cast<int>(positions.size()));
                const int nrmIdx = ResolveIndex(rawVn, static_cast<int>(normals  .size()));
                if (posIdx < 0 || nrmIdx < 0) {
                    if (outError) {
                        char tmp[160];
                        std::snprintf(tmp, sizeof(tmp),
                                      "OBJ line %d: face index out of range (v=%d, vn=%d)",
                                      lineNo, rawV, rawVn);
                        *outError = tmp;
                    }
                    return false;
                }

                const uint64_t key =
                    (static_cast<uint64_t>(static_cast<uint32_t>(posIdx)) << 32) |
                     static_cast<uint64_t>(static_cast<uint32_t>(nrmIdx));
                auto it = dedupe.find(key);
                uint32_t vIdx;
                if (it != dedupe.end()) {
                    vIdx = it->second;
                } else {
                    vIdx = static_cast<uint32_t>(out.Vertices.size());
                    out.Vertices.push_back({ positions[posIdx], normals[nrmIdx] });
                    dedupe.emplace(key, vIdx);
                }
                faceVerts.push_back(vIdx);

                p = SkipWhitespace(p);
            }

            // Fan-triangulate: (v0, vi, vi+1) for i = 1..N-2. Quads → 2 tris.
            if (faceVerts.size() >= 3) {
                for (size_t i = 1; i + 1 < faceVerts.size(); ++i) {
                    out.Indices.push_back(faceVerts[0]);
                    out.Indices.push_back(faceVerts[i]);
                    out.Indices.push_back(faceVerts[i + 1]);
                }
            }
        } else if ((p[0] == 'g' || p[0] == 'o') && (p[1] == ' ' || p[1] == '\t')) {
            p = SkipWhitespace(p + 2);
            std::string name(p, lineEnd);
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
            if (name.empty()) name = "default";
            beginSubmesh(std::move(name));
        }
        // Everything else (mtllib, usemtl, s, ...) intentionally ignored.
    }

    // Close the trailing submesh's index range.
    if (!out.Submeshes.empty()) {
        ParsedSubmesh& last = out.Submeshes.back();
        last.IndexCount = static_cast<uint32_t>(out.Indices.size()) - last.FirstIndex;
    } else if (!out.Indices.empty()) {
        // No 'g'/'o' line ever appeared but we did parse faces — synthesise one.
        ParsedSubmesh sm{};
        sm.Name       = "default";
        sm.FirstIndex = 0;
        sm.IndexCount = static_cast<uint32_t>(out.Indices.size());
        out.Submeshes.push_back(std::move(sm));
    }

    if (out.Vertices.empty() || out.Indices.empty()) {
        if (outError) *outError = "OBJ parsed but produced no triangles";
        return false;
    }

    out.AABBMin = aabbMin;
    out.AABBMax = aabbMax;

    RS_LOG_INFO("OBJ parsed: %s -> %zu verts, %zu indices (%zu tris), %zu submeshes",
                path, out.Vertices.size(), out.Indices.size(),
                out.Indices.size() / 3, out.Submeshes.size());
    return true;
}

} // namespace RS
