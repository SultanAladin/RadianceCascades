// Source/Scene/ObjLoader.h — Phase 5
// In-house OBJ parser (NO tinyobjloader, NO new third-party deps).
// Parses v / vn / f lines (drops vt for v1 — Phase 5 doesn't sample textures).
// Faces are fan-triangulated; (positionIndex, normalIndex) tuples are deduped
// so the produced vertex stream is render-ready.
//
// OBJ groups: each `g <name>` or `o <name>` starts a new submesh range. ShaderBall
// is single-group (`g default`) so the output is one submesh.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace RS {

struct ParsedVertex {
    glm::vec3 Position;
    glm::vec3 Normal;
};

struct ParsedSubmesh {
    std::string Name;             // group/object name (defaults to "default")
    uint32_t    FirstIndex = 0;   // index into ParsedMesh::Indices
    uint32_t    IndexCount = 0;   // triangulated tri-list count (multiple of 3)
};

struct ParsedMesh {
    std::vector<ParsedVertex>  Vertices;
    std::vector<uint32_t>      Indices;
    std::vector<ParsedSubmesh> Submeshes;
    glm::vec3                  AABBMin = glm::vec3(0.0f);
    glm::vec3                  AABBMax = glm::vec3(0.0f);
};

// Returns true on success. On failure, `outError` (if non-null) gets a one-line
// message; the partial parse state is discarded.
bool ParseObjFile(const char* path, ParsedMesh& outMesh, std::string* outError);

} // namespace RS
