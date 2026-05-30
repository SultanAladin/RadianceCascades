// Source/Scene/FloorMesh.cpp — Phase 11.5
#include "Scene/FloorMesh.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"

namespace RS {

MeshHandle FloorMeshInscribe(Scene& scene, const FloorPlaneDesc& desc) {
    const float e = desc.HalfExtent;
    const float y = desc.Y;

    ParsedMesh parsed;
    parsed.Vertices.resize(4);
    parsed.Vertices[0] = { glm::vec3(-e, y, -e), glm::vec3(0.0f, 1.0f, 0.0f) };
    parsed.Vertices[1] = { glm::vec3( e, y, -e), glm::vec3(0.0f, 1.0f, 0.0f) };
    parsed.Vertices[2] = { glm::vec3( e, y,  e), glm::vec3(0.0f, 1.0f, 0.0f) };
    parsed.Vertices[3] = { glm::vec3(-e, y,  e), glm::vec3(0.0f, 1.0f, 0.0f) };

    // CCW from +Y so the GBuffer pass's BACK_BIT cull keeps the top face.
    parsed.Indices = { 0, 2, 1,   0, 3, 2 };

    ParsedSubmesh sub{};
    sub.Name       = "floor";
    sub.FirstIndex = 0;
    sub.IndexCount = static_cast<uint32_t>(parsed.Indices.size());
    parsed.Submeshes.push_back(std::move(sub));

    parsed.AABBMin = glm::vec3(-e, y, -e);
    parsed.AABBMax = glm::vec3( e, y,  e);

    const MeshHandle h = SceneInscribeProceduralMesh(scene, parsed);
    if (h == 0) {
        RS_LOG_ERROR("FloorMesh: inscribe failed");
    } else {
        RS_LOG_INFO("FloorMesh: handle %u, half-extent %.1fm at y=%.3f",
                    h, e, y);
    }
    return h;
}

} // namespace RS
