// Source/Scene/ScaleRefSphere.cpp
#include "Scene/ScaleRefSphere.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"

#include <cmath>

namespace RS {

MeshHandle ScaleRefSphereInscribe(Scene& scene, const ScaleRefSphereDesc& desc) {
    const float    r        = desc.Radius;
    const uint32_t segments = desc.Segments < 3u ? 3u : desc.Segments;
    const uint32_t rings    = desc.Rings    < 2u ? 2u : desc.Rings;

    ParsedMesh parsed;

    // Vertices: (rings+1) latitude bands x (segments+1) longitude columns. The
    // seam column is duplicated so UVs/normals stay simple (we only need pos+N).
    for (uint32_t y = 0; y <= rings; ++y) {
        const float v     = static_cast<float>(y) / static_cast<float>(rings);
        const float phi   = v * 3.14159265358979323846f;          // 0..pi
        const float sinP  = std::sin(phi);
        const float cosP  = std::cos(phi);
        for (uint32_t x = 0; x <= segments; ++x) {
            const float u    = static_cast<float>(x) / static_cast<float>(segments);
            const float theta = u * 2.0f * 3.14159265358979323846f; // 0..2pi
            const glm::vec3 n(std::cos(theta) * sinP, cosP, std::sin(theta) * sinP);
            ParsedVertex pv;
            pv.Position = n * r;
            pv.Normal   = n;          // unit sphere: normal == direction
            parsed.Vertices.push_back(pv);
        }
    }

    const uint32_t stride = segments + 1u;
    for (uint32_t y = 0; y < rings; ++y) {
        for (uint32_t x = 0; x < segments; ++x) {
            const uint32_t i0 = y * stride + x;
            const uint32_t i1 = i0 + 1u;
            const uint32_t i2 = i0 + stride;
            const uint32_t i3 = i2 + 1u;
            // CCW when viewed from outside (matches the GBuffer BACK_BIT cull).
            parsed.Indices.push_back(i0);
            parsed.Indices.push_back(i2);
            parsed.Indices.push_back(i1);

            parsed.Indices.push_back(i1);
            parsed.Indices.push_back(i2);
            parsed.Indices.push_back(i3);
        }
    }

    ParsedSubmesh sub{};
    sub.Name       = "scaleref";
    sub.FirstIndex = 0;
    sub.IndexCount = static_cast<uint32_t>(parsed.Indices.size());
    parsed.Submeshes.push_back(std::move(sub));

    parsed.AABBMin = glm::vec3(-r);
    parsed.AABBMax = glm::vec3( r);

    const MeshHandle h = SceneInscribeProceduralMesh(scene, parsed);
    if (h == 0) {
        RS_LOG_ERROR("ScaleRefSphere: inscribe failed");
    } else {
        RS_LOG_INFO("ScaleRefSphere: handle %u, radius %.3fm (%u tris)",
                    h, r, static_cast<uint32_t>(parsed.Indices.size() / 3));
    }
    return h;
}

} // namespace RS
