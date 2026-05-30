// Source/Scene/FloorMesh.h — Phase 11.5 floor-in-GBuffer
// Synthesises a vast XZ plane (one quad, two triangles, normals = +Y) as a
// ParsedMesh and inscribes it through MeshRegistry. The result is a real
// GBuffer participant — shadows from any IShadowAlgorithm (PCF/PCSS/VSM) catch
// it for free because every shadow record loops InstanceRegistry. GI in
// Phase 14b/c will see it the same way.
//
// The checker pattern is procedural in gbuffer.frag (Floor flag in the push
// block); no texture is allocated. Floor extent defaults to 500m half-size
// (vast but finite — long-term scalable, no infinite-plane numerics).
#pragma once

#include "Scene/MeshRegistry.h"
#include "RS/Scene.h"   // MeshHandle

namespace RS {

struct FloorPlaneDesc {
    float HalfExtent = 500.0f;   // metres from origin along X and Z (full plane = 1km square)
    float Y          = 0.0f;     // world-Y of the plane
};

// Synthesise the plane mesh and upload it through MeshRegistry. Returns the
// uploaded handle (0 on failure). The mesh has one submesh named "floor".
MeshHandle FloorMeshInscribe(Scene& scene, const FloorPlaneDesc& desc);

} // namespace RS
