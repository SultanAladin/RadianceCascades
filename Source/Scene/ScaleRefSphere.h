// Source/Scene/ScaleRefSphere.h
// Procedural UV-sphere inscribed as a real GBuffer mesh, mirroring FloorMesh's
// pattern. Purpose: a small, known-radius sphere placed above the first
// ShaderBall so the working world scale is visible at a glance — important
// because the ShaderBall OBJ is in centimetres and gets scaled down 100x, so
// "how big is one metre here" is otherwise hard to eyeball when tuning the
// radiance-cascade probe spacing.
#pragma once

#include "Scene/MeshRegistry.h"
#include "RS/Scene.h"   // MeshHandle

namespace RS {

struct ScaleRefSphereDesc {
    float    Radius   = 0.5f;                    // world metres
    uint32_t Segments = 24;                      // longitude divisions
    uint32_t Rings    = 16;                      // latitude divisions
};

// Synthesise a unit-ish UV sphere of the given radius (centred at the origin in
// mesh-local space) and upload it through MeshRegistry. Returns the uploaded
// handle (0 on failure). The mesh has one submesh named "scaleref". Position it
// with the instance transform.
MeshHandle ScaleRefSphereInscribe(Scene& scene, const ScaleRefSphereDesc& desc);

} // namespace RS
