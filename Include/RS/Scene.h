// Include/RS/Scene.h
// Mesh and instance registries. Each OBJ 'g'/'o' group becomes a submesh that
// holds an independent material slot per instance.
#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace RS {

struct ParsedMesh;   // Source/Scene/ObjLoader.h — used by the procedural-inscribe friend

using MeshHandle     = uint32_t;
using SubmeshIndex   = uint32_t;
using InstanceHandle = uint32_t;
using MaterialHandle = uint32_t;

struct MeshSource {
    const char* ObjPath = nullptr;   // absolute or relative to the working dir
};

struct InstanceSource {
    MeshHandle Mesh      = 0;
    glm::mat4  Transform = glm::mat4(1.0f);
};

class Scene {
public:
    Scene();
    ~Scene();

    MeshHandle     InscribeMesh    (const MeshSource& src);
    InstanceHandle InscribeInstance(const InstanceSource& src);

    uint32_t    SubmeshCount    (MeshHandle mesh) const;
    const char* SubmeshGroupName(MeshHandle mesh, SubmeshIndex submesh) const;

    void BindMaterial   (InstanceHandle instance, SubmeshIndex submesh, MaterialHandle material);
    void DestroyInstance(InstanceHandle instance);
    void DestroyMesh    (MeshHandle mesh);     // refuses if instances still reference it

private:
    struct Impl;
    Impl* m_Impl = nullptr;

    // Project-internal hooks defined in Source/Scene/SceneInternal.h. They
    // need direct access to m_Impl so the Vulkan-aware MeshRegistry can be
    // initialised lazily without exposing it through this public header.
    friend struct VulkanContext;
    friend void SceneAttachVulkan(Scene&, const struct VulkanContext&);
    friend void SceneDetachVulkan(Scene&);
    friend const class MeshRegistry&     SceneMeshes      (const Scene&);
    friend const class InstanceRegistry& SceneInstances   (const Scene&);
    friend       class InstanceRegistry& SceneInstancesMut(Scene&);
    friend MeshHandle SceneInscribeProceduralMesh(Scene&, const ParsedMesh&);
};

} // namespace RS
