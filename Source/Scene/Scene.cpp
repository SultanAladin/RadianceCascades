// Source/Scene/Scene.cpp — Phase 5
// Concrete facade. InscribeMesh parses the OBJ + uploads to the MeshRegistry;
// InscribeInstance records a transform + per-submesh material binding table.
// The VulkanContext pointer is bound through Source/Scene/SceneInternal.h —
// the public RS::Scene type stays Vulkan-agnostic so Pigment/CAD hosts can
// still construct one for query-only purposes if they ever need that.
#include "RS/Scene.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"

namespace RS {

struct Scene::Impl {
    const VulkanContext* Vk = nullptr;
    MeshRegistry         Meshes;
    InstanceRegistry     Instances;
};

Scene::Scene()  : m_Impl(new Impl()) {}
Scene::~Scene() {
    if (m_Impl) {
        // Buffer cleanup must already have happened via SceneDetachVulkan; if
        // a caller forgot, log so we surface the leak instead of crashing.
        if (m_Impl->Vk) {
            RS_LOG_WARN("Scene destroyed with VulkanContext still attached -- did you forget SceneDetachVulkan?");
        }
        delete m_Impl;
        m_Impl = nullptr;
    }
}

MeshHandle Scene::InscribeMesh(const MeshSource& src) {
    if (!m_Impl || !m_Impl->Vk) {
        RS_LOG_ERROR("Scene::InscribeMesh before SceneAttachVulkan");
        return 0;
    }
    if (!src.ObjPath) {
        RS_LOG_ERROR("Scene::InscribeMesh: null path");
        return 0;
    }
    ParsedMesh parsed;
    std::string err;
    if (!ParseObjFile(src.ObjPath, parsed, &err)) {
        RS_LOG_ERROR("OBJ parse failed: %s", err.c_str());
        return 0;
    }
    return m_Impl->Meshes.Upload(*m_Impl->Vk, parsed);
}

InstanceHandle Scene::InscribeInstance(const InstanceSource& src) {
    if (!m_Impl) return 0;
    const GpuMesh* mesh = m_Impl->Meshes.Get(src.Mesh);
    if (!mesh) {
        RS_LOG_ERROR("Scene::InscribeInstance: unknown MeshHandle %u", src.Mesh);
        return 0;
    }
    return m_Impl->Instances.Create(src.Mesh, src.Transform,
                                    static_cast<uint32_t>(mesh->Submeshes.size()));
}

uint32_t Scene::SubmeshCount(MeshHandle mesh) const {
    if (!m_Impl) return 0;
    const GpuMesh* m = m_Impl->Meshes.Get(mesh);
    return m ? static_cast<uint32_t>(m->Submeshes.size()) : 0;
}

const char* Scene::SubmeshGroupName(MeshHandle mesh, SubmeshIndex submesh) const {
    if (!m_Impl) return "";
    const GpuMesh* m = m_Impl->Meshes.Get(mesh);
    if (!m || submesh >= m->Submeshes.size()) return "";
    return m->Submeshes[submesh].Name.c_str();
}

void Scene::BindMaterial(InstanceHandle inst, SubmeshIndex submesh, MaterialHandle mat) {
    if (m_Impl) m_Impl->Instances.BindMaterial(inst, submesh, mat);
}

void Scene::DestroyInstance(InstanceHandle inst) {
    if (m_Impl) m_Impl->Instances.Destroy(inst);
}

void Scene::DestroyMesh(MeshHandle /*mesh*/) {
    // v1: meshes outlive the Scene by design (we don't ref-count instances yet).
    // Phase 9+ tracks usage and releases buffers here.
}

// --- internal hooks --------------------------------------------------------

void SceneAttachVulkan(Scene& scene, const VulkanContext& ctx) {
    if (!scene.m_Impl) return;
    scene.m_Impl->Vk = &ctx;
    scene.m_Impl->Meshes.Initialize(ctx);
}

void SceneDetachVulkan(Scene& scene) {
    if (!scene.m_Impl || !scene.m_Impl->Vk) return;
    scene.m_Impl->Meshes.Terminate(*scene.m_Impl->Vk);
    scene.m_Impl->Vk = nullptr;
}

const MeshRegistry& SceneMeshes(const Scene& scene) {
    return scene.m_Impl->Meshes;
}

const InstanceRegistry& SceneInstances(const Scene& scene) {
    return scene.m_Impl->Instances;
}

InstanceRegistry& SceneInstancesMut(Scene& scene) {
    return scene.m_Impl->Instances;
}

} // namespace RS
