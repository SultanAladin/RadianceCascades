// Source/Scene/InstanceRegistry.cpp — Phase 5
#include "Scene/InstanceRegistry.h"

namespace RS {

InstanceHandle InstanceRegistry::Create(MeshHandle mesh, const glm::mat4& transform,
                                        uint32_t submeshCount) {
    if (m_Instances.empty()) m_Instances.emplace_back();  // sentinel slot
    GpuInstance inst{};
    inst.Mesh      = mesh;
    inst.Transform = transform;
    inst.MaterialBindings.assign(submeshCount, 0u);
    m_Instances.push_back(std::move(inst));
    return static_cast<InstanceHandle>(m_Instances.size() - 1);
}

void InstanceRegistry::Destroy(InstanceHandle handle) {
    if (handle == 0 || handle >= m_Instances.size()) return;
    m_Instances[handle] = GpuInstance{};   // mark slot dead (Mesh==0)
}

void InstanceRegistry::BindMaterial(InstanceHandle inst, SubmeshIndex submesh,
                                    MaterialHandle material) {
    if (inst == 0 || inst >= m_Instances.size()) return;
    GpuInstance& gi = m_Instances[inst];
    if (submesh >= gi.MaterialBindings.size()) return;
    gi.MaterialBindings[submesh] = material;
}

const GpuInstance* InstanceRegistry::Get(InstanceHandle handle) const {
    if (handle == 0 || handle >= m_Instances.size()) return nullptr;
    if (m_Instances[handle].Mesh == 0) return nullptr;
    return &m_Instances[handle];
}

} // namespace RS
