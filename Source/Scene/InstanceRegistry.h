// Source/Scene/InstanceRegistry.h — Phase 5
// One transform per instance + a per-submesh material binding table. v1 has
// no material textures yet (Phase 6+), so the MaterialHandles slots stay 0.
#pragma once

#include "RS/Scene.h"

#include <glm/glm.hpp>
#include <vector>

namespace RS {

struct GpuInstance {
    MeshHandle             Mesh      = 0;
    glm::mat4              Transform = glm::mat4(1.0f);
    std::vector<MaterialHandle> MaterialBindings;  // size = mesh's submesh count
};

class InstanceRegistry {
public:
    InstanceRegistry()  = default;
    ~InstanceRegistry() = default;

    InstanceHandle Create(MeshHandle mesh, const glm::mat4& transform, uint32_t submeshCount);
    void           Destroy(InstanceHandle handle);

    void BindMaterial(InstanceHandle inst, SubmeshIndex submesh, MaterialHandle material);

    const GpuInstance* Get(InstanceHandle handle) const;
    size_t             Count() const { return m_Instances.size(); }

    // Iterate live instances (skips destroyed slots). Visitor: (handle, instance).
    template <typename Fn>
    void ForEach(Fn&& fn) const {
        for (size_t i = 1; i < m_Instances.size(); ++i) {
            if (m_Instances[i].Mesh != 0) fn(static_cast<InstanceHandle>(i), m_Instances[i]);
        }
    }

private:
    std::vector<GpuInstance> m_Instances;   // index 0 = sentinel
};

} // namespace RS
