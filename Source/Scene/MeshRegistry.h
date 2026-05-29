// Source/Scene/MeshRegistry.h — Phase 5
// Owns vertex/index VkBuffers per mesh. v1 uses host-visible memory + memcpy
// so we don't need a staging-buffer + transfer-queue path until Phase 7 wants
// device-local performance. ShaderBall is ~2 MB after dedup — easily fits.
#pragma once

#include "Core/VulkanContext.h"
#include "Scene/ObjLoader.h"

#include <string>
#include <vector>

namespace RS {

using MeshHandle = uint32_t;

struct GpuSubmesh {
    std::string Name;
    uint32_t    FirstIndex = 0;
    uint32_t    IndexCount = 0;
};

struct GpuMesh {
    VkBuffer        VertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory  VertexMemory       = VK_NULL_HANDLE;
    VkBuffer        IndexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory  IndexMemory        = VK_NULL_HANDLE;
    uint32_t        VertexCount        = 0;
    uint32_t        IndexCount         = 0;
    std::vector<GpuSubmesh> Submeshes;
    glm::vec3       AABBMin            = glm::vec3(0.0f);
    glm::vec3       AABBMax            = glm::vec3(0.0f);
};

class MeshRegistry {
public:
    MeshRegistry()  = default;
    ~MeshRegistry() = default;

    bool       Initialize(const VulkanContext& ctx);
    void       Terminate (const VulkanContext& ctx);

    // Upload a parsed mesh and return its handle (1-based; 0 = invalid).
    MeshHandle Upload(const VulkanContext& ctx, const ParsedMesh& parsed);

    const GpuMesh* Get(MeshHandle handle) const;
    size_t         Count() const { return m_Meshes.size(); }

private:
    bool m_Initialized = false;
    std::vector<GpuMesh> m_Meshes;     // index 0 is a sentinel
};

} // namespace RS
