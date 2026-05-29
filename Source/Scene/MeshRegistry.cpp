// Source/Scene/MeshRegistry.cpp — Phase 5
// Allocates two VkBuffers per mesh (vertex + index) in host-visible memory so
// the upload is a single vkMapMemory/memcpy. Phase 7's staged-upload path will
// take over once we care about device-local throughput for large scenes.
#include "Scene/MeshRegistry.h"
#include "Core/Logger.h"

#include <cstring>

namespace RS {

namespace {

int FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool CreateHostBuffer(const VulkanContext& ctx, VkDeviceSize bytes,
                      VkBufferUsageFlags usage,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bytes;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &outBuf) != VK_SUCCESS) {
        RS_LOG_ERROR("MeshRegistry: vkCreateBuffer failed (size=%llu)",
                     static_cast<unsigned long long>(bytes));
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, outBuf, &req);

    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType < 0) {
        RS_LOG_ERROR("MeshRegistry: no HOST_VISIBLE|HOST_COHERENT memory type");
        return false;
    }
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &outMem) != VK_SUCCESS) {
        RS_LOG_ERROR("MeshRegistry: vkAllocateMemory failed");
        return false;
    }
    if (vkBindBufferMemory(ctx.Device, outBuf, outMem, 0) != VK_SUCCESS) {
        RS_LOG_ERROR("MeshRegistry: vkBindBufferMemory failed");
        return false;
    }
    return true;
}

} // namespace

bool MeshRegistry::Initialize(const VulkanContext&) {
    if (m_Initialized) return true;
    m_Meshes.clear();
    m_Meshes.emplace_back();   // sentinel — handle 0 = invalid
    m_Initialized = true;
    return true;
}

void MeshRegistry::Terminate(const VulkanContext& ctx) {
    if (!m_Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (size_t i = 1; i < m_Meshes.size(); ++i) {
        GpuMesh& m = m_Meshes[i];
        if (m.VertexBuffer) vkDestroyBuffer(ctx.Device, m.VertexBuffer, nullptr);
        if (m.VertexMemory) vkFreeMemory   (ctx.Device, m.VertexMemory, nullptr);
        if (m.IndexBuffer)  vkDestroyBuffer(ctx.Device, m.IndexBuffer,  nullptr);
        if (m.IndexMemory)  vkFreeMemory   (ctx.Device, m.IndexMemory,  nullptr);
    }
    m_Meshes.clear();
    m_Initialized = false;
}

MeshHandle MeshRegistry::Upload(const VulkanContext& ctx, const ParsedMesh& parsed) {
    if (!m_Initialized) {
        RS_LOG_ERROR("MeshRegistry::Upload before Initialize");
        return 0;
    }
    if (parsed.Vertices.empty() || parsed.Indices.empty()) return 0;

    GpuMesh m{};
    m.VertexCount = static_cast<uint32_t>(parsed.Vertices.size());
    m.IndexCount  = static_cast<uint32_t>(parsed.Indices.size());
    m.AABBMin     = parsed.AABBMin;
    m.AABBMax     = parsed.AABBMax;

    const VkDeviceSize vBytes = sizeof(ParsedVertex) * parsed.Vertices.size();
    const VkDeviceSize iBytes = sizeof(uint32_t)     * parsed.Indices.size();

    if (!CreateHostBuffer(ctx, vBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          m.VertexBuffer, m.VertexMemory)) return 0;
    if (!CreateHostBuffer(ctx, iBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          m.IndexBuffer,  m.IndexMemory))  return 0;

    void* mapped = nullptr;
    if (vkMapMemory(ctx.Device, m.VertexMemory, 0, vBytes, 0, &mapped) != VK_SUCCESS) return 0;
    std::memcpy(mapped, parsed.Vertices.data(), static_cast<size_t>(vBytes));
    vkUnmapMemory(ctx.Device, m.VertexMemory);

    if (vkMapMemory(ctx.Device, m.IndexMemory, 0, iBytes, 0, &mapped) != VK_SUCCESS) return 0;
    std::memcpy(mapped, parsed.Indices.data(), static_cast<size_t>(iBytes));
    vkUnmapMemory(ctx.Device, m.IndexMemory);

    m.Submeshes.reserve(parsed.Submeshes.size());
    for (const ParsedSubmesh& s : parsed.Submeshes) {
        GpuSubmesh gs{};
        gs.Name       = s.Name;
        gs.FirstIndex = s.FirstIndex;
        gs.IndexCount = s.IndexCount;
        m.Submeshes.push_back(std::move(gs));
    }

    m_Meshes.push_back(std::move(m));
    const MeshHandle handle = static_cast<MeshHandle>(m_Meshes.size() - 1);
    RS_LOG_INFO("MeshRegistry: uploaded handle %u (%u verts, %u indices)",
                handle, m_Meshes.back().VertexCount, m_Meshes.back().IndexCount);
    return handle;
}

const GpuMesh* MeshRegistry::Get(MeshHandle handle) const {
    if (handle == 0 || handle >= m_Meshes.size()) return nullptr;
    return &m_Meshes[handle];
}

} // namespace RS
