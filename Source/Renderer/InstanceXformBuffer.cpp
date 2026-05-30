// Source/Renderer/InstanceXformBuffer.cpp — Phase 13.5
#include "Renderer/InstanceXformBuffer.h"
#include "Scene/SceneInternal.h"
#include "Scene/InstanceRegistry.h"
#include "Core/Logger.h"

#include <glm/gtc/matrix_inverse.hpp>
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

} // namespace

bool InstanceXformBufferInitialize(InstanceXformBuffer& xb, const VulkanContext& ctx) {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = InstanceXformBuffer::kBytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(ctx.Device, &bci, nullptr, &xb.Buffers[i]) != VK_SUCCESS) {
            RS_LOG_ERROR("InstanceXformBuffer: vkCreateBuffer failed (slot %u)", i);
            return false;
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(ctx.Device, xb.Buffers[i], &req);
        const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memType < 0) {
            RS_LOG_ERROR("InstanceXformBuffer: no host-visible memory type");
            return false;
        }

        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(memType);
        if (vkAllocateMemory(ctx.Device, &mai, nullptr, &xb.Memories[i]) != VK_SUCCESS) return false;
        vkBindBufferMemory(ctx.Device, xb.Buffers[i], xb.Memories[i], 0);

        if (vkMapMemory(ctx.Device, xb.Memories[i], 0, req.size, 0, &xb.Mapped[i]) != VK_SUCCESS) {
            RS_LOG_ERROR("InstanceXformBuffer: vkMapMemory failed (slot %u)", i);
            return false;
        }
        // Identity by default so out-of-range fetches map world -> world.
        glm::mat4* dst = static_cast<glm::mat4*>(xb.Mapped[i]);
        for (uint32_t k = 0; k < InstanceXformBuffer::kMaxInstances; ++k) {
            dst[k] = glm::mat4(1.0f);
        }
    }
    return true;
}

void InstanceXformBufferTerminate(InstanceXformBuffer& xb, const VulkanContext& ctx) {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (xb.Mapped[i]) {
            vkUnmapMemory(ctx.Device, xb.Memories[i]);
            xb.Mapped[i] = nullptr;
        }
        if (xb.Buffers[i])  vkDestroyBuffer(ctx.Device, xb.Buffers[i],  nullptr);
        if (xb.Memories[i]) vkFreeMemory   (ctx.Device, xb.Memories[i], nullptr);
        xb.Buffers[i]  = VK_NULL_HANDLE;
        xb.Memories[i] = VK_NULL_HANDLE;
    }
}

void InstanceXformBufferRefresh(InstanceXformBuffer& xb,
                                uint32_t frameSlot,
                                const Scene& scene,
                                MeshHandle sdfMesh,
                                InstanceHandle sdfAnchorInstance) {
    if (frameSlot >= VulkanContext::kFramesInFlight || !xb.Mapped[frameSlot]) return;

    struct alignas(16) Layout {
        uint32_t count;
        uint32_t pad[3];
        glm::mat4 sdfInvModels[256];
    };
    Layout* dst = static_cast<Layout*>(xb.Mapped[frameSlot]);

    dst->count = 0;

    const InstanceRegistry& reg = SceneInstances(scene);

    if (sdfMesh != 0) {
        reg.ForEach([&](InstanceHandle handle, const GpuInstance& inst) {
            if (inst.Mesh == sdfMesh && dst->count < InstanceXformBuffer::kMaxInstances) {
                dst->sdfInvModels[dst->count++] = glm::inverse(inst.Transform);
            }
        });
    }
}

} // namespace RS
