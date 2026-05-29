// Source/Renderer/Picking.cpp — Phase 9
#include "Renderer/Picking.h"
#include "Core/Logger.h"

#include <cstring>

namespace RS {

namespace {

constexpr VkDeviceSize kSlotBytes = sizeof(uint32_t) * 2;   // (instance, submesh)

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

bool CreateStaging(const VulkanContext& ctx, PickingSlot& slot) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = kSlotBytes;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &slot.Staging) != VK_SUCCESS) {
        RS_LOG_ERROR("Picking: vkCreateBuffer failed");
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, slot.Staging, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType < 0) {
        RS_LOG_ERROR("Picking: no HOST_VISIBLE|HOST_COHERENT memory type");
        return false;
    }
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &slot.Memory) != VK_SUCCESS) {
        RS_LOG_ERROR("Picking: vkAllocateMemory failed");
        return false;
    }
    if (vkBindBufferMemory(ctx.Device, slot.Staging, slot.Memory, 0) != VK_SUCCESS) {
        RS_LOG_ERROR("Picking: vkBindBufferMemory failed");
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx.Device, slot.Memory, 0, kSlotBytes, 0, &mapped) != VK_SUCCESS) {
        RS_LOG_ERROR("Picking: vkMapMemory failed");
        return false;
    }
    slot.Mapped = reinterpret_cast<uint32_t*>(mapped);
    slot.Mapped[0] = 0;
    slot.Mapped[1] = 0;
    return true;
}

} // namespace

bool PickingInitialize(PickingSystem& sys, const VulkanContext& ctx) {
    if (sys.Initialized) return true;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (!CreateStaging(ctx, sys.Slots[i])) return false;
    }
    sys.Initialized = true;
    RS_LOG_INFO("PickingSystem ready: %u host-visible 8-byte readback slots",
                VulkanContext::kFramesInFlight);
    return true;
}

void PickingTerminate(PickingSystem& sys, const VulkanContext& ctx) {
    if (!sys.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (PickingSlot& s : sys.Slots) {
        if (s.Memory) {
            vkUnmapMemory(ctx.Device, s.Memory);
            vkFreeMemory (ctx.Device, s.Memory, nullptr);
        }
        if (s.Staging) vkDestroyBuffer(ctx.Device, s.Staging, nullptr);
        s = PickingSlot{};
    }
    sys.Initialized = false;
}

void PickingRequest(PickingSystem& sys, uint32_t frameSlot,
                    uint32_t x, uint32_t y) {
    if (!sys.Initialized || frameSlot >= VulkanContext::kFramesInFlight) return;
    PickingSlot& s = sys.Slots[frameSlot];
    s.Requested  = true;
    s.RequestedX = x;
    s.RequestedY = y;
}

void PickingRecordCopy(PickingSystem& sys, VkCommandBuffer cmd,
                       uint32_t frameSlot,
                       VkImage identityImage, VkExtent2D extent) {
    if (!sys.Initialized || frameSlot >= VulkanContext::kFramesInFlight) return;
    PickingSlot& s = sys.Slots[frameSlot];
    if (!s.Requested) return;
    s.Requested = false;

    // Clamp to the actual rendered extent so out-of-range clicks pull from a
    // valid texel (decodes as 0/0 → "no pick").
    const uint32_t px = (extent.width  > 0) ? (s.RequestedX < extent.width  ? s.RequestedX : extent.width  - 1) : 0;
    const uint32_t py = (extent.height > 0) ? (s.RequestedY < extent.height ? s.RequestedY : extent.height - 1) : 0;

    // 1. identity image: SHADER_READ_ONLY_OPTIMAL -> TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier toSrc{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toSrc.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toSrc.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toSrc.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.image         = identityImage;
    toSrc.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    toSrc.subresourceRange.baseMipLevel   = 0;
    toSrc.subresourceRange.levelCount     = 1;
    toSrc.subresourceRange.baseArrayLayer = 0;
    toSrc.subresourceRange.layerCount     = 1;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toSrc);

    // 2. 1x1 copy at the cursor.
    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = { static_cast<int32_t>(px),
                                               static_cast<int32_t>(py), 0 };
    region.imageExtent                     = { 1u, 1u, 1u };
    vkCmdCopyImageToBuffer(cmd, identityImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           s.Staging, 1, &region);

    // 3. identity image: TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL so
    //    the GBufferPreview sample (and the next GBufferPass entry) find the
    //    layout the rest of the pipeline expects.
    VkImageMemoryBarrier toShader = toSrc;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShader.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toShader.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toShader);

    s.ResultReady = true;
}

bool PickingTryRead(PickingSystem& sys, uint32_t frameSlot,
                    uint32_t& outInstance, uint32_t& outSubmesh) {
    if (!sys.Initialized || frameSlot >= VulkanContext::kFramesInFlight) return false;
    PickingSlot& s = sys.Slots[frameSlot];
    if (!s.ResultReady) return false;

    // BeginFrame has waited the fence for this slot, so the copy submitted on
    // the previous cycle is GPU-complete and the host pointer is safe.
    outInstance = s.Mapped[0];
    outSubmesh  = s.Mapped[1];
    s.ResultReady = false;
    return true;
}

} // namespace RS
