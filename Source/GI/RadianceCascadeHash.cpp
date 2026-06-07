// Source/GI/RadianceCascadeHash.cpp — sparse radiance cascades storage core.
#include "GI/RadianceCascadeHash.h"
#include "Core/Logger.h"

#include <algorithm>
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

bool CreateBuffer(const VulkanContext& ctx, VkDeviceSize bytes,
                  VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                  VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &outBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, outBuf, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits, props);
    if (memType < 0) return false;

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &outMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(ctx.Device, outBuf, outMem, 0);
    return true;
}

bool CreateSetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    // 5 bindings: keys, payload, params, cell list, resolve.
    VkDescriptorSetLayoutBinding b[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = (i == 2) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 5;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 4 * VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 1 * VulkanContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    return vkCreateDescriptorPool(device, &pci, nullptr, &outPool) == VK_SUCCESS;
}

} // namespace

bool RadianceCascadeHashInitialize(RadianceCascadeHash& h,
                                   const VulkanContext& ctx,
                                   uint32_t baseLog2,
                                   uint32_t cascades) {
    if (h.Initialized) return true;

    baseLog2 = std::max(12u, std::min(18u, baseLog2));
    cascades = std::max(1u, std::min(RadianceCascadeHash::kMaxCascades, cascades));
    // Smallest sub-table must hold at least one workgroup's worth of slots.
    while (cascades > 1 && (1u << baseLog2) >> (2u * (cascades - 1)) < 64u) {
        --cascades;
    }
    h.BaseLog2  = baseLog2;
    h.BaseSlots = 1u << baseLog2;
    h.Cascades  = cascades;

    uint32_t totalKeySlots = 0;
    for (uint32_t c = 0; c < cascades; ++c) totalKeySlots += h.SlotsOf(c);

    h.KeyBytes      = VkDeviceSize(totalKeySlots) * sizeof(uint32_t);
    // Payload: cascades blocks of BaseSlots·D0 vec4 texels (constant per level).
    h.PayloadBytes  = VkDeviceSize(cascades) * h.BaseSlots *
                      RadianceCascadeHash::kD0 * 16u;
    h.CellListBytes = VkDeviceSize(cascades) * (h.BaseSlots + 1u) * sizeof(uint32_t);
    h.ResolveBytes  = VkDeviceSize(h.BaseSlots) * 3u * 16u;
    const VkDeviceSize paramsBytes = sizeof(RcHashParams);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (!CreateBuffer(ctx, h.KeyBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          h.KeyBuffers[i], h.KeyMemory[i])) {
            RS_LOG_ERROR("RCHash: key alloc failed slot %u", i); return false;
        }
        if (!CreateBuffer(ctx, h.PayloadBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          h.PayloadBuffers[i], h.PayloadMemory[i])) {
            RS_LOG_ERROR("RCHash: payload alloc failed slot %u", i); return false;
        }
        if (!CreateBuffer(ctx, h.CellListBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          h.CellListBuffers[i], h.CellListMemory[i])) {
            RS_LOG_ERROR("RCHash: celllist alloc failed slot %u", i); return false;
        }
        if (!CreateBuffer(ctx, h.ResolveBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          h.ResolveBuffers[i], h.ResolveMemory[i])) {
            RS_LOG_ERROR("RCHash: resolve alloc failed slot %u", i); return false;
        }

        if (!CreateBuffer(ctx, sizeof(uint32_t),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          h.ReadbackBuffers[i], h.ReadbackMemory[i])) {
            RS_LOG_ERROR("RCHash: readback alloc failed slot %u", i); return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(ctx.Device, h.ReadbackMemory[i], 0, sizeof(uint32_t), 0,
                        &mapped) != VK_SUCCESS) {
            return false;
        }
        h.ReadbackMapped[i] = reinterpret_cast<uint32_t*>(mapped);
        h.ReadbackMapped[i][0] = 0u;

        if (!CreateBuffer(ctx, paramsBytes,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          h.ParamsBuffers[i], h.ParamsMemory[i])) {
            RS_LOG_ERROR("RCHash: params alloc failed slot %u", i); return false;
        }
        if (vkMapMemory(ctx.Device, h.ParamsMemory[i], 0, paramsBytes, 0,
                        &h.ParamsMapped[i]) != VK_SUCCESS) {
            return false;
        }
        std::memset(h.ParamsMapped[i], 0, paramsBytes);
    }

    if (!CreateSetLayout(ctx.Device, h.SetLayout)) {
        RS_LOG_ERROR("RCHash: set layout failed"); return false;
    }
    if (!CreateDescriptorPool(ctx.Device, h.DescriptorPool)) {
        RS_LOG_ERROR("RCHash: descriptor pool failed"); return false;
    }

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = h.SetLayout;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = h.DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(ctx.Device, &ai, h.Sets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("RCHash: descriptor set alloc failed"); return false;
    }

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorBufferInfo bi[5]{};
        bi[0].buffer = h.KeyBuffers[i];      bi[0].range = h.KeyBytes;
        bi[1].buffer = h.PayloadBuffers[i];  bi[1].range = h.PayloadBytes;
        bi[2].buffer = h.ParamsBuffers[i];   bi[2].range = paramsBytes;
        bi[3].buffer = h.CellListBuffers[i]; bi[3].range = h.CellListBytes;
        bi[4].buffer = h.ResolveBuffers[i];  bi[4].range = h.ResolveBytes;

        VkWriteDescriptorSet w[5]{};
        for (uint32_t k = 0; k < 5; ++k) {
            w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet          = h.Sets[i];
            w[k].dstBinding      = k;
            w[k].descriptorCount = 1;
            w[k].descriptorType  = (k == 2) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[k].pBufferInfo     = &bi[k];
        }
        vkUpdateDescriptorSets(ctx.Device, 5, w, 0, nullptr);
    }

    h.Initialized = true;
    RS_LOG_INFO("RCHash ready: base 2^%u, %u cascades, payload %llu MB x%u frames",
                h.BaseLog2, h.Cascades,
                static_cast<unsigned long long>(h.PayloadBytes / (1024 * 1024)),
                VulkanContext::kFramesInFlight);
    return true;
}

void RadianceCascadeHashTerminate(RadianceCascadeHash& h, const VulkanContext& ctx) {
    if (!h.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (h.ParamsMapped[i]) {
            vkUnmapMemory(ctx.Device, h.ParamsMemory[i]);
            h.ParamsMapped[i] = nullptr;
        }
        if (h.ParamsBuffers[i]) vkDestroyBuffer(ctx.Device, h.ParamsBuffers[i], nullptr);
        if (h.ParamsMemory[i])  vkFreeMemory   (ctx.Device, h.ParamsMemory[i],  nullptr);

        if (h.ReadbackMapped[i]) {
            vkUnmapMemory(ctx.Device, h.ReadbackMemory[i]);
            h.ReadbackMapped[i] = nullptr;
        }
        if (h.ReadbackBuffers[i]) vkDestroyBuffer(ctx.Device, h.ReadbackBuffers[i], nullptr);
        if (h.ReadbackMemory[i])  vkFreeMemory   (ctx.Device, h.ReadbackMemory[i],  nullptr);

        if (h.ResolveBuffers[i])  vkDestroyBuffer(ctx.Device, h.ResolveBuffers[i],  nullptr);
        if (h.ResolveMemory[i])   vkFreeMemory   (ctx.Device, h.ResolveMemory[i],   nullptr);
        if (h.CellListBuffers[i]) vkDestroyBuffer(ctx.Device, h.CellListBuffers[i], nullptr);
        if (h.CellListMemory[i])  vkFreeMemory   (ctx.Device, h.CellListMemory[i],  nullptr);
        if (h.PayloadBuffers[i])  vkDestroyBuffer(ctx.Device, h.PayloadBuffers[i],  nullptr);
        if (h.PayloadMemory[i])   vkFreeMemory   (ctx.Device, h.PayloadMemory[i],   nullptr);
        if (h.KeyBuffers[i])      vkDestroyBuffer(ctx.Device, h.KeyBuffers[i],      nullptr);
        if (h.KeyMemory[i])       vkFreeMemory   (ctx.Device, h.KeyMemory[i],       nullptr);
    }
    if (h.DescriptorPool) vkDestroyDescriptorPool     (ctx.Device, h.DescriptorPool, nullptr);
    if (h.SetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, h.SetLayout,      nullptr);

    h = RadianceCascadeHash{};
}

void RadianceCascadeHashWriteParams(RadianceCascadeHash& h,
                                    uint32_t frameSlot,
                                    const RcHashParams& params) {
    if (frameSlot >= VulkanContext::kFramesInFlight) return;
    if (!h.ParamsMapped[frameSlot]) return;
    std::memcpy(h.ParamsMapped[frameSlot], &params, sizeof(RcHashParams));
}

void RadianceCascadeHashClearForFrame(const RadianceCascadeHash& h,
                                      VkCommandBuffer cmd,
                                      uint32_t frameSlot) {
    if (!h.Initialized) return;
    if (frameSlot >= VulkanContext::kFramesInFlight) return;
    VkBuffer keyBuf  = h.KeyBuffers[frameSlot];
    VkBuffer cellBuf = h.CellListBuffers[frameSlot];

    // The frame fence has retired this slot's previous frame, so no shader is
    // touching these buffers; only the transfer→compute ordering below matters.
    vkCmdFillBuffer(cmd, keyBuf, 0, h.KeyBytes, 0u);
    for (uint32_t c = 0; c < h.Cascades; ++c) {
        vkCmdFillBuffer(cmd, cellBuf,
                        VkDeviceSize(h.CellListOffset(c)) * sizeof(uint32_t),
                        sizeof(uint32_t), 0u);
    }

    VkBufferMemoryBarrier b[2]{};
    b[0].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[0].buffer = keyBuf;
    b[0].offset = 0;
    b[0].size   = VK_WHOLE_SIZE;
    b[1] = b[0];
    b[1].buffer = cellBuf;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 2, b, 0, nullptr);
}

void RadianceCascadeHashRecordReadback(const RadianceCascadeHash& h,
                                       VkCommandBuffer cmd,
                                       uint32_t frameSlot) {
    if (!h.Initialized) return;
    if (frameSlot >= VulkanContext::kFramesInFlight) return;
    if (h.ReadbackBuffers[frameSlot] == VK_NULL_HANDLE) return;

    VkBufferMemoryBarrier pre{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    pre.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.buffer              = h.CellListBuffers[frameSlot];
    pre.offset              = 0;
    pre.size                = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &pre, 0, nullptr);

    VkBufferCopy region{};
    region.size = sizeof(uint32_t);   // cascade-0 counter sits at offset 0
    vkCmdCopyBuffer(cmd, h.CellListBuffers[frameSlot],
                    h.ReadbackBuffers[frameSlot], 1, &region);
}

uint32_t RadianceCascadeHashReadProbeCount(const RadianceCascadeHash& h,
                                           uint32_t frameSlot) {
    if (!h.Initialized) return 0;
    if (frameSlot >= VulkanContext::kFramesInFlight) return 0;
    if (!h.ReadbackMapped[frameSlot]) return 0;
    return h.ReadbackMapped[frameSlot][0];
}

} // namespace RS
