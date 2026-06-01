// Source/GI/RadianceCascadeHash.cpp — Phase 14b
#include "GI/RadianceCascadeHash.h"
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
    // 4 bindings, all available to compute and fragment-equivalent
    // (lighting compose is compute too, so just COMPUTE_BIT).
    VkDescriptorSetLayoutBinding b[4]{};
    b[0].binding         = 0;
    b[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // HashKeys
    b[0].descriptorCount = 1;
    b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    b[1].binding         = 1;
    b[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // ProbePayload
    b[1].descriptorCount = 1;
    b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    b[2].binding         = 2;
    b[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;   // ParamsUbo
    b[2].descriptorCount = 1;
    b[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    b[3].binding         = 3;
    b[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // CellList
    b[3].descriptorCount = 1;
    b[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 4;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 3 * VulkanContext::kFramesInFlight;   // keys + payload + celllist
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 1 * VulkanContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    return vkCreateDescriptorPool(device, &pci, nullptr, &outPool) == VK_SUCCESS;
}

void WriteFrameSet(VkDevice device, VkDescriptorSet set,
                   VkBuffer keys, VkBuffer payload,
                   VkBuffer params, VkBuffer cellList,
                   VkDeviceSize keysBytes, VkDeviceSize payloadBytes,
                   VkDeviceSize paramsBytes, VkDeviceSize cellListBytes) {
    VkDescriptorBufferInfo bi[4]{};
    bi[0].buffer = keys;     bi[0].offset = 0; bi[0].range = keysBytes;
    bi[1].buffer = payload;  bi[1].offset = 0; bi[1].range = payloadBytes;
    bi[2].buffer = params;   bi[2].offset = 0; bi[2].range = paramsBytes;
    bi[3].buffer = cellList; bi[3].offset = 0; bi[3].range = cellListBytes;

    VkWriteDescriptorSet w[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = set;
        w[i].dstBinding      = i;
        w[i].descriptorCount = 1;
        w[i].pBufferInfo     = &bi[i];
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkUpdateDescriptorSets(device, 4, w, 0, nullptr);
}

} // namespace

bool RadianceCascadeHashInitialize(RadianceCascadeHash& h,
                                   const VulkanContext& ctx,
                                   uint32_t hashLog2) {
    if (h.Initialized) return true;

    if (hashLog2 < 16) hashLog2 = 16;
    if (hashLog2 > RadianceCascadeHash::kMaxHashLog2) {
        hashLog2 = RadianceCascadeHash::kMaxHashLog2;
    }
    h.HashLog2  = hashLog2;
    h.SlotCount = 1u << hashLog2;

    const VkDeviceSize keyBytes      =
        static_cast<VkDeviceSize>(h.SlotCount) * sizeof(uint32_t);
    const VkDeviceSize payloadBytes  =
        static_cast<VkDeviceSize>(h.SlotCount) *
        static_cast<VkDeviceSize>(RadianceCascadeHash::kBytesPerPayload);
    // CellList[0] = atomic counter, CellList[1..N] = compacted keys. Sized N+1
    // for headroom even though we cap insertion at N anyway.
    const VkDeviceSize cellListBytes =
        static_cast<VkDeviceSize>(h.SlotCount + 1) * sizeof(uint32_t);
    const VkDeviceSize paramsBytes   = sizeof(RcHashParams);

    if (!CreateBuffer(ctx, keyBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      h.KeyBuffer, h.KeyMemory)) {
        RS_LOG_ERROR("RadianceCascadeHash: KeyBuffer alloc failed");
        return false;
    }
    if (!CreateBuffer(ctx, payloadBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      h.PayloadBuffer, h.PayloadMemory)) {
        RS_LOG_ERROR("RadianceCascadeHash: PayloadBuffer alloc failed");
        return false;
    }
    if (!CreateBuffer(ctx, cellListBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      h.CellListBuffer, h.CellListMemory)) {
        RS_LOG_ERROR("RadianceCascadeHash: CellListBuffer alloc failed");
        return false;
    }

    // Phase 14c: 4-byte host-visible readback ring for CellList[0].
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (!CreateBuffer(ctx, sizeof(uint32_t),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          h.ReadbackBuffers[i], h.ReadbackMemory[i])) {
            RS_LOG_ERROR("RadianceCascadeHash: ReadbackBuffer alloc failed slot %u", i);
            return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(ctx.Device, h.ReadbackMemory[i], 0, sizeof(uint32_t), 0,
                        &mapped) != VK_SUCCESS) {
            RS_LOG_ERROR("RadianceCascadeHash: ReadbackBuffer map failed slot %u", i);
            return false;
        }
        h.ReadbackMapped[i] = reinterpret_cast<uint32_t*>(mapped);
        h.ReadbackMapped[i][0] = 0u;
    }

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (!CreateBuffer(ctx, paramsBytes,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          h.ParamsBuffers[i], h.ParamsMemory[i])) {
            RS_LOG_ERROR("RadianceCascadeHash: ParamsBuffer alloc failed slot %u", i);
            return false;
        }
        if (vkMapMemory(ctx.Device, h.ParamsMemory[i], 0, paramsBytes, 0,
                        &h.ParamsMapped[i]) != VK_SUCCESS) {
            RS_LOG_ERROR("RadianceCascadeHash: ParamsBuffer map failed slot %u", i);
            return false;
        }
        std::memset(h.ParamsMapped[i], 0, paramsBytes);
    }

    if (!CreateSetLayout(ctx.Device, h.SetLayout)) {
        RS_LOG_ERROR("RadianceCascadeHash: set layout failed"); return false;
    }
    if (!CreateDescriptorPool(ctx.Device, h.DescriptorPool)) {
        RS_LOG_ERROR("RadianceCascadeHash: descriptor pool failed"); return false;
    }

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = h.SetLayout;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = h.DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(ctx.Device, &ai, h.Sets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("RadianceCascadeHash: alloc descriptor sets failed");
        return false;
    }
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        WriteFrameSet(ctx.Device, h.Sets[i],
                      h.KeyBuffer, h.PayloadBuffer,
                      h.ParamsBuffers[i], h.CellListBuffer,
                      keyBytes, payloadBytes, paramsBytes, cellListBytes);
    }

    h.Initialized = true;
    RS_LOG_INFO("RadianceCascadeHash ready: 2^%u slots (%llu KB keys + %llu MB payload)",
                h.HashLog2,
                static_cast<unsigned long long>(keyBytes / 1024),
                static_cast<unsigned long long>(payloadBytes / (1024 * 1024)));
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
        h.ParamsBuffers[i] = VK_NULL_HANDLE;
        h.ParamsMemory[i]  = VK_NULL_HANDLE;

        if (h.ReadbackMapped[i]) {
            vkUnmapMemory(ctx.Device, h.ReadbackMemory[i]);
            h.ReadbackMapped[i] = nullptr;
        }
        if (h.ReadbackBuffers[i]) vkDestroyBuffer(ctx.Device, h.ReadbackBuffers[i], nullptr);
        if (h.ReadbackMemory[i])  vkFreeMemory   (ctx.Device, h.ReadbackMemory[i],  nullptr);
        h.ReadbackBuffers[i] = VK_NULL_HANDLE;
        h.ReadbackMemory[i]  = VK_NULL_HANDLE;
    }
    if (h.DescriptorPool)  vkDestroyDescriptorPool     (ctx.Device, h.DescriptorPool,  nullptr);
    if (h.SetLayout)       vkDestroyDescriptorSetLayout(ctx.Device, h.SetLayout,       nullptr);
    if (h.CellListBuffer)  vkDestroyBuffer(ctx.Device, h.CellListBuffer,  nullptr);
    if (h.CellListMemory)  vkFreeMemory   (ctx.Device, h.CellListMemory,  nullptr);
    if (h.PayloadBuffer)   vkDestroyBuffer(ctx.Device, h.PayloadBuffer,   nullptr);
    if (h.PayloadMemory)   vkFreeMemory   (ctx.Device, h.PayloadMemory,   nullptr);
    if (h.KeyBuffer)       vkDestroyBuffer(ctx.Device, h.KeyBuffer,       nullptr);
    if (h.KeyMemory)       vkFreeMemory   (ctx.Device, h.KeyMemory,       nullptr);

    h.KeyBuffer = h.PayloadBuffer = h.CellListBuffer = VK_NULL_HANDLE;
    h.KeyMemory = h.PayloadMemory = h.CellListMemory = VK_NULL_HANDLE;
    h.DescriptorPool = VK_NULL_HANDLE;
    h.SetLayout      = VK_NULL_HANDLE;
    h.Sets           = {};
    h.Initialized    = false;
}

void RadianceCascadeHashWriteParams(RadianceCascadeHash& h,
                                    uint32_t frameSlot,
                                    const RcHashParams& params) {
    if (frameSlot >= VulkanContext::kFramesInFlight) return;
    if (!h.ParamsMapped[frameSlot]) return;
    std::memcpy(h.ParamsMapped[frameSlot], &params, sizeof(RcHashParams));
}

void RadianceCascadeHashClearForFrame(const RadianceCascadeHash& h,
                                      VkCommandBuffer cmd) {
    if (!h.Initialized) return;
    const VkDeviceSize keyBytes      =
        static_cast<VkDeviceSize>(h.SlotCount) * sizeof(uint32_t);
    // CellList[0] = counter; clear just the counter slot (4 bytes). The
    // remaining slots are overwritten by atomic appends and the relight
    // shader only reads up to `count`, so leftover stale keys are inert.
    vkCmdFillBuffer(cmd, h.KeyBuffer,      0, keyBytes,         0u);
    vkCmdFillBuffer(cmd, h.CellListBuffer, 0, sizeof(uint32_t), 0u);

    VkBufferMemoryBarrier b[2]{};
    b[0].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[0].buffer = h.KeyBuffer;
    b[0].offset = 0;
    b[0].size   = VK_WHOLE_SIZE;
    b[1] = b[0];
    b[1].buffer = h.CellListBuffer;
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

    // Compute-write on CellListBuffer (the atomic counter) → transfer-read.
    VkBufferMemoryBarrier pre{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    pre.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.buffer              = h.CellListBuffer;
    pre.offset              = 0;
    pre.size                = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &pre, 0, nullptr);

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, h.CellListBuffer, h.ReadbackBuffers[frameSlot],
                    1, &region);

    // Host-coherent: no host-barrier needed beyond the fence wait at BeginFrame.
}

uint32_t RadianceCascadeHashReadProbeCount(const RadianceCascadeHash& h,
                                           uint32_t frameSlot) {
    if (!h.Initialized) return 0;
    if (frameSlot >= VulkanContext::kFramesInFlight) return 0;
    if (!h.ReadbackMapped[frameSlot]) return 0;
    return h.ReadbackMapped[frameSlot][0];
}

} // namespace RS
