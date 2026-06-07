// Source/GI/RadianceCascades.cpp — RC rewrite #2, pass 1 host
#include "GI/RadianceCascades.h"
#include "SDF/GlobalSDF.h"
#include "Core/Logger.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace RS {

namespace {

// Mirrors SDFConeShadow / GlobalSDF: linear scan of the device's memory types.
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

uint32_t Log2Ceil(uint32_t v) {
    uint32_t r = 0;
    while ((1u << r) < v) ++r;
    return r;
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool LoadSpirv(const char* path, std::vector<char>& out) {
    std::ifstream stream(path, std::ios::ate | std::ios::binary);
    if (!stream.is_open()) return false;
    const size_t bytes = static_cast<size_t>(stream.tellg());
    out.resize(bytes);
    stream.seekg(0);
    stream.read(out.data(), static_cast<std::streamsize>(bytes));
    return bytes > 0;
}

// Create a host-visible coherent UBO and persistently map it.
bool CreateHostBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkBuffer& buffer, VkDeviceMemory& memory, void** mapped) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = size;
    bci.usage = usage;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, buffer, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType < 0) return false;

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(ctx.Device, buffer, memory, 0);

    if (mapped) {
        if (vkMapMemory(ctx.Device, memory, 0, size, 0, mapped) != VK_SUCCESS) return false;
        std::memset(*mapped, 0, size);
    }
    return true;
}

} // namespace

//------------------------------------------------------------------------------------------------------------------------
//                                                    HIERARCHY + LAYOUTS
//------------------------------------------------------------------------------------------------------------------------

static void ResolveLevels(RadianceCascades& rc) {
    const std::vector<Cascades::CascadeDescriptor> hierarchy = Cascades::ResolveHierarchy(rc.Spec);
    rc.CascadeCount = static_cast<uint32_t>(hierarchy.size());
    if (rc.CascadeCount > kMaxCascades) rc.CascadeCount = kMaxCascades;   // atlases capped by the array
    for (uint32_t c = 0; c < rc.CascadeCount; ++c) rc.Levels[c] = hierarchy[c];
}

static bool CreateSetLayouts(RadianceCascades& rc) {
    VkDevice device = rc.Ctx->Device;

    // set 0: sparse SDF (idx, pool, params, instance-xform).
    VkDescriptorSetLayoutBinding sparse[4]{};
    sparse[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    sparse[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    sparse[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    sparse[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo sci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    sci.bindingCount = 4;
    sci.pBindings    = sparse;
    if (vkCreateDescriptorSetLayout(device, &sci, nullptr, &rc.SetLayoutSparse) != VK_SUCCESS) return false;

    // set 1: cascade build (build UBO, lighting UBO, atlas storage image).
    VkDescriptorSetLayoutBinding cascade[3]{};
    cascade[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    cascade[1] = { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    cascade[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo cci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    cci.bindingCount = 3;
    cci.pBindings    = cascade;
    if (vkCreateDescriptorSetLayout(device, &cci, nullptr, &rc.SetLayoutCascade) != VK_SUCCESS) return false;

    return true;
}

static bool CreateDummyBuffer(RadianceCascades& rc) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = 16;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(rc.Ctx->Device, &bci, nullptr, &rc.DummyBuffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(rc.Ctx->Device, rc.DummyBuffer, &req);
    const int memType = FindMemoryType(rc.Ctx->PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(rc.Ctx->Device, &mai, nullptr, &rc.DummyMemory) != VK_SUCCESS) return false;
    vkBindBufferMemory(rc.Ctx->Device, rc.DummyBuffer, rc.DummyMemory, 0);
    return true;
}

static bool CreateUboRings(RadianceCascades& rc) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(rc.Ctx->PhysicalDevice, &props);
    rc.CascadeBuildStride = AlignUp(sizeof(CascadeBuildParams), props.limits.minUniformBufferOffsetAlignment);

    for (RcFrameResources& frame : rc.Frames) {
        if (!CreateHostBuffer(*rc.Ctx, rc.CascadeBuildStride * kMaxCascades,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              frame.CascadeBuildBuffer, frame.CascadeBuildMemory, &frame.CascadeBuildMapped))
            return false;
        if (!CreateHostBuffer(*rc.Ctx, sizeof(RcLightingParams),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              frame.LightingBuffer, frame.LightingMemory, &frame.LightingMapped))
            return false;
        if (!CreateHostBuffer(*rc.Ctx, sizeof(RcSparseParams),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              frame.SparseParamsBuffer, frame.SparseParamsMemory, &frame.SparseParamsMapped))
            return false;
    }
    return true;
}

static bool CreateDescriptorPoolAndSets(RadianceCascades& rc) {
    VkDevice device = rc.Ctx->Device;
    const uint32_t frames = VulkanContext::kFramesInFlight;

    // Per frame: set 0 (2 SSBO idx/pool + 1 SSBO xform + 1 UBO params) + set 1 x kMaxCascades (2 UBO + 1 image).
    VkDescriptorPoolSize sizes[3]{};
    sizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames * 3 };                         // idx + pool + xform
    sizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames * (1 + 2 * kMaxCascades) };    // sparse params + build/lighting per cascade
    sizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  frames * kMaxCascades };              // one atlas image per cascade

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = frames * (1 + kMaxCascades);
    pci.poolSizeCount = 3;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &rc.DescriptorPool) != VK_SUCCESS) return false;

    for (RcFrameResources& frame : rc.Frames) {
        VkDescriptorSetAllocateInfo sa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        sa.descriptorPool     = rc.DescriptorPool;
        sa.descriptorSetCount = 1;
        sa.pSetLayouts        = &rc.SetLayoutSparse;
        if (vkAllocateDescriptorSets(device, &sa, &frame.SparseSet) != VK_SUCCESS) return false;

        VkDescriptorSetLayout cascadeLayouts[kMaxCascades];
        for (uint32_t c = 0; c < kMaxCascades; ++c) cascadeLayouts[c] = rc.SetLayoutCascade;

        VkDescriptorSetAllocateInfo ca{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ca.descriptorPool     = rc.DescriptorPool;
        ca.descriptorSetCount = kMaxCascades;
        ca.pSetLayouts        = cascadeLayouts;
        if (vkAllocateDescriptorSets(device, &ca, frame.CascadeSets.data()) != VK_SUCCESS) return false;

        // Static set-0 UBO binding (binding 2 = sparse params); SSBOs (0,1,3) written by RewriteSparseSets.
        VkDescriptorBufferInfo paramInfo{ frame.SparseParamsBuffer, 0, sizeof(RcSparseParams) };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = frame.SparseSet;
        w.dstBinding = 2;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &paramInfo;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    return true;
}

static bool CreatePipeline(RadianceCascades& rc) {
    VkDevice device = rc.Ctx->Device;

    std::vector<char> spirv;
    if (!LoadSpirv("Artifacts/Shaders/rc_build.comp.spv", spirv)) {
        RS_LOG_ERROR("RadianceCascades: rc_build.comp.spv not found in Artifacts/Shaders");
        return false;
    }

    VkShaderModuleCreateInfo mci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    mci.codeSize = spirv.size();
    mci.pCode    = reinterpret_cast<const uint32_t*>(spirv.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &mci, nullptr, &module) != VK_SUCCESS) return false;

    VkDescriptorSetLayout setLayouts[2] = { rc.SetLayoutSparse, rc.SetLayoutCascade };
    VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    lci.setLayoutCount = 2;
    lci.pSetLayouts    = setLayouts;
    if (vkCreatePipelineLayout(device, &lci, nullptr, &rc.BuildPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, module, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage  = stage;
    cpci.layout = rc.BuildPipelineLayout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &rc.BuildPipeline);
    vkDestroyShaderModule(device, module, nullptr);
    return r == VK_SUCCESS;
}

bool RadianceCascadesInitialize(RadianceCascades& rc, const VulkanContext& ctx) {
    rc.Ctx = &ctx;
    ResolveLevels(rc);

    if (!Cascades::ConfirmAngularAxisPowerOfTwo(rc.Spec))
        RS_LOG_ERROR("RadianceCascades: BaseAngularAxisCount is not a power of two — parent merge will break (B2)");
    if (!Cascades::ConfirmTopCascadeCoversRegion(rc.Spec))
        RS_LOG_ERROR("RadianceCascades: top cascade does not cover the region diameter (A4)");

    if (!CreateSetLayouts(rc))           { RS_LOG_ERROR("RadianceCascades: set layouts failed");    return false; }
    if (!CreateDummyBuffer(rc))          { RS_LOG_ERROR("RadianceCascades: dummy buffer failed");   return false; }
    if (!CreateUboRings(rc))             { RS_LOG_ERROR("RadianceCascades: UBO rings failed");      return false; }
    if (!CreateDescriptorPoolAndSets(rc)){ RS_LOG_ERROR("RadianceCascades: descriptor sets failed");return false; }
    if (!CreatePipeline(rc))             { RS_LOG_ERROR("RadianceCascades: pipeline failed");       return false; }

    RS_LOG_INFO("RadianceCascades initialized (%u cascades, awaiting atlases + SetSDF)", rc.CascadeCount);
    return true;
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    ATLAS CONSTRUCTION
//------------------------------------------------------------------------------------------------------------------------

// One-shot UNDEFINED -> GENERAL transition for every atlas (mirrors GlobalSDF::SubmitOneShot pattern).
static void TransitionAtlasesToGeneral(RadianceCascades& rc) {
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool        = rc.Ctx->CommandPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(rc.Ctx->Device, &cai, &cmd) != VK_SUCCESS) return;

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    for (uint32_t c = 0; c < rc.CascadeCount; ++c) {
        CascadeAtlas& atlas = rc.Atlases[c];
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = atlas.Image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        atlas.LayoutNow = VK_IMAGE_LAYOUT_GENERAL;
    }

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(rc.Ctx->GraphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(rc.Ctx->GraphicsQueue);
    vkFreeCommandBuffers(rc.Ctx->Device, rc.Ctx->CommandPool, 1, &cmd);
}

static void WriteCascadeDescriptors(RadianceCascades& rc) {
    VkDevice device = rc.Ctx->Device;
    for (RcFrameResources& frame : rc.Frames) {
        for (uint32_t c = 0; c < rc.CascadeCount; ++c) {
            VkDescriptorBufferInfo buildInfo{ frame.CascadeBuildBuffer, rc.CascadeBuildStride * c, sizeof(CascadeBuildParams) };
            VkDescriptorBufferInfo lightInfo{ frame.LightingBuffer, 0, sizeof(RcLightingParams) };
            VkDescriptorImageInfo  atlasInfo{};
            atlasInfo.imageView   = rc.Atlases[c].View;
            atlasInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet w[3]{};
            w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.CascadeSets[c], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buildInfo, nullptr };
            w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.CascadeSets[c], 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightInfo, nullptr };
            w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.CascadeSets[c], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &atlasInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
        }
    }
}

bool RadianceCascadesConstructAtlases(RadianceCascades& rc, const VulkanContext& ctx) {
    VkDevice device = ctx.Device;

    for (uint32_t c = 0; c < rc.CascadeCount; ++c) {
        const Cascades::CascadeDescriptor& level = rc.Levels[c];
        CascadeAtlas& atlas = rc.Atlases[c];
        atlas.Extent = glm::uvec3(level.AtlasWidth, level.AtlasHeight, level.AtlasDepth);

        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType     = VK_IMAGE_TYPE_3D;
        ici.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent        = { atlas.Extent.x, atlas.Extent.y, atlas.Extent.z };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &atlas.Image) != VK_SUCCESS) {
            RS_LOG_ERROR("RadianceCascades: atlas image create failed (cascade %u)", c);
            return false;
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, atlas.Image, &req);
        const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memType < 0) return false;

        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(memType);
        if (vkAllocateMemory(device, &mai, nullptr, &atlas.Memory) != VK_SUCCESS) return false;
        vkBindImageMemory(device, atlas.Image, atlas.Memory, 0);

        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image                       = atlas.Image;
        vci.viewType                    = VK_IMAGE_VIEW_TYPE_3D;
        vci.format                      = VK_FORMAT_R16G16B16A16_SFLOAT;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &vci, nullptr, &atlas.View) != VK_SUCCESS) return false;

        atlas.LayoutNow = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    TransitionAtlasesToGeneral(rc);
    WriteCascadeDescriptors(rc);
    RS_LOG_INFO("RadianceCascades atlases constructed (%u cascades, c0 = %ux%ux%u)",
                rc.CascadeCount, rc.Atlases[0].Extent.x, rc.Atlases[0].Extent.y, rc.Atlases[0].Extent.z);
    return true;
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    SPARSE RESIDENCY (push model)
//------------------------------------------------------------------------------------------------------------------------

// Rewrite set-0 SSBOs (idx, pool, instance-xform) across all frames under device-idle.
static void RewriteSparseSets(RadianceCascades& rc) {
    if (!rc.Ctx || !rc.SetLayoutSparse) return;
    vkDeviceWaitIdle(rc.Ctx->Device);

    for (RcFrameResources& frame : rc.Frames) {
        VkDescriptorBufferInfo idxInfo{}, poolInfo{}, xformInfo{};
        idxInfo.buffer  = (rc.HasSDF && rc.IndexBuffer) ? rc.IndexBuffer : rc.DummyBuffer;
        idxInfo.offset  = 0;
        idxInfo.range   = (rc.HasSDF && rc.IndexBuffer) ? rc.IndexBytes : 16;

        poolInfo.buffer = (rc.HasSDF && rc.PoolBuffer) ? rc.PoolBuffer : rc.DummyBuffer;
        poolInfo.offset = 0;
        poolInfo.range  = (rc.HasSDF && rc.PoolBuffer) ? rc.PoolBytes : 16;

        VkWriteDescriptorSet writes[3]{};
        uint32_t n = 0;
        writes[n++] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.SparseSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &idxInfo,  nullptr };
        writes[n++] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.SparseSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &poolInfo, nullptr };

        const uint32_t slot = static_cast<uint32_t>(&frame - rc.Frames.data());
        if (rc.InstXformBuffers[slot] != VK_NULL_HANDLE && rc.InstXformBytes > 0) {
            xformInfo.buffer = rc.InstXformBuffers[slot];
            xformInfo.offset = 0;
            xformInfo.range  = rc.InstXformBytes;
            writes[n++] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.SparseSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &xformInfo, nullptr };
        } else {
            // Bind the dummy so binding 3 is never left unwritten (the shader indexes it; an
            // unwritten storage binding trips validation even if InstanceIndex never selects it).
            xformInfo.buffer = rc.DummyBuffer;
            xformInfo.offset = 0;
            xformInfo.range  = 16;
            writes[n++] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.SparseSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &xformInfo, nullptr };
        }

        vkUpdateDescriptorSets(rc.Ctx->Device, n, writes, 0, nullptr);
    }
}

void RadianceCascadesSetSDF(RadianceCascades& rc, const ResidentSparseSDF* sparse,
                            bool hasSDF, uint32_t instanceIndex) {
    if (sparse && hasSDF) {
        rc.HasSDF        = true;
        rc.IndexBuffer   = sparse->IndexBuffer;
        rc.PoolBuffer    = sparse->PoolBuffer;
        rc.IndexBytes    = sparse->IndexBytes;
        rc.PoolBytes     = sparse->PoolBytes;
        rc.InstanceIndex = instanceIndex;

        const uint32_t bs     = sparse->BrickSize ? sparse->BrickSize : 8u;
        const float    decode = sparse->MaxDist > 0.0f ? (sparse->MaxDist / 32767.0f) : 0.0f;
        rc.SparseParams.AABBMin = glm::vec4(sparse->AABBMin, sparse->MaxDist);
        rc.SparseParams.AABBMax = glm::vec4(sparse->AABBMax, decode);
        rc.SparseParams.Dims    = glm::uvec4(sparse->Resolution, bs, sparse->BrickGrid, Log2Ceil(bs));
    } else {
        rc.HasSDF        = false;
        rc.IndexBuffer   = VK_NULL_HANDLE;
        rc.PoolBuffer    = VK_NULL_HANDLE;
        rc.IndexBytes    = 0;
        rc.PoolBytes     = 0;
        rc.SparseParams  = {};
        rc.InstanceIndex = 0;
    }

    // Refresh the mapped sparse-params UBO on every frame slot, then rewrite SSBO bindings.
    for (RcFrameResources& frame : rc.Frames) {
        if (frame.SparseParamsMapped)
            std::memcpy(frame.SparseParamsMapped, &rc.SparseParams, sizeof(RcSparseParams));
    }
    RewriteSparseSets(rc);
}

void RadianceCascadesSetInstanceXformBuffer(RadianceCascades& rc,
                                            const VkBuffer* ssbosByFrame,
                                            VkDeviceSize bytesPerSlot) {
    if (!ssbosByFrame || bytesPerSlot == 0) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i)
        rc.InstXformBuffers[i] = ssbosByFrame[i];
    rc.InstXformBytes = bytesPerSlot;
    RewriteSparseSets(rc);
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    RECORD BUILD
//------------------------------------------------------------------------------------------------------------------------

void RadianceCascadesRecordBuild(RadianceCascades& rc,
                                 VkCommandBuffer cmd,
                                 uint32_t frameSlot,
                                 const glm::vec3& cameraEyeWorld,
                                 const glm::vec3& sunDirectionWorld,
                                 const glm::vec3& sunColour) {
    RcFrameResources& frame = rc.Frames[frameSlot];

    // Lighting UBO (shared across cascades).
    RcLightingParams lighting{};
    lighting.SunDirectionWorld = glm::vec4(glm::normalize(sunDirectionWorld), 0.0f);
    lighting.SunColour         = glm::vec4(sunColour, 0.0f);
    lighting.AmbientColour     = glm::vec4(0.03f, 0.035f, 0.045f, 0.0f);   // faint sky ambient; tune later
    lighting.SurfaceAlbedo     = glm::vec4(0.8f, 0.8f, 0.8f, 0.0f);        // placeholder constant albedo
    std::memcpy(frame.LightingMapped, &lighting, sizeof(RcLightingParams));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rc.BuildPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rc.BuildPipelineLayout,
                            0, 1, &frame.SparseSet, 0, nullptr);

    for (uint32_t c = 0; c < rc.CascadeCount; ++c) {
        const Cascades::CascadeDescriptor& level = rc.Levels[c];
        const float pitch = Cascades::ResolveBaseProbePitch(rc.Spec) * std::pow(2.0f, static_cast<float>(c));

        // Clipmap snap: region origin = camera - radius, rounded down to the probe pitch.
        const glm::vec3 rawOrigin     = cameraEyeWorld - glm::vec3(kRcQueryRadiusMetres);
        const glm::vec3 snappedOrigin = glm::floor(rawOrigin / pitch) * pitch;

        CascadeBuildParams build{};
        build.RegionOriginWorld = glm::vec4(snappedOrigin, pitch);
        build.IntervalMetres    = glm::vec4(level.IntervalStart,
                                            level.IntervalEnd,
                                            0.01f,    // MinStepWorld (metres)
                                            1.0f);    // ambientScalar
        build.GridDims          = glm::uvec4(level.ProbeAxisCount, level.OctSide, c, rc.InstanceIndex);

        std::memcpy(static_cast<char*>(frame.CascadeBuildMapped) + rc.CascadeBuildStride * c,
                    &build, sizeof(CascadeBuildParams));

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rc.BuildPipelineLayout,
                                1, 1, &frame.CascadeSets[c], 0, nullptr);

        // Flattened-atlas dispatch: X/Y fold probe*OctSide+dir, Z = ProbeAxisCount (<= 64).
        const uint32_t groupsX = (level.AtlasWidth  + 7u) / 8u;
        const uint32_t groupsY = (level.AtlasHeight + 7u) / 8u;
        const uint32_t groupsZ = level.AtlasDepth;
        vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
    }

    // Write -> read barrier so a later debug/merge pass can sample the atlases.
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    DEBUG SLICE BLIT
//------------------------------------------------------------------------------------------------------------------------

void RadianceCascadesRecordDebugSlice(RadianceCascades& rc,
                                      VkCommandBuffer cmd,
                                      VkImage destination,
                                      VkImageLayout destinationLayout,
                                      uint32_t destinationWidth,
                                      uint32_t destinationHeight,
                                      uint32_t sliceZ) {
    CascadeAtlas& source = rc.Atlases[0];
    if (source.Image == VK_NULL_HANDLE) return;

    const uint32_t z = (sliceZ < source.Extent.z) ? sliceZ : (source.Extent.z - 1);

    VkImageBlit blit{};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[0]  = { 0, 0, static_cast<int32_t>(z) };
    blit.srcOffsets[1]  = { static_cast<int32_t>(source.Extent.x), static_cast<int32_t>(source.Extent.y), static_cast<int32_t>(z + 1) };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstOffsets[0]  = { 0, 0, 0 };
    blit.dstOffsets[1]  = { static_cast<int32_t>(destinationWidth), static_cast<int32_t>(destinationHeight), 1 };

    vkCmdBlitImage(cmd,
                   source.Image, VK_IMAGE_LAYOUT_GENERAL,
                   destination,  destinationLayout,
                   1, &blit, VK_FILTER_NEAREST);
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    TERMINATE
//------------------------------------------------------------------------------------------------------------------------

void RadianceCascadesTerminate(RadianceCascades& rc, const VulkanContext& ctx) {
    VkDevice device = ctx.Device;
    vkDeviceWaitIdle(device);

    for (CascadeAtlas& atlas : rc.Atlases) {
        if (atlas.View)   vkDestroyImageView(device, atlas.View, nullptr);
        if (atlas.Image)  vkDestroyImage(device, atlas.Image, nullptr);
        if (atlas.Memory) vkFreeMemory(device, atlas.Memory, nullptr);
        atlas = CascadeAtlas{};
    }

    for (RcFrameResources& frame : rc.Frames) {
        if (frame.CascadeBuildMapped) vkUnmapMemory(device, frame.CascadeBuildMemory);
        if (frame.LightingMapped)     vkUnmapMemory(device, frame.LightingMemory);
        if (frame.SparseParamsMapped) vkUnmapMemory(device, frame.SparseParamsMemory);
        if (frame.CascadeBuildBuffer) vkDestroyBuffer(device, frame.CascadeBuildBuffer, nullptr);
        if (frame.CascadeBuildMemory) vkFreeMemory(device, frame.CascadeBuildMemory, nullptr);
        if (frame.LightingBuffer)     vkDestroyBuffer(device, frame.LightingBuffer, nullptr);
        if (frame.LightingMemory)     vkFreeMemory(device, frame.LightingMemory, nullptr);
        if (frame.SparseParamsBuffer) vkDestroyBuffer(device, frame.SparseParamsBuffer, nullptr);
        if (frame.SparseParamsMemory) vkFreeMemory(device, frame.SparseParamsMemory, nullptr);
        frame = RcFrameResources{};
    }

    if (rc.DummyBuffer)         vkDestroyBuffer(device, rc.DummyBuffer, nullptr);
    if (rc.DummyMemory)         vkFreeMemory(device, rc.DummyMemory, nullptr);
    if (rc.BuildPipeline)       vkDestroyPipeline(device, rc.BuildPipeline, nullptr);
    if (rc.BuildPipelineLayout) vkDestroyPipelineLayout(device, rc.BuildPipelineLayout, nullptr);
    if (rc.DescriptorPool)      vkDestroyDescriptorPool(device, rc.DescriptorPool, nullptr);
    if (rc.SetLayoutCascade)    vkDestroyDescriptorSetLayout(device, rc.SetLayoutCascade, nullptr);
    if (rc.SetLayoutSparse)     vkDestroyDescriptorSetLayout(device, rc.SetLayoutSparse, nullptr);

    rc.DummyBuffer = VK_NULL_HANDLE; rc.DummyMemory = VK_NULL_HANDLE;
    rc.BuildPipeline = VK_NULL_HANDLE; rc.BuildPipelineLayout = VK_NULL_HANDLE;
    rc.DescriptorPool = VK_NULL_HANDLE;
    rc.SetLayoutCascade = VK_NULL_HANDLE; rc.SetLayoutSparse = VK_NULL_HANDLE;
    rc.HasSDF = false;
    rc.Ctx = nullptr;
}

} // namespace RS
