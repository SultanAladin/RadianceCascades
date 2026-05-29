// Source/Shadow/PCSSShadow.cpp — Phase 11
#include "Shadow/PCSSShadow.h"
#include "Renderer/FrameContext.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"
#include "imgui.h"

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

} // namespace

PCSSShadow::PCSSShadow()  = default;
PCSSShadow::~PCSSShadow() = default;

void PCSSShadow::Initialize(const VulkanContext& ctx,
                            VkFormat /*depthFormat*/,
                            VkExtent2D /*shadowMapResolution*/) {
    m_Ctx = &ctx;

    if (!ShadowMapInitialize(m_Map, ctx, "Artifacts/Shaders")) {
        RS_LOG_ERROR("PCSSShadow: ShadowMapInitialize failed"); return;
    }
    if (!CreateSamplers())       { RS_LOG_ERROR("PCSSShadow: samplers failed");  return; }
    if (!CreateSetLayout())      { RS_LOG_ERROR("PCSSShadow: layout failed");    return; }
    if (!CreateAlgoUbos())       { RS_LOG_ERROR("PCSSShadow: UBOs failed");      return; }
    if (!CreateDescriptorSets()) { RS_LOG_ERROR("PCSSShadow: sets failed");      return; }

    RS_LOG_INFO("PCSSShadow ready: %u cascades @ %u^2 (blocker-search + variable PCF)",
                ShadowMap::kCascadeCount, ShadowMap::kCascadeSize);
}

void PCSSShadow::Terminate(const VulkanContext& ctx) {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(ctx.Device);

    if (m_DescriptorPool)
        vkDestroyDescriptorPool(ctx.Device, m_DescriptorPool, nullptr);
    if (m_LightingSetLayout)
        vkDestroyDescriptorSetLayout(ctx.Device, m_LightingSetLayout, nullptr);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (m_AlgoUboMapped[i]) {
            vkUnmapMemory(ctx.Device, m_AlgoUboMemory[i]);
            m_AlgoUboMapped[i] = nullptr;
        }
        if (m_AlgoUboBuffers[i]) vkDestroyBuffer(ctx.Device, m_AlgoUboBuffers[i], nullptr);
        if (m_AlgoUboMemory [i]) vkFreeMemory   (ctx.Device, m_AlgoUboMemory [i], nullptr);
    }
    m_AlgoUboBuffers = {};
    m_AlgoUboMemory  = {};

    if (m_CompareSampler)    vkDestroySampler(ctx.Device, m_CompareSampler,    nullptr);
    if (m_NonCompareSampler) vkDestroySampler(ctx.Device, m_NonCompareSampler, nullptr);

    ShadowMapTerminate(m_Map, ctx);

    m_DescriptorPool     = VK_NULL_HANDLE;
    m_LightingSetLayout  = VK_NULL_HANDLE;
    m_CompareSampler     = VK_NULL_HANDLE;
    m_NonCompareSampler  = VK_NULL_HANDLE;
    m_FrameSets          = {};
    m_Ctx                = nullptr;
}

void PCSSShadow::DrawImGuiParams() {
    ImGui::Checkbox  ("Shadows enabled", &m_Params.Enabled);
    ImGui::SliderFloat("Light size (UV)", &m_Params.LightSizeUv, 0.0f, 0.05f, "%.4f");
    ImGui::SliderInt  ("Blocker samples", &m_Params.BlockerSearchSamples, 4, 32);
    ImGui::SliderInt  ("PCF samples",     &m_Params.PcfSamples,           4, 64);
    ImGui::SliderFloat("Min penumbra (texels)", &m_Params.MinPenumbra, 0.0f, 8.0f);
    ImGui::SliderFloat("Depth bias",   &m_Params.DepthBias,  0.0f, 0.01f, "%.5f");
    ImGui::SliderFloat("Normal bias",  &m_Params.NormalBias, 0.0f, 0.1f,  "%.4f");
    ImGui::Separator();
    ImGui::TextDisabled("CSM atlas: 4 x %u^2 D32_SFLOAT (shared layout)", ShadowMap::kCascadeSize);
    ImGui::TextDisabled("Light size widens penumbra with distance from blocker.");
}

void PCSSShadow::RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Ctx || !m_Map.Initialized) return;
    if (!m_Params.Enabled) {
        UpdateAlgoUbo(frame.FrameSlot);
        return;
    }
    ShadowMapBuildCascades(m_Map, frame.Cam, frame.SunDirection);
    UpdateAlgoUbo(frame.FrameSlot);

    if (frame.ScenePtr == nullptr) return;
    const MeshRegistry&     meshes    = SceneMeshes   (*frame.ScenePtr);
    const InstanceRegistry& instances = SceneInstances(*frame.ScenePtr);
    ShadowMapRecord(m_Map, *m_Ctx, cmd, meshes, instances);
}

void PCSSShadow::BindLightingDescriptorSet(VkCommandBuffer cmd,
                                           VkPipelineLayout lightingLayout,
                                           uint32_t frameSlot,
                                           uint32_t set) {
    if (!m_FrameSets[frameSlot]) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lightingLayout, set, 1, &m_FrameSets[frameSlot],
                            0, nullptr);
}

bool PCSSShadow::CreateSamplers() {
    {
        VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sci.compareEnable = VK_TRUE;
        sci.compareOp     = VK_COMPARE_OP_LESS;
        sci.minLod = 0.0f; sci.maxLod = 0.0f;
        if (vkCreateSampler(m_Ctx->Device, &sci, nullptr, &m_CompareSampler) != VK_SUCCESS) return false;
    }
    {
        VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sci.magFilter    = VK_FILTER_NEAREST;   // raw depth reads — no filtering
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sci.compareEnable = VK_FALSE;
        sci.minLod = 0.0f; sci.maxLod = 0.0f;
        if (vkCreateSampler(m_Ctx->Device, &sci, nullptr, &m_NonCompareSampler) != VK_SUCCESS) return false;
    }
    return true;
}

bool PCSSShadow::CreateSetLayout() {
    VkDescriptorSetLayoutBinding b[3]{};
    b[0].binding         = 0;
    b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding         = 1;
    b[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding         = 2;
    b[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[2].descriptorCount = 1;
    b[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr, &m_LightingSetLayout) == VK_SUCCESS;
}

bool PCSSShadow::CreateAlgoUbos() {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = sizeof(AlgoUbo);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (vkCreateBuffer(m_Ctx->Device, &bci, nullptr, &m_AlgoUboBuffers[i]) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_Ctx->Device, m_AlgoUboBuffers[i], &req);
        const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memType < 0) return false;

        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(memType);
        if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_AlgoUboMemory[i]) != VK_SUCCESS) return false;
        vkBindBufferMemory(m_Ctx->Device, m_AlgoUboBuffers[i], m_AlgoUboMemory[i], 0);
        if (vkMapMemory(m_Ctx->Device, m_AlgoUboMemory[i], 0, req.size, 0, &m_AlgoUboMapped[i]) != VK_SUCCESS) return false;
        std::memset(m_AlgoUboMapped[i], 0, sizeof(AlgoUbo));
    }
    return true;
}

bool PCSSShadow::CreateDescriptorSets() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 2 * VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 1 * VulkanContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr, &m_DescriptorPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight] = {};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = m_LightingSetLayout;

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = m_DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai, m_FrameSets.data()) != VK_SUCCESS) return false;

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo imgCompare{};
        imgCompare.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgCompare.imageView   = m_Map.AtlasArrayView;
        imgCompare.sampler     = m_CompareSampler;

        VkDescriptorImageInfo imgNonCompare{};
        imgNonCompare.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgNonCompare.imageView   = m_Map.AtlasArrayView;
        imgNonCompare.sampler     = m_NonCompareSampler;

        VkDescriptorBufferInfo buf{};
        buf.buffer = m_AlgoUboBuffers[i];
        buf.offset = 0;
        buf.range  = sizeof(AlgoUbo);

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = m_FrameSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &imgCompare;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = m_FrameSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo     = &buf;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = m_FrameSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo      = &imgNonCompare;

        vkUpdateDescriptorSets(m_Ctx->Device, 3, writes, 0, nullptr);
    }
    return true;
}

void PCSSShadow::UpdateAlgoUbo(uint32_t frameSlot) {
    AlgoUbo ubo{};
    for (uint32_t c = 0; c < ShadowMap::kCascadeCount; ++c) {
        ubo.CascadeViewProj[c] = m_Map.CascadeViewProj[c];
    }
    ubo.CascadeSplits = glm::vec4(m_Map.CascadeSplitDistance[0],
                                  m_Map.CascadeSplitDistance[1],
                                  m_Map.CascadeSplitDistance[2],
                                  m_Map.CascadeSplitDistance[3]);
    const float enabledFlag = m_Params.Enabled ? 1.0f : 0.0f;
    ubo.PCSSParamsVec = glm::vec4(m_Params.LightSizeUv * enabledFlag,
                                  m_Params.DepthBias,
                                  m_Params.NormalBias,
                                  m_Params.MinPenumbra);
    const uint32_t blocker = static_cast<uint32_t>(std::clamp(m_Params.BlockerSearchSamples, 1, 64));
    const uint32_t pcf     = static_cast<uint32_t>(std::clamp(m_Params.PcfSamples,           1, 64));
    const uint32_t packed  = (blocker & 0xFFFFu) | (pcf << 16);
    float packedAsFloat;
    std::memcpy(&packedAsFloat, &packed, sizeof(float));
    ubo.AtlasParams = glm::vec4(1.0f / static_cast<float>(ShadowMap::kCascadeSize),
                                static_cast<float>(ShadowMap::kCascadeSize),
                                static_cast<float>(ShadowMap::kCascadeCount),
                                packedAsFloat);
    std::memcpy(m_AlgoUboMapped[frameSlot], &ubo, sizeof(ubo));
}

} // namespace RS
