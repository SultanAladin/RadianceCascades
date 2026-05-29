// Source/Shadow/PCFShadow.cpp — Phase 10
#include "Shadow/PCFShadow.h"
#include "Renderer/FrameContext.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"
#include "imgui.h"

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

PCFShadow::PCFShadow()  = default;
PCFShadow::~PCFShadow() = default;

void PCFShadow::Initialize(const VulkanContext& ctx,
                           VkFormat /*depthFormat*/,
                           VkExtent2D /*shadowMapResolution*/) {
    m_Ctx = &ctx;

    // Shadow atlas + depth-only pipeline. ShadowMap loads shadow_csm_vert.spv
    // relative to "Artifacts/Shaders" — match the convention every other pass
    // uses.
    if (!ShadowMapInitialize(m_Map, ctx, "Artifacts/Shaders")) {
        RS_LOG_ERROR("PCFShadow: ShadowMapInitialize failed");
        return;
    }

    if (!CreateSampler())        { RS_LOG_ERROR("PCFShadow: sampler failed");   return; }
    if (!CreateSetLayout())      { RS_LOG_ERROR("PCFShadow: set layout failed"); return; }
    if (!CreateAlgoUbos())       { RS_LOG_ERROR("PCFShadow: UBOs failed");      return; }
    if (!CreateDescriptorSets()) { RS_LOG_ERROR("PCFShadow: sets failed");      return; }

    RS_LOG_INFO("PCFShadow ready: %u cascades @ %u^2 D32_SFLOAT",
                ShadowMap::kCascadeCount, ShadowMap::kCascadeSize);
}

void PCFShadow::Terminate(const VulkanContext& ctx) {
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

    if (m_CompareSampler) vkDestroySampler(ctx.Device, m_CompareSampler, nullptr);

    ShadowMapTerminate(m_Map, ctx);

    m_DescriptorPool    = VK_NULL_HANDLE;
    m_LightingSetLayout = VK_NULL_HANDLE;
    m_CompareSampler    = VK_NULL_HANDLE;
    m_FrameSets         = {};
    m_Ctx               = nullptr;
}

// -- ImGui --------------------------------------------------------------------

void PCFShadow::DrawImGuiParams() {
    ImGui::Checkbox  ("Shadows enabled", &m_Params.Enabled);
    ImGui::SliderInt ("Kernel radius",   &m_Params.KernelRadius, 0, 3);
    ImGui::TextDisabled("  0 = 1 tap, 1 = 3x3, 2 = 5x5, 3 = 7x7");
    ImGui::SliderFloat("Depth bias",     &m_Params.DepthBias,     0.0f, 0.01f, "%.5f");
    ImGui::SliderFloat("Normal bias",    &m_Params.NormalBias,    0.0f, 0.1f,  "%.4f");
    ImGui::SliderFloat("Jitter strength",&m_Params.JitterStrength,0.0f, 2.0f);
    ImGui::Separator();
    ImGui::TextDisabled("CSM atlas: 4 x %u^2 D32_SFLOAT", ShadowMap::kCascadeSize);
    ImGui::TextDisabled("Splits: 10%% / 25%% / 50%% / 100%% of frustum");
}

// -- record -------------------------------------------------------------------

void PCFShadow::RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Ctx || !m_Map.Initialized) return;
    if (!m_Params.Enabled) {
        // Disabled = still need to update the UBO so the lighting shader's
        // shadow factor stays at 1.0 (the compose shader gates on
        // PCFParamsVec.w going to zero).
        UpdateAlgoUbo(frame.FrameSlot);
        return;
    }

    // 1) Re-fit cascades against the live camera + sun.
    ShadowMapBuildCascades(m_Map, frame.Cam, frame.SunDirection);

    // 2) Push the algo UBO for the lighting compose to consume.
    UpdateAlgoUbo(frame.FrameSlot);

    // 3) Render each cascade. The render pass's finalLayout pins the atlas in
    //    SHADER_READ_ONLY_OPTIMAL on the way out.
    if (frame.ScenePtr == nullptr) return;
    const MeshRegistry&     meshes    = SceneMeshes   (*frame.ScenePtr);
    const InstanceRegistry& instances = SceneInstances(*frame.ScenePtr);

    ShadowMapRecord(m_Map, *m_Ctx, cmd, meshes, instances);
}

// -- lighting binding ---------------------------------------------------------

void PCFShadow::BindLightingDescriptorSet(VkCommandBuffer cmd,
                                          VkPipelineLayout lightingLayout,
                                          uint32_t frameSlot,
                                          uint32_t set) {
    if (!m_FrameSets[frameSlot]) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lightingLayout, set, 1, &m_FrameSets[frameSlot],
                            0, nullptr);
}

// -- one-time GPU resource creation ------------------------------------------

bool PCFShadow::CreateSampler() {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;   // outside cascade = lit
    sci.compareEnable = VK_TRUE;
    sci.compareOp     = VK_COMPARE_OP_LESS;
    sci.minLod       = 0.0f;
    sci.maxLod       = 0.0f;
    return vkCreateSampler(m_Ctx->Device, &sci, nullptr, &m_CompareSampler) == VK_SUCCESS;
}

bool PCFShadow::CreateSetLayout() {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 2;
    lci.pBindings    = bindings;
    return vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr, &m_LightingSetLayout) == VK_SUCCESS;
}

bool PCFShadow::CreateAlgoUbos() {
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

bool PCFShadow::CreateDescriptorSets() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = VulkanContext::kFramesInFlight;

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

    // Both bindings are stable across frames (the atlas view and the per-
    // frame UBO addresses don't move), so we write once at init.
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo img{};
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img.imageView   = m_Map.AtlasArrayView;
        img.sampler     = m_CompareSampler;

        VkDescriptorBufferInfo buf{};
        buf.buffer = m_AlgoUboBuffers[i];
        buf.offset = 0;
        buf.range  = sizeof(AlgoUbo);

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = m_FrameSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &img;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = m_FrameSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo     = &buf;

        vkUpdateDescriptorSets(m_Ctx->Device, 2, writes, 0, nullptr);
    }
    return true;
}

void PCFShadow::UpdateAlgoUbo(uint32_t frameSlot) {
    AlgoUbo ubo{};
    for (uint32_t c = 0; c < ShadowMap::kCascadeCount; ++c) {
        ubo.CascadeViewProj[c] = m_Map.CascadeViewProj[c];
    }
    ubo.CascadeSplits = glm::vec4(m_Map.CascadeSplitDistance[0],
                                  m_Map.CascadeSplitDistance[1],
                                  m_Map.CascadeSplitDistance[2],
                                  m_Map.CascadeSplitDistance[3]);
    ubo.PCFParamsVec  = glm::vec4(static_cast<float>(m_Params.KernelRadius),
                                  m_Params.DepthBias,
                                  m_Params.NormalBias,
                                  m_Params.Enabled ? m_Params.JitterStrength : 0.0f);
    ubo.AtlasParams   = glm::vec4(1.0f / static_cast<float>(ShadowMap::kCascadeSize),
                                  static_cast<float>(ShadowMap::kCascadeSize),
                                  static_cast<float>(ShadowMap::kCascadeCount),
                                  0.0f);
    std::memcpy(m_AlgoUboMapped[frameSlot], &ubo, sizeof(ubo));
}

} // namespace RS
