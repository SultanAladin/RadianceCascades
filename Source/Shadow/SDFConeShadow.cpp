// Source/Shadow/SDFConeShadow.cpp — Phase 13
#include "Shadow/SDFConeShadow.h"
#include "Renderer/FrameContext.h"
#include "Core/Logger.h"
#include "imgui.h"

#include <cmath>
#include <cstring>

namespace RS {

namespace {

constexpr float kPi = 3.14159265358979323846f;

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

// -- lifecycle ----------------------------------------------------------------

void SDFConeShadow::Initialize(const VulkanContext& ctx,
                               VkFormat /*depthFormat*/,
                               VkExtent2D /*shadowMapResolution*/) {
    m_Ctx = &ctx;
    if (!CreateSampler())        { RS_LOG_ERROR("SDFConeShadow: sampler failed");    return; }
    if (!CreateSetLayout())      { RS_LOG_ERROR("SDFConeShadow: set layout failed"); return; }
    if (!CreateAlgoUbos())       { RS_LOG_ERROR("SDFConeShadow: UBOs failed");       return; }
    if (!CreateDescriptorSets()) { RS_LOG_ERROR("SDFConeShadow: sets failed");       return; }

    // Initial descriptor write uses our linear/clamp sampler against a null
    // view — that's not valid, so SetSDF must be called before the first
    // RecordShadowPass. Main.cpp does this right after Initialize(); until then
    // the algo is "not ready". To keep validation quiet on the very first
    // frame we'll write the sampler with a placeholder view in RewriteAllSets
    // only after SetSDF has been called.
    RS_LOG_INFO("SDFConeShadow ready: awaiting SetSDF() to bind the residency view");
}

void SDFConeShadow::Terminate(const VulkanContext& ctx) {
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

    if (m_LinearClampSampler)
        vkDestroySampler(ctx.Device, m_LinearClampSampler, nullptr);

    m_LinearClampSampler = VK_NULL_HANDLE;
    m_LightingSetLayout  = VK_NULL_HANDLE;
    m_DescriptorPool     = VK_NULL_HANDLE;
    m_FrameSets          = {};
    m_SdfView            = VK_NULL_HANDLE;
    m_SdfSampler         = VK_NULL_HANDLE;
    m_HasSDF             = false;
    m_Ctx                = nullptr;
}

// -- ImGui --------------------------------------------------------------------

void SDFConeShadow::DrawImGuiParams() {
    ImGui::Checkbox  ("Shadows enabled",   &m_Params.Enabled);
    ImGui::SliderFloat("Half-angle (°)",   &m_Params.HalfAngleDeg,  0.05f, 8.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cone half-angle widens the soft penumbra.\n"
                          "Larger ≈ softer shadows; 1.5° ≈ realistic sun.");
    }
    ImGui::SliderFloat("Max distance (m)", &m_Params.MaxDistance,   0.5f, 64.0f);
    ImGui::SliderFloat("Surface offset",   &m_Params.SurfaceOffset, 0.0f, 0.2f, "%.4f");
    ImGui::SliderInt  ("Max steps",        &m_Params.MaxSteps,      8,    128);
    ImGui::SliderFloat("Min step",         &m_Params.MinStep,       0.001f, 0.05f, "%.4f");
    ImGui::SliderFloat("Strength",         &m_Params.Strength,      0.0f, 1.0f);
    ImGui::Separator();
    ImGui::TextDisabled(m_HasSDF
        ? "SDF resident — tracing the active mesh."
        : "No SDF resident — output stays lit (no shadow).");
}

// -- record -------------------------------------------------------------------

void SDFConeShadow::RecordShadowPass(VkCommandBuffer /*cmd*/,
                                     const FrameContext& frame) {
    if (!m_Ctx) return;
    // No render pass to record — the lighting compose itself traces. We only
    // need to refresh the per-frame UBO with the latest sun + tunables.
    UpdateAlgoUbo(frame.FrameSlot, frame);
}

// -- lighting binding ---------------------------------------------------------

void SDFConeShadow::BindLightingDescriptorSet(VkCommandBuffer cmd,
                                              VkPipelineLayout lightingLayout,
                                              uint32_t frameSlot,
                                              uint32_t set) {
    if (!m_FrameSets[frameSlot]) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lightingLayout, set, 1, &m_FrameSets[frameSlot],
                            0, nullptr);
}

// -- residency hook -----------------------------------------------------------

void SDFConeShadow::SetSDF(VkImageView sdfView, VkSampler sampler,
                           const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                           float maxDist, bool hasSDF) {
    m_SdfView    = sdfView;
    m_SdfSampler = sampler;
    m_AabbMin    = aabbMin;
    m_AabbMax    = aabbMax;
    m_MaxDist    = maxDist;
    m_HasSDF     = hasSDF;
    if (!m_Ctx || !m_LightingSetLayout) return;
    RewriteAllSets();
}

void SDFConeShadow::SetInstanceXformBuffer(const VkBuffer* ssbosByFrame,
                                           VkDeviceSize bytesPerSlot) {
    if (!ssbosByFrame || bytesPerSlot == 0) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        m_InstXformBuffers[i] = ssbosByFrame[i];
    }
    m_XformBytes = bytesPerSlot;
    if (!m_Ctx || !m_LightingSetLayout) return;
    RewriteAllSets();
}

// -- one-time GPU resource creation ------------------------------------------

bool SDFConeShadow::CreateSampler() {
    // Mirror GlobalSDF::Sampler's filter/clamp settings — the GLSL sampler is
    // a sampler3D so we use TEXTURE_3D + linear filtering for trilinear taps.
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod       = 0.0f;
    sci.maxLod       = 0.0f;
    return vkCreateSampler(m_Ctx->Device, &sci, nullptr, &m_LinearClampSampler) == VK_SUCCESS;
}

bool SDFConeShadow::CreateSetLayout() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3;
    lci.pBindings    = bindings;
    return vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr, &m_LightingSetLayout) == VK_SUCCESS;
}

bool SDFConeShadow::CreateAlgoUbos() {
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

bool SDFConeShadow::CreateDescriptorSets() {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = VulkanContext::kFramesInFlight;
    sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[2].descriptorCount = VulkanContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 3;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr, &m_DescriptorPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight] = {};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = m_LightingSetLayout;

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = m_DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai, m_FrameSets.data()) != VK_SUCCESS) return false;

    // UBO bindings are stable (only contents move). Image binding (0) is left
    // unwritten until SetSDF() supplies a real view — validation accepts this
    // because we never bind the set before then; Main.cpp calls SetSDF before
    // the first frame.
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorBufferInfo buf{};
        buf.buffer = m_AlgoUboBuffers[i];
        buf.offset = 0;
        buf.range  = sizeof(AlgoUbo);

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = m_FrameSets[i];
        w.dstBinding      = 1;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo     = &buf;
        vkUpdateDescriptorSets(m_Ctx->Device, 1, &w, 0, nullptr);
    }
    return true;
}

void SDFConeShadow::RewriteAllSets() {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(m_Ctx->Device);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkWriteDescriptorSet writes[2]{};
        VkDescriptorImageInfo  img{};
        VkDescriptorBufferInfo buf{};
        uint32_t n = 0;

        if (m_SdfView != VK_NULL_HANDLE) {
            img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            img.imageView   = m_SdfView;
            // Prefer the panel-supplied sampler (GlobalSDF::Sampler) when present;
            // fall back to our own linear/clamp sampler. They're functionally
            // identical — this just avoids a redundant VkSampler.
            img.sampler     = (m_SdfSampler != VK_NULL_HANDLE)
                                  ? m_SdfSampler : m_LinearClampSampler;

            writes[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n].dstSet          = m_FrameSets[i];
            writes[n].dstBinding      = 0;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].pImageInfo      = &img;
            ++n;
        }

        if (m_InstXformBuffers[i] != VK_NULL_HANDLE && m_XformBytes > 0) {
            buf.buffer = m_InstXformBuffers[i];
            buf.offset = 0;
            buf.range  = m_XformBytes;

            writes[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n].dstSet          = m_FrameSets[i];
            writes[n].dstBinding      = 2;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].pBufferInfo     = &buf;
            ++n;
        }

        if (n > 0) {
            vkUpdateDescriptorSets(m_Ctx->Device, n, writes, 0, nullptr);
        }
    }
}

void SDFConeShadow::UpdateAlgoUbo(uint32_t frameSlot, const FrameContext& frame) {
    AlgoUbo ubo{};
    // FrameContext.SunDirection points TOWARD the sun (matches lighting.comp's
    // pc.SunDirAndIntensity convention).
    const float halfAngleRad = m_Params.HalfAngleDeg * (kPi / 180.0f);
    ubo.SunDirAndHalfAngle   = glm::vec4(glm::normalize(frame.SunDirection),
                                         halfAngleRad);
    ubo.AABBMinAndMaxDist    = glm::vec4(m_AabbMin, m_MaxDist);
    ubo.AABBMaxAndSurfaceOff = glm::vec4(m_AabbMax, m_Params.SurfaceOffset);
    ubo.TraceParams          = glm::vec4(static_cast<float>(m_Params.MaxSteps),
                                         m_Params.MaxDistance,
                                         m_Params.MinStep,
                                         m_Params.Strength);
    ubo.Flags                = glm::vec4(m_Params.Enabled ? 1.0f : 0.0f,
                                         m_HasSDF         ? 1.0f : 0.0f,
                                         0.0f, 0.0f);
    std::memcpy(m_AlgoUboMapped[frameSlot], &ubo, sizeof(ubo));
}

} // namespace RS
