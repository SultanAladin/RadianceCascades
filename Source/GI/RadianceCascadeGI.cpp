// Source/GI/RadianceCascadeGI.cpp — Phase 14b
#include "GI/RadianceCascadeGI.h"
#include "Renderer/OffscreenTargets.h"
#include "Renderer/FrameContext.h"
#include "SDF/GlobalSDF.h"
#include "Core/Logger.h"

#include "imgui.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace RS {

namespace {

bool ReadFile(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("RC GI: cannot open shader: %s", path); return false; }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0) { std::fclose(f); return false; }
    out.resize(static_cast<size_t>(len));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

VkShaderModule LoadModule(VkDevice device, const std::string& path) {
    std::vector<char> buf;
    if (!ReadFile(path.c_str(), buf)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = buf.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

bool BuildComputePipeline(VkDevice device, VkPipelineLayout layout,
                          const std::string& spvPath, VkPipeline& outPipeline) {
    VkShaderModule m = LoadModule(device, spvPath);
    if (!m) { RS_LOG_ERROR("RC GI: shader load failed: %s", spvPath.c_str()); return false; }
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = m;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage  = stage;
    cpci.layout = layout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &outPipeline);
    vkDestroyShaderModule(device, m, nullptr);
    return r == VK_SUCCESS;
}

} // namespace

// ---------------- public API ----------------------------------------------

void RadianceCascadeGI::SetFrameResources(const OffscreenTargets* targets,
                                          const GlobalSDF* sdf,
                                          const char* shaderArtifactsDir) {
    m_Targets   = targets;
    m_GlobalSdf = sdf;
    m_ShaderDir = shaderArtifactsDir ? shaderArtifactsDir : "Artifacts/Shaders";
}

void RadianceCascadeGI::SetSDFView(VkImageView view, VkSampler sampler,
                                   const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                                   float maxDist, bool hasSDF) {
    m_SdfView  = view;
    if (sampler != VK_NULL_HANDLE) m_SdfSampler = sampler;
    m_AabbMin  = aabbMin;
    m_AabbMax  = aabbMax;
    m_MaxDist  = maxDist;
    m_HasSDF   = hasSDF;
    if (m_RelightSdfSet != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_Ctx->Device);
        RewriteSdfDescriptors();
    }
}

void RadianceCascadeGI::Initialize(const VulkanContext& ctx, const GlobalSDF& sdf) {
    m_Ctx       = &ctx;
    m_GlobalSdf = &sdf;
    m_SdfSampler= sdf.Sampler;
    m_SdfView   = sdf.DummyView;     // safe default until SetSDFView lands real one

    const int  log2Want = m_Settings ? m_Settings->HashLog2 : 20;
    const uint32_t log2 = static_cast<uint32_t>(std::max(16, std::min(22, log2Want)));
    if (!RadianceCascadeHashInitialize(m_Hash, ctx, log2)) {
        RS_LOG_ERROR("RC GI: hash core init failed");
        return;
    }

    if (!CreateGBufferDescriptors()) return;
    if (!CreateSdfDescriptors())     return;
    if (!CreateLightHdrDescriptors())return;
    if (!CreateInsertPipeline())     return;
    if (!CreateRelightPipeline())    return;
    if (!CreateComposePipeline())    return;

    RS_LOG_INFO("RadianceCascadeGI initialized: hash 2^%u", m_Hash.HashLog2);
}

void RadianceCascadeGI::Terminate() {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(m_Ctx->Device);

    auto destroyPipeline = [&](VkPipeline& p, VkPipelineLayout& pl) {
        if (p)  vkDestroyPipeline      (m_Ctx->Device, p, nullptr);
        if (pl) vkDestroyPipelineLayout(m_Ctx->Device, pl, nullptr);
        p = VK_NULL_HANDLE; pl = VK_NULL_HANDLE;
    };
    destroyPipeline(m_InsertPipeline,  m_InsertLayout);
    destroyPipeline(m_RelightPipeline, m_RelightLayout);
    destroyPipeline(m_ComposePipeline, m_ComposeLayout);

    if (m_InsertGBufferPool)  vkDestroyDescriptorPool     (m_Ctx->Device, m_InsertGBufferPool, nullptr);
    if (m_InsertGBufferSetLayout) vkDestroyDescriptorSetLayout(m_Ctx->Device, m_InsertGBufferSetLayout, nullptr);
    if (m_RelightSdfPool)     vkDestroyDescriptorPool     (m_Ctx->Device, m_RelightSdfPool, nullptr);
    if (m_RelightSdfSetLayout)vkDestroyDescriptorSetLayout(m_Ctx->Device, m_RelightSdfSetLayout, nullptr);
    if (m_ComposePool)        vkDestroyDescriptorPool     (m_Ctx->Device, m_ComposePool, nullptr);
    if (m_ComposeGBufferSetLayout)  vkDestroyDescriptorSetLayout(m_Ctx->Device, m_ComposeGBufferSetLayout, nullptr);
    if (m_ComposeLightHdrSetLayout) vkDestroyDescriptorSetLayout(m_Ctx->Device, m_ComposeLightHdrSetLayout, nullptr);

    m_InsertGBufferPool = m_RelightSdfPool = m_ComposePool = VK_NULL_HANDLE;
    m_InsertGBufferSetLayout = m_RelightSdfSetLayout = VK_NULL_HANDLE;
    m_ComposeGBufferSetLayout = m_ComposeLightHdrSetLayout = VK_NULL_HANDLE;
    m_InsertGBufferSets = {};
    m_ComposeGBufferSets = {};
    m_ComposeLightHdrSets = {};
    m_RelightSdfSet = VK_NULL_HANDLE;

    RadianceCascadeHashTerminate(m_Hash, *m_Ctx);
    m_Ctx = nullptr;
}

void RadianceCascadeGI::DrawImGuiParams() {
    if (!m_Settings) {
        ImGui::TextDisabled("RadianceCascades: no settings bound.");
        return;
    }
    GISettings& s = *m_Settings;

    ImGui::SliderInt  ("Cascades",     &s.CascadeCount, 1, 6);
    ImGui::SliderFloat("s0 (m)",       &s.S0Meters,     0.25f, 4.0f, "%.2f");
    ImGui::SliderInt  ("r0 (rays/c0)", &s.R0Rays,       8, 64);
    ImGui::SliderInt  ("Hash log2",    &s.HashLog2,     16, 22);
    ImGui::Checkbox   ("Bilinear fix", &s.BilinearFix);
    ImGui::SliderFloat("Indirect boost", &s.IndirectBoost, 0.0f, 4.0f, "%.2f");
    ImGui::TextDisabled("Hash log2 changes apply on algo restart.");

    ImGui::Separator();
    static const char* kViewLabels[] = {
        "Off (lit)", "Hash cells", "Cascades", "Probe rays", "Irradiance only"
    };
    int viewIdx = static_cast<int>(s.DebugView);
    if (ImGui::Combo("Debug view", &viewIdx, kViewLabels, IM_ARRAYSIZE(kViewLabels))) {
        s.DebugView = static_cast<GIDebugView>(viewIdx);
    }

    ImGui::TextDisabled("Hash slots: 2^%u (%u). SDF resident: %s.",
                        m_Hash.HashLog2, m_Hash.SlotCount, m_HasSDF ? "yes" : "no");
}

// --------------------------- descriptors / pipelines --------------------------

bool RadianceCascadeGI::CreateGBufferDescriptors() {
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1] = b[0]; b[1].binding = 1;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 2; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr,
                                    &m_InsertGBufferSetLayout) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: insert GB set layout failed"); return false;
    }

    VkDescriptorPoolSize sz{};
    sz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sz.descriptorCount = 2 * VulkanContext::kFramesInFlight;
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 1; pci.pPoolSizes = &sz;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr,
                               &m_InsertGBufferPool) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: insert GB pool failed"); return false;
    }

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        layouts[i] = m_InsertGBufferSetLayout;
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = m_InsertGBufferPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai,
                                 m_InsertGBufferSets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: insert GB set alloc failed"); return false;
    }
    if (m_Targets) RewriteGBufferDescriptors();
    return true;
}

void RadianceCascadeGI::RewriteGBufferDescriptors() {
    if (!m_Targets || !m_Targets->Initialized) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        const OffscreenFrame& f = m_Targets->Frames[i];
        VkDescriptorImageInfo ii[2]{};
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[0].imageView   = f.Depth.View;
        ii[0].sampler     = m_Targets->SamplerNearest;
        ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1].imageView   = f.Identity.View;
        ii[1].sampler     = m_Targets->SamplerNearest;
        VkWriteDescriptorSet w[2]{};
        for (uint32_t k = 0; k < 2; ++k) {
            w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet          = m_InsertGBufferSets[i];
            w[k].dstBinding      = k;
            w[k].descriptorCount = 1;
            w[k].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[k].pImageInfo      = &ii[k];
        }
        vkUpdateDescriptorSets(m_Ctx->Device, 2, w, 0, nullptr);
    }
}

bool RadianceCascadeGI::CreateSdfDescriptors() {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 1; lci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr,
                                    &m_RelightSdfSetLayout) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: relight SDF layout failed"); return false;
    }

    VkDescriptorPoolSize sz{};
    sz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sz.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &sz;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr,
                               &m_RelightSdfPool) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: relight SDF pool failed"); return false;
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = m_RelightSdfPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &m_RelightSdfSetLayout;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai, &m_RelightSdfSet) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: relight SDF set alloc failed"); return false;
    }
    RewriteSdfDescriptors();
    return true;
}

void RadianceCascadeGI::RewriteSdfDescriptors() {
    if (m_RelightSdfSet == VK_NULL_HANDLE) return;
    VkDescriptorImageInfo ii{};
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii.imageView   = m_SdfView ? m_SdfView : (m_GlobalSdf ? m_GlobalSdf->DummyView : VK_NULL_HANDLE);
    ii.sampler     = m_SdfSampler;
    if (ii.imageView == VK_NULL_HANDLE || ii.sampler == VK_NULL_HANDLE) return;
    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = m_RelightSdfSet; w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(m_Ctx->Device, 1, &w, 0, nullptr);
}

bool RadianceCascadeGI::CreateLightHdrDescriptors() {
    // Compose set 0 = GBuffer (6 SRVs), set 1 = LightHDR storage image.
    {
        VkDescriptorSetLayoutBinding b[6]{};
        for (uint32_t i = 0; i < 6; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 6; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr,
                                        &m_ComposeGBufferSetLayout) != VK_SUCCESS) {
            RS_LOG_ERROR("RC GI: compose GB layout failed"); return false;
        }
    }
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 1; lci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr,
                                        &m_ComposeLightHdrSetLayout) != VK_SUCCESS) {
            RS_LOG_ERROR("RC GI: compose LightHDR layout failed"); return false;
        }
    }

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 6 * VulkanContext::kFramesInFlight;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 1 * VulkanContext::kFramesInFlight;
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 2 * VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr,
                               &m_ComposePool) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: compose pool failed"); return false;
    }

    VkDescriptorSetLayout gbLayouts[VulkanContext::kFramesInFlight]{};
    VkDescriptorSetLayout lhLayouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        gbLayouts[i] = m_ComposeGBufferSetLayout;
        lhLayouts[i] = m_ComposeLightHdrSetLayout;
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = m_ComposePool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts = gbLayouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai,
                                 m_ComposeGBufferSets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: compose GB set alloc failed"); return false;
    }
    ai.pSetLayouts = lhLayouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai,
                                 m_ComposeLightHdrSets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: compose LH set alloc failed"); return false;
    }

    if (m_Targets) RewriteLightHdrDescriptors();
    return true;
}

void RadianceCascadeGI::RewriteLightHdrDescriptors() {
    if (!m_Targets || !m_Targets->Initialized) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        const OffscreenFrame& f = m_Targets->Frames[i];
        VkDescriptorImageInfo gbInfos[6]{};
        const VkImageView views[6] = {
            f.Albedo.View, f.Normal.View, f.RoughMetalF0.View,
            f.Emissive.View, f.Depth.View, f.Identity.View
        };
        for (uint32_t k = 0; k < 6; ++k) {
            gbInfos[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            gbInfos[k].imageView   = views[k];
            gbInfos[k].sampler     = (k == 1 || k == 4 || k == 5)
                                       ? m_Targets->SamplerNearest
                                       : m_Targets->SamplerLinear;
        }
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t k = 0; k < 6; ++k) {
            writes[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[k].dstSet          = m_ComposeGBufferSets[i];
            writes[k].dstBinding      = k;
            writes[k].descriptorCount = 1;
            writes[k].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[k].pImageInfo      = &gbInfos[k];
        }
        VkDescriptorImageInfo lhInfo{};
        lhInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        lhInfo.imageView   = f.LightHDR.View;
        lhInfo.sampler     = VK_NULL_HANDLE;
        writes[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet          = m_ComposeLightHdrSets[i];
        writes[6].dstBinding      = 0;
        writes[6].descriptorCount = 1;
        writes[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[6].pImageInfo      = &lhInfo;
        vkUpdateDescriptorSets(m_Ctx->Device, 7, writes, 0, nullptr);
    }
}

bool RadianceCascadeGI::CreateInsertPipeline() {
    VkDescriptorSetLayout sets[2] = { m_InsertGBufferSetLayout, m_Hash.SetLayout };
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size   = 16;   // 4 uints
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 2; plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_Ctx->Device, &plci, nullptr,
                               &m_InsertLayout) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: insert pipeline layout failed"); return false;
    }
    if (!BuildComputePipeline(m_Ctx->Device, m_InsertLayout,
                              m_ShaderDir + "/rc_insert.spv",
                              m_InsertPipeline)) {
        RS_LOG_ERROR("RC GI: insert pipeline build failed"); return false;
    }
    return true;
}

bool RadianceCascadeGI::CreateRelightPipeline() {
    VkDescriptorSetLayout sets[2] = { m_RelightSdfSetLayout, m_Hash.SetLayout };
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0; pc.size = 16;
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 2; plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_Ctx->Device, &plci, nullptr,
                               &m_RelightLayout) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: relight pipeline layout failed"); return false;
    }
    if (!BuildComputePipeline(m_Ctx->Device, m_RelightLayout,
                              m_ShaderDir + "/rc_relight.spv",
                              m_RelightPipeline)) {
        RS_LOG_ERROR("RC GI: relight pipeline build failed"); return false;
    }
    return true;
}

bool RadianceCascadeGI::CreateComposePipeline() {
    VkDescriptorSetLayout sets[3] = {
        m_ComposeGBufferSetLayout, m_ComposeLightHdrSetLayout, m_Hash.SetLayout
    };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 3; plci.pSetLayouts = sets;
    if (vkCreatePipelineLayout(m_Ctx->Device, &plci, nullptr,
                               &m_ComposeLayout) != VK_SUCCESS) {
        RS_LOG_ERROR("RC GI: compose pipeline layout failed"); return false;
    }
    if (!BuildComputePipeline(m_Ctx->Device, m_ComposeLayout,
                              m_ShaderDir + "/gi_compose.spv",
                              m_ComposePipeline)) {
        RS_LOG_ERROR("RC GI: compose pipeline build failed"); return false;
    }
    return true;
}

// --------------------------- frame recording ----------------------------------

void RadianceCascadeGI::BindGIResourceForLighting(VkCommandBuffer /*cmd*/,
                                                  VkPipelineLayout /*layout*/,
                                                  uint32_t /*set*/, uint32_t /*binding*/) {
    // Compose lands in RecordCompose after lighting; the LightingPass keeps its
    // existing GiStubSet at set=3. Nothing to do here.
}

namespace {

void FillParams(RcHashParams& p, const RadianceCascadeHash& h,
                const FrameContext& frame, const GISettings& gi,
                const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                float decodeScale) {
    p.EyePosAndHashLog2 = glm::vec4(frame.Cam.EyePositionWorld, float(h.HashLog2));
    p.SunDirAndIntensity = glm::vec4(glm::normalize(frame.SunDirection), frame.SunIntensity);
    p.SunColor          = glm::vec4(frame.SunColor, 0.20f);     // .w = ambient luma
    p.CascadeParams     = glm::vec4(gi.S0Meters,
                                    float(std::max(1, gi.CascadeCount)),
                                    float(std::max(4, gi.R0Rays)),
                                    gi.IndirectBoost);
    glm::mat4 ivp = glm::inverse(frame.Cam.Projection * frame.Cam.View);
    // glm matrices are column-major; the shader reconstructs as mat4(c0, c1, c2, c3).
    p.InvViewProj0 = ivp[0];
    p.InvViewProj1 = ivp[1];
    p.InvViewProj2 = ivp[2];
    p.InvViewProj3 = ivp[3];
    p.RenderExtentAndFlags = glm::vec4(
        float(frame.RenderExtent.width),
        float(frame.RenderExtent.height),
        gi.BilinearFix ? 1.0f : 0.0f,
        float(static_cast<uint32_t>(gi.DebugView))
    );
    p.SdfAabbMin = glm::vec4(aabbMin, 0.0f);
    p.SdfAabbMax = glm::vec4(aabbMax, decodeScale);
    p.SecondaryParams = glm::vec4(3.0f, 1.0f, float(frame.FrameIndex), 0.0f);
}

void BufferBarrier(VkCommandBuffer cmd, VkBuffer buf,
                   VkAccessFlags src, VkAccessFlags dst,
                   VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkBufferMemoryBarrier b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    b.srcAccessMask = src; b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buf; b.offset = 0; b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

} // namespace

void RadianceCascadeGI::RecordPreFrame(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Hash.Initialized || !m_InsertPipeline || !m_Settings) return;
    if (!m_Settings->Enabled) return;

    // 1) Refresh params UBO for this slot.
    RcHashParams params{};
    FillParams(params, m_Hash, frame, *m_Settings, m_AabbMin, m_AabbMax, m_MaxDist);
    RadianceCascadeHashWriteParams(m_Hash, frame.FrameSlot, params);

    // 2) Relight first (consume LAST frame's hash + payload). On frame 0 the
    //    hash is empty; the dispatch is cheap (the cell-list count is 0).
    if (m_HasSDF) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_RelightPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_RelightLayout, 0, 1, &m_RelightSdfSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_RelightLayout, 1, 1,
                                &m_Hash.Sets[frame.FrameSlot], 0, nullptr);
        struct Push { uint32_t maxSteps; float minStep; float maxDistance; uint32_t pad; } pc{};
        pc.maxSteps    = 32;
        pc.minStep     = 0.005f;
        pc.maxDistance = 16.0f;
        vkCmdPushConstants(cmd, m_RelightLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        // CellList[0] holds the previous-frame count. Worst case = SlotCount.
        const uint32_t maxCells = m_Hash.SlotCount;
        const uint32_t groups   = (maxCells + 63u) / 64u;
        vkCmdDispatch(cmd, groups, 1, 1);

        BufferBarrier(cmd, m_Hash.PayloadBuffer,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    // 3) Clear hash keys + cell-list counter so this frame's inserts start
    //    from a clean table.
    RadianceCascadeHashClearForFrame(m_Hash, cmd);
}

void RadianceCascadeGI::RecordGather(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Hash.Initialized || !m_InsertPipeline || !m_Settings) return;
    if (!m_Settings->Enabled) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_InsertPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_InsertLayout, 0, 1,
                            &m_InsertGBufferSets[frame.FrameSlot], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_InsertLayout, 1, 1,
                            &m_Hash.Sets[frame.FrameSlot], 0, nullptr);

    // ---- Primary: 4× downsampled visible-pixel insertion --------------------
    {
        struct Push { uint32_t mode; uint32_t downSh; uint32_t secR; uint32_t pad; } pc{};
        pc.mode   = 0;
        pc.downSh = 2;        // >>2 = 4× downsample
        pc.secR   = 0;
        vkCmdPushConstants(cmd, m_InsertLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        const uint32_t loW = (frame.RenderExtent.width  >> 2);
        const uint32_t loH = (frame.RenderExtent.height >> 2);
        const uint32_t gx = (loW + 7u) / 8u;
        const uint32_t gy = (loH + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    BufferBarrier(cmd, m_Hash.KeyBuffer,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ---- Secondary: frustum-extended insertion ------------------------------
    {
        const uint32_t R = 2;
        struct Push { uint32_t mode; uint32_t downSh; uint32_t secR; uint32_t pad; } pc{};
        pc.mode = 1; pc.downSh = 0; pc.secR = R; pc.pad = 0;
        vkCmdPushConstants(cmd, m_InsertLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        const uint32_t side = 2u * R + 1u;
        vkCmdDispatch(cmd, side, side, side);
    }

    // Make inserts visible to the post-lighting compose pass.
    BufferBarrier(cmd, m_Hash.KeyBuffer,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    BufferBarrier(cmd, m_Hash.PayloadBuffer,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void RadianceCascadeGI::RecordCompose(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Hash.Initialized || !m_ComposePipeline || !m_Settings) return;
    if (!m_Settings->Enabled) return;
    if (!m_Targets) return;
    const OffscreenFrame& f = m_Targets->Frames[frame.FrameSlot];

    // LightHDR is currently SHADER_READ_ONLY_OPTIMAL (lighting pass post-barrier).
    // Need GENERAL for storage rw.
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = f.LightHDR.Image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComposePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_ComposeLayout, 0, 1,
                            &m_ComposeGBufferSets[frame.FrameSlot], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_ComposeLayout, 1, 1,
                            &m_ComposeLightHdrSets[frame.FrameSlot], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_ComposeLayout, 2, 1,
                            &m_Hash.Sets[frame.FrameSlot], 0, nullptr);
    const uint32_t gx = (frame.RenderExtent.width  + 7u) / 8u;
    const uint32_t gy = (frame.RenderExtent.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition LightHDR back to SHADER_READ_ONLY for the preview blit.
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace RS
