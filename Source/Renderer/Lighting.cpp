// Source/Renderer/Lighting.cpp — Phase 10
#include "Renderer/Lighting.h"
#include "Shadow/IShadowAlgorithm.h"
#include "Core/Logger.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

constexpr uint32_t kFrameBindingCount = 10;   // 0..9 (GBuffer + IBL + LightHDR)
constexpr uint32_t kLightHDRBinding   = 9;

bool ReadFile(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("Lighting: cannot open shader: %s", path); return false; }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0) { std::fclose(f); return false; }
    out.resize(static_cast<size_t>(len));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

VkShaderModule LoadModule(VkDevice device, const char* path) {
    std::vector<char> buf;
    if (!ReadFile(path, buf)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = buf.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) {
        RS_LOG_ERROR("Lighting: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

bool CreateFrameSetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    VkDescriptorSetLayoutBinding b[kFrameBindingCount]{};
    // 0..5 GBuffer SRVs (combined image samplers)
    for (uint32_t i = 0; i < 6; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // 6..8 IBL bindings (cube irradiance, cube prefilter, BRDF LUT)
    for (uint32_t i = 6; i < 9; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // 9 LightHDR (storage image, write-only)
    b[9].binding         = kLightHDRBinding;
    b[9].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[9].descriptorCount = 1;
    b[9].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = kFrameBindingCount;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateEmptySetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 0;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 9 * VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 1 * VulkanContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight + 1;    // +1 for the GI stub set
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    return vkCreateDescriptorPool(device, &pci, nullptr, &outPool) == VK_SUCCESS;
}

bool AllocateFrameSets(VkDevice device, LightingPass& lp,
                       const OffscreenTargets& targets, const SkyAtmosphere& sky) {
    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = lp.SetLayoutFrame;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = lp.DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device, &ai, lp.FrameSets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("Lighting: alloc frame sets failed");
        return false;
    }

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        const OffscreenFrame& f = targets.Frames[i];
        VkDescriptorImageInfo infos[kFrameBindingCount]{};

        const VkImageView gbufferViews[6] = {
            f.Albedo.View, f.Normal.View, f.RoughMetalF0.View,
            f.Emissive.View, f.Depth.View, f.Identity.View
        };
        for (uint32_t b = 0; b < 6; ++b) {
            infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[b].imageView   = gbufferViews[b];
            // Normal + identity = nearest (no filtering across pixels). Depth
            // uses nearest too — we only need a per-pixel sample.
            infos[b].sampler     = (b == 1 || b == 4 || b == 5)
                                       ? targets.SamplerNearest
                                       : targets.SamplerLinear;
        }
        infos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[6].imageView   = sky.Irradiance.View;
        infos[6].sampler     = sky.SamplerCubeLinear;
        infos[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[7].imageView   = sky.Prefilter.View;
        infos[7].sampler     = sky.SamplerCubeLinear;
        infos[8].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[8].imageView   = sky.LutView;
        infos[8].sampler     = sky.SamplerLut;

        // LightHDR storage write. Sampler is ignored for STORAGE_IMAGE.
        infos[9].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        infos[9].imageView   = f.LightHDR.View;
        infos[9].sampler     = VK_NULL_HANDLE;

        VkWriteDescriptorSet writes[kFrameBindingCount]{};
        for (uint32_t b = 0; b < kFrameBindingCount; ++b) {
            writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet          = lp.FrameSets[i];
            writes[b].dstBinding      = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType  = (b == kLightHDRBinding)
                                            ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo      = &infos[b];
        }
        vkUpdateDescriptorSets(device, kFrameBindingCount, writes, 0, nullptr);
    }
    return true;
}

bool AllocateGiStubSet(VkDevice device, LightingPass& lp) {
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = lp.DescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &lp.SetLayoutGiStub;
    return vkAllocateDescriptorSets(device, &ai, &lp.GiStubSet) == VK_SUCCESS;
}

bool BuildPipelineLayout(LightingPass& lp, VkDevice device,
                         VkDescriptorSetLayout shadowLayout) {
    // Pipeline layout = [set0=empty (GI placeholder), set1=Frame,
    //                    set2=shadow algo, set3=GI stub].
    // The GLSL shader puts its bindings on sets 1/2/3, so set 0 must still be
    // declared to keep set indices contiguous.
    VkDescriptorSetLayout setLayouts[4] = {
        lp.SetLayoutGiStub,    // set=0 (empty — placeholder)
        lp.SetLayoutFrame,     // set=1
        shadowLayout,          // set=2
        lp.SetLayoutGiStub,    // set=3
    };

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(LightingPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 4;
    plci.pSetLayouts            = setLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    return vkCreatePipelineLayout(device, &plci, nullptr, &lp.PipelineLayout) == VK_SUCCESS;
}

bool BuildPipeline(LightingPass& lp, VkDevice device, const char* shaderDir,
                   uint32_t variant) {
    // Each shadow variant is its own SPV — the set=2 descriptor layout differs
    // across variants (PCF: 2 bindings, PCSS: 3, VSM: 2 different types,
    // SDFCone: sampler3D + UBO), and GLSL can't redeclare a sampler at the
    // same binding with different types, so a single SPV with a spec-constant
    // branch isn't viable.
    const char* shaderName = "lighting.spv";
    switch (variant) {
        case 0: shaderName = "lighting.spv";          break;   // PCF
        case 1: shaderName = "lighting_pcss.spv";     break;   // PCSS
        case 2: shaderName = "lighting_vsm.spv";      break;   // VSM
        case 3: shaderName = "lighting_sdfcone.spv";  break;   // SDFCone (Phase 13)
        default: shaderName = "lighting.spv";         break;
    }
    const std::string sp = std::string(shaderDir) + "/" + shaderName;
    VkShaderModule cs = LoadModule(device, sp.c_str());
    if (!cs) return false;

    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage  = stage;
    cpci.layout = lp.PipelineLayout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &lp.Pipeline);
    vkDestroyShaderModule(device, cs, nullptr);
    return r == VK_SUCCESS;
}

} // namespace

// -- public API ---------------------------------------------------------------

bool LightingPassInitialize(LightingPass& lp, const VulkanContext& ctx,
                            const OffscreenTargets& targets,
                            const SkyAtmosphere& sky,
                            const IShadowAlgorithm& shadowAlgo,
                            const char* shaderArtifactsDir) {
    if (lp.Initialized) return true;
    lp.RenderExtent = targets.Extent;

    if (!CreateFrameSetLayout(ctx.Device, lp.SetLayoutFrame)) {
        RS_LOG_ERROR("Lighting: frame set layout failed"); return false;
    }
    if (!CreateEmptySetLayout(ctx.Device, lp.SetLayoutGiStub)) {
        RS_LOG_ERROR("Lighting: GI stub layout failed"); return false;
    }
    if (!CreateDescriptorPool(ctx.Device, lp.DescriptorPool)) {
        RS_LOG_ERROR("Lighting: descriptor pool failed"); return false;
    }
    if (!AllocateFrameSets(ctx.Device, lp, targets, sky)) return false;
    if (!AllocateGiStubSet(ctx.Device, lp)) {
        RS_LOG_ERROR("Lighting: GI stub set alloc failed"); return false;
    }
    if (!BuildPipelineLayout(lp, ctx.Device, shadowAlgo.LightingSetLayout())) {
        RS_LOG_ERROR("Lighting: pipeline layout failed"); return false;
    }
    if (!BuildPipeline(lp, ctx.Device, shaderArtifactsDir, shadowAlgo.AlgoVariant())) {
        RS_LOG_ERROR("Lighting: pipeline failed"); return false;
    }
    lp.BuiltVariant      = shadowAlgo.AlgoVariant();
    lp.BuiltShadowLayout = shadowAlgo.LightingSetLayout();
    lp.Initialized       = true;
    RS_LOG_INFO("LightingPass ready: variant=%u, %ux%u",
                lp.BuiltVariant, lp.RenderExtent.width, lp.RenderExtent.height);
    return true;
}

void LightingPassTerminate(LightingPass& lp, const VulkanContext& ctx) {
    if (!lp.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    if (lp.Pipeline)         vkDestroyPipeline           (ctx.Device, lp.Pipeline,         nullptr);
    if (lp.PipelineLayout)   vkDestroyPipelineLayout     (ctx.Device, lp.PipelineLayout,   nullptr);
    if (lp.DescriptorPool)   vkDestroyDescriptorPool     (ctx.Device, lp.DescriptorPool,   nullptr);
    if (lp.SetLayoutFrame)   vkDestroyDescriptorSetLayout(ctx.Device, lp.SetLayoutFrame,   nullptr);
    if (lp.SetLayoutGiStub)  vkDestroyDescriptorSetLayout(ctx.Device, lp.SetLayoutGiStub,  nullptr);
    lp.Pipeline         = VK_NULL_HANDLE;
    lp.PipelineLayout   = VK_NULL_HANDLE;
    lp.DescriptorPool   = VK_NULL_HANDLE;
    lp.SetLayoutFrame   = VK_NULL_HANDLE;
    lp.SetLayoutGiStub  = VK_NULL_HANDLE;
    lp.FrameSets        = {};
    lp.GiStubSet        = VK_NULL_HANDLE;
    lp.BuiltVariant     = 0xFFFFFFFFu;
    lp.BuiltShadowLayout= VK_NULL_HANDLE;
    lp.Initialized      = false;
}

bool LightingPassSetShadowAlgorithm(LightingPass& lp, const VulkanContext& ctx,
                                    const IShadowAlgorithm& newAlgo,
                                    const char* shaderArtifactsDir) {
    if (!lp.Initialized) return false;
    const uint32_t              newVariant = newAlgo.AlgoVariant();
    const VkDescriptorSetLayout newLayout  = newAlgo.LightingSetLayout();
    if (newVariant == lp.BuiltVariant && newLayout == lp.BuiltShadowLayout) {
        return true;
    }
    vkDeviceWaitIdle(ctx.Device);
    if (lp.Pipeline)       vkDestroyPipeline      (ctx.Device, lp.Pipeline,       nullptr);
    if (lp.PipelineLayout) vkDestroyPipelineLayout(ctx.Device, lp.PipelineLayout, nullptr);
    lp.Pipeline       = VK_NULL_HANDLE;
    lp.PipelineLayout = VK_NULL_HANDLE;
    if (!BuildPipelineLayout(lp, ctx.Device, newLayout)) return false;
    if (!BuildPipeline      (lp, ctx.Device, shaderArtifactsDir, newVariant)) return false;
    lp.BuiltVariant      = newVariant;
    lp.BuiltShadowLayout = newLayout;
    RS_LOG_INFO("LightingPass rebuilt: variant=%u", newVariant);
    return true;
}

void LightingPassRecord(LightingPass& lp, const VulkanContext& /*ctx*/,
                        VkCommandBuffer cmd, uint32_t frameSlot,
                        const OffscreenTargets& targets,
                        IShadowAlgorithm& shadowAlgo,
                        const CameraView& camera,
                        const glm::vec3& sunDirectionToward,
                        const glm::vec3& sunColor,
                        float sunIntensity,
                        float ambient,
                        bool  iblEnabled,
                        float iblIntensity) {
    if (!lp.Initialized) return;
    const OffscreenFrame& f = targets.Frames[frameSlot];

    // 1) Transition LightHDR to GENERAL for the storage write.
    //    Previous state: UNDEFINED on first frame, SHADER_READ_ONLY after a
    //    previous Lit preview consumed it. Either way we discard the old
    //    contents.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;     // discard previous contents
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = f.LightHDR.Image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // 2) Bind pipeline + sets.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.Pipeline);

    // set=0 = GI stub (empty), set=1 = frame, set=3 = GI stub.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.PipelineLayout,
                            0, 1, &lp.GiStubSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.PipelineLayout,
                            1, 1, &lp.FrameSets[frameSlot], 0, nullptr);
    shadowAlgo.BindLightingDescriptorSet(cmd, lp.PipelineLayout, frameSlot, 2);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.PipelineLayout,
                            3, 1, &lp.GiStubSet, 0, nullptr);

    // 3) Push constants.
    LightingPushConstants pc{};
    pc.InvViewProj        = glm::inverse(camera.Projection * camera.View);
    pc.EyePosWorld        = glm::vec4(camera.EyePositionWorld, ambient);
    pc.SunDirAndIntensity = glm::vec4(glm::normalize(sunDirectionToward), sunIntensity);
    pc.SunColor           = glm::vec4(sunColor, 0.0f);
    pc.IblParams          = glm::vec4(iblEnabled ? 1.0f : 0.0f,
                                      iblIntensity,
                                      static_cast<float>(SkyAtmosphere::kPrefilterMips),
                                      0.0f);
    pc.ResAndFlags        = glm::vec4(static_cast<float>(lp.RenderExtent.width),
                                      static_cast<float>(lp.RenderExtent.height),
                                      0.0f, 0.0f);
    vkCmdPushConstants(cmd, lp.PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // 4) Dispatch.
    const uint32_t gx = (lp.RenderExtent.width  + 7u) / 8u;
    const uint32_t gy = (lp.RenderExtent.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    // 5) Transition LightHDR back to SHADER_READ_ONLY so the preview blit can
    //    sample it as a texture.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = f.LightHDR.Image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }
}

} // namespace RS
