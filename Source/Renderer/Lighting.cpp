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

constexpr uint32_t kFrameBindingCount = 11;   // 0..10 (GBuffer + IBL + LightHDR + Specular)
constexpr uint32_t kLightHDRBinding   = 9;
constexpr uint32_t kSpecularBinding   = 10;
// Phase 3 radiance-cascades resolve: C0 atlas (storage image) + metadata UBO.
// These two are distinct descriptor types from the GBuffer block above, so they
// are declared/written separately rather than folded into the image loop.
constexpr uint32_t kCascadeAtlasBinding  = 11;
constexpr uint32_t kCascadeResolveBinding = 12;

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
    VkDescriptorSetLayoutBinding b[kFrameBindingCount + 2]{};
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
    // 10 Specular SRV (KHR_materials_specular)
    b[10].binding         = kSpecularBinding;
    b[10].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[10].descriptorCount = 1;
    b[10].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    // 11 Cascade C0 atlas (storage image, read-only in shader)
    b[11].binding         = kCascadeAtlasBinding;
    b[11].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[11].descriptorCount = 1;
    b[11].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    // 12 Cascade resolve metadata (uniform buffer)
    b[12].binding         = kCascadeResolveBinding;
    b[12].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[12].descriptorCount = 1;
    b[12].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = kFrameBindingCount + 2;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateEmptySetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 0;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 10 * VulkanContext::kFramesInFlight; // 9 + Specular
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 2  * VulkanContext::kFramesInFlight; // LightHDR + C0 atlas
    sizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[2].descriptorCount = 1  * VulkanContext::kFramesInFlight; // cascade resolve UBO

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight + 1;    // +1 for unused set=0
    pci.poolSizeCount = 3;
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
        // Specular SRV.
        infos[10].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[10].imageView   = f.Specular.View;
        infos[10].sampler     = targets.SamplerLinear;

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

bool AllocateEmptySet0(VkDevice device, LightingPass& lp) {
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = lp.DescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &lp.SetLayoutEmpty0;
    return vkAllocateDescriptorSets(device, &ai, &lp.EmptySet0) == VK_SUCCESS;
}

bool BuildPipelineLayout(LightingPass& lp, VkDevice device,
                         VkDescriptorSetLayout shadowLayout) {
    // Pipeline layout = [set0=empty, set1=Frame, set2=shadow algo].
    // The GLSL shader puts its bindings on sets 1/2, so set 0 must still be
    // declared to keep set indices contiguous.
    VkDescriptorSetLayout setLayouts[3] = {
        lp.SetLayoutEmpty0,    // set=0 (unused)
        lp.SetLayoutFrame,     // set=1
        shadowLayout,          // set=2
    };

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(LightingPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 3;
    plci.pSetLayouts            = setLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    return vkCreatePipelineLayout(device, &plci, nullptr, &lp.PipelineLayout) == VK_SUCCESS;
}

bool BuildPipeline(LightingPass& lp, VkDevice device, const char* shaderDir,
                   uint32_t variant) {
    // Each shadow variant is its own SPV — the set=2 descriptor layout differs
    // across variants (SDFCone uses sparse-SDF buffers and UBOs),
    // SDFCone: sampler3D + UBO), and GLSL can't redeclare a sampler at the
    // same binding with different types, so a single SPV with a spec-constant
    // branch isn't viable.
    (void)variant;
    const char* shaderName = "lighting_sdfcone.spv";
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
    if (!CreateEmptySetLayout(ctx.Device, lp.SetLayoutEmpty0)) {
        RS_LOG_ERROR("Lighting: empty set=0 layout failed"); return false;
    }
    if (!CreateDescriptorPool(ctx.Device, lp.DescriptorPool)) {
        RS_LOG_ERROR("Lighting: descriptor pool failed"); return false;
    }
    if (!AllocateFrameSets(ctx.Device, lp, targets, sky)) return false;
    if (!AllocateEmptySet0(ctx.Device, lp)) {
        RS_LOG_ERROR("Lighting: empty set=0 alloc failed"); return false;
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
    if (lp.SetLayoutEmpty0)  vkDestroyDescriptorSetLayout(ctx.Device, lp.SetLayoutEmpty0,  nullptr);
    lp.Pipeline         = VK_NULL_HANDLE;
    lp.PipelineLayout   = VK_NULL_HANDLE;
    lp.DescriptorPool   = VK_NULL_HANDLE;
    lp.SetLayoutFrame   = VK_NULL_HANDLE;
    lp.SetLayoutEmpty0  = VK_NULL_HANDLE;
    lp.FrameSets        = {};
    lp.EmptySet0        = VK_NULL_HANDLE;
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

void LightingPassRegisterRadianceCascades(LightingPass& lp, const VulkanContext& ctx,
                                          VkImageView cascadeAtlasView,
                                          const VkBuffer* resolveParamBuffersByFrame) {
    if (!lp.Initialized || cascadeAtlasView == VK_NULL_HANDLE || !resolveParamBuffersByFrame)
        return;

    vkDeviceWaitIdle(ctx.Device);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo atlasInfo{};
        atlasInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;       // C0 stays GENERAL (compute store + read)
        atlasInfo.imageView   = cascadeAtlasView;
        atlasInfo.sampler     = VK_NULL_HANDLE;                // storage image, sampler ignored

        VkDescriptorBufferInfo resolveInfo{};
        resolveInfo.buffer = resolveParamBuffersByFrame[i];
        resolveInfo.offset = 0;
        resolveInfo.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = lp.FrameSets[i];
        writes[0].dstBinding      = kCascadeAtlasBinding;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &atlasInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = lp.FrameSets[i];
        writes[1].dstBinding      = kCascadeResolveBinding;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo     = &resolveInfo;

        vkUpdateDescriptorSets(ctx.Device, 2, writes, 0, nullptr);
    }
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
                        float iblIntensity,
                        bool  realisticPbr) {
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

    // set=0 is unused by the shader, but must be bound so set=1 and set=2 stay
    // at their declared indices.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.PipelineLayout,
                            0, 1, &lp.EmptySet0, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lp.PipelineLayout,
                            1, 1, &lp.FrameSets[frameSlot], 0, nullptr);
    shadowAlgo.BindLightingDescriptorSet(cmd, lp.PipelineLayout, frameSlot, 2);

    // 3) Push constants.
    LightingPushConstants pc{};
    pc.InvViewProj        = glm::inverse(camera.Projection * camera.View);
    pc.EyePosWorld        = glm::vec4(camera.EyePositionWorld, ambient);
    pc.SunDirAndIntensity = glm::vec4(glm::normalize(sunDirectionToward), sunIntensity);
    pc.SunColor           = glm::vec4(sunColor, 0.0f);
    // IblParams.w carries the PBR flag word as float-bits (shader decodes via
    // floatBitsToUint). Bit 0 = realistic PBR (EON + energy comp).
    const uint32_t pbrFlags = realisticPbr ? PbrFlags::Realistic : 0u;
    float pbrFlagsAsFloat;
    std::memcpy(&pbrFlagsAsFloat, &pbrFlags, sizeof(pbrFlagsAsFloat));
    pc.IblParams          = glm::vec4(iblEnabled ? 1.0f : 0.0f,
                                      iblIntensity,
                                      static_cast<float>(SkyAtmosphere::kPrefilterMips),
                                      pbrFlagsAsFloat);
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
