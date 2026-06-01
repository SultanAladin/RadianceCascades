// Source/Renderer/Tonemap.cpp — Phase 16
#include "Renderer/Tonemap.h"
#include "Core/Logger.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

bool ReadFile(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("Tonemap: cannot open shader: %s", path); return false; }
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
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

bool CreateSetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 1;
    lci.pBindings    = &b;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    VkDescriptorPoolSize size{};
    size.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    size.descriptorCount = VulkanContext::kFramesInFlight;
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &size;
    return vkCreateDescriptorPool(device, &pci, nullptr, &outPool) == VK_SUCCESS;
}

bool AllocateFrameSets(VkDevice device, TonemapPass& tp,
                       const OffscreenTargets& targets) {
    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight]{};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = tp.SetLayout;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = tp.DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device, &ai, tp.FrameSets.data()) != VK_SUCCESS) {
        RS_LOG_ERROR("Tonemap: alloc frame sets failed");
        return false;
    }
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo info{};
        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        info.imageView   = targets.Frames[i].LightHDR.View;
        info.sampler     = VK_NULL_HANDLE;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = tp.FrameSets[i];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo      = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    return true;
}

bool BuildPipelineLayout(TonemapPass& tp, VkDevice device) {
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(TonemapPushConstants);
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &tp.SetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    return vkCreatePipelineLayout(device, &plci, nullptr, &tp.PipelineLayout) == VK_SUCCESS;
}

bool BuildPipeline(TonemapPass& tp, VkDevice device, const char* shaderDir) {
    const std::string sp = std::string(shaderDir) + "/tonemap.spv";
    VkShaderModule cs = LoadModule(device, sp.c_str());
    if (!cs) { RS_LOG_ERROR("Tonemap: load tonemap.spv failed"); return false; }
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage  = stage;
    cpci.layout = tp.PipelineLayout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &tp.Pipeline);
    vkDestroyShaderModule(device, cs, nullptr);
    return r == VK_SUCCESS;
}

} // namespace

bool TonemapPassInitialize(TonemapPass& tp, const VulkanContext& ctx,
                           const OffscreenTargets& targets,
                           const char* shaderArtifactsDir) {
    if (tp.Initialized) return true;
    tp.Extent = targets.Extent;
    if (!CreateSetLayout     (ctx.Device, tp.SetLayout))      { RS_LOG_ERROR("Tonemap: set layout");      return false; }
    if (!CreateDescriptorPool(ctx.Device, tp.DescriptorPool)) { RS_LOG_ERROR("Tonemap: descriptor pool"); return false; }
    if (!AllocateFrameSets   (ctx.Device, tp, targets))       { return false; }
    if (!BuildPipelineLayout (tp, ctx.Device))                { RS_LOG_ERROR("Tonemap: pipeline layout"); return false; }
    if (!BuildPipeline       (tp, ctx.Device, shaderArtifactsDir)) { RS_LOG_ERROR("Tonemap: pipeline"); return false; }
    tp.Initialized = true;
    RS_LOG_INFO("TonemapPass ready: %ux%u", tp.Extent.width, tp.Extent.height);
    return true;
}

void TonemapPassTerminate(TonemapPass& tp, const VulkanContext& ctx) {
    if (!tp.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    if (tp.Pipeline)       vkDestroyPipeline           (ctx.Device, tp.Pipeline,       nullptr);
    if (tp.PipelineLayout) vkDestroyPipelineLayout     (ctx.Device, tp.PipelineLayout, nullptr);
    if (tp.DescriptorPool) vkDestroyDescriptorPool     (ctx.Device, tp.DescriptorPool, nullptr);
    if (tp.SetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, tp.SetLayout,      nullptr);
    tp = {};
}

void TonemapPassRecord(TonemapPass& tp, VkCommandBuffer cmd, uint32_t frameSlot,
                       const OffscreenTargets& targets,
                       const TonemapSettings& settings) {
    if (!tp.Initialized) return;
    const OffscreenFrame& f = targets.Frames[frameSlot];

    // Caller chain leaves LightHDR in SHADER_READ_ONLY_OPTIMAL (post-Lighting
    // or post-GI-compose). Flip to GENERAL for the in-place storage write.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image            = f.LightHDR.Image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    vkCmdBindPipeline      (cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tp.Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tp.PipelineLayout,
                            0, 1, &tp.FrameSets[frameSlot], 0, nullptr);

    uint32_t flags = 0u;
    if (settings.AcesEnabled)  flags |= 1u;
    if (settings.GammaEnabled) flags |= 2u;
    float flagsAsFloat;
    std::memcpy(&flagsAsFloat, &flags, sizeof(flagsAsFloat));

    TonemapPushConstants pc{};
    pc.Params    = glm::vec4(std::exp2(settings.ExposureEV), flagsAsFloat, 0.0f, 0.0f);
    pc.ResAndPad = glm::vec4(static_cast<float>(tp.Extent.width),
                             static_cast<float>(tp.Extent.height), 0.0f, 0.0f);
    vkCmdPushConstants(cmd, tp.PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    const uint32_t gx = (tp.Extent.width  + 7u) / 8u;
    const uint32_t gy = (tp.Extent.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition back to SHADER_READ_ONLY for the preview sampler.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image            = f.LightHDR.Image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }
}

} // namespace RS
