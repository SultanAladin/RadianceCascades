// Source/Renderer/Tonemap.h — Phase 16
// In-place compute pass over LightHDR: exposure -> ACES Narkowicz -> sRGB gamma.
// Slotted between the GI compose (or LightingPassRecord, when GI is off) and
// GBufferPreviewRecord. LightHDR is the only image touched, so there's no new
// descriptor pool entry on the preview side and no extra image allocation.
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/OffscreenTargets.h"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace RS {

struct TonemapSettings {
    float ExposureEV   = 0.0f;     // stops; final scale = exp2(EV).
    bool  AcesEnabled  = true;     // false = simple clamp 0..1.
    bool  GammaEnabled = true;     // sRGB encode (swapchain is _UNORM).
};

struct TonemapPass {
    VkDescriptorSetLayout SetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      DescriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout      PipelineLayout = VK_NULL_HANDLE;
    VkPipeline            Pipeline       = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> FrameSets{};

    VkExtent2D Extent      = { 0, 0 };
    bool       Initialized = false;
};

struct TonemapPushConstants {
    glm::vec4 Params;     // x = linear exposure, y = flagsAsFloat, z/w unused
    glm::vec4 ResAndPad;  // xy = extent, zw unused
};
static_assert(sizeof(TonemapPushConstants) == 32,
              "TonemapPushConstants layout pinned for the shader");

bool TonemapPassInitialize(TonemapPass& tp, const VulkanContext& ctx,
                           const OffscreenTargets& targets,
                           const char* shaderArtifactsDir);
void TonemapPassTerminate (TonemapPass& tp, const VulkanContext& ctx);

void TonemapPassRecord(TonemapPass& tp, VkCommandBuffer cmd, uint32_t frameSlot,
                       const OffscreenTargets& targets,
                       const TonemapSettings& settings);

} // namespace RS
