// Source/Renderer/Lighting.h — Phase 10
// Compute pass that composes the lit image from the GBuffer + the active
// shadow algorithm's set=2 contract. Writes RGBA16F into OffscreenFrame::LightHDR
// (storage). Per-frame-in-flight descriptor sets for set=1 (GBuffer + IBL +
// LightHDR storage); set=2 is owned by SDFConeShadow and bound at record time.
//
// Pipeline rebuild: `LightingPassSetShadowAlgorithm` re-creates the pipeline
// when the active variant changes (specialisation constant kAlgoVariant). The
// pipeline layout itself depends on the shadow algo's LightingSetLayout(); we
// rebuild that too.
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/OffscreenTargets.h"
#include "Renderer/SkyAtmosphere.h"
#include "RS/Renderer.h"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace RS {

struct IShadowAlgorithm;

struct LightingPass {
    // Owned descriptor / pipeline objects.
    VkDescriptorSetLayout SetLayoutFrame    = VK_NULL_HANDLE;  // set=1
    VkDescriptorSetLayout SetLayoutEmpty0   = VK_NULL_HANDLE;  // set=0 (unused by shader)
    VkDescriptorPool      DescriptorPool    = VK_NULL_HANDLE;
    VkPipelineLayout      PipelineLayout    = VK_NULL_HANDLE;
    VkPipeline            Pipeline          = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> FrameSets{};
    VkDescriptorSet       EmptySet0         = VK_NULL_HANDLE;

    // Variant that the current pipeline was built against.
    uint32_t              BuiltVariant      = 0xFFFFFFFFu;
    VkDescriptorSetLayout BuiltShadowLayout = VK_NULL_HANDLE;

    VkExtent2D            RenderExtent      = { 0, 0 };
    bool                  Initialized       = false;
};

struct LightingPushConstants {
    glm::mat4 InvViewProj;
    glm::vec4 EyePosWorld;        // xyz = eye, w = ambient fallback
    glm::vec4 SunDirAndIntensity; // xyz = toward sun, w = intensity
    glm::vec4 SunColor;           // rgb, w unused
    glm::vec4 IblParams;          // x = on/off, y = intensity, z = prefilter mips, w
    glm::vec4 ResAndFlags;        // x/y = render extent, z/w unused
};
static_assert(sizeof(LightingPushConstants) == 64 + 5 * 16,
              "LightingPushConstants layout pinned for the shader to match");

bool LightingPassInitialize(LightingPass& lp, const VulkanContext& ctx,
                            const OffscreenTargets& targets,
                            const SkyAtmosphere& sky,
                            const IShadowAlgorithm& shadowAlgo,
                            const char* shaderArtifactsDir);
void LightingPassTerminate (LightingPass& lp, const VulkanContext& ctx);

// Rebuilds pipeline + pipeline layout for a new shadow algo. Call after the
// new algo has been Initialize()'d.
bool LightingPassSetShadowAlgorithm(LightingPass& lp, const VulkanContext& ctx,
                                    const IShadowAlgorithm& newAlgo,
                                    const char* shaderArtifactsDir);

// Phase 3: bind the radiance-cascades C0 atlas (storage image3D, GENERAL) and the
// per-frame resolve-metadata UBO ring into set=1 bindings 11/12. Rewrites those
// two bindings across all frame slots under vkDeviceWaitIdle (mirrors the
// SDFConeShadow SetSDF descriptor-rewrite pattern). Must be called once after
// LightingPassInitialize and after the cascade atlas + UBO ring exist.
void LightingPassRegisterRadianceCascades(LightingPass& lp, const VulkanContext& ctx,
                                          VkImageView cascadeAtlasView,
                                          const VkBuffer* resolveParamBuffersByFrame);

void LightingPassRecord(LightingPass& lp, const VulkanContext& ctx,
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
                        bool  realisticPbr = true);

// PBR flag bits packed into LightingPushConstants::IblParams.w (as float bits).
// Kept here so Main.cpp / panels can build the word without including the
// shader-side include.
namespace PbrFlags {
    constexpr uint32_t Realistic = 1u << 0;
}

} // namespace RS
