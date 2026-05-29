// Source/Renderer/SkyAtmosphere.h — Phase 8
// Owns the three image-based-lighting textures the PbrLit shader samples:
//
//   * Sky cubemap         — RGBA16F, 6 faces × kSkySize²,  mip 0 only.
//                           Procedural 3-band gradient (zenith / horizon /
//                           ground) + soft sun disc, baked from sky_bake.comp.
//   * Irradiance cubemap  — RGBA16F, 6 faces × kIrrSize²,  mip 0 only.
//                           Lambertian convolution of the sky, baked from
//                           ibl_irradiance.comp.
//   * Prefilter cubemap   — RGBA16F, 6 faces × kPrefilterSize², kPrefilterMips.
//                           GGX split-sum prefilter, one mip per roughness,
//                           baked from ibl_prefilter.comp.
//   * BRDF LUT            — RG16F,   kBrdfLutSize² 2D image. Karis scale/bias.
//
// The bakes run inside SkyAtmosphereEnsureBaked() — called once per frame from
// Main; the function early-outs when the settings hash matches the previous
// bake, so per-frame cost is a hash compare and a couple of memcmps. Changing
// the sun direction or any colour in the panel re-bakes (taking sub-millisecond
// for the sky + irradiance, low single-digit ms for the GGX prefilter).
#pragma once

#include "Core/VulkanContext.h"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

namespace RS {

struct SkySettings {
    glm::vec3 ZenithColor    = glm::vec3(0.20f, 0.36f, 0.66f);
    glm::vec3 HorizonColor   = glm::vec3(0.72f, 0.78f, 0.88f);
    glm::vec3 GroundColor    = glm::vec3(0.18f, 0.16f, 0.14f);
    float     SkyIntensity   = 1.0f;
    float     HorizonExponent = 1.4f;
    float     GroundExponent  = 1.0f;

    glm::vec3 SunColor       = glm::vec3(1.0f, 0.96f, 0.88f);
    float     SunIntensity   = 8.0f;          // multiplied into sky cubemap
    float     SunAngularSizeDeg = 1.5f;       // soft disc half-angle

    bool      EnableIBL      = true;          // Sun/Sky panel toggle, plumbed to PbrLit
    float     IBLIntensity   = 1.0f;          // overall IBL gain
};

struct SkyCubeImage {
    VkImage        Image  = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView    View   = VK_NULL_HANDLE;   // CUBE view for sampling
    VkImageView    Storage[16] = {};          // per-mip 2D-array view for imageStore
    uint32_t       MipCount = 1;
    uint32_t       FaceSize = 0;
    VkFormat       Format   = VK_FORMAT_UNDEFINED;
};

struct SkyAtmosphere {
    bool Initialized = false;

    SkyCubeImage Sky;         // mip 0 only
    SkyCubeImage Irradiance;  // mip 0 only
    SkyCubeImage Prefilter;   // kPrefilterMips
    // BRDF LUT (2D, RG16F)
    VkImage        LutImage   = VK_NULL_HANDLE;
    VkDeviceMemory LutMemory  = VK_NULL_HANDLE;
    VkImageView    LutView    = VK_NULL_HANDLE;

    VkSampler      SamplerCubeLinear = VK_NULL_HANDLE;   // mips + linear, edge clamp
    VkSampler      SamplerLut        = VK_NULL_HANDLE;   // linear, edge clamp, no mips

    // Compute pipelines
    VkDescriptorPool      DescriptorPool          = VK_NULL_HANDLE;
    VkDescriptorSetLayout SkySetLayout            = VK_NULL_HANDLE;
    VkDescriptorSetLayout IrradianceSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout PrefilterSetLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout BrdfSetLayout           = VK_NULL_HANDLE;

    VkPipelineLayout SkyPipelineLayout        = VK_NULL_HANDLE;
    VkPipelineLayout IrradiancePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout PrefilterPipelineLayout  = VK_NULL_HANDLE;
    VkPipelineLayout BrdfPipelineLayout       = VK_NULL_HANDLE;

    VkPipeline SkyPipeline        = VK_NULL_HANDLE;
    VkPipeline IrradiancePipeline = VK_NULL_HANDLE;
    VkPipeline PrefilterPipeline  = VK_NULL_HANDLE;
    VkPipeline BrdfPipeline       = VK_NULL_HANDLE;

    VkDescriptorSet SkyDescriptor        = VK_NULL_HANDLE;
    VkDescriptorSet IrradianceDescriptor = VK_NULL_HANDLE;
    // One prefilter descriptor per mip — each one writes a different storage
    // image view into binding 1.
    std::array<VkDescriptorSet, 16> PrefilterDescriptors{};
    VkDescriptorSet BrdfDescriptor       = VK_NULL_HANDLE;

    // Cached hash of the SkySettings the current textures were baked from.
    // 0 = no bake has run yet.
    uint64_t LastBakedHash = 0;
    bool     LutBaked      = false;

    // Sizes (constants for v1 — could be promoted to init params later).
    static constexpr uint32_t kSkySize        = 256;
    static constexpr uint32_t kIrrSize        =  32;
    static constexpr uint32_t kPrefilterSize  = 128;
    static constexpr uint32_t kPrefilterMips  =   6;   // 128 -> 64 -> 32 -> 16 -> 8 -> 4
    static constexpr uint32_t kBrdfLutSize    = 256;
    static constexpr uint32_t kPrefilterSampleCount = 64;
    static constexpr uint32_t kBrdfSampleCount      = 1024;
};

bool SkyAtmosphereInitialize(SkyAtmosphere& sa, const VulkanContext& ctx,
                             const char* shaderArtifactsDir);
void SkyAtmosphereTerminate (SkyAtmosphere& sa, const VulkanContext& ctx);

// Records a one-shot compute job to (re)bake the sky / irradiance / prefilter
// cubemaps when `settings` changed since last call. The BRDF LUT only bakes on
// the first call (it's independent of settings). Safe to call every frame —
// the hash compare gates the actual GPU work.
void SkyAtmosphereEnsureBaked(SkyAtmosphere& sa, const VulkanContext& ctx,
                              VkCommandBuffer cmd,
                              const SkySettings& settings);

} // namespace RS
