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
    // Sun appearance. SunColor + SunAngularSizeDeg drive the analytic sun disc
    // baked into the sky cube; SunDirection is mirrored from the Sun panel by
    // Main.cpp so the cube tracks the same sun the lighting compose uses.
    glm::vec3 SunColor          = glm::vec3(1.0f, 0.96f, 0.88f);
    float     SunIntensity      = 8.0f;          // analytic disc brightness only
    float     SunAngularSizeDeg = 1.5f;
    glm::vec3 SunDirection      = glm::normalize(glm::vec3(0.3f, 0.85f, 0.5f));

    bool      EnableIBL    = true;               // Sun/Sky panel toggle, plumbed to PbrLit
    float     IBLIntensity = 1.0f;               // overall IBL gain

    // Master gain on the scattered (non-sun-disc) sky radiance. Keeps the IBL
    // ambient from overpowering the rest of the lighting when sun intensity is
    // pushed up for the analytic disc.
    float     SkyIntensity = 1.0f;

    // Bake the 3D Aerial-Perspective LUT. No consumer yet — wires up future
    // fog-on-geometry. Off by default to keep per-frame cost minimal.
    bool      BakeAerialPerspective = false;

    // Atmosphere parameters. Defaults match Bruneton 2017 / Hillaire 2020
    // sea-level coefficients (units: km for radii, 1/m for β).
    float     PlanetRadiusKm        = 6360.0f;
    float     AtmosphereRadiusKm    = 6460.0f;
    glm::vec3 RayleighScattering    = glm::vec3(5.802e-6f, 13.558e-6f, 33.1e-6f);
    float     MieScattering         = 3.996e-6f;
    float     MieG                  = 0.8f;
    glm::vec3 OzoneAbsorption       = glm::vec3(0.650e-6f, 1.881e-6f, 0.085e-6f);

    // Camera altitude above sea level (m). Plumbs through to the SkyView
    // raymarch; for ground-level cameras keep near 0. Bumped to the planet
    // surface (epsilon) by the bake to avoid r == R_planet edge case.
    float     CameraAltitudeMeters  = 2.0f;
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

// Plain 2D / 3D image — used by the Hillaire LUTs (transmittance, multiscatter,
// skyview = 2D RGBA16F; aerial perspective = 3D RGBA16F).
struct LutImage2D {
    VkImage        Image  = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView    View   = VK_NULL_HANDLE;
    uint32_t       Width  = 0;
    uint32_t       Height = 0;
    VkFormat       Format = VK_FORMAT_UNDEFINED;
};
struct LutImage3D {
    VkImage        Image  = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView    View   = VK_NULL_HANDLE;
    uint32_t       Width  = 0;
    uint32_t       Height = 0;
    uint32_t       Depth  = 0;
    VkFormat       Format = VK_FORMAT_UNDEFINED;
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

    // Hillaire LUTs (Phase B).
    LutImage2D Transmittance;     // 256×64 RGBA16F
    LutImage2D MultiScatter;      // 32×32 RGBA16F
    LutImage2D SkyView;           // 192×108 RGBA16F
    LutImage3D AerialPerspective; // 32×32×32 RGBA16F

    VkSampler      SamplerCubeLinear = VK_NULL_HANDLE;   // mips + linear, edge clamp
    VkSampler      SamplerLut        = VK_NULL_HANDLE;   // linear, edge clamp, no mips
    VkSampler      SamplerLutLinear  = VK_NULL_HANDLE;   // linear, edge clamp, no mips (Hillaire 2D LUTs)

    // Compute pipelines — all Hillaire 2020 + IBL chain.
    VkDescriptorPool      DescriptorPool          = VK_NULL_HANDLE;
    VkDescriptorSetLayout IrradianceSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout PrefilterSetLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout BrdfSetLayout           = VK_NULL_HANDLE;
    VkDescriptorSetLayout TransmittanceSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout MultiScatterSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout SkyViewSetLayout        = VK_NULL_HANDLE;
    VkDescriptorSetLayout AerialSetLayout         = VK_NULL_HANDLE;
    VkDescriptorSetLayout CubeFromSkyViewLayout   = VK_NULL_HANDLE;

    VkPipelineLayout IrradiancePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout PrefilterPipelineLayout  = VK_NULL_HANDLE;
    VkPipelineLayout BrdfPipelineLayout       = VK_NULL_HANDLE;
    VkPipelineLayout TransmittancePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout MultiScatterPipelineLayout  = VK_NULL_HANDLE;
    VkPipelineLayout SkyViewPipelineLayout       = VK_NULL_HANDLE;
    VkPipelineLayout AerialPipelineLayout        = VK_NULL_HANDLE;
    VkPipelineLayout CubeFromSkyViewPipelineLayout = VK_NULL_HANDLE;

    VkPipeline IrradiancePipeline = VK_NULL_HANDLE;
    VkPipeline PrefilterPipeline  = VK_NULL_HANDLE;
    VkPipeline BrdfPipeline       = VK_NULL_HANDLE;
    VkPipeline TransmittancePipeline = VK_NULL_HANDLE;
    VkPipeline MultiScatterPipeline  = VK_NULL_HANDLE;
    VkPipeline SkyViewPipeline       = VK_NULL_HANDLE;
    VkPipeline AerialPipeline        = VK_NULL_HANDLE;
    VkPipeline CubeFromSkyViewPipeline = VK_NULL_HANDLE;

    VkDescriptorSet IrradianceDescriptor = VK_NULL_HANDLE;
    // One prefilter descriptor per mip — each one writes a different storage
    // image view into binding 1.
    std::array<VkDescriptorSet, 16> PrefilterDescriptors{};
    VkDescriptorSet BrdfDescriptor       = VK_NULL_HANDLE;
    VkDescriptorSet TransmittanceDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet MultiScatterDescriptor  = VK_NULL_HANDLE;
    VkDescriptorSet SkyViewDescriptor       = VK_NULL_HANDLE;
    VkDescriptorSet AerialDescriptor        = VK_NULL_HANDLE;
    VkDescriptorSet CubeFromSkyViewDescriptor = VK_NULL_HANDLE;

    // Cached hash of the SkySettings the current textures were baked from.
    // 0 = no bake has run yet.
    uint64_t LastBakedHash       = 0;
    uint64_t LastAtmosphereHash  = 0;  // For Transmittance / MultiScatter gating.
    bool     TransmittanceBaked  = false;
    bool     MultiScatterBaked   = false;
    bool     LutBaked            = false;

    // Sizes (constants for v1 — could be promoted to init params later).
    static constexpr uint32_t kSkySize        = 256;
    static constexpr uint32_t kIrrSize        =  32;
    static constexpr uint32_t kPrefilterSize  = 128;
    static constexpr uint32_t kPrefilterMips  =   6;   // 128 -> 64 -> 32 -> 16 -> 8 -> 4
    static constexpr uint32_t kBrdfLutSize    = 256;
    static constexpr uint32_t kPrefilterSampleCount = 64;
    static constexpr uint32_t kBrdfSampleCount      = 1024;

    static constexpr uint32_t kTransmittanceWidth   = 256;
    static constexpr uint32_t kTransmittanceHeight  = 64;
    static constexpr uint32_t kTransmittanceSteps   = 40;
    static constexpr uint32_t kMultiScatterSize     = 32;
    static constexpr uint32_t kMultiScatterSteps    = 20;
    static constexpr uint32_t kMultiScatterSqrtN    = 8;  // 8×8 = 64 sphere samples
    static constexpr uint32_t kSkyViewWidth         = 192;
    static constexpr uint32_t kSkyViewHeight        = 108;
    static constexpr uint32_t kSkyViewSteps         = 32;
    static constexpr uint32_t kAerialSize           = 32;
    static constexpr uint32_t kAerialStepsPerSlice  = 4;
    // Aerial-Perspective fog covers ~32 km — 1 km per depth slice.
    static constexpr float    kAerialSliceMeters    = 1000.0f;
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
