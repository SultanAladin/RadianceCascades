// Source/Renderer/GBufferPreview.h — Phase 6
// Fullscreen pass on the swapchain that samples one frame's GBuffer attachment
// (chosen by Debug panel) and writes it where geometry exists. Behind the
// ShaderBall silhouette the grid still shows because the pass `discard`s where
// the GBuffer depth attachment is 1.0.
//
// Owns a descriptor pool + per-frame-in-flight descriptor sets (one set per
// OffscreenFrame), a render-pass on the swapchain (loadOp=LOAD), pipeline,
// and per-swapchain-image framebuffers.
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/OffscreenTargets.h"
#include "Renderer/SkyAtmosphere.h"
#include "RS/Renderer.h"      // CameraView

#include <array>
#include <vector>
#include <glm/glm.hpp>

namespace RS {

enum class GBufferPreviewMode : int {
    Lit       = 0,
    Albedo    = 1,
    Normal    = 2,
    RMF       = 3,
    Emissive  = 4,
    Depth     = 5,
    Identity  = 6,
    Matcap    = 7,
    SDFSlice  = 8,   // Phase 12 — visualises the active mesh SDF on a Y-plane
};

// Phase 12: SDF-slice debug view state. The active mesh's SDF image view + a
// few uniforms get bound at set=0 binding=10/11. When no SDF is resident the
// preview falls back to a 1^3 dummy and the slice mode is a flat colour.
// Phase 12.5: how the SDF slice mode visualises the field. Mirrored to the
// CPU slice preview in SdfBakerPanel.cpp so on-screen and panel look identical.
enum class SDFVizMode : int {
    SignedRGB  = 0,   // original Phase 12 viz (red=inside, blue=outside, green iso band)
    Heatmap    = 1,   // jet/inferno ramp on |distance|
    Grayscale  = 2,   // |distance| → linear grey ramp
    SignedBW   = 3,   // hard step: inside black, outside white, thin iso band
    GradientMag= 4,   // central-difference |∇SDF|, deviation from 1.0
};

// Phase 15f — slice can read from either the dense .rsdf 3D texture (legacy)
// or the sparse .rsdfvdb SSBOs. The two sources have independent AABBs so the
// panel picks whichever residency is present for that mesh.
enum class SDFSliceSource : int {
    Dense  = 0,
    Sparse = 1,
};

struct SDFSliceState {
    VkImageView View      = VK_NULL_HANDLE;  // resident image view (or dummy)
    VkSampler   Sampler   = VK_NULL_HANDLE;  // GlobalSDF::Sampler
    glm::vec3   AABBMin   = glm::vec3(0.0f);
    glm::vec3   AABBMax   = glm::vec3(1.0f);
    float       MaxDist   = 1.0f;            // R16_SNORM decode scale
    float       PlaneY    = 0.0f;            // world-space Y for the slice plane
    SDFVizMode  VizMode   = SDFVizMode::SignedRGB;
    bool        HasSDF    = false;

    // Phase 15f sparse-side mirror. Independent AABB because the sparse bake
    // may have different padding than the legacy dense one.
    SDFSliceSource Source         = SDFSliceSource::Dense;
    glm::vec3      SparseAABBMin  = glm::vec3(0.0f);
    glm::vec3      SparseAABBMax  = glm::vec3(1.0f);
    float          SparseMaxDist  = 1.0f;
    uint32_t       SparseRes      = 0;
    uint32_t       SparseBrickSz  = 0;
    uint32_t       SparseBrickGrid= 0;
};

// Procedural matcap settings — driven entirely from the Matcap ImGui panel.
// Shader expands these into the view-space N → colour lookup analytically.
struct MatcapSettings {
    glm::vec3 ColorTop      = glm::vec3(0.92f, 0.90f, 0.88f);
    glm::vec3 ColorBottom   = glm::vec3(0.18f, 0.20f, 0.24f);
    float     GradientCurve = 1.0f;

    glm::vec3 RimColor      = glm::vec3(0.95f, 0.85f, 0.65f);
    float     RimPower      = 3.0f;
    float     RimStrength   = 0.6f;

    glm::vec3 HighlightColor   = glm::vec3(1.0f);
    glm::vec2 HighlightOffset  = glm::vec2(0.35f, 0.35f);
    float     HighlightSize    = 0.18f;
    float     HighlightStrength = 1.2f;

    glm::vec3 Tint            = glm::vec3(1.0f);
    float     Exposure        = 1.0f;

    // Material-ish controls. Roughness widens the highlight (low roughness =
    // sharp lobe). Metallic tints the highlight by the top gradient colour and
    // attenuates the diffuse term.
    float     Roughness       = 0.35f;
    float     Metallic        = 0.0f;
};

struct GBufferPreviewSettings {
    GBufferPreviewMode Mode          = GBufferPreviewMode::Lit;
    glm::vec3          SunDirection  = glm::vec3(0.3f, 0.85f, 0.5f);
    float              SunIntensity  = 1.4f;
    float              Ambient       = 0.10f;
    bool               UseIBL        = true;   // mirrored from SkySettings; lit path
    float              IBLIntensity  = 1.0f;
    MatcapSettings     Matcap;
};

struct GBufferPreviewPass {
    VkRenderPass          RenderPass        = VK_NULL_HANDLE;
    VkDescriptorSetLayout SetLayout         = VK_NULL_HANDLE;
    VkPipelineLayout      PipelineLayout    = VK_NULL_HANDLE;
    VkPipeline            Pipeline          = VK_NULL_HANDLE;
    VkDescriptorPool      DescriptorPool    = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> FrameSets{};
    std::vector<VkFramebuffer> Framebuffers;

    // Phase 12: per-frame UBO carrying SDF-slice params (now 96 bytes after
    // Phase 15f added sparse AABB + Dims). Host-visible coherent so we can
    // memcpy from the main thread before recording.
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> SdfUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> SdfUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> SdfUboMapped{};

    // Phase 15f: dummy 16-byte device-local SSBO bound to bindings 13/14 when
    // no sparse residency exists. Always-bound layout keeps the descriptor set
    // validation-clean even before the first sparse bake.
    VkBuffer       DummySparseBuffer = VK_NULL_HANDLE;
    VkDeviceMemory DummySparseMemory = VK_NULL_HANDLE;
    bool                  Initialized       = false;
};

bool GBufferPreviewInitialize(GBufferPreviewPass& pp, const VulkanContext& ctx,
                              const OffscreenTargets& targets,
                              const SkyAtmosphere& sky,
                              VkSampler sdfFallbackSampler,
                              VkImageView sdfFallbackView,
                              const char* shaderArtifactsDir);
void GBufferPreviewTerminate (GBufferPreviewPass& pp, const VulkanContext& ctx);

// Update the SDF descriptor binding (set=0 binding=10) to point at a new view +
// sampler. Safe to call between frames after vkDeviceWaitIdle, or once at
// startup after the GlobalSDF residency is established.
void GBufferPreviewSetSDF(GBufferPreviewPass& pp, const VulkanContext& ctx,
                          VkImageView view, VkSampler sampler);

// Phase 15f: update the sparse SDF SSBO bindings (set=0 bindings 13+14). Pass
// VK_NULL_HANDLE buffers to revert to the always-bound dummy. Safe to call
// after vkDeviceWaitIdle (the residency-changed branch already gates the
// caller on that).
void GBufferPreviewSetSparseSDF(GBufferPreviewPass& pp, const VulkanContext& ctx,
                                VkBuffer indexBuffer, VkDeviceSize indexBytes,
                                VkBuffer poolBuffer,  VkDeviceSize poolBytes);

void GBufferPreviewRecord(GBufferPreviewPass& pp, const VulkanContext& ctx,
                          VkCommandBuffer cmd, uint32_t imageIndex,
                          uint32_t frameSlot,
                          const CameraView& camera,
                          const GBufferPreviewSettings& settings,
                          const SDFSliceState& sdfSlice);

} // namespace RS
