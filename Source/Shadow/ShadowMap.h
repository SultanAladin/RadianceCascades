// Source/Shadow/ShadowMap.h — Phase 10
// 4-cascade CSM atlas used by PCF / PCSS / VSM. SDFCone bypasses this entirely
// (it traces the global SDF) — that algo doesn't construct a ShadowMap.
//
// Layout decisions (locked in chat 2026-05-28):
//   * Single VkImage, VK_FORMAT_D32_SFLOAT, arrayLayers = kCascadeCount (4).
//     One 2D-array view sampled with sampler2DArrayShadow in PCF/PCSS; one
//     per-layer 2D view used as a depth attachment when rendering each cascade.
//   * Per-cascade splits: 0.1 / 0.25 / 0.5 / 1.0 of (FarClip - NearClip), then
//     each split's view-frustum slice is wrapped in a sphere fit and snapped to
//     a stable texel grid so cascade edges don't shimmer when the camera moves
//     (Stable CSM, MJP-style).
//   * Depth-only pipeline: vertex stream is the same {pos, normal} ParsedVertex
//     the GBufferPass reads; the shadow vertex shader only consumes position.
//     No fragment shader bound (depthOnly subpass with rasterizationDiscard
//     left off so depth writes still happen).
//
// Output for PCF/PCSS to consume:
//   * Image (D32_SFLOAT, 4 layers)
//   * ArrayView (VIEW_TYPE_2D_ARRAY) — sample with sampler2DArrayShadow
//   * CascadeViewProj[4]   — light-space view-proj per cascade (Vulkan-style,
//                            depth range [0,1], Y not flipped)
//   * CascadeSplitDistance[4] — the far-plane distance of each cascade slice
//                               in view-space (used by the lighting compute to
//                               pick which cascade to sample per fragment)
#pragma once

#include "Core/VulkanContext.h"
#include "RS/Renderer.h"   // CameraView
#include "Scene/MeshRegistry.h"
#include "Scene/InstanceRegistry.h"

#include <array>
#include <glm/glm.hpp>

namespace RS {

struct ShadowMap {
    static constexpr uint32_t kCascadeCount = 4;
    // Per-cascade resolution (square). One 4-layer D32_SFLOAT image at this
    // size. 2048 → 64 MB. Bumping to 4096 (256 MB) is a v2 quality preset.
    static constexpr uint32_t kCascadeSize  = 2048;

    // -------- GPU resources --------
    VkImage        AtlasImage     = VK_NULL_HANDLE;
    VkDeviceMemory AtlasMemory    = VK_NULL_HANDLE;
    VkImageView    AtlasArrayView = VK_NULL_HANDLE;                       // sampled
    std::array<VkImageView, kCascadeCount> CascadeLayerViews{};           // depth-attached per cascade

    // Depth-only pipeline + per-cascade framebuffers. The pipeline is shared;
    // we render each cascade by binding the right framebuffer.
    VkRenderPass     RenderPass       = VK_NULL_HANDLE;
    VkPipelineLayout PipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       Pipeline         = VK_NULL_HANDLE;
    std::array<VkFramebuffer, kCascadeCount> CascadeFramebuffers{};

    // -------- Frame-local state (refreshed in BuildFrame()) --------
    std::array<glm::mat4, kCascadeCount> CascadeViewProj{};
    std::array<float,     kCascadeCount> CascadeSplitDistance{};   // view-space far plane per cascade
    std::array<float,     kCascadeCount> CascadeSphereRadius{};    // for texel-snap reuse / debug

    VkFormat Format     = VK_FORMAT_D32_SFLOAT;
    bool     Initialized = false;
};

bool ShadowMapInitialize(ShadowMap& sm, const VulkanContext& ctx,
                         const char* shaderArtifactsDir);
void ShadowMapTerminate (ShadowMap& sm, const VulkanContext& ctx);

// Computes per-cascade split planes, fits each split's view frustum into a
// world-space sphere, snaps to a stable texel grid, and writes
// CascadeViewProj/CascadeSplitDistance/CascadeSphereRadius. Cheap — no GPU
// work. Call once per frame before ShadowMapRecord.
void ShadowMapBuildCascades(ShadowMap& sm,
                            const CameraView& camera,
                            const glm::vec3& sunDirectionToward);

// Records 4 depth-only render passes (one per cascade) drawing every instance
// in `instances` once per cascade. The depth-only pipeline reads positions
// from the same vertex stream the GBufferPass binds, so MeshRegistry doesn't
// need a second upload.
void ShadowMapRecord(ShadowMap& sm, const VulkanContext& ctx,
                     VkCommandBuffer cmd,
                     const MeshRegistry& meshes,
                     const InstanceRegistry& instances);

} // namespace RS
