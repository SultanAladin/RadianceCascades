// Source/Renderer/GBuffer.h — Phase 6
// Replaces the throw-away ForwardPass. Writes 5 colour attachments + depth
// into the OffscreenTargets ring, transitioning each to SHADER_READ_ONLY_OPTIMAL
// at the end of the pass (via finalLayout) so the preview blit / Phase 7
// lighting compose can sample them without an explicit barrier.
//
// One framebuffer per frame-in-flight, keyed off VulkanContext.FrameIndex (NOT
// the swapchain image index — GBuffer never touches the swapchain).
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/OffscreenTargets.h"
#include "RS/Material.h"      // MaterialRegistry, MaterialHandle
#include "RS/Renderer.h"      // CameraView
#include "Scene/MeshRegistry.h"
#include "Scene/InstanceRegistry.h"

#include <array>
#include <glm/glm.hpp>

namespace RS {

// GPU-side flattened slice of a PbrMaterial — what the GBuffer push constant
// actually carries. Phase 9 builds one of these per submesh draw from the
// instance's MaterialBinding. The Phase 6 "one global material" pathway lives
// on as a fallback used when an instance's binding resolves to an invalid
// handle (and when Phase 6's Sun panel hot-reloads a tweaked default).
struct GBufferMaterial {
    glm::vec3 BaseColor = glm::vec3(0.78f, 0.78f, 0.80f);
    float     Roughness = 0.55f;
    float     Metallic  = 0.0f;
    float     F0        = 0.04f;     // dielectric reference
    glm::vec3 Emissive  = glm::vec3(0.0f);
};

// Flatten a PbrMaterial into the smaller GPU push-constant shape. F0 is the
// red channel of F0Flat for v1 — multi-channel F0 is a Phase 11+ topic that
// also brings textured Fresnel.
GBufferMaterial GBufferMaterialFromPbr(const PbrMaterial& m);

struct GBufferPass {
    VkRenderPass     RenderPass     = VK_NULL_HANDLE;
    VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
    VkPipeline       Pipeline       = VK_NULL_HANDLE;

    std::array<VkFramebuffer, VulkanContext::kFramesInFlight> Framebuffers{};
    bool             Initialized    = false;
};

bool GBufferPassInitialize(GBufferPass& gp, const VulkanContext& ctx,
                           const OffscreenTargets& targets,
                           const char* shaderArtifactsDir);
void GBufferPassTerminate (GBufferPass& gp, const VulkanContext& ctx);

// Phase 9: loop submeshes per instance. The MaterialRegistry is consulted to
// resolve each instance's per-submesh MaterialHandle binding into a
// GBufferMaterial. `fallback` is what we use when the binding resolves to a
// handle outside the registry (e.g. a stale binding after Destroy) — typically
// the seeded "Default" material from MaterialSeed.
void GBufferPassRecord(GBufferPass& gp, const VulkanContext& ctx,
                       VkCommandBuffer cmd, uint32_t frameSlot,
                       const CameraView& camera,
                       const MeshRegistry& meshes,
                       const InstanceRegistry& instances,
                       MaterialRegistry& materials,
                       const GBufferMaterial& fallback);

} // namespace RS
