// Source/Renderer/GridPass.h — Phase 4
// Full-screen ray-marched ground grid. Owns a render-pass + pipeline + per-
// swap-image framebuffer, draws a single fullscreen triangle each frame, hands
// off the swapchain image in COLOR_ATTACHMENT_OPTIMAL for ImGui to paint over.
//
// Lifetime mirrors ImGuiContextRS: init after VulkanContext, terminate before.
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/SkyAtmosphere.h"
#include "RS/Renderer.h"      // CameraView

namespace RS {

struct GridPass {
    VkRenderPass                 RenderPass     = VK_NULL_HANDLE;
    VkDescriptorSetLayout        SetLayout      = VK_NULL_HANDLE;
    VkPipelineLayout             PipelineLayout = VK_NULL_HANDLE;
    VkPipeline                   Pipeline       = VK_NULL_HANDLE;
    VkDescriptorPool             DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet              SkySet         = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>   Framebuffers;
    bool                         Initialized    = false;
};

// Knobs the panel will eventually edit; CamearView + these constants are all
// the pass needs to record.
struct GridSettings {
    float MajorSpacing   = 1.0f;
    float MinorSpacing   = 0.2f;
    float MajorThickness = 1.4f;
    float MinorThickness = 1.0f;
    float Opacity        = 1.0f;
    float FadeDistance   = 60.0f;
    float FalloffStart   = 8.0f;
    float FalloffCurve   = 1.4f;
    bool  Infinite       = true;
    bool  AxisHighlight  = true;

    // Visual palette. SkyColor is now a tint multiplier on top of the baked sky
    // cubemap — Phase 8 SkyAtmosphere already provides the gradient colour, so
    // the default is plain white.
    float MajorColor[3]  = { 0.78f, 0.78f, 0.82f };
    float MinorColor[3]  = { 0.52f, 0.52f, 0.56f };
    float SkyColor[3]    = { 1.00f, 1.00f, 1.00f };
    float GroundColor[3] = { 0.10f, 0.11f, 0.14f };
    float Extent         = 50.0f;   // only used when Infinite=false

    // Phase 8 polish: classic dev-engine checker floor. CheckerSpacing is in
    // metres; CheckerStrength fades the checker into the GroundColor (0 = off,
    // 1 = full contrast). Two-band greyscale by default, matches Unity/Godot.
    //
    // Phase 11.5: now defaults to 0 because the floor lives in the GBuffer as
    // a real mesh (Source/Scene/FloorMesh.{h,cpp}) which paints the same
    // checker pattern and catches shadows. The GridPass branch survives so
    // panels-with-no-floor (e.g. an editor host that turns the floor off) can
    // still get a procedural ground at glancing angles.
    float CheckerSpacing  = 1.0f;
    float CheckerStrength = 0.00f;
    float CheckerLight[3] = { 0.58f, 0.58f, 0.60f };
    float CheckerDark [3] = { 0.34f, 0.34f, 0.36f };
};

bool GridPassInitialize(GridPass& gp, const VulkanContext& ctx,
                        const SkyAtmosphere& sky,
                        const char* shaderArtifactsDir);
void GridPassTerminate (GridPass& gp, const VulkanContext& ctx);

void GridPassRecord(GridPass& gp, const VulkanContext& ctx,
                    VkCommandBuffer cmd, uint32_t imageIndex,
                    const CameraView& camera,
                    const GridSettings& settings);

} // namespace RS
