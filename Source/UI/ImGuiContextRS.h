// Source/UI/ImGuiContextRS.h — Phase 3
// Owns the ImGui Vulkan backend + render-pass + per-swapchain-image framebuffer.
// Lifetime: bring up after VulkanContext, tear down before VulkanContext.
//
// Per frame:
//   ImGuiContextRSNewFrame()                 // Win32 + Vulkan backend new-frame, ImGui::NewFrame
//   ImGui::Begin(...)/End(...)               // user
//   ImGuiContextRSRender(frame.Cmd, idx,
//                        clearRGBA)          // render-pass clears the swap image + draws ImGui
#pragma once

#include "Core/VulkanContext.h"

namespace RS {

struct ImGuiContextRS {
    // Vulkan owned objects.
    VkRenderPass                 RenderPass     = VK_NULL_HANDLE;
    VkDescriptorPool             DescriptorPool = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>   Framebuffers;
    bool                         Initialized    = false;
};

bool ImGuiContextRSInitialize(ImGuiContextRS& gui, const VulkanContext& ctx, HWND hwnd);
void ImGuiContextRSTerminate (ImGuiContextRS& gui, const VulkanContext& ctx);

void ImGuiContextRSNewFrame  ();   // Win32 + Vulkan NewFrame + ImGui::NewFrame
// Phase 4+: the prior pass (GridPass) already owns the swap-image clear and
// left it in COLOR_ATTACHMENT_OPTIMAL — this pass uses loadOp=LOAD and paints
// ImGui on top, no clear colour needed.
void ImGuiContextRSRender    (ImGuiContextRS& gui, const VulkanContext& ctx,
                              VkCommandBuffer cmd, uint32_t imageIndex);

} // namespace RS
