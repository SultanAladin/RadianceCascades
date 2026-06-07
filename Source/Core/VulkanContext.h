// Source/Core/VulkanContext.h
// Vulkan instance / surface / device / swapchain / sync plumbing used by the
// standalone exe. When linked into a host (Pigment / future game engine), the
// host owns the device and we never construct this type.
//
// v1 simplifications:
//   * Non-resizable window — swapchain is created once and torn down at exit.
//   * FIFO present mode (vsync). No mailbox / immediate modes.
//   * BGRA8_UNORM swapchain format, picked from the first compatible surface format.
//   * Validation layers gated on _DEBUG (off in /O2 build).
#pragma once

#define WIN32_LEAN_AND_MEAN
// Suppress the windows.h min/max macros — they mangle std::min / std::max in any
// TU that combines this header with the STL (e.g. GI/CascadeSpecification.hpp).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// vkCreateWin32SurfaceKHR + VK_KHR_WIN32_SURFACE_EXTENSION_NAME live behind
// this platform define in vulkan_win32.h. Must be defined before the include.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace RS {

struct VulkanContext {
    // ---- Construction handles --------------------------------------------
    VkInstance         Instance            = VK_NULL_HANDLE;
    VkPhysicalDevice   PhysicalDevice      = VK_NULL_HANDLE;
    VkDevice           Device              = VK_NULL_HANDLE;
    VkSurfaceKHR       Surface             = VK_NULL_HANDLE;

    VkQueue            GraphicsQueue       = VK_NULL_HANDLE;
    uint32_t           GraphicsQueueFamily = 0;

    // ---- Swapchain --------------------------------------------------------
    VkSwapchainKHR             Swapchain        = VK_NULL_HANDLE;
    VkFormat                   SwapchainFormat  = VK_FORMAT_UNDEFINED;
    VkExtent2D                 SwapchainExtent  = { 0, 0 };
    std::vector<VkImage>       SwapchainImages;
    std::vector<VkImageView>   SwapchainViews;

    // ---- Per-frame sync + command pool -----------------------------------
    // Double-buffered: at most 2 frames in flight.
    static constexpr uint32_t kFramesInFlight = 2;

    VkCommandPool   CommandPool                       = VK_NULL_HANDLE;
    VkCommandBuffer CommandBuffers[kFramesInFlight]   = {};
    VkSemaphore     ImageAvailableSem[kFramesInFlight]= {};   // signalled by vkAcquireNextImageKHR
    VkSemaphore     RenderFinishedSem[kFramesInFlight]= {};   // signalled by submit
    VkFence         FrameFences      [kFramesInFlight]= {};   // CPU/GPU rendezvous per frame

    uint32_t        FrameIndex                        = 0;    // cycles through kFramesInFlight

    // ---- Lifecycle --------------------------------------------------------
    bool Initialize(HINSTANCE hInstance, HWND hwnd,
                    uint32_t clientWidth, uint32_t clientHeight,
                    bool enableValidation);
    void Terminate();
};

// Single helper used by Main.cpp in Phase 2 — records a clear of the current
// swapchain image, transitions to PRESENT_SRC, submits, and presents. Returns
// false on swapchain failure (e.g. resize, but v1 ignores resize).
bool VulkanContextRenderClearFrame(VulkanContext& ctx, const float clearRGBA[4]);

// Split API used from Phase 3 onward (ImGui draws into the same command buffer
// as the clear). `BeginFrame` acquires an image, transitions it to
// COLOR_ATTACHMENT_OPTIMAL, begins the command buffer, and returns the acquired
// image index plus the recording command buffer. `EndFrame` transitions to
// PRESENT_SRC_KHR, submits with the acquire/present semaphores, and presents.
//
// Caller pattern:
//   FrameInfo f;
//   if (VulkanContextBeginFrame(ctx, &f)) {
//       vkCmdClearAttachments / record render passes against f.Cmd, f.ImageIndex
//       VulkanContextEndFrame(ctx, f);
//   }
struct FrameInfo {
    VkCommandBuffer Cmd        = VK_NULL_HANDLE;
    uint32_t        ImageIndex = 0;
    uint32_t        FrameSlot  = 0;     // index into kFramesInFlight
};
bool VulkanContextBeginFrame(VulkanContext& ctx, FrameInfo* outFrame);
bool VulkanContextEndFrame  (VulkanContext& ctx, const FrameInfo& frame);

} // namespace RS
