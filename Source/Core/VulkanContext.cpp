// Source/Core/VulkanContext.cpp — Phase 2
// Instance / surface / device / swapchain / sync + a single clear-and-present
// helper. Plain raw Vulkan, no helper libraries.
#include "Core/VulkanContext.h"
#include "Core/Logger.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace RS {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

#define VKCHECK(expr)                                                                 \
    do {                                                                              \
        VkResult _r = (expr);                                                         \
        if (_r != VK_SUCCESS) {                                                       \
            RS_LOG_ERROR("VK call failed (%d) at %s:%d : %s", _r, __FILE__, __LINE__, \
                         #expr);                                                      \
            return false;                                                             \
        }                                                                             \
    } while (0)

bool LayerAvailable(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> props(count);
    vkEnumerateInstanceLayerProperties(&count, props.data());
    for (const auto& p : props) {
        if (std::strcmp(p.layerName, name) == 0) return true;
    }
    return false;
}

// Pick a graphics queue family that also supports presenting to our surface.
int FindGraphicsQueueFamily(VkPhysicalDevice pd, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        const bool gfx = (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        if (gfx && present) return static_cast<int>(i);
    }
    return -1;
}

VkSurfaceFormatKHR PickSurfaceFormat(VkPhysicalDevice pd, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats.data());

    // Prefer BGRA8_UNORM + SRGB_NONLINEAR; fall back to first.
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.empty() ? VkSurfaceFormatKHR{ VK_FORMAT_B8G8R8A8_UNORM,
                                                 VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }
                           : formats[0];
}

VkExtent2D ClampExtent(VkExtent2D requested, const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }
    VkExtent2D e = requested;
    e.width  = std::clamp(e.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    e.height = std::clamp(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return e;
}

} // namespace

bool VulkanContext::Initialize(HINSTANCE hInstance, HWND hwnd,
                               uint32_t clientWidth, uint32_t clientHeight,
                               bool enableValidation) {
    // ---- 1. Instance ------------------------------------------------------
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName   = "RenderingSubsystem";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName        = "RenderingSubsystem";
    app.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_2;

    std::vector<const char*> instanceExts = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    };
    std::vector<const char*> instanceLayers;
    if (enableValidation && LayerAvailable(kValidationLayer)) {
        instanceLayers.push_back(kValidationLayer);
        RS_LOG_INFO("Validation layer enabled");
    } else if (enableValidation) {
        RS_LOG_WARN("Validation layer requested but VK_LAYER_KHRONOS_validation not installed");
    }

    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo        = &app;
    ici.enabledExtensionCount   = static_cast<uint32_t>(instanceExts.size());
    ici.ppEnabledExtensionNames = instanceExts.data();
    ici.enabledLayerCount       = static_cast<uint32_t>(instanceLayers.size());
    ici.ppEnabledLayerNames     = instanceLayers.data();

    VKCHECK(vkCreateInstance(&ici, nullptr, &Instance));

    // ---- 2. Surface (Win32) ----------------------------------------------
    VkWin32SurfaceCreateInfoKHR sci{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    sci.hinstance = hInstance;
    sci.hwnd      = hwnd;
    VKCHECK(vkCreateWin32SurfaceKHR(Instance, &sci, nullptr, &Surface));

    // ---- 3. Physical device + queue family --------------------------------
    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(Instance, &pdCount, nullptr);
    if (pdCount == 0) {
        RS_LOG_ERROR("No Vulkan physical devices found");
        return false;
    }
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(Instance, &pdCount, pds.data());

    // Prefer a discrete GPU; fall back to first compatible.
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    int              chosenQueue = -1;
    for (auto pd : pds) {
        const int q = FindGraphicsQueueFamily(pd, Surface);
        if (q < 0) continue;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            chosen      = pd;
            chosenQueue = q;
            break;
        }
        if (chosen == VK_NULL_HANDLE) {
            chosen      = pd;
            chosenQueue = q;
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        RS_LOG_ERROR("No physical device with graphics + present support");
        return false;
    }
    PhysicalDevice      = chosen;
    GraphicsQueueFamily = static_cast<uint32_t>(chosenQueue);

    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(PhysicalDevice, &props);
        RS_LOG_INFO("Vulkan device: %s (queue family %u)",
                    props.deviceName, GraphicsQueueFamily);
    }

    // ---- 4. Logical device + queue ---------------------------------------
    const float qPriority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = GraphicsQueueFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &qPriority;

    std::array<const char*, 1> deviceExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<uint32_t>(deviceExts.size());
    dci.ppEnabledExtensionNames = deviceExts.data();
    VKCHECK(vkCreateDevice(PhysicalDevice, &dci, nullptr, &Device));
    vkGetDeviceQueue(Device, GraphicsQueueFamily, 0, &GraphicsQueue);

    // ---- 5. Swapchain -----------------------------------------------------
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(PhysicalDevice, Surface, &caps);

    const VkSurfaceFormatKHR fmt = PickSurfaceFormat(PhysicalDevice, Surface);
    SwapchainFormat = fmt.format;
    SwapchainExtent = ClampExtent({ clientWidth, clientHeight }, caps);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR scci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    scci.surface          = Surface;
    scci.minImageCount    = imageCount;
    scci.imageFormat      = fmt.format;
    scci.imageColorSpace  = fmt.colorSpace;
    scci.imageExtent      = SwapchainExtent;
    scci.imageArrayLayers = 1;
    // TRANSFER_DST so we can vkCmdClearColorImage + vkCmdBlitImage later.
    scci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    scci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    scci.preTransform     = caps.currentTransform;
    scci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    scci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;            // vsync
    scci.clipped          = VK_TRUE;
    VKCHECK(vkCreateSwapchainKHR(Device, &scci, nullptr, &Swapchain));

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(Device, Swapchain, &actualCount, nullptr);
    SwapchainImages.resize(actualCount);
    vkGetSwapchainImagesKHR(Device, Swapchain, &actualCount, SwapchainImages.data());

    SwapchainViews.resize(actualCount);
    for (uint32_t i = 0; i < actualCount; ++i) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image      = SwapchainImages[i];
        vci.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        vci.format     = SwapchainFormat;
        vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VKCHECK(vkCreateImageView(Device, &vci, nullptr, &SwapchainViews[i]));
    }
    RS_LOG_INFO("Swapchain: %ux%u x %u images, format %d",
                SwapchainExtent.width, SwapchainExtent.height, actualCount,
                static_cast<int>(SwapchainFormat));

    // ---- 6. Command pool + per-frame command buffers ----------------------
    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = GraphicsQueueFamily;
    VKCHECK(vkCreateCommandPool(Device, &cpci, nullptr, &CommandPool));

    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool        = CommandPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kFramesInFlight;
    VKCHECK(vkAllocateCommandBuffers(Device, &cbai, CommandBuffers));

    // ---- 7. Sync objects (per frame in flight) ---------------------------
    VkSemaphoreCreateInfo sciSem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo     fci   { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // first frame's wait must succeed immediately
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VKCHECK(vkCreateSemaphore(Device, &sciSem, nullptr, &ImageAvailableSem[i]));
        VKCHECK(vkCreateSemaphore(Device, &sciSem, nullptr, &RenderFinishedSem[i]));
        VKCHECK(vkCreateFence    (Device, &fci,    nullptr, &FrameFences      [i]));
    }

    return true;
}

void VulkanContext::Terminate() {
    if (Device) {
        vkDeviceWaitIdle(Device);
    }

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (FrameFences      [i]) vkDestroyFence    (Device, FrameFences      [i], nullptr);
        if (ImageAvailableSem[i]) vkDestroySemaphore(Device, ImageAvailableSem[i], nullptr);
        if (RenderFinishedSem[i]) vkDestroySemaphore(Device, RenderFinishedSem[i], nullptr);
        FrameFences      [i] = VK_NULL_HANDLE;
        ImageAvailableSem[i] = VK_NULL_HANDLE;
        RenderFinishedSem[i] = VK_NULL_HANDLE;
    }
    if (CommandPool) {
        vkDestroyCommandPool(Device, CommandPool, nullptr);
        CommandPool = VK_NULL_HANDLE;
    }

    for (VkImageView v : SwapchainViews) {
        if (v) vkDestroyImageView(Device, v, nullptr);
    }
    SwapchainViews.clear();
    SwapchainImages.clear();
    if (Swapchain) {
        vkDestroySwapchainKHR(Device, Swapchain, nullptr);
        Swapchain = VK_NULL_HANDLE;
    }
    if (Device) {
        vkDestroyDevice(Device, nullptr);
        Device = VK_NULL_HANDLE;
    }
    if (Surface) {
        vkDestroySurfaceKHR(Instance, Surface, nullptr);
        Surface = VK_NULL_HANDLE;
    }
    if (Instance) {
        vkDestroyInstance(Instance, nullptr);
        Instance = VK_NULL_HANDLE;
    }
}

// ----------------------------------------------------------------------------
// Phase 3 split-frame API. `BeginFrame` acquires + transitions the swapchain
// image into COLOR_ATTACHMENT_OPTIMAL (so an ImGui render-pass loadOp=CLEAR can
// own the clear-colour); `EndFrame` transitions to PRESENT_SRC and submits.
// ----------------------------------------------------------------------------
bool VulkanContextBeginFrame(VulkanContext& ctx, FrameInfo* outFrame) {
    const uint32_t frame = ctx.FrameIndex;
    vkWaitForFences(ctx.Device, 1, &ctx.FrameFences[frame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(ctx.Device, ctx.Swapchain, UINT64_MAX,
                                         ctx.ImageAvailableSem[frame], VK_NULL_HANDLE,
                                         &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;     // v1 ignores resize
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        RS_LOG_ERROR("vkAcquireNextImageKHR failed: %d", static_cast<int>(acq));
        return false;
    }

    vkResetFences(ctx.Device, 1, &ctx.FrameFences[frame]);

    VkCommandBuffer cmd = ctx.CommandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // UNDEFINED → COLOR_ATTACHMENT_OPTIMAL. The ImGui render-pass uses
    // loadOp=CLEAR with initialLayout=COLOR_ATTACHMENT_OPTIMAL so the implicit
    // subpass barrier inside the render-pass doesn't have to do this.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = ctx.SwapchainImages[imageIndex];
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    if (outFrame) {
        outFrame->Cmd        = cmd;
        outFrame->ImageIndex = imageIndex;
        outFrame->FrameSlot  = frame;
    }
    return true;
}

bool VulkanContextEndFrame(VulkanContext& ctx, const FrameInfo& frame) {
    VkCommandBuffer cmd = frame.Cmd;

    // COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR for the surface composite.
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask       = 0;
        b.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = ctx.SwapchainImages[frame.ImageIndex];
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &ctx.ImageAvailableSem[frame.FrameSlot];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &ctx.RenderFinishedSem[frame.FrameSlot];
    vkQueueSubmit(ctx.GraphicsQueue, 1, &si, ctx.FrameFences[frame.FrameSlot]);

    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &ctx.RenderFinishedSem[frame.FrameSlot];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &ctx.Swapchain;
    pi.pImageIndices      = &frame.ImageIndex;
    VkResult pr = vkQueuePresentKHR(ctx.GraphicsQueue, &pi);
    if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
        // v1: don't recreate on OUT_OF_DATE; just continue.
    }

    ctx.FrameIndex = (frame.FrameSlot + 1) % VulkanContext::kFramesInFlight;
    return true;
}

// ----------------------------------------------------------------------------
// Phase 2 single-pass clear helper. Kept compiling so any caller that still
// needs a plain clear-and-present can use it, but Main.cpp moved to the
// split BeginFrame/EndFrame API in Phase 3.
// ----------------------------------------------------------------------------
bool VulkanContextRenderClearFrame(VulkanContext& ctx, const float clearRGBA[4]) {
    FrameInfo f{};
    if (!VulkanContextBeginFrame(ctx, &f)) return false;

    VkClearColorValue clr{};
    clr.float32[0] = clearRGBA[0];
    clr.float32[1] = clearRGBA[1];
    clr.float32[2] = clearRGBA[2];
    clr.float32[3] = clearRGBA[3];

    // We're already in COLOR_ATTACHMENT_OPTIMAL after BeginFrame; transition
    // briefly to TRANSFER_DST so vkCmdClearColorImage is legal, then back.
    auto transition = [&](VkImageLayout oldL, VkImageLayout newL,
                          VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                          VkPipelineStageFlags srcStage,
                          VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = oldL;
        b.newLayout           = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = ctx.SwapchainImages[f.ImageIndex];
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(f.Cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    transition(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(f.Cmd, ctx.SwapchainImages[f.ImageIndex],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &range);

    transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    return VulkanContextEndFrame(ctx, f);
}

} // namespace RS
