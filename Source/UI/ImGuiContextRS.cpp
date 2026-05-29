// Source/UI/ImGuiContextRS.cpp — Phase 3
// Bring-up of the ImGui Win32 + Vulkan backends, plus a small render-pass that
// targets the swapchain image directly. The pass uses loadOp=CLEAR so the
// caller's clear colour shows through wherever ImGui doesn't paint — which is
// the entire window in v1 because there's no 3D content yet.
#include "ImGuiContextRS.h"
#include "Core/Logger.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_vulkan.h"

namespace RS {

namespace {

void CheckVkResult(VkResult r) {
    if (r != VK_SUCCESS) {
        RS_LOG_ERROR("ImGui Vulkan backend reported VkResult=%d", static_cast<int>(r));
    }
}

bool CreateRenderPass(VkDevice device, VkFormat format, VkRenderPass& outPass) {
    // Phase 4+: the GridPass already cleared the swap image and left it in
    // COLOR_ATTACHMENT_OPTIMAL. We preserve its contents (loadOp=LOAD) so
    // ImGui paints on top of the grid.
    VkAttachmentDescription color{};
    color.format         = format;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Caller transitions UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL before the pass
    // and COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC after, so the pass itself
    // can keep both layouts equal.
    color.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    // External -> subpass dependency, color-attachment-output stage.
    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &color;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    return vkCreateRenderPass(device, &rpci, nullptr, &outPass) == VK_SUCCESS;
}

bool CreateDescriptorPool(VkDevice device, VkDescriptorPool& outPool) {
    // ImGui uses one combined-image-sampler per font atlas + per user texture.
    // The "all 1000" recipe from the ImGui examples is overkill for our needs
    // but lets users register textures freely without rebuilding the pool.
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
    };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 256;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = sizes;
    return vkCreateDescriptorPool(device, &dpci, nullptr, &outPool) == VK_SUCCESS;
}

bool CreateFramebuffers(VkDevice device, VkRenderPass pass,
                        const std::vector<VkImageView>& views,
                        VkExtent2D extent,
                        std::vector<VkFramebuffer>& outFbs) {
    outFbs.resize(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
        VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbci.renderPass      = pass;
        fbci.attachmentCount = 1;
        fbci.pAttachments    = &views[i];
        fbci.width           = extent.width;
        fbci.height          = extent.height;
        fbci.layers          = 1;
        if (vkCreateFramebuffer(device, &fbci, nullptr, &outFbs[i]) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

} // namespace

bool ImGuiContextRSInitialize(ImGuiContextRS& gui, const VulkanContext& ctx, HWND hwnd) {
    if (gui.Initialized) return true;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // We don't want ImGui writing imgui.ini until Phase 16; suppress it for now.
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();

    if (!CreateRenderPass(ctx.Device, ctx.SwapchainFormat, gui.RenderPass)) {
        RS_LOG_ERROR("ImGui: render pass creation failed");
        return false;
    }
    if (!CreateDescriptorPool(ctx.Device, gui.DescriptorPool)) {
        RS_LOG_ERROR("ImGui: descriptor pool creation failed");
        return false;
    }
    if (!CreateFramebuffers(ctx.Device, gui.RenderPass, ctx.SwapchainViews,
                            ctx.SwapchainExtent, gui.Framebuffers)) {
        RS_LOG_ERROR("ImGui: framebuffer creation failed");
        return false;
    }

    if (!ImGui_ImplWin32_Init(hwnd)) {
        RS_LOG_ERROR("ImGui_ImplWin32_Init failed");
        return false;
    }

    // Note: as of ImGui 2025/09/26, RenderPass/Subpass/MSAASamples migrated
    // off ImGui_ImplVulkan_InitInfo onto a nested PipelineInfoMain struct.
    ImGui_ImplVulkan_InitInfo init{};
    init.ApiVersion                       = VK_API_VERSION_1_2;
    init.Instance                         = ctx.Instance;
    init.PhysicalDevice                   = ctx.PhysicalDevice;
    init.Device                           = ctx.Device;
    init.QueueFamily                      = ctx.GraphicsQueueFamily;
    init.Queue                            = ctx.GraphicsQueue;
    init.DescriptorPool                   = gui.DescriptorPool;
    init.MinImageCount                    = static_cast<uint32_t>(ctx.SwapchainImages.size());
    init.ImageCount                       = static_cast<uint32_t>(ctx.SwapchainImages.size());
    init.PipelineInfoMain.RenderPass      = gui.RenderPass;
    init.PipelineInfoMain.Subpass         = 0;
    init.PipelineInfoMain.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    init.CheckVkResultFn                  = CheckVkResult;
    if (!ImGui_ImplVulkan_Init(&init)) {
        RS_LOG_ERROR("ImGui_ImplVulkan_Init failed");
        return false;
    }
    // Recent ImGui versions build the font atlas implicitly on first NewFrame
    // (CreateFontsTexture is internal); nothing extra to do here.

    gui.Initialized = true;
    RS_LOG_INFO("ImGui ready: %zu framebuffers, render-pass + descriptor-pool live",
                gui.Framebuffers.size());
    return true;
}

void ImGuiContextRSTerminate(ImGuiContextRS& gui, const VulkanContext& ctx) {
    if (!gui.Initialized) return;

    vkDeviceWaitIdle(ctx.Device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    for (VkFramebuffer fb : gui.Framebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    gui.Framebuffers.clear();
    if (gui.DescriptorPool) {
        vkDestroyDescriptorPool(ctx.Device, gui.DescriptorPool, nullptr);
        gui.DescriptorPool = VK_NULL_HANDLE;
    }
    if (gui.RenderPass) {
        vkDestroyRenderPass(ctx.Device, gui.RenderPass, nullptr);
        gui.RenderPass = VK_NULL_HANDLE;
    }
    gui.Initialized = false;
}

void ImGuiContextRSNewFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
}

void ImGuiContextRSRender(ImGuiContextRS& gui, const VulkanContext& ctx,
                          VkCommandBuffer cmd, uint32_t imageIndex) {
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();

    // loadOp=LOAD → no clear value needed (and Vulkan will warn if we pass one).
    VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass        = gui.RenderPass;
    rpbi.framebuffer       = gui.Framebuffers[imageIndex];
    rpbi.renderArea.offset = { 0, 0 };
    rpbi.renderArea.extent = ctx.SwapchainExtent;
    rpbi.clearValueCount   = 0;
    rpbi.pClearValues      = nullptr;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    if (drawData) {
        ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    }

    vkCmdEndRenderPass(cmd);
}

} // namespace RS
