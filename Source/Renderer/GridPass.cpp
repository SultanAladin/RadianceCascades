// Source/Renderer/GridPass.cpp — Phase 4
// Tiny full-screen pass that paints the ground grid. Reads SPIR-V from disk on
// init (mirroring SolidArc's pattern) so shader recompiles flow in without an
// app rebuild. Push-constant layout mirrors the GLSL block in grid.frag.
#include "Renderer/GridPass.h"
#include "Core/Logger.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

#pragma pack(push, 16)
struct GridPushConstants {
    glm::mat4 InvViewProj;
    glm::vec4 MajorColorAndSize;
    glm::vec4 MinorColorAndInfinite;
    glm::vec4 SpacingAndThickness;
    glm::vec4 FadeAndOpacity;
    glm::vec4 SkyAndGround;
    glm::vec4 GroundColor;
    // Phase 8 polish: checker floor.
    glm::vec4 CheckerLightAndSpacing;   // rgb = light tile, a = spacing (metres)
    glm::vec4 CheckerDarkAndStrength;   // rgb = dark tile,  a = strength [0,1]
};
#pragma pack(pop)
static_assert(sizeof(GridPushConstants) == 64 + 16 * 8,
              "GridPushConstants must stay tightly packed");
static_assert(sizeof(GridPushConstants) <= 256,
              "GridPushConstants must fit the 256-byte push budget");

bool ReadFile(const char* path, std::vector<char>& outBuf) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        RS_LOG_ERROR("GridPass: cannot open shader: %s", path);
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0) { std::fclose(f); return false; }
    outBuf.resize(static_cast<size_t>(len));
    const size_t got = std::fread(outBuf.data(), 1, outBuf.size(), f);
    std::fclose(f);
    return got == outBuf.size();
}

VkShaderModule LoadModule(VkDevice device, const char* path) {
    std::vector<char> buf;
    if (!ReadFile(path, buf)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = buf.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) {
        RS_LOG_ERROR("GridPass: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

bool CreateRenderPass(VkDevice device, VkFormat format, VkRenderPass& outPass) {
    // GridPass owns the swapchain clear: loadOp=CLEAR, layout transitions are
    // already done outside the pass (BeginFrame -> COLOR_ATTACHMENT_OPTIMAL),
    // so the pass's initial/final layouts both stay COLOR_ATTACHMENT_OPTIMAL,
    // and the ImGui pass that follows uses loadOp=LOAD against the same layout.
    VkAttachmentDescription color{};
    color.format         = format;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

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

bool CreatePipeline(VkDevice device, VkRenderPass pass, VkExtent2D extent,
                    VkDescriptorSetLayout setLayout,
                    const char* shaderDir,
                    VkPipelineLayout& outLayout, VkPipeline& outPipe) {
    const std::string vertPath = std::string(shaderDir) + "/grid_vert.spv";
    const std::string fragPath = std::string(shaderDir) + "/grid_frag.spv";

    VkShaderModule vert = LoadModule(device, vertPath.c_str());
    VkShaderModule frag = LoadModule(device, fragPath.c_str());
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(device, vert, nullptr);
        if (frag) vkDestroyShaderModule(device, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    // no vertex buffers; gl_VertexIndex drives the fullscreen triangle.

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{ 0, 0, static_cast<float>(extent.width),
                               static_cast<float>(extent.height), 0.0f, 1.0f };
    VkRect2D   scissor { { 0, 0 }, extent };
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    att.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments    = &att;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(GridPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &outLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pColorBlendState    = &cb;
    gpci.layout              = outLayout;
    gpci.renderPass          = pass;
    gpci.subpass             = 0;

    const VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci,
                                                 nullptr, &outPipe);
    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
    return r == VK_SUCCESS;
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

bool GridPassInitialize(GridPass& gp, const VulkanContext& ctx,
                        const SkyAtmosphere& sky,
                        const char* shaderArtifactsDir) {
    if (gp.Initialized) return true;
    if (!CreateRenderPass(ctx.Device, ctx.SwapchainFormat, gp.RenderPass)) {
        RS_LOG_ERROR("GridPass: render-pass creation failed");
        return false;
    }

    // Sky cubemap binding (set 0, binding 0).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 1;
        lci.pBindings    = &b;
        if (vkCreateDescriptorSetLayout(ctx.Device, &lci, nullptr, &gp.SetLayout) != VK_SUCCESS) {
            RS_LOG_ERROR("GridPass: descriptor set layout creation failed");
            return false;
        }
    }

    if (!CreatePipeline(ctx.Device, gp.RenderPass, ctx.SwapchainExtent, gp.SetLayout,
                        shaderArtifactsDir, gp.PipelineLayout, gp.Pipeline)) {
        RS_LOG_ERROR("GridPass: pipeline creation failed");
        return false;
    }
    if (!CreateFramebuffers(ctx.Device, gp.RenderPass, ctx.SwapchainViews,
                            ctx.SwapchainExtent, gp.Framebuffers)) {
        RS_LOG_ERROR("GridPass: framebuffer creation failed");
        return false;
    }

    // Single descriptor pool + set referencing the sky cubemap; the view is
    // stable across the SkyAtmosphere's lifetime (the cube image is recreated
    // only at re-init, not re-bake).
    {
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 1;
        VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pci.maxSets       = 1;
        pci.poolSizeCount = 1;
        pci.pPoolSizes    = &ps;
        if (vkCreateDescriptorPool(ctx.Device, &pci, nullptr, &gp.DescriptorPool) != VK_SUCCESS) {
            return false;
        }
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool     = gp.DescriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &gp.SetLayout;
        if (vkAllocateDescriptorSets(ctx.Device, &ai, &gp.SkySet) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorImageInfo dii{};
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        dii.imageView   = sky.Sky.View;
        dii.sampler     = sky.SamplerCubeLinear;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = gp.SkySet;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &dii;
        vkUpdateDescriptorSets(ctx.Device, 1, &w, 0, nullptr);
    }

    gp.Initialized = true;
    RS_LOG_INFO("GridPass ready: %zu framebuffers", gp.Framebuffers.size());
    return true;
}

void GridPassTerminate(GridPass& gp, const VulkanContext& ctx) {
    if (!gp.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (VkFramebuffer fb : gp.Framebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    gp.Framebuffers.clear();
    if (gp.DescriptorPool) vkDestroyDescriptorPool     (ctx.Device, gp.DescriptorPool, nullptr);
    if (gp.Pipeline)       vkDestroyPipeline           (ctx.Device, gp.Pipeline,       nullptr);
    if (gp.PipelineLayout) vkDestroyPipelineLayout     (ctx.Device, gp.PipelineLayout, nullptr);
    if (gp.SetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, gp.SetLayout,      nullptr);
    if (gp.RenderPass)     vkDestroyRenderPass         (ctx.Device, gp.RenderPass,     nullptr);
    gp.DescriptorPool = VK_NULL_HANDLE;
    gp.Pipeline       = VK_NULL_HANDLE;
    gp.PipelineLayout = VK_NULL_HANDLE;
    gp.SetLayout      = VK_NULL_HANDLE;
    gp.RenderPass     = VK_NULL_HANDLE;
    gp.SkySet         = VK_NULL_HANDLE;
    gp.Initialized    = false;
}

void GridPassRecord(GridPass& gp, const VulkanContext& ctx,
                    VkCommandBuffer cmd, uint32_t imageIndex,
                    const CameraView& camera,
                    const GridSettings& s) {
    VkClearValue clear{};
    clear.color.float32[0] = s.SkyColor[0];
    clear.color.float32[1] = s.SkyColor[1];
    clear.color.float32[2] = s.SkyColor[2];
    clear.color.float32[3] = 1.0f;

    VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass        = gp.RenderPass;
    rpbi.framebuffer       = gp.Framebuffers[imageIndex];
    rpbi.renderArea.offset = { 0, 0 };
    rpbi.renderArea.extent = ctx.SwapchainExtent;
    rpbi.clearValueCount   = 1;
    rpbi.pClearValues      = &clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gp.Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gp.PipelineLayout,
                            0, 1, &gp.SkySet, 0, nullptr);

    GridPushConstants pc{};
    pc.InvViewProj          = glm::inverse(camera.Projection * camera.View);
    pc.MajorColorAndSize    = glm::vec4(s.MajorColor[0], s.MajorColor[1],
                                        s.MajorColor[2], s.Extent);
    pc.MinorColorAndInfinite= glm::vec4(s.MinorColor[0], s.MinorColor[1],
                                        s.MinorColor[2], s.Infinite ? 1.0f : 0.0f);
    pc.SpacingAndThickness  = glm::vec4(s.MajorSpacing, s.MinorSpacing,
                                        s.MajorThickness, s.MinorThickness);
    const float signedOpacity = s.AxisHighlight ?  s.Opacity : -s.Opacity;
    pc.FadeAndOpacity       = glm::vec4(s.FadeDistance, s.FalloffStart,
                                        s.FalloffCurve, signedOpacity);
    pc.SkyAndGround         = glm::vec4(s.SkyColor[0], s.SkyColor[1],
                                        s.SkyColor[2], 0.0f);
    pc.GroundColor          = glm::vec4(s.GroundColor[0], s.GroundColor[1],
                                        s.GroundColor[2], 0.0f);
    pc.CheckerLightAndSpacing = glm::vec4(s.CheckerLight[0], s.CheckerLight[1],
                                          s.CheckerLight[2], s.CheckerSpacing);
    pc.CheckerDarkAndStrength = glm::vec4(s.CheckerDark[0],  s.CheckerDark[1],
                                          s.CheckerDark[2],  s.CheckerStrength);
    vkCmdPushConstants(cmd, gp.PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

} // namespace RS
