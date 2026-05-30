// Source/Renderer/GBuffer.cpp — Phase 6
// 5-target GBuffer render-pass + pipeline. Each attachment's finalLayout is
// SHADER_READ_ONLY_OPTIMAL so the preview blit (and the upcoming lighting
// compose) can sample without an extra image barrier.
#include "Renderer/GBuffer.h"
#include "Core/Logger.h"
#include "Scene/ObjLoader.h"   // ParsedVertex

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

#pragma pack(push, 16)
struct GBufferPushConstants {
    glm::mat4 MVP;
    glm::mat4 Model;
    glm::vec4 BaseColor;             // rgb = albedo, a = unused
    glm::vec4 RoughMetalF0Emissive;  // r=roughness g=metallic b=F0 a=emissive scale
    glm::vec4 EmissiveColor;         // rgb = emissive colour, a = unused
    glm::uvec4 IdentityIds;          // x = instanceId, y = submeshId
    glm::vec4 FloorParams;           // x = checker spacing (m), y = strength,
                                     // z = flag (>=0.5 → floor), w = dark tint scale
};
#pragma pack(pop)
static_assert(sizeof(GBufferPushConstants) <= 256,
              "GBufferPushConstants must fit in the 256-byte push-constant budget");

bool ReadFile(const char* path, std::vector<char>& outBuf) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("GBuffer: cannot open shader: %s", path); return false; }
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
        RS_LOG_ERROR("GBuffer: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

bool CreateRenderPass(VkDevice device, VkRenderPass& outPass) {
    // 5 colour + 1 depth. Each colour attachment clears to black at start, the
    // depth attachment clears to 1.0. After the pass everything ends up in
    // SHADER_READ_ONLY_OPTIMAL so downstream samplers don't need barriers.
    VkAttachmentDescription atts[6]{};

    auto initColor = [&](uint32_t i, VkFormat fmt) {
        atts[i].format         = fmt;
        atts[i].samples        = VK_SAMPLE_COUNT_1_BIT;
        atts[i].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[i].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        atts[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[i].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[i].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    };

    initColor(0, OffscreenFormatAlbedo  ());
    initColor(1, OffscreenFormatNormal  ());
    initColor(2, OffscreenFormatRMF     ());
    initColor(3, OffscreenFormatEmissive());
    initColor(4, OffscreenFormatIdentity());

    atts[5].format         = OffscreenFormatDepth();
    atts[5].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[5].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[5].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;   // sampled by preview's depth view
    atts[5].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[5].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[5].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[5].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRefs[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        colorRefs[i].attachment = i;
        colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    VkAttachmentReference depthRef{ 5, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 5;
    subpass.pColorAttachments       = colorRefs;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 6;
    rpci.pAttachments    = atts;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;
    return vkCreateRenderPass(device, &rpci, nullptr, &outPass) == VK_SUCCESS;
}

bool CreatePipeline(VkDevice device, VkRenderPass pass, VkExtent2D extent,
                    const char* shaderDir,
                    VkPipelineLayout& outLayout, VkPipeline& outPipe) {
    const std::string vertPath = std::string(shaderDir) + "/gbuffer_vert.spv";
    const std::string fragPath = std::string(shaderDir) + "/gbuffer_frag.spv";
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

    VkVertexInputBindingDescription   binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(ParsedVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribs[2]{};
    attribs[0].location = 0; attribs[0].binding = 0;
    attribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attribs[0].offset   = offsetof(ParsedVertex, Position);
    attribs[1].location = 1; attribs[1].binding = 0;
    attribs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attribs[1].offset   = offsetof(ParsedVertex, Normal);

    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions    = attribs;

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
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState atts[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        atts[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        atts[i].blendEnable    = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 5;
    cb.pAttachments    = atts;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(GBufferPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
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
    gpci.pDepthStencilState  = &ds;
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
                        const OffscreenTargets& targets,
                        std::array<VkFramebuffer, VulkanContext::kFramesInFlight>& outFbs) {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        const OffscreenFrame& f = targets.Frames[i];
        VkImageView att[6] = {
            f.Albedo.View,
            f.Normal.View,
            f.RoughMetalF0.View,
            f.Emissive.View,
            f.Identity.View,
            f.Depth.View,
        };
        VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbci.renderPass      = pass;
        fbci.attachmentCount = 6;
        fbci.pAttachments    = att;
        fbci.width           = targets.Extent.width;
        fbci.height          = targets.Extent.height;
        fbci.layers          = 1;
        if (vkCreateFramebuffer(device, &fbci, nullptr, &outFbs[i]) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

} // namespace

bool GBufferPassInitialize(GBufferPass& gp, const VulkanContext& ctx,
                           const OffscreenTargets& targets,
                           const char* shaderArtifactsDir) {
    if (gp.Initialized) return true;
    if (!CreateRenderPass(ctx.Device, gp.RenderPass)) {
        RS_LOG_ERROR("GBufferPass: render pass failed");
        return false;
    }
    if (!CreatePipeline(ctx.Device, gp.RenderPass, targets.Extent,
                        shaderArtifactsDir, gp.PipelineLayout, gp.Pipeline)) {
        RS_LOG_ERROR("GBufferPass: pipeline failed");
        return false;
    }
    if (!CreateFramebuffers(ctx.Device, gp.RenderPass, targets, gp.Framebuffers)) {
        RS_LOG_ERROR("GBufferPass: framebuffers failed");
        return false;
    }
    gp.Initialized = true;
    RS_LOG_INFO("GBufferPass ready: %u framebuffers, %ux%u",
                VulkanContext::kFramesInFlight,
                targets.Extent.width, targets.Extent.height);
    return true;
}

void GBufferPassTerminate(GBufferPass& gp, const VulkanContext& ctx) {
    if (!gp.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (VkFramebuffer fb : gp.Framebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    gp.Framebuffers = {};
    if (gp.Pipeline)       vkDestroyPipeline      (ctx.Device, gp.Pipeline,       nullptr);
    if (gp.PipelineLayout) vkDestroyPipelineLayout(ctx.Device, gp.PipelineLayout, nullptr);
    if (gp.RenderPass)     vkDestroyRenderPass    (ctx.Device, gp.RenderPass,     nullptr);
    gp.Pipeline       = VK_NULL_HANDLE;
    gp.PipelineLayout = VK_NULL_HANDLE;
    gp.RenderPass     = VK_NULL_HANDLE;
    gp.Initialized    = false;
}

GBufferMaterial GBufferMaterialFromPbr(const PbrMaterial& m) {
    GBufferMaterial g{};
    g.BaseColor = m.AlbedoFlat;
    g.Roughness = m.RoughnessFlat;
    g.Metallic  = m.MetallicFlat;
    g.F0        = m.F0Flat.x;            // v1 single-channel F0
    g.Emissive  = m.EmissiveFlat;
    return g;
}

void GBufferPassRecord(GBufferPass& gp, const VulkanContext& ctx,
                       VkCommandBuffer cmd, uint32_t frameSlot,
                       const CameraView& camera,
                       const MeshRegistry& meshes,
                       const InstanceRegistry& instances,
                       MaterialRegistry& materials,
                       const GBufferMaterial& fallback,
                       const GBufferFloorConfig& floor) {
    VkClearValue clears[6]{};
    // Albedo / normal / RMF / emissive / identity all clear to 0.
    clears[5].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass        = gp.RenderPass;
    rpbi.framebuffer       = gp.Framebuffers[frameSlot];
    rpbi.renderArea.offset = { 0, 0 };
    rpbi.renderArea.extent = ctx.SwapchainExtent;
    rpbi.clearValueCount   = 6;
    rpbi.pClearValues      = clears;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gp.Pipeline);

    const glm::mat4 viewProj      = camera.Projection * camera.View;
    const uint32_t  matCount      = static_cast<uint32_t>(materials.Count());
    instances.ForEach([&](InstanceHandle handle, const GpuInstance& inst) {
        const GpuMesh* mesh = meshes.Get(inst.Mesh);
        if (!mesh) return;

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->VertexBuffer, &offset);
        vkCmdBindIndexBuffer  (cmd, mesh->IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // One draw per submesh — push constants carry the resolved material
        // and identity ids (instance, submesh) for picking.
        const uint32_t subCount = static_cast<uint32_t>(mesh->Submeshes.size());
        for (uint32_t s = 0; s < subCount; ++s) {
            const GpuSubmesh& sub = mesh->Submeshes[s];
            if (sub.IndexCount == 0) continue;

            GBufferMaterial gm = fallback;
            if (s < inst.MaterialBindings.size()) {
                const MaterialHandle h = inst.MaterialBindings[s];
                if (h < matCount) {
                    gm = GBufferMaterialFromPbr(materials.Get(h));
                }
            }

            GBufferPushConstants pc{};
            pc.Model                = inst.Transform;
            pc.MVP                  = viewProj * inst.Transform;
            pc.BaseColor            = glm::vec4(gm.BaseColor, 1.0f);
            pc.RoughMetalF0Emissive = glm::vec4(gm.Roughness, gm.Metallic,
                                                gm.F0, 1.0f);
            pc.EmissiveColor        = glm::vec4(gm.Emissive, 0.0f);
            pc.IdentityIds          = glm::uvec4(static_cast<uint32_t>(handle), s, 0u, 0u);

            const bool isFloor = (floor.FloorInstance != 0 &&
                                  handle == floor.FloorInstance);
            pc.FloorParams = isFloor
                ? glm::vec4(floor.CheckerSpacing, floor.CheckerStrength,
                            1.0f, floor.DarkTintScale)
                : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            vkCmdPushConstants(cmd, gp.PipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            vkCmdDrawIndexed(cmd, sub.IndexCount, 1, sub.FirstIndex, 0, 0);
        }
    });

    vkCmdEndRenderPass(cmd);
}

} // namespace RS
