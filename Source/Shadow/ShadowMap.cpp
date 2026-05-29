// Source/Shadow/ShadowMap.cpp — Phase 10
#include "Shadow/ShadowMap.h"
#include "Core/Logger.h"
#include "Scene/ObjLoader.h"   // ParsedVertex

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

#pragma pack(push, 16)
struct ShadowPushConstants {
    glm::mat4 LightViewProj;
    glm::mat4 Model;
};
#pragma pack(pop)
static_assert(sizeof(ShadowPushConstants) <= 256, "ShadowPushConstants budget");

int FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool ReadFile(const char* path, std::vector<char>& outBuf) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("ShadowMap: cannot open shader: %s", path); return false; }
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
        RS_LOG_ERROR("ShadowMap: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

bool CreateAtlas(ShadowMap& sm, const VulkanContext& ctx) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = sm.Format;
    ici.extent        = { ShadowMap::kCascadeSize, ShadowMap::kCascadeSize, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = ShadowMap::kCascadeCount;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &sm.AtlasImage) != VK_SUCCESS) {
        RS_LOG_ERROR("ShadowMap: vkCreateImage failed");
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx.Device, sm.AtlasImage, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) {
        RS_LOG_ERROR("ShadowMap: no DEVICE_LOCAL memory type");
        return false;
    }
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &sm.AtlasMemory) != VK_SUCCESS) {
        RS_LOG_ERROR("ShadowMap: vkAllocateMemory failed");
        return false;
    }
    vkBindImageMemory(ctx.Device, sm.AtlasImage, sm.AtlasMemory, 0);

    // Sampled view: 2D-array over all 4 layers.
    {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image            = sm.AtlasImage;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format           = sm.Format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, ShadowMap::kCascadeCount };
        if (vkCreateImageView(ctx.Device, &vci, nullptr, &sm.AtlasArrayView) != VK_SUCCESS) {
            RS_LOG_ERROR("ShadowMap: array view failed");
            return false;
        }
    }
    // Per-cascade single-layer 2D views (depth attachments).
    for (uint32_t i = 0; i < ShadowMap::kCascadeCount; ++i) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image            = sm.AtlasImage;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = sm.Format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, i, 1 };
        if (vkCreateImageView(ctx.Device, &vci, nullptr, &sm.CascadeLayerViews[i]) != VK_SUCCESS) {
            RS_LOG_ERROR("ShadowMap: layer view %u failed", i);
            return false;
        }
    }
    return true;
}

bool CreateRenderPass(ShadowMap& sm, VkDevice device) {
    VkAttachmentDescription depth{};
    depth.format         = sm.Format;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    // Wait for prior reads of the atlas (lighting compose of previous frame).
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &depth;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;
    return vkCreateRenderPass(device, &rpci, nullptr, &sm.RenderPass) == VK_SUCCESS;
}

bool CreatePipeline(ShadowMap& sm, VkDevice device, const char* shaderDir) {
    const std::string vertPath = std::string(shaderDir) + "/shadow_csm_vert.spv";
    VkShaderModule vert = LoadModule(device, vertPath.c_str());
    if (!vert) return false;

    VkPipelineShaderStageCreateInfo stages[1]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(ParsedVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrib{};
    attrib.location = 0;
    attrib.binding  = 0;
    attrib.format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrib.offset   = offsetof(ParsedVertex, Position);

    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions    = &attrib;

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{
        0.0f, 0.0f,
        static_cast<float>(ShadowMap::kCascadeSize),
        static_cast<float>(ShadowMap::kCascadeSize),
        0.0f, 1.0f
    };
    VkRect2D scissor{ { 0, 0 }, { ShadowMap::kCascadeSize, ShadowMap::kCascadeSize } };
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    // Front-face culling for shadow maps reduces self-shadow acne on the lit
    // surface (we draw back faces into the depth atlas). Combined with a small
    // depth bias this is the same trick most CSM impls use.
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode             = VK_POLYGON_MODE_FILL;
    rs.cullMode                = VK_CULL_MODE_FRONT_BIT;
    rs.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth               = 1.0f;
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = 1.25f;
    rs.depthBiasSlopeFactor    = 1.75f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    // No colour attachments — pure depth pass.
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(ShadowPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &sm.PipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, vert, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gpci.stageCount          = 1;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pDepthStencilState  = &ds;
    gpci.pColorBlendState    = &cb;
    gpci.layout              = sm.PipelineLayout;
    gpci.renderPass          = sm.RenderPass;
    gpci.subpass             = 0;

    const VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci,
                                                 nullptr, &sm.Pipeline);
    vkDestroyShaderModule(device, vert, nullptr);
    return r == VK_SUCCESS;
}

bool CreateFramebuffers(ShadowMap& sm, VkDevice device) {
    for (uint32_t i = 0; i < ShadowMap::kCascadeCount; ++i) {
        VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbci.renderPass      = sm.RenderPass;
        fbci.attachmentCount = 1;
        fbci.pAttachments    = &sm.CascadeLayerViews[i];
        fbci.width           = ShadowMap::kCascadeSize;
        fbci.height          = ShadowMap::kCascadeSize;
        fbci.layers          = 1;
        if (vkCreateFramebuffer(device, &fbci, nullptr, &sm.CascadeFramebuffers[i]) != VK_SUCCESS) {
            RS_LOG_ERROR("ShadowMap: framebuffer %u failed", i);
            return false;
        }
    }
    return true;
}

} // namespace

// -- public API ---------------------------------------------------------------

bool ShadowMapInitialize(ShadowMap& sm, const VulkanContext& ctx,
                         const char* shaderArtifactsDir) {
    if (sm.Initialized) return true;
    if (!CreateAtlas      (sm, ctx))                      return false;
    if (!CreateRenderPass (sm, ctx.Device))               return false;
    if (!CreatePipeline   (sm, ctx.Device, shaderArtifactsDir)) return false;
    if (!CreateFramebuffers(sm, ctx.Device))              return false;
    sm.Initialized = true;
    RS_LOG_INFO("ShadowMap ready: D32_SFLOAT %ux%ux%u",
                ShadowMap::kCascadeSize, ShadowMap::kCascadeSize,
                ShadowMap::kCascadeCount);
    return true;
}

void ShadowMapTerminate(ShadowMap& sm, const VulkanContext& ctx) {
    if (!sm.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);

    for (VkFramebuffer fb : sm.CascadeFramebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    sm.CascadeFramebuffers = {};

    if (sm.Pipeline)       vkDestroyPipeline      (ctx.Device, sm.Pipeline,       nullptr);
    if (sm.PipelineLayout) vkDestroyPipelineLayout(ctx.Device, sm.PipelineLayout, nullptr);
    if (sm.RenderPass)     vkDestroyRenderPass    (ctx.Device, sm.RenderPass,     nullptr);

    for (VkImageView v : sm.CascadeLayerViews) {
        if (v) vkDestroyImageView(ctx.Device, v, nullptr);
    }
    sm.CascadeLayerViews = {};
    if (sm.AtlasArrayView) vkDestroyImageView(ctx.Device, sm.AtlasArrayView, nullptr);
    if (sm.AtlasImage)     vkDestroyImage    (ctx.Device, sm.AtlasImage,     nullptr);
    if (sm.AtlasMemory)    vkFreeMemory      (ctx.Device, sm.AtlasMemory,    nullptr);

    sm.Pipeline       = VK_NULL_HANDLE;
    sm.PipelineLayout = VK_NULL_HANDLE;
    sm.RenderPass     = VK_NULL_HANDLE;
    sm.AtlasArrayView = VK_NULL_HANDLE;
    sm.AtlasImage     = VK_NULL_HANDLE;
    sm.AtlasMemory    = VK_NULL_HANDLE;
    sm.Initialized    = false;
}

// -- cascade fit ---------------------------------------------------------------

void ShadowMapBuildCascades(ShadowMap& sm,
                            const CameraView& camera,
                            const glm::vec3& sunDirectionToward) {
    // Per-cascade fraction of (near, far) — practical-CSM with hand-picked
    // splits favouring near range (matches the plan's 0.1/0.25/0.5/1.0).
    const float splits[ShadowMap::kCascadeCount] = { 0.1f, 0.25f, 0.5f, 1.0f };

    const float nearZ = camera.NearClip;
    const float farZ  = camera.FarClip;
    const float range = farZ - nearZ;

    const glm::mat4 invViewProj = glm::inverse(camera.Projection * camera.View);

    // The light looks DOWN the sun-direction-toward (i.e. from the sun to the
    // scene). Build an orthonormal basis up = world-Y unless degenerate.
    const glm::vec3 lightDir = glm::normalize(-sunDirectionToward);   // direction of light travel
    glm::vec3 upRef = (std::abs(lightDir.y) > 0.95f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

    float prevSplit = 0.0f;
    for (uint32_t c = 0; c < ShadowMap::kCascadeCount; ++c) {
        const float thisSplit = splits[c];

        // View-frustum corners in NDC. Vulkan depth [0,1]; near=0 far=1.
        const float nNDC = prevSplit;   // mapped to NDC z via lerp; see below
        const float fNDC = thisSplit;

        // Slice frustum corners: take the full near→far frustum in world space
        // and lerp between near and far face for this slice's bounds.
        glm::vec3 corners[8] = {
            { -1.0f, -1.0f, 0.0f }, {  1.0f, -1.0f, 0.0f },
            { -1.0f,  1.0f, 0.0f }, {  1.0f,  1.0f, 0.0f },
            { -1.0f, -1.0f, 1.0f }, {  1.0f, -1.0f, 1.0f },
            { -1.0f,  1.0f, 1.0f }, {  1.0f,  1.0f, 1.0f },
        };
        for (int i = 0; i < 8; ++i) {
            glm::vec4 ws = invViewProj * glm::vec4(corners[i], 1.0f);
            corners[i] = glm::vec3(ws) / ws.w;
        }
        // Lerp near→far edges to carve this cascade's slice.
        glm::vec3 sliceCorners[8];
        for (int i = 0; i < 4; ++i) {
            const glm::vec3 dir = corners[i + 4] - corners[i];
            sliceCorners[i    ] = corners[i] + dir * nNDC;
            sliceCorners[i + 4] = corners[i] + dir * fNDC;
        }

        // Sphere fit — same radius regardless of camera rotation gives stable
        // sample positions.
        glm::vec3 centerWS(0.0f);
        for (int i = 0; i < 8; ++i) centerWS += sliceCorners[i];
        centerWS *= (1.0f / 8.0f);

        float radius = 0.0f;
        for (int i = 0; i < 8; ++i) {
            radius = std::max(radius, glm::length(sliceCorners[i] - centerWS));
        }
        // Snap radius up to a stable value so it doesn't oscillate frame-to-
        // frame for any given camera pose.
        radius = std::ceil(radius * 16.0f) / 16.0f;
        sm.CascadeSphereRadius[c] = radius;

        // Build the light view at the sphere centre + light-direction offset.
        const float kBackoff = radius * 1.5f;   // pull back so geometry behind the centre still casts
        const glm::vec3 lightEye = centerWS - lightDir * kBackoff;
        glm::mat4 lightView = glm::lookAtRH(lightEye, centerWS, upRef);

        // Texel-snap the center to kill cascade shimmer when the camera pans.
        const float texelsPerWorld = static_cast<float>(ShadowMap::kCascadeSize) / (2.0f * radius);
        glm::vec4 originLS = lightView * glm::vec4(centerWS, 1.0f);
        originLS.x = std::floor(originLS.x * texelsPerWorld) / texelsPerWorld;
        originLS.y = std::floor(originLS.y * texelsPerWorld) / texelsPerWorld;
        // Re-derive the light eye from the snapped origin.
        const glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightView) * originLS);
        const glm::vec3 snappedEye    = snappedCenter - lightDir * kBackoff;
        lightView = glm::lookAtRH(snappedEye, snappedCenter, upRef);

        // Tight ortho — depth covers [eye → centre + radius]. Padding above
        // (negative side) catches casters behind the centre.
        const float zNear = 0.0f;
        const float zFar  = kBackoff + radius * 2.0f;
        glm::mat4 lightProj = glm::orthoRH_ZO(-radius, radius, -radius, radius, zNear, zFar);
        // Vulkan: flip Y for the depth-only pass (matches gbuffer pipeline).
        lightProj[1][1] *= -1.0f;

        sm.CascadeViewProj    [c] = lightProj * lightView;
        sm.CascadeSplitDistance[c] = nearZ + range * thisSplit;
        prevSplit = thisSplit;
    }
}

// -- record --------------------------------------------------------------------

void ShadowMapRecord(ShadowMap& sm, const VulkanContext& /*ctx*/,
                     VkCommandBuffer cmd,
                     const MeshRegistry& meshes,
                     const InstanceRegistry& instances) {
    VkClearValue clear{};
    clear.depthStencil = { 1.0f, 0 };

    for (uint32_t c = 0; c < ShadowMap::kCascadeCount; ++c) {
        VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rpbi.renderPass        = sm.RenderPass;
        rpbi.framebuffer       = sm.CascadeFramebuffers[c];
        rpbi.renderArea.offset = { 0, 0 };
        rpbi.renderArea.extent = { ShadowMap::kCascadeSize, ShadowMap::kCascadeSize };
        rpbi.clearValueCount   = 1;
        rpbi.pClearValues      = &clear;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sm.Pipeline);

        const glm::mat4& lightVP = sm.CascadeViewProj[c];
        instances.ForEach([&](InstanceHandle /*handle*/, const GpuInstance& inst) {
            const GpuMesh* mesh = meshes.Get(inst.Mesh);
            if (!mesh) return;

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->VertexBuffer, &offset);
            vkCmdBindIndexBuffer  (cmd, mesh->IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            ShadowPushConstants pc{};
            pc.LightViewProj = lightVP;
            pc.Model         = inst.Transform;
            vkCmdPushConstants(cmd, sm.PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(pc), &pc);

            // One draw covering every submesh — the depth-only pass doesn't
            // care about per-submesh material binding, so collapse to a single
            // indexed draw over the whole mesh.
            uint32_t indexCount = 0;
            for (const GpuSubmesh& sub : mesh->Submeshes) indexCount += sub.IndexCount;
            if (indexCount == 0) return;
            vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
        });

        vkCmdEndRenderPass(cmd);
    }
}

} // namespace RS
