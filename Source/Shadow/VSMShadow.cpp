// Source/Shadow/VSMShadow.cpp — Phase 11
#include "Shadow/VSMShadow.h"
#include "Renderer/FrameContext.h"
#include "Scene/SceneInternal.h"
#include "Scene/ObjLoader.h"     // ParsedVertex
#include "Core/Logger.h"
#include "imgui.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

#pragma pack(push, 16)
struct VsmPushConstants {
    glm::mat4 LightViewProj;
    glm::mat4 Model;
};
#pragma pack(pop)
static_assert(sizeof(VsmPushConstants) <= 256, "VsmPushConstants budget");

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
    if (!f) { RS_LOG_ERROR("VSMShadow: cannot open shader: %s", path); return false; }
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
        RS_LOG_ERROR("VSMShadow: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

} // namespace

VSMShadow::VSMShadow()  = default;
VSMShadow::~VSMShadow() = default;

void VSMShadow::Initialize(const VulkanContext& ctx,
                           VkFormat /*depthFormat*/,
                           VkExtent2D /*shadowMapResolution*/) {
    m_Ctx = &ctx;

    if (!CreateAtlas())            { RS_LOG_ERROR("VSMShadow: atlas failed");        return; }
    if (!CreateTransientDepth())   { RS_LOG_ERROR("VSMShadow: depth failed");        return; }
    if (!CreateRenderPass())       { RS_LOG_ERROR("VSMShadow: render pass failed");  return; }
    if (!CreatePipeline("Artifacts/Shaders"))
                                    { RS_LOG_ERROR("VSMShadow: pipeline failed");    return; }
    if (!CreateFramebuffers())     { RS_LOG_ERROR("VSMShadow: framebuffers failed"); return; }
    if (!CreateSampler())          { RS_LOG_ERROR("VSMShadow: sampler failed");      return; }
    if (!CreateSetLayout())        { RS_LOG_ERROR("VSMShadow: set layout failed");   return; }
    if (!CreateAlgoUbos())         { RS_LOG_ERROR("VSMShadow: UBOs failed");         return; }
    if (!CreateDescriptorSets())   { RS_LOG_ERROR("VSMShadow: sets failed");         return; }

    RS_LOG_INFO("VSMShadow ready: %u cascades @ %u^2 RG16F (Chebyshev + LBR)",
                kCascadeCount, kCascadeSize);
}

void VSMShadow::Terminate(const VulkanContext& ctx) {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(ctx.Device);

    if (m_DescriptorPool)    vkDestroyDescriptorPool     (ctx.Device, m_DescriptorPool,    nullptr);
    if (m_LightingSetLayout) vkDestroyDescriptorSetLayout(ctx.Device, m_LightingSetLayout, nullptr);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (m_AlgoUboMapped[i]) {
            vkUnmapMemory(ctx.Device, m_AlgoUboMemory[i]);
            m_AlgoUboMapped[i] = nullptr;
        }
        if (m_AlgoUboBuffers[i]) vkDestroyBuffer(ctx.Device, m_AlgoUboBuffers[i], nullptr);
        if (m_AlgoUboMemory [i]) vkFreeMemory   (ctx.Device, m_AlgoUboMemory [i], nullptr);
    }
    m_AlgoUboBuffers = {};
    m_AlgoUboMemory  = {};

    if (m_LinearSampler) vkDestroySampler(ctx.Device, m_LinearSampler, nullptr);

    for (VkFramebuffer fb : m_CascadeFramebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    m_CascadeFramebuffers = {};

    if (m_Pipeline)       vkDestroyPipeline      (ctx.Device, m_Pipeline,       nullptr);
    if (m_PipelineLayout) vkDestroyPipelineLayout(ctx.Device, m_PipelineLayout, nullptr);
    if (m_RenderPass)     vkDestroyRenderPass    (ctx.Device, m_RenderPass,     nullptr);

    for (VkImageView v : m_CascadeColorViews) {
        if (v) vkDestroyImageView(ctx.Device, v, nullptr);
    }
    m_CascadeColorViews = {};
    if (m_AtlasArrayView) vkDestroyImageView(ctx.Device, m_AtlasArrayView, nullptr);
    if (m_AtlasImage)     vkDestroyImage    (ctx.Device, m_AtlasImage,     nullptr);
    if (m_AtlasMemory)    vkFreeMemory      (ctx.Device, m_AtlasMemory,    nullptr);

    if (m_TransientDepthView)   vkDestroyImageView(ctx.Device, m_TransientDepthView,   nullptr);
    if (m_TransientDepthImage)  vkDestroyImage    (ctx.Device, m_TransientDepthImage,  nullptr);
    if (m_TransientDepthMemory) vkFreeMemory      (ctx.Device, m_TransientDepthMemory, nullptr);

    m_Pipeline             = VK_NULL_HANDLE;
    m_PipelineLayout       = VK_NULL_HANDLE;
    m_RenderPass           = VK_NULL_HANDLE;
    m_AtlasArrayView       = VK_NULL_HANDLE;
    m_TransientDepthView   = VK_NULL_HANDLE;
    m_AtlasImage           = VK_NULL_HANDLE;
    m_TransientDepthImage  = VK_NULL_HANDLE;
    m_AtlasMemory          = VK_NULL_HANDLE;
    m_TransientDepthMemory = VK_NULL_HANDLE;
    m_DescriptorPool       = VK_NULL_HANDLE;
    m_LightingSetLayout    = VK_NULL_HANDLE;
    m_LinearSampler        = VK_NULL_HANDLE;
    m_FrameSets            = {};
    m_Ctx                  = nullptr;
}

void VSMShadow::DrawImGuiParams() {
    ImGui::Checkbox  ("Shadows enabled", &m_Params.Enabled);
    ImGui::SliderFloat("Min variance",         &m_Params.MinVariance,         0.0f, 1e-3f, "%.6f");
    ImGui::SliderFloat("Light-bleed reduction",&m_Params.LightBleedReduction, 0.0f, 0.95f);
    ImGui::SliderFloat("Depth bias",  &m_Params.DepthBias,  0.0f, 0.01f, "%.5f");
    ImGui::SliderFloat("Normal bias", &m_Params.NormalBias, 0.0f, 0.1f,  "%.4f");
    ImGui::Separator();
    ImGui::TextDisabled("Moment atlas: 4 x %u^2 RG16F", kCascadeSize);
    ImGui::TextDisabled("Linear filter gives a free pre-blur; no separable pass.");
}

void VSMShadow::RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) {
    if (!m_Ctx) return;
    if (!m_Params.Enabled) {
        UpdateAlgoUbo(frame.FrameSlot);
        return;
    }
    BuildCascades(frame);
    UpdateAlgoUbo(frame.FrameSlot);
    if (frame.ScenePtr == nullptr) return;
    RecordMomentPass(cmd, frame);
}

void VSMShadow::BindLightingDescriptorSet(VkCommandBuffer cmd,
                                          VkPipelineLayout lightingLayout,
                                          uint32_t frameSlot,
                                          uint32_t set) {
    if (!m_FrameSets[frameSlot]) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lightingLayout, set, 1, &m_FrameSets[frameSlot],
                            0, nullptr);
}

// ---- cascade fit (mirror of ShadowMapBuildCascades, kept local so VSM owns its math) ----

void VSMShadow::BuildCascades(const FrameContext& frame) {
    const CameraView& camera = frame.Cam;
    const glm::vec3&  sunDirectionToward = frame.SunDirection;

    const float splits[kCascadeCount] = { 0.1f, 0.25f, 0.5f, 1.0f };

    const float nearZ = camera.NearClip;
    const float farZ  = camera.FarClip;
    const float range = farZ - nearZ;

    const glm::mat4 invViewProj = glm::inverse(camera.Projection * camera.View);

    const glm::vec3 lightDir = glm::normalize(-sunDirectionToward);
    glm::vec3 upRef = (std::abs(lightDir.y) > 0.95f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

    float prevSplit = 0.0f;
    for (uint32_t c = 0; c < kCascadeCount; ++c) {
        const float thisSplit = splits[c];

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
        glm::vec3 sliceCorners[8];
        for (int i = 0; i < 4; ++i) {
            const glm::vec3 dir = corners[i + 4] - corners[i];
            sliceCorners[i    ] = corners[i] + dir * prevSplit;
            sliceCorners[i + 4] = corners[i] + dir * thisSplit;
        }
        glm::vec3 centerWS(0.0f);
        for (int i = 0; i < 8; ++i) centerWS += sliceCorners[i];
        centerWS *= (1.0f / 8.0f);

        float radius = 0.0f;
        for (int i = 0; i < 8; ++i) {
            radius = std::max(radius, glm::length(sliceCorners[i] - centerWS));
        }
        radius = std::ceil(radius * 16.0f) / 16.0f;

        const float kBackoff = radius * 1.5f;
        const glm::vec3 lightEye = centerWS - lightDir * kBackoff;
        glm::mat4 lightView = glm::lookAtRH(lightEye, centerWS, upRef);

        const float texelsPerWorld = static_cast<float>(kCascadeSize) / (2.0f * radius);
        glm::vec4 originLS = lightView * glm::vec4(centerWS, 1.0f);
        originLS.x = std::floor(originLS.x * texelsPerWorld) / texelsPerWorld;
        originLS.y = std::floor(originLS.y * texelsPerWorld) / texelsPerWorld;
        const glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightView) * originLS);
        const glm::vec3 snappedEye    = snappedCenter - lightDir * kBackoff;
        lightView = glm::lookAtRH(snappedEye, snappedCenter, upRef);

        const float zNear = 0.0f;
        const float zFar  = kBackoff + radius * 2.0f;
        glm::mat4 lightProj = glm::orthoRH_ZO(-radius, radius, -radius, radius, zNear, zFar);
        lightProj[1][1] *= -1.0f;

        m_CascadeViewProj    [c] = lightProj * lightView;
        m_CascadeSplitDistance[c] = nearZ + range * thisSplit;
        prevSplit = thisSplit;
    }
}

void VSMShadow::RecordMomentPass(VkCommandBuffer cmd, const FrameContext& frame) {
    const MeshRegistry&     meshes    = SceneMeshes   (*frame.ScenePtr);
    const InstanceRegistry& instances = SceneInstances(*frame.ScenePtr);

    VkClearValue clears[2]{};
    // The moment frag writes (depth, depth²); clear to (1, 1) — far + far² — so
    // empty pixels read as fully lit ("blocker at infinity, zero variance").
    clears[0].color        = { { 1.0f, 1.0f, 0.0f, 0.0f } };
    clears[1].depthStencil = { 1.0f, 0 };

    for (uint32_t c = 0; c < kCascadeCount; ++c) {
        VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rpbi.renderPass        = m_RenderPass;
        rpbi.framebuffer       = m_CascadeFramebuffers[c];
        rpbi.renderArea.offset = { 0, 0 };
        rpbi.renderArea.extent = { kCascadeSize, kCascadeSize };
        rpbi.clearValueCount   = 2;
        rpbi.pClearValues      = clears;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

        const glm::mat4& lightVP = m_CascadeViewProj[c];
        instances.ForEach([&](InstanceHandle /*handle*/, const GpuInstance& inst) {
            const GpuMesh* mesh = meshes.Get(inst.Mesh);
            if (!mesh) return;

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->VertexBuffer, &offset);
            vkCmdBindIndexBuffer  (cmd, mesh->IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            VsmPushConstants pc{};
            pc.LightViewProj = lightVP;
            pc.Model         = inst.Transform;
            vkCmdPushConstants(cmd, m_PipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            uint32_t indexCount = 0;
            for (const GpuSubmesh& sub : mesh->Submeshes) indexCount += sub.IndexCount;
            if (indexCount == 0) return;
            vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
        });

        vkCmdEndRenderPass(cmd);
    }
}

// ---- one-time GPU resource creation -----------------------------------------

bool VSMShadow::CreateAtlas() {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = kMomentFormat;
    ici.extent      = { kCascadeSize, kCascadeSize, 1 };
    ici.mipLevels   = 1;
    ici.arrayLayers = kCascadeCount;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_Ctx->Device, &ici, nullptr, &m_AtlasImage) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_Ctx->Device, m_AtlasImage, &req);
    const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_AtlasMemory) != VK_SUCCESS) return false;
    vkBindImageMemory(m_Ctx->Device, m_AtlasImage, m_AtlasMemory, 0);

    {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image    = m_AtlasImage;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format   = kMomentFormat;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, kCascadeCount };
        if (vkCreateImageView(m_Ctx->Device, &vci, nullptr, &m_AtlasArrayView) != VK_SUCCESS) return false;
    }
    for (uint32_t i = 0; i < kCascadeCount; ++i) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image    = m_AtlasImage;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = kMomentFormat;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1 };
        if (vkCreateImageView(m_Ctx->Device, &vci, nullptr, &m_CascadeColorViews[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool VSMShadow::CreateTransientDepth() {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = m_TransientDepthFormat;
    ici.extent      = { kCascadeSize, kCascadeSize, 1 };
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_Ctx->Device, &ici, nullptr, &m_TransientDepthImage) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_Ctx->Device, m_TransientDepthImage, &req);
    const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_TransientDepthMemory) != VK_SUCCESS) return false;
    vkBindImageMemory(m_Ctx->Device, m_TransientDepthImage, m_TransientDepthMemory, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image    = m_TransientDepthImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = m_TransientDepthFormat;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(m_Ctx->Device, &vci, nullptr, &m_TransientDepthView) == VK_SUCCESS;
}

bool VSMShadow::CreateRenderPass() {
    VkAttachmentDescription atts[2]{};
    // Color (moments)
    atts[0].format         = kMomentFormat;
    atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Depth (transient — written, never sampled)
    atts[1].format         = m_TransientDepthFormat;
    atts[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 2;
    rpci.pAttachments    = atts;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;
    return vkCreateRenderPass(m_Ctx->Device, &rpci, nullptr, &m_RenderPass) == VK_SUCCESS;
}

bool VSMShadow::CreatePipeline(const char* shaderDir) {
    const std::string vertPath = std::string(shaderDir) + "/shadow_csm_vert.spv";
    const std::string fragPath = std::string(shaderDir) + "/shadow_vsm_frag.spv";
    VkShaderModule vert = LoadModule(m_Ctx->Device, vertPath.c_str());
    if (!vert) return false;
    VkShaderModule frag = LoadModule(m_Ctx->Device, fragPath.c_str());
    if (!frag) { vkDestroyShaderModule(m_Ctx->Device, vert, nullptr); return false; }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

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

    VkViewport viewport{ 0.0f, 0.0f,
        static_cast<float>(kCascadeSize), static_cast<float>(kCascadeSize),
        0.0f, 1.0f };
    VkRect2D scissor{ { 0, 0 }, { kCascadeSize, kCascadeSize } };
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    // VSM doesn't need front-face culling (it stores depth² regardless), so we
    // can use back-face culling for performance. No slope bias either — VSM's
    // light bleed is handled in the compose shader.
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

    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    blend.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments    = &blend;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(VsmPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(m_Ctx->Device, &plci, nullptr, &m_PipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(m_Ctx->Device, vert, nullptr);
        vkDestroyShaderModule(m_Ctx->Device, frag, nullptr);
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
    gpci.layout              = m_PipelineLayout;
    gpci.renderPass          = m_RenderPass;
    gpci.subpass             = 0;

    const VkResult r = vkCreateGraphicsPipelines(m_Ctx->Device, VK_NULL_HANDLE, 1, &gpci,
                                                 nullptr, &m_Pipeline);
    vkDestroyShaderModule(m_Ctx->Device, vert, nullptr);
    vkDestroyShaderModule(m_Ctx->Device, frag, nullptr);
    return r == VK_SUCCESS;
}

bool VSMShadow::CreateFramebuffers() {
    for (uint32_t i = 0; i < kCascadeCount; ++i) {
        VkImageView atts[2] = { m_CascadeColorViews[i], m_TransientDepthView };
        VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbci.renderPass      = m_RenderPass;
        fbci.attachmentCount = 2;
        fbci.pAttachments    = atts;
        fbci.width           = kCascadeSize;
        fbci.height          = kCascadeSize;
        fbci.layers          = 1;
        if (vkCreateFramebuffer(m_Ctx->Device, &fbci, nullptr, &m_CascadeFramebuffers[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool VSMShadow::CreateSampler() {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.compareEnable = VK_FALSE;
    sci.minLod = 0.0f; sci.maxLod = 0.0f;
    return vkCreateSampler(m_Ctx->Device, &sci, nullptr, &m_LinearSampler) == VK_SUCCESS;
}

bool VSMShadow::CreateSetLayout() {
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding         = 0;
    b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding         = 1;
    b[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 2;
    lci.pBindings    = b;
    return vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr, &m_LightingSetLayout) == VK_SUCCESS;
}

bool VSMShadow::CreateAlgoUbos() {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = sizeof(AlgoUbo);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (vkCreateBuffer(m_Ctx->Device, &bci, nullptr, &m_AlgoUboBuffers[i]) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_Ctx->Device, m_AlgoUboBuffers[i], &req);
        const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memType < 0) return false;
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(memType);
        if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_AlgoUboMemory[i]) != VK_SUCCESS) return false;
        vkBindBufferMemory(m_Ctx->Device, m_AlgoUboBuffers[i], m_AlgoUboMemory[i], 0);
        if (vkMapMemory(m_Ctx->Device, m_AlgoUboMemory[i], 0, req.size, 0, &m_AlgoUboMapped[i]) != VK_SUCCESS) return false;
        std::memset(m_AlgoUboMapped[i], 0, sizeof(AlgoUbo));
    }
    return true;
}

bool VSMShadow::CreateDescriptorSets() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = VulkanContext::kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = VulkanContext::kFramesInFlight;
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(m_Ctx->Device, &pci, nullptr, &m_DescriptorPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight] = {};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) layouts[i] = m_LightingSetLayout;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = m_DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(m_Ctx->Device, &ai, m_FrameSets.data()) != VK_SUCCESS) return false;

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo img{};
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img.imageView   = m_AtlasArrayView;
        img.sampler     = m_LinearSampler;

        VkDescriptorBufferInfo buf{};
        buf.buffer = m_AlgoUboBuffers[i];
        buf.offset = 0;
        buf.range  = sizeof(AlgoUbo);

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = m_FrameSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &img;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = m_FrameSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo     = &buf;

        vkUpdateDescriptorSets(m_Ctx->Device, 2, writes, 0, nullptr);
    }
    return true;
}

void VSMShadow::UpdateAlgoUbo(uint32_t frameSlot) {
    AlgoUbo ubo{};
    for (uint32_t c = 0; c < kCascadeCount; ++c) {
        ubo.CascadeViewProj[c] = m_CascadeViewProj[c];
    }
    ubo.CascadeSplits = glm::vec4(m_CascadeSplitDistance[0],
                                  m_CascadeSplitDistance[1],
                                  m_CascadeSplitDistance[2],
                                  m_CascadeSplitDistance[3]);
    ubo.VSMParamsVec  = glm::vec4(m_Params.MinVariance,
                                  m_Params.LightBleedReduction,
                                  m_Params.DepthBias,
                                  m_Params.NormalBias);
    ubo.AtlasParams   = glm::vec4(1.0f / static_cast<float>(kCascadeSize),
                                  static_cast<float>(kCascadeSize),
                                  static_cast<float>(kCascadeCount),
                                  m_Params.Enabled ? 1.0f : 0.0f);
    std::memcpy(m_AlgoUboMapped[frameSlot], &ubo, sizeof(ubo));
}

} // namespace RS
