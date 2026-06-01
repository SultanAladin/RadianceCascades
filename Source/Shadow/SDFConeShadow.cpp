// Source/Shadow/SDFConeShadow.cpp — Phase 13 (+ Phase 15d sparse rewire)
#include "Shadow/SDFConeShadow.h"
#include "SDF/GlobalSDF.h"
#include "Renderer/FrameContext.h"
#include "Core/Logger.h"
#include "imgui.h"

#include <cmath>
#include <cstring>

namespace RS {

namespace {

constexpr float kPi = 3.14159265358979323846f;

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

uint32_t Log2Ceil(uint32_t v) {
    uint32_t r = 0;
    while ((1u << r) < v) ++r;
    return r;
}

} // namespace

// -- lifecycle ----------------------------------------------------------------

void SDFConeShadow::Initialize(const VulkanContext& ctx,
                               VkFormat /*depthFormat*/,
                               VkExtent2D /*shadowMapResolution*/) {
    m_Ctx = &ctx;
    if (!CreateSetLayout())          { RS_LOG_ERROR("SDFConeShadow: set layout failed"); return; }
    if (!CreateAlgoUbos())           { RS_LOG_ERROR("SDFConeShadow: AlgoUbos failed");   return; }
    if (!CreateSparseParamsUbos())   { RS_LOG_ERROR("SDFConeShadow: SparseUbos failed"); return; }
    if (!CreateDummySparseBuffers()) { RS_LOG_ERROR("SDFConeShadow: dummies failed");    return; }
    if (!CreateDescriptorSets())     { RS_LOG_ERROR("SDFConeShadow: sets failed");       return; }

    // Initial sparse params = identity (1^3 dummy AABB). All descriptor sets
    // are written against the dummy SSBOs so binding is always valid even
    // before SetSDF supplies a real residency. The shader gates the trace
    // loop on the StrengthAndFlags.z (hasSDF) flag, so a missing SDF leaves
    // every pixel fully lit.
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        UpdateSparseParamsUbo(i);
    }
    RewriteAllSets();
    RS_LOG_INFO("SDFConeShadow ready (sparse VDB binding, awaiting SetSDF)");
}

void SDFConeShadow::Terminate(const VulkanContext& ctx) {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(ctx.Device);

    if (m_DescriptorPool)
        vkDestroyDescriptorPool(ctx.Device, m_DescriptorPool, nullptr);
    if (m_LightingSetLayout)
        vkDestroyDescriptorSetLayout(ctx.Device, m_LightingSetLayout, nullptr);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (m_AlgoUboMapped[i]) {
            vkUnmapMemory(ctx.Device, m_AlgoUboMemory[i]);
            m_AlgoUboMapped[i] = nullptr;
        }
        if (m_AlgoUboBuffers[i]) vkDestroyBuffer(ctx.Device, m_AlgoUboBuffers[i], nullptr);
        if (m_AlgoUboMemory [i]) vkFreeMemory   (ctx.Device, m_AlgoUboMemory [i], nullptr);

        if (m_SparseUboMapped[i]) {
            vkUnmapMemory(ctx.Device, m_SparseUboMemory[i]);
            m_SparseUboMapped[i] = nullptr;
        }
        if (m_SparseUboBuffers[i]) vkDestroyBuffer(ctx.Device, m_SparseUboBuffers[i], nullptr);
        if (m_SparseUboMemory [i]) vkFreeMemory   (ctx.Device, m_SparseUboMemory [i], nullptr);
    }
    m_AlgoUboBuffers   = {};
    m_AlgoUboMemory    = {};
    m_SparseUboBuffers = {};
    m_SparseUboMemory  = {};

    if (m_DummyBuffer) vkDestroyBuffer(ctx.Device, m_DummyBuffer, nullptr);
    if (m_DummyMemory) vkFreeMemory   (ctx.Device, m_DummyMemory, nullptr);
    m_DummyBuffer = VK_NULL_HANDLE;
    m_DummyMemory = VK_NULL_HANDLE;

    m_LightingSetLayout  = VK_NULL_HANDLE;
    m_DescriptorPool     = VK_NULL_HANDLE;
    m_FrameSets          = {};
    m_HasSDF             = false;
    m_IndexBuffer        = VK_NULL_HANDLE;
    m_PoolBuffer         = VK_NULL_HANDLE;
    m_Ctx                = nullptr;
}

// -- ImGui --------------------------------------------------------------------

void SDFConeShadow::DrawImGuiParams() {
    ImGui::Checkbox  ("Shadows enabled",   &m_Params.Enabled);
    ImGui::SliderFloat("Half-angle (°)",   &m_Params.HalfAngleDeg,  0.05f, 8.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cone half-angle widens the soft penumbra.\n"
                          "Larger ≈ softer shadows; 1.5° ≈ realistic sun.");
    }
    ImGui::SliderFloat("Max distance (m)", &m_Params.MaxDistance,   0.5f, 64.0f);
    ImGui::SliderFloat("Surface offset",   &m_Params.SurfaceOffset, 0.0f, 0.2f, "%.4f");
    ImGui::SliderInt  ("Max steps",        &m_Params.MaxSteps,      8,    128);
    ImGui::SliderFloat("Min step",         &m_Params.MinStep,       0.001f, 0.05f, "%.4f");
    ImGui::SliderFloat("Strength",         &m_Params.Strength,      0.0f, 1.0f);
    ImGui::Separator();
    ImGui::TextDisabled(m_HasSDF
        ? "Sparse SDF resident — tracing the active mesh."
        : "No SDF resident — output stays lit (no shadow).");
}

// -- record -------------------------------------------------------------------

void SDFConeShadow::RecordShadowPass(VkCommandBuffer /*cmd*/,
                                     const FrameContext& frame) {
    if (!m_Ctx) return;
    UpdateAlgoUbo(frame.FrameSlot, frame);
}

// -- lighting binding ---------------------------------------------------------

void SDFConeShadow::BindLightingDescriptorSet(VkCommandBuffer cmd,
                                              VkPipelineLayout lightingLayout,
                                              uint32_t frameSlot,
                                              uint32_t set) {
    if (!m_FrameSets[frameSlot]) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lightingLayout, set, 1, &m_FrameSets[frameSlot],
                            0, nullptr);
}

// -- residency hook -----------------------------------------------------------

void SDFConeShadow::SetSDF(const ResidentSparseSDF* sparse, bool hasSDF) {
    if (sparse && hasSDF) {
        m_HasSDF       = true;
        m_IndexBuffer  = sparse->IndexBuffer;
        m_PoolBuffer   = sparse->PoolBuffer;
        m_IndexBytes   = sparse->IndexBytes;
        m_PoolBytes    = sparse->PoolBytes;

        const uint32_t bs = sparse->BrickSize ? sparse->BrickSize : 8u;
        const float decode = sparse->MaxDist > 0.0f
                                 ? (sparse->MaxDist / 32767.0f)
                                 : 0.0f;
        m_SparseParams.AABBMin = glm::vec4(sparse->AABBMin, sparse->MaxDist);
        m_SparseParams.AABBMax = glm::vec4(sparse->AABBMax, decode);
        m_SparseParams.Dims    = glm::uvec4(sparse->Resolution,
                                            bs,
                                            sparse->BrickGrid,
                                            Log2Ceil(bs));
    } else {
        m_HasSDF       = false;
        m_IndexBuffer  = VK_NULL_HANDLE;
        m_PoolBuffer   = VK_NULL_HANDLE;
        m_IndexBytes   = 0;
        m_PoolBytes    = 0;
        m_SparseParams = {};
    }

    if (!m_Ctx || !m_LightingSetLayout) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        UpdateSparseParamsUbo(i);
    }
    RewriteAllSets();
}

void SDFConeShadow::SetInstanceXformBuffer(const VkBuffer* ssbosByFrame,
                                           VkDeviceSize bytesPerSlot) {
    if (!ssbosByFrame || bytesPerSlot == 0) return;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        m_InstXformBuffers[i] = ssbosByFrame[i];
    }
    m_XformBytes = bytesPerSlot;
    if (!m_Ctx || !m_LightingSetLayout) return;
    RewriteAllSets();
}

// -- one-time GPU resource creation ------------------------------------------

bool SDFConeShadow::CreateSetLayout() {
    VkDescriptorSetLayoutBinding bindings[5]{};
    // 0,1 = sparse SSBOs (BrickIndex + BrickPool)
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1] = bindings[0]; bindings[1].binding = 1;

    // 2 = SparseParams UBO
    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    // 3 = AlgoUbo
    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    // 4 = InstanceXform SSBO
    bindings[4].binding         = 4;
    bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 5;
    lci.pBindings    = bindings;
    return vkCreateDescriptorSetLayout(m_Ctx->Device, &lci, nullptr, &m_LightingSetLayout) == VK_SUCCESS;
}

bool SDFConeShadow::CreateAlgoUbos() {
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

bool SDFConeShadow::CreateSparseParamsUbos() {
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = sizeof(SparseParams);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (vkCreateBuffer(m_Ctx->Device, &bci, nullptr, &m_SparseUboBuffers[i]) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_Ctx->Device, m_SparseUboBuffers[i], &req);
        const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memType < 0) return false;

        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = static_cast<uint32_t>(memType);
        if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_SparseUboMemory[i]) != VK_SUCCESS) return false;
        vkBindBufferMemory(m_Ctx->Device, m_SparseUboBuffers[i], m_SparseUboMemory[i], 0);

        if (vkMapMemory(m_Ctx->Device, m_SparseUboMemory[i], 0, req.size, 0, &m_SparseUboMapped[i]) != VK_SUCCESS) return false;
        std::memset(m_SparseUboMapped[i], 0, sizeof(SparseParams));
    }
    return true;
}

bool SDFConeShadow::CreateDummySparseBuffers() {
    // A single 16-byte device-local STORAGE_BUFFER that backs both binding 0
    // and binding 1 when no real sparse SDF is resident. The shader never
    // reads it (hasSDF flag short-circuits the trace) — this just keeps the
    // descriptor write valid.
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = 16;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(m_Ctx->Device, &bci, nullptr, &m_DummyBuffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_Ctx->Device, m_DummyBuffer, &req);
    const int memType = FindMemoryType(m_Ctx->PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(m_Ctx->Device, &mai, nullptr, &m_DummyMemory) != VK_SUCCESS) return false;
    vkBindBufferMemory(m_Ctx->Device, m_DummyBuffer, m_DummyMemory, 0);
    return true;
}

bool SDFConeShadow::CreateDescriptorSets() {
    VkDescriptorPoolSize sizes[2]{};
    // Sparse SSBOs (2) + xform SSBO (1) = 3 per frame
    sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 3 * VulkanContext::kFramesInFlight;
    // SparseParams + AlgoUbo = 2 per frame
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 2 * VulkanContext::kFramesInFlight;

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

    // Stable UBO bindings (binding 2 = SparseParams, binding 3 = AlgoUbo).
    // SSBO bindings (0, 1, 4) are written by RewriteAllSets — but the dummy
    // sparse buffer + InstXform=null path means the first call from
    // Initialize already writes valid storage to keep validation quiet.
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorBufferInfo sparseUbo{};
        sparseUbo.buffer = m_SparseUboBuffers[i];
        sparseUbo.offset = 0;
        sparseUbo.range  = sizeof(SparseParams);
        VkDescriptorBufferInfo algoUbo{};
        algoUbo.buffer = m_AlgoUboBuffers[i];
        algoUbo.offset = 0;
        algoUbo.range  = sizeof(AlgoUbo);

        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = m_FrameSets[i];
        w[0].dstBinding = 2;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &sparseUbo;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = m_FrameSets[i];
        w[1].dstBinding = 3;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[1].pBufferInfo = &algoUbo;

        vkUpdateDescriptorSets(m_Ctx->Device, 2, w, 0, nullptr);
    }
    return true;
}

void SDFConeShadow::RewriteAllSets() {
    if (!m_Ctx) return;
    vkDeviceWaitIdle(m_Ctx->Device);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        // Bindings 0 (BrickIndex), 1 (BrickPool), and 4 (InstXform).
        VkDescriptorBufferInfo idxInfo{}, poolInfo{}, xformInfo{};
        idxInfo.buffer = m_HasSDF && m_IndexBuffer ? m_IndexBuffer : m_DummyBuffer;
        idxInfo.offset = 0;
        idxInfo.range  = m_HasSDF && m_IndexBuffer ? m_IndexBytes : 16;

        poolInfo.buffer = m_HasSDF && m_PoolBuffer ? m_PoolBuffer : m_DummyBuffer;
        poolInfo.offset = 0;
        poolInfo.range  = m_HasSDF && m_PoolBuffer ? m_PoolBytes : 16;

        VkWriteDescriptorSet writes[3]{};
        uint32_t n = 0;

        writes[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet          = m_FrameSets[i];
        writes[n].dstBinding      = 0;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[n].pBufferInfo     = &idxInfo;
        ++n;

        writes[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet          = m_FrameSets[i];
        writes[n].dstBinding      = 1;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[n].pBufferInfo     = &poolInfo;
        ++n;

        if (m_InstXformBuffers[i] != VK_NULL_HANDLE && m_XformBytes > 0) {
            xformInfo.buffer = m_InstXformBuffers[i];
            xformInfo.offset = 0;
            xformInfo.range  = m_XformBytes;

            writes[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n].dstSet          = m_FrameSets[i];
            writes[n].dstBinding      = 4;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].pBufferInfo     = &xformInfo;
            ++n;
        }

        vkUpdateDescriptorSets(m_Ctx->Device, n, writes, 0, nullptr);
    }
}

void SDFConeShadow::UpdateAlgoUbo(uint32_t frameSlot, const FrameContext& frame) {
    AlgoUbo ubo{};
    const float halfAngleRad = m_Params.HalfAngleDeg * (kPi / 180.0f);
    ubo.SunDirAndHalfAngle    = glm::vec4(glm::normalize(frame.SunDirection),
                                          halfAngleRad);
    ubo.TraceParamsAndSurfOff = glm::vec4(static_cast<float>(m_Params.MaxSteps),
                                          m_Params.MaxDistance,
                                          m_Params.MinStep,
                                          m_Params.SurfaceOffset);
    ubo.StrengthAndFlags      = glm::vec4(m_Params.Strength,
                                          m_Params.Enabled ? 1.0f : 0.0f,
                                          m_HasSDF         ? 1.0f : 0.0f,
                                          0.0f);
    std::memcpy(m_AlgoUboMapped[frameSlot], &ubo, sizeof(ubo));
}

void SDFConeShadow::UpdateSparseParamsUbo(uint32_t frameSlot) {
    if (!m_SparseUboMapped[frameSlot]) return;
    std::memcpy(m_SparseUboMapped[frameSlot], &m_SparseParams, sizeof(SparseParams));
}

} // namespace RS
