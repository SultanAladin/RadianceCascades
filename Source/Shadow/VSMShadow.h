// Source/Shadow/VSMShadow.h — Phase 11
// Variance Shadow Maps (Donnelly & Lauritzen 2006). Stores (depth, depth²) per
// texel into an RG16F 4-cascade array, then the lighting compose evaluates
// Chebyshev's inequality to estimate visibility. No hardware compare; the
// linear filter just averages moments, which is mathematically valid and gives
// a free pre-filter without a separate blur pass.
//
// Light bleed (a classic VSM artifact) is mitigated by raising the lower bound
// `pMax = (x − μ) / (variance + (x − μ)²)` away from zero with a slider
// (LightBleedReduction).
//
// This algo owns its own atlas (separate from PCF/PCSS's D32_SFLOAT atlas) +
// a color-output render pass + its own depth-only-for-test attachment to
// reject hidden geometry while writing moments. ShadowMap stays single-
// responsibility for the depth atlas.
//
// Descriptor contract (set=2):
//   binding 0 = sampler2DArray (linear, RG16F moments)
//   binding 1 = AlgoUbo
#pragma once

#include "Shadow/IShadowAlgorithm.h"
#include "Shadow/ShadowMap.h"   // reuse kCascadeCount + cascade fit logic

#include <array>
#include <glm/glm.hpp>

namespace RS {

struct VSMParams {
    float MinVariance         = 2.0e-5f;  // floor on variance - kills numerical noise
    float LightBleedReduction = 0.20f;    // 0..1; bigger = harder shadow but tighter penumbra
    float DepthBias           = 0.001f;
    float NormalBias          = 0.02f;
    bool  Enabled             = true;
};

class VSMShadow final : public IShadowAlgorithm {
public:
    VSMShadow();
    ~VSMShadow() override;

    const char* Name() const override { return "VSM"; }
    uint32_t    AlgoVariant() const override { return 2; }

    void Initialize(const VulkanContext& ctx,
                    VkFormat depthFormat,
                    VkExtent2D shadowMapResolution) override;
    void Terminate (const VulkanContext& ctx) override;

    void DrawImGuiParams() override;
    void RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) override;

    VkDescriptorSetLayout LightingSetLayout() const override { return m_LightingSetLayout; }
    void BindLightingDescriptorSet(VkCommandBuffer cmd,
                                   VkPipelineLayout lightingLayout,
                                   uint32_t frameSlot,
                                   uint32_t set) override;

private:
    // Same per-cascade resolution as ShadowMap — keeps the layout symmetric and
    // means the cascade-fit math drops in unchanged.
    static constexpr uint32_t kCascadeCount = ShadowMap::kCascadeCount;
    static constexpr uint32_t kCascadeSize  = ShadowMap::kCascadeSize;
    static constexpr VkFormat kMomentFormat = VK_FORMAT_R16G16_SFLOAT;

    // ---- Per-frame cascade transforms (live) ----
    std::array<glm::mat4, kCascadeCount> m_CascadeViewProj{};
    std::array<float,     kCascadeCount> m_CascadeSplitDistance{};

    // ---- GPU resources ----
    VkImage        m_AtlasImage      = VK_NULL_HANDLE;
    VkDeviceMemory m_AtlasMemory     = VK_NULL_HANDLE;
    VkImageView    m_AtlasArrayView  = VK_NULL_HANDLE;     // sampled (linear)
    std::array<VkImageView, kCascadeCount> m_CascadeColorViews{};   // color attachment per cascade

    // Per-cascade transient depth (used only as a depth attachment so the
    // moment frag's discard / depth-test still rejects back-facing fragments).
    // A single shared single-layer image; cleared per cascade.
    VkImage        m_TransientDepthImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_TransientDepthMemory = VK_NULL_HANDLE;
    VkImageView    m_TransientDepthView   = VK_NULL_HANDLE;
    VkFormat       m_TransientDepthFormat = VK_FORMAT_D32_SFLOAT;

    VkRenderPass     m_RenderPass     = VK_NULL_HANDLE;
    VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_Pipeline       = VK_NULL_HANDLE;
    std::array<VkFramebuffer, kCascadeCount> m_CascadeFramebuffers{};

    VkSampler             m_LinearSampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescriptorPool    = VK_NULL_HANDLE;

    // ---- Algo UBO (same shape as PCF/PCSS for descriptor-write ergonomics) ----
    struct AlgoUbo {
        glm::mat4 CascadeViewProj[kCascadeCount];
        glm::vec4 CascadeSplits;
        // x = MinVariance, y = LightBleedReduction, z = DepthBias, w = NormalBias
        glm::vec4 VSMParamsVec;
        // x = 1/cascadeSize, y = cascadeSize, z = cascadeCount, w = enabledFlag
        glm::vec4 AtlasParams;
    };
    static_assert(sizeof(AlgoUbo) == 256 + 48, "VSM AlgoUbo layout pinned");

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_AlgoUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_AlgoUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_AlgoUboMapped{};

    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_FrameSets{};

    const VulkanContext* m_Ctx = nullptr;
    VSMParams            m_Params{};

    // ---- Internals ----
    bool CreateAtlas();
    bool CreateTransientDepth();
    bool CreateRenderPass();
    bool CreatePipeline(const char* shaderArtifactsDir);
    bool CreateFramebuffers();
    bool CreateSampler();
    bool CreateSetLayout();
    bool CreateAlgoUbos();
    bool CreateDescriptorSets();
    void UpdateAlgoUbo(uint32_t frameSlot);
    void BuildCascades(const struct FrameContext& frame);
    void RecordMomentPass(VkCommandBuffer cmd, const FrameContext& frame);
};

} // namespace RS
