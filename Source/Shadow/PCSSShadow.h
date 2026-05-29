// Source/Shadow/PCSSShadow.h — Phase 11
// Second concrete IShadowAlgorithm. Reuses the same 4-cascade D32_SFLOAT atlas
// as PCFShadow (no second image), but its set=2 layout exposes BOTH a
// sampler2DArrayShadow (for the final PCF taps with hardware compare) AND a
// non-compare sampler2DArray (for the blocker-search reads that need raw
// depths). AlgoVariant() = 1 → lighting pipeline loads "lighting_pcss.spv".
//
// Descriptor contract (set=2 — differs from PCF):
//   binding 0 = sampler2DArrayShadow  (compare-LESS, the same compare sampler PCF uses)
//   binding 1 = AlgoUbo (CascadeViewProj[4] + Splits + PCSSParams + Atlas)
//   binding 2 = sampler2DArray         (linear non-compare, for blocker search)
#pragma once

#include "Shadow/IShadowAlgorithm.h"
#include "Shadow/ShadowMap.h"

#include <array>
#include <glm/glm.hpp>

namespace RS {

struct PCSSParams {
    float LightSizeUv         = 0.012f;  // virtual light radius in UV space (0..1 over cascade extent)
    int   BlockerSearchSamples = 16;     // taps for the blocker-search step
    int   PcfSamples           = 32;     // taps for the variable-radius PCF
    float MinPenumbra          = 1.0f;   // floor on penumbra in shadowmap texels
    float DepthBias            = 0.001f;
    float NormalBias           = 0.02f;
    bool  Enabled              = true;
};

class PCSSShadow final : public IShadowAlgorithm {
public:
    PCSSShadow();
    ~PCSSShadow() override;

    const char* Name() const override { return "PCSS"; }
    uint32_t    AlgoVariant() const override { return 1; }

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
    bool CreateSamplers();
    bool CreateSetLayout();
    bool CreateAlgoUbos();
    bool CreateDescriptorSets();
    void UpdateAlgoUbo(uint32_t frameSlot);

    const VulkanContext* m_Ctx = nullptr;
    ShadowMap            m_Map{};
    PCSSParams           m_Params{};

    VkSampler m_CompareSampler   = VK_NULL_HANDLE;   // sampler2DArrayShadow (LESS)
    VkSampler m_NonCompareSampler = VK_NULL_HANDLE;  // sampler2DArray (linear, no compare)

    // 256+48 byte UBO — same shape as PCFShadow but the .w slots are PCSS-specific.
    struct AlgoUbo {
        glm::mat4 CascadeViewProj[ShadowMap::kCascadeCount];
        glm::vec4 CascadeSplits;
        // x = LightSizeUv, y = DepthBias, z = NormalBias, w = MinPenumbra-in-texels
        glm::vec4 PCSSParamsVec;
        // x = 1/cascadeSize, y = cascadeSize, z = cascadeCount,
        // w = packed (BlockerSamples & 0xFFFF) | (PcfSamples << 16)
        glm::vec4 AtlasParams;
    };
    static_assert(sizeof(AlgoUbo) == 256 + 48, "PCSS AlgoUbo layout pinned");

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_AlgoUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_AlgoUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_AlgoUboMapped{};

    VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescriptorPool    = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_FrameSets{};
};

} // namespace RS
