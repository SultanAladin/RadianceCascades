// Source/Shadow/PCFShadow.h — Phase 10
// First concrete IShadowAlgorithm. Owns:
//   * a ShadowMap (4-cascade D32_SFLOAT atlas, depth-only render pass)
//   * a sampler2DArrayShadow (compareOp = LESS, linear filtering)
//   * a per-frame-in-flight 256-byte algo UBO with cascade view-projs + splits
//   * a descriptor-set layout exposed to Lighting via LightingSetLayout()
//   * per-frame-in-flight descriptor sets that bind both above
//
// Public params live in PCFParams and the Shadows panel mutates them through
// DrawImGuiParams(). The lighting compose reads the UBO at set=2 binding=1 and
// samples set=2 binding=0 with sampler2DArrayShadow.
#pragma once

#include "Shadow/IShadowAlgorithm.h"
#include "Shadow/ShadowMap.h"

#include <array>
#include <glm/glm.hpp>

namespace RS {

struct PCFParams {
    int   KernelRadius   = 1;       // 1 = 3x3 taps, 2 = 5x5, 3 = 7x7
    float DepthBias      = 0.001f;  // added to receiver depth before compare
    float NormalBias     = 0.02f;   // world-space normal offset before projection
    float JitterStrength = 1.0f;    // 0 = grid PCF; >0 = blue-noise jittered taps
    bool  Enabled        = true;
};

class PCFShadow final : public IShadowAlgorithm {
public:
    PCFShadow();
    ~PCFShadow() override;

    const char* Name() const override { return "PCF"; }
    uint32_t    AlgoVariant() const override { return 0; }

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
    bool CreateSampler();
    bool CreateSetLayout();
    bool CreateAlgoUbos();
    bool CreateDescriptorSets();
    void UpdateAlgoUbo(uint32_t frameSlot);

    // Owned subsystems / context.
    const VulkanContext* m_Ctx = nullptr;
    ShadowMap            m_Map{};
    PCFParams            m_Params{};

    // Compare sampler — sampler2DArrayShadow on the shader side.
    VkSampler m_CompareSampler = VK_NULL_HANDLE;

    // Per-frame algo UBO. Plan §6.3 reserves a "256-byte algo UBO" at the
    // set=2 binding=1 contract slot — that's a guideline, not a Vulkan hard
    // limit (the device's maxUniformBufferRange is ≥16 KB). We stay just past
    // it (304 bytes); the lighting shader declares the matching layout.
    struct AlgoUbo {
        glm::mat4 CascadeViewProj[ShadowMap::kCascadeCount];   // 256 bytes
        glm::vec4 CascadeSplits;       // x..w = view-space far plane per cascade
        glm::vec4 PCFParamsVec;        // x=kernelRadius, y=depthBias, z=normalBias, w=jitter
        glm::vec4 AtlasParams;         // x=1/cascadeSize, y=cascadeSize, z=cascadeCount
    };
    static_assert(sizeof(AlgoUbo) == 256 + 48, "AlgoUbo layout pinned for the shader to match");

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_AlgoUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_AlgoUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_AlgoUboMapped{};

    VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescriptorPool    = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_FrameSets{};
};

} // namespace RS
