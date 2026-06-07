// Source/Shadow/IShadowAlgorithm.h
// Minimal shadow algorithm contract. The only concrete implementation left is
// SDFConeShadow; Lighting binds its descriptor set at set=2.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace RS {

struct VulkanContext;
struct FrameContext;

struct IShadowAlgorithm {
    virtual ~IShadowAlgorithm() = default;

    virtual const char* Name() const = 0;

    // Stable id for lighting pipeline rebuild bookkeeping.
    virtual uint32_t AlgoVariant() const = 0;

    virtual void Initialize(const VulkanContext& ctx,
                            VkFormat depthFormat,
                            VkExtent2D shadowMapResolution) = 0;
    virtual void Terminate(const VulkanContext& ctx) = 0;

    virtual void DrawImGuiParams() = 0;
    virtual void RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) = 0;

    virtual VkDescriptorSetLayout LightingSetLayout() const = 0;
    virtual void BindLightingDescriptorSet(VkCommandBuffer cmd,
                                           VkPipelineLayout lightingLayout,
                                           uint32_t frameSlot,
                                           uint32_t set) = 0;
};

} // namespace RS
