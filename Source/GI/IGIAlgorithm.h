// Source/GI/IGIAlgorithm.h
// Plugin interface for GI algorithms. v1 has one implementor (SDFGI); voxel-
// cone and radiance-cascades drop in later without renderer surgery.
#pragma once

#include <vulkan/vulkan.h>

namespace RS {

struct VulkanContext;
struct FrameContext;
struct GlobalSDF;

struct IGIAlgorithm {
    virtual ~IGIAlgorithm() = default;

    virtual const char* Name() const = 0;

    virtual void Initialize(const VulkanContext& ctx, const GlobalSDF& sdf) = 0;
    virtual void Terminate() = 0;

    virtual void DrawImGuiParams() = 0;

    virtual void RecordPreFrame(VkCommandBuffer cmd, const FrameContext& frame) = 0;  // probe relight
    virtual void RecordGather  (VkCommandBuffer cmd, const FrameContext& frame) = 0;  // screen gather

    virtual void BindGIResourceForLighting(VkCommandBuffer cmd,
                                           VkPipelineLayout lightingLayout,
                                           uint32_t set,
                                           uint32_t binding) = 0;
};

} // namespace RS
