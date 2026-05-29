// Source/GI/SDFGI.h — Phases 14-15
// Cascaded SDFGI probe grid: 4 cascades × 32^3 probes, SH band-2 irradiance,
// 512 probes/frame round-robin relight, half-res screen-space gather.
#pragma once

#include "GI/IGIAlgorithm.h"

namespace RS {

class SDFGI final : public IGIAlgorithm {
public:
    const char* Name() const override { return "SDFGI"; }
    void Initialize(const VulkanContext&, const GlobalSDF&) override {}
    void Terminate() override {}
    void DrawImGuiParams() override {}
    void RecordPreFrame(VkCommandBuffer, const FrameContext&) override {}
    void RecordGather  (VkCommandBuffer, const FrameContext&) override {}
    void BindGIResourceForLighting(VkCommandBuffer, VkPipelineLayout, uint32_t, uint32_t) override {}
};

} // namespace RS
