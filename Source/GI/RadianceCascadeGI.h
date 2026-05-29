// Source/GI/RadianceCascadeGI.h — Phase 14a groundwork
// Stub IGIAlgorithm implementor for the RC × hash open-world GI design
// (Docs/RadianceCascades3D_OpenWorld.md). Phase 14a is wiring only: registers
// against the algo combo, reads RenderSettings, no compute work yet. The
// shader/SSBO buildout lands in Phase 14b once the harness extension validates
// the merge math.
#pragma once

#include "GI/IGIAlgorithm.h"
#include "RS/RenderSettings.h"

namespace RS {

class RadianceCascadeGI final : public IGIAlgorithm {
public:
    const char* Name() const override { return "RadianceCascades"; }

    // The settings pointer lets the algo read live UI values without growing
    // the IGIAlgorithm vtable. Main.cpp installs it after construction.
    void SetSettings(GISettings* gi) { m_Settings = gi; }

    void Initialize(const VulkanContext&, const GlobalSDF&) override {}
    void Terminate() override {}

    void DrawImGuiParams() override;

    void RecordPreFrame(VkCommandBuffer, const FrameContext&) override {}
    void RecordGather  (VkCommandBuffer, const FrameContext&) override {}

    void BindGIResourceForLighting(VkCommandBuffer, VkPipelineLayout,
                                   uint32_t, uint32_t) override {}

private:
    GISettings* m_Settings = nullptr;   // borrowed; owned by Main.cpp.
};

} // namespace RS
