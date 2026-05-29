// Source/GI/RadianceCascadeGI.cpp — Phase 14a groundwork
// DrawImGuiParams() draws the RC-specific knobs that live alongside the algo
// combo in the Render Settings panel. Real compute work arrives in 14b.
#include "GI/RadianceCascadeGI.h"

#include "imgui.h"

namespace RS {

void RadianceCascadeGI::DrawImGuiParams() {
    if (!m_Settings) {
        ImGui::TextDisabled("RadianceCascades: no settings bound.");
        return;
    }
    GISettings& s = *m_Settings;

    ImGui::SliderInt  ("Cascades",     &s.CascadeCount, 1, 6);
    ImGui::SliderFloat("s0 (m)",       &s.S0Meters,     0.25f, 4.0f, "%.2f");
    ImGui::SliderInt  ("r0 (rays/c0)", &s.R0Rays,       8, 128);
    ImGui::SliderInt  ("Hash log2",    &s.HashLog2,     16, 22);
    ImGui::Checkbox   ("Bilinear fix", &s.BilinearFix);
    ImGui::SliderFloat("Indirect boost", &s.IndirectBoost, 0.0f, 4.0f, "%.2f");

    ImGui::Separator();
    static const char* kViewLabels[] = {
        "Off (lit)", "Hash cells", "Cascades", "Probe rays", "Irradiance only"
    };
    int viewIdx = static_cast<int>(s.DebugView);
    if (ImGui::Combo("Debug view", &viewIdx, kViewLabels, IM_ARRAYSIZE(kViewLabels))) {
        s.DebugView = static_cast<GIDebugView>(viewIdx);
    }

    ImGui::TextDisabled("Phase 14a: wiring only — no GI is shaded yet.");
}

} // namespace RS
