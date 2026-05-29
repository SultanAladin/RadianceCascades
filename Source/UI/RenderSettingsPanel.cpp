// Source/UI/RenderSettingsPanel.cpp — Phase 14a groundwork
#include "UI/RenderSettingsPanel.h"

#include "RS/RenderSettings.h"
#include "GI/IGIAlgorithm.h"

#include "imgui.h"

namespace RS {

int DrawRenderSettingsPanel(RenderSettings& settings, IGIAlgorithm* activeGI) {
    int requestedSwap = -1;

    ImGui::SetNextWindowPos (ImVec2(960.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 380.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render Settings")) {
        ImGui::TextUnformatted("Global Illumination");
        ImGui::Checkbox("Enable GI", &settings.GI.Enabled);

        // Greying the rest out when GI is disabled keeps the panel honest —
        // changing a slider while disabled won't accidentally feel meaningful.
        ImGui::BeginDisabled(!settings.GI.Enabled);

        static const char* kAlgoLabels[] = {
            "SDFGI",
            "Radiance Cascades",
        };
        int algoIdx = static_cast<int>(settings.GI.Algorithm);
        if (ImGui::Combo("Algorithm", &algoIdx, kAlgoLabels, IM_ARRAYSIZE(kAlgoLabels))) {
            if (algoIdx != static_cast<int>(settings.GI.Algorithm)) {
                requestedSwap = algoIdx;
            }
        }
        ImGui::TextDisabled("SDFGI is the v0 placeholder. RadianceCascades is the\n"
                            "open-world target (Phase 14 — wiring only for now).");

        ImGui::Separator();
        if (activeGI) {
            ImGui::TextUnformatted(activeGI->Name());
            activeGI->DrawImGuiParams();
        } else {
            ImGui::TextDisabled("No GI algorithm bound.");
        }

        ImGui::EndDisabled();
    }
    ImGui::End();
    return requestedSwap;
}

} // namespace RS
