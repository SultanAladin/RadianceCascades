// Source/UI/RenderSettingsPanel.cpp — Phase 14a groundwork
#include "UI/RenderSettingsPanel.h"

#include "RS/RenderSettings.h"
#include "Renderer/SkyAtmosphere.h"
#include "GI/IGIAlgorithm.h"

#include "imgui.h"

#include <glm/glm.hpp>

namespace RS {

void DrawAtmospherePanel(SkySettings& sky) {
    ImGui::SetNextWindowPos (ImVec2(640.0f, 700.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 280.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Atmosphere")) {
        ImGui::TextUnformatted("Hillaire 2020 atmosphere parameters");
        ImGui::SliderFloat("Planet radius (km)",   &sky.PlanetRadiusKm,     6300.0f, 6400.0f);
        ImGui::SliderFloat("Atmosphere top (km)",  &sky.AtmosphereRadiusKm, 6400.0f, 6500.0f);
        ImGui::SliderFloat("Mie anisotropy g",     &sky.MieG,               0.0f,    0.99f);
        // β coefficients are in 1/m and very small — edit as 10⁻⁶ scale.
        constexpr float kRayleighScale = 1.0e-6f;
        constexpr float kMieScale      = 1.0e-6f;
        constexpr float kOzoneScale    = 1.0e-6f;
        glm::vec3 ray = sky.RayleighScattering / kRayleighScale;
        if (ImGui::SliderFloat3("Rayleigh β (×1e-6 /m)", &ray.x, 0.0f, 60.0f)) {
            sky.RayleighScattering = ray * kRayleighScale;
        }
        float mie = sky.MieScattering / kMieScale;
        if (ImGui::SliderFloat("Mie β (×1e-6 /m)", &mie, 0.0f, 20.0f)) {
            sky.MieScattering = mie * kMieScale;
        }
        glm::vec3 ozo = sky.OzoneAbsorption / kOzoneScale;
        if (ImGui::SliderFloat3("Ozone abs (×1e-6 /m)", &ozo.x, 0.0f, 5.0f)) {
            sky.OzoneAbsorption = ozo * kOzoneScale;
        }
        ImGui::SliderFloat("Camera altitude (m)", &sky.CameraAltitudeMeters, 0.0f, 5000.0f);
        ImGui::Separator();
        ImGui::Checkbox("Bake AerialPerspective LUT", &sky.BakeAerialPerspective);
        ImGui::TextDisabled("Not consumed yet — wires up future fog-on-geometry.");
    }
    ImGui::End();
}

int DrawRenderSettingsPanel(RenderSettings& settings, IGIAlgorithm* activeGI,
                            SkySettings* /*sky*/) {
    int requestedSwap = -1;

    ImGui::SetNextWindowPos (ImVec2(960.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 460.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render Settings")) {
        ImGui::TextUnformatted("PBR");
        ImGui::Checkbox("Realistic PBR (EON + energy comp + KHR specular)",
                        &settings.PBR.RealisticPbr);
        ImGui::TextDisabled("Off = classic Cook-Torrance (Lambert + uncompensated GGX).");
        ImGui::Separator();

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
