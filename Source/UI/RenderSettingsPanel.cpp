// Source/UI/RenderSettingsPanel.cpp
#include "UI/RenderSettingsPanel.h"

#include "RS/RenderSettings.h"
#include "Renderer/SkyAtmosphere.h"

#include "imgui.h"

#include <glm/glm.hpp>

namespace RS {

void DrawAtmospherePanel(SkySettings& sky) {
    ImGui::SetNextWindowPos (ImVec2(640.0f, 700.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 280.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Atmosphere")) {
        ImGui::TextUnformatted("Hillaire 2020 atmosphere parameters");
        ImGui::SliderFloat("Planet radius (km)",  &sky.PlanetRadiusKm,     6300.0f, 6400.0f);
        ImGui::SliderFloat("Atmosphere top (km)", &sky.AtmosphereRadiusKm, 6400.0f, 6500.0f);
        ImGui::SliderFloat("Mie anisotropy g",    &sky.MieG,               0.0f,    0.99f);

        constexpr float kRayleighScale = 1.0e-6f;
        constexpr float kMieScale      = 1.0e-6f;
        constexpr float kOzoneScale    = 1.0e-6f;
        glm::vec3 ray = sky.RayleighScattering / kRayleighScale;
        if (ImGui::SliderFloat3("Rayleigh beta (x1e-6 /m)", &ray.x, 0.0f, 60.0f)) {
            sky.RayleighScattering = ray * kRayleighScale;
        }
        float mie = sky.MieScattering / kMieScale;
        if (ImGui::SliderFloat("Mie beta (x1e-6 /m)", &mie, 0.0f, 20.0f)) {
            sky.MieScattering = mie * kMieScale;
        }
        glm::vec3 ozo = sky.OzoneAbsorption / kOzoneScale;
        if (ImGui::SliderFloat3("Ozone abs (x1e-6 /m)", &ozo.x, 0.0f, 5.0f)) {
            sky.OzoneAbsorption = ozo * kOzoneScale;
        }
        ImGui::SliderFloat("Camera altitude (m)", &sky.CameraAltitudeMeters, 0.0f, 5000.0f);
        ImGui::Separator();
        ImGui::Checkbox("Bake AerialPerspective LUT", &sky.BakeAerialPerspective);
        ImGui::TextDisabled("Not consumed yet - wires up future fog-on-geometry.");
    }
    ImGui::End();
}

void DrawRenderSettingsPanel(RenderSettings& settings) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 150.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render Settings")) {
        ImGui::TextUnformatted("PBR");
        ImGui::Checkbox("Realistic PBR (EON + energy comp + KHR specular)",
                        &settings.PBR.RealisticPbr);
        ImGui::TextDisabled("Off = classic Cook-Torrance (Lambert + uncompensated GGX).");
    }
    ImGui::End();
}

} // namespace RS
