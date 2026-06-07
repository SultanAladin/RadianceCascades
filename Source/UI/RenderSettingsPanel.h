// Source/UI/RenderSettingsPanel.h
#pragma once

namespace RS {

struct RenderSettings;
struct SkySettings;

void DrawRenderSettingsPanel(RenderSettings& settings);

// Standalone Hillaire 2020 atmosphere panel: Rayleigh / Mie / Ozone / radii /
// MieG / camera altitude / aerial-perspective bake toggle.
void DrawAtmospherePanel(SkySettings& sky);

} // namespace RS
