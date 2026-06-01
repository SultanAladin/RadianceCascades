// Source/UI/RenderSettingsPanel.h — Phase 14a groundwork
// Render Settings ImGui panel. Master GI on/off plus algo combo; the active
// algo's own params are drawn inline (mirrors DrawShadowsPanel). Returns the
// requested GI algo variant if the combo changed this frame, -1 otherwise.
#pragma once

namespace RS {

struct RenderSettings;
struct SkySettings;
struct IGIAlgorithm;

// Render Settings — PBR + GI knobs. `sky` is kept in the signature for source
// compatibility; atmosphere lives in its own panel now (see DrawAtmospherePanel).
int DrawRenderSettingsPanel(RenderSettings& settings, IGIAlgorithm* activeGI,
                            SkySettings* sky = nullptr);

// Standalone Hillaire 2020 atmosphere panel — Rayleigh / Mie / Ozone / radii /
// MieG / camera altitude / aerial-perspective bake toggle.
void DrawAtmospherePanel(SkySettings& sky);

} // namespace RS
