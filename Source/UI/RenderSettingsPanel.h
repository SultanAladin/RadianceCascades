// Source/UI/RenderSettingsPanel.h — Phase 14a groundwork
// Render Settings ImGui panel. Master GI on/off plus algo combo; the active
// algo's own params are drawn inline (mirrors DrawShadowsPanel). Returns the
// requested GI algo variant if the combo changed this frame, -1 otherwise.
#pragma once

namespace RS {

struct RenderSettings;
struct IGIAlgorithm;

int DrawRenderSettingsPanel(RenderSettings& settings, IGIAlgorithm* activeGI);

} // namespace RS
