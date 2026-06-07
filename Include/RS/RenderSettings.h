// Include/RS/RenderSettings.h
// Shared state for the Render Settings UI panel. Owned by Main.cpp and mutated
// in place by the panel.
#pragma once

namespace RS {

// Per-frame PBR settings. RealisticPbr enables the EON 2024 diffuse + Fdez-
// Aguera 2019 energy-compensated GGX + KHR_materials_specular semantics in
// the lighting compose. Default: on (cheap path stays available for A/B).
struct PbrSettings {
    bool RealisticPbr = true;
};

struct RenderSettings {
    PbrSettings PBR;
};

} // namespace RS
