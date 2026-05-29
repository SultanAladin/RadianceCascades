// Include/RS/RenderSettings.h — Phase 14 groundwork
// Shared state for the Render Settings UI panel and the GI algo path. Owned by
// Main.cpp, mutated in place by the panel, read by the active IGIAlgorithm
// each frame. Kept header-only and POD so Application/, Source/UI/, and
// Source/GI/ can all see the same definition without a circular include.
#pragma once

#include "RS/GI.h"

#include <cstdint>

namespace RS {

// Debug visualisation modes for the radiance-cascade hash. Picked from a
// combo in the Render Settings panel; consumed by the lighting compose path
// once Phase 14b lands. Stays in this header so the panel can drive the combo
// without including any GI internals.
enum class GIDebugView : uint32_t {
    Off       = 0,   // Lit (default).
    HashCells = 1,   // Colour occupied hash cells by SH band-0 magnitude.
    Cascades  = 2,   // Tint each cascade level a distinct hue.
    ProbeRays = 3,   // Step-count visualisation of c0 rays.
    Irradiance= 4,   // Final irradiance only (no albedo, no direct).
};

struct GISettings {
    bool             Enabled        = false;             // master GI toggle.
    GIAlgorithmKind  Algorithm      = GIAlgorithmKind::SDFGI;

    // --- Radiance-cascade specific (consumed by RadianceCascadeGI). All
    // values match the defaults in Docs/RadianceCascades3D_OpenWorld.md §5.1.
    int   CascadeCount  = 4;        // c0..c3 by default; up to 6 supported.
    float S0Meters      = 1.0f;     // c0 probe spacing in metres.
    int   R0Rays        = 32;       // c0 rays per probe. Doc note: 16 caused
                                    // banding on curved surfaces in 3D;
                                    // bumped to 32 as the Phase 14a default.
    int   HashLog2      = 20;       // 2^N hash entries (20 → ~1M slots,
                                    // ~20 MiB payload at 20 B/cell).
    bool  BilinearFix   = true;     // Sannikov's bilinear-fix at cascade
                                    // boundaries — A/B toggle for ringing.
    float IndirectBoost = 1.0f;     // Multiplier on the indirect term in the
                                    // compose; mirrors rc3d_viewer.py.

    GIDebugView DebugView = GIDebugView::Off;
};

// One umbrella settings struct. New render-wide knobs slot in here so the
// panel + Main.cpp don't have to thread a growing parameter list. v1 holds
// GI; later phases may add an exposure block, SSR toggles, etc.
struct RenderSettings {
    GISettings GI;
};

} // namespace RS
