// Source/UI/PerfWidget.h — Phase 16
// Per-frame timing HUD. Reads CPU delta from ImGui's IO and GPU per-pass ms
// from PerfTimers's read-back ring. Maintains a small rolling history for the
// frame-time graph.
#pragma once

#include "Renderer/PerfTimers.h"

#include <array>
#include <cstdint>

namespace RS {

struct PerfWidgetState {
    static constexpr uint32_t kHistory = 120;

    std::array<float, kHistory> CpuMsHistory{};
    std::array<float, kHistory> GpuMsHistory{};
    uint32_t HistoryHead = 0;       // newest index written
    bool     Visible     = true;
};

// Records the latest CPU + GPU samples and draws the HUD. CPU ms is read from
// ImGui::GetIO().DeltaTime; GPU ms is summed from `pt.LastMs[frameSlot]`.
void PerfWidgetDraw(PerfWidgetState& ws, const PerfTimers& pt, uint32_t frameSlot);

} // namespace RS
