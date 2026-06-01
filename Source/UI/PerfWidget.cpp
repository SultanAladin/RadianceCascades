// Source/UI/PerfWidget.cpp — Phase 16
#include "UI/PerfWidget.h"
#include "Renderer/PerfTimers.h"

#include "imgui.h"

#include <algorithm>

namespace RS {

namespace {

// Compact per-pass row: name, ms, percentage of total. Skips passes that
// weren't recorded this cycle (the WasRecorded bitmap is owned by PerfTimers).
void DrawPassRows(const PerfTimers& pt, uint32_t frameSlot, float gpuTotalMs) {
    if (!ImGui::BeginTable("##perf_passes", 3,
                           ImGuiTableFlags_BordersInnerH |
                           ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_SizingStretchProp)) {
        return;
    }
    ImGui::TableSetupColumn("Pass",   ImGuiTableColumnFlags_WidthStretch, 0.5f);
    ImGui::TableSetupColumn("ms",     ImGuiTableColumnFlags_WidthStretch, 0.25f);
    ImGui::TableSetupColumn("%",      ImGuiTableColumnFlags_WidthStretch, 0.25f);
    ImGui::TableHeadersRow();

    for (uint32_t i = 0; i < kPerfPassCount; ++i) {
        if (!pt.WasRecorded[frameSlot][i]) continue;
        const float ms = pt.LastMs[frameSlot][i];
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(PerfPassName(static_cast<PerfPass>(i)));
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%6.3f", ms);
        ImGui::TableSetColumnIndex(2);
        const float pct = (gpuTotalMs > 0.0f) ? (ms / gpuTotalMs) * 100.0f : 0.0f;
        ImGui::Text("%5.1f", pct);
    }
    ImGui::EndTable();
}

} // namespace

void PerfWidgetDraw(PerfWidgetState& ws, const PerfTimers& pt, uint32_t frameSlot) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 510.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 380.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Perf", &ws.Visible)) {
        ImGui::End();
        return;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const float cpuMs = io.DeltaTime * 1000.0f;
    // Pick the slot whose readback is freshest. Panels draw *before*
    // BeginFrame, so the slot the caller passes in is the one that will be
    // recorded next — its current `LastMs` is one full cycle stale. The other
    // slot was filled during this frame's last completed BeginFrame and is
    // the freshest data we have.
    const uint32_t displaySlot =
        (frameSlot + VulkanContext::kFramesInFlight - 1u) % VulkanContext::kFramesInFlight;
    float gpuTotalMs = 0.0f;
    for (uint32_t i = 0; i < kPerfPassCount; ++i) {
        if (pt.WasRecorded[displaySlot][i]) {
            gpuTotalMs += pt.LastMs[displaySlot][i];
        }
    }

    // Push into the rolling ring.
    ws.HistoryHead = (ws.HistoryHead + 1u) % PerfWidgetState::kHistory;
    ws.CpuMsHistory[ws.HistoryHead] = cpuMs;
    ws.GpuMsHistory[ws.HistoryHead] = gpuTotalMs;

    ImGui::Text("CPU frame: %5.2f ms  (%.0f FPS)", cpuMs,
                cpuMs > 0.0001f ? 1000.0f / cpuMs : 0.0f);
    if (pt.Supported) {
        ImGui::Text("GPU total: %5.2f ms", gpuTotalMs);
    } else {
        ImGui::TextDisabled("GPU timings unavailable (timestampComputeAndGraphics=0)");
    }

    // Frame-time graph (ms). Show the newest sample on the right by feeding
    // ImGui::PlotLines with the offset that maps history-head → graph end.
    const float graphHeight = 60.0f;
    const int   count       = static_cast<int>(PerfWidgetState::kHistory);
    const int   offset      = static_cast<int>((ws.HistoryHead + 1u) % PerfWidgetState::kHistory);

    ImGui::Separator();
    ImGui::TextUnformatted("Frame ms (CPU, last 120 frames)");
    ImGui::PlotLines("##cpu_ms", ws.CpuMsHistory.data(), count, offset,
                     nullptr, 0.0f, 33.3f,
                     ImVec2(0.0f, graphHeight));

    if (pt.Supported) {
        ImGui::TextUnformatted("Frame ms (GPU total)");
        ImGui::PlotLines("##gpu_ms", ws.GpuMsHistory.data(), count, offset,
                         nullptr, 0.0f, 33.3f,
                         ImVec2(0.0f, graphHeight));
    }

    ImGui::Separator();
    if (pt.Supported) {
        ImGui::TextUnformatted("GPU per-pass breakdown");
        DrawPassRows(pt, displaySlot, gpuTotalMs);
    }

    ImGui::End();
}

} // namespace RS
