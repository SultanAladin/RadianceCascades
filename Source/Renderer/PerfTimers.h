// Source/Renderer/PerfTimers.h — Phase 16 (GPU timestamp infrastructure)
// One VkQueryPool, kPassCount * 2 timestamps per frame-in-flight slot. Each pass
// brackets work with BeginPass/EndPass; BeginFrame resets the slot's queries
// and reads back the previous-cycle slot's results (which BeginFrame's fence
// wait has just made GPU-complete).
//
// timestampPeriod is read from VkPhysicalDeviceProperties at init; ticks are
// converted to ms at readback.
//
// Pass indices are enum'd so call sites stay readable and ordering changes can
// shuffle the table in one place. kPassCount sizes the pool.
#pragma once

#include "Core/VulkanContext.h"

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

namespace RS {

enum class PerfPass : uint32_t {
    SkyBake = 0,
    GBuffer,
    PickingCopy,
    Shadow,
    GIPreFrame,
    GIGather,
    Lighting,
    GICompose,
    Tonemap,
    Preview,
    ImGuiDraw,
    kCount
};

constexpr uint32_t kPerfPassCount = static_cast<uint32_t>(PerfPass::kCount);

struct PerfTimers {
    VkQueryPool QueryPool        = VK_NULL_HANDLE;
    float       TimestampPeriodNs = 1.0f;       // ticks → ns multiplier
    bool        Supported        = false;       // false on devices without
                                                //   timestampComputeAndGraphics=1.
    bool        Initialized      = false;

    // Per-slot, per-pass ms results (filled at BeginFrame from the previous
    // cycle's writes). Index by uint32_t(PerfPass).
    std::array<std::array<float, kPerfPassCount>, VulkanContext::kFramesInFlight> LastMs{};

    // Per-slot, per-pass "was bracketed this cycle" flag. Used at readback to
    // skip passes that were gated off (e.g. GI when GI is disabled).
    std::array<std::array<uint8_t, kPerfPassCount>, VulkanContext::kFramesInFlight> WasRecorded{};

    // The pass set the host marks via BeginPass during recording. Cleared at
    // BeginFrame; consulted at the next BeginFrame for readback.
    std::array<uint8_t, kPerfPassCount> CurrentRecorded{};
};

bool PerfTimersInitialize(PerfTimers& pt, const VulkanContext& ctx);
void PerfTimersTerminate (PerfTimers& pt, const VulkanContext& ctx);

// Call at BeginFrame: reset this slot's queries and read back the previous
// cycle's results for this slot into LastMs[frameSlot]. Safe before any
// BeginPass on this slot's command buffer.
void PerfTimersBeginFrame(PerfTimers& pt, const VulkanContext& ctx,
                          VkCommandBuffer cmd, uint32_t frameSlot);

// Bracket helpers. Both write timestamps at the BOTTOM_OF_PIPE stage so the
// query records when the previous work has *finished*, which gives a clean
// pass-end signal (and a clean pass-start for the next pass).
void PerfTimersBeginPass(PerfTimers& pt, VkCommandBuffer cmd,
                         uint32_t frameSlot, PerfPass pass);
void PerfTimersEndPass  (PerfTimers& pt, VkCommandBuffer cmd,
                         uint32_t frameSlot, PerfPass pass);

const char* PerfPassName(PerfPass pass);

} // namespace RS
