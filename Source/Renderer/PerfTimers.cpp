// Source/Renderer/PerfTimers.cpp — Phase 16
#include "Renderer/PerfTimers.h"
#include "Core/Logger.h"

#include <array>
#include <cstdint>

namespace RS {

namespace {

constexpr uint32_t kTimestampsPerPass     = 2;
constexpr uint32_t kTimestampsPerSlot     = kPerfPassCount * kTimestampsPerPass;
constexpr uint32_t kTotalTimestamps       =
    kTimestampsPerSlot * VulkanContext::kFramesInFlight;

uint32_t SlotBase(uint32_t frameSlot) { return frameSlot * kTimestampsPerSlot; }
uint32_t QueryIndex(uint32_t frameSlot, PerfPass pass, bool isEnd) {
    return SlotBase(frameSlot)
         + static_cast<uint32_t>(pass) * kTimestampsPerPass
         + (isEnd ? 1u : 0u);
}

} // namespace

bool PerfTimersInitialize(PerfTimers& pt, const VulkanContext& ctx) {
    if (pt.Initialized) return true;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.PhysicalDevice, &props);
    pt.TimestampPeriodNs = props.limits.timestampPeriod;
    pt.Supported = (props.limits.timestampComputeAndGraphics != VK_FALSE)
                   && pt.TimestampPeriodNs > 0.0f;
    if (!pt.Supported) {
        RS_LOG_INFO("PerfTimers: GPU timestamps unsupported (period=%.2f) — perf widget will show CPU only",
                    pt.TimestampPeriodNs);
        pt.Initialized = true;
        return true;
    }

    VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kTotalTimestamps;
    if (vkCreateQueryPool(ctx.Device, &qpci, nullptr, &pt.QueryPool) != VK_SUCCESS) {
        RS_LOG_ERROR("PerfTimers: vkCreateQueryPool failed");
        return false;
    }

    pt.Initialized = true;
    RS_LOG_INFO("PerfTimers ready: %u queries (%u pass × %u slot × 2 ts), period=%.2f ns",
                kTotalTimestamps, kPerfPassCount,
                VulkanContext::kFramesInFlight, pt.TimestampPeriodNs);
    return true;
}

void PerfTimersTerminate(PerfTimers& pt, const VulkanContext& ctx) {
    if (!pt.Initialized) return;
    if (pt.QueryPool) vkDestroyQueryPool(ctx.Device, pt.QueryPool, nullptr);
    pt = {};
}

void PerfTimersBeginFrame(PerfTimers& pt, const VulkanContext& ctx,
                          VkCommandBuffer cmd, uint32_t frameSlot) {
    if (!pt.Initialized || !pt.Supported) return;

    // 1) Previous-cycle readback: the fence wait in VulkanContextBeginFrame for
    //    this slot has just completed, so any timestamps written into this slot
    //    last cycle are now safe to read. Skip on the very first cycle (all
    //    WasRecorded zero).
    bool anyRecorded = false;
    for (uint8_t r : pt.WasRecorded[frameSlot]) { if (r) { anyRecorded = true; break; } }
    if (anyRecorded) {
        std::array<uint64_t, kTimestampsPerSlot> raw{};
        // No WAIT_BIT: the fence has already made them ready. Pass
        // VK_QUERY_RESULT_64_BIT so we don't lose precision on the diff.
        vkGetQueryPoolResults(ctx.Device, pt.QueryPool,
                              SlotBase(frameSlot), kTimestampsPerSlot,
                              raw.size() * sizeof(uint64_t),
                              raw.data(), sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT);
        auto& lastMs = pt.LastMs[frameSlot];
        for (uint32_t i = 0; i < kPerfPassCount; ++i) {
            if (!pt.WasRecorded[frameSlot][i]) { lastMs[i] = 0.0f; continue; }
            const uint64_t t0 = raw[i * 2 + 0];
            const uint64_t t1 = raw[i * 2 + 1];
            const double   dn = static_cast<double>(t1 - t0) * pt.TimestampPeriodNs;
            lastMs[i] = static_cast<float>(dn / 1.0e6);
        }
    }

    // 2) Reset this slot's queries before this cycle writes into them.
    vkCmdResetQueryPool(cmd, pt.QueryPool, SlotBase(frameSlot), kTimestampsPerSlot);
    pt.WasRecorded[frameSlot] = {};
    pt.CurrentRecorded         = {};
}

void PerfTimersBeginPass(PerfTimers& pt, VkCommandBuffer cmd,
                         uint32_t frameSlot, PerfPass pass) {
    if (!pt.Initialized || !pt.Supported) return;
    // BOTTOM_OF_PIPE: the timestamp lands once all previously-submitted work
    // has finished — i.e. the pass start sees a clean baseline regardless of
    // the prior pass's stage mix.
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        pt.QueryPool, QueryIndex(frameSlot, pass, false));
}

void PerfTimersEndPass(PerfTimers& pt, VkCommandBuffer cmd,
                       uint32_t frameSlot, PerfPass pass) {
    if (!pt.Initialized || !pt.Supported) return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        pt.QueryPool, QueryIndex(frameSlot, pass, true));
    pt.WasRecorded[frameSlot][static_cast<uint32_t>(pass)] = 1u;
    pt.CurrentRecorded[static_cast<uint32_t>(pass)] = 1u;
}

const char* PerfPassName(PerfPass pass) {
    switch (pass) {
        case PerfPass::SkyBake:     return "SkyBake";
        case PerfPass::GBuffer:     return "GBuffer";
        case PerfPass::PickingCopy: return "PickingCopy";
        case PerfPass::Shadow:      return "Shadow";
        case PerfPass::Lighting:    return "Lighting";
        case PerfPass::Tonemap:     return "Tonemap";
        case PerfPass::Preview:     return "Preview";
        case PerfPass::ImGuiDraw:   return "ImGui";
        case PerfPass::kCount:      return "?";
    }
    return "?";
}

} // namespace RS
