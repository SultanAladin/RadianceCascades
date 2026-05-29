// Source/Renderer/Picking.h — Phase 9
// Identity-image readback for click-to-select. The renderer writes (instanceId,
// submeshId) into a R32G32_UINT attachment in the GBuffer pass. On click we
// queue "want pixel (x, y) for FrameSlot N"; that frame's command buffer
// transitions identity SHADER_READ_ONLY -> TRANSFER_SRC, copies 1x1 into a
// host-visible ring buffer, and transitions back. The next time FrameSlot N
// wakes (1 cycle later, after its fence is waited inside BeginFrame), the host
// reads the result and the panel updates.
//
// No vkQueueWaitIdle, no per-frame background copies, and no new sync primitives
// — the existing per-frame fence ring is the read-after-write boundary.
#pragma once

#include "Core/VulkanContext.h"

#include <array>
#include <cstdint>

namespace RS {

struct PickingSlot {
    VkBuffer       Staging      = VK_NULL_HANDLE;
    VkDeviceMemory Memory       = VK_NULL_HANDLE;
    uint32_t*      Mapped       = nullptr;   // points at 2 * uint32_t

    // Request state for this slot, set by RequestPick + cleared by RecordPickCopy.
    bool           Requested    = false;
    uint32_t       RequestedX   = 0;
    uint32_t       RequestedY   = 0;

    // Result state for this slot, set by RecordPickCopy + consumed by TryRead.
    bool           ResultReady  = false;
};

struct PickingSystem {
    std::array<PickingSlot, VulkanContext::kFramesInFlight> Slots;
    bool Initialized = false;
};

bool PickingInitialize(PickingSystem& sys, const VulkanContext& ctx);
void PickingTerminate (PickingSystem& sys, const VulkanContext& ctx);

// Queue a pick at viewport coordinate (x, y) to be serviced when the next
// command buffer for `frameSlot` records its copy. Overwrites any previous
// unsubmitted request for that slot.
void PickingRequest(PickingSystem& sys, uint32_t frameSlot,
                    uint32_t x, uint32_t y);

// Record (inside the frame's command buffer, AFTER GBufferPass has finished
// and BEFORE GBufferPreview begins so the identity image's expected layout is
// preserved) a 1x1 copy from the identity attachment into this slot's staging
// buffer. No-op if PickingRequest hasn't been called for this slot.
void PickingRecordCopy(PickingSystem& sys, VkCommandBuffer cmd,
                       uint32_t frameSlot,
                       VkImage identityImage, VkExtent2D extent);

// Read out the result for `frameSlot` if one is pending. Returns true iff
// outInstance/outSubmesh were written; clears the slot's pending flag.
bool PickingTryRead(PickingSystem& sys, uint32_t frameSlot,
                    uint32_t& outInstance, uint32_t& outSubmesh);

} // namespace RS
