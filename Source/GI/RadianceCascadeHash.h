// Source/GI/RadianceCascadeHash.h — sparse radiance cascades storage core.
//
// Canonical 3D RC over a per-cascade sparse hash. Cascade c (0-based):
//   probe pitch     = s0 · 2^c
//   slots(c)        = BaseSlots >> 2c        (sub-table inside one key buffer)
//   dirs(c)         = D0 · 4^c               (octahedral res 4·2^c per probe)
//   payload texel   = vec4(rgb merged radiance, β transparency)
//   payload block   = BaseSlots · D0 texels per cascade (slots·dirs constant)
//
// Buffers (all per-frame-in-flight so concurrent frames never share memory):
//   binding 0: Keys      — uint per slot, all cascades packed (offset by
//                          RC_KeyOffset). 0 = empty sentinel.
//   binding 1: Payload   — vec4 texels, cascade c at c·BaseSlots·D0.
//   binding 2: ParamsUbo — RcHashParams.
//   binding 3: CellList  — per-cascade: [offset] = atomic counter,
//                          [offset+1..offset+slots] = packed keys.
//                          offset = c · (BaseSlots + 1).
//   binding 4: Resolve   — cascade-0 SH-L1: 3 vec4 per c0 slot
//                          (BaseSlots · 3 vec4). Written by rc_resolve,
//                          read by gi_compose.
//
// Frame protocol (single command buffer, barriers between steps):
//   1) clear keys + cell-list counters (+ payload/resolve once per slot)
//   2) insert     (screen + eye volume + parent chains)
//   3) relight    top-down, one dispatch per cascade c = N−1 … 0
//   4) resolve    cascade-0 dirs → SH-L1
//   5) compose    trilinear c0 gather → LightHDR
#pragma once

#include "Core/VulkanContext.h"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace RS {

struct GISettings;

// Mirrored in shaders/include/rc_hash.glsl. Keep sizes in sync.
struct RcHashParams {
    glm::vec4 EyePosAndBaseLog2;     // xyz = eye world, w = log2(BaseSlots)
    glm::vec4 SunDirAndIntensity;    // xyz = toward sun, w = intensity
    glm::vec4 SunColor;              // rgb, w = ambient sky luma
    glm::vec4 CascadeParams;         // x = s0 (m), y = cascadeCount,
                                     // z = D0 (dirs at c0), w = indirectBoost
    glm::vec4 InvViewProj0;          // columns of InvViewProj
    glm::vec4 InvViewProj1;
    glm::vec4 InvViewProj2;
    glm::vec4 InvViewProj3;
    glm::vec4 RenderExtentAndFlags;  // x/y = render w/h, z = bilinearFix,
                                     // w = debugView
    glm::vec4 MiscParams;            // x = frameIndex, y = scaleLocalPerWorld
    glm::vec4 WorldToLocal0;         // world→mesh-local, column-major
    glm::vec4 WorldToLocal1;
    glm::vec4 WorldToLocal2;
    glm::vec4 WorldToLocal3;
    glm::vec4 SdfAlbedo;             // rgb = SDF mesh albedo bounce tint
};
static_assert(sizeof(RcHashParams) == 15 * 16,
              "RcHashParams pinned at 240 bytes — mirror in rc_hash.glsl");

struct RadianceCascadeHash {
    static constexpr uint32_t kMaxCascades  = 6;
    static constexpr uint32_t kD0           = 16;   // dirs per c0 probe (4×4 oct)
    static constexpr uint32_t kProbeCap     = 16;   // linear-probe limit

    // BaseSlots = slots of cascade 0; cascade c gets BaseSlots >> 2c. With
    // BaseLog2 = 16 and 5 cascades: 65536 + 16384 + 4096 + 1024 + 256 keys,
    // payload = 5 · 65536 · 16 · 16 B = 80 MB... too fat. BaseLog2 = 14:
    // payload = cascades · 16384 · 16 texels · 16 B = 4 MB per cascade.
    uint32_t BaseLog2  = 14;
    uint32_t BaseSlots = 1u << 14;
    uint32_t Cascades  = 4;

    // Derived sizes (filled by Initialize).
    VkDeviceSize KeyBytes      = 0;
    VkDeviceSize PayloadBytes  = 0;
    VkDeviceSize CellListBytes = 0;
    VkDeviceSize ResolveBytes  = 0;

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> KeyBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> KeyMemory{};

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> PayloadBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> PayloadMemory{};

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> CellListBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> CellListMemory{};

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> ResolveBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> ResolveMemory{};

    // ParamsUbo: per-frame host-visible ring.
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> ParamsBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> ParamsMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> ParamsMapped{};

    // 4-byte host-visible ring for the cascade-0 probe-count readback.
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> ReadbackBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> ReadbackMemory{};
    std::array<uint32_t*,      VulkanContext::kFramesInFlight> ReadbackMapped{};

    VkDescriptorSetLayout SetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      DescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> Sets{};

    bool Initialized = false;

    // Host mirrors of the shader's sub-table math.
    uint32_t SlotsOf(uint32_t c)      const { return BaseSlots >> (2u * c); }
    uint32_t DirsOf(uint32_t c)       const { return kD0 << (2u * c); }
    uint32_t CellListOffset(uint32_t c) const { return c * (BaseSlots + 1u); }
};

bool RadianceCascadeHashInitialize(RadianceCascadeHash& h,
                                   const VulkanContext& ctx,
                                   uint32_t baseLog2,
                                   uint32_t cascades);
void RadianceCascadeHashTerminate (RadianceCascadeHash& h,
                                   const VulkanContext& ctx);

void RadianceCascadeHashWriteParams(RadianceCascadeHash& h,
                                    uint32_t frameSlot,
                                    const RcHashParams& params);

// Clear keys + every cascade's cell-list counter for `frameSlot`. Must be the
// first GI op of the frame (the frame fence guarantees this slot's previous
// use has fully retired, so no cross-frame hazard).
void RadianceCascadeHashClearForFrame(const RadianceCascadeHash& h,
                                      VkCommandBuffer cmd,
                                      uint32_t frameSlot);

// Copy cascade-0's cell counter into the readback ring for host stats.
void RadianceCascadeHashRecordReadback(const RadianceCascadeHash& h,
                                       VkCommandBuffer cmd,
                                       uint32_t frameSlot);

uint32_t RadianceCascadeHashReadProbeCount(const RadianceCascadeHash& h,
                                           uint32_t frameSlot);

} // namespace RS
