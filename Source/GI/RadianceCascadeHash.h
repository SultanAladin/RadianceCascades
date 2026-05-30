// Source/GI/RadianceCascadeHash.h — Phase 14b
// World-space hash table + SH-L1 payload for the Radiance Cascade × Hash GI
// algorithm. Owned by RadianceCascadeGI; bound at set=3 of the lighting compose
// AND set=0 of the rc_insert / rc_relight compute shaders.
//
// Sizing (locked from Taichi C3 sweep):
//   Hash slots          = 2^20 (1 048 576)            — 4 MB key SSBO
//   Payload per slot    = 4 floats (SH-L1 luma, 1 vec4) — 16 MB payload SSBO
//   Linear-probe cap    = 8 (Taichi C3 worst = 3 @ 2^20)
//
// Phase 14b ships **luma-only** SH-L1 (4 bands × 1 channel = 1 vec4). The
// compose path modulates by GBuffer albedo to recover chroma — this is the
// standard low-budget RC GI compaction and keeps the payload at 16 MB instead
// of 48 MB for RGB SH. Phase 14c can promote to RGB SH if the smoke test
// shows obvious chroma bleeding (red ball doesn't bleed onto white floor).
//
// SSBOs:
//   binding 0: HashKeys[N]        — uint32 packed cell key, 0 = empty sentinel
//   binding 1: ProbePayload[N]    — vec4 = (Y_00, Y_1y, Y_1z, Y_1x) of luma
//   binding 2: ParamsUbo          — hash size, cascade params, eye pos, sun
//   binding 3: CellList[N]        — uint32 packed cell key for occupied slots,
//                                   index 0 is the running counter (atomic).
//                                   Relight reads `count = CellList[0]` then
//                                   iterates `CellList[1..count]`.
//
// All buffers are device-local, persistent. Each frame:
//   1) PreFrame: clear HashKeys to 0 + reset CellList[0] = 0 (vkCmdFillBuffer)
//   2) Insert dispatch(es): populate HashKeys + append occupied keys to CellList
//   3) Relight dispatch:   walk CellList[1..count], trace SDF, write payload
//   4) Lighting/Compose:   read HashKeys + ProbePayload to fetch SH at worldPos
//
// The shared set=3 layout is exposed via `LightingSetLayout()` so the lighting
// pass can build its pipeline layout against it. The same descriptor set is
// also bound at compute-set=0 of rc_insert/rc_relight (different pipeline
// layout but binding-compatible).
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
    glm::vec4 EyePosAndHashLog2;     // xyz = camera eye world, w = log2(N)
    glm::vec4 SunDirAndIntensity;    // xyz = direction toward sun, w = sun intensity
    glm::vec4 SunColor;              // rgb, w = ambient sky contribution
    glm::vec4 CascadeParams;         // x = s0 (meters), y = cascadeCount,
                                     // z = r0Rays, w = indirectBoost
    glm::vec4 InvViewProj0;          // rows of InvViewProj — packed as 4 vec4s
    glm::vec4 InvViewProj1;
    glm::vec4 InvViewProj2;
    glm::vec4 InvViewProj3;
    glm::vec4 RenderExtentAndFlags;  // x/y = render width/height,
                                     // z = bilinearFix, w = debugView
    glm::vec4 SdfAabbMin;            // xyz = mesh-local AABB min (for relight)
    glm::vec4 SdfAabbMax;            // xyz = mesh-local AABB max, w = decode scale
    glm::vec4 SecondaryParams;       // x = secondary radius L0 (cells),
                                     // y = grow per cascade,
                                     // z = frameIndex (for jitter),
                                     // w = unused
};
static_assert(sizeof(RcHashParams) == 12 * 16,
              "RcHashParams pinned at 192 bytes — mirror in rc_hash.glsl");

struct RadianceCascadeHash {
    static constexpr uint32_t kMaxHashLog2     = 22;    // upper bound for sizing
    static constexpr uint32_t kBytesPerPayload = 16;    // 4 floats / cell (luma SH-L1)
    static constexpr uint32_t kProbeCap        = 8;     // linear probe limit

    // Active size (matches GISettings::HashLog2 at Initialize time). Re-create
    // the hash core if HashLog2 changes — Phase 14b ships with the slider
    // disabled at runtime, only the boot-time value matters.
    uint32_t HashLog2     = 20;
    uint32_t SlotCount    = 1u << 20;

    VkBuffer        KeyBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory  KeyMemory        = VK_NULL_HANDLE;

    VkBuffer        PayloadBuffer    = VK_NULL_HANDLE;
    VkDeviceMemory  PayloadMemory    = VK_NULL_HANDLE;

    VkBuffer        CellListBuffer   = VK_NULL_HANDLE;   // [0]=count, [1..]=keys
    VkDeviceMemory  CellListMemory   = VK_NULL_HANDLE;

    // ParamsUbo: per-frame-in-flight host-visible ring.
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> ParamsBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> ParamsMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> ParamsMapped{};

    VkDescriptorSetLayout SetLayout      = VK_NULL_HANDLE;   // set=3 / set=0
    VkDescriptorPool      DescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> Sets{};

    bool Initialized = false;
};

bool RadianceCascadeHashInitialize(RadianceCascadeHash& h,
                                   const VulkanContext& ctx,
                                   uint32_t hashLog2);
void RadianceCascadeHashTerminate (RadianceCascadeHash& h,
                                   const VulkanContext& ctx);

// Refresh the params UBO for `frameSlot`. Reads from GISettings + the per-frame
// camera / SDF residency snapshot in `params`. Host-visible coherent → no
// barrier needed.
void RadianceCascadeHashWriteParams(RadianceCascadeHash& h,
                                    uint32_t frameSlot,
                                    const RcHashParams& params);

// Clear HashKeys[*] = 0 and CellList[0] = 0. Records vkCmdFillBuffer +
// pipeline barrier on STORAGE/COMPUTE. Must be the first compute op of the
// frame.
void RadianceCascadeHashClearForFrame(const RadianceCascadeHash& h,
                                      VkCommandBuffer cmd);

} // namespace RS
