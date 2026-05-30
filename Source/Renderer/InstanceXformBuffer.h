// Source/Renderer/InstanceXformBuffer.h — Phase 13.5
// Per-instance inverse-model SSBO. Bound by SDFCone shadows (and Phase 14b
// rc_relight) so the trace can map world-space pixels into mesh-local space
// of whichever instance the pixel belongs to. The instance handle comes from
// the GBuffer identity attachment's R channel.
//
// Layout (matches sdf_trace consumers):
//   mat4 InverseModel[256]
//
// Host-visible coherent, ring-of-kFramesInFlight to avoid touching an
// in-flight slot. Refresh() rewrites the active slot each frame from the
// live instance registry — cheap (256 × 64 B = 16 KB memcpy worst-case).
#pragma once

#include "Core/VulkanContext.h"
#include "RS/Scene.h"   // MeshHandle, InstanceHandle
#include <glm/glm.hpp>
#include <array>

namespace RS {

class Scene;

struct InstanceXformBuffer {
    static constexpr uint32_t kMaxInstances = 256;
    static constexpr VkDeviceSize kBytes =
        static_cast<VkDeviceSize>(kMaxInstances) * sizeof(glm::mat4);

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> Buffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> Memories{};
    std::array<void*,          VulkanContext::kFramesInFlight> Mapped{};
};

bool InstanceXformBufferInitialize(InstanceXformBuffer& xb, const VulkanContext& ctx);
void InstanceXformBufferTerminate (InstanceXformBuffer& xb, const VulkanContext& ctx);

// Rewrite the per-frame slot with current inverse-model matrices keyed by
// instance handle. Empty slots get identity so out-of-range fetches in the
// shader stay benign.
//
// `sdfMesh` and `sdfAnchorInstance` solve Phase 13.5's floor-shadow bug:
// SDFCone traces in mesh-local space (cm) but the floor instance has an
// identity transform (world meters). Without a fallback, the floor's shader
// invocation transforms world→local with `identity`, sampling the cm SDF as
// if 1 m == 1 cm — so the resident ball's shadow appears 100× larger on the
// floor. We patch around it here by routing every non-SDF-mesh instance to
// the SDF anchor's invModel. That gives the floor a single correctly-scaled
// shadow under the anchor ball; multi-ball shadows on the floor land with
// the multi-mesh SDF in Phase 14b/16.
//
// Pass `sdfMesh = 0` to disable the fallback (every non-SDF instance gets
// identity, restoring pre-fix behavior).
void InstanceXformBufferRefresh(InstanceXformBuffer& xb,
                                uint32_t frameSlot,
                                const Scene& scene,
                                MeshHandle sdfMesh = 0,
                                InstanceHandle sdfAnchorInstance = 0);

} // namespace RS
