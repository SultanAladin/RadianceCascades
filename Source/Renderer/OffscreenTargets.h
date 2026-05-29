// Source/Renderer/OffscreenTargets.h — Phase 6
// Per-frame-in-flight ring of GBuffer attachments. Sized to the swapchain at
// init; v1 doesn't resize. Each frame in flight owns its own set so the GPU
// can keep sampling last frame's GBuffer while we record into the next.
//
// Attachment list (mirrors FrameContext.h):
//   * Albedo         RGBA8_UNORM
//   * Normal         A2B10G10R10_UNORM_PACK32 (encoded N*0.5+0.5)
//   * RoughMetalF0   RGBA8_UNORM   (R=roughness, G=metallic, B=F0, A=unused)
//   * Emissive       RGBA16_SFLOAT
//   * Depth          D32_SFLOAT
//   * Identity       R32G32_UINT   (instanceId, submeshId) — TRANSFER_SRC for picking
//   * LightHDR       RGBA16_SFLOAT (Phase 10) — written by Shaders/lighting.comp
//                                  as a storage image, sampled by the Lit-mode
//                                  GBufferPreview.
//
// All color/depth images carry SAMPLED_BIT so the Phase 6 preview blit (and the
// Phase 7+ lighting compute) can read them as sampled images after the GBuffer
// pass transitions them to SHADER_READ_ONLY_OPTIMAL via the render-pass's
// finalLayout.
#pragma once

#include "Core/VulkanContext.h"

#include <array>

namespace RS {

struct OffscreenAttachment {
    VkImage        Image  = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView    View   = VK_NULL_HANDLE;
    VkFormat       Format = VK_FORMAT_UNDEFINED;
};

struct OffscreenFrame {
    OffscreenAttachment Albedo;
    OffscreenAttachment Normal;
    OffscreenAttachment RoughMetalF0;
    OffscreenAttachment Emissive;
    OffscreenAttachment Depth;
    OffscreenAttachment Identity;
    OffscreenAttachment LightHDR;     // Phase 10: lighting.comp writes here, Lit preview reads.
};

struct OffscreenTargets {
    static constexpr uint32_t kColorAttachmentCount = 5;   // albedo, normal, rmf, emissive, identity
    static constexpr uint32_t kAttachmentCount      = 6;   // + depth

    std::array<OffscreenFrame, VulkanContext::kFramesInFlight> Frames;
    VkExtent2D Extent      = { 0, 0 };
    bool       Initialized = false;

    // Reusable samplers for the preview blit + (later) lighting compose.
    VkSampler  SamplerLinear  = VK_NULL_HANDLE;
    VkSampler  SamplerNearest = VK_NULL_HANDLE;
};

bool OffscreenTargetsInitialize(OffscreenTargets& ot, const VulkanContext& ctx);
void OffscreenTargetsTerminate (OffscreenTargets& ot, const VulkanContext& ctx);

// Static descriptors mirrored from FrameContext.h (handy for the preview blit
// and Phase 7's lighting compose).
inline VkFormat OffscreenFormatAlbedo  () { return VK_FORMAT_R8G8B8A8_UNORM;             }
inline VkFormat OffscreenFormatNormal  () { return VK_FORMAT_A2B10G10R10_UNORM_PACK32;   }
inline VkFormat OffscreenFormatRMF     () { return VK_FORMAT_R8G8B8A8_UNORM;             }
inline VkFormat OffscreenFormatEmissive() { return VK_FORMAT_R16G16B16A16_SFLOAT;        }
inline VkFormat OffscreenFormatDepth   () { return VK_FORMAT_D32_SFLOAT;                 }
inline VkFormat OffscreenFormatIdentity() { return VK_FORMAT_R32G32_UINT;                }
inline VkFormat OffscreenFormatLightHDR() { return VK_FORMAT_R16G16B16A16_SFLOAT;        }

} // namespace RS
