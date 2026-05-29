// Source/Renderer/OffscreenTargets.cpp — Phase 6
#include "Renderer/OffscreenTargets.h"
#include "Core/Logger.h"

namespace RS {

namespace {

int FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool CreateAttachment(const VulkanContext& ctx, VkExtent2D extent,
                      VkFormat format, VkImageUsageFlags usage,
                      VkImageAspectFlags aspect,
                      OffscreenAttachment& out) {
    out.Format = format;

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { extent.width, extent.height, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &out.Image) != VK_SUCCESS) {
        RS_LOG_ERROR("OffscreenTargets: vkCreateImage failed (format=%d)", (int)format);
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx.Device, out.Image, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) {
        RS_LOG_ERROR("OffscreenTargets: no DEVICE_LOCAL memory type");
        return false;
    }
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &out.Memory) != VK_SUCCESS) {
        RS_LOG_ERROR("OffscreenTargets: vkAllocateMemory failed");
        return false;
    }
    vkBindImageMemory(ctx.Device, out.Image, out.Memory, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image            = out.Image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = format;
    vci.subresourceRange = { aspect, 0, 1, 0, 1 };
    if (vkCreateImageView(ctx.Device, &vci, nullptr, &out.View) != VK_SUCCESS) {
        RS_LOG_ERROR("OffscreenTargets: vkCreateImageView failed");
        return false;
    }
    return true;
}

void DestroyAttachment(VkDevice device, OffscreenAttachment& a) {
    if (a.View)   vkDestroyImageView(device, a.View,   nullptr);
    if (a.Image)  vkDestroyImage    (device, a.Image,  nullptr);
    if (a.Memory) vkFreeMemory      (device, a.Memory, nullptr);
    a = {};
}

bool CreateSamplers(VkDevice device, OffscreenTargets& ot) {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sci.minLod       = 0.0f;
    sci.maxLod       = 0.0f;
    if (vkCreateSampler(device, &sci, nullptr, &ot.SamplerLinear) != VK_SUCCESS) return false;

    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(device, &sci, nullptr, &ot.SamplerNearest) != VK_SUCCESS) return false;
    return true;
}

} // namespace

bool OffscreenTargetsInitialize(OffscreenTargets& ot, const VulkanContext& ctx) {
    if (ot.Initialized) return true;
    ot.Extent = ctx.SwapchainExtent;

    constexpr VkImageUsageFlags kColorSampled =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    constexpr VkImageUsageFlags kDepthSampled =
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    constexpr VkImageUsageFlags kIdentityUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // Phase 10: storage so lighting.comp can imageStore, sampled so the Lit
    // preview blit can read. No COLOR_ATTACHMENT — compute pipeline writes
    // directly.
    constexpr VkImageUsageFlags kLightHDRUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        OffscreenFrame& f = ot.Frames[i];
        const bool ok =
            CreateAttachment(ctx, ot.Extent, OffscreenFormatAlbedo  (),
                             kColorSampled, VK_IMAGE_ASPECT_COLOR_BIT, f.Albedo)       &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatNormal  (),
                             kColorSampled, VK_IMAGE_ASPECT_COLOR_BIT, f.Normal)       &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatRMF     (),
                             kColorSampled, VK_IMAGE_ASPECT_COLOR_BIT, f.RoughMetalF0) &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatEmissive(),
                             kColorSampled, VK_IMAGE_ASPECT_COLOR_BIT, f.Emissive)     &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatDepth   (),
                             kDepthSampled, VK_IMAGE_ASPECT_DEPTH_BIT, f.Depth)        &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatIdentity(),
                             kIdentityUsage, VK_IMAGE_ASPECT_COLOR_BIT, f.Identity) &&
            CreateAttachment(ctx, ot.Extent, OffscreenFormatLightHDR(),
                             kLightHDRUsage, VK_IMAGE_ASPECT_COLOR_BIT, f.LightHDR);
        if (!ok) {
            RS_LOG_ERROR("OffscreenTargets: failed to create frame %u attachments", i);
            return false;
        }
    }

    if (!CreateSamplers(ctx.Device, ot)) {
        RS_LOG_ERROR("OffscreenTargets: failed to create samplers");
        return false;
    }

    ot.Initialized = true;
    RS_LOG_INFO("OffscreenTargets ready: %ux%u, %u frames",
                ot.Extent.width, ot.Extent.height,
                VulkanContext::kFramesInFlight);
    return true;
}

void OffscreenTargetsTerminate(OffscreenTargets& ot, const VulkanContext& ctx) {
    if (!ot.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        OffscreenFrame& f = ot.Frames[i];
        DestroyAttachment(ctx.Device, f.Albedo);
        DestroyAttachment(ctx.Device, f.Normal);
        DestroyAttachment(ctx.Device, f.RoughMetalF0);
        DestroyAttachment(ctx.Device, f.Emissive);
        DestroyAttachment(ctx.Device, f.Depth);
        DestroyAttachment(ctx.Device, f.Identity);
        DestroyAttachment(ctx.Device, f.LightHDR);
    }
    if (ot.SamplerLinear)  vkDestroySampler(ctx.Device, ot.SamplerLinear,  nullptr);
    if (ot.SamplerNearest) vkDestroySampler(ctx.Device, ot.SamplerNearest, nullptr);
    ot.SamplerLinear  = VK_NULL_HANDLE;
    ot.SamplerNearest = VK_NULL_HANDLE;
    ot.Initialized    = false;
}

} // namespace RS
