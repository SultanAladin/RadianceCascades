// Include/RS/Vulkan.h
// Thin Vulkan handle aliases so consumers of RenderingSubsystem.lib can avoid
// pulling <vulkan/vulkan.h> into their own translation units when all they need
// is to pass through opaque handles.
#pragma once

#include <vulkan/vulkan.h>

namespace RS {

using VkInstanceHandle       = VkInstance;
using VkPhysicalDeviceHandle = VkPhysicalDevice;
using VkDeviceHandle         = VkDevice;
using VkQueueHandle          = VkQueue;
using VkImageViewHandle      = VkImageView;
using VkCommandBufferHandle  = VkCommandBuffer;

} // namespace RS
