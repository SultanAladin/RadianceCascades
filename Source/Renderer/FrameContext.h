// Source/Renderer/FrameContext.h
// Per-frame snapshot passed to every IShadowAlgorithm / IGIAlgorithm record
// callback. Internal — never exposed in Include/.
#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "RS/Renderer.h"

namespace RS {

class Scene;

struct FrameContext {
    CameraView    Cam{};
    glm::vec3     SunDirection = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3     SunColor     = glm::vec3(1.0f, 0.95f, 0.85f);
    float         SunIntensity = 3.0f;
    float         TimeSeconds  = 0.0f;
    uint32_t      FrameIndex   = 0;     // monotonic frame counter
    uint32_t      FrameSlot    = 0;     // index into kFramesInFlight rings
    const Scene*  ScenePtr     = nullptr;

    VkImageView   GBufferAlbedo        = VK_NULL_HANDLE;
    VkImageView   GBufferNormal        = VK_NULL_HANDLE;
    VkImageView   GBufferRoughMetalF0  = VK_NULL_HANDLE;
    VkImageView   GBufferEmissive      = VK_NULL_HANDLE;
    VkImageView   GBufferDepth         = VK_NULL_HANDLE;
    VkImageView   GBufferIdentity      = VK_NULL_HANDLE;

    VkExtent2D    RenderExtent         = { 0, 0 };
};

} // namespace RS
