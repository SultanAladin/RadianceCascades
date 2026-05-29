// Include/RS/Renderer.h
// Public facade. Hosts (standalone exe or Pigment/game engine/CAD) talk to this
// type only. All subsystem state lives behind a private Impl pointer.
#pragma once

#include <cstdint>
#include <memory>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "RS/Shadow.h"
#include "RS/GI.h"

namespace RS {

class Scene;
class MaterialRegistry;

struct InitDesc {
    VkInstance       Instance            = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice      = VK_NULL_HANDLE;
    VkDevice         Device              = VK_NULL_HANDLE;
    VkQueue          GraphicsQueue       = VK_NULL_HANDLE;
    uint32_t         GraphicsQueueFamily = 0;
    uint32_t         RenderWidth         = 1280;
    uint32_t         RenderHeight        = 720;
    const char*      ShaderArtifactsDir  = "Artifacts/Shaders";
};

struct CameraView {
    glm::mat4 View;
    glm::mat4 Projection;        // Y-flipped for Vulkan
    glm::vec3 EyePositionWorld;
    float     NearClip;
    float     FarClip;
};

struct FrameTimings {
    float GBufferMs    = 0.0f;
    float ShadowMs     = 0.0f;
    float GIGatherMs   = 0.0f;
    float GIRelightMs  = 0.0f;
    float LightingMs   = 0.0f;
    float TonemapMs    = 0.0f;
    float TotalGpuMs   = 0.0f;
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool Initialize(const InitDesc& desc);
    void Terminate();
    void Resize(uint32_t width, uint32_t height);

    void RenderFrame(VkCommandBuffer cmd,
                     const CameraView& camera,
                     FrameTimings*     outTimings);

    void SetShadowAlgorithm(ShadowAlgorithmKind kind);
    void SetGIAlgorithm    (GIAlgorithmKind     kind);

    VkImageView OffscreenColorImage   () const;   // RGBA16F, post-tonemap
    VkImageView OffscreenIdentityImage() const;   // RG32UI, instance/submesh ID

    Scene&            GetScene    ();
    MaterialRegistry& GetMaterials();

    // Host calls this between its own ImGui::NewFrame() and ImGui::Render().
    void DrawImGuiPanel();

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace RS
