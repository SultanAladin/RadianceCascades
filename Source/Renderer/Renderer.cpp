// Source/Renderer/Renderer.cpp
// Facade implementation. Phase 1 ships a no-op skeleton so the public API
// links; Phases 6+ wire up the GBuffer/Lighting/Tonemap stages.
#include "RS/Renderer.h"
#include "RS/Camera.h"
#include "RS/Scene.h"
#include "RS/Material.h"

#include "Core/Logger.h"

#include <utility>

namespace RS {

struct Renderer::Impl {
    InitDesc Init{};
    bool     Initialised = false;

    ShadowAlgorithmKind ShadowKind = ShadowAlgorithmKind::PCF;
    GIAlgorithmKind     GIKind     = GIAlgorithmKind::SDFGI;

    // Phase 1 stubs — Phase 5 replaces these with the real registries.
    Scene            Scene_;
    MaterialRegistry Materials_;
};

Renderer::Renderer()
    : m_Impl(std::make_unique<Impl>()) {}

Renderer::~Renderer() {
    if (m_Impl && m_Impl->Initialised) {
        Terminate();
    }
}

bool Renderer::Initialize(const InitDesc& desc) {
    if (!m_Impl) return false;
    m_Impl->Init        = desc;
    m_Impl->Initialised = true;
    RS_LOG_INFO("Renderer::Initialize stub: render target %ux%u", desc.RenderWidth, desc.RenderHeight);
    return true;
}

void Renderer::Terminate() {
    if (!m_Impl) return;
    m_Impl->Initialised = false;
}

void Renderer::Resize(uint32_t width, uint32_t height) {
    if (!m_Impl) return;
    m_Impl->Init.RenderWidth  = width;
    m_Impl->Init.RenderHeight = height;
}

void Renderer::RenderFrame(VkCommandBuffer /*cmd*/,
                           const CameraView& /*camera*/,
                           FrameTimings*     outTimings) {
    if (outTimings) {
        *outTimings = FrameTimings{};
    }
}

void Renderer::SetShadowAlgorithm(ShadowAlgorithmKind kind) {
    if (m_Impl) m_Impl->ShadowKind = kind;
}

void Renderer::SetGIAlgorithm(GIAlgorithmKind kind) {
    if (m_Impl) m_Impl->GIKind = kind;
}

VkImageView Renderer::OffscreenColorImage   () const { return VK_NULL_HANDLE; }
VkImageView Renderer::OffscreenIdentityImage() const { return VK_NULL_HANDLE; }

Scene&            Renderer::GetScene    () { return m_Impl->Scene_; }
MaterialRegistry& Renderer::GetMaterials() { return m_Impl->Materials_; }

void Renderer::DrawImGuiPanel() {
    // Phase 3+ replaces this with the full panel.
}

} // namespace RS
