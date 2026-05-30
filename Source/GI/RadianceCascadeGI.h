// Source/GI/RadianceCascadeGI.h — Phase 14b
// Concrete IGIAlgorithm implementor for the RC × Hash design. Owns:
//   - RadianceCascadeHash (key + payload + cell-list + params SSBOs)
//   - rc_insert.comp pipeline (primary + secondary insertion)
//   - rc_relight.comp pipeline (R0 rays + SH-L1 accumulate)
//   - gi_compose.comp pipeline (modulate LightHDR after lighting)
//
// Frame ordering inside the host:
//   RecordPreFrame  → hash clear (KeyBuf + CellList counter) + relight dispatch
//                     against PREVIOUS frame's hash (so the first frame just
//                     does nothing useful — fine, irradiance ramps up).
//   RecordGather    → insert primary + insert secondary, repopulating hash.
//                     Compose runs separately via BindGIResourceForLighting.
// The compose dispatch is fired by RadianceCascadeGI::RecordCompose, called
// from Main.cpp after LightingPassRecord.
#pragma once

#include "GI/IGIAlgorithm.h"
#include "GI/RadianceCascadeHash.h"
#include "RS/RenderSettings.h"

#include <array>
#include <cstdint>
#include <string>
#include <glm/glm.hpp>

namespace RS {

struct OffscreenTargets;
struct GlobalSDF;

class RadianceCascadeGI final : public IGIAlgorithm {
public:
    const char* Name() const override { return "RadianceCascades"; }

    void SetSettings(GISettings* gi) { m_Settings = gi; }

    // Phase 14b: hook the GI algo onto the GBuffer + LightHDR ring + global SDF.
    // Lives outside the IGIAlgorithm interface so SDFGI (which doesn't need
    // any of this) stays a clean stub. Main.cpp wires it after Initialize.
    void SetFrameResources(const OffscreenTargets* targets, const GlobalSDF* sdf,
                          const char* shaderArtifactsDir);

    // Push the resident SDF view in. Mirrors SDFConeShadow::SetSDF — called
    // from the same residency-changed branch in Main.cpp.
    void SetSDFView(VkImageView view, VkSampler sampler,
                    const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                    float maxDist, bool hasSDF);

    void Initialize(const VulkanContext& ctx, const GlobalSDF& sdf) override;
    void Terminate() override;

    void DrawImGuiParams() override;

    void RecordPreFrame(VkCommandBuffer cmd, const FrameContext& frame) override;
    void RecordGather  (VkCommandBuffer cmd, const FrameContext& frame) override;

    // After LightingPassRecord — modulate LightHDR by GI + handle debug viz.
    // Not in IGIAlgorithm (SDFGI's path will diverge); called directly from
    // Main.cpp on the RC algo.
    void RecordCompose (VkCommandBuffer cmd, const FrameContext& frame);

    void BindGIResourceForLighting(VkCommandBuffer cmd, VkPipelineLayout layout,
                                   uint32_t set, uint32_t binding) override;

    // Hash core exposed so the lighting pass can match its set=3 layout.
    VkDescriptorSetLayout HashSetLayout() const { return m_Hash.SetLayout; }

private:
    bool CreateInsertPipeline();
    bool CreateRelightPipeline();
    bool CreateComposePipeline();
    bool CreateGBufferDescriptors();
    bool CreateSdfDescriptors();
    bool CreateLightHdrDescriptors();
    void RewriteSdfDescriptors();
    void RewriteGBufferDescriptors();
    void RewriteLightHdrDescriptors();

    GISettings*               m_Settings = nullptr;
    const VulkanContext*      m_Ctx      = nullptr;
    const OffscreenTargets*   m_Targets  = nullptr;
    const GlobalSDF*          m_GlobalSdf= nullptr;
    std::string               m_ShaderDir;

    RadianceCascadeHash m_Hash;

    // ---- Insert pipeline ----
    VkDescriptorSetLayout m_InsertGBufferSetLayout = VK_NULL_HANDLE; // set=0
    VkDescriptorPool      m_InsertGBufferPool      = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_InsertGBufferSets{};
    VkPipelineLayout      m_InsertLayout   = VK_NULL_HANDLE;
    VkPipeline            m_InsertPipeline = VK_NULL_HANDLE;

    // ---- Relight pipeline ----
    VkSampler             m_SdfSampler             = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_RelightSdfSetLayout    = VK_NULL_HANDLE; // set=0
    VkDescriptorPool      m_RelightSdfPool         = VK_NULL_HANDLE;
    VkDescriptorSet       m_RelightSdfSet          = VK_NULL_HANDLE;
    VkPipelineLayout      m_RelightLayout          = VK_NULL_HANDLE;
    VkPipeline            m_RelightPipeline        = VK_NULL_HANDLE;

    // ---- Compose pipeline (post-lighting) ----
    VkDescriptorSetLayout m_ComposeGBufferSetLayout = VK_NULL_HANDLE;  // set=0
    VkDescriptorSetLayout m_ComposeLightHdrSetLayout = VK_NULL_HANDLE; // set=1
    VkDescriptorPool      m_ComposePool             = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_ComposeGBufferSets{};
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_ComposeLightHdrSets{};
    VkPipelineLayout      m_ComposeLayout   = VK_NULL_HANDLE;
    VkPipeline            m_ComposePipeline = VK_NULL_HANDLE;

    // SDF residency snapshot, pushed via SetSDFView.
    VkImageView m_SdfView   = VK_NULL_HANDLE;
    glm::vec3   m_AabbMin   = glm::vec3(0.0f);
    glm::vec3   m_AabbMax   = glm::vec3(1.0f);
    float       m_MaxDist   = 1.0f;
    bool        m_HasSDF    = false;
};

} // namespace RS
