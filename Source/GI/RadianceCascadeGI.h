// Source/GI/RadianceCascadeGI.h — sparse radiance cascades (canonical).
// Concrete IGIAlgorithm implementor. Owns:
//   - RadianceCascadeHash (keys + payload + cell-lists + resolve + params)
//   - rc_insert.comp  (screen + eye-volume insertion w/ parent merge chains)
//   - rc_relight.comp (per-cascade dispatch, disjoint intervals, top-down merge)
//   - rc_resolve.comp (cascade-0 directional radiance → SH-L1)
//   - gi_compose.comp (trilinear c0 gather → LightHDR after lighting)
//
// Frame ordering inside the host (single-frame pipeline, no cross-frame state):
//   RecordPreFrame → clear this slot's keys + cell-list counters.
//   RecordGather   → insert, then relight cascades N−1…0 (barrier between
//                    each so child merges read final parent radiance), then
//                    resolve c0 to SH.
//   RecordCompose  → gather + add to LightHDR (called after LightingPassRecord).
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
struct ResidentSparseSDF;

class RadianceCascadeGI final : public IGIAlgorithm {
public:
    const char* Name() const override { return "RadianceCascades"; }

    void SetSettings(GISettings* gi) { m_Settings = gi; }

    // Phase 14b: hook the GI algo onto the GBuffer + LightHDR ring + global SDF.
    // Lives outside the IGIAlgorithm interface so SDFGI (which doesn't need
    // any of this) stays a clean stub. Main.cpp wires it after Initialize.
    void SetFrameResources(const OffscreenTargets* targets, const GlobalSDF* sdf,
                          const char* shaderArtifactsDir);

    // Phase 15e: push the sparse SDF residency in. Mirrors SDFConeShadow::SetSDF
    // — called from the same residency-changed branch in Main.cpp. Pass nullptr
    // to bind dummies (GI degrades to sky/escape only).
    void SetSDFView(const ResidentSparseSDF* sparse, bool hasSDF);

    // Phase 14b GI fix: the relight pass traces probes (world-space metres)
    // against the SDF, whose AABB + brick grid are authored in mesh-local
    // (OBJ) space. Push the SDF mesh's world→local transform — inverse of the
    // anchor instance's model matrix — so relight can map probes into local
    // before tracing, mirroring lighting_sdfcone's per-pixel invModel mapping.
    // Pass identity (the default) when no SDF instance is live.
    void SetSDFWorldTransform(const glm::mat4& meshLocalToWorld);

    // Phase 14c: albedo of the SDF mesh, used to tint radiance leaving a probe
    // hit (so the bounce carries the bouncing surface's colour). Single mesh-
    // wide tint until per-hit material (GBuffer reprojection / baked bricks).
    void SetSDFAlbedo(const glm::vec3& albedo) { m_SdfAlbedo = albedo; }

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
    bool CreateResolvePipeline();
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

    // ---- Relight pipeline (sparse SDF traced per cascade) ----
    VkDescriptorSetLayout m_RelightSdfSetLayout    = VK_NULL_HANDLE; // set=0
    VkDescriptorPool      m_RelightSdfPool         = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_RelightSdfSets{};
    VkPipelineLayout      m_RelightLayout          = VK_NULL_HANDLE;
    VkPipeline            m_RelightPipeline        = VK_NULL_HANDLE;

    // ---- Resolve pipeline (c0 dirs → SH-L1) ----
    VkPipelineLayout      m_ResolveLayout          = VK_NULL_HANDLE;
    VkPipeline            m_ResolvePipeline        = VK_NULL_HANDLE;

    // SparseParams UBO ring + dummy buffer for the no-SDF fallback.
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_SparseUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_SparseUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_SparseUboMapped{};
    VkBuffer       m_DummySparseBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_DummySparseMemory = VK_NULL_HANDLE;

    // ---- Compose pipeline (post-lighting) ----
    VkDescriptorSetLayout m_ComposeGBufferSetLayout = VK_NULL_HANDLE;  // set=0
    VkDescriptorSetLayout m_ComposeLightHdrSetLayout = VK_NULL_HANDLE; // set=1
    VkDescriptorPool      m_ComposePool             = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_ComposeGBufferSets{};
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_ComposeLightHdrSets{};
    VkPipelineLayout      m_ComposeLayout   = VK_NULL_HANDLE;
    VkPipeline            m_ComposePipeline = VK_NULL_HANDLE;

    // Sparse SDF residency snapshot, pushed via SetSDFView.
    bool         m_HasSDF       = false;
    VkBuffer     m_IndexBuffer  = VK_NULL_HANDLE;
    VkBuffer     m_PoolBuffer   = VK_NULL_HANDLE;
    VkDeviceSize m_IndexBytes   = 0;
    VkDeviceSize m_PoolBytes    = 0;
    glm::vec3    m_AabbMin      = glm::vec3(0.0f);
    glm::vec3    m_AabbMax      = glm::vec3(1.0f);
    float        m_MaxDist      = 1.0f;
    struct SparseParams {
        glm::vec4  AABBMin;
        glm::vec4  AABBMax;
        glm::uvec4 Dims;
    };
    static_assert(sizeof(SparseParams) == 48, "RC SparseParams pinned 48 bytes");
    SparseParams m_SparseParams{};

    // World→mesh-local transform for the relight SDF trace (inverse of the SDF
    // instance's model matrix). Identity when no SDF instance is live, which
    // is correct when the bake AABB is already world-space.
    glm::mat4 m_WorldToLocal = glm::mat4(1.0f);

    // SDF mesh albedo for the relight bounce tint (Phase 14c RGB SH).
    glm::vec3 m_SdfAlbedo = glm::vec3(0.8f);

    // Phase 14c: tracks which Readback slot was most recently submitted, so
    // DrawImGuiParams reads from the slot whose copy has actually been GPU-
    // completed by the time ImGui runs.
    uint32_t    m_LastReadbackSlot = 0;
    bool        m_HasReadback      = false;
};

} // namespace RS
