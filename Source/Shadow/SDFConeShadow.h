// Source/Shadow/SDFConeShadow.h — Phase 13 (+ Phase 15d sparse rewire)
// IShadowAlgorithm impl that does NOT render to a shadow atlas. The lighting
// compose itself does the cone-trace (improved Iñigo Quilez soft-shadow) via
// the sparse-VDB SDF sampler (Phase 15c brick-aware ray helper) so empty
// space crosses in one step.
//
// Descriptor contract at set=2 for the SDFCone variant (Phase 15d):
//   binding 0 = SparseIdx  SSBO  (uint BrickIndex[BrickGrid^3])
//   binding 1 = SparsePool SSBO  (uint BrickPool[occupied*bs^3/2])
//   binding 2 = SparseParams UBO (SparseSDFParams: AABB + dims + MaxDist + decode)
//   binding 3 = AlgoUbo          (48 bytes — sun, trace params, surface off, flags)
//   binding 4 = InstanceXform SSBO (mat4 InverseModel[256]) — Phase 13.5
//
// The sparse storage at bindings 0/1/2 tracks the GlobalSDF::ResidentSparse
// table: `SetSDF` is called from Main.cpp's residency-changed branch the same
// way the old dense path was. We pre-allocate kFramesInFlight sets and rewrite
// all of them under vkDeviceWaitIdle so no in-flight frame samples stale data.
//
// **Why mesh-local trace stays.** Meshes can move at runtime. Re-baking on
// every move is infeasible; the right primitive is one bake per mesh + a
// per-instance world->mesh-local transform applied in the shader. The C++
// invModel SSBO already exists from Phase 13.5; sparse just swaps the sampler.
#pragma once

#include "Shadow/IShadowAlgorithm.h"

#include <array>
#include <glm/glm.hpp>

#include "Core/VulkanContext.h"

namespace RS {

struct GlobalSDF;
struct ResidentSparseSDF;

struct SDFConeParams {
    bool  Enabled        = true;
    float HalfAngleDeg   = 1.5f;     // sun-disc half-angle approximation
    float MaxDistance    = 8.0f;     // world units along the ray
    float SurfaceOffset  = 0.02f;    // push start past the surface along N
    int   MaxSteps       = 64;       // sphere-trace iteration cap
    float MinStep        = 0.002f;   // floor on per-step march to avoid stalls
    float Strength       = 1.0f;     // shadow intensity (1 = full block, 0 = no shadow)
};

class SDFConeShadow final : public IShadowAlgorithm {
public:
    SDFConeShadow()  = default;
    ~SDFConeShadow() = default;

    const char* Name() const override { return "SDF Cone"; }
    uint32_t    AlgoVariant() const override { return 0; }

    void Initialize(const VulkanContext& ctx,
                    VkFormat depthFormat,
                    VkExtent2D shadowMapResolution) override;
    void Terminate (const VulkanContext& ctx) override;

    void DrawImGuiParams() override;
    void RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) override;

    VkDescriptorSetLayout LightingSetLayout() const override { return m_LightingSetLayout; }
    void BindLightingDescriptorSet(VkCommandBuffer cmd,
                                   VkPipelineLayout lightingLayout,
                                   uint32_t frameSlot,
                                   uint32_t set) override;

    // Phase 15d: hook the shadow algo onto the sparse SDF residency. Called
    // from Main.cpp at boot + every time GlobalSDF residency changes for the
    // tracked mesh. Re-writes all kFramesInFlight descriptor sets under
    // vkDeviceWaitIdle so no in-flight frame can sample dangling SSBOs.
    // Pass nullptr / hasSDF=false to bind a dummy (lights stay full-bright).
    void SetSDF(const ResidentSparseSDF* sparse, bool hasSDF);

    // Phase 13.5: per-instance inverse-model SSBO ring (one per frame-in-flight).
    // Bound at set=2 binding 4. Main.cpp wires this once after Initialize.
    void SetInstanceXformBuffer(const VkBuffer* ssbosByFrame, VkDeviceSize bytesPerSlot);

    bool HasSDF() const { return m_HasSDF; }

private:
    bool CreateSetLayout();
    bool CreateAlgoUbos();
    bool CreateSparseParamsUbos();
    bool CreateDummySparseBuffers();
    bool CreateDescriptorSets();
    void RewriteAllSets();
    void UpdateAlgoUbo(uint32_t frameSlot, const FrameContext& frame);
    void UpdateSparseParamsUbo(uint32_t frameSlot);

    const VulkanContext* m_Ctx = nullptr;
    SDFConeParams        m_Params{};

    // Sparse residency snapshot, updated only via SetSDF().
    bool       m_HasSDF        = false;
    VkBuffer   m_IndexBuffer   = VK_NULL_HANDLE;
    VkBuffer   m_PoolBuffer    = VK_NULL_HANDLE;
    VkDeviceSize m_IndexBytes  = 0;
    VkDeviceSize m_PoolBytes   = 0;

    // SparseParams UBO contents (mirrors SparseSDFParams in sdf_sparse.glsl).
    // Refreshed by UpdateSparseParamsUbo on residency change.
    struct SparseParams {
        glm::vec4  AABBMin;   // xyz + MaxDist in .w
        glm::vec4  AABBMax;   // xyz + decodeScale in .w
        glm::uvec4 Dims;      // x=Resolution y=BrickSize z=BrickGrid w=log2(BrickSize)
    };
    static_assert(sizeof(SparseParams) == 48, "SDFCone SparseParams pinned 48 bytes");
    SparseParams m_SparseParams{};

    // Dummy SSBOs (one 4-byte each) bound when no sparse SDF is resident, so
    // descriptor writes never see VK_NULL_HANDLE storage buffers. The shader
    // never reads them because hasSDF=false short-circuits the trace.
    VkBuffer        m_DummyBuffer = VK_NULL_HANDLE;
    VkDeviceMemory  m_DummyMemory = VK_NULL_HANDLE;

    // Per-frame AlgoUbo (host-visible coherent).
    struct AlgoUbo {
        glm::vec4 SunDirAndHalfAngle;     // xyz = direction TOWARD sun, w = half-angle (rad)
        glm::vec4 TraceParamsAndSurfOff;  // x = maxSteps, y = maxDistance, z = minStep, w = surfaceOffset
        glm::vec4 StrengthAndFlags;       // x = strength, y = enabled, z = hasSDF, w = 0
    };
    static_assert(sizeof(AlgoUbo) == 48, "SDFConeShadow AlgoUbo pinned at 48 bytes");

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_AlgoUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_AlgoUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_AlgoUboMapped{};

    // Per-frame SparseParams UBO (host-visible coherent). Updated on SetSDF().
    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_SparseUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_SparseUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_SparseUboMapped{};

    // Per-instance inverse-model SSBO ring (owned by Main.cpp).
    std::array<VkBuffer, VulkanContext::kFramesInFlight> m_InstXformBuffers{};
    VkDeviceSize                                         m_XformBytes = 0;

    VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescriptorPool    = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_FrameSets{};
};

} // namespace RS
