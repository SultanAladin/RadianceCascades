// Source/Shadow/SDFConeShadow.h — Phase 13
// IShadowAlgorithm impl that does NOT render to a shadow atlas. The lighting
// compose itself does the cone-trace (improved Iñigo Quilez soft-shadow,
// h*h/(2*y) penumbra tracking — eliminates the banding the classic k*h/t form
// shows at grazing angles, and ramps the penumbra width proportional to
// distance which is what we want for soft sun shadows).
//
// Descriptor contract at set=2 for the SDFCone variant:
//   binding 0 = sampler3D R16_SNORM (the resident mesh SDF — fallback dummy
//               when no mesh is resident, so the binding is always safe)
//   binding 1 = AlgoUbo (64 bytes — SunDir, AABBMin/Max+decode, trace params)
//
// The view bound at binding 0 has to track the GlobalSDF residency table:
// `Set` is called from Main.cpp's residency-changed branch (the same spot
// that calls GBufferPreviewSetSDF). We pre-allocate kFramesInFlight sets and
// rewrite all of them with a vkDeviceWaitIdle barrier so no in-flight frame
// reads a stale image view.
#pragma once

#include "Shadow/IShadowAlgorithm.h"

#include <array>
#include <glm/glm.hpp>

#include "Core/VulkanContext.h"

namespace RS {

struct GlobalSDF;

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
    uint32_t    AlgoVariant() const override { return 3; }

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

    // Phase 13: hook the shadow algo onto the resident SDF. Called from
    // Main.cpp at boot + every time GlobalSDF residency changes for the
    // tracked mesh. Re-writes all kFramesInFlight descriptor sets under
    // vkDeviceWaitIdle so no in-flight frame can sample a dangling view.
    void SetSDF(VkImageView sdfView, VkSampler sampler,
                const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                float maxDist, bool hasSDF);

    // Latest-known residency state. SetSDF stores into these so RecordShadowPass
    // can stamp the UBO without the caller passing the geometry every frame.
    bool HasSDF() const { return m_HasSDF; }

private:
    bool CreateSampler();
    bool CreateSetLayout();
    bool CreateAlgoUbos();
    bool CreateDescriptorSets();
    void RewriteAllSets();
    void UpdateAlgoUbo(uint32_t frameSlot, const FrameContext& frame);

    const VulkanContext* m_Ctx = nullptr;
    SDFConeParams        m_Params{};

    // Linear/clamp sampler. The SDF image itself comes from GlobalSDF, but the
    // sampler lives here so we don't have to ferry it through. Matches
    // GlobalSDF::Sampler's filter/clamp settings.
    VkSampler m_LinearClampSampler = VK_NULL_HANDLE;

    // Residency snapshot — updated only via SetSDF().
    VkImageView m_SdfView   = VK_NULL_HANDLE;
    VkSampler   m_SdfSampler= VK_NULL_HANDLE;
    glm::vec3   m_AabbMin   = glm::vec3(0.0f);
    glm::vec3   m_AabbMax   = glm::vec3(1.0f);
    float       m_MaxDist   = 1.0f;
    bool        m_HasSDF    = false;

    // Per-frame algo UBO (host-visible coherent — matches PCFShadow's pattern).
    // 64 bytes — well under the 256-byte budget; lighting_sdfcone reads it at
    // set=2 binding=1.
    struct AlgoUbo {
        glm::vec4 SunDirAndHalfAngle;   // xyz = direction TOWARD sun, w = half-angle (rad)
        glm::vec4 AABBMinAndMaxDist;    // xyz = AABB min, w = R16_SNORM decode scale
        glm::vec4 AABBMaxAndSurfaceOff; // xyz = AABB max, w = surface offset (along N)
        glm::vec4 TraceParams;          // x = maxSteps, y = maxDist, z = minStep, w = strength
        glm::vec4 Flags;                // x = enabled, y = hasSDF, z = padding, w = padding
    };
    static_assert(sizeof(AlgoUbo) == 80, "SDFConeShadow AlgoUbo pinned at 80 bytes");

    std::array<VkBuffer,       VulkanContext::kFramesInFlight> m_AlgoUboBuffers{};
    std::array<VkDeviceMemory, VulkanContext::kFramesInFlight> m_AlgoUboMemory{};
    std::array<void*,          VulkanContext::kFramesInFlight> m_AlgoUboMapped{};

    VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescriptorPool    = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, VulkanContext::kFramesInFlight> m_FrameSets{};
};

} // namespace RS
