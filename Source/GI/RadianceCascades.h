// Source/GI/RadianceCascades.h — RC rewrite #2, pass 1 host
// Owns one rgba16f image3D atlas per cascade, the rc_build compute pipeline, the
// per-frame UBO rings, and the descriptor wiring. Mirrors SDFConeShadow's sparse
// residency model: the SDF is PUSHED in via SetSDF() from Main.cpp (not pulled by
// index), and the per-instance inverse-model SSBO ring is handed over the same way
// SDFConeShadow::SetInstanceXformBuffer does.
//
// Conventions: this file follows the RS struct + free-function style (see
// GlobalSDF / InstanceXformBuffer). RadianceCascades is a plain struct; every
// operation is a free function taking it by reference. VulkanContext fields are
// public (ctx.Device, ctx.PhysicalDevice) — no accessor methods.
//
// Descriptor contract (matches rc_build.comp byte-for-byte):
//   set 0 (sparse SDF, shared across cascades within a frame):
//     binding 0 = SparseIdx  SSBO  (uint BrickIndex[])
//     binding 1 = SparsePool SSBO  (uint BrickPool[])
//     binding 2 = SparseParams UBO (SparseParams: AABB + dims + MaxDist + decode)
//     binding 3 = InstanceXform SSBO (mat4 InvModel[256])
//   set 1 (cascade build, one set per cascade per frame — atlas image differs):
//     binding 0 = CascadeBuildBuf UBO (CascadeBuildParams)
//     binding 1 = LightingBuf     UBO (sun / ambient / albedo)
//     binding 2 = uCascadeAtlas   storage image3D (rgba16f, GENERAL layout)
//
// The sparse set uses bindings 0/1/2/3 (NOT the cone shader's 0/1/2/4) because RC
// owns its own set layout; the binding-4 gap in the cone shader was a Phase-13.5
// artifact and does not need to be preserved here.
#pragma once

#include "Core/VulkanContext.h"
#include "GI/CascadeSpecification.hpp"
#include "Renderer/InstanceXformBuffer.h"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

namespace RS {

struct GlobalSDF;
struct ResidentSparseSDF;

//------------------------------------------------------------------------------------------------------------------------
//                                                    TUNABLES
//------------------------------------------------------------------------------------------------------------------------

// Camera-anchored half-extent the cascades must cover (A2). Mirrors the locked
// HierarchySpecification default; restated here so RecordBuild's clipmap snap has it.
constexpr float kRcQueryRadiusMetres = 8.0f;

// Upper bound on cascade levels we allocate atlases / descriptor slots for. The
// locked spec (t0=0.25, 64 probes, R0=4) resolves to 4; this caps the std::array
// sizes. ResolveHierarchy fills only the cascades the spec actually produces.
constexpr uint32_t kMaxCascades = 4;

//------------------------------------------------------------------------------------------------------------------------
//                                                    UBO LAYOUTS
//------------------------------------------------------------------------------------------------------------------------

// std140 UBO consumed by rc_build.comp at set 1 binding 0. MUST mirror
// CascadeBuildParams in the shader exactly (three 16-byte rows = 48 bytes).
struct CascadeBuildParams {
    glm::vec4  RegionOriginWorld;   // [m]  xyz = snapped world min corner; .w = ProbePitchWorld
    glm::vec4  IntervalMetres;      // [m]  x = IntervalStart, y = IntervalEnd, z = MinStepWorld, w = ambientScalar
    glm::uvec4 GridDims;            // [-]  x = ProbeAxisCount, y = OctSide, z = CascadeIndex, w = InstanceIndex
};
static_assert(sizeof(CascadeBuildParams) == 48, "CascadeBuildParams must stay 48 bytes (std140, matches rc_build.comp)");

// std140 UBO at set 1 binding 1. Mirrors LightingBuf in the shader (4 x vec4).
struct RcLightingParams {
    glm::vec4 SunDirectionWorld;    // [-]  xyz = direction TOWARD the sun (unit); .w unused
    glm::vec4 SunColour;            // [-]  rgb = sun colour * intensity; .w unused
    glm::vec4 AmbientColour;        // [-]  rgb = flat ambient; .w unused
    glm::vec4 SurfaceAlbedo;        // [-]  rgb = constant albedo (placeholder until per-hit material); .w unused
};
static_assert(sizeof(RcLightingParams) == 64, "RcLightingParams must stay 64 bytes (std140, matches rc_build.comp)");

// std140 UBO at set 0 binding 0 of the merge pass. Mirrors MergeParamsBuf in rc_merge.comp (5 x vec4/uvec4 = 80 bytes).
struct RcMergeParams {
    glm::vec4  ProbeOriginN;      // xyz = snapped world min corner cascade N; w = probe pitch N
    glm::vec4  ProbeOriginUpper;  // xyz = snapped world min corner cascade N+1; w = probe pitch Upper
    glm::vec4  IntervalEndN;      // x = IntervalEnd N (metres); yzw unused
    glm::uvec4 DimsN;             // x = octSide N, y = probeAxis N, z = atlasWidth N, .w unused
    glm::uvec4 DimsUpper;         // x = octSide Upper, y = probeAxis Upper, z = atlasWidth Upper, .w unused
};
static_assert(sizeof(RcMergeParams) == 80, "RcMergeParams must stay 80 bytes (std140, matches rc_merge.comp)");

// std140 UBO at set 0 binding 2. Mirrors SparseSDFParams in sdf_sparse.glsl and
// SDFConeShadow::SparseParams (pinned 48 bytes) — fill it the same way.
struct RcSparseParams {
    glm::vec4  AABBMin;             // [cm] xyz = mesh-local AABB min; .w = MaxDist
    glm::vec4  AABBMax;             // [cm] xyz = mesh-local AABB max; .w = decodeScale (MaxDist/32767)
    glm::uvec4 Dims;                // [-]  x = Resolution, y = BrickSize, z = BrickGrid, w = log2(BrickSize)
};
static_assert(sizeof(RcSparseParams) == 48, "RcSparseParams must stay 48 bytes (std140, matches sdf_sparse.glsl)");

//------------------------------------------------------------------------------------------------------------------------
//                                                    OWNED RESOURCES
//------------------------------------------------------------------------------------------------------------------------

// One owned 3D atlas (rgba16f, image3D, GENERAL layout for compute store + blit src).
struct CascadeAtlas {
    VkImage        Image     = VK_NULL_HANDLE;
    VkDeviceMemory Memory    = VK_NULL_HANDLE;
    VkImageView    View      = VK_NULL_HANDLE;
    glm::uvec3     Extent    = glm::uvec3(0);                 // [px] (ProbeAxis*OctSide, same, ProbeAxis)
    VkImageLayout  LayoutNow = VK_IMAGE_LAYOUT_UNDEFINED;
};

// Per-frame-in-flight UBO ring + descriptor sets.
struct RcFrameResources {
    // CascadeBuildParams ring: kMaxCascades aligned slots in one buffer; offset selects the cascade.
    VkBuffer       CascadeBuildBuffer = VK_NULL_HANDLE;
    VkDeviceMemory CascadeBuildMemory = VK_NULL_HANDLE;
    void*          CascadeBuildMapped = nullptr;

    VkBuffer       LightingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory LightingMemory = VK_NULL_HANDLE;
    void*          LightingMapped = nullptr;

    VkBuffer       SparseParamsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory SparseParamsMemory = VK_NULL_HANDLE;
    void*          SparseParamsMapped = nullptr;

    VkDescriptorSet SparseSet = VK_NULL_HANDLE;                          // set 0 (shared across cascades this frame)
    std::array<VkDescriptorSet, kMaxCascades> CascadeSets{};            // set 1 (one per cascade, distinct atlas)

    // Merge pass: per-cascade MergeParams UBO ring + descriptor sets (one per cascade, distinct atlas pair).
    VkBuffer       MergeParamsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory MergeParamsMemory = VK_NULL_HANDLE;
    void*          MergeParamsMapped = nullptr;
    std::array<VkDescriptorSet, kMaxCascades> MergeSets{};              // set 0 (merge UBO + atlas pair per cascade)
};

//------------------------------------------------------------------------------------------------------------------------
//                                                    SUBSYSTEM STATE
//------------------------------------------------------------------------------------------------------------------------

struct RadianceCascades {
    // Resolved hierarchy (from CascadeSpecification.hpp). Only [0, CascadeCount) are valid.
    Cascades::HierarchySpecification          Spec{};
    std::array<Cascades::CascadeDescriptor, kMaxCascades> Levels{};
    uint32_t                                  CascadeCount = 0;

    std::array<CascadeAtlas, kMaxCascades>    Atlases{};

    // rc_build pipeline.
    VkPipeline            BuildPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      BuildPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout SetLayoutSparse     = VK_NULL_HANDLE;   // set 0
    VkDescriptorSetLayout SetLayoutCascade    = VK_NULL_HANDLE;   // set 1
    VkDescriptorPool      DescriptorPool      = VK_NULL_HANDLE;

    // rc_merge pipeline (Phase 2: cascade merge pass).
    VkPipeline            MergePipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      MergePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout SetLayoutMerge      = VK_NULL_HANDLE;   // set 0 (merge UBO + images)

    std::array<RcFrameResources, VulkanContext::kFramesInFlight> Frames{};
    VkDeviceSize          CascadeBuildStride  = 0;                // [B] aligned stride between cascade UBO slots
    VkDeviceSize          MergeParamsStride   = 0;                // [B] aligned stride between merge UBO slots

    // Per-frame snapped origins cached by RecordBuild for use by RecordMerge.
    std::array<glm::vec3, kMaxCascades> LastSnappedOrigins{};

    // Sparse residency snapshot (pushed via RadianceCascadesSetSDF, mirrors SDFConeShadow).
    bool         HasSDF       = false;
    VkBuffer     IndexBuffer  = VK_NULL_HANDLE;
    VkBuffer     PoolBuffer   = VK_NULL_HANDLE;
    VkDeviceSize IndexBytes   = 0;
    VkDeviceSize PoolBytes    = 0;
    RcSparseParams SparseParams{};
    uint32_t     InstanceIndex = 0;          // which InvModel[] slot the resident SDF mesh occupies

    // Dummy SSBO bound at set 0 bindings 0/1 when no sparse SDF is resident (keeps
    // descriptor writes valid; the trace never reads it because the atlas just
    // fills with misses — there is no hasSDF flag in rc_build, so a missing SDF
    // simply produces an all-miss (transmittance=1) atlas, which is correct).
    VkBuffer       DummyBuffer = VK_NULL_HANDLE;
    VkDeviceMemory DummyMemory = VK_NULL_HANDLE;

    // Per-instance inverse-model SSBO ring (owned by Main.cpp, handed over by ref).
    std::array<VkBuffer, VulkanContext::kFramesInFlight> InstXformBuffers{};
    VkDeviceSize InstXformBytes = 0;

    const VulkanContext* Ctx = nullptr;
};

//------------------------------------------------------------------------------------------------------------------------
//                                                    PUBLIC API
//------------------------------------------------------------------------------------------------------------------------

// Resolve the cascade hierarchy, create the set layouts, UBO rings, dummy SSBO,
// and the rc_build pipeline. Does NOT allocate atlas images yet — call
// RadianceCascadesConstructAtlases after, so the caller owns the command buffer
// used for the initial layout transition. Returns false on any GPU failure.
bool RadianceCascadesInitialize(RadianceCascades& rc, const VulkanContext& ctx);

// Allocate every cascade atlas, transition them UNDEFINED -> GENERAL via a
// one-shot command buffer, then write the set-1 (cascade) descriptors. Call once
// after Initialize. Returns false on failure.
bool RadianceCascadesConstructAtlases(RadianceCascades& rc, const VulkanContext& ctx);

void RadianceCascadesTerminate(RadianceCascades& rc, const VulkanContext& ctx);

// Push sparse SDF residency (mirrors SDFConeShadow::SetSDF). Rewrites the set-0
// descriptors across all frames under vkDeviceWaitIdle. Pass nullptr/false to
// unbind (atlas then fills all-miss). `instanceIndex` is the InvModel[] slot of
// the resident mesh, written into CascadeBuildParams.GridDims.w each frame.
void RadianceCascadesSetSDF(RadianceCascades& rc, const ResidentSparseSDF* sparse,
                            bool hasSDF, uint32_t instanceIndex);

// Hand over the per-instance inverse-model SSBO ring (mirrors
// SDFConeShadow::SetInstanceXformBuffer). Rewrites the set-0 binding-3 across all frames.
void RadianceCascadesSetInstanceXformBuffer(RadianceCascades& rc,
                                            const VkBuffer* ssbosByFrame,
                                            VkDeviceSize bytesPerSlot);

// Record pass 1 for the active frame: fill UBOs from camera/sun, dispatch rc_build
// once per cascade (flattened-atlas grid), then a write->read barrier so a later
// debug/merge pass can sample the atlases.
void RadianceCascadesRecordBuild(RadianceCascades& rc,
                                 VkCommandBuffer cmd,
                                 uint32_t frameSlot,
                                 const glm::vec3& cameraEyeWorld,
                                 const glm::vec3& sunDirectionWorld,
                                 const glm::vec3& sunColour);

// Record pass 2 for the active frame: merge cascades from C-2 down to 0, compositing
// the upper cascade's far-field radiance into each lower cascade via software trilinear
// parallax interpolation. Must be called after RadianceCascadesRecordBuild in the same
// command buffer.
void RadianceCascadesRecordMerge(RadianceCascades& rc,
                                 VkCommandBuffer cmd,
                                 uint32_t frameSlot);

// Debug: blit one Z-slice of a cascade atlas into a destination colour image. Proves
// the atlas has non-zero texels independent of any merge/compose logic. Note:
// blitting rgba16f HDR to an LDR target clips/looks dim — black here but light in
// a later merge means suspect the blit target's format, not the trace.
void RadianceCascadesRecordDebugSlice(RadianceCascades& rc,
                                      VkCommandBuffer cmd,
                                      VkImage destination,
                                      VkImageLayout destinationLayout,
                                      uint32_t destinationWidth,
                                      uint32_t destinationHeight,
                                      uint32_t cascadeIndex,
                                      uint32_t sliceZ);

} // namespace RS
