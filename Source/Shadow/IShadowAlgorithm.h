// Source/Shadow/IShadowAlgorithm.h
// The plugin pattern's prototype. Every shadow algorithm implements this; the
// renderer holds std::unique_ptr<IShadowAlgorithm> and swaps on enum change.
// Lighting compute shader binds to a fixed descriptor contract — see
// `Source/Shadow/PCFShadow.cpp` and `Shaders/lighting.comp` for the layout.
//
// Phase 10 wiring:
//   * Initialize takes the VulkanContext, the depth format CSM should bake at
//     (PCF/PCSS/VSM use D32_SFLOAT, SDFCone ignores), and a shadow-map size
//     (per-cascade-layer side length).
//   * RecordShadowPass is called between GBufferPassRecord and Lighting compose;
//     the algo writes its result into images it owns.
//   * The lighting compose binds the algo's set=2 just before vkCmdDispatch.
//     The algo guarantees:
//       - set=2 binding=0 = result image (sampler — type varies by variant)
//       - set=2 binding=1 = 256-byte UBO with algo-private constants
//     Lighting holds a pipeline-layout-compatible empty set layout for set=2
//     when no algo is active.
//   * AlgoVariant() returns the spec-constant id (0..3) the lighting compute
//     was compiled against. Renderer compares this to the lighting pipeline's
//     baked variant and rebuilds on mismatch.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace RS {

struct VulkanContext;
struct FrameContext;

struct IShadowAlgorithm {
    virtual ~IShadowAlgorithm() = default;

    virtual const char* Name() const = 0;

    // Specialisation-constant id baked into the lighting pipeline.
    //   0 = PCF  (sampler2DArrayShadow + per-cascade UV)
    //   1 = PCSS (same input, different filter)
    //   2 = VSM  (sampler2DArray RG32F moments)
    //   3 = SDFCone (sampler2D R8 occlusion at full res)
    virtual uint32_t AlgoVariant() const = 0;

    virtual void Initialize(const VulkanContext& ctx,
                            VkFormat depthFormat,
                            VkExtent2D shadowMapResolution) = 0;
    virtual void Terminate(const VulkanContext& ctx) = 0;

    // Algo owns its params struct internally and exposes it through ImGui.
    virtual void DrawImGuiParams() = 0;

    // Called between GBuffer and Lighting. Algo records its own passes; it
    // also updates its private UBO with anything dependent on this frame's
    // sun direction / camera split planes.
    virtual void RecordShadowPass(VkCommandBuffer cmd, const FrameContext& frame) = 0;

    // Descriptor-set layout for the lighting compute's set=2. The renderer
    // queries this once at Initialize so the lighting pipeline layout can be
    // built. Lifetime: owned by the algo, valid until Terminate().
    virtual VkDescriptorSetLayout LightingSetLayout() const = 0;

    // Algo writes its result into the lighting shader's set=2 by binding the
    // per-frame descriptor set the algo allocated and updated.
    virtual void BindLightingDescriptorSet(VkCommandBuffer cmd,
                                           VkPipelineLayout lightingLayout,
                                           uint32_t frameSlot,
                                           uint32_t set) = 0;
};

} // namespace RS
