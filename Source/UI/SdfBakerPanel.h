// Source/UI/SdfBakerPanel.h — Phase 12
// Dockable ImGui window that drives the SDF baking workflow:
//   - target selector (which mesh)
//   - resolution combo (32 / 64 / 96 / 128, default 64)
//   - algorithm combo (only Exact BVH brute-force enabled in v1)
//   - Bake / Re-bake / Load .rsdf / Evict buttons + an async progress bar
//   - slice preview (Y-slider + CPU-rasterised ImGui::Image)
//   - main-viewport SDF-slice mode shares the same Y-plane setting
//
// The baker runs in a worker std::thread; the main thread polls progress and
// uploads the resulting BakedSDF to the GlobalSDF table on completion.
#pragma once

#include "Core/VulkanContext.h"
#include "Renderer/GBufferPreview.h"   // SDFSliceState
#include "SDF/SDFCache.h"
#include "SDF/GlobalSDF.h"
#include "SDF/MeshSDFBaker.h"
#include "RS/Scene.h"
#include "Scene/MeshRegistry.h"

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>
#include "imgui.h"

namespace RS {

struct SdfBakerState {
    // User-facing settings
    int      ResolutionChoice = 1;   // index into kResolutions = {32, 64, 96, 128}
    int      AlgorithmChoice  = 0;   // 0 = Exact BVH (only enabled option)
    int      VizModeChoice    = 0;   // see SDFVizMode (0..4)
    bool     ShowSlicePreview = true;
    float    SlicePlaneY01    = 0.5f; // 0..1 across the SDF AABB

    // Per-mesh state. Phase 12 only has ShaderBall so a flat vector is plenty.
    MeshHandle TargetMesh = 0;       // resolved from picked instance / dropdown
    const char* TargetSourcePath = nullptr;  // OBJ path, owned by Main.cpp

    // Phase 12.5: editable output target. Defaults derive from the mesh source
    // path on first Bake (Cache/SDF + <basename>). The text buffers are owned
    // by the panel so ImGui::InputText can mutate them in place.
    bool                    OutputCustomised = false;
    std::array<char, 260>   OutputDirBuf{};   // null-terminated
    std::array<char, 96>    OutputNameBuf{};  // basename without _<res>.rsdf

    // Async bake state
    bool                          BakeInFlight    = false;
    std::shared_ptr<BakeProgress> Progress;            // shared with worker
    std::thread                   Worker;
    std::shared_ptr<BakedSDF>     PendingUpload;       // worker writes, main reads
    std::atomic<bool>             WorkerDone { false };

    // CPU slice preview (regenerated when the slider moves or a fresh bake
    // lands). RGBA8 128x128, uploaded into an ImGui texture handle.
    bool                  SliceDirty = true;
    std::vector<uint32_t> SliceRGBA;
    uint32_t              SliceResolution = 128;
    // ImGui sees a sampled-image descriptor. We allocate one per-bake from
    // imgui_impl_vulkan via ImGui_ImplVulkan_AddTexture; the previous handle
    // is released on the next regeneration.
    VkSampler             SlicePreviewSampler = VK_NULL_HANDLE;
    VkImage               SlicePreviewImage   = VK_NULL_HANDLE;
    VkDeviceMemory        SlicePreviewMemory  = VK_NULL_HANDLE;
    VkImageView           SlicePreviewView    = VK_NULL_HANDLE;
    ImTextureID           SlicePreviewTex     = 0;

    // Cached copy of the most recent successful bake — drives the slice
    // preview without round-tripping through GPU readback. Cleared on Evict.
    std::shared_ptr<BakedSDF> ResidentBakedCopy;

    // Phase 12.5 stats. Updated on bake completion / Load / Evict.
    struct LastBakeStats {
        bool     Valid          = false;
        bool     FromCache      = false;
        uint32_t Resolution     = 0;
        uint64_t VoxelCount     = 0;
        uint64_t CpuBytes       = 0;        // sizeof(int16) × voxelCount
        uint64_t GpuBytes       = 0;        // R16_SNORM × voxelCount (+small overhead)
        uint64_t DiskBytes      = 0;        // on-disk file size, 0 if not saved
        double   BakeDurationMs = 0.0;
        uint32_t TriangleCount  = 0;
        float    MaxDist        = 0.0f;
        std::string LastSavedPath;          // last successful save destination
    } Stats;
};

bool SdfBakerInitialize(SdfBakerState& s, const VulkanContext& ctx);
void SdfBakerTerminate (SdfBakerState& s, const VulkanContext& ctx);

// Pump the panel + the async upload step. Call once per frame from Main.cpp
// after Vulkan BeginFrame but before any render-pass recording (so that an
// upload — which submits a one-shot cmd + waits idle — can land before the
// frame is in flight). Returns true if the GlobalSDF residency changed this
// frame (so callers can refresh the GBufferPreview SDF binding).
bool SdfBakerPanelDrawAndPump(SdfBakerState& s,
                              const VulkanContext& ctx,
                              GlobalSDF& globalSdf,
                              const Scene& scene);

// Translate panel resolution choice → cube resolution.
uint32_t SdfBakerResolutionPx(const SdfBakerState& s);

// Build the SDFSliceState that the GBufferPreview record consumes from the
// current panel state + the currently-resident SDF (if any).
SDFSliceState SdfBakerComposeSliceState(const SdfBakerState& s,
                                        const GlobalSDF& globalSdf);

} // namespace RS
