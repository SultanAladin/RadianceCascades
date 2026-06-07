// Application/Main.cpp â€” Phase 10
// Phase 9 added the multi-instance grid + per-submesh material binding + click
// picking. Phase 10 adds the shadow plugin chain (IShadowAlgorithm) + the
// compute lighting pass. LightingPass composes Karis/Schlick/Smith + IBL +
// SDF cone shadows into
// LightHDR; the GBufferPreview Lit mode samples that. Frame ordering is now
// Sky â†’ Grid â†’ GBuffer â†’ PickingCopy â†’ Shadow â†’ Lighting â†’ Preview â†’ ImGui.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "Win32Host.h"
#include "Core/VulkanContext.h"
#include "Core/Logger.h"
#include "RS/Renderer.h"
#include "RS/Camera.h"
#include "RS/Material.h"
#include "RS/Scene.h"
#include "Renderer/GridPass.h"
#include "Renderer/GBuffer.h"
#include "Renderer/GBufferPreview.h"
#include "Renderer/Lighting.h"
#include "Renderer/Tonemap.h"
#include "Renderer/PerfTimers.h"
#include "Renderer/OffscreenTargets.h"
#include "Renderer/Picking.h"
#include "Renderer/InstanceXformBuffer.h"
#include "Renderer/SkyAtmosphere.h"
#include "Renderer/FrameContext.h"
#include "Shadow/IShadowAlgorithm.h"
#include "Shadow/SDFConeShadow.h"
#include "GI/RadianceCascades.h"
#include "RS/RenderSettings.h"
#include "SDF/GlobalSDF.h"
#include "Material/MaterialSeed.h"
#include "Scene/FloorMesh.h"
#include "Scene/ScaleRefSphere.h"
#include "Scene/SceneInternal.h"
#include "UI/ImGuiContextRS.h"
#include "UI/ImGuiPanel.h"
#include "UI/RenderSettingsPanel.h"
#include "UI/SdfBakerPanel.h"
#include "UI/PerfWidget.h"
#include "imgui.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <memory>
#include <vector>

namespace {

constexpr const char* kShaderBallPath =
    "../ShaderBall-master/shaderBallNoCrease/shaderBall.obj";

glm::mat4 FitMeshOnGrid(const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    constexpr float kCmToMeters = 0.01f;
    const glm::vec3 centerCm   = 0.5f * (aabbMin + aabbMax);
    glm::vec3 anchor = centerCm * kCmToMeters;
    anchor.y = aabbMin.y * kCmToMeters;
    return glm::translate(glm::mat4(1.0f), -anchor) *
           glm::scale    (glm::mat4(1.0f), glm::vec3(kCmToMeters));
}

void DrawSunPanel(RS::GBufferPreviewSettings& preview,
                  RS::SkySettings& sky) {
    ImGui::SetNextWindowPos (ImVec2(16.0f, 470.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 230.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Sun / IBL")) {
        ImGui::TextUnformatted("Sun");
        ImGui::SliderFloat3("Direction", &preview.SunDirection.x, -1.0f, 1.0f);
        ImGui::SliderFloat ("Intensity", &preview.SunIntensity,    0.0f, 4.0f);
        ImGui::SliderFloat ("Ambient",   &preview.Ambient,         0.0f, 1.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("IBL");
        bool ibl = preview.UseIBL;
        if (ImGui::Checkbox("Enable IBL", &ibl)) {
            preview.UseIBL = ibl;
            sky.EnableIBL  = ibl;
        }
        if (ImGui::SliderFloat("IBL intensity", &preview.IBLIntensity, 0.0f, 4.0f)) {
            sky.IBLIntensity = preview.IBLIntensity;
        }
        ImGui::TextDisabled("Material sliders moved to the Materials panel.");
    }
    ImGui::End();
}

void DrawSkyPanel(RS::SkySettings& s) {
    ImGui::SetNextWindowPos (ImVec2(640.0f, 470.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 220.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Sky")) {
        ImGui::TextUnformatted("Sky inscatter");
        ImGui::SliderFloat("Sky intensity", &s.SkyIntensity, 0.0f, 4.0f);
        ImGui::TextDisabled("Master gain on scattered sky radiance (decoupled\n"
                            "from sun intensity so IBL stays balanced).");

        ImGui::Separator();
        ImGui::TextUnformatted("Sun disc (in sky cube)");
        ImGui::ColorEdit3 ("Sun colour",     &s.SunColor.x);
        ImGui::SliderFloat("Sun intensity",  &s.SunIntensity,    0.0f, 32.0f);
        ImGui::SliderFloat("Sun angular Â°",  &s.SunAngularSizeDeg, 0.1f, 8.0f);
        ImGui::TextDisabled("Edits re-bake IBL cubemaps (~sub-ms each).");
    }
    ImGui::End();
}

void DrawShadowsPanel(RS::IShadowAlgorithm& active) {
    ImGui::SetNextWindowPos (ImVec2(640.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 320.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Shadows")) {
        ImGui::TextUnformatted("SDF Cone");
        ImGui::TextDisabled("Cone-trace soft shadows against the active mesh SDF â€”\n"
                            "bake one in the SDF Baker panel first.");
        ImGui::Separator();
        active.DrawImGuiParams();
    }
    ImGui::End();
}

void DrawTonemapPanel(RS::TonemapSettings& t) {
    ImGui::SetNextWindowPos (ImVec2(16.0f, 710.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 140.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Tonemap")) {
        ImGui::SliderFloat("Exposure (EV)", &t.ExposureEV, -4.0f, 4.0f);
        ImGui::Checkbox("ACES (Narkowicz)",  &t.AcesEnabled);
        ImGui::SameLine();
        ImGui::Checkbox("sRGB gamma",        &t.GammaEnabled);
        ImGui::TextDisabled("In-place on LightHDR. Swapchain is _UNORM, so the\n"
                            "sRGB encode lives here (not in the framebuffer).");
    }
    ImGui::End();
}

struct RcDebugSettings {
    bool BuildEnabled   = true;
    bool ShowDebugSlice = false;
    int  CascadeIndex   = 0;
    int  SliceZ         = 0;
};

void DrawDebugPanel(RS::GBufferPreviewSettings& preview,
                    RcDebugSettings& rcDebug,
                    const RS::RadianceCascades* rcGi) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 170.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0f, 250.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Debug")) {
        static const char* kModeLabels[] = {
            "Lit preview", "Albedo", "World normal", "Rough+Metal+F0",
            "Emissive", "Depth (linearised)", "Identity (colourised)",
            "Matcap", "SDF Slice"
        };
        int mode = static_cast<int>(preview.Mode);
        if (ImGui::Combo("GBuffer view", &mode, kModeLabels, IM_ARRAYSIZE(kModeLabels))) {
            preview.Mode = static_cast<RS::GBufferPreviewMode>(mode);
        }
        ImGui::TextDisabled("Choose which GBuffer channel paints over the grid.");

        ImGui::Separator();
        ImGui::TextUnformatted("Radiance Cascades");
        if (!rcGi || rcGi->CascadeCount == 0) {
            bool disabled = false;
            ImGui::BeginDisabled();
            ImGui::Checkbox("Build RC atlas", &disabled);
            ImGui::Checkbox("Show RC atlas slice", &disabled);
            ImGui::SliderInt("Cascade", &rcDebug.CascadeIndex, 0, 0);
            ImGui::SliderInt("Slice Z", &rcDebug.SliceZ, 0, 0);
            ImGui::EndDisabled();
            ImGui::TextDisabled("RC resource allocation failed; see log.");
        } else {
            const int maxCascade = static_cast<int>(rcGi->CascadeCount) - 1;
            rcDebug.CascadeIndex = std::clamp(rcDebug.CascadeIndex, 0, maxCascade);

            const RS::CascadeAtlas& atlas =
                rcGi->Atlases[static_cast<uint32_t>(rcDebug.CascadeIndex)];
            const int maxSlice = atlas.Extent.z > 0
                ? static_cast<int>(atlas.Extent.z) - 1
                : 0;
            rcDebug.SliceZ = std::clamp(rcDebug.SliceZ, 0, maxSlice);

            ImGui::Checkbox("Build RC atlas", &rcDebug.BuildEnabled);
            ImGui::Checkbox("Show RC atlas slice", &rcDebug.ShowDebugSlice);
            ImGui::BeginDisabled(!rcDebug.ShowDebugSlice);
            if (ImGui::SliderInt("Cascade", &rcDebug.CascadeIndex, 0, maxCascade)) {
                rcDebug.CascadeIndex = std::clamp(rcDebug.CascadeIndex, 0, maxCascade);
            }
            const RS::CascadeAtlas& selected =
                rcGi->Atlases[static_cast<uint32_t>(rcDebug.CascadeIndex)];
            const int selectedMaxSlice = selected.Extent.z > 0
                ? static_cast<int>(selected.Extent.z) - 1
                : 0;
            rcDebug.SliceZ = std::clamp(rcDebug.SliceZ, 0, selectedMaxSlice);
            ImGui::SliderInt("Slice Z", &rcDebug.SliceZ, 0, selectedMaxSlice);
            ImGui::EndDisabled();
            ImGui::Text("Atlas: %ux%ux%u",
                        selected.Extent.x, selected.Extent.y, selected.Extent.z);
            ImGui::Text("SDF: %s", rcGi->HasSDF ? "resident" : "none");
        }
    }
    ImGui::End();
}

void DrawMatcapPanel(RS::MatcapSettings& m) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 300.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 360.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Matcap")) {
        ImGui::TextUnformatted("Gradient");
        ImGui::ColorEdit3 ("Top",            &m.ColorTop.x);
        ImGui::ColorEdit3 ("Bottom",         &m.ColorBottom.x);
        ImGui::SliderFloat("Gradient curve", &m.GradientCurve, 0.1f, 6.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("Rim");
        ImGui::ColorEdit3 ("Rim colour",   &m.RimColor.x);
        ImGui::SliderFloat("Rim power",    &m.RimPower,    0.5f, 12.0f);
        ImGui::SliderFloat("Rim strength", &m.RimStrength, 0.0f,  3.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("Highlight");
        ImGui::ColorEdit3 ("Hi colour",   &m.HighlightColor.x);
        ImGui::SliderFloat2("Hi offset",  &m.HighlightOffset.x, -1.0f, 1.0f);
        ImGui::SliderFloat("Hi size",     &m.HighlightSize,     0.01f, 1.0f);
        ImGui::SliderFloat("Hi strength", &m.HighlightStrength, 0.0f,  3.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("Material");
        ImGui::SliderFloat("Roughness", &m.Roughness, 0.04f, 1.0f);
        ImGui::SliderFloat("Metallic",  &m.Metallic,  0.0f,  1.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("Output");
        ImGui::ColorEdit3 ("Tint",     &m.Tint.x);
        ImGui::SliderFloat("Exposure", &m.Exposure, 0.0f, 4.0f);
        ImGui::TextDisabled("Switch Debug â†’ Matcap to view.");
    }
    ImGui::End();
}

// Build the per-instance transform for cell (row, col) in an N x N square
// grid centred at the world origin, sitting on the ground plane. Cell pitch
// scales with the mesh footprint so the balls don't intersect.
glm::mat4 CellTransform(int row, int col, int n,
                        const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    const glm::mat4 base = FitMeshOnGrid(aabbMin, aabbMax);

    constexpr float kCmToMeters = 0.01f;
    const glm::vec3 spanCm = aabbMax - aabbMin;
    const float footprint  = std::max(spanCm.x, spanCm.z) * kCmToMeters;
    const float pitch      = footprint * 1.20f;          // 20% gap between balls
    const float offset     = -0.5f * pitch * (static_cast<float>(n) - 1.0f);
    const float x = offset + pitch * static_cast<float>(col);
    const float z = offset + pitch * static_cast<float>(row);
    return glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, z)) * base;
}

// Diff the requested grid against the slot table. Slot (r, c) keeps its
// instance handle (and therefore its material bindings) if (r, c) survives
// the resize; cells outside the new range are destroyed, cells newly inside
// are inscribed. `forceReseed` clears everything first â€” the "Reload
// ShaderBall" button uses it.
void RebuildShaderBallGrid(RS::Scene& scene, std::vector<RS::InstanceHandle>& slots,
                           int& currentN, int requestedN,
                           RS::MeshHandle mesh, const glm::vec3& aabbMin,
                           const glm::vec3& aabbMax, bool forceReseed) {
    if (mesh == 0 || requestedN <= 0) return;

    if (forceReseed) {
        for (RS::InstanceHandle h : slots) {
            if (h != 0) scene.DestroyInstance(h);
        }
        slots.clear();
        currentN = 0;
    }
    if (requestedN == currentN && !forceReseed) return;

    const int oldN = currentN;
    std::vector<RS::InstanceHandle> next(static_cast<size_t>(requestedN * requestedN), 0u);

    // Carry over instances that still fit. Anything outside the new range
    // gets destroyed; the rest is re-indexed into row-major order so the
    // (row, col) -> InstanceHandle relationship survives a resize.
    for (int r = 0; r < oldN; ++r) {
        for (int c = 0; c < oldN; ++c) {
            const size_t srcIdx = static_cast<size_t>(r * oldN + c);
            const RS::InstanceHandle h = (srcIdx < slots.size()) ? slots[srcIdx] : 0u;
            if (h == 0) continue;
            if (r < requestedN && c < requestedN) {
                next[static_cast<size_t>(r * requestedN + c)] = h;
            } else {
                scene.DestroyInstance(h);
            }
        }
    }

    // Inscribe the new cells.
    for (int r = 0; r < requestedN; ++r) {
        for (int c = 0; c < requestedN; ++c) {
            const size_t dstIdx = static_cast<size_t>(r * requestedN + c);
            if (next[dstIdx] != 0) continue;
            next[dstIdx] = scene.InscribeInstance({
                mesh, CellTransform(r, c, requestedN, aabbMin, aabbMax)
            });
        }
    }

    slots     = std::move(next);
    currentN  = requestedN;
}

} // namespace

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE /*hPrev*/,
                     LPSTR /*lpCmdLine*/, int /*nCmdShow*/) {
    constexpr uint32_t kWidth  = 1280;
    constexpr uint32_t kHeight = 720;

    RS::Win32Host window;
    if (!window.Create(hInstance, L"Rendering Subsystem â€” Phase 9", kWidth, kHeight)) {
        RS_LOG_ERROR("Win32Host::Create failed");
        return 1;
    }

#ifdef NDEBUG
    const bool enableValidation = false;
#else
    const bool enableValidation = true;
#endif

    RS::VulkanContext vk;
    if (!vk.Initialize(hInstance, window.WindowHandle(), kWidth, kHeight, enableValidation)) {
        RS_LOG_ERROR("VulkanContext::Initialize failed");
        return 2;
    }

    RS::ImGuiContextRS gui;
    if (!RS::ImGuiContextRSInitialize(gui, vk, window.WindowHandle())) {
        RS_LOG_ERROR("ImGuiContextRSInitialize failed");
        return 3;
    }

    RS::SkyAtmosphere sky;
    if (!RS::SkyAtmosphereInitialize(sky, vk, "Artifacts/Shaders")) {
        RS_LOG_ERROR("SkyAtmosphereInitialize failed");
        return 4;
    }

    RS::GridPass grid;
    if (!RS::GridPassInitialize(grid, vk, sky, "Artifacts/Shaders")) {
        RS_LOG_ERROR("GridPassInitialize failed");
        return 5;
    }

    RS::OffscreenTargets targets;
    if (!RS::OffscreenTargetsInitialize(targets, vk)) {
        RS_LOG_ERROR("OffscreenTargetsInitialize failed");
        return 6;
    }

    RS::GBufferPass gbuffer;
    if (!RS::GBufferPassInitialize(gbuffer, vk, targets, "Artifacts/Shaders")) {
        RS_LOG_ERROR("GBufferPassInitialize failed");
        return 7;
    }

    // Phase 12: GlobalSDF must initialize before the preview so the preview
    // can bind the dummy 1^3 SDF as a safe fallback at startup.
    RS::GlobalSDF globalSdf;
    if (!RS::GlobalSDFInitialize(globalSdf, vk)) {
        RS_LOG_ERROR("GlobalSDFInitialize failed");
        return 7;
    }

    RS::GBufferPreviewPass preview;
    if (!RS::GBufferPreviewInitialize(preview, vk, targets, sky,
                                      globalSdf.Sampler, globalSdf.DummyView,
                                      "Artifacts/Shaders")) {
        RS_LOG_ERROR("GBufferPreviewInitialize failed");
        return 8;
    }

    RS::PickingSystem picking;
    if (!RS::PickingInitialize(picking, vk)) {
        RS_LOG_ERROR("PickingInitialize failed");
        return 9;
    }

    std::unique_ptr<RS::IShadowAlgorithm> shadow = std::make_unique<RS::SDFConeShadow>();
    shadow->Initialize(vk, RS::OffscreenFormatDepth(), { 0, 0 });

    // Phase 13.5 â€” per-instance inverse-model SSBO ring. Fixes the SDF-shadow
    // coord-space mismatch (SDF baked in mesh-local; pixels are in world).
    // Bound at SDFCone set=2 binding 4.
    RS::InstanceXformBuffer instanceXforms;
    if (!RS::InstanceXformBufferInitialize(instanceXforms, vk)) {
        RS_LOG_ERROR("InstanceXformBufferInitialize failed");
        return 11;
    }
    auto pushInstanceXformsIfSDFCone = [&]() {
        if (auto* cone = dynamic_cast<RS::SDFConeShadow*>(shadow.get())) {
            cone->SetInstanceXformBuffer(instanceXforms.Buffers.data(),
                                         RS::InstanceXformBuffer::kBytes);
        }
    };
    pushInstanceXformsIfSDFCone();

    // ---- Radiance Cascades GI (rewrite #2) ---------------------------------
    // Resources are allocated unconditionally so the feature is driven entirely
    // by the in-app Debug panel toggles (Build RC atlas / Show RC atlas slice)
    // instead of a launch flag. rcGiReady is false only if GPU allocation fails;
    // the toggles then stay disabled. The debug blit (off by default) overwrites
    // the swapchain with a chosen atlas slice for first-light inspection.
    RS::RadianceCascades rcGi{};
    bool rcGiReady = true;
    if (!RS::RadianceCascadesInitialize(rcGi, vk)) {
        RS_LOG_ERROR("RadianceCascadesInitialize failed; RC disabled this session");
        rcGiReady = false;
    } else if (!RS::RadianceCascadesConstructAtlases(rcGi, vk)) {
        RS_LOG_ERROR("RadianceCascadesConstructAtlases failed; RC disabled this session");
        rcGiReady = false;
    } else {
        // Same per-instance inverse-model ring the cone shadow uses; RC reads
        // InvModel[InstanceIndex] to map each probe ray world -> mesh-local.
        RS::RadianceCascadesSetInstanceXformBuffer(
            rcGi, instanceXforms.Buffers.data(), RS::InstanceXformBuffer::kBytes);
        RS_LOG_INFO("RCGI ready: %u cascades constructed", rcGi.CascadeCount);
    }

    RS::LightingPass lighting;
    if (!RS::LightingPassInitialize(lighting, vk, targets, sky, *shadow,
                                    "Artifacts/Shaders")) {
        RS_LOG_ERROR("LightingPassInitialize failed");
        return 10;
    }

    RS::TonemapPass tonemap;
    if (!RS::TonemapPassInitialize(tonemap, vk, targets, "Artifacts/Shaders")) {
        RS_LOG_ERROR("TonemapPassInitialize failed");
        return 12;
    }
    RS::TonemapSettings tonemapSettings;

    // Phase 16: GPU timestamp ring (per-pass start/end timestamps; readback
    // lands at the next BeginFrame on the same slot, gated on the fence wait).
    RS::PerfTimers perfTimers;
    if (!RS::PerfTimersInitialize(perfTimers, vk)) {
        RS_LOG_ERROR("PerfTimersInitialize failed");
        return 13;
    }

    RS::RenderSettings renderSettings;

    RS::Renderer renderer;
    RS::InitDesc rd{};
    rd.Instance            = vk.Instance;
    rd.PhysicalDevice      = vk.PhysicalDevice;
    rd.Device              = vk.Device;
    rd.GraphicsQueue       = vk.GraphicsQueue;
    rd.GraphicsQueueFamily = vk.GraphicsQueueFamily;
    rd.RenderWidth         = vk.SwapchainExtent.width;
    rd.RenderHeight        = vk.SwapchainExtent.height;
    rd.ShaderArtifactsDir  = "Artifacts/Shaders";
    renderer.Initialize(rd);

    RS::Scene& scene = renderer.GetScene();
    RS::SceneAttachVulkan(scene, vk);

    RS::MaterialRegistry& materials = renderer.GetMaterials();
    const RS::SeededMaterials seeded = RS::SeedDemoMaterials(materials);
    RS_LOG_INFO("Seeded %u demo materials (Default=%u, RedPlastic=%u, "
                "PolishedGold=%u, BrushedAluminium=%u, MatteBlackRubber=%u, "
                "EmissiveCyan=%u, Floor=%u)",
                RS::SeededMaterials::kCount, seeded.Default, seeded.RedPlastic,
                seeded.PolishedGold, seeded.BrushedAluminium,
                seeded.MatteBlackRubber, seeded.EmissiveCyan, seeded.Floor);

    const RS::MeshHandle shaderBall = scene.InscribeMesh({ kShaderBallPath });
    glm::vec3 aabbMin(0.0f), aabbMax(0.0f);
    if (shaderBall != 0) {
        if (const RS::GpuMesh* mesh = RS::SceneMeshes(scene).Get(shaderBall)) {
            aabbMin = mesh->AABBMin;
            aabbMax = mesh->AABBMax;
        }
    }

    std::vector<RS::InstanceHandle> gridSlots;   // row-major, size = N*N
    int                             gridCurrentN = 0;
    RebuildShaderBallGrid(scene, gridSlots, gridCurrentN, /*requested*/ 3,
                          shaderBall, aabbMin, aabbMax, /*forceReseed*/ false);

    // Phase 11.5 â€” vast checker floor as a real GBuffer mesh. Shadows from any
    // SDF cone shadows sample it through the GBuffer like the other scene geometry.
    RS::FloorPlaneDesc floorDesc{};
    floorDesc.HalfExtent = 500.0f;
    floorDesc.Y          = 0.0f;
    const RS::MeshHandle     floorMesh     = RS::FloorMeshInscribe(scene, floorDesc);
    RS::InstanceHandle       floorInstance = 0;
    if (floorMesh != 0) {
        floorInstance = scene.InscribeInstance({ floorMesh, glm::mat4(1.0f) });
        if (floorInstance != 0 && seeded.Floor != 0) {
            scene.BindMaterial(floorInstance, 0, seeded.Floor);
        }
        RS_LOG_INFO("Floor: mesh=%u instance=%u material=%u",
                    floorMesh, floorInstance, seeded.Floor);
    }
    // Scale-reference sphere: a 0.5m-radius ball hovering above the first
    // ShaderBall so the working world scale is visible (the ShaderBall is
    // authored in cm and scaled down 100x â€” easy to misjudge probe spacing
    // without a known-size reference). Placed over grid cell (0,0).
    {
        RS::ScaleRefSphereDesc sphereDesc{};
        sphereDesc.Radius = 0.5f;   // metres
        const RS::MeshHandle sphereMesh = RS::ScaleRefSphereInscribe(scene, sphereDesc);
        if (sphereMesh != 0) {
            constexpr float kCmToMeters = 0.01f;
            const glm::mat4 cell0 = CellTransform(0, 0, gridCurrentN, aabbMin, aabbMax);
            const glm::vec3 ballXZ(cell0[3].x, 0.0f, cell0[3].z);
            const float ballTopY = aabbMax.y * kCmToMeters;          // ball height in world m
            const glm::vec3 spherePos =
                ballXZ + glm::vec3(0.0f, ballTopY + 0.5f + sphereDesc.Radius, 0.0f);
            const RS::InstanceHandle sphereInstance =
                scene.InscribeInstance({ sphereMesh,
                    glm::translate(glm::mat4(1.0f), spherePos) });
            if (sphereInstance != 0 && seeded.RedPlastic != 0) {
                scene.BindMaterial(sphereInstance, 0, seeded.RedPlastic);
            }
            RS_LOG_INFO("ScaleRefSphere: instance %u at (%.2f, %.2f, %.2f), r=0.5m",
                        sphereInstance, spherePos.x, spherePos.y, spherePos.z);
        }
    }

    RS::GBufferFloorConfig floorCfg{};
    floorCfg.FloorInstance   = floorInstance;
    floorCfg.CheckerSpacing  = 1.0f;
    floorCfg.CheckerStrength = 0.85f;
    floorCfg.DarkTintScale   = 0.55f;

    RS::OrbitCamera             cam;
    RS::GridSettings            gridSettings;
    RS::GBufferMaterial         legacyMaterial;       // soon-to-die Phase 6 fallback
    RS::GBufferPreviewSettings  previewSettings;
    RcDebugSettings             rcDebug;
    RS::SkySettings             skySettings;
    RS::PanelSelection          panelSel;
    RS::SdfBakerState           sdfBaker;
    RS::PerfWidgetState         perfWidget;
    RS::MeshHandle              activeSdfMesh = shaderBall;

    sdfBaker.TargetMesh       = shaderBall;
    sdfBaker.TargetSourcePath = kShaderBallPath;
    RS::SdfBakerInitialize(sdfBaker, vk);

    // Phase 12 boot auto-load. Phase 15g flipped the priority: try the sparse
    // .rsdfvdb first (matches the new default UseSparse=true), fall back to the
    // legacy dense .rsdf only if no sparse file exists for the resolution.
    // Either residency may end up populated; both paths feed pushSdfToConsumers-
    // equivalent wiring below so the cone shadow and preview binders see
    // whichever residency exists.
    if (shaderBall != 0) {
        const uint32_t bootRes   = RS::SdfBakerResolutionPx(sdfBaker);
        const uint32_t bootBrick = 8u;   // matches cached shaderBall_*_b8.rsdfvdb

        bool sparseHit = RS::GlobalSDFTryLoadSparseFromCache(
            globalSdf, vk, shaderBall, kShaderBallPath, bootRes, bootBrick);

        // Dense load is still attempted: the dense view drives the SDFSlice
        // debug viewer's legacy "Dense .rsdf" mode and seeds the CPU slice
        // preview (sparse â†’ CPU decode lands in Phase 15g-follow-on). On a
        // pure-sparse machine the dense file is just absent and this no-ops.
        bool denseHit = RS::GlobalSDFTryLoadFromCache(globalSdf, vk, shaderBall,
                                                     kShaderBallPath, bootRes);
        if (denseHit) {
            auto baked = std::make_shared<RS::BakedSDF>();
            const std::string p = RS::SDFCacheDerivePath(kShaderBallPath, bootRes);
            if (RS::SDFCacheLoad(p.c_str(), *baked)) {
                sdfBaker.ResidentBakedCopy = baked;
                sdfBaker.SliceDirty = true;
            }
            if (const RS::ResidentSDF* r = RS::GlobalSDFGet(globalSdf, shaderBall)) {
                RS::GBufferPreviewSetSDF(preview, vk, r->View, globalSdf.Sampler);
            }
        }

        if (sparseHit) {
            if (const RS::ResidentSparseSDF* rs = RS::GlobalSDFGetSparse(globalSdf, shaderBall)) {
                if (auto* cone = dynamic_cast<RS::SDFConeShadow*>(shadow.get())) {
                    cone->SetSDF(rs, true);
                }
                RS::GBufferPreviewSetSparseSDF(preview, vk,
                                               rs->IndexBuffer, rs->IndexBytes,
                                               rs->PoolBuffer,  rs->PoolBytes);
                // RCGI: same residency, pushed the SDFConeShadow way. instanceIndex 0
                // because InstanceXformBufferRefresh packs the single SDF-mesh anchor
                // contiguously from slot 0 (it is the only resident sparse mesh).
                if (rcGiReady) {
                    RS::RadianceCascadesSetSDF(rcGi, rs, true, /*instanceIndex*/ 0u);
                }
            }
        }

        if (sparseHit && denseHit) {
            RS_LOG_INFO("Phase 15g: SDF cache hit for ShaderBall at %u^3 (sparse + dense)", bootRes);
        } else if (sparseHit) {
            RS_LOG_INFO("Phase 15g: SDF cache hit for ShaderBall at %u^3 (sparse only â€” legacy dense absent)", bootRes);
        } else if (denseHit) {
            RS_LOG_INFO("Phase 15g: SDF cache hit for ShaderBall at %u^3 (dense only â€” bake to produce .rsdfvdb)", bootRes);
        } else {
            RS_LOG_INFO("Phase 15g: no SDF cache for ShaderBall â€” open the SDF Baker window");
        }
    }

    while (window.PumpMessages()) {
        if (window.ConsumeF1Pressed()) {
            cam = RS::OrbitCamera{};
            RS_LOG_INFO("F1: camera reset");
        }

        RS::ImGuiContextRSNewFrame();

        const ImGuiIO& io = ImGui::GetIO();
        RS::CameraInput cin{};
        if (!io.WantCaptureMouse) {
            cin.MouseDeltaX = io.MouseDelta.x;
            cin.MouseDeltaY = io.MouseDelta.y;
            cin.WheelDelta  = io.MouseWheel;
            cin.MiddleDown  = io.MouseDown[2];
            cin.RightDown   = io.MouseDown[1];
        }
        cin.ShiftDown    = io.KeyShift;
        cin.DeltaSeconds = io.DeltaTime;
        cin.ViewportW    = static_cast<float>(vk.SwapchainExtent.width);
        cin.ViewportH    = static_cast<float>(vk.SwapchainExtent.height);
        RS::UpdateCamera(cam, cin);

        const float aspect = cin.ViewportW / (cin.ViewportH > 0.0f ? cin.ViewportH : 1.0f);
        const RS::CameraView view = RS::BuildCameraView(cam, aspect);

        // Keep sky's IBL state in sync with the Sun panel's checkbox so the
        // SkyAtmosphere re-bake gate only looks at one source of truth.
        skySettings.EnableIBL    = previewSettings.UseIBL;
        skySettings.IBLIntensity = previewSettings.IBLIntensity;

        // Mirror the panel's GridN through the slider so DrawScenePanel sees
        // the canonical value each frame.
        panelSel.GridN = gridCurrentN > 0 ? gridCurrentN : 3;
        panelSel.GridReloadPressed = false;

        RS::DrawCameraPanel   (cam);
        RS::DrawGridPanel     (gridSettings);
        DrawSunPanel          (previewSettings, skySettings);
        DrawSkyPanel          (skySettings);
        RS::DrawAtmospherePanel(skySettings);
        RS::DrawScenePanel    (panelSel, shaderBall, scene,
                               static_cast<size_t>(gridCurrentN * gridCurrentN));
        RS::DrawMaterialsPanel(panelSel, materials,
                               RS::SceneInstancesMut(scene),
                               shaderBall, scene);
        DrawDebugPanel        (previewSettings, rcDebug, rcGiReady ? &rcGi : nullptr);
        DrawMatcapPanel       (previewSettings.Matcap);
        DrawTonemapPanel      (tonemapSettings);
        RS::PerfWidgetDraw    (perfWidget, perfTimers, vk.FrameIndex);

        // Phase 12: SDF Baker window. Returns true if the GlobalSDF residency
        // changed this frame (bake completed, evict, load-from-disk, etc.) so
        // we can refresh the GBufferPreview's binding-10 descriptor and the
        // SDFConeShadow's set=2 binding-0 view.
        const bool sdfResidencyChanged =
            RS::SdfBakerPanelDrawAndPump(sdfBaker, vk, globalSdf, scene);

        // Helper â€” push the active resident SDF (or dummy) into both the
        // preview pass and the shadow algo if it's currently SDFConeShadow.
        auto pushSdfToConsumers = [&]() {
            activeSdfMesh = sdfBaker.TargetMesh;
            // Dense residency drives the GBufferPreview SDFSlice debug view
            // (Phase 15f will move it onto sparse too).
            const RS::ResidentSDF* r =
                RS::GlobalSDFGet(globalSdf, sdfBaker.TargetMesh);
            VkImageView denseView = r ? r->View : globalSdf.DummyView;
            RS::GBufferPreviewSetSDF(preview, vk, denseView, globalSdf.Sampler);

            // Phase 15d/e: cone shadow reads sparse residency.
            const RS::ResidentSparseSDF* rs =
                RS::GlobalSDFGetSparse(globalSdf, sdfBaker.TargetMesh);
            const bool hasSparse = (rs != nullptr);
            if (auto* cone = dynamic_cast<RS::SDFConeShadow*>(shadow.get())) {
                cone->SetSDF(rs, hasSparse);
            }
            if (rcGiReady) {
                RS::RadianceCascadesSetSDF(rcGi, rs, hasSparse, /*instanceIndex*/ 0u);
            }
            // Phase 15f â€” also wire sparse SSBOs into the debug-slice preview.
            // Falls back to the always-bound dummy when no sparse residency exists.
            RS::GBufferPreviewSetSparseSDF(preview, vk,
                                           hasSparse ? rs->IndexBuffer : VK_NULL_HANDLE,
                                           hasSparse ? rs->IndexBytes  : 0,
                                           hasSparse ? rs->PoolBuffer  : VK_NULL_HANDLE,
                                           hasSparse ? rs->PoolBytes   : 0);
        };
        if (sdfResidencyChanged) pushSdfToConsumers();

        RS::DrawRenderSettingsPanel(renderSettings);

        DrawShadowsPanel(*shadow);

        // Apply Scene panel actions before recording the frame so any new
        // instances render in this same submit.
        if (panelSel.GridReloadPressed) {
            RebuildShaderBallGrid(scene, gridSlots, gridCurrentN,
                                  panelSel.GridNRequested, shaderBall,
                                  aabbMin, aabbMax, /*forceReseed*/ true);
        } else if (panelSel.GridNRequested != gridCurrentN) {
            RebuildShaderBallGrid(scene, gridSlots, gridCurrentN,
                                  panelSel.GridNRequested, shaderBall,
                                  aabbMin, aabbMax, /*forceReseed*/ false);
        }

        RS::FrameInfo frame{};
        if (!RS::VulkanContextBeginFrame(vk, &frame)) {
            ImGui::EndFrame();
            continue;
        }

        // Phase 16: reset this slot's timestamp queries + readback last cycle's.
        // The fence wait inside VulkanContextBeginFrame already gated readback.
        RS::PerfTimersBeginFrame(perfTimers, vk, frame.Cmd, frame.FrameSlot);

        // Phase 9 picking â€” BeginFrame has just waited this slot's fence so
        // the previous-cycle copy is GPU-complete and the host-mapped pixel
        // is safe to read.
        uint32_t pickInst = 0, pickSub = 0;
        if (RS::PickingTryRead(picking, frame.FrameSlot, pickInst, pickSub)) {
            if (pickInst != 0) {
                panelSel.SelectedInstance = pickInst;
                panelSel.SelectedSubmesh  = pickSub;
                panelSel.OpenMaterialsNextFrame = true;
                if (const RS::GpuInstance* gi =
                        RS::SceneInstances(scene).Get(pickInst)) {
                    if (pickSub < gi->MaterialBindings.size()) {
                        panelSel.SelectedMaterial = gi->MaterialBindings[pickSub];
                    }
                    // Phase 12: clicking a cell also retargets the SDF Baker
                    // window to that cell's mesh (per-mesh residency, so all
                    // instances of the same mesh share the bake).
                    sdfBaker.TargetMesh = gi->Mesh;
                    // Source path: only ShaderBall is in scope for v1.
                    sdfBaker.TargetSourcePath = kShaderBallPath;
                }
                const char* subName = (shaderBall != 0)
                    ? scene.SubmeshGroupName(shaderBall, pickSub) : "";
                RS_LOG_INFO("pick: instance=%u submesh=%u (%s)",
                            pickInst, pickSub, subName);
            }
        }

        // Click-to-pick: queue a 1x1 readback on left-down when ImGui isn't
        // capturing the mouse. The actual copy is recorded after GBufferPass.
        if (!io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const ImVec2 mp = io.MousePos;
            const uint32_t mx = mp.x > 0.0f ? static_cast<uint32_t>(mp.x) : 0u;
            const uint32_t my = mp.y > 0.0f ? static_cast<uint32_t>(mp.y) : 0u;
            RS::PickingRequest(picking, frame.FrameSlot, mx, my);
        }

        // Phase 8: bake sky / IBL cubemaps if the settings changed since last
        // frame. Hash gate makes the steady-state cost a single comparison.
        // Phase B bug fix: sync the live sun direction (driven by Sun panel)
        // into SkySettings so the sky cube tracks the same sun the lighting
        // compose uses.
        skySettings.SunDirection = glm::normalize(previewSettings.SunDirection);
        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::SkyBake);
        RS::SkyAtmosphereEnsureBaked(sky, vk, frame.Cmd, skySettings);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::SkyBake);

        RS::GridPassRecord       (grid, vk, frame.Cmd, frame.ImageIndex, view, gridSettings);

        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::GBuffer);
        RS::GBufferPassRecord    (gbuffer, vk, frame.Cmd, frame.FrameSlot, view,
                                  RS::SceneMeshes(scene), RS::SceneInstances(scene),
                                  materials, legacyMaterial, floorCfg);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::GBuffer);

        // After GBufferPass, identity is SHADER_READ_ONLY_OPTIMAL. RecordCopy
        // briefly transitions to TRANSFER_SRC + back so the preview pass below
        // still finds the layout it expects.
        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::PickingCopy);
        RS::PickingRecordCopy    (picking, frame.Cmd, frame.FrameSlot,
                                  targets.Frames[frame.FrameSlot].Identity.Image,
                                  targets.Extent);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::PickingCopy);

        // Phase 10: shadow + lighting compose. Build a FrameContext snapshot
        // so the algo + downstream callbacks see one consistent state.
        RS::FrameContext frameCtx{};
        frameCtx.Cam          = view;
        frameCtx.SunDirection = previewSettings.SunDirection;
        frameCtx.SunColor     = glm::vec3(1.0f);
        frameCtx.SunIntensity = previewSettings.SunIntensity;
        frameCtx.FrameSlot    = frame.FrameSlot;
        frameCtx.FrameIndex   = vk.FrameIndex;
        frameCtx.ScenePtr     = &scene;
        frameCtx.RenderExtent = vk.SwapchainExtent;
        // GBuffer views for any pass that wants to sample them through the
        // FrameContext rather than reach into OffscreenTargets directly.
        {
            const RS::OffscreenFrame& f = targets.Frames[frame.FrameSlot];
            frameCtx.GBufferAlbedo       = f.Albedo.View;
            frameCtx.GBufferNormal       = f.Normal.View;
            frameCtx.GBufferRoughMetalF0 = f.RoughMetalF0.View;
            frameCtx.GBufferEmissive     = f.Emissive.View;
            frameCtx.GBufferDepth        = f.Depth.View;
            frameCtx.GBufferIdentity     = f.Identity.View;
        }
        // Refresh per-instance inverse-model SSBO for this frame's slot.
        // SDFCone reads this at set=2 binding 2; other variants ignore it.
        // Pass the SDF target mesh + the first live grid ball as the "anchor"
        // so non-ShaderBall instances (the floor) route to a correctly-scaled
        // invModel instead of identity. Without this, the floor traces in
        // world meters against the cm-AABB and the shadow appears 100Ã— too
        // large. Multi-ball floor shadows land with multi-mesh SDF (Phase 14b/16).
        RS::InstanceHandle sdfAnchorInstance = 0;
        for (RS::InstanceHandle h : gridSlots) {
            if (h != 0) { sdfAnchorInstance = h; break; }
        }
        RS::InstanceXformBufferRefresh(instanceXforms, frame.FrameSlot, scene,
                                       activeSdfMesh, sdfAnchorInstance);

        // RCGI passes: dispatch rc_build per cascade now that this slot's xform
        // ring is current, then merge cascades C-2..0. Leaves all atlases in
        // GENERAL (debug blit reads them).
        if (rcGiReady && rcDebug.BuildEnabled) {
            RS::RadianceCascadesRecordBuild(
                rcGi, frame.Cmd, frame.FrameSlot,
                view.EyePositionWorld,
                previewSettings.SunDirection,
                glm::vec3(previewSettings.SunIntensity));
            RS::RadianceCascadesRecordMerge(rcGi, frame.Cmd, frame.FrameSlot);
        }

        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Shadow);
        shadow->RecordShadowPass(frame.Cmd, frameCtx);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Shadow);

        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Lighting);
        RS::LightingPassRecord(lighting, vk, frame.Cmd, frame.FrameSlot, targets,
                               *shadow, view, previewSettings.SunDirection,
                               glm::vec3(1.0f), previewSettings.SunIntensity,
                               previewSettings.Ambient,
                               previewSettings.UseIBL, previewSettings.IBLIntensity,
                               renderSettings.PBR.RealisticPbr);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Lighting);

        // Phase 16: ACES + exposure + sRGB encode, in-place over LightHDR.
        // LightHDR comes in as SHADER_READ_ONLY_OPTIMAL from Lighting; Tonemap
        // flips to GENERAL, writes, flips back.
        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Tonemap);
        RS::TonemapPassRecord(tonemap, frame.Cmd, frame.FrameSlot, targets,
                              tonemapSettings);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Tonemap);

        const RS::SDFSliceState sdfSliceState =
            RS::SdfBakerComposeSliceState(sdfBaker, globalSdf);
        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Preview);
        RS::GBufferPreviewRecord (preview, vk, frame.Cmd, frame.ImageIndex, frame.FrameSlot,
                                  view, previewSettings, sdfSliceState);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::Preview);

        // RCGI debug: blit the selected atlas Z-slice over the scene, then draw
        // ImGui after it so the debug controls stay reachable while inspecting.
        // The swapchain image is COLOR_ATTACHMENT_OPTIMAL after the preview pass;
        // bracket the blit to TRANSFER_DST and back so ImGui can load from it.
        if (rcGiReady && rcDebug.ShowDebugSlice) {
            VkImage swap = vk.SwapchainImages[frame.ImageIndex];

            VkImageMemoryBarrier toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            toDst.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            toDst.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image               = swap;
            toDst.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdPipelineBarrier(frame.Cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);

            RS::RadianceCascadesRecordDebugSlice(
                rcGi, frame.Cmd, swap, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.SwapchainExtent.width, vk.SwapchainExtent.height,
                static_cast<uint32_t>(rcDebug.CascadeIndex),
                static_cast<uint32_t>(rcDebug.SliceZ));

            VkImageMemoryBarrier toColor = toDst;
            toColor.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            toColor.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toColor.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            vkCmdPipelineBarrier(frame.Cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &toColor);
        }

        RS::PerfTimersBeginPass(perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::ImGuiDraw);
        RS::ImGuiContextRSRender (gui,  vk, frame.Cmd, frame.ImageIndex);
        RS::PerfTimersEndPass  (perfTimers, frame.Cmd, frame.FrameSlot, RS::PerfPass::ImGuiDraw);

        RS::VulkanContextEndFrame(vk,   frame);
    }

    RS::SdfBakerTerminate     (sdfBaker, vk);
    RS::SceneDetachVulkan(scene);
    RS::PerfTimersTerminate   (perfTimers, vk);
    RS::TonemapPassTerminate  (tonemap,  vk);
    RS::LightingPassTerminate (lighting, vk);
    shadow->Terminate         (vk);
    shadow.reset();
    if (rcGiReady) RS::RadianceCascadesTerminate(rcGi, vk);
    RS::InstanceXformBufferTerminate(instanceXforms, vk);
    RS::PickingTerminate(picking, vk);
    RS::GBufferPreviewTerminate(preview, vk);
    RS::GBufferPassTerminate   (gbuffer, vk);
    RS::OffscreenTargetsTerminate(targets, vk);
    RS::GridPassTerminate     (grid,    vk);
    RS::SkyAtmosphereTerminate(sky,     vk);
    RS::GlobalSDFTerminate    (globalSdf, vk);
    RS::ImGuiContextRSTerminate(gui,    vk);
    renderer.Terminate();
    vk.Terminate();
    window.Destroy();
    return 0;
}
