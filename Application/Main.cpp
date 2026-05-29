// Application/Main.cpp — Phase 10
// Phase 9 added the multi-instance grid + per-submesh material binding + click
// picking. Phase 10 adds the shadow plugin chain (IShadowAlgorithm) + the
// compute lighting pass (lighting.comp). PCFShadow stands up a 4-cascade CSM
// atlas; LightingPass composes Karis/Schlick/Smith + IBL + shadow into
// LightHDR; the GBufferPreview Lit mode samples that. Frame ordering is now
// Sky → Grid → GBuffer → PickingCopy → Shadow → Lighting → Preview → ImGui.
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
#include "Renderer/OffscreenTargets.h"
#include "Renderer/Picking.h"
#include "Renderer/SkyAtmosphere.h"
#include "Renderer/FrameContext.h"
#include "Shadow/IShadowAlgorithm.h"
#include "Shadow/PCFShadow.h"
#include "Shadow/PCSSShadow.h"
#include "Shadow/VSMShadow.h"
#include "Shadow/SDFConeShadow.h"
#include "SDF/GlobalSDF.h"
#include "Material/MaterialSeed.h"
#include "Scene/SceneInternal.h"
#include "UI/ImGuiContextRS.h"
#include "UI/ImGuiPanel.h"
#include "UI/SdfBakerPanel.h"
#include "imgui.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
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
    ImGui::SetNextWindowSize(ImVec2(320.0f, 320.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Sky")) {
        ImGui::TextUnformatted("Gradient");
        ImGui::ColorEdit3 ("Zenith",     &s.ZenithColor.x);
        ImGui::ColorEdit3 ("Horizon",    &s.HorizonColor.x);
        ImGui::ColorEdit3 ("Ground",     &s.GroundColor.x);
        ImGui::SliderFloat("Horizon exp", &s.HorizonExponent, 0.2f, 6.0f);
        ImGui::SliderFloat("Ground exp",  &s.GroundExponent,  0.2f, 6.0f);
        ImGui::SliderFloat("Sky intensity", &s.SkyIntensity,  0.0f, 4.0f);

        ImGui::Separator();
        ImGui::TextUnformatted("Sun disc (in sky cube)");
        ImGui::ColorEdit3 ("Sun colour",     &s.SunColor.x);
        ImGui::SliderFloat("Sun intensity",  &s.SunIntensity,    0.0f, 32.0f);
        ImGui::SliderFloat("Sun angular °",  &s.SunAngularSizeDeg, 0.1f, 8.0f);
        ImGui::TextDisabled("Edits re-bake IBL cubemaps (~sub-ms each).");
    }
    ImGui::End();
}

// Returns the variant the user requested (0=PCF, 1=PCSS, 2=VSM, 3=SDFCone) or
// -1 if no change. Active algo's own params are drawn inline.
int DrawShadowsPanel(RS::IShadowAlgorithm& active) {
    int requestedSwap = -1;
    ImGui::SetNextWindowPos (ImVec2(640.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 320.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Shadows")) {
        static const char* kAlgoLabels[] = {
            "PCF",
            "PCSS",
            "VSM",
            "SDF Cone",
        };
        int algoIdx = static_cast<int>(active.AlgoVariant());
        if (ImGui::Combo("Algorithm", &algoIdx, kAlgoLabels, IM_ARRAYSIZE(kAlgoLabels))) {
            if (algoIdx != static_cast<int>(active.AlgoVariant())) {
                requestedSwap = algoIdx;
            }
        }
        ImGui::TextDisabled("PCF / PCSS / VSM / SDF Cone live. SDF Cone uses the\n"
                            "active mesh SDF — bake one in the SDF Baker panel first.");

        ImGui::Separator();
        active.DrawImGuiParams();
    }
    ImGui::End();
    return requestedSwap;
}

void DrawDebugPanel(RS::GBufferPreviewSettings& preview) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 170.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 120.0f), ImGuiCond_FirstUseEver);
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
        ImGui::TextDisabled("Switch Debug → Matcap to view.");
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
// are inscribed. `forceReseed` clears everything first — the "Reload
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
    if (!window.Create(hInstance, L"Rendering Subsystem — Phase 9", kWidth, kHeight)) {
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

    auto makeShadowAlgo = [](int variant) -> std::unique_ptr<RS::IShadowAlgorithm> {
        switch (variant) {
            case 1:  return std::make_unique<RS::PCSSShadow>();
            case 2:  return std::make_unique<RS::VSMShadow>();
            case 3:  return std::make_unique<RS::SDFConeShadow>();
            case 0:
            default: return std::make_unique<RS::PCFShadow>();
        }
    };
    int initialShadowVariant = 0;
    {
        LPSTR cmdLine = GetCommandLineA();
        if (cmdLine) {
            if      (std::strstr(cmdLine, "-pcss"))    initialShadowVariant = 1;
            else if (std::strstr(cmdLine, "-vsm"))     initialShadowVariant = 2;
            else if (std::strstr(cmdLine, "-sdfcone")) initialShadowVariant = 3;
        }
    }
    std::unique_ptr<RS::IShadowAlgorithm> shadow = makeShadowAlgo(initialShadowVariant);
    shadow->Initialize(vk, RS::OffscreenFormatDepth(),
                       { RS::ShadowMap::kCascadeSize, RS::ShadowMap::kCascadeSize });

    RS::LightingPass lighting;
    if (!RS::LightingPassInitialize(lighting, vk, targets, sky, *shadow,
                                    "Artifacts/Shaders")) {
        RS_LOG_ERROR("LightingPassInitialize failed");
        return 10;
    }

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
                "EmissiveCyan=%u)",
                RS::SeededMaterials::kCount, seeded.Default, seeded.RedPlastic,
                seeded.PolishedGold, seeded.BrushedAluminium,
                seeded.MatteBlackRubber, seeded.EmissiveCyan);

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

    RS::OrbitCamera             cam;
    RS::GridSettings            gridSettings;
    RS::GBufferMaterial         legacyMaterial;       // soon-to-die Phase 6 fallback
    RS::GBufferPreviewSettings  previewSettings;
    RS::SkySettings             skySettings;
    RS::PanelSelection          panelSel;
    RS::SdfBakerState           sdfBaker;

    sdfBaker.TargetMesh       = shaderBall;
    sdfBaker.TargetSourcePath = kShaderBallPath;
    RS::SdfBakerInitialize(sdfBaker, vk);

    // Phase 12: try to auto-load a previously-cached SDF for the ShaderBall at
    // the default 64^3. If the cache hit, also load the CPU copy so the slice
    // preview is populated; otherwise the panel shows "Click Bake".
    if (shaderBall != 0) {
        const uint32_t bootRes = RS::SdfBakerResolutionPx(sdfBaker);
        if (RS::GlobalSDFTryLoadFromCache(globalSdf, vk, shaderBall,
                                          kShaderBallPath, bootRes)) {
            auto baked = std::make_shared<RS::BakedSDF>();
            const std::string p = RS::SDFCacheDerivePath(kShaderBallPath, bootRes);
            if (RS::SDFCacheLoad(p.c_str(), *baked)) {
                sdfBaker.ResidentBakedCopy = baked;
                sdfBaker.SliceDirty = true;
            }
            // Refresh GBufferPreview's binding to point at the freshly-resident
            // SDF instead of the dummy, and hand the residency to the shadow
            // algo if it's SDFConeShadow (a no-op for the other variants).
            if (const RS::ResidentSDF* r = RS::GlobalSDFGet(globalSdf, shaderBall)) {
                RS::GBufferPreviewSetSDF(preview, vk, r->View, globalSdf.Sampler);
                if (auto* cone = dynamic_cast<RS::SDFConeShadow*>(shadow.get())) {
                    cone->SetSDF(r->View, globalSdf.Sampler,
                                 r->AABBMin, r->AABBMax, r->MaxDist, true);
                }
            }
            RS_LOG_INFO("Phase 12: SDF cache hit for ShaderBall at %u^3", bootRes);
        } else {
            RS_LOG_INFO("Phase 12: no SDF cache for ShaderBall — open the SDF Baker window");
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
        RS::DrawScenePanel    (panelSel, shaderBall, scene,
                               static_cast<size_t>(gridCurrentN * gridCurrentN));
        RS::DrawMaterialsPanel(panelSel, materials,
                               RS::SceneInstancesMut(scene),
                               shaderBall, scene);
        DrawDebugPanel        (previewSettings);
        DrawMatcapPanel       (previewSettings.Matcap);

        // Phase 12: SDF Baker window. Returns true if the GlobalSDF residency
        // changed this frame (bake completed, evict, load-from-disk, etc.) so
        // we can refresh the GBufferPreview's binding-10 descriptor and the
        // SDFConeShadow's set=2 binding-0 view.
        const bool sdfResidencyChanged =
            RS::SdfBakerPanelDrawAndPump(sdfBaker, vk, globalSdf, scene);

        // Helper — push the active resident SDF (or dummy) into both the
        // preview pass and the shadow algo if it's currently SDFConeShadow.
        auto pushSdfToConsumers = [&]() {
            const RS::ResidentSDF* r =
                RS::GlobalSDFGet(globalSdf, sdfBaker.TargetMesh);
            VkImageView view = r ? r->View : globalSdf.DummyView;
            glm::vec3   aMin = r ? r->AABBMin : glm::vec3(0.0f);
            glm::vec3   aMax = r ? r->AABBMax : glm::vec3(1.0f);
            float       md   = r ? r->MaxDist : 1.0f;
            bool        has  = (r != nullptr);
            RS::GBufferPreviewSetSDF(preview, vk, view, globalSdf.Sampler);
            if (auto* cone = dynamic_cast<RS::SDFConeShadow*>(shadow.get())) {
                cone->SetSDF(view, globalSdf.Sampler, aMin, aMax, md, has);
            }
        };
        if (sdfResidencyChanged) pushSdfToConsumers();

        const int requestedShadowSwap = DrawShadowsPanel(*shadow);
        if (requestedShadowSwap >= 0) {
            // Hot-swap algo: tear down old, build new, then rebuild the
            // lighting pipeline against the new set=2 layout + variant SPV.
            vkDeviceWaitIdle(vk.Device);
            shadow->Terminate(vk);
            shadow = makeShadowAlgo(requestedShadowSwap);
            shadow->Initialize(vk, RS::OffscreenFormatDepth(),
                               { RS::ShadowMap::kCascadeSize, RS::ShadowMap::kCascadeSize });
            if (!RS::LightingPassSetShadowAlgorithm(lighting, vk, *shadow,
                                                    "Artifacts/Shaders")) {
                RS_LOG_ERROR("Shadow swap: LightingPassSetShadowAlgorithm failed");
            }
            // SDFConeShadow needs the current resident SDF wired in before
            // its first RecordShadowPass. Other variants ignore the call.
            pushSdfToConsumers();
        }

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

        // Phase 9 picking — BeginFrame has just waited this slot's fence so
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
        RS::SkyAtmosphereEnsureBaked(sky, vk, frame.Cmd, skySettings);
        RS::GridPassRecord       (grid, vk, frame.Cmd, frame.ImageIndex, view, gridSettings);
        RS::GBufferPassRecord    (gbuffer, vk, frame.Cmd, frame.FrameSlot, view,
                                  RS::SceneMeshes(scene), RS::SceneInstances(scene),
                                  materials, legacyMaterial);
        // After GBufferPass, identity is SHADER_READ_ONLY_OPTIMAL. RecordCopy
        // briefly transitions to TRANSFER_SRC + back so the preview pass below
        // still finds the layout it expects.
        RS::PickingRecordCopy    (picking, frame.Cmd, frame.FrameSlot,
                                  targets.Frames[frame.FrameSlot].Identity.Image,
                                  targets.Extent);

        // Phase 10: shadow + lighting compose. Build a FrameContext snapshot
        // so the algo + downstream callbacks see one consistent state.
        RS::FrameContext frameCtx{};
        frameCtx.Cam          = view;
        frameCtx.SunDirection = previewSettings.SunDirection;
        frameCtx.SunColor     = glm::vec3(1.0f);
        frameCtx.SunIntensity = previewSettings.SunIntensity;
        frameCtx.FrameSlot    = frame.FrameSlot;
        frameCtx.ScenePtr     = &scene;
        frameCtx.RenderExtent = vk.SwapchainExtent;
        shadow->RecordShadowPass(frame.Cmd, frameCtx);

        RS::LightingPassRecord(lighting, vk, frame.Cmd, frame.FrameSlot, targets,
                               *shadow, view, previewSettings.SunDirection,
                               glm::vec3(1.0f), previewSettings.SunIntensity,
                               previewSettings.Ambient,
                               previewSettings.UseIBL, previewSettings.IBLIntensity);

        const RS::SDFSliceState sdfSliceState =
            RS::SdfBakerComposeSliceState(sdfBaker, globalSdf);
        RS::GBufferPreviewRecord (preview, vk, frame.Cmd, frame.ImageIndex, frame.FrameSlot,
                                  view, previewSettings, sdfSliceState);
        RS::ImGuiContextRSRender (gui,  vk, frame.Cmd, frame.ImageIndex);
        RS::VulkanContextEndFrame(vk,   frame);
    }

    RS::SdfBakerTerminate     (sdfBaker, vk);
    RS::SceneDetachVulkan(scene);
    RS::LightingPassTerminate (lighting, vk);
    shadow->Terminate         (vk);
    shadow.reset();
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
