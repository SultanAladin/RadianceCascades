// Source/UI/ImGuiPanel.cpp — Phase 4 / Phase 9
// Standalone panels for the dev-loop. Phase 4 added Camera + Grid; Phase 9
// added Scene (N x N grid) and Materials (registry editor + per-instance
// submesh binding table). The full single-window layout (left rail, right
// pane, splitter) from the master plan lands when the panel count crosses 6.
#include "imgui.h"
#include "UI/ImGuiPanel.h"
#include "RS/Material.h"
#include "Renderer/GridPass.h"
#include "Scene/InstanceRegistry.h"
#include "Scene/MeshRegistry.h"

#include <cstdio>

namespace RS {

void DrawCameraPanel(OrbitCamera& cam) {
    ImGui::SetNextWindowPos (ImVec2(16.0f, 16.0f),  ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 230.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Camera")) {
        const ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("FPS:      %7.1f", io.Framerate);
        ImGui::Text("Frame:    %7.3f ms",
                    1000.0f / (io.Framerate > 0 ? io.Framerate : 1.0f));
        ImGui::Separator();
        ImGui::Text("Mode:     %s", cam.Mode == CameraMode::Orbit ? "Orbit" : "Fly");
        ImGui::Text("Yaw:      %7.2f deg", cam.YawDeg);
        ImGui::Text("Pitch:    %7.2f deg", cam.PitchDeg);
        ImGui::Text("Distance: %7.3f",     cam.Distance);
        ImGui::Text("Target:   (%.2f, %.2f, %.2f)",
                    cam.Target.x, cam.Target.y, cam.Target.z);
        ImGui::Separator();
        ImGui::Checkbox("Invert X (yaw)",   &cam.InvertX);
        ImGui::SameLine();
        ImGui::Checkbox("Invert Y (pitch)", &cam.InvertY);
        ImGui::TextDisabled("MMB orbit  Shift+MMB pan  Wheel zoom");
    }
    ImGui::End();
}

void DrawGridPanel(GridSettings& s) {
    ImGui::SetNextWindowPos (ImVec2(16.0f, 200.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 260.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Grid")) {
        ImGui::SliderFloat("Major spacing", &s.MajorSpacing, 0.1f,  10.0f);
        ImGui::SliderFloat("Minor spacing", &s.MinorSpacing, 0.01f,  s.MajorSpacing);
        ImGui::SliderFloat("Major weight",  &s.MajorThickness, 0.2f, 4.0f);
        ImGui::SliderFloat("Minor weight",  &s.MinorThickness, 0.2f, 4.0f);
        ImGui::SliderFloat("Opacity",       &s.Opacity, 0.0f, 1.0f);
        ImGui::Checkbox  ("Infinite",       &s.Infinite);
        if (!s.Infinite) {
            ImGui::SliderFloat("Extent", &s.Extent, 1.0f, 200.0f);
        }
        ImGui::Checkbox  ("Axis highlight", &s.AxisHighlight);
        ImGui::SliderFloat("Fade distance", &s.FadeDistance, 0.0f, 200.0f);
        ImGui::SliderFloat("Falloff start", &s.FalloffStart, 0.0f, s.FadeDistance);
        ImGui::SliderFloat("Falloff curve", &s.FalloffCurve, 0.1f, 4.0f);
        ImGui::ColorEdit3("Major colour",   s.MajorColor);
        ImGui::ColorEdit3("Minor colour",   s.MinorColor);
        ImGui::ColorEdit3("Sky tint",       s.SkyColor);
        ImGui::ColorEdit3("Ground colour",  s.GroundColor);

        ImGui::Separator();
        ImGui::TextUnformatted("Checker floor");
        ImGui::SliderFloat("Checker strength", &s.CheckerStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("Checker spacing",  &s.CheckerSpacing,  0.05f, 8.0f);
        ImGui::ColorEdit3 ("Tile light",       s.CheckerLight);
        ImGui::ColorEdit3 ("Tile dark",        s.CheckerDark);
    }
    ImGui::End();
}

// ---- Phase 9 panels --------------------------------------------------------

void DrawScenePanel(PanelSelection& sel, MeshHandle mesh, const Scene& scene,
                    size_t liveInstanceCount) {
    ImGui::SetNextWindowPos (ImVec2(960.0f, 16.0f),  ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(310.0f, 160.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scene")) {
        if (mesh == 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "ShaderBall failed to load");
        } else {
            const uint32_t subCount = scene.SubmeshCount(mesh);
            ImGui::Text("Mesh handle: %u   Submeshes: %u", mesh, subCount);
            ImGui::Separator();

            int n = sel.GridN;
            if (ImGui::SliderInt("Grid N", &n, 1, 10)) {
                sel.GridNRequested = n;
            } else {
                sel.GridNRequested = sel.GridN;
            }
            ImGui::Text("Instances: %zu (%dx%d)",
                        liveInstanceCount, sel.GridN, sel.GridN);
            if (ImGui::Button("Reload ShaderBall")) {
                sel.GridReloadPressed = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(destructive reset)");

            ImGui::Separator();
            if (sel.SelectedInstance != 0) {
                ImGui::Text("Selected: instance %u, submesh %u (%s)",
                            sel.SelectedInstance, sel.SelectedSubmesh,
                            scene.SubmeshGroupName(mesh, sel.SelectedSubmesh));
            } else {
                ImGui::TextDisabled("Click a ShaderBall to select.");
            }
        }
    }
    ImGui::End();
}

// Pretty-print a 12-byte hex preview of a colour so the binding combo can
// distinguish "Red Plastic" from "Polished Gold" by swatch when the names get
// too long. Kept inline so we don't drag a Color helper into Source/UI.
static ImU32 SwatchColor(const glm::vec3& c) {
    auto pack = [](float x) {
        const float v = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    };
    return IM_COL32(pack(c.r), pack(c.g), pack(c.b), 255);
}

void DrawMaterialsPanel(PanelSelection& sel, MaterialRegistry& materials,
                        InstanceRegistry& instances, MeshHandle mesh,
                        const Scene& scene) {
    if (sel.OpenMaterialsNextFrame) {
        ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
        ImGui::SetNextWindowFocus();
        sel.OpenMaterialsNextFrame = false;
    }
    ImGui::SetNextWindowPos (ImVec2(960.0f, 200.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 600.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Materials")) {
        const auto matCount = static_cast<MaterialHandle>(materials.Count());

        // ---- List of materials ------------------------------------------
        ImGui::TextUnformatted("Library");
        ImGui::Separator();
        if (ImGui::BeginChild("##matlist", ImVec2(0.0f, 140.0f), true)) {
            for (MaterialHandle h = 0; h < matCount; ++h) {
                PbrMaterial& mat = materials.Get(h);
                ImGui::PushID(static_cast<int>(h));
                const ImU32 swatch = SwatchColor(mat.AlbedoFlat);
                ImGui::ColorButton("##swatch",
                                   ImVec4(((swatch >> IM_COL32_R_SHIFT) & 0xFF) / 255.0f,
                                          ((swatch >> IM_COL32_G_SHIFT) & 0xFF) / 255.0f,
                                          ((swatch >> IM_COL32_B_SHIFT) & 0xFF) / 255.0f,
                                          1.0f),
                                   ImGuiColorEditFlags_NoTooltip |
                                   ImGuiColorEditFlags_NoBorder,
                                   ImVec2(16.0f, 16.0f));
                ImGui::SameLine();
                char label[96];
                std::snprintf(label, sizeof(label), "[%u] %s", h, mat.Name);
                if (ImGui::Selectable(label, sel.SelectedMaterial == h)) {
                    sel.SelectedMaterial = h;
                }
                ImGui::PopID();
            }
        }
        ImGui::EndChild();

        // ---- Channels editor --------------------------------------------
        ImGui::TextUnformatted("Channels");
        ImGui::Separator();
        if (sel.SelectedMaterial < matCount) {
            PbrMaterial& m = materials.Get(sel.SelectedMaterial);
            ImGui::InputText("Name", m.Name, sizeof(m.Name));
            ImGui::ColorEdit3 ("Albedo",    &m.AlbedoFlat.x);
            ImGui::SliderFloat("AO",        &m.AOFlat,        0.0f, 1.0f);
            ImGui::SliderFloat("Metallic",  &m.MetallicFlat,  0.0f, 1.0f);
            ImGui::SliderFloat("Roughness", &m.RoughnessFlat, 0.04f, 1.0f);
            ImGui::ColorEdit3 ("F0",        &m.F0Flat.x);
            ImGui::ColorEdit3 ("Emissive",  &m.EmissiveFlat.x,
                               ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
        } else {
            ImGui::TextDisabled("No material selected.");
        }

        // ---- Binding table (per-instance, per-submesh) ------------------
        ImGui::Separator();
        ImGui::TextUnformatted("Binding table");
        if (sel.SelectedInstance == 0 || mesh == 0) {
            ImGui::TextDisabled("Click an instance in the viewport to bind materials.");
        } else {
            const GpuInstance* inst = instances.Get(sel.SelectedInstance);
            if (!inst) {
                ImGui::TextDisabled("Selected instance no longer exists.");
            } else {
                ImGui::Text("Instance #%u", sel.SelectedInstance);
                if (ImGui::BeginTable("##bind", 2,
                                      ImGuiTableFlags_Borders |
                                      ImGuiTableFlags_RowBg   |
                                      ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Submesh", ImGuiTableColumnFlags_WidthStretch, 0.45f);
                    ImGui::TableSetupColumn("Material", ImGuiTableColumnFlags_WidthStretch, 0.55f);
                    ImGui::TableHeadersRow();

                    const uint32_t subCount =
                        static_cast<uint32_t>(inst->MaterialBindings.size());
                    for (uint32_t s = 0; s < subCount; ++s) {
                        ImGui::TableNextRow();
                        const bool isPicked = (s == sel.SelectedSubmesh);
                        if (isPicked) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                                   IM_COL32(255, 200, 80, 50));
                        }
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%u  %s", s, scene.SubmeshGroupName(mesh, s));

                        ImGui::TableSetColumnIndex(1);
                        ImGui::PushID(static_cast<int>(s));
                        const MaterialHandle cur = inst->MaterialBindings[s];
                        const char* curName = (cur < matCount)
                            ? materials.Get(cur).Name : "<invalid>";
                        if (ImGui::BeginCombo("##mat", curName)) {
                            for (MaterialHandle h = 0; h < matCount; ++h) {
                                const bool sel2 = (h == cur);
                                if (ImGui::Selectable(materials.Get(h).Name, sel2)) {
                                    instances.BindMaterial(sel.SelectedInstance, s, h);
                                    sel.SelectedMaterial = h;
                                    sel.SelectedSubmesh  = s;
                                }
                                if (sel2) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
            }
        }
    }
    ImGui::End();
}

} // namespace RS
