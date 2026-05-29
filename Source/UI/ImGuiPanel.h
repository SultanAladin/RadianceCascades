// Source/UI/ImGuiPanel.h — Phase 9
// Public surface of the developer-loop ImGui panels. Main.cpp constructs a
// PanelSelection per-frame and passes it through DrawScenePanel /
// DrawMaterialsPanel; the panels mutate the selection in place. Phase 4's
// Camera + Grid panels stay stateless and live next to these.
#pragma once

#include "RS/Camera.h"
#include "RS/Scene.h"

namespace RS {

struct GridSettings;
class  MaterialRegistry;
class  InstanceRegistry;
class  Scene;

// Per-frame transient panel state shared between Scene + Materials.
struct PanelSelection {
    // N x N grid slider. Main.cpp diffs GridNRequested against the current
    // m_InstanceSlots size and inscribes/destroys instances to converge.
    int  GridN              = 3;
    int  GridNRequested     = 3;
    bool GridReloadPressed  = false;

    // Click-pick result + materials-panel state.
    InstanceHandle SelectedInstance       = 0;
    SubmeshIndex   SelectedSubmesh        = 0;
    MaterialHandle SelectedMaterial       = 0;
    bool           OpenMaterialsNextFrame = false;
};

void DrawCameraPanel   (OrbitCamera& cam);
void DrawGridPanel     (GridSettings& s);
void DrawScenePanel    (PanelSelection& sel, MeshHandle mesh, const Scene& scene,
                        size_t liveInstanceCount);
void DrawMaterialsPanel(PanelSelection& sel, MaterialRegistry& materials,
                        InstanceRegistry& instances, MeshHandle mesh,
                        const Scene& scene);

} // namespace RS
