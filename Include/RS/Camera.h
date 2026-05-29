// Include/RS/Camera.h
// Orbit camera (yaw/pitch/distance) with optional WASD fly mode.
// Matches SolidArc UI/UIState.h:36-51 + PigmentGridRenderer.cpp:188-220 patterns
// so behaviour is familiar across the user's stack.
#pragma once

#include <glm/glm.hpp>

namespace RS {

enum class CameraMode {
    Orbit,
    Fly,
};

struct OrbitCamera {
    // Orbit state -------------------------------------------------------------
    glm::vec3 Target      = glm::vec3(0.0f, 0.0f, 0.0f);
    float     YawDeg      = 45.0f;
    float     PitchDeg    = -28.0f;
    float     Distance    = 5.0f;

    // Fly state ---------------------------------------------------------------
    glm::vec3 FlyPosition = glm::vec3(0.0f, 1.5f, 5.0f);

    // Shared ------------------------------------------------------------------
    CameraMode Mode     = CameraMode::Orbit;
    float      FovDeg   = 60.0f;
    float      NearClip = 0.05f;
    float      FarClip  = 200.0f;
    bool       Ortho    = false;

    // Mouse axis-invert. Applies to orbit drag AND fly mouse-look so the
    // user picks the rotation feel once and it sticks across modes.
    // InvertX defaults true because users consistently expect left-drag to
    // sweep the camera left (i.e. world rotates right).
    bool       InvertX  = true;
    bool       InvertY  = false;
};

struct CameraInput {
    float MouseDeltaX  = 0.0f;
    float MouseDeltaY  = 0.0f;
    float WheelDelta   = 0.0f;
    bool  MiddleDown   = false;
    bool  RightDown    = false;
    bool  ShiftDown    = false;
    bool  WDown        = false;
    bool  ADown        = false;
    bool  SDown        = false;
    bool  DDown        = false;
    bool  QDown        = false;
    bool  EDown        = false;
    float DeltaSeconds = 0.016f;
    float ViewportW    = 1024.0f;
    float ViewportH    = 1024.0f;
};

// CameraView is declared by Renderer.h; we forward-declare here to keep the
// header dependency one-directional.
struct CameraView;

void       UpdateCamera   (OrbitCamera& cam, const CameraInput& in);
CameraView BuildCameraView(const OrbitCamera& cam, float aspect);

} // namespace RS
