// Source/Camera/OrbitCamera.cpp
// Ported from SolidArc UI/UIState.h:36-51 + Vulkan/PigmentGridRenderer.cpp:188-220.
// Orbit camera with optional WASD fly mode.
#include "RS/Camera.h"
#include "RS/Renderer.h"   // for CameraView

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace RS {

namespace {

// Spherical (yaw, pitch) → unit forward vector. Matches the SolidArc form so
// camera behaviour is consistent across this project, SolidArc, and Pigment.
glm::vec3 ForwardFromYawPitch(float yawDeg, float pitchDeg) {
    const float yawRad   = glm::radians(yawDeg);
    const float pitchRad = glm::radians(pitchDeg);
    return glm::vec3(
        std::cos(pitchRad) * std::sin(yawRad),
        std::sin(pitchRad),
        std::cos(pitchRad) * std::cos(yawRad)
    );
}

} // namespace

void UpdateCamera(OrbitCamera& cam, const CameraInput& in) {
    const float xSign = cam.InvertX ? -1.0f : 1.0f;
    const float ySign = cam.InvertY ? -1.0f : 1.0f;

    if (cam.Mode == CameraMode::Orbit) {
        if (in.MiddleDown && !in.ShiftDown) {
            // Middle-drag orbit: dx → yaw, dy → pitch. 0.5°/pixel matches
            // SolidArc's orbit sensitivity.
            cam.YawDeg   += xSign * in.MouseDeltaX * 0.5f;
            cam.PitchDeg -= ySign * in.MouseDeltaY * 0.5f;
            cam.PitchDeg = std::clamp(cam.PitchDeg, -89.0f, 89.0f);
        } else if (in.MiddleDown && in.ShiftDown) {
            // Shift+middle pan: translate the target along the view basis.
            const glm::vec3 forward = ForwardFromYawPitch(cam.YawDeg, cam.PitchDeg);
            const glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
            const glm::vec3 upBasis = glm::normalize(glm::cross(right, forward));
            const float panScale = cam.Distance * 0.0015f;
            cam.Target -= right   * (in.MouseDeltaX * panScale);
            cam.Target += upBasis * (in.MouseDeltaY * panScale);
        }
        if (in.WheelDelta != 0.0f) {
            // Exponential zoom keeps slider feel even across magnitudes.
            cam.Distance *= std::pow(0.9f, in.WheelDelta);
            cam.Distance  = std::clamp(cam.Distance, 0.1f, 500.0f);
        }
    } else {
        // Fly mode: WASDQE on view basis. Mouse-look gated on right-button.
        if (in.RightDown) {
            cam.YawDeg   += xSign * in.MouseDeltaX * 0.2f;
            cam.PitchDeg -= ySign * in.MouseDeltaY * 0.2f;
            cam.PitchDeg = std::clamp(cam.PitchDeg, -89.0f, 89.0f);
        }
        const glm::vec3 forward = ForwardFromYawPitch(cam.YawDeg, cam.PitchDeg);
        const glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        const float speed = 2.0f * (in.ShiftDown ? 4.0f : 1.0f) * in.DeltaSeconds;
        if (in.WDown) cam.FlyPosition += forward * speed;
        if (in.SDown) cam.FlyPosition -= forward * speed;
        if (in.DDown) cam.FlyPosition += right   * speed;
        if (in.ADown) cam.FlyPosition -= right   * speed;
        if (in.EDown) cam.FlyPosition.y += speed;
        if (in.QDown) cam.FlyPosition.y -= speed;
    }
}

CameraView BuildCameraView(const OrbitCamera& cam, float aspect) {
    const glm::vec3 forward = ForwardFromYawPitch(cam.YawDeg, cam.PitchDeg);

    glm::vec3 eye;
    glm::vec3 focus;
    if (cam.Mode == CameraMode::Orbit) {
        focus = cam.Target;
        eye   = focus - forward * cam.Distance;
    } else {
        eye   = cam.FlyPosition;
        focus = eye + forward;
    }

    CameraView v;
    v.View             = glm::lookAtRH(eye, focus, glm::vec3(0, 1, 0));
    v.Projection       = glm::perspectiveRH_ZO(glm::radians(cam.FovDeg),
                                               aspect, cam.NearClip, cam.FarClip);
    v.Projection[1][1] *= -1.0f;     // Vulkan clip-space Y-flip
    v.EyePositionWorld = eye;
    v.NearClip         = cam.NearClip;
    v.FarClip          = cam.FarClip;
    return v;
}

} // namespace RS
