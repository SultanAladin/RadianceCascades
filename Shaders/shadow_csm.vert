#version 450
// Phase 10 CSM depth-only vertex shader. One cascade per render-pass instance;
// the C++ side rebinds the framebuffer (per-layer depth view) and pushes a
// fresh LightViewProj for each cascade. No fragment shader is bound — the
// pipeline state has zero colour attachments and depth-only output.
//
// Vertex stream matches gbuffer.vert so MeshRegistry doesn't need a separate
// upload: (position, normal). Normal is consumed by the input layout but
// ignored by this shader.

layout(location = 0) in vec3 aPos;

layout(push_constant) uniform Push {
    mat4 LightViewProj;
    mat4 Model;
} pc;

void main() {
    gl_Position = pc.LightViewProj * pc.Model * vec4(aPos, 1.0);
}
