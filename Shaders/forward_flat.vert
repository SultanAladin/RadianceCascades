#version 450
// Phase 5 throw-away forward pass: world-space Lambert against a hard-coded sun.
// vertex inputs come from the interleaved `ParsedVertex` stream: position +
// normal, both vec3.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Push {
    mat4 MVP;
    mat4 Model;
    vec4 SunDirAndIntensity;   // xyz = direction toward the sun, w = intensity
    vec4 BaseColor;            // rgb = albedo, a = ambient term
} pc;

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) out vec3 vWorldPos;

void main() {
    vec4 wp     = pc.Model * vec4(inPosition, 1.0);
    vWorldPos   = wp.xyz;
    // No non-uniform scale on Phase 5 instances, so the model 3×3 is fine for
    // normals. Phase 6+ will switch to a proper inverse-transpose.
    vWorldNormal = normalize(mat3(pc.Model) * inNormal);
    gl_Position  = pc.MVP * vec4(inPosition, 1.0);
}
