#version 450
// Phase 6 GBuffer vertex: emit world-space position + normal alongside the
// clip-space output. Same {pos, normal} input stream as forward_flat.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Push {
    mat4 MVP;
    mat4 Model;
    vec4 BaseColor;
    vec4 RoughMetalF0Emissive;
    vec4 EmissiveColor;
    uvec4 IdentityIds;
    vec4 FloorParams;
} pc;

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) out vec3 vWorldPos;

void main() {
    vec4 wp      = pc.Model * vec4(inPosition, 1.0);
    vWorldPos    = wp.xyz;
    // Phase 5 still avoids non-uniform scale, so model 3x3 works for normals.
    // Phase 7+ will pass an explicit normal matrix.
    vWorldNormal = normalize(mat3(pc.Model) * inNormal);
    gl_Position  = pc.MVP * vec4(inPosition, 1.0);
}
