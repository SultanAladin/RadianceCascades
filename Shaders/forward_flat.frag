#version 450
// Throw-away Lambert: NdotL against the push-const sun direction + a tiny
// ambient lift so back-facing pixels aren't pitch black. The real Phase 7
// PBR path replaces this whole pass.
layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec3 vWorldPos;

layout(push_constant) uniform Push {
    mat4 MVP;
    mat4 Model;
    vec4 SunDirAndIntensity;
    vec4 BaseColor;
} pc;

layout(location = 0) out vec4 oColor;

void main() {
    vec3  N        = normalize(vWorldNormal);
    vec3  L        = normalize(pc.SunDirAndIntensity.xyz);
    float NdotL    = max(dot(N, L), 0.0);
    float sun      = NdotL * pc.SunDirAndIntensity.w;
    float ambient  = pc.BaseColor.a;

    vec3  lit = pc.BaseColor.rgb * (sun + ambient);
    oColor    = vec4(lit, 1.0);
}
