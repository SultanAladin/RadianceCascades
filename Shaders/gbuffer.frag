#version 450
// Phase 6 GBuffer fragment: write 5 colour attachments.
//   0: albedo                       RGBA8_UNORM
//   1: encoded normal (N*0.5+0.5)   A2B10G10R10_UNORM_PACK32
//   2: (roughness, metallic, F0)    RGBA8_UNORM
//   3: emissive                     RGBA16_SFLOAT
//   4: (instanceId, submeshId)      R32G32_UINT
layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec3 vWorldPos;

layout(push_constant) uniform Push {
    mat4 MVP;
    mat4 Model;
    vec4 BaseColor;             // rgb = albedo
    vec4 RoughMetalF0Emissive;  // r=roughness g=metallic b=F0 a=emissive scale
    vec4 EmissiveColor;         // rgb = emissive colour
    uvec4 IdentityIds;          // x = instanceId, y = submeshId
} pc;

layout(location = 0) out vec4  oAlbedo;
layout(location = 1) out vec4  oNormal;
layout(location = 2) out vec4  oRMF;
layout(location = 3) out vec4  oEmissive;
layout(location = 4) out uvec2 oIdentity;

void main() {
    vec3 N = normalize(vWorldNormal);

    oAlbedo   = vec4(pc.BaseColor.rgb, 1.0);
    oNormal   = vec4(N * 0.5 + 0.5, 1.0);
    oRMF      = vec4(clamp(pc.RoughMetalF0Emissive.r, 0.0, 1.0),
                     clamp(pc.RoughMetalF0Emissive.g, 0.0, 1.0),
                     clamp(pc.RoughMetalF0Emissive.b, 0.0, 1.0),
                     1.0);
    oEmissive = vec4(pc.EmissiveColor.rgb * pc.RoughMetalF0Emissive.a, 1.0);
    oIdentity = uvec2(pc.IdentityIds.x, pc.IdentityIds.y);
}
