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
    vec4 FloorParams;           // x = checker spacing (m), y = checker strength,
                                // z = mode flag (>=0.5 → procedural floor checker),
                                // w = checker dark tint scale (final dark = BaseColor * w)
} pc;

layout(location = 0) out vec4  oAlbedo;
layout(location = 1) out vec4  oNormal;
layout(location = 2) out vec4  oRMF;
layout(location = 3) out vec4  oEmissive;
layout(location = 4) out uvec2 oIdentity;

// Anti-aliased two-tile checker on world XZ. Returns the parity-blended albedo
// given a base tint and a dark-tint scale. Uses fwidth so distant tiles fade to
// their mean colour instead of flickering.
vec3 ProceduralFloorAlbedo(vec3 worldPos, vec3 tint, float spacing,
                           float strength, float darkScale) {
    vec2  scaled = worldPos.xz / max(spacing, 1e-4);
    vec2  cell   = floor(scaled);
    float parity = mod(cell.x + cell.y, 2.0);     // 0 or 1
    // Edge-AA: blend toward 0.5 as derivative grows so far tiles → mean.
    vec2  fw     = max(fwidth(scaled), vec2(1e-4));
    float aa     = clamp(max(fw.x, fw.y), 0.0, 1.0);
    float p      = mix(parity, 0.5, aa);
    vec3  dark   = tint * clamp(darkScale, 0.0, 1.0);
    vec3  light  = tint;
    vec3  checker = mix(dark, light, p);
    return mix(tint, checker, clamp(strength, 0.0, 1.0));
}

void main() {
    vec3 N = normalize(vWorldNormal);

    vec3 albedo = pc.BaseColor.rgb;
    if (pc.FloorParams.z >= 0.5) {
        albedo = ProceduralFloorAlbedo(vWorldPos, pc.BaseColor.rgb,
                                       pc.FloorParams.x, pc.FloorParams.y,
                                       pc.FloorParams.w);
    }

    oAlbedo   = vec4(albedo, 1.0);
    oNormal   = vec4(N * 0.5 + 0.5, 1.0);
    oRMF      = vec4(clamp(pc.RoughMetalF0Emissive.r, 0.0, 1.0),
                     clamp(pc.RoughMetalF0Emissive.g, 0.0, 1.0),
                     clamp(pc.RoughMetalF0Emissive.b, 0.0, 1.0),
                     1.0);
    oEmissive = vec4(pc.EmissiveColor.rgb * pc.RoughMetalF0Emissive.a, 1.0);
    oIdentity = uvec2(pc.IdentityIds.x, pc.IdentityIds.y);
}
