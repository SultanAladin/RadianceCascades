#version 450
// Phase 6 + KHR-specular GBuffer fragment: writes 6 colour attachments.
//   0: albedo + AO                  RGBA8_UNORM   (rgb = albedo, a = AO)
//   1: encoded normal (N*0.5+0.5)   A2B10G10R10_UNORM_PACK32
//   2: (roughness, metallic, specF) RGBA8_UNORM   (b = KHR specularFactor, a unused)
//   3: emissive                     RGBA16_SFLOAT
//   4: specular colour              RGBA8_UNORM   (rgb = KHR specularColor)
//   5: (instanceId, submeshId)      R32G32_UINT
layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec3 vWorldPos;

layout(push_constant) uniform Push {
    mat4 MVP;
    mat4 Model;
    vec4 BaseColor;             // rgb = albedo, a = AO
    vec4 RoughMetalSpecFactor;  // r=roughness g=metallic b=SpecularFactor a=emissive scale
    vec4 EmissiveColor;         // rgb = emissive colour
    vec4 SpecularColor;         // rgb = KHR specularColor (dielectric tint)
    uvec4 IdentityIds;          // x = instanceId, y = submeshId
    vec4 FloorParams;           // x = checker spacing (m), y = checker strength,
                                // z = mode flag (>=0.5 → procedural floor checker),
                                // w = checker dark tint scale (final dark = BaseColor * w)
} pc;

layout(location = 0) out vec4  oAlbedo;
layout(location = 1) out vec4  oNormal;
layout(location = 2) out vec4  oRMF;
layout(location = 3) out vec4  oEmissive;
layout(location = 4) out vec4  oSpecular;
layout(location = 5) out uvec2 oIdentity;

// Anti-aliased two-tile checker on world XZ. Returns the parity-blended albedo
// given a base tint and a dark-tint scale. Uses fwidth so distant tiles fade to
// their mean colour instead of flickering.
vec3 ProceduralFloorAlbedo(vec3 worldPos, vec3 tint, float spacing,
                           float strength, float darkScale) {
    vec2  scaled = worldPos.xz / max(spacing, 1e-4);
    vec2  cell   = floor(scaled);
    float parity = mod(cell.x + cell.y, 2.0);     // 0 or 1
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

    oAlbedo   = vec4(albedo, clamp(pc.BaseColor.a, 0.0, 1.0));   // a = AO
    oNormal   = vec4(N * 0.5 + 0.5, 1.0);
    oRMF      = vec4(clamp(pc.RoughMetalSpecFactor.r, 0.0, 1.0),
                     clamp(pc.RoughMetalSpecFactor.g, 0.0, 1.0),
                     clamp(pc.RoughMetalSpecFactor.b, 0.0, 1.0),
                     1.0);
    oEmissive = vec4(pc.EmissiveColor.rgb * pc.RoughMetalSpecFactor.a, 1.0);
    oSpecular = vec4(clamp(pc.SpecularColor.rgb, vec3(0.0), vec3(1.0)), 1.0);
    oIdentity = uvec2(pc.IdentityIds.x, pc.IdentityIds.y);
}
