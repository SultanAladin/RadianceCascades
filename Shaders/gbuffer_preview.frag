#version 450
// Phase 6 + 7 GBuffer preview. Picks one attachment, writes it to the swapchain.
// Uses sampled depth as a presence mask so the grid pass behind us still shows
// through wherever GBuffer didn't write geometry (depth==1.0).
//
// Mode codes match the C++ enum in GBufferPreview.h:
//   0 = lit preview (Phase 7 Cook-Torrance GGX, sun only, no shadows/IBL)
//   1 = albedo
//   2 = world normal
//   3 = (roughness, metallic, F0)
//   4 = emissive
//   5 = linearised depth
//   6 = identity colourised
//   7 = matcap (procedural, view-space N driven)
//   8 = SDF slice (Phase 12 — visualises the mesh SDF on a world-Y plane)
layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 oColor;

layout(set = 0, binding = 0) uniform sampler2D   uAlbedo;
layout(set = 0, binding = 1) uniform sampler2D   uNormal;
layout(set = 0, binding = 2) uniform sampler2D   uRMF;
layout(set = 0, binding = 3) uniform sampler2D   uEmissive;
layout(set = 0, binding = 4) uniform sampler2D   uDepth;
layout(set = 0, binding = 5) uniform usampler2D  uIdentity;
// Phase 8 IBL bindings.
layout(set = 0, binding = 6) uniform samplerCube uIrradiance;
layout(set = 0, binding = 7) uniform samplerCube uPrefilter;
layout(set = 0, binding = 8) uniform sampler2D   uBrdfLut;
// Phase 10: LightHDR (lit-mode source). Replaces the inline PbrLit path; the
// lighting.comp shader now does the Cook-Torrance + IBL + shadow compose into
// this image.
layout(set = 0, binding = 9) uniform sampler2D   uLightHDR;
// Phase 12: per-mesh SDF (R16_SNORM) sampled as a 3D texture in world space.
// Falls back to a 1^3 +1 dummy when no SDF is resident, so this binding is
// always safe to declare.
layout(set = 0, binding = 10) uniform sampler3D  uMeshSDF;
layout(set = 0, binding = 11) uniform SdfParams {
    vec4 AABBMinAndPlaneY;   // xyz = SDF AABB min, w = world-Y of slice plane
    vec4 AABBMaxAndMaxDist;  // xyz = SDF AABB max, w = R16_SNORM decode scale
    vec4 PreviewParams;      // x = viz mode (float),  y/z/w = reserved
} uSdf;

layout(push_constant) uniform Push {
    vec4 SunDirAndIntensity;   // xyz = direction toward sun, w = intensity
    vec4 AmbientAndNearFar;    // x = ambient, y = nearClip, z = farClip, w = mode (float)

    // Matcap settings (active when mode == 7).
    vec4 MatcapTop;            // rgb = top, a = gradient curve
    vec4 MatcapBottom;         // rgb = bottom, a = exposure
    vec4 MatcapRim;            // rgb = rim, a = rim power
    vec4 MatcapHighlight;      // rgb = highlight, a = highlight size
    vec4 MatcapParams;         // x = rim strength, y = highlight strength,
                               // z = hi off x, w = hi off y
    vec4 MatcapTintRow0;       // view rot row 0 + tint.x
    vec4 MatcapTintRow1;       // view rot row 1 + tint.y
    vec4 MatcapTintRow2;       // view rot row 2 + tint.z
    vec4 MatcapMaterial;       // x = roughness, y = metallic
    vec4 IblParams;            // x = on/off, y = intensity, z = prefilter mip count, w = unused
    mat4 InvViewProj;          // for Phase 7 world-pos reconstruction
} pc;

// -- helpers ------------------------------------------------------------------

vec3 IdentityColor(uvec2 id) {
    uint h = id.x * 2654435761u ^ id.y * 40503u;
    return vec3(
        float((h >>  0) & 0xFFu) / 255.0,
        float((h >>  8) & 0xFFu) / 255.0,
        float((h >> 16) & 0xFFu) / 255.0);
}

float Linearise(float depth, float nearZ, float farZ) {
    return clamp((depth - 0.0) / max(farZ - nearZ, 1e-3), 0.0, 1.0);
}

// -- matcap -------------------------------------------------------------------

vec3 Matcap(vec3 worldN) {
    mat3 viewRot = mat3(
        vec3(pc.MatcapTintRow0.x, pc.MatcapTintRow1.x, pc.MatcapTintRow2.x),
        vec3(pc.MatcapTintRow0.y, pc.MatcapTintRow1.y, pc.MatcapTintRow2.y),
        vec3(pc.MatcapTintRow0.z, pc.MatcapTintRow1.z, pc.MatcapTintRow2.z));
    vec3 viewN = normalize(viewRot * worldN);

    float roughness = clamp(pc.MatcapMaterial.x, 0.04, 1.0);
    float metallic  = clamp(pc.MatcapMaterial.y, 0.0,  1.0);

    // 1. Vertical gradient. Roughness blurs by lifting the bottom end slightly,
    //    metallic darkens the diffuse term (metals have no diffuse).
    float ny = clamp(viewN.y * 0.5 + 0.5, 0.0, 1.0);
    ny = pow(ny, max(pc.MatcapTop.a, 0.01));
    vec3 gradient = mix(pc.MatcapBottom.rgb, pc.MatcapTop.rgb, ny);
    gradient *= mix(1.0, 0.15, metallic);

    // 2. Rim — unchanged by roughness/metal (it's mostly Fresnel).
    float rim = pow(clamp(1.0 - viewN.z, 0.0, 1.0), max(pc.MatcapRim.a, 0.01));
    vec3 rimColor = pc.MatcapRim.rgb * rim * pc.MatcapParams.x;

    // 3. Highlight: roughness widens the smoothstep radius (rough = broader,
    //    softer spot); metallic tints the highlight by the top gradient
    //    colour instead of leaving it white-ish.
    vec2  hOffset    = vec2(pc.MatcapParams.z, pc.MatcapParams.w);
    float baseSize   = max(pc.MatcapHighlight.a, 0.001);
    float widenedSize = baseSize * mix(0.4, 3.0, roughness);
    float d         = length(viewN.xy - hOffset);
    float hMask     = 1.0 - smoothstep(0.0, widenedSize, d);

    vec3  hiColor   = mix(pc.MatcapHighlight.rgb, pc.MatcapTop.rgb, metallic);
    // Rough surfaces scatter the highlight energy — dim it proportionally so
    // the integral stays roughly constant.
    float hiEnergy  = mix(1.5, 0.35, roughness);
    vec3  highlight = hiColor * hMask * pc.MatcapParams.y * hiEnergy;

    vec3 tint     = vec3(pc.MatcapTintRow0.w, pc.MatcapTintRow1.w, pc.MatcapTintRow2.w);
    vec3 color    = (gradient + rimColor + highlight) * tint;
    return color * pc.MatcapBottom.a;
}

// -- Phase 7 PBR --------------------------------------------------------------
// Karis 2013 GGX + Schlick + Smith, sun light only, no shadows / IBL.

const float kPI = 3.14159265359;

float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (kPI * d * d + 1e-5);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness) {
    float a    = roughness * roughness;
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a) + a);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a) + a);
    return 0.5 / max(ggxV + ggxL, 1e-5);
}

vec3 F_Schlick(float VdotH, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// Karis Fresnel-with-roughness for IBL — clamps the Fresnel grazing-angle term
// when sampling a low-resolution prefilter, otherwise rough metals look too hot
// at glancing angles.
vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    return F0 + Fr * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 PbrLit(vec2 uv) {
    vec3  albedo  = texture(uAlbedo, uv).rgb;
    vec3  encN    = texture(uNormal, uv).rgb;
    vec3  rmf     = texture(uRMF,    uv).rgb;
    vec3  emiss   = texture(uEmissive, uv).rgb;
    float depth   = texture(uDepth,   uv).r;

    vec3  N         = normalize(encN * 2.0 - 1.0);
    float roughness = max(rmf.r, 0.04);
    float metallic  = rmf.g;
    float F0Scalar  = rmf.b;

    // World-pos reconstruction: NDC = (uv*2-1, depth). Vulkan depth range is
    // [0,1] (GLM_FORCE_DEPTH_ZERO_TO_ONE) and our projection Y is pre-flipped
    // on the CPU side, so uv.y maps to NDC.y directly.
    vec4 ndc       = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 worldH    = pc.InvViewProj * ndc;
    vec3 worldPos  = worldH.xyz / worldH.w;

    vec4 eyeH      = pc.InvViewProj * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 eyePos    = eyeH.xyz / max(eyeH.w, 1e-5);
    vec3 V         = normalize(eyePos - worldPos);

    vec3  L     = normalize(pc.SunDirAndIntensity.xyz);
    vec3  H     = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-3);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    vec3 F0 = mix(vec3(F0Scalar), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float Vt = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(VdotH, F0);

    vec3 specular = D * Vt * F;
    vec3 kS       = F;
    vec3 kD       = (1.0 - kS) * (1.0 - metallic);
    vec3 diffuse  = kD * albedo / kPI;

    vec3 sunRadiance = vec3(1.0) * pc.SunDirAndIntensity.w;
    vec3 lit = (diffuse + specular) * sunRadiance * NdotL;

    // Phase 8 IBL: replaces the flat ambient when enabled. Split-sum
    // approximation — diffuse from irradiance cube, specular from prefiltered
    // radiance + BRDF integration LUT.
    vec3 ambient;
    if (pc.IblParams.x >= 0.5) {
        vec3  R         = reflect(-V, N);
        vec3  irr       = texture(uIrradiance, N).rgb;
        vec3  Fr        = F_SchlickRoughness(NdotV, F0, roughness);
        vec3  kD_ibl    = (vec3(1.0) - Fr) * (1.0 - metallic);

        float mipCount  = max(pc.IblParams.z - 1.0, 0.0);
        vec3  prefilt   = textureLod(uPrefilter, R, roughness * mipCount).rgb;
        vec2  brdf      = texture(uBrdfLut, vec2(NdotV, roughness)).rg;
        vec3  specIBL   = prefilt * (Fr * brdf.x + brdf.y);

        ambient = (kD_ibl * irr * albedo + specIBL) * pc.IblParams.y;
    } else {
        ambient = albedo * pc.AmbientAndNearFar.x;
    }

    return lit + ambient + emiss;
}

// -- SDF slice (Phase 12) ----------------------------------------------------
// Intersect a world-space horizontal plane at y = uSdf.AABBMinAndPlaneY.w with
// the camera ray, then sample the 3D SDF and return a sign-coloured ramp.
//   negative (inside) → red
//   zero (iso)        → green
//   positive (outside)→ blue
// Returns rgb=0 + a=0 when the ray misses the plane or the SDF box.
vec4 SDFSlice(vec2 uv, float depth) {
    vec3 aabbMin = uSdf.AABBMinAndPlaneY.xyz;
    vec3 aabbMax = uSdf.AABBMaxAndMaxDist.xyz;
    float planeY = uSdf.AABBMinAndPlaneY.w;
    float maxDist = max(uSdf.AABBMaxAndMaxDist.w, 1e-4);

    // Reconstruct camera ray from NDC (matches PbrLit's world-pos reconstruction
    // — depth==1.0 means far plane).
    vec4 nearH = pc.InvViewProj * vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 farH  = pc.InvViewProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 nearP = nearH.xyz / max(nearH.w, 1e-6);
    vec3 farP  = farH.xyz  / max(farH.w,  1e-6);
    vec3 rd    = normalize(farP - nearP);
    vec3 ro    = nearP;

    // Solve ro.y + t*rd.y = planeY. If parallel or behind camera, no hit.
    if (abs(rd.y) < 1e-4) return vec4(0.0);
    float t = (planeY - ro.y) / rd.y;
    if (t <= 0.0) return vec4(0.0);

    // If the GBuffer has geometry in front of this slice intersection, render
    // the geometry colour instead (slice is occluded). Conservative: if
    // geometry exists at this pixel and is closer, fall back to a darkened
    // lit colour so the silhouette stays readable.
    vec3 hitPos = ro + rd * t;
    if (hitPos.x < aabbMin.x || hitPos.x > aabbMax.x ||
        hitPos.z < aabbMin.z || hitPos.z > aabbMax.z ||
        planeY  < aabbMin.y || planeY  > aabbMax.y) {
        return vec4(0.0);
    }

    // World → SDF UV (per-axis box mapping).
    vec3 boxSize = max(aabbMax - aabbMin, vec3(1e-6));
    vec3 uvw = (hitPos - aabbMin) / boxSize;
    // R16_SNORM hardware gives [-1, 1]; decode to world units.
    float snorm = texture(uMeshSDF, uvw).r;
    float d     = snorm * maxDist;

    // Saturate to the cell-distance scale so a bake at 64^3 still shows readable
    // gradient: divide by 0.5 × longest box axis as the visual unit, clamped.
    float visScale = 0.5 * max(boxSize.x, max(boxSize.y, boxSize.z));
    float vis = clamp(d / visScale, -1.0, 1.0);
    int vizMode = int(uSdf.PreviewParams.x);
    vec3 col;
    if (vizMode == 1) {
        // Heatmap (jet-ish ramp on absolute distance).
        float t = clamp(abs(vis), 0.0, 1.0);
        vec3 c0 = vec3(0.00, 0.00, 0.50);
        vec3 c1 = vec3(0.00, 0.80, 1.00);
        vec3 c2 = vec3(0.00, 1.00, 0.30);
        vec3 c3 = vec3(1.00, 0.95, 0.10);
        vec3 c4 = vec3(0.95, 0.10, 0.05);
        col = mix(c0, c1, smoothstep(0.0, 0.25, t));
        col = mix(col, c2, smoothstep(0.25, 0.50, t));
        col = mix(col, c3, smoothstep(0.50, 0.75, t));
        col = mix(col, c4, smoothstep(0.75, 1.00, t));
        if (vis < 0.0) col *= 0.55;
    } else if (vizMode == 2) {
        // Grayscale magnitude.
        float t = clamp(abs(vis), 0.0, 1.0);
        col = vec3(t);
    } else if (vizMode == 3) {
        // Signed B/W: inside black, outside white, thin iso band.
        col = (vis < 0.0) ? vec3(0.02) : vec3(0.95);
        float iso = exp(-220.0 * vis * vis);
        col = mix(col, vec3(0.95, 0.80, 0.20), iso);
    } else if (vizMode == 4) {
        // Gradient magnitude via voxel-spaced finite differences (central).
        vec3 vox = 1.0 / vec3(textureSize(uMeshSDF, 0));
        float dx = (texture(uMeshSDF, uvw + vec3(vox.x, 0.0, 0.0)).r -
                    texture(uMeshSDF, uvw - vec3(vox.x, 0.0, 0.0)).r) * maxDist;
        float dy = (texture(uMeshSDF, uvw + vec3(0.0, vox.y, 0.0)).r -
                    texture(uMeshSDF, uvw - vec3(0.0, vox.y, 0.0)).r) * maxDist;
        float dz = (texture(uMeshSDF, uvw + vec3(0.0, 0.0, vox.z)).r -
                    texture(uMeshSDF, uvw - vec3(0.0, 0.0, vox.z)).r) * maxDist;
        // Cell width along each axis: (2 * vox) * boxSize. Magnitude in
        // world units / world units → ideally 1.0 for a clean SDF.
        vec3 cell = 2.0 * vox * boxSize;
        vec3 g = vec3(dx / max(cell.x, 1e-6),
                      dy / max(cell.y, 1e-6),
                      dz / max(cell.z, 1e-6));
        float gmag = length(g);
        // Deviation from ideal 1.0 → blue=under, green=ideal, red=over.
        float dev = clamp(gmag, 0.0, 2.0);
        if (dev <= 1.0) col = mix(vec3(0.05, 0.10, 0.55), vec3(0.10, 0.85, 0.20), dev);
        else            col = mix(vec3(0.10, 0.85, 0.20), vec3(0.90, 0.15, 0.05), dev - 1.0);
    } else {
        // 0 = Signed RGB (default — original Phase 12 viz).
        if (vis < 0.0) {
            col = mix(vec3(0.05, 0.4, 0.05), vec3(0.9, 0.05, 0.05), -vis);
        } else {
            col = mix(vec3(0.05, 0.4, 0.05), vec3(0.05, 0.15, 0.9), vis);
        }
        float iso = exp(-160.0 * vis * vis);
        col = mix(col, vec3(0.2, 1.0, 0.4), iso);
    }

    // Composite onto whatever the GBuffer / lighting already produced when the
    // geometry sits in front of the slice. depth==1.0 → sky → slice fully
    // opaque; depth<1.0 + geometry closer than the slice → blend so the
    // silhouette stays readable.
    float alpha = (depth >= 1.0) ? 1.0 : 0.7;
    return vec4(col, alpha);
}

// -- entry --------------------------------------------------------------------

void main() {
    int mode = int(pc.AmbientAndNearFar.w);

    float depth = texture(uDepth, vUV).r;
    // SDF slice mode needs to render against the sky too, so it owns its own
    // discard rule.
    if (mode != 8 && depth >= 1.0) {
        discard;
    }

    vec3 result = vec3(0.0);
    if (mode == 1) {
        result = texture(uAlbedo, vUV).rgb;
    } else if (mode == 2) {
        result = texture(uNormal, vUV).rgb;
    } else if (mode == 3) {
        result = texture(uRMF, vUV).rgb;
    } else if (mode == 4) {
        vec3 e = texture(uEmissive, vUV).rgb;
        result = e / (e + 1.0);
    } else if (mode == 5) {
        result = vec3(Linearise(depth, pc.AmbientAndNearFar.y, pc.AmbientAndNearFar.z));
    } else if (mode == 6) {
        result = IdentityColor(texture(uIdentity, vUV).rg);
    } else if (mode == 7) {
        vec3 encN = texture(uNormal, vUV).rgb;
        vec3 N    = normalize(encN * 2.0 - 1.0);
        result = Matcap(N);
    } else if (mode == 8) {
        vec4 slice = SDFSlice(vUV, depth);
        if (slice.a <= 0.001) {
            // No slice contribution at this pixel — paint sky/grid (let grid
            // pass below show through) by discarding.
            discard;
        }
        result = slice.rgb;
    } else {
        // Phase 10: Lit mode is now just a sample of LightHDR, which
        // lighting.comp filled this frame with Karis/Schlick/Smith + IBL +
        // shadow compose.
        result = texture(uLightHDR, vUV).rgb;
    }

    oColor = vec4(result, 1.0);
}
