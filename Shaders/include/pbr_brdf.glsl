// Shaders/include/pbr_brdf.glsl
// Shared PBR BRDF library for lighting.comp, lighting_pcss.comp,
// lighting_vsm.comp, gbuffer_preview.frag, and any future variant.
//
// "Cheap" path: classic Karis 2013 Cook-Torrance — Lambert diffuse,
// uncompensated Smith-GGX specular, scalar Schlick Fresnel.
//
// "Realistic" path (bit 0 of LightingFlags):
//   * Diffuse  : Bosch et al. 2024, "EON — energy-preserving Oren-Nayar"
//                (JCGT, https://www.jcgt.org/published/0013/04/01/).
//                Replaces Lambert with a closed-form energy-conserving
//                rough-diffuse model that subsumes Oren-Nayar.
//   * Specular : adds Fdez-Aguera 2019 multi-scatter energy compensation
//                ("A Multiple-Scattering Microfacet Model for Real-Time
//                Image-Based Lighting"). Uses the two channels already in
//                the BRDF split-sum LUT — no extra LUT needed.
//   * Fresnel  : KHR_materials_specular semantics. Dielectric F0 = 0.04 *
//                specularFactor * specularColor; metallic ignores both
//                (F0 = baseColor, as before).

#ifndef PBR_BRDF_GLSL_INCLUDED
#define PBR_BRDF_GLSL_INCLUDED

const float kPI    = 3.14159265359;
const float kInvPI = 0.31830988618;

// Flag bits packed into LightingPushConstants.IblParams.w (as float bits).
const uint kPbrFlagRealistic = 1u << 0;

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

// Karis Fresnel-with-roughness for IBL — clamps the Fresnel grazing-angle
// term when sampling a low-resolution prefilter.
vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    return F0 + Fr * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// KHR_materials_specular F0:
//   dielectric F0 = 0.04 * specFactor * specColor   (default 0.04 white)
//   metallic   F0 = albedo                          (specColor ignored)
vec3 ComputeF0_KHR(vec3 albedo, float metallic, vec3 specColor, float specFactor) {
    vec3 dielectric = vec3(0.04) * specFactor * specColor;
    return mix(dielectric, albedo, metallic);
}

// Energy-preserving Oren-Nayar (Bosch et al. 2024).
// LdotV = dot(L, V). All other dots are NdotX with N normalised.
//
// Reduces to Lambert as roughness → 0; correctly preserves energy at high
// roughness (Lambert does not). The multi-scatter term (rho² * 0.1159 * r)
// is the closed-form fit from the paper.
vec3 Diffuse_EON(vec3 albedo, float NdotV, float NdotL, float LdotV, float roughness) {
    float r       = max(roughness, 1e-3);
    float s       = LdotV - NdotL * NdotV;
    float sOverT  = (s <= 0.0) ? s : s / max(max(NdotL, NdotV), 1e-4);
    float A       = 1.0 / (1.0 + (0.5 - 2.0 / (3.0 * kPI)) * r);
    float B       = r * A;
    vec3  single  = albedo * kInvPI * (A + B * sOverT);
    vec3  multi   = albedo * albedo * (0.1159 * r);
    return single + multi;
}

// Fdez-Aguera 2019 single-bounce -> multi-bounce energy compensation.
// E_ss = directional albedo of single-scattering specular = brdfLut.r + brdfLut.g.
// Returns a multiplier you apply to ANY specular term (direct or IBL) to
// add back the energy lost to multi-bounce GGX.
//
// brdfLut here is `texture(uBrdfLut, vec2(NdotV, roughness)).rg`.
vec3 EnergyCompFactor(vec3 F0, vec2 brdfLut) {
    float Ess = max(brdfLut.x + brdfLut.y, 1e-4);
    return vec3(1.0) + F0 * (1.0 / Ess - 1.0);
}

// Decode the flag word from IblParams.w (transported as float bits).
uint UnpackPbrFlags(float packed) {
    return floatBitsToUint(packed);
}

bool IsRealisticPbr(uint flags) {
    return (flags & kPbrFlagRealistic) != 0u;
}

#endif // PBR_BRDF_GLSL_INCLUDED
