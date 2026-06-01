// Shaders/include/atmosphere.glsl — Phase B
// Shared constants + helpers for the Hillaire 2020 sky/atmosphere LUTs.
//
// All distances/altitudes are in **kilometres** inside this file. The world
// space used by the rest of the renderer is metres, so the SkyView consumer
// (sky_cube_from_skyview.comp + future fog-on-geometry) scales at the LUT
// lookup boundary, not inside.
//
// Coefficient reference: Bruneton 2017 "Precomputed Atmospheric Scattering: a
// New Implementation" §4 sea-level numbers; Hillaire 2020 SIGGRAPH course
// notes adopt the same.
#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL

// ---- Push-constant feed --------------------------------------------------
// Every Hillaire compute shader pulls the same atmosphere parameters from a
// shared push-constant struct. The host packs this so we never have to keep
// per-shader copies in sync.
struct AtmosphereParams {
    vec4 RayleighScatteringAndPlanetRadius; // rgb = beta_R (1/m), w = R_planet (km)
    vec4 MieAndAtmoTopAndG;                 // x = beta_M (1/m), y = R_top (km), z = mie g, w = unused
    vec4 OzoneAbsorption;                   // rgb = ozone absorption (1/m), w = unused
};

// ---- Constants -----------------------------------------------------------
// Atmospheric scale heights (Bruneton 2017 §4). Layer thickness above which
// density falls by 1/e.
const float kRayleighScaleHeightKm = 8.0;
const float kMieScaleHeightKm      = 1.2;
// Ozone layer is a Chapman bump centred at 25 km with ±15 km tent.
const float kOzoneCenterKm         = 25.0;
const float kOzoneWidthKm          = 15.0;
const float kPi                    = 3.14159265359;
const float kFourPi                = 12.5663706144;

// ---- Density profiles (altitude in km above planet centre minus R_planet) -
float RayleighDensity(float altKm) {
    return exp(-max(altKm, 0.0) / kRayleighScaleHeightKm);
}
float MieDensity(float altKm) {
    return exp(-max(altKm, 0.0) / kMieScaleHeightKm);
}
float OzoneDensity(float altKm) {
    // Tent shape, peak 1.0 at 25 km, zero outside [10, 40] km. Matches the
    // Bruneton "atmosphere.glsl" reference (Δ-shaped profile).
    float d = abs(altKm - kOzoneCenterKm) / kOzoneWidthKm;
    return max(0.0, 1.0 - d);
}

// ---- Phase functions -----------------------------------------------------
float RayleighPhase(float cosTheta) {
    // 3 / (16π) * (1 + cos²θ)
    return (3.0 / (16.0 * kPi)) * (1.0 + cosTheta * cosTheta);
}
float MiePhase(float cosTheta, float g) {
    // Henyey-Greenstein.
    float g2     = g * g;
    float denom  = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (kFourPi * pow(max(denom, 1e-4), 1.5));
}

// ---- Ray / sphere intersection -------------------------------------------
// Returns the nearest non-negative t along (ro + t*rd) where the ray hits a
// sphere of `radius` centred at origin. Returns -1 if no positive hit.
float RaySphereIntersectNearest(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float d = b * b - c;
    if (d < 0.0) return -1.0;
    float s = sqrt(d);
    float t0 = -b - s;
    float t1 = -b + s;
    if (t0 >= 0.0) return t0;
    if (t1 >= 0.0) return t1;
    return -1.0;
}

// Convenience for "exit the atmosphere shell" — assumes ro is inside.
float RaySphereExit(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float d = b * b - c;
    if (d < 0.0) return 0.0;
    return -b + sqrt(d); // far root, always >= 0 when inside the sphere
}

// ---- Coefficients packed from push constants -----------------------------
// Helper to pull beta_R / beta_M / ozone / radii out of the shared struct so
// callers don't repeat the unpack.
struct AtmoCoeffs {
    vec3  BetaRayleigh;    // 1/m
    float BetaMie;         // 1/m, isotropic in colour (single scalar)
    vec3  OzoneAbs;        // 1/m
    float PlanetRadiusKm;
    float AtmoRadiusKm;
    float MieG;
};

AtmoCoeffs UnpackAtmoCoeffs(AtmosphereParams p) {
    AtmoCoeffs c;
    c.BetaRayleigh   = p.RayleighScatteringAndPlanetRadius.rgb;
    c.PlanetRadiusKm = p.RayleighScatteringAndPlanetRadius.w;
    c.BetaMie        = p.MieAndAtmoTopAndG.x;
    c.AtmoRadiusKm   = p.MieAndAtmoTopAndG.y;
    c.MieG           = p.MieAndAtmoTopAndG.z;
    c.OzoneAbs       = p.OzoneAbsorption.rgb;
    return c;
}

// ---- Extinction at a sample altitude -------------------------------------
// Returns extinction coefficient (1/m) in RGB. Used by every raymarch step.
vec3 SampleExtinction(AtmoCoeffs c, float altKm) {
    float rho_r = RayleighDensity(altKm);
    float rho_m = MieDensity(altKm);
    float rho_o = OzoneDensity(altKm);
    // Mie extinction is roughly 1.1 × scattering (small absorption term per
    // Bruneton); we approximate by 1.0 here so the LUT stays single-scatter
    // clean. The 10% under-extinction is buried under the multiscatter pass.
    return c.BetaRayleigh * rho_r
         + vec3(c.BetaMie) * rho_m
         + c.OzoneAbs      * rho_o;
}

// Scattering coefficients (no ozone, no Mie absorption — for inscatter integration).
void SampleScattering(AtmoCoeffs c, float altKm, out vec3 sRayleigh, out float sMie) {
    sRayleigh = c.BetaRayleigh * RayleighDensity(altKm);
    sMie      = c.BetaMie      * MieDensity(altKm);
}

// ---- Altitude helper -----------------------------------------------------
// pos is in km relative to the planet centre. Returns altitude in km above
// the planet surface.
float AltitudeKm(vec3 pos, float planetRadiusKm) {
    return max(length(pos) - planetRadiusKm, 0.0);
}

// ---- LUT parameterisation ------------------------------------------------
// Transmittance LUT: u = cos(view-zenith) mapped via Bruneton-style remap so
// the horizon resolves with more texels. We use the simpler "linear in
// altitudeRatio, nonlinear-in-mu" mapping from Hillaire's reference code.
vec2 TransmittanceUVFromRMu(float r, float mu, float planetR, float atmoR) {
    // r  = current radius (km, planet centre).
    // mu = cos(view zenith).
    float H   = sqrt(max(atmoR * atmoR - planetR * planetR, 1e-6));
    float rho = sqrt(max(r * r - planetR * planetR, 0.0));
    float discriminant = r * r * (mu * mu - 1.0) + atmoR * atmoR;
    float d = max(-r * mu + sqrt(max(discriminant, 0.0)), 0.0);
    float dMin = atmoR - r;
    float dMax = rho + H;
    float xMu  = (dMax > dMin) ? (d - dMin) / (dMax - dMin) : 0.0;
    float xR   = (H > 0.0)     ? (rho / H)                 : 0.0;
    return vec2(clamp(xMu, 0.0, 1.0), clamp(xR, 0.0, 1.0));
}

void RMuFromTransmittanceUV(vec2 uv, float planetR, float atmoR,
                            out float r, out float mu) {
    float H   = sqrt(max(atmoR * atmoR - planetR * planetR, 1e-6));
    float rho = uv.y * H;
    r = sqrt(rho * rho + planetR * planetR);
    float dMin = atmoR - r;
    float dMax = rho + H;
    float d    = dMin + uv.x * (dMax - dMin);
    mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

// SkyView LUT is parameterised by (longitude, latitude-from-horizon) so the
// horizon band gets dense texels. We use a square-root remap in latitude to
// stretch the horizon — matches Hillaire 2020 §5.3.
vec2 SkyViewUVFromDir(vec3 dir, vec3 up) {
    // Pull the latitude from horizon (= 90° - view-zenith angle).
    float cosTheta = clamp(dot(dir, up), -1.0, 1.0);
    float lat      = asin(cosTheta);                  // [-π/2, +π/2]
    float v        = 0.5 + 0.5 * sign(lat) * sqrt(abs(lat) * 2.0 / kPi);
    // Longitude: project dir onto plane orthogonal to up, atan2 of x/z.
    vec3 perp = dir - up * cosTheta;
    float lon = atan(perp.x, -perp.z);                // wraps [-π, π]
    float u   = (lon + kPi) / (2.0 * kPi);
    return vec2(u, v);
}

vec3 DirFromSkyViewUV(vec2 uv, vec3 up, vec3 right, vec3 forward) {
    // Inverse of SkyViewUVFromDir under the (right, up, forward) local frame.
    float lon = uv.x * 2.0 * kPi - kPi;
    float v   = uv.y * 2.0 - 1.0;
    float lat = sign(v) * (v * v) * 0.5 * kPi;
    float c   = cos(lat);
    return normalize(right * (c * sin(lon)) + up * sin(lat) + forward * (-c * cos(lon)));
}

#endif // ATMOSPHERE_GLSL
