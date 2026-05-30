// rc_hash.glsl — Phase 14b shared hash + SH primitives.
//
// Consumers must declare the 4 set-bindings (HashKeys, ProbePayload,
// ParamsUbo, CellList) at whichever set index they need before #include-ing
// this. The helpers below operate on `gRC_Keys`, `gRC_Payload`, etc. so a
// macro wires the per-shader binding into the helper's name.
//
// Bindings expected by the helpers (declared by the consumer):
//   layout(std430, set=S, binding=0) buffer RC_KeyBuf     { uint  gRC_Keys[];     };
//   layout(std430, set=S, binding=1) buffer RC_PayloadBuf { vec4  gRC_Payload[];  };
//   layout(std140, set=S, binding=2) uniform RC_ParamsBuf { RcParams gRC_Params; } gRC;
//   layout(std430, set=S, binding=3) buffer RC_CellList   { uint  gRC_CellList[]; };
//
// Payload layout per cell (1 vec4 = 16 bytes, matches kBytesPerPayload):
//   gRC_Payload[cellIdx] = vec4(Y_00, Y_1y, Y_1z, Y_1x)   // luma SH-L1
// Chroma is recovered at compose time by multiplying SH irradiance × GBuffer
// albedo. Phase 14c can promote to full RGB SH (48 B / cell) if the smoke
// test shows obvious chroma bleeding.

#ifndef RC_HASH_GLSL_INCLUDED
#define RC_HASH_GLSL_INCLUDED

// Pinned: matches RcHashParams in C++ (Source/GI/RadianceCascadeHash.h).
struct RcParams {
    vec4 EyePosAndHashLog2;
    vec4 SunDirAndIntensity;
    vec4 SunColor;
    vec4 CascadeParams;       // x=s0, y=cascadeCount, z=r0Rays, w=indirectBoost
    vec4 InvViewProj0;
    vec4 InvViewProj1;
    vec4 InvViewProj2;
    vec4 InvViewProj3;
    vec4 RenderExtentAndFlags;
    vec4 SdfAabbMin;
    vec4 SdfAabbMax;          // .w = decode scale
    vec4 SecondaryParams;     // x=radiusL0, y=growPerCascade, z=frameIdx, w=_
};

// ---------------------------------------------------------------------------
// Cell key packing
// ---------------------------------------------------------------------------
// Key layout (uint32, 0 = empty sentinel):
//   bits [0..6]   = x (7 bits, 0..127)
//   bits [7..13]  = y
//   bits [14..20] = z
//   bits [21..23] = cascade (3 bits, 0..7)
//   bits [24..30] = (reserved / extra entropy)
//   bit  31       = occupied flag (always 1 for valid keys, so key != 0)
//
// World → cell:
//   cellSpacing = s0 * 2^cascade
//   cellIdx = floor((worldPos - eye) / cellSpacing)
// We anchor to camera eye so the cascades follow the viewer.

uint RC_PackKey(ivec3 cell, uint cascade) {
    uvec3 u = uvec3(cell & ivec3(0x7F));   // mask to 7 bits each
    uint k =  (u.x      )
            | (u.y <<  7)
            | (u.z << 14)
            | ((cascade & 0x7u) << 21)
            | (1u << 31);                  // occupied flag
    return k;
}

void RC_UnpackKey(uint key, out ivec3 cell, out uint cascade) {
    cell.x  = int( key        & 0x7Fu);
    cell.y  = int((key >>  7) & 0x7Fu);
    cell.z  = int((key >> 14) & 0x7Fu);
    cascade = (key >> 21) & 0x7u;
    // Re-extend sign so cells "behind the eye" don't lookup the wrong slot.
    // 7-bit signed has range -64..63; bit 6 is the sign bit.
    if ((cell.x & 0x40) != 0) cell.x |= ~0x7F;
    if ((cell.y & 0x40) != 0) cell.y |= ~0x7F;
    if ((cell.z & 0x40) != 0) cell.z |= ~0x7F;
}

// Splitmix-style integer hash. Cheap, decent avalanche. Drop-in fine for the
// 2^20-slot table; Taichi C3 worst probe length = 3.
uint RC_HashWord(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// ---------------------------------------------------------------------------
// Cascade selection
// ---------------------------------------------------------------------------
// Cascade c covers radius ~ s0 * 2^c. Pick the lowest cascade whose half-extent
// contains the point relative to the eye.
uint RC_PickCascade(vec3 worldPos, vec3 eyePos, float s0, int cascadeCount) {
    float dist = length(worldPos - eyePos);
    // Conservative: cascade c covers [s0 * 2^c * 4, s0 * 2^(c+1) * 4) in radius.
    // Anchor the 4× factor so c0 reaches several metres before we drop to c1.
    float d = max(dist, 0.0001);
    float lvlF = log2(d / max(s0, 0.0001)) - 2.0;     // -2 → 4× slack
    int   lvl  = int(clamp(floor(lvlF), 0.0, float(cascadeCount - 1)));
    return uint(lvl);
}

// Cell pitch (world units) for a given cascade.
float RC_CellPitch(float s0, uint cascade) {
    return s0 * float(1u << cascade);
}

ivec3 RC_WorldToCell(vec3 worldPos, vec3 eyePos, float s0, uint cascade) {
    float pitch = RC_CellPitch(s0, cascade);
    vec3  rel   = (worldPos - eyePos) / pitch;
    return ivec3(floor(rel));
}

vec3 RC_CellCenterWorld(ivec3 cell, vec3 eyePos, float s0, uint cascade) {
    float pitch = RC_CellPitch(s0, cascade);
    return eyePos + (vec3(cell) + vec3(0.5)) * pitch;
}

// ---------------------------------------------------------------------------
// SH-L1 helpers
// ---------------------------------------------------------------------------
// 4 bands × 1 channel (luma SH). Y_00 = 0.282, Y_1m1/Y_10/Y_11 = 0.488 · dir.
// We modulate by the GBuffer albedo at compose time so cells stay compact.

const float kSH_Y00 = 0.282094792;
const float kSH_Y1  = 0.488602512;

// Project a directional radiance sample into 4-band luma SH.
vec4 RC_ProjectShL1(vec3 dir, float radianceLuma) {
    return radianceLuma * vec4(kSH_Y00,
                               kSH_Y1 * dir.y,
                               kSH_Y1 * dir.z,
                               kSH_Y1 * dir.x);
}

// Evaluate luma irradiance from a 4-band SH coeff vector at normal N.
// Lambert convolution coefficients (Ramamoorthi/Hanrahan): A0=pi, A1=2pi/3.
float RC_EvalShL1Irradiance(vec4 sh, vec3 N) {
    const float A0 = 3.14159265 * kSH_Y00;
    const float A1 = (2.0 * 3.14159265 / 3.0) * kSH_Y1;
    return max(sh.x * A0
             + sh.y * A1 * N.y
             + sh.z * A1 * N.z
             + sh.w * A1 * N.x, 0.0);
}

#endif // RC_HASH_GLSL_INCLUDED
