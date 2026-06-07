// rc_hash.glsl — canonical 3D radiance cascades primitives (sparse hash).
//
// Faithful to Sannikov's paper:
//   - probe spacing      Δp(c) = s0 · 2^c          (world-anchored grid)
//   - directions/probe   D(c)  = D0 · 4^c          (octahedral map, res R0·2^c)
//   - radiance interval  [s0·(2^c − 1), s0·(2^(c+1) − 1))   — disjoint, contiguous,
//                        doubling lengths, cascade 0 starts at 0
//   - merge operator     L = near.rgb + near.β · far.rgb ;  β = near.β · far.β
//     (β = transparency of the interval: 1 = ray escaped, 0 = hit)
//
// Sparsity: per-cascade hash sub-tables. Cascade c gets Base >> 2c slots, so
// slots·dirs = Base·D0 is constant per cascade and every cascade's payload
// block is the same size (the paper's memory bound). All passes for one frame
// (insert → relight top-down → resolve → compose) run on the same per-frame
// buffer set, so probes are rebuilt deterministically every frame with FIXED
// octahedral directions — no jitter, no temporal blend, no noise.
//
// Consumers declare the set bindings before #include-ing:
//   layout(std430, set=S, binding=0) buffer RC_KeyBuf     { uint gRC_Keys[];     };
//   layout(std430, set=S, binding=1) buffer RC_PayloadBuf { vec4 gRC_Payload[];  };
//   layout(std140, set=S, binding=2) uniform RC_ParamsBuf { RcParams gRC_P; };
//   layout(std430, set=S, binding=3) buffer RC_CellList   { uint gRC_CellList[]; };
//   layout(std430, set=S, binding=4) buffer RC_ResolveBuf { vec4 gRC_Resolve[];  };
// (declare only the bindings the shader actually uses)

#ifndef RC_HASH_GLSL_INCLUDED
#define RC_HASH_GLSL_INCLUDED

// Pinned: matches RcHashParams in C++ (Source/GI/RadianceCascadeHash.h).
struct RcParams {
    vec4 EyePosAndBaseLog2;   // xyz = camera eye world, w = log2(base slots)
    vec4 SunDirAndIntensity;  // xyz = direction toward sun, w = intensity
    vec4 SunColor;            // rgb, w = ambient sky luma
    vec4 CascadeParams;       // x = s0 (m), y = cascadeCount, z = D0, w = boost
    vec4 InvViewProj0;
    vec4 InvViewProj1;
    vec4 InvViewProj2;
    vec4 InvViewProj3;
    vec4 RenderExtentAndFlags;// x/y = render w/h, z = bilinearFix, w = debugView
    vec4 MiscParams;          // x = frameIndex, y = scaleLocalPerWorld, zw unused
    vec4 WorldToLocal0;       // world→mesh-local (column-major) for SDF tracing
    vec4 WorldToLocal1;
    vec4 WorldToLocal2;
    vec4 WorldToLocal3;
    vec4 SdfAlbedo;           // rgb = SDF mesh albedo tint for bounced light
};

// ---------------------------------------------------------------------------
// Cascade geometry (world-anchored — probes do NOT move with the camera)
// ---------------------------------------------------------------------------

float RC_CellPitch(float s0, uint cascade) {
    return s0 * float(1u << cascade);
}

ivec3 RC_WorldToCell(vec3 worldPos, float pitch) {
    return ivec3(floor(worldPos / pitch));
}

vec3 RC_CellCenterWorld(ivec3 cell, float pitch) {
    return (vec3(cell) + vec3(0.5)) * pitch;
}

// Disjoint radiance interval for cascade c: [s0·(2^c − 1), s0·(2^(c+1) − 1)).
float RC_IntervalStart(float s0, uint cascade) {
    return s0 * (float(1u << cascade) - 1.0);
}
float RC_IntervalEnd(float s0, uint cascade) {
    return s0 * (float(2u << cascade) - 1.0);
}

// ---------------------------------------------------------------------------
// Per-cascade sub-table layout
// ---------------------------------------------------------------------------
// base = 1 << BaseLog2 (from params). Cascade c:
//   slots(c)          = base >> 2c            (min base log2 14 with 5 cascades
//                                              keeps slots(4) = 64 ≥ workgroup)
//   keyOffset(c)      = Σ_{i<c} slots(i)
//   dirCount(c)       = D0 << 2c              (D0 = 16 → oct res 4·2^c)
//   payloadOffset(c)  = c · base · D0         (slots·dirs is constant!)
//   cellListOffset(c) = c · (base + 1)        ([offset] = atomic counter,
//                                              [offset+1 ..] = packed keys)

uint RC_CascadeSlots(uint base, uint c)    { return base >> (2u * c); }
uint RC_DirCount(uint d0, uint c)          { return d0 << (2u * c); }
uint RC_OctRes(uint c)                     { return 4u << c; }        // sqrt(D), D0=16

uint RC_KeyOffset(uint base, uint c) {
    uint off = 0u;
    for (uint i = 0u; i < c; ++i) off += base >> (2u * i);
    return off;
}
uint RC_PayloadOffset(uint base, uint d0, uint c)  { return c * base * d0; }
uint RC_CellListOffset(uint base, uint c)          { return c * (base + 1u); }

// ---------------------------------------------------------------------------
// Cell key packing — 9 signed bits per axis (range −256..255 cells), 3 bits
// cascade (bits 27..29), bit 30 spare, bit 31 = occupied flag so any valid
// key is nonzero (0 stays the empty-slot sentinel).
// ---------------------------------------------------------------------------

uint RC_PackKey(ivec3 cell, uint cascade) {
    uvec3 u = uvec3(cell & ivec3(0x1FF));
    return  (u.x       )
          | (u.y <<  9 )
          | (u.z << 18 )
          | ((cascade & 0x7u) << 27)
          | (1u << 31);
}

void RC_UnpackKey(uint key, out ivec3 cell, out uint cascade) {
    cell.x  = int( key         & 0x1FFu);
    cell.y  = int((key >>  9u) & 0x1FFu);
    cell.z  = int((key >> 18u) & 0x1FFu);
    cascade = (key >> 27u) & 0x7u;
    if ((cell.x & 0x100) != 0) cell.x |= ~0x1FF;
    if ((cell.y & 0x100) != 0) cell.y |= ~0x1FF;
    if ((cell.z & 0x100) != 0) cell.z |= ~0x1FF;
}

uint RC_HashWord(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

#define RC_PROBE_CAP 16u

// ---------------------------------------------------------------------------
// Octahedral direction mapping (fixed, deterministic — RC's noiselessness
// depends on every probe tracing the SAME directions every frame).
// ---------------------------------------------------------------------------

vec3 RC_OctDecode(vec2 e) {                    // e in [-1,1]^2
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Texel (x, y) of an R×R octahedral map → unit direction.
vec3 RC_TexelDir(uint x, uint y, uint res) {
    vec2 uv = (vec2(float(x), float(y)) + 0.5) / float(res);
    return RC_OctDecode(uv * 2.0 - 1.0);
}

// ---------------------------------------------------------------------------
// Sky radiance — escape term for the top cascade and for missing parents.
// ---------------------------------------------------------------------------

vec3 RC_Sky(vec3 dir, vec3 sunColor, float ambient) {
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return sunColor * (ambient * (0.5 + 0.5 * up));
}

// ---------------------------------------------------------------------------
// SH-L1 (resolve output: 3 vec4 per c0 probe, one 4-band SH per channel)
// ---------------------------------------------------------------------------

const float kSH_Y00 = 0.282094792;
const float kSH_Y1  = 0.488602512;

vec4 RC_ProjectShL1(vec3 dir, float radiance) {
    return radiance * vec4(kSH_Y00,
                           kSH_Y1 * dir.y,
                           kSH_Y1 * dir.z,
                           kSH_Y1 * dir.x);
}

float RC_EvalShL1Irradiance(vec4 sh, vec3 N) {
    // Returns irradiance / π (the cosine-weighted mean incident radiance), so
    // compose's `albedo * irr` is correct Lambertian exit radiance — the BRDF's
    // 1/π is already folded in here. (Raw E would over-light by π; see notes.)
    const float A0 = kSH_Y00;
    const float A1 = (2.0 / 3.0) * kSH_Y1;
    return max(sh.x * A0
             + sh.y * A1 * N.y
             + sh.z * A1 * N.z
             + sh.w * A1 * N.x, 0.0);
}

#endif // RC_HASH_GLSL_INCLUDED
