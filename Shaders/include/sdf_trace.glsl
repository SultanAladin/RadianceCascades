// sdf_trace.glsl — shared SDF sphere-tracing primitives.
//
// Used by:
//   - lighting_sdfcone.comp (Phase 13 soft shadows)
//   - rc_relight.comp        (Phase 14b radiance-cascade probe relight)
//
// Design rules
// ------------
// 1. Binding-agnostic. The SDF sampler and its AABB/decode params are passed
//    in by the caller. lighting_sdfcone has them on set=2; rc_relight will
//    bind them on set=3 alongside the hash/payload SSBOs.
// 2. No global state. Every function reads only its parameters. Easier to
//    audit, easier to reuse, no surprise dependency on AlgoUbo layout.
// 3. The R16_SNORM SDF stores normalised [-1, 1] distance — multiply by
//    decodeScale (the bake's MaxDist) to get world units.
//
// Consumers must add this near the top of the file:
//   #extension GL_GOOGLE_include_directive : require
// and the build script must pass `-I<Shaders>` so the include path resolves.

#ifndef SDF_TRACE_GLSL_INCLUDED
#define SDF_TRACE_GLSL_INCLUDED

// Sample the SDF at a world-space point. Returns signed distance in world
// units. Points outside the AABB return a large positive sentinel (1e6) so
// downstream loops see "no occluder" and bail out cleanly.
float SDF_Sample(sampler3D sdfTex,
                 vec3 worldPos,
                 vec3 aabbMin,
                 vec3 aabbMax,
                 float decodeScale)
{
    vec3 boxSize = max(aabbMax - aabbMin, vec3(1e-6));
    vec3 uvw     = (worldPos - aabbMin) / boxSize;
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
        return 1e6;
    }
    float snorm = texture(sdfTex, uvw).r;
    return snorm * decodeScale;
}

// Conservative slab intersection. Returns world-space t to enter the SDF
// AABB from a ray (ro, rd). Negative if ro is already inside. -1.0 on miss.
// `rd` MUST be normalised.
float SDF_RayBoxEnter(vec3 ro, vec3 rd, vec3 aabbMin, vec3 aabbMax) {
    vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6)) * sign(rd);
    vec3 t1 = (aabbMin - ro) * invRd;
    vec3 t2 = (aabbMax - ro) * invRd;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    float tEnter = max(tmin.x, max(tmin.y, tmin.z));
    float tExit  = min(tmax.x, min(tmax.y, tmax.z));
    if (tExit < 0.0 || tEnter > tExit) return -1.0;
    return tEnter;
}

// -----------------------------------------------------------------------------
// Soft shadow (improved Iñigo Quilez, h*h/(2*y))
// -----------------------------------------------------------------------------
// Returns visibility in [0, 1]: 1.0 = fully lit, 0.0 = fully shadowed.
// `k` is the penumbra coefficient (1 / tan(half_angle)): larger = sharper.
//   ref: https://iquilezles.org/articles/rmshadows/
//
// Step floor `minStep` is in the same units as the SDF (mesh-local for the
// resident mesh SDF). The Taichi prototype `rc3d_render.py::soft_shadow` is
// the byte-equivalent reference — match its k * w / (t - y) form, not the
// inverted d / (k * (t - y)) form (which makes small k = soft, opposite of
// the standard IQ semantics).
float SDF_SoftShadow(sampler3D sdfTex,
                    vec3 ro,
                    vec3 rd,
                    float maxDist,
                    float k,
                    int maxSteps,
                    float minStep,
                    vec3 aabbMin,
                    vec3 aabbMax,
                    float decodeScale)
{
    // Skip empty space outside the SDF box. If ro is already inside (tEnter
    // < 0) the loop just starts at t=0, which is what we want.
    float tEnter = SDF_RayBoxEnter(ro, rd, aabbMin, aabbMax);
    if (tEnter < 0.0 && SDF_Sample(sdfTex, ro, aabbMin, aabbMax, decodeScale) >= 1e6) {
        // Ray misses the AABB entirely — no occluder along it.
        return 1.0;
    }
    float t = max(0.0, tEnter);
    if (t > maxDist) return 1.0;

    float res = 1.0;
    for (int i = 0; i < maxSteps; ++i) {
        if (t > maxDist) break;
        float h = SDF_Sample(sdfTex, ro + rd * t, aabbMin, aabbMax, decodeScale);
        if (h >= 1e5) {
            break;
        }
        if (h < 0.001) return 0.0;
        
        // Classic IQ soft shadow form. We drop the improved back-projection
        // (y = h^2/(2*ph)) because tri-linear interpolation of a discrete
        // voxel SDF breaks the exact Eikonal distance property (|grad| != 1).
        // This causes h to fluctuate non-monotonically, breaking the w term
        // and causing extreme banding / noisy "hazy spots" in the penumbra.
        // The classic form is much more robust to voxel SDFs.
        res = min(res, k * h / max(t, 1e-4));
        t  += max(h, minStep);
    }
    
    // Smoothstep to soften the hard transitions at the penumbra edges
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// -----------------------------------------------------------------------------
// Hard hit query (needed for Phase 14b radiance-interval extension)
// -----------------------------------------------------------------------------
// rc_relight.comp traces a ray at cascade c0 with budget [tMin, tMax]. If the
// ray hits within budget, sample direct lighting at the hit point. If it
// escapes (`hit == false`), the caller looks up the next-cascade SH probe at
// `outPos` — this is the C4-validated radiance-interval extension.
//
// Returns:
//   outPos        = world-space hit point (if hit) or ray endpoint at tMax (if escape)
//   outNormal     = surface normal at hit (central differences) — undefined on miss
//   outDistTravelled = parametric t along the ray
//   return        = true if hit, false if escaped maxDist
bool SDF_TraceHit(sampler3D sdfTex,
                  vec3 ro,
                  vec3 rd,
                  float tMin,
                  float tMax,
                  int maxSteps,
                  float minStep,
                  vec3 aabbMin,
                  vec3 aabbMax,
                  float decodeScale,
                  out vec3 outPos,
                  out vec3 outNormal,
                  out float outDistTravelled)
{
    // Start at max(tMin, slab enter) — caller's tMin is the bias-off-surface
    // offset (rc_relight wants ~half a cell to avoid self-intersection).
    float tEnter = SDF_RayBoxEnter(ro, rd, aabbMin, aabbMax);
    float t      = max(tMin, max(0.0, tEnter));

    for (int i = 0; i < maxSteps; ++i) {
        if (t > tMax) break;
        vec3 p = ro + rd * t;
        float h = SDF_Sample(sdfTex, p, aabbMin, aabbMax, decodeScale);
        if (h >= 1e5) {
            break;
        }
        if (h < 0.001) {
            outPos = p;
            outDistTravelled = t;
            // Central-difference normal in world units. Step size = half a
            // voxel decoded against the AABB — picks up real curvature
            // without aliasing the R16_SNORM quantisation.
            vec3 boxSize = max(aabbMax - aabbMin, vec3(1e-6));
            vec3 eps = boxSize * (0.5 / 64.0);   // half a voxel at 64^3 baseline
            float dx = SDF_Sample(sdfTex, p + vec3(eps.x, 0, 0), aabbMin, aabbMax, decodeScale)
                     - SDF_Sample(sdfTex, p - vec3(eps.x, 0, 0), aabbMin, aabbMax, decodeScale);
            float dy = SDF_Sample(sdfTex, p + vec3(0, eps.y, 0), aabbMin, aabbMax, decodeScale)
                     - SDF_Sample(sdfTex, p - vec3(0, eps.y, 0), aabbMin, aabbMax, decodeScale);
            float dz = SDF_Sample(sdfTex, p + vec3(0, 0, eps.z), aabbMin, aabbMax, decodeScale)
                     - SDF_Sample(sdfTex, p - vec3(0, 0, eps.z), aabbMin, aabbMax, decodeScale);
            outNormal = normalize(vec3(dx, dy, dz));
            return true;
        }
        t += max(h, minStep);
    }

    // Escape — endpoint at tMax (or wherever the loop bailed). Caller will
    // chain into the next cascade's SH probe lookup here.
    outPos = ro + rd * min(t, tMax);
    outDistTravelled = min(t, tMax);
    outNormal = -rd;   // sentinel — caller checks `hit` flag
    return false;
}

// -----------------------------------------------------------------------------
// Sparse variants (Phase 15d/e). Same math as the dense helpers above, but
// sample through the sparse-VDB SSBOs via the macros in sdf_sparse.glsl. We
// can't pass SSBO arrays as function args (GLSL/SPIR-V wants static binding
// references), so the helpers are macro-defined: the consumer pastes
// SDFTRACE_DEFINE_SPARSE_HELPERS once after instantiating both the point
// helper (SDFSPARSE_DEFINE_HELPER) AND the ray helper (SDFSPARSE_DEFINE_RAY_HELPER).
//
// The point helper backs central-difference normals + the soft-shadow penumbra
// reads; the ray helper is the hot sphere-trace step (Phase 15c brick skip).
//
// Required setup in the consumer file (order matters):
//   #include "include/sdf_sparse.glsl"
//   layout(...) readonly buffer SparseIdx  { uint Data[]; } uSparseIdx;
//   layout(...) readonly buffer SparsePool { uint Data[]; } uSparsePool;
//   layout(...) uniform SparseParamsBuf    { SparseSDFParams P; } uSparseP;
//   SDFSPARSE_DEFINE_HELPER    (uSparseIdx, uSparsePool)
//   SDFSPARSE_DEFINE_RAY_HELPER(uSparseIdx, uSparsePool)
//   #include "include/sdf_trace.glsl"
//   SDFTRACE_DEFINE_SPARSE_HELPERS()
//
// After expansion, callers use:
//   float vis = SDF_SoftShadowSparse(ro, rd, maxDist, k, maxSteps, minStep, P);
//   bool  hit = SDF_TraceHitSparse  (ro, rd, tMin, tMax, maxSteps, minStep, P,
//                                    outPos, outNormal, outDistTravelled);
// -----------------------------------------------------------------------------

#define SDFTRACE_DEFINE_SPARSE_HELPERS()                                          \
float SDF_SoftShadowSparse(vec3 ro, vec3 rd,                                      \
                           float maxDist, float k,                                \
                           int maxSteps, float minStep,                           \
                           SparseSDFParams params)                                \
{                                                                                 \
    /* Slab skip into the SDF box. If ro is inside (tEnter<0) start at 0. */      \
    float tEnter = SDF_RayBoxEnter(ro, rd, params.AABBMin.xyz, params.AABBMax.xyz);\
    if (tEnter < 0.0 && SDFSparseSampleHelper(ro, params) >= 1e5) {               \
        return 1.0;                                                               \
    }                                                                             \
    float t = max(0.0, tEnter);                                                   \
    if (t > maxDist) return 1.0;                                                  \
    float res = 1.0;                                                              \
    for (int i = 0; i < maxSteps; ++i) {                                          \
        if (t > maxDist) break;                                                   \
        float h = SDFSparseRaySampleHelper(ro + rd * t, rd, params);              \
        if (h >= 1e5) {                                                           \
            break;                                                                \
        }                                                                         \
        if (h < 0.001) return 0.0;                                                \
        res = min(res, k * h / max(t, 1e-4));                                     \
        t  += max(h, minStep);                                                    \
    }                                                                             \
    res = clamp(res, 0.0, 1.0);                                                   \
    return res * res * (3.0 - 2.0 * res);                                         \
}                                                                                 \
                                                                                  \
bool SDF_TraceHitSparse(vec3 ro, vec3 rd,                                         \
                        float tMin, float tMax,                                   \
                        int maxSteps, float minStep,                              \
                        SparseSDFParams params,                                   \
                        out vec3 outPos, out vec3 outNormal,                      \
                        out float outDistTravelled)                               \
{                                                                                 \
    float tEnter = SDF_RayBoxEnter(ro, rd, params.AABBMin.xyz, params.AABBMax.xyz);\
    float t      = max(tMin, max(0.0, tEnter));                                   \
    for (int i = 0; i < maxSteps; ++i) {                                          \
        if (t > tMax) break;                                                      \
        vec3 p = ro + rd * t;                                                     \
        float h = SDFSparseRaySampleHelper(p, rd, params);                        \
        if (h >= 1e5) {                                                           \
            break;                                                                \
        }                                                                         \
        if (h < 0.001) {                                                          \
            outPos = p;                                                           \
            outDistTravelled = t;                                                 \
            /* Central diff normal. Use the point helper (trilinear); the ray */  \
            /* helper would short-circuit AllOutside bricks and break the   */    \
            /* gradient. Step = half a voxel along each axis.               */    \
            vec3 boxSize = max(params.AABBMax.xyz - params.AABBMin.xyz, vec3(1e-6));\
            vec3 eps = boxSize / (2.0 * vec3(float(params.Dims.x)));              \
            float dx = SDFSparseSampleHelper(p + vec3(eps.x, 0, 0), params)       \
                     - SDFSparseSampleHelper(p - vec3(eps.x, 0, 0), params);      \
            float dy = SDFSparseSampleHelper(p + vec3(0, eps.y, 0), params)       \
                     - SDFSparseSampleHelper(p - vec3(0, eps.y, 0), params);      \
            float dz = SDFSparseSampleHelper(p + vec3(0, 0, eps.z), params)       \
                     - SDFSparseSampleHelper(p - vec3(0, 0, eps.z), params);      \
            outNormal = normalize(vec3(dx, dy, dz));                              \
            return true;                                                          \
        }                                                                         \
        t += max(h, minStep);                                                     \
    }                                                                             \
    outPos = ro + rd * min(t, tMax);                                              \
    outDistTravelled = min(t, tMax);                                              \
    outNormal = -rd;                                                              \
    return false;                                                                 \
}

#endif // SDF_TRACE_GLSL_INCLUDED
