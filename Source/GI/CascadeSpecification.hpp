/*====================================================================================================================================
                                                  CASCADESPECIFICATION.HPP
====================================================================================================================================*/
// Pure Radiance-Cascades math: interval series, direction counts, probe counts, atlas sizing, parent/child mapping (metres)

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

namespace Cascades
{

//------------------------------------------------------------------------------------------------------------------------
//                                                       CONSTANTS
//------------------------------------------------------------------------------------------------------------------------

// The two structural ratios that define Radiance Cascades. NOT free parameters - they ARE the penumbra condition.
// Per cascade: angular resolution rises 4x, spatial probe density falls 8x (1/2 per axis, three axes). Interval length
// grows 4x so the geometric ray-length series tiles space. In 3D the ray budget HALVES per cascade (8x spatial drop
// vs 4x angular rise) - intended; the top cascades are cheap.
constexpr uint32_t AngularBranchFactor  = 4;     // [-] - child rays per parent ray (octahedral 2x2 subdivision)
constexpr uint32_t SpatialReductionAxis = 2;     // [-] - probe-spacing doubling per axis per cascade
constexpr float    IntervalLengthFactor = 4.0f;  // [-] - interval-length growth per cascade (== AngularBranchFactor)

//------------------------------------------------------------------------------------------------------------------------
//                                                    INPUT STRUCTURES
//------------------------------------------------------------------------------------------------------------------------

// Everything the renderer needs to derive a full cascade hierarchy. ALL LENGTHS ARE METRES (A1). The cascades cover a
// camera-anchored bounded region of half-extent RcQueryRadius (A2) - NOT a whole-scene diagonal. The SDF is sampled in
// mesh-local centimetres, but that conversion happens in-shader via scaleLocalPerWorld; this struct stays pure metres.
//
// Locked after A4 verification: t0=0.25 + 64 probes + R0=4 derives to 4 cascades, top covers exactly 16.0 m (PASS),
// c0 atlas = 32 MB, total ~60 MB. t0 raised from 0.15 -> 0.25 to satisfy coverage at 64 probes instead of paying 8x
// VRAM for 128 probes. Near-field still well under the 0.5 m reference sphere radius. If contact GI bleeds later,
// decouple via ResolveBaseProbePitch (shorten t0, keep probe pitch) rather than re-densifying.
struct HierarchySpecification
{
    float    RcQueryRadius        = 8.0f;   // [m] - camera-anchored half-extent the cascades must cover (A2)
    float    BaseIntervalLength   = 0.25f;  // [m] - cascade-0 interval length t0 (A1; raised 0.15->0.25 for A4 coverage)
    uint32_t BaseProbeAxisCount   = 64;     // [-] - cascade-0 probes along one axis (cubic grid)
    uint32_t BaseAngularAxisCount = 4;      // [-] - cascade-0 octahedral side R0 (dirs = R0*R0); MUST be power of two (B2)
    uint32_t CascadeCountOverride = 0;      // [-] - 0 => derive from RcQueryRadius (yields 4 at these defaults)
};

//------------------------------------------------------------------------------------------------------------------------
//                                                    OUTPUT STRUCTURES
//------------------------------------------------------------------------------------------------------------------------

// Resolved geometry for one cascade level. OctSide doubles per cascade so parent->child is an exact 2x2 block.
// ATLAS LAYOUT INVARIANT (A5): the OctSide x OctSide direction tile lives in X and Y ONLY; Z is the bare probe-Z axis
// (no direction tiling in Z). A probe at (probeX,probeY,probeZ) owns texels:
//     X in [probeX*OctSide .. probeX*OctSide + OctSide-1]
//     Y in [probeY*OctSide .. probeY*OctSide + OctSide-1]
//     Z == probeZ
// rc_merge depends on this: the 2x2 oct child block lives in X/Y of the child cascade; the spatial 2x2x2 probe gather
// spans probeZ via the Z axis.
struct CascadeDescriptor
{
    uint32_t CascadeIndex   = 0;    // [idx] - cascade level c (0 == finest spatially, coarsest angularly)
    uint32_t ProbeAxisCount = 0;    // [-]   - probes along one axis at this level
    uint32_t OctSide        = 0;    // [-]   - octahedral map side length R (directions = R*R)
    uint32_t DirectionCount = 0;    // [-]   - R*R
    float    IntervalStart  = 0.0f; // [m]   - near end of this cascade's radiance interval
    float    IntervalEnd    = 0.0f; // [m]   - far end of this cascade's radiance interval

    uint32_t AtlasWidth     = 0;    // [px]  - ProbeAxisCount * OctSide  (oct tile in X)
    uint32_t AtlasHeight    = 0;    // [px]  - ProbeAxisCount * OctSide  (oct tile in Y)
    uint32_t AtlasDepth     = 0;    // [px]  - ProbeAxisCount            (bare probe-Z)
};

// Per-frame placement (A3). World-anchored + clipmap-snapped so probes do NOT swim under sub-pitch camera motion -
// this was a root cause of last attempt's flicker. Filled by C++ each frame, NOT part of the static spec, but the
// fields are defined now so the shader UBO layout is fixed early.
struct CascadePlacement
{
    float RegionOriginWorldX = 0.0f; // [m] - world-space min corner X of THIS cascade's probe grid (snapped)
    float RegionOriginWorldY = 0.0f; // [m] - world-space min corner Y
    float RegionOriginWorldZ = 0.0f; // [m] - world-space min corner Z
    float ProbePitchWorld    = 0.0f; // [m] - s0 * 2^CascadeIndex (metres between probes)
};

//------------------------------------------------------------------------------------------------------------------------
//                                                    PUBLIC FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------

// Geometric ray-length series. Near end of cascade c is the sum of all shorter intervals:
//     start(c) = t0 * (4^c - 1) / 3
//     end(c)   = t0 * (4^(c+1) - 1) / 3      (== start(c+1), so intervals tile with no gaps)
inline float ResolveIntervalStart(float BaseIntervalLength, uint32_t CascadeIndex)
{
    const float IntervalPower = std::pow(IntervalLengthFactor, static_cast<float>(CascadeIndex));
    return BaseIntervalLength * (IntervalPower - 1.0f) / 3.0f;
}

inline float ResolveIntervalEnd(float BaseIntervalLength, uint32_t CascadeIndex)
{
    return ResolveIntervalStart(BaseIntervalLength, CascadeIndex + 1);
}

// Cascade-0 probe pitch s0 in metres. Equal to the c0 interval length t0 by RC convention (interval ~ probe spacing).
// Exposed so C++ placement can derive ProbePitchWorld = s0 * 2^c without re-deriving from the spec.
inline float ResolveBaseProbePitch(const HierarchySpecification& Specification)
{
    return Specification.BaseIntervalLength;
}

// Derive cascade count so the top cascade's far interval reaches >= the region DIAMETER (A2):
//     end(C-1) >= 2*RcQueryRadius  ==>  C = ceil( log4( 3*(2*R)/t0 + 1 ) )
inline uint32_t ResolveCascadeCount(const HierarchySpecification& Specification)
{
    if (Specification.CascadeCountOverride != 0)
        return Specification.CascadeCountOverride;

    const float RegionDiameter = 2.0f * Specification.RcQueryRadius;
    const float SeriesRatio    = 3.0f * RegionDiameter / Specification.BaseIntervalLength + 1.0f;
    const float CascadeLevels  = std::ceil(std::log(SeriesRatio) / std::log(IntervalLengthFactor));
    return static_cast<uint32_t>(std::max(1.0f, CascadeLevels));
}

// Resolve one cascade's full descriptor. Probe axis halves (clamped to 2) per level; octahedral side doubles.
inline CascadeDescriptor ResolveCascade(const HierarchySpecification& Specification, uint32_t CascadeIndex)
{
    CascadeDescriptor Descriptor;
    Descriptor.CascadeIndex = CascadeIndex;

    // Spatial: probes halve per axis per cascade, never below 2 (A4 coverage caveat checked separately).
    uint32_t ProbeAxis = Specification.BaseProbeAxisCount >> CascadeIndex;
    if (ProbeAxis < 2) ProbeAxis = 2;
    Descriptor.ProbeAxisCount = ProbeAxis;

    // Angular: octahedral side doubles per cascade ==> directions quadruple (4x branch). Requires R0 power of two (B2).
    const uint32_t OctSide = Specification.BaseAngularAxisCount << CascadeIndex;
    Descriptor.OctSide        = OctSide;
    Descriptor.DirectionCount = OctSide * OctSide;

    Descriptor.IntervalStart = ResolveIntervalStart(Specification.BaseIntervalLength, CascadeIndex);
    Descriptor.IntervalEnd   = ResolveIntervalEnd(Specification.BaseIntervalLength, CascadeIndex);

    Descriptor.AtlasWidth  = ProbeAxis * OctSide;   // oct tile in X (A5)
    Descriptor.AtlasHeight = ProbeAxis * OctSide;   // oct tile in Y (A5)
    Descriptor.AtlasDepth  = ProbeAxis;             // bare probe-Z   (A5)

    return Descriptor;
}

// Resolve the whole hierarchy. Index 0 is the finest spatial / coarsest angular cascade.
inline std::vector<CascadeDescriptor> ResolveHierarchy(const HierarchySpecification& Specification)
{
    const uint32_t CascadeCount = ResolveCascadeCount(Specification);
    std::vector<CascadeDescriptor> Hierarchy;
    Hierarchy.reserve(CascadeCount);
    for (uint32_t CascadeIndex = 0; CascadeIndex < CascadeCount; ++CascadeIndex)
        Hierarchy.push_back(ResolveCascade(Specification, CascadeIndex));
    return Hierarchy;
}

//------------------------------------------------------------------------------------------------------------------------
//                                                  VALIDATION FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------

// B2: nesting is exact only if R0 is a power of two (4 -> 4,8,16,32...). A tweak to e.g. 6 silently breaks parent
// merge. Returns true if valid; the renderer logs/asserts on false at boot.
inline bool ConfirmAngularAxisPowerOfTwo(const HierarchySpecification& Specification)
{
    const uint32_t Axis = Specification.BaseAngularAxisCount;
    return Axis != 0 && (Axis & (Axis - 1)) == 0;
}

// A4: once a cascade clamps ProbeAxis to 2, its world coverage (ProbeAxis * pitch) may no longer reach the region
// diameter, breaking the "intervals tile all of space" promise. The TOP cascade must satisfy
//     ProbeAxis * (s0 * 2^c)  >=  2 * RcQueryRadius.
// Returns true if the top cascade covers the diameter; renderer surfaces this in the boot log. If false: raise
// BaseProbeAxisCount or lower CascadeCount.
inline bool ConfirmTopCascadeCoversRegion(const HierarchySpecification& Specification)
{
    const uint32_t CascadeCount = ResolveCascadeCount(Specification);
    const uint32_t TopIndex     = CascadeCount - 1;
    const CascadeDescriptor Top = ResolveCascade(Specification, TopIndex);

    const float TopPitch    = ResolveBaseProbePitch(Specification) * std::pow(2.0f, static_cast<float>(TopIndex));
    const float TopCoverage = static_cast<float>(Top.ProbeAxisCount) * TopPitch;
    return TopCoverage >= 2.0f * Specification.RcQueryRadius;
}

//------------------------------------------------------------------------------------------------------------------------
//                                                    INTERNAL FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------

// Total ray budget per cascade = probes^3 * directions. The RC 3D invariant: this HALVES per level (probes fall 8x,
// directions rise 4x). Exposed for the debug HUD so you can confirm the budget halves rather than exploding (16^c bug).
inline uint64_t EstimateRayBudget(const CascadeDescriptor& Descriptor)
{
    const uint64_t ProbeTotal = static_cast<uint64_t>(Descriptor.ProbeAxisCount)
                              * Descriptor.ProbeAxisCount * Descriptor.ProbeAxisCount;
    return ProbeTotal * Descriptor.DirectionCount;
}

} // namespace Cascades
