// Standalone self-test for CascadeSpecification.hpp + the octahedral mapping.
// Not part of the engine build; compiled ad hoc to verify the RC math core
// before any GPU pass depends on it. Prints the per-cascade table, runs the
// A4/B2 validators, sums VRAM, and runs the B1 octahedral round-trip (10k dirs).
//
// Build (from project root, inside a vcvars64 shell):
//   cl /std:c++17 /EHsc /O2 /nologo Source\GI\spec_selftest.cpp /Fe:spec_selftest.exe

#include "CascadeSpecification.hpp"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>

// ----- minimal vec3 + the octahedral mapping, CPU port of octahedral.glsl -----
struct V3 { float x, y, z; };
static V3   v3(float a, float b, float c) { return {a, b, c}; }
static float dot3(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3   normalize3(V3 a) { float l = std::sqrt(dot3(a,a)); return {a.x/l, a.y/l, a.z/l}; }

struct V2 { float x, y; };

static V2 OctEncode(V3 Direction) {
    V3 AbsoluteDirection = { std::fabs(Direction.x), std::fabs(Direction.y), std::fabs(Direction.z) };
    float denom = AbsoluteDirection.x + AbsoluteDirection.y + AbsoluteDirection.z;
    V2 Encoded = { Direction.x / denom, Direction.z / denom };   // .xz
    if (Direction.y < 0.0f) {
        V2 Folded;
        Folded.x = (1.0f - std::fabs(Encoded.y)) * (Encoded.x >= 0.0f ? 1.0f : -1.0f);
        Folded.y = (1.0f - std::fabs(Encoded.x)) * (Encoded.y >= 0.0f ? 1.0f : -1.0f);
        Encoded = Folded;
    }
    return { Encoded.x * 0.5f + 0.5f, Encoded.y * 0.5f + 0.5f };
}

static V3 OctDecode(V2 Encoded) {
    V2 Centred = { Encoded.x * 2.0f - 1.0f, Encoded.y * 2.0f - 1.0f };
    V3 Direction = { Centred.x, 1.0f - std::fabs(Centred.x) - std::fabs(Centred.y), Centred.y };
    float Fold = std::fmax(-Direction.y, 0.0f);
    Direction.x += Direction.x >= 0.0f ? -Fold : Fold;
    Direction.z += Direction.z >= 0.0f ? -Fold : Fold;
    return normalize3(Direction);
}

int main() {
    using namespace Cascades;

    HierarchySpecification spec;   // defaults: R=8, t0=0.15, probes=64, R0=4

    std::printf("=== HierarchySpecification ===\n");
    std::printf("RcQueryRadius=%.3f m  t0=%.3f m  s0=%.3f m  BaseProbeAxis=%u  R0=%u\n\n",
                spec.RcQueryRadius, spec.BaseIntervalLength, ResolveBaseProbePitch(spec),
                spec.BaseProbeAxisCount, spec.BaseAngularAxisCount);

    const bool r0pow2  = ConfirmAngularAxisPowerOfTwo(spec);
    const bool covered = ConfirmTopCascadeCoversRegion(spec);
    std::printf("ConfirmAngularAxisPowerOfTwo  : %s\n", r0pow2  ? "PASS" : "FAIL");
    std::printf("ConfirmTopCascadeCoversRegion : %s\n\n", covered ? "PASS" : "FAIL");

    const uint32_t count = ResolveCascadeCount(spec);
    std::printf("ResolveCascadeCount = %u\n\n", count);

    std::printf(" c | ProbeAxis | OctSide |  IntStart |  IntEnd  |   Pitch  |   AtlasW x H x D    |   RayBudget   |  VRAM(MB f16vec4)\n");
    std::printf("---+-----------+---------+-----------+----------+----------+---------------------+---------------+------------------\n");

    double totalBytes = 0.0;
    auto hier = ResolveHierarchy(spec);
    for (const auto& d : hier) {
        const float pitch = ResolveBaseProbePitch(spec) * std::pow(2.0f, (float)d.CascadeIndex);
        const uint64_t budget = EstimateRayBudget(d);
        // payload = vec4 f16 (rgb + transmittance) = 8 bytes/texel
        const double bytes = (double)d.AtlasWidth * d.AtlasHeight * d.AtlasDepth * 8.0;
        totalBytes += bytes;
        std::printf(" %u |    %5u  |   %4u  |  %7.4f  | %7.4f  | %7.4f  | %5u x %5u x %4u | %13llu | %8.2f\n",
                    d.CascadeIndex, d.ProbeAxisCount, d.OctSide,
                    d.IntervalStart, d.IntervalEnd, pitch,
                    d.AtlasWidth, d.AtlasHeight, d.AtlasDepth,
                    (unsigned long long)budget, bytes / (1024.0*1024.0));
    }
    std::printf("\nTotal atlas VRAM (f16 vec4, single buffer) = %.2f MB\n", totalBytes / (1024.0*1024.0));
    std::printf("With a transient merge ping copy (worst case) = %.2f MB\n\n", 2.0 * totalBytes / (1024.0*1024.0));

    // Top-cascade coverage detail (A4)
    {
        const auto& top = hier.back();
        const float pitch = ResolveBaseProbePitch(spec) * std::pow(2.0f, (float)top.CascadeIndex);
        std::printf("A4 top-cascade coverage: ProbeAxis(%u) * pitch(%.3f) = %.3f m  vs  diameter %.3f m\n\n",
                    top.ProbeAxisCount, pitch, top.ProbeAxisCount * pitch, 2.0f * spec.RcQueryRadius);
    }

    // ----- B1: octahedral round-trip, 10k random unit dirs -----
    std::mt19937 rng(1234567u);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    double maxErr = 0.0; double sumErr = 0.0; int n = 0; int worstReported = 0;
    for (int i = 0; i < 10000; ++i) {
        V3 d = { uni(rng), uni(rng), uni(rng) };
        float len2 = dot3(d, d);
        if (len2 < 1e-6f) continue;
        d = normalize3(d);
        V3 r = OctDecode(OctEncode(d));
        V3 diff = { r.x - d.x, r.y - d.y, r.z - d.z };
        double e = std::sqrt(dot3(diff, diff));
        maxErr = std::fmax(maxErr, e);
        sumErr += e; ++n;
        if (e > 1e-3 && worstReported < 5) {
            std::printf("  round-trip outlier: d=(%.3f,%.3f,%.3f) -> r=(%.3f,%.3f,%.3f) err=%.5f\n",
                        d.x,d.y,d.z, r.x,r.y,r.z, e);
            ++worstReported;
        }
    }
    std::printf("B1 octahedral round-trip over %d dirs: maxErr=%.3e meanErr=%.3e  %s (threshold 1e-5)\n",
                n, maxErr, sumErr / n, (maxErr < 1e-5) ? "PASS" : "FAIL");

    // Also check texel-centre directions are unit-length for c0 grid (4x4)
    {
        double maxOff = 0.0;
        // replicate DirectionFromOctTexel for OctSide=4
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) {
                V2 e = { (x + 0.5f) / 4.0f, (y + 0.5f) / 4.0f };
                V3 dir = OctDecode(e);
                maxOff = std::fmax(maxOff, std::fabs(std::sqrt(dot3(dir,dir)) - 1.0));
            }
        std::printf("Texel-centre dirs unit-length (c0 4x4): max |len-1| = %.3e %s\n",
                    maxOff, (maxOff < 1e-5) ? "PASS" : "FAIL");
    }

    return (r0pow2 && covered && maxErr < 1e-5) ? 0 : 1;
}
