// Source/Material/MaterialSeed.h — Phase 9
// Boot-time seeding of a small library of demo PBR materials so the Materials
// panel + per-submesh binding table have something interesting to bind right
// out of the box. Slot 0 is always "Default" — the InstanceRegistry assigns it
// to every new submesh slot, and the renderer falls back to it whenever an
// instance's binding is missing or invalid.
//
// The seed list is intentionally hand-curated rather than procedurally
// generated: each demo exercises a distinct corner of the GGX/Schlick BRDF
// (dielectric red plastic, polished metal, brushed metal, matte rubber,
// emissive) so a single 3x3 grid + a few clicks paint a clearly multi-material
// scene without anyone having to author a material first.
#pragma once

#include "RS/Material.h"

namespace RS {

struct SeededMaterials {
    MaterialHandle Default          = 0;
    MaterialHandle RedPlastic       = 0;
    MaterialHandle PolishedGold     = 0;
    MaterialHandle BrushedAluminium = 0;
    MaterialHandle MatteBlackRubber = 0;
    MaterialHandle EmissiveCyan     = 0;
    MaterialHandle Floor            = 0;   // Phase 11.5 — dev-engine checker floor

    static constexpr uint32_t kCount = 7;
};

// Push the 6 demos into the registry in fixed order. Caller is responsible for
// keeping the returned handles around (Main.cpp parks them on the heap so the
// Materials panel can refer to them by name when rendering the binding combo).
SeededMaterials SeedDemoMaterials(MaterialRegistry& registry);

} // namespace RS
