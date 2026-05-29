// Source/Material/MaterialSeed.cpp — Phase 9
#include "Material/MaterialSeed.h"

#include <cstring>

namespace RS {

namespace {

void SetName(PbrMaterial& mat, const char* name) {
    std::strncpy(mat.Name, name, sizeof(mat.Name) - 1);
    mat.Name[sizeof(mat.Name) - 1] = '\0';
}

} // namespace

SeededMaterials SeedDemoMaterials(MaterialRegistry& registry) {
    SeededMaterials s{};

    // 0 - Default. Matches the Phase 6 GBufferMaterial defaults so existing
    // scenes look identical before the user touches anything.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(0.78f, 0.78f, 0.80f);
        m.RoughnessFlat = 0.55f;
        m.MetallicFlat  = 0.0f;
        m.F0Flat        = glm::vec3(0.04f);
        m.EmissiveFlat  = glm::vec3(0.0f);
        SetName(m, "Default");
        s.Default = registry.Create(m);
    }

    // 1 - Red plastic dielectric. Saturated albedo, medium-rough, no metal.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(0.78f, 0.08f, 0.06f);
        m.RoughnessFlat = 0.40f;
        m.MetallicFlat  = 0.0f;
        m.F0Flat        = glm::vec3(0.04f);
        SetName(m, "Red Plastic");
        s.RedPlastic = registry.Create(m);
    }

    // 2 - Polished gold. Metal => albedo doubles as F0 tint; very smooth.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(1.00f, 0.78f, 0.34f);
        m.RoughnessFlat = 0.10f;
        m.MetallicFlat  = 1.0f;
        m.F0Flat        = glm::vec3(1.00f, 0.78f, 0.34f);
        SetName(m, "Polished Gold");
        s.PolishedGold = registry.Create(m);
    }

    // 3 - Brushed aluminium. Cool metal tint, rougher than polished gold so
    // the highlight broadens instead of catching a single point of sky.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(0.91f, 0.92f, 0.94f);
        m.RoughnessFlat = 0.42f;
        m.MetallicFlat  = 1.0f;
        m.F0Flat        = glm::vec3(0.91f, 0.92f, 0.94f);
        SetName(m, "Brushed Aluminium");
        s.BrushedAluminium = registry.Create(m);
    }

    // 4 - Matte black rubber. Near-black albedo, very rough, dielectric F0.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(0.02f, 0.02f, 0.02f);
        m.RoughnessFlat = 0.90f;
        m.MetallicFlat  = 0.0f;
        m.F0Flat        = glm::vec3(0.04f);
        SetName(m, "Matte Black Rubber");
        s.MatteBlackRubber = registry.Create(m);
    }

    // 5 - Emissive cyan. Acts as a soft area light in the scene; useful for
    // sanity-checking the emissive channel survives the GBuffer round-trip.
    {
        PbrMaterial m{};
        m.AlbedoFlat    = glm::vec3(0.0f);
        m.RoughnessFlat = 0.50f;
        m.MetallicFlat  = 0.0f;
        m.F0Flat        = glm::vec3(0.04f);
        m.EmissiveFlat  = glm::vec3(0.10f, 1.80f, 2.20f);
        SetName(m, "Emissive Cyan");
        s.EmissiveCyan = registry.Create(m);
    }

    return s;
}

} // namespace RS
