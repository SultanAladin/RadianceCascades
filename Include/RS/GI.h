// Include/RS/GI.h
// Public enum for the GI algorithm family. SDFGI is the only v1 implementor;
// voxel-cone and radiance-cascades slot in later behind IGIAlgorithm.
#pragma once

#include <cstdint>

namespace RS {

enum class GIAlgorithmKind : uint32_t {
    SDFGI            = 0,   // cascaded probe grid + SH against global signed distance field
    RadianceCascades = 1,   // Sannikov-style RC × sparse world-space hash (Phase 14)
};

} // namespace RS
