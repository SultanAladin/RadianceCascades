// Include/RS/Shadow.h
// Public enum identifying which swappable shadow algorithm the renderer should run.
// Implementations live behind IShadowAlgorithm in Source/Shadow/.
#pragma once

#include <cstdint>

namespace RS {

enum class ShadowAlgorithmKind : uint32_t {
    PCF     = 0,   // 5x5 (configurable) percentage-closer filter on CSM atlas
    PCSS    = 1,   // percentage-closer soft shadows, variable penumbra
    VSM     = 2,   // variance shadow maps with moment blur pass
    SDFCone = 3,   // Inigo Quilez cone-trace soft shadows against global SDF
};

} // namespace RS
