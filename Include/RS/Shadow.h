// Include/RS/Shadow.h
// Public enum identifying the renderer shadow path.
#pragma once

#include <cstdint>

namespace RS {

enum class ShadowAlgorithmKind : uint32_t {
    SDFCone = 0,   // Inigo Quilez cone-trace soft shadows against the resident SDF
};

} // namespace RS
