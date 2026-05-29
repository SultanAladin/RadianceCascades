// Source/Material/PbrMaterial.cpp
// Phase 1 ships a minimal MaterialRegistry that just stores PbrMaterial copies
// in a std::vector with stable handles. Phase 7+ adds GPU-side descriptor data
// alongside.
#include "RS/Material.h"

#include <vector>

namespace RS {

struct MaterialRegistry::Impl {
    std::vector<PbrMaterial> Materials;
};

MaterialRegistry::MaterialRegistry() : m_Impl(new Impl()) {}
MaterialRegistry::~MaterialRegistry() { delete m_Impl; m_Impl = nullptr; }

MaterialHandle MaterialRegistry::Create(const PbrMaterial& material) {
    m_Impl->Materials.push_back(material);
    return static_cast<MaterialHandle>(m_Impl->Materials.size() - 1);
}

PbrMaterial& MaterialRegistry::Get(MaterialHandle handle) {
    return m_Impl->Materials.at(handle);
}

void MaterialRegistry::Destroy(MaterialHandle handle) {
    // Phase 1 sentinel: blank the slot rather than shifting indices, so handles
    // stay stable.
    if (handle < m_Impl->Materials.size()) {
        m_Impl->Materials[handle] = PbrMaterial{};
    }
}

size_t MaterialRegistry::Count() const {
    return m_Impl->Materials.size();
}

} // namespace RS
