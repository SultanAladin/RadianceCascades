// Include/RS/Material.h
// PBR material — six channels per spec: Albedo, AO, Metallic, Roughness, F0,
// Normal, Emissive. Each channel exposes a flat-value path and an optional
// texture-slot path. Normal has no flat fallback; absent normal texture means
// "use geometric normal".
#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

namespace RS {

struct TextureSlot {
    // Both fields null => the corresponding flat value is used.
    VkImageView View    = VK_NULL_HANDLE;
    VkSampler   Sampler = VK_NULL_HANDLE;
};

struct PbrMaterial {
    glm::vec3   AlbedoFlat    = glm::vec3(0.8f, 0.8f, 0.8f);
    TextureSlot AlbedoTex;

    float       AOFlat        = 1.0f;
    TextureSlot AOTex;

    float       MetallicFlat  = 0.0f;
    TextureSlot MetallicTex;

    float       RoughnessFlat = 0.5f;
    TextureSlot RoughnessTex;

    // KHR_materials_specular semantics on dielectrics:
    //   F0 = vec3(0.04) * SpecularFactorFlat * F0Flat
    // (F0Flat plays the "specularColor" role — RGB tint of the dielectric
    // reflection; SpecularFactorFlat scales the magnitude.)
    // On metals (Metallic = 1) both are ignored — F0 = AlbedoFlat.
    // Kept the F0Flat name to avoid breaking external code; semantics
    // changed from "scalar F0 broadcast" to "dielectric specular colour".
    glm::vec3   F0Flat            = glm::vec3(1.0f, 1.0f, 1.0f);
    TextureSlot F0Tex;
    float       SpecularFactorFlat = 1.0f;

    // Normal has no flat fallback — when NormalTex is unbound, the GBuffer
    // pass falls back to the per-fragment geometric normal.
    TextureSlot NormalTex;

    glm::vec3   EmissiveFlat  = glm::vec3(0.0f, 0.0f, 0.0f);
    TextureSlot EmissiveTex;

    char Name[64] = "Untitled";
};

using MaterialHandle = uint32_t;

class MaterialRegistry {
public:
    MaterialRegistry();
    ~MaterialRegistry();

    MaterialHandle Create (const PbrMaterial& material);
    PbrMaterial&   Get    (MaterialHandle handle);
    void           Destroy(MaterialHandle handle);
    size_t         Count() const;

private:
    struct Impl;
    Impl* m_Impl = nullptr;
};

} // namespace RS
