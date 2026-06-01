// Source/Renderer/SkyAtmosphere.cpp — Phase 8
#include "Renderer/SkyAtmosphere.h"
#include "Core/Logger.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

// --- generic helpers --------------------------------------------------------

bool ReadFile(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("SkyAtmosphere: cannot open shader %s", path); return false; }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0) { std::fclose(f); return false; }
    out.resize(static_cast<size_t>(len));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

VkShaderModule LoadModule(VkDevice device, const char* path) {
    std::vector<char> buf;
    if (!ReadFile(path, buf)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = buf.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) {
        RS_LOG_ERROR("SkyAtmosphere: vkCreateShaderModule failed %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

int FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool CreateCubeImage(const VulkanContext& ctx, uint32_t faceSize, uint32_t mipCount,
                     VkFormat format, SkyCubeImage& out) {
    out.Format   = format;
    out.FaceSize = faceSize;
    out.MipCount = mipCount;

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { faceSize, faceSize, 1 };
    ici.mipLevels     = mipCount;
    ici.arrayLayers   = 6;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &out.Image) != VK_SUCCESS) {
        RS_LOG_ERROR("SkyAtmosphere: vkCreateImage (cube) failed");
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx.Device, out.Image, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &out.Memory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.Device, out.Image, out.Memory, 0);

    // Cube view spanning all mips, for textureLod sampling.
    VkImageViewCreateInfo sampleView{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    sampleView.image            = out.Image;
    sampleView.viewType         = VK_IMAGE_VIEW_TYPE_CUBE;
    sampleView.format           = format;
    sampleView.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount, 0, 6 };
    if (vkCreateImageView(ctx.Device, &sampleView, nullptr, &out.View) != VK_SUCCESS) {
        return false;
    }

    // Per-mip 2D-array views for imageStore in the compute bakers (cube views
    // can't be used as storage images in GLSL; we need an array view of the
    // 6 faces of a single mip).
    for (uint32_t m = 0; m < mipCount; ++m) {
        VkImageViewCreateInfo storeView{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        storeView.image            = out.Image;
        storeView.viewType         = VK_IMAGE_VIEW_TYPE_CUBE;
        storeView.format           = format;
        storeView.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 6 };
        if (vkCreateImageView(ctx.Device, &storeView, nullptr, &out.Storage[m]) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

void DestroyCubeImage(VkDevice device, SkyCubeImage& img) {
    for (uint32_t m = 0; m < img.MipCount; ++m) {
        if (img.Storage[m]) vkDestroyImageView(device, img.Storage[m], nullptr);
        img.Storage[m] = VK_NULL_HANDLE;
    }
    if (img.View)   vkDestroyImageView(device, img.View,   nullptr);
    if (img.Image)  vkDestroyImage    (device, img.Image,  nullptr);
    if (img.Memory) vkFreeMemory      (device, img.Memory, nullptr);
    img = {};
}

// 2D RGBA16F image used for the Hillaire transmittance/multiscatter/skyview LUTs.
bool CreateLut2D(const VulkanContext& ctx, uint32_t w, uint32_t h, VkFormat fmt,
                 LutImage2D& out) {
    out.Width  = w;
    out.Height = h;
    out.Format = fmt;
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = fmt;
    ici.extent        = { w, h, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &out.Image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(ctx.Device, out.Image, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &out.Memory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.Device, out.Image, out.Memory, 0);
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image            = out.Image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(ctx.Device, &vci, nullptr, &out.View) == VK_SUCCESS;
}
void DestroyLut2D(VkDevice device, LutImage2D& img) {
    if (img.View)   vkDestroyImageView(device, img.View,   nullptr);
    if (img.Image)  vkDestroyImage    (device, img.Image,  nullptr);
    if (img.Memory) vkFreeMemory      (device, img.Memory, nullptr);
    img = {};
}

// 3D RGBA16F image for the Aerial Perspective LUT.
bool CreateLut3D(const VulkanContext& ctx, uint32_t w, uint32_t h, uint32_t d,
                 VkFormat fmt, LutImage3D& out) {
    out.Width  = w; out.Height = h; out.Depth = d; out.Format = fmt;
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_3D;
    ici.format        = fmt;
    ici.extent        = { w, h, d };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &out.Image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(ctx.Device, out.Image, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &out.Memory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.Device, out.Image, out.Memory, 0);
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image            = out.Image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_3D;
    vci.format           = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(ctx.Device, &vci, nullptr, &out.View) == VK_SUCCESS;
}
void DestroyLut3D(VkDevice device, LutImage3D& img) {
    if (img.View)   vkDestroyImageView(device, img.View,   nullptr);
    if (img.Image)  vkDestroyImage    (device, img.Image,  nullptr);
    if (img.Memory) vkFreeMemory      (device, img.Memory, nullptr);
    img = {};
}

bool CreateLutImage(const VulkanContext& ctx, SkyAtmosphere& sa) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R16G16_SFLOAT;
    ici.extent        = { SkyAtmosphere::kBrdfLutSize, SkyAtmosphere::kBrdfLutSize, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &sa.LutImage) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx.Device, sa.LutImage, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &sa.LutMemory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.Device, sa.LutImage, sa.LutMemory, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image            = sa.LutImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R16G16_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(ctx.Device, &vci, nullptr, &sa.LutView) == VK_SUCCESS;
}

bool CreateSamplers(VkDevice device, SkyAtmosphere& sa) {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod       = 0.0f;
    sci.maxLod       = static_cast<float>(SkyAtmosphere::kPrefilterMips);
    if (vkCreateSampler(device, &sci, nullptr, &sa.SamplerCubeLinear) != VK_SUCCESS) return false;

    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.minLod     = 0.0f;
    sci.maxLod     = 0.0f;
    if (vkCreateSampler(device, &sci, nullptr, &sa.SamplerLut) != VK_SUCCESS) return false;
    // Same linear-clamp sampler reused for the Hillaire 2D LUTs.
    return vkCreateSampler(device, &sci, nullptr, &sa.SamplerLutLinear) == VK_SUCCESS;
}

// --- descriptor / pipeline helpers ------------------------------------------

bool MakeDescriptorSetLayout(VkDevice device,
                             std::initializer_list<VkDescriptorSetLayoutBinding> bindings,
                             VkDescriptorSetLayout& out) {
    std::vector<VkDescriptorSetLayoutBinding> v(bindings.begin(), bindings.end());
    VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ci.bindingCount = static_cast<uint32_t>(v.size());
    ci.pBindings    = v.data();
    return vkCreateDescriptorSetLayout(device, &ci, nullptr, &out) == VK_SUCCESS;
}

bool MakeComputePipeline(VkDevice device, const char* spvPath,
                         VkDescriptorSetLayout setLayout,
                         uint32_t pushSize,
                         VkPipelineLayout& outLayout, VkPipeline& outPipe) {
    VkShaderModule mod = LoadModule(device, spvPath);
    if (!mod) return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = pushSize;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &outLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, mod, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = outLayout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &outPipe);
    vkDestroyShaderModule(device, mod, nullptr);
    return r == VK_SUCCESS;
}

VkDescriptorSet AllocSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &layout;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(device, &ai, &ds);
    return ds;
}

// --- image-layout transitions ----------------------------------------------

void TransitionToGeneral(VkCommandBuffer cmd, VkImage image,
                         uint32_t baseMip, uint32_t mipCount,
                         uint32_t layerCount,
                         VkImageLayout oldLayout) {
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.oldLayout           = oldLayout;
    b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, layerCount };
    b.srcAccessMask       = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                          ? 0
                          : VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

void TransitionToShaderRead(VkCommandBuffer cmd, VkImage image,
                            uint32_t baseMip, uint32_t mipCount,
                            uint32_t layerCount) {
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, layerCount };
    b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

// --- push-constant layouts mirroring the GLSL blocks ------------------------

#pragma pack(push, 16)
struct IrradiancePush {
    glm::vec4 FaceSize;       // x = face size, y = sample step radians
};
struct PrefilterPush {
    glm::vec4 FaceSizeRoughnessSamples;  // x size, y roughness, z samples, w src size
};
struct BrdfPush {
    glm::vec4 SizeAndSamples;  // x width, y height, z samples
};

// Shared atmosphere parameter block — mirrors atmosphere.glsl AtmosphereParams.
struct AtmosphereParamsGpu {
    glm::vec4 RayleighScatteringAndPlanetRadius; // rgb = β_R (1/m), w = R_planet (km)
    glm::vec4 MieAndAtmoTopAndG;                 // x β_M, y R_top (km), z g, w unused
    glm::vec4 OzoneAbsorption;                   // rgb = ozone abs (1/m), w unused
};

struct TransmittancePush {
    AtmosphereParamsGpu Atmo;
    glm::vec4           LutSize;  // x w, y h, z steps
};
struct MultiScatterPush {
    AtmosphereParamsGpu Atmo;
    glm::vec4           Sizes;   // x = MS size, y = TR w, z = TR h, w = steps
    glm::vec4           SunIrr;  // rgb = sun irradiance gain (TOA), w = sqrt sample count
};
struct SkyViewPush {
    AtmosphereParamsGpu Atmo;
    glm::vec4           SunDirAndSkyGain;   // xyz dir, w = sky inscatter gain (SkyIntensity)
    glm::vec4           SunColorAndSize;    // rgb = sun color × SunIntensity, w = aperture (rad)
    glm::vec4           ViewSize;           // x w, y h, z steps, w camAltKm
};
struct AerialPush {
    AtmosphereParamsGpu Atmo;
    glm::vec4           SunDirAndIntensity;
    glm::vec4           ViewParams;     // x res, y stepsPerSlice, z sliceMeters, w camAltKm
    glm::vec4           FrustumX;       // basis X
    glm::vec4           FrustumY;       // basis Y
    glm::vec4           FrustumForward; // forward
};
struct CubeFromSkyViewPush {
    glm::vec4 FaceSize;
    glm::vec4 UpAxis;
    glm::vec4 ForwardAxis;
    glm::vec4 RightAxis;
    glm::vec4 SunDirSize;
    glm::vec4 SunColor;
};
#pragma pack(pop)

// --- hash of SkySettings for the bake gate ----------------------------------

// FNV-1a primitive.
inline void HashMix(uint64_t& h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(b[i]);
        h *= 1099511628211ull;
    }
}

uint64_t HashSettings(const SkySettings& s) {
    // Everything that drives the *displayed* Sky cube. Atmosphere params are
    // included here too — the SkyView raymarch consumes them every frame, so
    // changing Rayleigh/Mie/Ozone/MieG/radii must trigger a cube rebake even
    // when the static LUTs haven't moved (which was the previous bug — sliders
    // only took effect when the camera-altitude path dirtied this hash).
    uint64_t h = 1469598103934665603ull;
    HashMix(h, &s.SunColor,              sizeof(s.SunColor));
    HashMix(h, &s.SunIntensity,          sizeof(s.SunIntensity));
    HashMix(h, &s.SunAngularSizeDeg,     sizeof(s.SunAngularSizeDeg));
    HashMix(h, &s.SunDirection,          sizeof(s.SunDirection));
    HashMix(h, &s.SkyIntensity,          sizeof(s.SkyIntensity));
    HashMix(h, &s.BakeAerialPerspective, sizeof(s.BakeAerialPerspective));
    HashMix(h, &s.CameraAltitudeMeters,  sizeof(s.CameraAltitudeMeters));
    HashMix(h, &s.PlanetRadiusKm,        sizeof(s.PlanetRadiusKm));
    HashMix(h, &s.AtmosphereRadiusKm,    sizeof(s.AtmosphereRadiusKm));
    HashMix(h, &s.RayleighScattering,    sizeof(s.RayleighScattering));
    HashMix(h, &s.MieScattering,         sizeof(s.MieScattering));
    HashMix(h, &s.MieG,                  sizeof(s.MieG));
    HashMix(h, &s.OzoneAbsorption,       sizeof(s.OzoneAbsorption));
    return h ? h : 1ull;
}

// Atmosphere-only hash — gates the static Transmittance + MultiScatter LUTs.
// Sun direction and SkyIntensity don't appear here (they only affect SkyView).
uint64_t HashAtmosphere(const SkySettings& s) {
    uint64_t h = 1469598103934665603ull;
    HashMix(h, &s.PlanetRadiusKm,     sizeof(s.PlanetRadiusKm));
    HashMix(h, &s.AtmosphereRadiusKm, sizeof(s.AtmosphereRadiusKm));
    HashMix(h, &s.RayleighScattering, sizeof(s.RayleighScattering));
    HashMix(h, &s.MieScattering,      sizeof(s.MieScattering));
    HashMix(h, &s.MieG,               sizeof(s.MieG));
    HashMix(h, &s.OzoneAbsorption,    sizeof(s.OzoneAbsorption));
    return h ? h : 1ull;
}

AtmosphereParamsGpu PackAtmosphere(const SkySettings& s) {
    AtmosphereParamsGpu p{};
    p.RayleighScatteringAndPlanetRadius = glm::vec4(s.RayleighScattering, s.PlanetRadiusKm);
    p.MieAndAtmoTopAndG                  = glm::vec4(s.MieScattering, s.AtmosphereRadiusKm, s.MieG, 0.0f);
    p.OzoneAbsorption                    = glm::vec4(s.OzoneAbsorption, 0.0f);
    return p;
}

} // namespace

bool SkyAtmosphereInitialize(SkyAtmosphere& sa, const VulkanContext& ctx,
                             const char* shaderArtifactsDir) {
    if (sa.Initialized) return true;
    constexpr VkFormat kCubeFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    if (!CreateCubeImage(ctx, SkyAtmosphere::kSkySize,       1,
                         kCubeFormat, sa.Sky) ||
        !CreateCubeImage(ctx, SkyAtmosphere::kIrrSize,       1,
                         kCubeFormat, sa.Irradiance) ||
        !CreateCubeImage(ctx, SkyAtmosphere::kPrefilterSize, SkyAtmosphere::kPrefilterMips,
                         kCubeFormat, sa.Prefilter)) {
        RS_LOG_ERROR("SkyAtmosphere: cube image creation failed");
        return false;
    }
    if (!CreateLutImage(ctx, sa)) {
        RS_LOG_ERROR("SkyAtmosphere: BRDF LUT creation failed");
        return false;
    }
    // Hillaire LUT images — same RGBA16F format as the sky cube. Failures here
    // tear down with the rest in Terminate; we treat them as a hard error
    // (memory should always be available at this size).
    if (!CreateLut2D(ctx, SkyAtmosphere::kTransmittanceWidth,
                     SkyAtmosphere::kTransmittanceHeight,
                     kCubeFormat, sa.Transmittance) ||
        !CreateLut2D(ctx, SkyAtmosphere::kMultiScatterSize,
                     SkyAtmosphere::kMultiScatterSize,
                     kCubeFormat, sa.MultiScatter) ||
        !CreateLut2D(ctx, SkyAtmosphere::kSkyViewWidth,
                     SkyAtmosphere::kSkyViewHeight,
                     kCubeFormat, sa.SkyView) ||
        !CreateLut3D(ctx, SkyAtmosphere::kAerialSize,
                     SkyAtmosphere::kAerialSize,
                     SkyAtmosphere::kAerialSize,
                     kCubeFormat, sa.AerialPerspective)) {
        RS_LOG_ERROR("SkyAtmosphere: Hillaire LUT image creation failed");
        return false;
    }
    if (!CreateSamplers(ctx.Device, sa)) {
        RS_LOG_ERROR("SkyAtmosphere: sampler creation failed");
        return false;
    }

    // -- descriptor set layouts --
    // ibl_irradiance.comp: 0 = sampled cube (sky), 1 = storage cube (irradiance mip0)
    if (!MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.IrradianceSetLayout)) return false;
    // ibl_prefilter.comp: 0 = sampled cube (sky), 1 = storage cube (prefilter per-mip)
    if (!MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.PrefilterSetLayout)) return false;
    // brdf_lut.comp: 0 = storage 2D
    if (!MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.BrdfSetLayout)) return false;

    // Hillaire set layouts.
    if (!MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.TransmittanceSetLayout) ||
        !MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.MultiScatterSetLayout) ||
        !MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.SkyViewSetLayout) ||
        !MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.AerialSetLayout) ||
        !MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.CubeFromSkyViewLayout)) {
        RS_LOG_ERROR("SkyAtmosphere: Hillaire set layouts failed");
        return false;
    }

    // -- pipelines --
    const std::string base = shaderArtifactsDir ? shaderArtifactsDir : "";
    const std::string irrSpv       = base + "/ibl_irradiance.spv";
    const std::string prefilterSpv = base + "/ibl_prefilter.spv";
    const std::string brdfSpv      = base + "/brdf_lut.spv";

    if (!MakeComputePipeline(ctx.Device, irrSpv.c_str(), sa.IrradianceSetLayout,
                             sizeof(IrradiancePush),
                             sa.IrradiancePipelineLayout, sa.IrradiancePipeline)) {
        RS_LOG_ERROR("SkyAtmosphere: ibl_irradiance pipeline failed");
        return false;
    }
    if (!MakeComputePipeline(ctx.Device, prefilterSpv.c_str(), sa.PrefilterSetLayout,
                             sizeof(PrefilterPush),
                             sa.PrefilterPipelineLayout, sa.PrefilterPipeline)) {
        RS_LOG_ERROR("SkyAtmosphere: ibl_prefilter pipeline failed");
        return false;
    }
    if (!MakeComputePipeline(ctx.Device, brdfSpv.c_str(), sa.BrdfSetLayout,
                             sizeof(BrdfPush),
                             sa.BrdfPipelineLayout, sa.BrdfPipeline)) {
        RS_LOG_ERROR("SkyAtmosphere: brdf_lut pipeline failed");
        return false;
    }

    // Hillaire compute pipelines — mandatory now that sky_bake.comp is gone.
    {
        const std::string trSpv  = base + "/sky_transmittance.spv";
        const std::string msSpv  = base + "/sky_multiscatter.spv";
        const std::string svSpv  = base + "/sky_skyview.spv";
        const std::string apSpv  = base + "/sky_aerial.spv";
        const std::string csvSpv = base + "/sky_cube_from_skyview.spv";

        if (!MakeComputePipeline(ctx.Device, trSpv.c_str(),  sa.TransmittanceSetLayout,
                                 sizeof(TransmittancePush),
                                 sa.TransmittancePipelineLayout, sa.TransmittancePipeline) ||
            !MakeComputePipeline(ctx.Device, msSpv.c_str(),  sa.MultiScatterSetLayout,
                                 sizeof(MultiScatterPush),
                                 sa.MultiScatterPipelineLayout,  sa.MultiScatterPipeline) ||
            !MakeComputePipeline(ctx.Device, svSpv.c_str(),  sa.SkyViewSetLayout,
                                 sizeof(SkyViewPush),
                                 sa.SkyViewPipelineLayout,       sa.SkyViewPipeline) ||
            !MakeComputePipeline(ctx.Device, apSpv.c_str(),  sa.AerialSetLayout,
                                 sizeof(AerialPush),
                                 sa.AerialPipelineLayout,        sa.AerialPipeline) ||
            !MakeComputePipeline(ctx.Device, csvSpv.c_str(), sa.CubeFromSkyViewLayout,
                                 sizeof(CubeFromSkyViewPush),
                                 sa.CubeFromSkyViewPipelineLayout, sa.CubeFromSkyViewPipeline)) {
            RS_LOG_ERROR("SkyAtmosphere: Hillaire pipelines failed to load");
            return false;
        }
    }

    // -- descriptor pool + sets --
    // Storage images: 1 irradiance + N prefilter + 1 brdf + 5 Hillaire
    //                 (Transmittance, MultiScatter, SkyView, AerialPerspective,
    //                  CubeFromSkyView-cube).
    // Sampled images: 1 irradiance + N prefilter + 8 Hillaire (MS: 1, SV: 2,
    //                 AP: 2, CSV: 1 + slack).
    const uint32_t hillaireStorage = 5u;
    const uint32_t hillaireSampled = 8u;
    const uint32_t hillaireSets    = 5u;

    const uint32_t storageNeeded = 1 + SkyAtmosphere::kPrefilterMips + 1 + hillaireStorage;
    const uint32_t sampledNeeded = 1 + SkyAtmosphere::kPrefilterMips + hillaireSampled;
    const uint32_t setsNeeded    = 1 + SkyAtmosphere::kPrefilterMips + 1 + hillaireSets;

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[0].descriptorCount = storageNeeded;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = sampledNeeded;

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = setsNeeded;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(ctx.Device, &pci, nullptr, &sa.DescriptorPool) != VK_SUCCESS) {
        RS_LOG_ERROR("SkyAtmosphere: descriptor pool failed");
        return false;
    }

    sa.IrradianceDescriptor = AllocSet(ctx.Device, sa.DescriptorPool, sa.IrradianceSetLayout);
    sa.BrdfDescriptor       = AllocSet(ctx.Device, sa.DescriptorPool, sa.BrdfSetLayout);
    for (uint32_t m = 0; m < SkyAtmosphere::kPrefilterMips; ++m) {
        sa.PrefilterDescriptors[m] = AllocSet(ctx.Device, sa.DescriptorPool, sa.PrefilterSetLayout);
    }
    if (!sa.IrradianceDescriptor || !sa.BrdfDescriptor) {
        RS_LOG_ERROR("SkyAtmosphere: descriptor set alloc failed");
        return false;
    }

    // Irradiance set — sampled sky + storage irradiance.
    {
        VkDescriptorImageInfo sampleInfo{};
        sampleInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sampleInfo.imageView   = sa.Sky.View;
        sampleInfo.sampler     = sa.SamplerCubeLinear;
        VkDescriptorImageInfo storeInfo{};
        storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        storeInfo.imageView   = sa.Irradiance.Storage[0];
        VkWriteDescriptorSet w[2]{};
        w[0].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet         = sa.IrradianceDescriptor;
        w[0].dstBinding     = 0;
        w[0].descriptorCount= 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo     = &sampleInfo;
        w[1].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet         = sa.IrradianceDescriptor;
        w[1].dstBinding     = 1;
        w[1].descriptorCount= 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo     = &storeInfo;
        vkUpdateDescriptorSets(ctx.Device, 2, w, 0, nullptr);
    }
    // Prefilter sets (one per mip) — sampled sky + storage prefilter[mip].
    for (uint32_t m = 0; m < SkyAtmosphere::kPrefilterMips; ++m) {
        VkDescriptorImageInfo sampleInfo{};
        sampleInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sampleInfo.imageView   = sa.Sky.View;
        sampleInfo.sampler     = sa.SamplerCubeLinear;
        VkDescriptorImageInfo storeInfo{};
        storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        storeInfo.imageView   = sa.Prefilter.Storage[m];
        VkWriteDescriptorSet w[2]{};
        w[0].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet         = sa.PrefilterDescriptors[m];
        w[0].dstBinding     = 0;
        w[0].descriptorCount= 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo     = &sampleInfo;
        w[1].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet         = sa.PrefilterDescriptors[m];
        w[1].dstBinding     = 1;
        w[1].descriptorCount= 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo     = &storeInfo;
        vkUpdateDescriptorSets(ctx.Device, 2, w, 0, nullptr);
    }
    // BRDF LUT set — storage image only.
    {
        VkDescriptorImageInfo storeInfo{};
        storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        storeInfo.imageView   = sa.LutView;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = sa.BrdfDescriptor;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo      = &storeInfo;
        vkUpdateDescriptorSets(ctx.Device, 1, &w, 0, nullptr);
    }

    // -- Hillaire descriptor sets --
    sa.TransmittanceDescriptor    = AllocSet(ctx.Device, sa.DescriptorPool, sa.TransmittanceSetLayout);
    sa.MultiScatterDescriptor     = AllocSet(ctx.Device, sa.DescriptorPool, sa.MultiScatterSetLayout);
    sa.SkyViewDescriptor          = AllocSet(ctx.Device, sa.DescriptorPool, sa.SkyViewSetLayout);
    sa.AerialDescriptor           = AllocSet(ctx.Device, sa.DescriptorPool, sa.AerialSetLayout);
    sa.CubeFromSkyViewDescriptor  = AllocSet(ctx.Device, sa.DescriptorPool, sa.CubeFromSkyViewLayout);
    if (!sa.TransmittanceDescriptor || !sa.MultiScatterDescriptor ||
        !sa.SkyViewDescriptor || !sa.AerialDescriptor ||
        !sa.CubeFromSkyViewDescriptor) {
        RS_LOG_ERROR("SkyAtmosphere: Hillaire descriptor alloc failed");
        return false;
    }
    {
        // Transmittance — storage only.
        {
            VkDescriptorImageInfo storeInfo{};
            storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            storeInfo.imageView   = sa.Transmittance.View;
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet          = sa.TransmittanceDescriptor;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo      = &storeInfo;
            vkUpdateDescriptorSets(ctx.Device, 1, &w, 0, nullptr);
        }
        // MultiScatter — read Transmittance, write MultiScatter.
        {
            VkDescriptorImageInfo sampleInfo{};
            sampleInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sampleInfo.imageView   = sa.Transmittance.View;
            sampleInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo storeInfo{};
            storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            storeInfo.imageView   = sa.MultiScatter.View;
            VkWriteDescriptorSet w[2]{};
            w[0].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet         = sa.MultiScatterDescriptor;
            w[0].dstBinding     = 0;
            w[0].descriptorCount= 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo     = &sampleInfo;
            w[1].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet         = sa.MultiScatterDescriptor;
            w[1].dstBinding     = 1;
            w[1].descriptorCount= 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = &storeInfo;
            vkUpdateDescriptorSets(ctx.Device, 2, w, 0, nullptr);
        }
        // SkyView — read Transmittance + MultiScatter, write SkyView.
        {
            VkDescriptorImageInfo trInfo{};
            trInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            trInfo.imageView   = sa.Transmittance.View;
            trInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo msInfo{};
            msInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            msInfo.imageView   = sa.MultiScatter.View;
            msInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo svInfo{};
            svInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            svInfo.imageView   = sa.SkyView.View;
            VkWriteDescriptorSet w[3]{};
            for (int i = 0; i < 3; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = sa.SkyViewDescriptor;
                w[i].dstBinding      = static_cast<uint32_t>(i);
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &trInfo;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &msInfo;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[2].pImageInfo = &svInfo;
            vkUpdateDescriptorSets(ctx.Device, 3, w, 0, nullptr);
        }
        // AerialPerspective — same inputs, 3D storage out.
        {
            VkDescriptorImageInfo trInfo{};
            trInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            trInfo.imageView   = sa.Transmittance.View;
            trInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo msInfo{};
            msInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            msInfo.imageView   = sa.MultiScatter.View;
            msInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo apInfo{};
            apInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            apInfo.imageView   = sa.AerialPerspective.View;
            VkWriteDescriptorSet w[3]{};
            for (int i = 0; i < 3; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = sa.AerialDescriptor;
                w[i].dstBinding      = static_cast<uint32_t>(i);
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &trInfo;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &msInfo;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[2].pImageInfo = &apInfo;
            vkUpdateDescriptorSets(ctx.Device, 3, w, 0, nullptr);
        }
        // CubeFromSkyView — read SkyView, write Sky cube.
        {
            VkDescriptorImageInfo svInfo{};
            svInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            svInfo.imageView   = sa.SkyView.View;
            svInfo.sampler     = sa.SamplerLutLinear;
            VkDescriptorImageInfo cubeInfo{};
            cubeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            cubeInfo.imageView   = sa.Sky.Storage[0];
            VkWriteDescriptorSet w[2]{};
            w[0].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet         = sa.CubeFromSkyViewDescriptor;
            w[0].dstBinding     = 0;
            w[0].descriptorCount= 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo     = &svInfo;
            w[1].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet         = sa.CubeFromSkyViewDescriptor;
            w[1].dstBinding     = 1;
            w[1].descriptorCount= 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = &cubeInfo;
            vkUpdateDescriptorSets(ctx.Device, 2, w, 0, nullptr);
        }
    }

    sa.Initialized = true;
    RS_LOG_INFO("SkyAtmosphere ready: sky=%u² irr=%u² prefilter=%u²×%umip lut=%u²",
                SkyAtmosphere::kSkySize,
                SkyAtmosphere::kIrrSize,
                SkyAtmosphere::kPrefilterSize,
                SkyAtmosphere::kPrefilterMips,
                SkyAtmosphere::kBrdfLutSize);
    return true;
}

void SkyAtmosphereTerminate(SkyAtmosphere& sa, const VulkanContext& ctx) {
    if (!sa.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);

    if (sa.IrradiancePipeline)       vkDestroyPipeline      (ctx.Device, sa.IrradiancePipeline,       nullptr);
    if (sa.PrefilterPipeline)        vkDestroyPipeline      (ctx.Device, sa.PrefilterPipeline,        nullptr);
    if (sa.BrdfPipeline)             vkDestroyPipeline      (ctx.Device, sa.BrdfPipeline,             nullptr);
    if (sa.TransmittancePipeline)    vkDestroyPipeline      (ctx.Device, sa.TransmittancePipeline,    nullptr);
    if (sa.MultiScatterPipeline)     vkDestroyPipeline      (ctx.Device, sa.MultiScatterPipeline,     nullptr);
    if (sa.SkyViewPipeline)          vkDestroyPipeline      (ctx.Device, sa.SkyViewPipeline,          nullptr);
    if (sa.AerialPipeline)           vkDestroyPipeline      (ctx.Device, sa.AerialPipeline,           nullptr);
    if (sa.CubeFromSkyViewPipeline)  vkDestroyPipeline      (ctx.Device, sa.CubeFromSkyViewPipeline,  nullptr);
    if (sa.IrradiancePipelineLayout) vkDestroyPipelineLayout(ctx.Device, sa.IrradiancePipelineLayout, nullptr);
    if (sa.PrefilterPipelineLayout)  vkDestroyPipelineLayout(ctx.Device, sa.PrefilterPipelineLayout,  nullptr);
    if (sa.BrdfPipelineLayout)       vkDestroyPipelineLayout(ctx.Device, sa.BrdfPipelineLayout,       nullptr);
    if (sa.TransmittancePipelineLayout) vkDestroyPipelineLayout(ctx.Device, sa.TransmittancePipelineLayout, nullptr);
    if (sa.MultiScatterPipelineLayout)  vkDestroyPipelineLayout(ctx.Device, sa.MultiScatterPipelineLayout,  nullptr);
    if (sa.SkyViewPipelineLayout)       vkDestroyPipelineLayout(ctx.Device, sa.SkyViewPipelineLayout,       nullptr);
    if (sa.AerialPipelineLayout)        vkDestroyPipelineLayout(ctx.Device, sa.AerialPipelineLayout,        nullptr);
    if (sa.CubeFromSkyViewPipelineLayout) vkDestroyPipelineLayout(ctx.Device, sa.CubeFromSkyViewPipelineLayout, nullptr);
    if (sa.IrradianceSetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, sa.IrradianceSetLayout, nullptr);
    if (sa.PrefilterSetLayout)       vkDestroyDescriptorSetLayout(ctx.Device, sa.PrefilterSetLayout,  nullptr);
    if (sa.BrdfSetLayout)            vkDestroyDescriptorSetLayout(ctx.Device, sa.BrdfSetLayout,       nullptr);
    if (sa.TransmittanceSetLayout)   vkDestroyDescriptorSetLayout(ctx.Device, sa.TransmittanceSetLayout, nullptr);
    if (sa.MultiScatterSetLayout)    vkDestroyDescriptorSetLayout(ctx.Device, sa.MultiScatterSetLayout,  nullptr);
    if (sa.SkyViewSetLayout)         vkDestroyDescriptorSetLayout(ctx.Device, sa.SkyViewSetLayout,       nullptr);
    if (sa.AerialSetLayout)          vkDestroyDescriptorSetLayout(ctx.Device, sa.AerialSetLayout,        nullptr);
    if (sa.CubeFromSkyViewLayout)    vkDestroyDescriptorSetLayout(ctx.Device, sa.CubeFromSkyViewLayout,  nullptr);
    if (sa.DescriptorPool)           vkDestroyDescriptorPool(ctx.Device, sa.DescriptorPool, nullptr);

    if (sa.LutView)   vkDestroyImageView(ctx.Device, sa.LutView,   nullptr);
    if (sa.LutImage)  vkDestroyImage    (ctx.Device, sa.LutImage,  nullptr);
    if (sa.LutMemory) vkFreeMemory      (ctx.Device, sa.LutMemory, nullptr);

    DestroyCubeImage(ctx.Device, sa.Sky);
    DestroyCubeImage(ctx.Device, sa.Irradiance);
    DestroyCubeImage(ctx.Device, sa.Prefilter);
    DestroyLut2D    (ctx.Device, sa.Transmittance);
    DestroyLut2D    (ctx.Device, sa.MultiScatter);
    DestroyLut2D    (ctx.Device, sa.SkyView);
    DestroyLut3D    (ctx.Device, sa.AerialPerspective);

    if (sa.SamplerCubeLinear) vkDestroySampler(ctx.Device, sa.SamplerCubeLinear, nullptr);
    if (sa.SamplerLut)        vkDestroySampler(ctx.Device, sa.SamplerLut,        nullptr);
    if (sa.SamplerLutLinear)  vkDestroySampler(ctx.Device, sa.SamplerLutLinear,  nullptr);

    sa = {};
}

void SkyAtmosphereEnsureBaked(SkyAtmosphere& sa, const VulkanContext& ctx,
                              VkCommandBuffer cmd,
                              const SkySettings& settings) {
    if (!sa.Initialized) return;

    const uint64_t hash         = HashSettings(settings);
    const uint64_t atmoHash     = HashAtmosphere(settings);
    const bool needCubes        = (hash != sa.LastBakedHash);
    const bool needBrdfLut      = !sa.LutBaked;
    const bool needTransmittance= !sa.TransmittanceBaked || atmoHash != sa.LastAtmosphereHash;
    const bool needMultiScatter = !sa.MultiScatterBaked  || atmoHash != sa.LastAtmosphereHash;
    if (!needCubes && !needBrdfLut && !needTransmittance && !needMultiScatter) return;

    // -- BRDF LUT (one-shot, settings-independent) --
    if (needBrdfLut) {
        TransitionToGeneral(cmd, sa.LutImage, 0, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.BrdfPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.BrdfPipelineLayout,
                                0, 1, &sa.BrdfDescriptor, 0, nullptr);
        BrdfPush bp{};
        bp.SizeAndSamples = glm::vec4(static_cast<float>(SkyAtmosphere::kBrdfLutSize),
                                      static_cast<float>(SkyAtmosphere::kBrdfLutSize),
                                      static_cast<float>(SkyAtmosphere::kBrdfSampleCount),
                                      0.0f);
        vkCmdPushConstants(cmd, sa.BrdfPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(bp), &bp);
        const uint32_t groups = (SkyAtmosphere::kBrdfLutSize + 7u) / 8u;
        vkCmdDispatch(cmd, groups, groups, 1);
        TransitionToShaderRead(cmd, sa.LutImage, 0, 1, 1);
        sa.LutBaked = true;
    }

    // ---- Hillaire static LUTs (Transmittance + MultiScatter) ----
    // These bake exactly once per atmosphere-parameter change. SkyView and
    // CubeFromSkyView run later under needCubes (sun direction is in the cubes
    // hash, atmosphere params are in atmoHash).
    if (needTransmittance) {
        TransitionToGeneral(cmd, sa.Transmittance.Image, 0, 1, 1,
                            sa.TransmittanceBaked ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_UNDEFINED);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.TransmittancePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sa.TransmittancePipelineLayout,
                                0, 1, &sa.TransmittanceDescriptor, 0, nullptr);
        TransmittancePush tp{};
        tp.Atmo    = PackAtmosphere(settings);
        tp.LutSize = glm::vec4(static_cast<float>(SkyAtmosphere::kTransmittanceWidth),
                               static_cast<float>(SkyAtmosphere::kTransmittanceHeight),
                               static_cast<float>(SkyAtmosphere::kTransmittanceSteps),
                               0.0f);
        vkCmdPushConstants(cmd, sa.TransmittancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(tp), &tp);
        const uint32_t gx = (SkyAtmosphere::kTransmittanceWidth  + 7u) / 8u;
        const uint32_t gy = (SkyAtmosphere::kTransmittanceHeight + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 1);
        TransitionToShaderRead(cmd, sa.Transmittance.Image, 0, 1, 1);
        sa.TransmittanceBaked = true;
    }
    if (needMultiScatter) {
        TransitionToGeneral(cmd, sa.MultiScatter.Image, 0, 1, 1,
                            sa.MultiScatterBaked ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.MultiScatterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sa.MultiScatterPipelineLayout,
                                0, 1, &sa.MultiScatterDescriptor, 0, nullptr);
        MultiScatterPush mp{};
        mp.Atmo  = PackAtmosphere(settings);
        mp.Sizes = glm::vec4(static_cast<float>(SkyAtmosphere::kMultiScatterSize),
                             static_cast<float>(SkyAtmosphere::kTransmittanceWidth),
                             static_cast<float>(SkyAtmosphere::kTransmittanceHeight),
                             static_cast<float>(SkyAtmosphere::kMultiScatterSteps));
        // Sun irradiance at TOA = unitless 1.0 — the renderer's sun gain
        // multiplies later in SkyView.
        mp.SunIrr = glm::vec4(1.0f, 1.0f, 1.0f, static_cast<float>(SkyAtmosphere::kMultiScatterSqrtN));
        vkCmdPushConstants(cmd, sa.MultiScatterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(mp), &mp);
        const uint32_t g = (SkyAtmosphere::kMultiScatterSize + 7u) / 8u;
        vkCmdDispatch(cmd, g, g, 1);
        TransitionToShaderRead(cmd, sa.MultiScatter.Image, 0, 1, 1);
        sa.MultiScatterBaked  = true;
        sa.LastAtmosphereHash = atmoHash;
    }

    if (!needCubes) return;

    // -- Sky cubemap fill: SkyView raymarch → CubeFromSkyView projection. --
    TransitionToGeneral(cmd, sa.Sky.Image, 0, 1, 6,
                        (sa.LastBakedHash == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    {
        // --- SkyView LUT bake ---
        // SkyView lives in its own image — needs GENERAL during write.
        // Because the SkyView image is in SHADER_READ_ONLY most of the time,
        // we need to transition both directions.
        TransitionToGeneral(cmd, sa.SkyView.Image, 0, 1, 1,
                            (sa.LastBakedHash == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.SkyViewPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sa.SkyViewPipelineLayout,
                                0, 1, &sa.SkyViewDescriptor, 0, nullptr);
        // Build a local frame: up = world Y (renderer is Y-up), forward =
        // projection of sun direction onto the horizon, right = up × forward.
        const glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 sunW = glm::normalize(settings.SunDirection);
        glm::vec3 horizSun = sunW - worldUp * glm::dot(sunW, worldUp);
        if (glm::length(horizSun) < 1e-3f) horizSun = glm::vec3(0.0f, 0.0f, -1.0f);
        const glm::vec3 forward = glm::normalize(horizSun);
        const glm::vec3 right   = glm::normalize(glm::cross(worldUp, forward));
        // Express sun in the (right, forward, up) local frame the LUT uses.
        const glm::vec3 sunLocal = glm::vec3(glm::dot(sunW, right),
                                             glm::dot(sunW, forward),
                                             glm::dot(sunW, worldUp));
        SkyViewPush svp{};
        svp.Atmo                = PackAtmosphere(settings);
        // .w = sky inscatter gain (decoupled from sun intensity so the IBL
        // ambient doesn't get blown out when the analytic disc is dialed up).
        svp.SunDirAndSkyGain    = glm::vec4(sunLocal, settings.SkyIntensity);
        constexpr float kDegToRad = 3.14159265359f / 180.0f;
        // Sun colour pre-multiplied by sun intensity — the analytic disc uses
        // this directly so we don't need a separate scalar in the shader.
        svp.SunColorAndSize     = glm::vec4(settings.SunColor * settings.SunIntensity,
                                            settings.SunAngularSizeDeg * kDegToRad);
        svp.ViewSize            = glm::vec4(static_cast<float>(SkyAtmosphere::kSkyViewWidth),
                                            static_cast<float>(SkyAtmosphere::kSkyViewHeight),
                                            static_cast<float>(SkyAtmosphere::kSkyViewSteps),
                                            settings.CameraAltitudeMeters * 0.001f);
        vkCmdPushConstants(cmd, sa.SkyViewPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(svp), &svp);
        const uint32_t gx = (SkyAtmosphere::kSkyViewWidth  + 7u) / 8u;
        const uint32_t gy = (SkyAtmosphere::kSkyViewHeight + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 1);
        TransitionToShaderRead(cmd, sa.SkyView.Image, 0, 1, 1);

        // --- Project SkyView into Sky cube ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.CubeFromSkyViewPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sa.CubeFromSkyViewPipelineLayout,
                                0, 1, &sa.CubeFromSkyViewDescriptor, 0, nullptr);
        CubeFromSkyViewPush csp{};
        csp.FaceSize    = glm::vec4(static_cast<float>(SkyAtmosphere::kSkySize), 0, 0, 0);
        csp.UpAxis      = glm::vec4(worldUp, 0.0f);
        csp.ForwardAxis = glm::vec4(forward, 0.0f);
        csp.RightAxis   = glm::vec4(right,   0.0f);
        csp.SunDirSize  = glm::vec4(sunW, settings.SunAngularSizeDeg * kDegToRad);
        csp.SunColor    = glm::vec4(settings.SunColor, settings.SunIntensity);
        vkCmdPushConstants(cmd, sa.CubeFromSkyViewPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(csp), &csp);
        const uint32_t groups = (SkyAtmosphere::kSkySize + 7u) / 8u;
        vkCmdDispatch(cmd, groups, groups, 6);

        // --- Optional: AerialPerspective LUT (toggle-gated). ---
        if (settings.BakeAerialPerspective) {
            TransitionToGeneral(cmd, sa.AerialPerspective.Image, 0, 1, 1,
                                VK_IMAGE_LAYOUT_UNDEFINED);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.AerialPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    sa.AerialPipelineLayout,
                                    0, 1, &sa.AerialDescriptor, 0, nullptr);
            AerialPush ap{};
            ap.Atmo               = PackAtmosphere(settings);
            ap.SunDirAndIntensity = glm::vec4(sunLocal, settings.SunIntensity);
            ap.ViewParams         = glm::vec4(static_cast<float>(SkyAtmosphere::kAerialSize),
                                              static_cast<float>(SkyAtmosphere::kAerialStepsPerSlice),
                                              SkyAtmosphere::kAerialSliceMeters,
                                              settings.CameraAltitudeMeters * 0.001f);
            // Placeholder frustum: 90° FOV looking forward. The future
            // fog-on-geometry consumer will fill in a real frustum.
            ap.FrustumX        = glm::vec4(right,   0.0f);
            ap.FrustumY        = glm::vec4(worldUp, 0.0f);
            ap.FrustumForward  = glm::vec4(forward, 0.0f);
            vkCmdPushConstants(cmd, sa.AerialPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(ap), &ap);
            const uint32_t g = (SkyAtmosphere::kAerialSize + 7u) / 8u;
            vkCmdDispatch(cmd, g, g, SkyAtmosphere::kAerialSize);
            TransitionToShaderRead(cmd, sa.AerialPerspective.Image, 0, 1, 1);
        }
    }
    TransitionToShaderRead(cmd, sa.Sky.Image, 0, 1, 6);

    // -- Irradiance --
    TransitionToGeneral(cmd, sa.Irradiance.Image, 0, 1, 6,
                        (sa.LastBakedHash == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.IrradiancePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.IrradiancePipelineLayout,
                            0, 1, &sa.IrradianceDescriptor, 0, nullptr);
    {
        IrradiancePush ip{};
        // Step ≈ 0.035 rad → 180 × 90 ≈ 16k samples per texel. With 32×32×6 =
        // 6144 texels, the bake is well under a millisecond on the target GPU.
        ip.FaceSize = glm::vec4(static_cast<float>(SkyAtmosphere::kIrrSize),
                                0.035f, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, sa.IrradiancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ip), &ip);
    }
    {
        const uint32_t groups = (SkyAtmosphere::kIrrSize + 7u) / 8u;
        vkCmdDispatch(cmd, groups, groups, 6);
    }
    TransitionToShaderRead(cmd, sa.Irradiance.Image, 0, 1, 6);

    // -- Prefilter (one dispatch per mip) --
    // Bring all prefilter mips to GENERAL, dispatch one at a time, then bring
    // everything to SHADER_READ_ONLY at the end so PbrLit can sample.
    TransitionToGeneral(cmd, sa.Prefilter.Image, 0, SkyAtmosphere::kPrefilterMips, 6,
                        (sa.LastBakedHash == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.PrefilterPipeline);
    for (uint32_t m = 0; m < SkyAtmosphere::kPrefilterMips; ++m) {
        const uint32_t mipSize = SkyAtmosphere::kPrefilterSize >> m;
        const float    roughness = (SkyAtmosphere::kPrefilterMips > 1)
            ? static_cast<float>(m) / static_cast<float>(SkyAtmosphere::kPrefilterMips - 1)
            : 0.0f;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sa.PrefilterPipelineLayout,
                                0, 1, &sa.PrefilterDescriptors[m], 0, nullptr);
        PrefilterPush pp{};
        pp.FaceSizeRoughnessSamples = glm::vec4(
            static_cast<float>(mipSize),
            roughness,
            static_cast<float>(SkyAtmosphere::kPrefilterSampleCount),
            static_cast<float>(SkyAtmosphere::kSkySize));
        vkCmdPushConstants(cmd, sa.PrefilterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pp), &pp);
        const uint32_t groups = (mipSize + 7u) / 8u;
        vkCmdDispatch(cmd, groups ? groups : 1u, groups ? groups : 1u, 6);
    }
    TransitionToShaderRead(cmd, sa.Prefilter.Image, 0, SkyAtmosphere::kPrefilterMips, 6);

    sa.LastBakedHash = hash;
}

} // namespace RS
