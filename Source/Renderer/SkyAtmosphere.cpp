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
    return vkCreateSampler(device, &sci, nullptr, &sa.SamplerLut) == VK_SUCCESS;
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
struct SkyBakePush {
    glm::vec4 ZenithIntensity;
    glm::vec4 Horizon;
    glm::vec4 Ground;
    glm::vec4 SunDirSize;
    glm::vec4 SunColor;
    glm::vec4 FaceSize;
};
struct IrradiancePush {
    glm::vec4 FaceSize;       // x = face size, y = sample step radians
};
struct PrefilterPush {
    glm::vec4 FaceSizeRoughnessSamples;  // x size, y roughness, z samples, w src size
};
struct BrdfPush {
    glm::vec4 SizeAndSamples;  // x width, y height, z samples
};
#pragma pack(pop)

// --- hash of SkySettings for the bake gate ----------------------------------

uint64_t HashSettings(const SkySettings& s) {
    // FNV-1a over the bytes (ignoring the IBL toggle/intensity which are
    // runtime-only and don't affect the baked images).
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(b[i]);
            h *= 1099511628211ull;
        }
    };
    mix(&s.ZenithColor,        sizeof(s.ZenithColor));
    mix(&s.HorizonColor,       sizeof(s.HorizonColor));
    mix(&s.GroundColor,        sizeof(s.GroundColor));
    mix(&s.SkyIntensity,       sizeof(s.SkyIntensity));
    mix(&s.HorizonExponent,    sizeof(s.HorizonExponent));
    mix(&s.GroundExponent,     sizeof(s.GroundExponent));
    mix(&s.SunColor,           sizeof(s.SunColor));
    mix(&s.SunIntensity,       sizeof(s.SunIntensity));
    mix(&s.SunAngularSizeDeg,  sizeof(s.SunAngularSizeDeg));
    return h ? h : 1ull;
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
    if (!CreateSamplers(ctx.Device, sa)) {
        RS_LOG_ERROR("SkyAtmosphere: sampler creation failed");
        return false;
    }

    // -- descriptor set layouts --
    // sky_bake.comp: binding 0 = storage cube (sky mip0)
    if (!MakeDescriptorSetLayout(ctx.Device, {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        }, sa.SkySetLayout)) return false;
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

    // -- pipelines --
    const std::string base = shaderArtifactsDir ? shaderArtifactsDir : "";
    const std::string skySpv       = base + "/sky_bake.spv";
    const std::string irrSpv       = base + "/ibl_irradiance.spv";
    const std::string prefilterSpv = base + "/ibl_prefilter.spv";
    const std::string brdfSpv      = base + "/brdf_lut.spv";

    if (!MakeComputePipeline(ctx.Device, skySpv.c_str(), sa.SkySetLayout,
                             sizeof(SkyBakePush),
                             sa.SkyPipelineLayout, sa.SkyPipeline)) {
        RS_LOG_ERROR("SkyAtmosphere: sky_bake pipeline failed");
        return false;
    }
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

    // -- descriptor pool + sets --
    // 1 sky + 1 irradiance + N prefilter + 1 brdf storage images
    // 1 irradiance + N prefilter sampled cubes
    const uint32_t storageNeeded = 1 + 1 + SkyAtmosphere::kPrefilterMips + 1;
    const uint32_t sampledNeeded = 1 + SkyAtmosphere::kPrefilterMips;
    const uint32_t setsNeeded    = storageNeeded; // 1 set per storage write target

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

    sa.SkyDescriptor        = AllocSet(ctx.Device, sa.DescriptorPool, sa.SkySetLayout);
    sa.IrradianceDescriptor = AllocSet(ctx.Device, sa.DescriptorPool, sa.IrradianceSetLayout);
    sa.BrdfDescriptor       = AllocSet(ctx.Device, sa.DescriptorPool, sa.BrdfSetLayout);
    for (uint32_t m = 0; m < SkyAtmosphere::kPrefilterMips; ++m) {
        sa.PrefilterDescriptors[m] = AllocSet(ctx.Device, sa.DescriptorPool, sa.PrefilterSetLayout);
    }
    if (!sa.SkyDescriptor || !sa.IrradianceDescriptor || !sa.BrdfDescriptor) {
        RS_LOG_ERROR("SkyAtmosphere: descriptor set alloc failed");
        return false;
    }

    // Sky set — storage image only.
    {
        VkDescriptorImageInfo storeInfo{};
        storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        storeInfo.imageView   = sa.Sky.Storage[0];
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = sa.SkyDescriptor;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo      = &storeInfo;
        vkUpdateDescriptorSets(ctx.Device, 1, &w, 0, nullptr);
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

    if (sa.SkyPipeline)              vkDestroyPipeline      (ctx.Device, sa.SkyPipeline,              nullptr);
    if (sa.IrradiancePipeline)       vkDestroyPipeline      (ctx.Device, sa.IrradiancePipeline,       nullptr);
    if (sa.PrefilterPipeline)        vkDestroyPipeline      (ctx.Device, sa.PrefilterPipeline,        nullptr);
    if (sa.BrdfPipeline)             vkDestroyPipeline      (ctx.Device, sa.BrdfPipeline,             nullptr);
    if (sa.SkyPipelineLayout)        vkDestroyPipelineLayout(ctx.Device, sa.SkyPipelineLayout,        nullptr);
    if (sa.IrradiancePipelineLayout) vkDestroyPipelineLayout(ctx.Device, sa.IrradiancePipelineLayout, nullptr);
    if (sa.PrefilterPipelineLayout)  vkDestroyPipelineLayout(ctx.Device, sa.PrefilterPipelineLayout,  nullptr);
    if (sa.BrdfPipelineLayout)       vkDestroyPipelineLayout(ctx.Device, sa.BrdfPipelineLayout,       nullptr);
    if (sa.SkySetLayout)             vkDestroyDescriptorSetLayout(ctx.Device, sa.SkySetLayout,        nullptr);
    if (sa.IrradianceSetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, sa.IrradianceSetLayout, nullptr);
    if (sa.PrefilterSetLayout)       vkDestroyDescriptorSetLayout(ctx.Device, sa.PrefilterSetLayout,  nullptr);
    if (sa.BrdfSetLayout)            vkDestroyDescriptorSetLayout(ctx.Device, sa.BrdfSetLayout,       nullptr);
    if (sa.DescriptorPool)           vkDestroyDescriptorPool(ctx.Device, sa.DescriptorPool, nullptr);

    if (sa.LutView)   vkDestroyImageView(ctx.Device, sa.LutView,   nullptr);
    if (sa.LutImage)  vkDestroyImage    (ctx.Device, sa.LutImage,  nullptr);
    if (sa.LutMemory) vkFreeMemory      (ctx.Device, sa.LutMemory, nullptr);

    DestroyCubeImage(ctx.Device, sa.Sky);
    DestroyCubeImage(ctx.Device, sa.Irradiance);
    DestroyCubeImage(ctx.Device, sa.Prefilter);

    if (sa.SamplerCubeLinear) vkDestroySampler(ctx.Device, sa.SamplerCubeLinear, nullptr);
    if (sa.SamplerLut)        vkDestroySampler(ctx.Device, sa.SamplerLut,        nullptr);

    sa = {};
}

void SkyAtmosphereEnsureBaked(SkyAtmosphere& sa, const VulkanContext& ctx,
                              VkCommandBuffer cmd,
                              const SkySettings& settings) {
    if (!sa.Initialized) return;

    const uint64_t hash    = HashSettings(settings);
    const bool needCubes   = (hash != sa.LastBakedHash);
    const bool needBrdfLut = !sa.LutBaked;
    if (!needCubes && !needBrdfLut) return;

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

    if (!needCubes) return;

    // -- Sky cubemap --
    TransitionToGeneral(cmd, sa.Sky.Image, 0, 1, 6,
                        (sa.LastBakedHash == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.SkyPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sa.SkyPipelineLayout,
                            0, 1, &sa.SkyDescriptor, 0, nullptr);
    {
        SkyBakePush sp{};
        sp.ZenithIntensity = glm::vec4(settings.ZenithColor, settings.SkyIntensity);
        sp.Horizon         = glm::vec4(settings.HorizonColor, settings.HorizonExponent);
        sp.Ground          = glm::vec4(settings.GroundColor,  settings.GroundExponent);
        constexpr float kDegToRad = 3.14159265359f / 180.0f;
        sp.SunDirSize      = glm::vec4(glm::normalize(glm::vec3(0.3f, 0.85f, 0.5f)),
                                       settings.SunAngularSizeDeg * kDegToRad);
        sp.SunColor        = glm::vec4(settings.SunColor, settings.SunIntensity);
        sp.FaceSize        = glm::vec4(static_cast<float>(SkyAtmosphere::kSkySize), 0, 0, 0);
        vkCmdPushConstants(cmd, sa.SkyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(sp), &sp);
    }
    {
        const uint32_t groups = (SkyAtmosphere::kSkySize + 7u) / 8u;
        vkCmdDispatch(cmd, groups, groups, 6);
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
