// Source/Renderer/GBufferPreview.cpp — Phase 6
#include "Renderer/GBufferPreview.h"
#include "Core/Logger.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RS {

namespace {

#pragma pack(push, 16)
struct PreviewPushConstants {
    glm::vec4 SunDirAndIntensity;   // xyz, w = intensity
    glm::vec4 AmbientAndNearFar;    // x = ambient, y = near, z = far, w = mode

    // Matcap (used only when mode == Matcap). Layout matches the GLSL block:
    //   MatcapTop      = (top.rgb, gradientCurve)
    //   MatcapBottom   = (bottom.rgb, exposure)
    //   MatcapRim      = (rim.rgb,   rimPower)
    //   MatcapHighlight= (hi.rgb,    highlightSize)
    //   MatcapParams   = (rimStrength, highlightStrength, hiOffset.x, hiOffset.y)
    //   MatcapTintRow0 = (viewRot[0][0..2], tint.x)
    //   MatcapTintRow1 = (viewRot[1][0..2], tint.y)
    //   MatcapTintRow2 = (viewRot[2][0..2], tint.z)
    glm::vec4 MatcapTop;
    glm::vec4 MatcapBottom;
    glm::vec4 MatcapRim;
    glm::vec4 MatcapHighlight;
    glm::vec4 MatcapParams;
    glm::vec4 MatcapTintRow0;
    glm::vec4 MatcapTintRow1;
    glm::vec4 MatcapTintRow2;
    glm::vec4 MatcapMaterial;        // x = roughness, y = metallic, z/w = unused

    // Phase 8 IBL toggle. x = on/off (>= 0.5), y = intensity, z = prefilter mip
    // count, w = unused.
    glm::vec4 IblParams;

    // Phase 7 PBR: world-position reconstruction from depth. Eye derives from
    // InvViewProj * (0,0,0,1).
    glm::mat4 InvViewProj;
};
#pragma pack(pop)
static_assert(sizeof(PreviewPushConstants) <= 256,
              "PreviewPushConstants must stay within 256-byte push budget");

bool ReadFile(const char* path, std::vector<char>& outBuf) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { RS_LOG_ERROR("GBufferPreview: cannot open shader: %s", path); return false; }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0) { std::fclose(f); return false; }
    outBuf.resize(static_cast<size_t>(len));
    const size_t got = std::fread(outBuf.data(), 1, outBuf.size(), f);
    std::fclose(f);
    return got == outBuf.size();
}

VkShaderModule LoadModule(VkDevice device, const char* path) {
    std::vector<char> buf;
    if (!ReadFile(path, buf)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = buf.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) {
        RS_LOG_ERROR("GBufferPreview: vkCreateShaderModule failed: %s", path);
        return VK_NULL_HANDLE;
    }
    return m;
}

bool CreateRenderPass(VkDevice device, VkFormat swapFormat, VkRenderPass& outPass) {
    // loadOp=LOAD so the GridPass output survives; layout stays in
    // COLOR_ATTACHMENT_OPTIMAL on both sides so ImGui (loadOp=LOAD) follows
    // unchanged.
    VkAttachmentDescription color{};
    color.format         = swapFormat;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &color;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    return vkCreateRenderPass(device, &rpci, nullptr, &outPass) == VK_SUCCESS;
}

bool CreateSetLayout(VkDevice device, VkDescriptorSetLayout& outLayout) {
    // 0..5  = GBuffer slots
    // 6     = irradiance cube
    // 7     = prefilter cube
    // 8     = BRDF LUT
    // 9     = LightHDR (Phase 10 — lit mode samples this)
    // 10    = SDF 3D image (Phase 12 — SDF-slice dense view)
    // 11    = SDF params UBO (Phase 12; grew to 96 B in 15f)
    // 12    = Specular SRV (KHR_materials_specular)
    // 13,14 = Sparse SDF SSBOs (Phase 15f — BrickIndex + BrickPool)
    VkDescriptorSetLayoutBinding bindings[15]{};
    for (uint32_t i = 0; i < 11; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[11].binding         = 11;
    bindings[11].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[12].binding         = 12;
    bindings[12].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[13].binding         = 13;
    bindings[13].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[14].binding         = 14;
    bindings[14].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 15;
    lci.pBindings    = bindings;
    return vkCreateDescriptorSetLayout(device, &lci, nullptr, &outLayout) == VK_SUCCESS;
}

// Phase 12 → 15f SDF UBO (must match GLSL std140 layout). 96 bytes after
// 15f added the sparse AABB/dims trio.
struct SdfPushUbo {
    glm::vec4  AABBMinAndPlaneY;  // xyz = dense AABB min, w = world-Y of slice plane
    glm::vec4  AABBMaxAndMaxDist; // xyz = dense AABB max, w = R16_SNORM decode scale
    glm::vec4  PreviewParams;     // x = SDFVizMode (float), y = SliceSource (0=dense,1=sparse), zw reserved
    glm::vec4  SparseAABBMin;     // xyz = sparse AABB min, w = sparse MaxDist
    glm::vec4  SparseAABBMax;     // xyz = sparse AABB max, w = sparse decode scale (MaxDist/32767)
    glm::uvec4 SparseDims;        // x = res, y = brickSize, z = brickGrid, w = bsLog2
};
static_assert(sizeof(SdfPushUbo) == 96, "SdfPushUbo must be 96 bytes");

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

bool CreateUboBuffer(const VulkanContext& ctx, VkDeviceSize bytes,
                     VkBuffer& outBuf, VkDeviceMemory& outMem, void*& outMapped) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bytes;
    bci.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &outBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, outBuf, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &outMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(ctx.Device, outBuf, outMem, 0);
    return vkMapMemory(ctx.Device, outMem, 0, bytes, 0, &outMapped) == VK_SUCCESS;
}

bool CreatePipeline(VkDevice device, VkRenderPass pass, VkExtent2D extent,
                    VkDescriptorSetLayout setLayout,
                    const char* shaderDir,
                    VkPipelineLayout& outLayout, VkPipeline& outPipe) {
    const std::string vertPath = std::string(shaderDir) + "/gbuffer_preview_vert.spv";
    const std::string fragPath = std::string(shaderDir) + "/gbuffer_preview_frag.spv";
    VkShaderModule vert = LoadModule(device, vertPath.c_str());
    VkShaderModule frag = LoadModule(device, fragPath.c_str());
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(device, vert, nullptr);
        if (frag) vkDestroyShaderModule(device, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{ 0, 0, static_cast<float>(extent.width),
                               static_cast<float>(extent.height), 0.0f, 1.0f };
    VkRect2D   scissor { { 0, 0 }, extent };
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    att.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments    = &att;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(PreviewPushConstants);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &outLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pColorBlendState    = &cb;
    gpci.layout              = outLayout;
    gpci.renderPass          = pass;
    gpci.subpass             = 0;

    const VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci,
                                                 nullptr, &outPipe);
    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
    return r == VK_SUCCESS;
}

bool CreateDescriptorSets(VkDevice device, GBufferPreviewPass& pp,
                          const OffscreenTargets& targets,
                          const SkyAtmosphere& sky,
                          VkSampler sdfFallbackSampler,
                          VkImageView sdfFallbackView) {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 12 * VulkanContext::kFramesInFlight; // 11 + Specular
    sizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 1  * VulkanContext::kFramesInFlight;
    sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[2].descriptorCount = 2  * VulkanContext::kFramesInFlight; // sparse idx + pool

    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = VulkanContext::kFramesInFlight;
    pci.poolSizeCount = 3;
    pci.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &pp.DescriptorPool) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetLayout layouts[VulkanContext::kFramesInFlight] = {};
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        layouts[i] = pp.SetLayout;
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = pp.DescriptorPool;
    ai.descriptorSetCount = VulkanContext::kFramesInFlight;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device, &ai, pp.FrameSets.data()) != VK_SUCCESS) {
        return false;
    }

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        const OffscreenFrame& f = targets.Frames[i];

        VkDescriptorImageInfo infos[12]{};
        // GBuffer (0..5)
        const VkImageView views[6] = {
            f.Albedo.View,
            f.Normal.View,
            f.RoughMetalF0.View,
            f.Emissive.View,
            f.Depth.View,
            f.Identity.View,
        };
        for (uint32_t b = 0; b < 6; ++b) {
            infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[b].imageView   = views[b];
            // Identity must use nearest, depth too (no filtering across pixels).
            infos[b].sampler     = (b == 1 || b == 5) ? targets.SamplerNearest
                                                      : targets.SamplerLinear;
        }
        // IBL bindings (6..8)
        infos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[6].imageView   = sky.Irradiance.View;
        infos[6].sampler     = sky.SamplerCubeLinear;
        infos[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[7].imageView   = sky.Prefilter.View;
        infos[7].sampler     = sky.SamplerCubeLinear;
        infos[8].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[8].imageView   = sky.LutView;
        infos[8].sampler     = sky.SamplerLut;
        // Phase 10: LightHDR sampled image. Layout matches the post-dispatch
        // barrier in LightingPassRecord (SHADER_READ_ONLY_OPTIMAL).
        infos[9].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[9].imageView   = f.LightHDR.View;
        infos[9].sampler     = targets.SamplerLinear;
        // Phase 12: SDF 3D image. Bound to the GlobalSDF dummy 1^3 at init
        // time; GBufferPreviewSetSDF later swaps to the resident view.
        infos[10].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[10].imageView   = sdfFallbackView;
        infos[10].sampler     = sdfFallbackSampler;
        // KHR specular SRV (binding 12, written below).
        infos[11].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[11].imageView   = f.Specular.View;
        infos[11].sampler     = targets.SamplerLinear;

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = pp.SdfUboBuffers[i];
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(SdfPushUbo);

        // Phase 15f — bind the dummy SSBO at 13/14 initially. Real residency
        // wires in via GBufferPreviewSetSparseSDF after the first sparse bake.
        VkDescriptorBufferInfo sparseInfos[2]{};
        sparseInfos[0].buffer = pp.DummySparseBuffer;
        sparseInfos[0].offset = 0;
        sparseInfos[0].range  = VK_WHOLE_SIZE;
        sparseInfos[1].buffer = pp.DummySparseBuffer;
        sparseInfos[1].offset = 0;
        sparseInfos[1].range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writes[15]{};
        for (uint32_t b = 0; b < 11; ++b) {
            writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet          = pp.FrameSets[i];
            writes[b].dstBinding      = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo      = &infos[b];
        }
        writes[11].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[11].dstSet          = pp.FrameSets[i];
        writes[11].dstBinding      = 11;
        writes[11].descriptorCount = 1;
        writes[11].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[11].pBufferInfo     = &uboInfo;
        writes[12].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[12].dstSet          = pp.FrameSets[i];
        writes[12].dstBinding      = 12;
        writes[12].descriptorCount = 1;
        writes[12].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[12].pImageInfo      = &infos[11];
        writes[13].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[13].dstSet          = pp.FrameSets[i];
        writes[13].dstBinding      = 13;
        writes[13].descriptorCount = 1;
        writes[13].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[13].pBufferInfo     = &sparseInfos[0];
        writes[14].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[14].dstSet          = pp.FrameSets[i];
        writes[14].dstBinding      = 14;
        writes[14].descriptorCount = 1;
        writes[14].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[14].pBufferInfo     = &sparseInfos[1];
        vkUpdateDescriptorSets(device, 15, writes, 0, nullptr);
    }
    return true;
}

// Phase 15f — 16-byte device-local SSBO that backs bindings 13/14 when no
// sparse residency exists. Contents are irrelevant (the slice-source flag
// gates reads), but the buffer must be a real VkBuffer for descriptor-write
// validation to pass.
bool CreateDummySparseBuffer(const VulkanContext& ctx, GBufferPreviewPass& pp) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = 16;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &pp.DummySparseBuffer) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, pp.DummySparseBuffer, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &pp.DummySparseMemory) != VK_SUCCESS) {
        return false;
    }
    return vkBindBufferMemory(ctx.Device, pp.DummySparseBuffer,
                              pp.DummySparseMemory, 0) == VK_SUCCESS;
}

bool CreateFramebuffers(VkDevice device, VkRenderPass pass,
                        const std::vector<VkImageView>& views,
                        VkExtent2D extent,
                        std::vector<VkFramebuffer>& outFbs) {
    outFbs.resize(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
        VkFramebufferCreateInfo fbci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbci.renderPass      = pass;
        fbci.attachmentCount = 1;
        fbci.pAttachments    = &views[i];
        fbci.width           = extent.width;
        fbci.height          = extent.height;
        fbci.layers          = 1;
        if (vkCreateFramebuffer(device, &fbci, nullptr, &outFbs[i]) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

} // namespace

bool GBufferPreviewInitialize(GBufferPreviewPass& pp, const VulkanContext& ctx,
                              const OffscreenTargets& targets,
                              const SkyAtmosphere& sky,
                              VkSampler sdfFallbackSampler,
                              VkImageView sdfFallbackView,
                              const char* shaderArtifactsDir) {
    if (pp.Initialized) return true;
    if (!CreateRenderPass(ctx.Device, ctx.SwapchainFormat, pp.RenderPass)) {
        RS_LOG_ERROR("GBufferPreview: render-pass failed");
        return false;
    }
    if (!CreateSetLayout(ctx.Device, pp.SetLayout)) {
        RS_LOG_ERROR("GBufferPreview: descriptor layout failed");
        return false;
    }
    if (!CreatePipeline(ctx.Device, pp.RenderPass, ctx.SwapchainExtent,
                        pp.SetLayout, shaderArtifactsDir,
                        pp.PipelineLayout, pp.Pipeline)) {
        RS_LOG_ERROR("GBufferPreview: pipeline failed");
        return false;
    }
    // Per-frame SDF UBOs must exist before descriptor sets so the buffer-info
    // pointers can be wired in CreateDescriptorSets.
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (!CreateUboBuffer(ctx, sizeof(SdfPushUbo),
                             pp.SdfUboBuffers[i], pp.SdfUboMemory[i],
                             pp.SdfUboMapped[i])) {
            RS_LOG_ERROR("GBufferPreview: SDF UBO alloc failed (slot %u)", i);
            return false;
        }
        // Seed with a benign default so an early frame before any update sees
        // a valid plane.
        SdfPushUbo seed{};
        seed.AABBMinAndPlaneY  = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        seed.AABBMaxAndMaxDist = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        seed.PreviewParams     = glm::vec4(0.0f);
        seed.SparseAABBMin     = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        seed.SparseAABBMax     = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f / 32767.0f);
        seed.SparseDims        = glm::uvec4(1u, 1u, 1u, 0u);
        std::memcpy(pp.SdfUboMapped[i], &seed, sizeof(seed));
    }
    // Phase 15f dummy SSBO must exist before descriptor writes reference it.
    if (!CreateDummySparseBuffer(ctx, pp)) {
        RS_LOG_ERROR("GBufferPreview: dummy sparse buffer alloc failed");
        return false;
    }
    if (!CreateDescriptorSets(ctx.Device, pp, targets, sky,
                              sdfFallbackSampler, sdfFallbackView)) {
        RS_LOG_ERROR("GBufferPreview: descriptor sets failed");
        return false;
    }
    if (!CreateFramebuffers(ctx.Device, pp.RenderPass, ctx.SwapchainViews,
                            ctx.SwapchainExtent, pp.Framebuffers)) {
        RS_LOG_ERROR("GBufferPreview: framebuffers failed");
        return false;
    }
    pp.Initialized = true;
    RS_LOG_INFO("GBufferPreview ready: %zu framebuffers (+SDF binding)",
                pp.Framebuffers.size());
    return true;
}

void GBufferPreviewSetSDF(GBufferPreviewPass& pp, const VulkanContext& ctx,
                          VkImageView view, VkSampler sampler) {
    if (!pp.Initialized || view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    // Caller is responsible for ensuring no frame is in flight that still
    // references the old binding (typically vkDeviceWaitIdle before swapping).
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorImageInfo info{};
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.imageView   = view;
        info.sampler     = sampler;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = pp.FrameSets[i];
        w.dstBinding      = 10;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &info;
        vkUpdateDescriptorSets(ctx.Device, 1, &w, 0, nullptr);
    }
}

void GBufferPreviewSetSparseSDF(GBufferPreviewPass& pp, const VulkanContext& ctx,
                                VkBuffer indexBuffer, VkDeviceSize indexBytes,
                                VkBuffer poolBuffer,  VkDeviceSize poolBytes) {
    if (!pp.Initialized) return;
    // Null buffers → revert to the dummy. Real residency wires in via the
    // residency-changed branch in Main.cpp (already gated on vkDeviceWaitIdle
    // via SdfBakerPanelDrawAndPump's upload-completion path).
    const VkBuffer     idx   = indexBuffer ? indexBuffer : pp.DummySparseBuffer;
    const VkDeviceSize iSize = indexBuffer ? indexBytes  : VK_WHOLE_SIZE;
    const VkBuffer     pool  = poolBuffer  ? poolBuffer  : pp.DummySparseBuffer;
    const VkDeviceSize pSize = poolBuffer  ? poolBytes   : VK_WHOLE_SIZE;
    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        VkDescriptorBufferInfo infos[2]{};
        infos[0].buffer = idx;
        infos[0].offset = 0;
        infos[0].range  = iSize;
        infos[1].buffer = pool;
        infos[1].offset = 0;
        infos[1].range  = pSize;
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = pp.FrameSets[i];
        writes[0].dstBinding      = 13;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo     = &infos[0];
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = pp.FrameSets[i];
        writes[1].dstBinding      = 14;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &infos[1];
        vkUpdateDescriptorSets(ctx.Device, 2, writes, 0, nullptr);
    }
}

void GBufferPreviewTerminate(GBufferPreviewPass& pp, const VulkanContext& ctx) {
    if (!pp.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (VkFramebuffer fb : pp.Framebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.Device, fb, nullptr);
    }
    pp.Framebuffers.clear();

    for (uint32_t i = 0; i < VulkanContext::kFramesInFlight; ++i) {
        if (pp.SdfUboMemory[i]) {
            vkUnmapMemory(ctx.Device, pp.SdfUboMemory[i]);
            vkFreeMemory (ctx.Device, pp.SdfUboMemory[i], nullptr);
            pp.SdfUboMemory[i] = VK_NULL_HANDLE;
        }
        if (pp.SdfUboBuffers[i]) {
            vkDestroyBuffer(ctx.Device, pp.SdfUboBuffers[i], nullptr);
            pp.SdfUboBuffers[i] = VK_NULL_HANDLE;
        }
        pp.SdfUboMapped[i] = nullptr;
    }
    if (pp.DummySparseBuffer) {
        vkDestroyBuffer(ctx.Device, pp.DummySparseBuffer, nullptr);
        pp.DummySparseBuffer = VK_NULL_HANDLE;
    }
    if (pp.DummySparseMemory) {
        vkFreeMemory(ctx.Device, pp.DummySparseMemory, nullptr);
        pp.DummySparseMemory = VK_NULL_HANDLE;
    }

    if (pp.DescriptorPool) vkDestroyDescriptorPool     (ctx.Device, pp.DescriptorPool, nullptr);
    if (pp.Pipeline)       vkDestroyPipeline           (ctx.Device, pp.Pipeline,       nullptr);
    if (pp.PipelineLayout) vkDestroyPipelineLayout     (ctx.Device, pp.PipelineLayout, nullptr);
    if (pp.SetLayout)      vkDestroyDescriptorSetLayout(ctx.Device, pp.SetLayout,      nullptr);
    if (pp.RenderPass)     vkDestroyRenderPass         (ctx.Device, pp.RenderPass,     nullptr);
    pp.DescriptorPool = VK_NULL_HANDLE;
    pp.Pipeline       = VK_NULL_HANDLE;
    pp.PipelineLayout = VK_NULL_HANDLE;
    pp.SetLayout      = VK_NULL_HANDLE;
    pp.RenderPass     = VK_NULL_HANDLE;
    pp.FrameSets      = {};
    pp.Initialized    = false;
}

void GBufferPreviewRecord(GBufferPreviewPass& pp, const VulkanContext& ctx,
                          VkCommandBuffer cmd, uint32_t imageIndex,
                          uint32_t frameSlot,
                          const CameraView& camera,
                          const GBufferPreviewSettings& settings,
                          const SDFSliceState& sdfSlice) {
    // Update this frame's SDF UBO before recording.
    if (pp.SdfUboMapped[frameSlot]) {
        // bsLog2 helper — same convention used by the sparse helper macros.
        auto log2Ceil = [](uint32_t v) {
            uint32_t r = 0; while ((1u << r) < v) ++r; return r;
        };
        const uint32_t bs     = sdfSlice.SparseBrickSz ? sdfSlice.SparseBrickSz : 1u;
        const float    decode = sdfSlice.SparseMaxDist > 0.0f
                                    ? (sdfSlice.SparseMaxDist / 32767.0f) : 0.0f;
        SdfPushUbo u{};
        u.AABBMinAndPlaneY  = glm::vec4(sdfSlice.AABBMin, sdfSlice.PlaneY);
        u.AABBMaxAndMaxDist = glm::vec4(sdfSlice.AABBMax, sdfSlice.MaxDist);
        u.PreviewParams     = glm::vec4(static_cast<float>(sdfSlice.VizMode),
                                        static_cast<float>(static_cast<int>(sdfSlice.Source)),
                                        0.0f, 0.0f);
        u.SparseAABBMin     = glm::vec4(sdfSlice.SparseAABBMin, sdfSlice.SparseMaxDist);
        u.SparseAABBMax     = glm::vec4(sdfSlice.SparseAABBMax, decode);
        u.SparseDims        = glm::uvec4(sdfSlice.SparseRes,
                                         bs,
                                         sdfSlice.SparseBrickGrid,
                                         log2Ceil(bs));
        std::memcpy(pp.SdfUboMapped[frameSlot], &u, sizeof(u));
    }
    VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass        = pp.RenderPass;
    rpbi.framebuffer       = pp.Framebuffers[imageIndex];
    rpbi.renderArea.offset = { 0, 0 };
    rpbi.renderArea.extent = ctx.SwapchainExtent;
    rpbi.clearValueCount   = 0;
    rpbi.pClearValues      = nullptr;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pp.Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pp.PipelineLayout,
                            0, 1, &pp.FrameSets[frameSlot], 0, nullptr);

    PreviewPushConstants pc{};
    pc.SunDirAndIntensity = glm::vec4(glm::normalize(settings.SunDirection),
                                      settings.SunIntensity);
    pc.AmbientAndNearFar  = glm::vec4(settings.Ambient,
                                      camera.NearClip, camera.FarClip,
                                      static_cast<float>(static_cast<int>(settings.Mode)));

    const MatcapSettings& m = settings.Matcap;
    pc.MatcapTop       = glm::vec4(m.ColorTop,        m.GradientCurve);
    pc.MatcapBottom    = glm::vec4(m.ColorBottom,     m.Exposure);
    pc.MatcapRim       = glm::vec4(m.RimColor,        m.RimPower);
    pc.MatcapHighlight = glm::vec4(m.HighlightColor,  m.HighlightSize);
    pc.MatcapParams    = glm::vec4(m.RimStrength, m.HighlightStrength,
                                   m.HighlightOffset.x, m.HighlightOffset.y);

    // Pack the upper-left 3×3 of the view matrix row-by-row so the shader can
    // rotate the world-space N into view-space without uploading a full mat4.
    const glm::mat3 viewRot(camera.View);
    pc.MatcapTintRow0 = glm::vec4(viewRot[0][0], viewRot[1][0], viewRot[2][0], m.Tint.x);
    pc.MatcapTintRow1 = glm::vec4(viewRot[0][1], viewRot[1][1], viewRot[2][1], m.Tint.y);
    pc.MatcapTintRow2 = glm::vec4(viewRot[0][2], viewRot[1][2], viewRot[2][2], m.Tint.z);
    pc.MatcapMaterial = glm::vec4(m.Roughness, m.Metallic, 0.0f, 0.0f);

    pc.IblParams = glm::vec4(settings.UseIBL ? 1.0f : 0.0f,
                             settings.IBLIntensity,
                             static_cast<float>(SkyAtmosphere::kPrefilterMips),
                             0.0f);

    pc.InvViewProj = glm::inverse(camera.Projection * camera.View);

    vkCmdPushConstants(cmd, pp.PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

} // namespace RS
