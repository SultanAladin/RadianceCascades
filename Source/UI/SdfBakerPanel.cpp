// Source/UI/SdfBakerPanel.cpp — Phase 12 (+ Phase 12.5 viz / output / stats)
#include "UI/SdfBakerPanel.h"
#include "Scene/SceneInternal.h"
#include "Core/Logger.h"

#include "backends/imgui_impl_vulkan.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <direct.h>
#include <sys/stat.h>

namespace RS {

namespace {

constexpr int kBrickSizes[3] = { 4, 8, 16 };
const char* kBrickSizeLabels[3] = { "4^3 (fine)", "8^3 (NanoVDB default)", "16^3 (coarse)" };

constexpr int kResolutions[7] = { 32, 64, 96, 128, 256, 512, 1024 };
const char* kResolutionLabels[7] = { "32^3", "64^3", "96^3", "128^3",
                                     "256^3 (heavy)", "512^3 (very heavy)", "1024^3 (extreme)" };
// VRAM cost at R16_SNORM: 32^3=64KB, 64^3=512KB, 128^3=4MB, 256^3=32MB,
// 512^3=256MB, 1024^3=2GB. CPU bake time scales with res^3 × triangle count.
const char* kAlgorithmLabels[4] = {
    "Exact (BVH brute-force)",
    "JFA (disabled in v1)",
    "Hybrid JFA + narrow-band (disabled in v1)",
    "Fast Sweeping Method (CPU)",
};
const char* kVizModeLabels[5] = {
    "Signed RGB (default)",
    "Heatmap",
    "Grayscale",
    "Signed B/W",
    "Gradient magnitude",
};

// Lightweight helpers that don't need vk state ---------------------------------

std::string DeriveBaseName(const char* sourcePath) {
    std::string s(sourcePath ? sourcePath : "");
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) s = s.substr(slash + 1);
    const size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
    if (s.empty()) s = "mesh";
    return s;
}

bool EnsureDirRecursive(const std::string& dir) {
    if (dir.empty()) return true;
    struct _stat st{};
    if (_stat(dir.c_str(), &st) == 0) return (st.st_mode & _S_IFDIR) != 0;
    // Walk parents.
    const size_t slash = dir.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::string parent = dir.substr(0, slash);
        if (!parent.empty()) EnsureDirRecursive(parent);
    }
    return _mkdir(dir.c_str()) == 0 || errno == EEXIST;
}

uint64_t QueryFileSize(const char* path) {
    struct _stat64 st{};
    if (_stat64(path, &st) != 0) return 0;
    if ((st.st_mode & _S_IFREG) == 0) return 0;
    return static_cast<uint64_t>(st.st_size);
}

void FormatBytes(uint64_t bytes, char* outBuf, size_t outSize) {
    constexpr double kKB = 1024.0;
    constexpr double kMB = 1024.0 * 1024.0;
    constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
    const double b = static_cast<double>(bytes);
    if      (b >= kGB) std::snprintf(outBuf, outSize, "%.2f GB", b / kGB);
    else if (b >= kMB) std::snprintf(outBuf, outSize, "%.2f MB", b / kMB);
    else if (b >= kKB) std::snprintf(outBuf, outSize, "%.2f KB", b / kKB);
    else               std::snprintf(outBuf, outSize, "%llu B",
                                     static_cast<unsigned long long>(bytes));
}

void SeedOutputBuffersFromMesh(SdfBakerState& s) {
    if (s.OutputDirBuf[0] == 0) {
        std::snprintf(s.OutputDirBuf.data(), s.OutputDirBuf.size(), "Cache/SDF");
    }
    if (s.OutputNameBuf[0] == 0 && s.TargetSourcePath) {
        const std::string base = DeriveBaseName(s.TargetSourcePath);
        std::snprintf(s.OutputNameBuf.data(), s.OutputNameBuf.size(), "%s",
                      base.c_str());
    }
}

// Compose the full output file path:  <dir>/<name>_<res>.rsdf
std::string ComposeOutputPath(const SdfBakerState& s, uint32_t resolution) {
    const char* dir  = s.OutputDirBuf[0]  ? s.OutputDirBuf.data()  : "Cache/SDF";
    const char* name = s.OutputNameBuf[0] ? s.OutputNameBuf.data() : "mesh";
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%u.rsdf", resolution);
    return std::string(dir) + "/" + name + tail;
}

std::string ComposeSparseOutputPath(const SdfBakerState& s, uint32_t resolution,
                                    uint32_t brickSize) {
    const char* dir  = s.OutputDirBuf[0]  ? s.OutputDirBuf.data()  : "Cache/SDF";
    const char* name = s.OutputNameBuf[0] ? s.OutputNameBuf.data() : "mesh";
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%u_b%u.rsdfvdb", resolution, brickSize);
    return std::string(dir) + "/" + name + tail;
}

uint32_t BrickSizePx(int brickSizeChoice) {
    const int idx = std::clamp(brickSizeChoice, 0,
                               static_cast<int>(IM_ARRAYSIZE(kBrickSizes)) - 1);
    return static_cast<uint32_t>(kBrickSizes[idx]);
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

bool CreatePreviewImage(const VulkanContext& ctx, uint32_t size,
                        VkImage& outImage, VkDeviceMemory& outMemory,
                        VkImageView& outView) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent        = { size, size, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &outImage) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx.Device, outImage, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &outMemory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.Device, outImage, outMemory, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image            = outImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(ctx.Device, &vci, nullptr, &outView) == VK_SUCCESS;
}

bool CreateHostBuffer(const VulkanContext& ctx, VkDeviceSize bytes,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bytes;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
    return true;
}

// Upload RGBA8 pixels into the preview image via a one-shot cmd buffer.
void UploadPreviewPixels(const VulkanContext& ctx, VkImage dst, uint32_t size,
                         const uint32_t* pixels) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(size) * size * 4;
    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    if (!CreateHostBuffer(ctx, bytes, staging, stagingMem)) return;
    void* mapped = nullptr;
    vkMapMemory(ctx.Device, stagingMem, 0, bytes, 0, &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.Device, stagingMem);

    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool        = ctx.CommandPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.Device, &cai, &cmd);

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image               = dst;
    toDst.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toDst.srcAccessMask       = 0;
    toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toDst);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { size, size, 1 };
    vkCmdCopyBufferToImage(cmd, staging, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toRead = toDst;
    toRead.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toRead);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.GraphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.GraphicsQueue);
    vkFreeCommandBuffers(ctx.Device, ctx.CommandPool, 1, &cmd);
    vkDestroyBuffer(ctx.Device, staging, nullptr);
    vkFreeMemory   (ctx.Device, stagingMem, nullptr);
}

// CPU sample of the BakedSDF at (x, y, z) in [0..res-1]. No interpolation —
// the preview pane shows raw voxels so quantisation is visible.
float SDFSampleCpu(const BakedSDF& b, int x, int y, int z) {
    const int r = static_cast<int>(b.Resolution);
    x = std::clamp(x, 0, r - 1);
    y = std::clamp(y, 0, r - 1);
    z = std::clamp(z, 0, r - 1);
    const size_t idx = (static_cast<size_t>(z) * r + y) * r + x;
    return static_cast<float>(b.Voxels[idx]) / 32767.0f * b.MaxDist;
}

// Central-difference gradient on the voxel grid (world-units / world-units).
// Used by the GradientMag viz; mirrors the GPU branch in gbuffer_preview.frag.
glm::vec3 SDFGradientCpu(const BakedSDF& b, int x, int y, int z) {
    const float dx = SDFSampleCpu(b, x + 1, y, z) - SDFSampleCpu(b, x - 1, y, z);
    const float dy = SDFSampleCpu(b, x, y + 1, z) - SDFSampleCpu(b, x, y - 1, z);
    const float dz = SDFSampleCpu(b, x, y, z + 1) - SDFSampleCpu(b, x, y, z - 1);
    const glm::vec3 size = b.AABBMax - b.AABBMin;
    const float r = static_cast<float>(std::max(1u, b.Resolution));
    const glm::vec3 cell = (2.0f / r) * size;
    return glm::vec3(dx / std::max(cell.x, 1e-6f),
                     dy / std::max(cell.y, 1e-6f),
                     dz / std::max(cell.z, 1e-6f));
}

glm::vec3 ColourSignedRGB(float vis) {
    glm::vec3 col;
    if (vis < 0.0f) col = glm::mix(glm::vec3(0.05f, 0.4f, 0.05f),
                                   glm::vec3(0.9f, 0.05f, 0.05f), -vis);
    else            col = glm::mix(glm::vec3(0.05f, 0.4f, 0.05f),
                                   glm::vec3(0.05f, 0.15f, 0.9f),  vis);
    const float iso = std::exp(-160.0f * vis * vis);
    return glm::mix(col, glm::vec3(0.2f, 1.0f, 0.4f), iso);
}

glm::vec3 ColourHeatmap(float vis) {
    const float t = std::clamp(std::fabs(vis), 0.0f, 1.0f);
    const glm::vec3 c0(0.00f, 0.00f, 0.50f);
    const glm::vec3 c1(0.00f, 0.80f, 1.00f);
    const glm::vec3 c2(0.00f, 1.00f, 0.30f);
    const glm::vec3 c3(1.00f, 0.95f, 0.10f);
    const glm::vec3 c4(0.95f, 0.10f, 0.05f);
    auto ss = [](float a, float b, float x) {
        const float t = std::clamp((x - a) / std::max(b - a, 1e-6f), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    glm::vec3 col = glm::mix(c0, c1, ss(0.00f, 0.25f, t));
    col = glm::mix(col, c2, ss(0.25f, 0.50f, t));
    col = glm::mix(col, c3, ss(0.50f, 0.75f, t));
    col = glm::mix(col, c4, ss(0.75f, 1.00f, t));
    if (vis < 0.0f) col *= 0.55f;
    return col;
}

glm::vec3 ColourGrayscale(float vis) {
    const float t = std::clamp(std::fabs(vis), 0.0f, 1.0f);
    return glm::vec3(t);
}

glm::vec3 ColourSignedBW(float vis) {
    glm::vec3 col = (vis < 0.0f) ? glm::vec3(0.02f) : glm::vec3(0.95f);
    const float iso = std::exp(-220.0f * vis * vis);
    return glm::mix(col, glm::vec3(0.95f, 0.80f, 0.20f), iso);
}

glm::vec3 ColourGradientMag(float gmag) {
    const float dev = std::clamp(gmag, 0.0f, 2.0f);
    if (dev <= 1.0f) return glm::mix(glm::vec3(0.05f, 0.10f, 0.55f),
                                     glm::vec3(0.10f, 0.85f, 0.20f), dev);
    return glm::mix(glm::vec3(0.10f, 0.85f, 0.20f),
                    glm::vec3(0.90f, 0.15f, 0.05f), dev - 1.0f);
}

// Bilinear lookup on a constant-Y slice for the preview pane. Sample world-Y
// at planeY01 ∈ [0..1] across the SDF box.
uint32_t SDFSliceColour(const BakedSDF& b, float u, float v, float planeY01,
                        int vizMode) {
    const int r = static_cast<int>(b.Resolution);
    const float fz = std::clamp(planeY01, 0.0f, 1.0f) * static_cast<float>(r - 1);
    const float fx = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(r - 1);
    const float fy = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(r - 1);
    // Match the world axis mapping used in the GPU shader: SDF's Y is
    // world-Y, U → world-X, V → world-Z. CPU indexes (x, y, z) in voxel
    // space; the slice's V scrolls Z.
    const int xi = static_cast<int>(std::floor(fx));
    const int yi = static_cast<int>(std::floor(fz));   // world-Y (the slice plane)
    const int zi = static_cast<int>(std::floor(fy));   // world-Z (preview vertical)

    const float d = SDFSampleCpu(b, xi, yi, zi);
    const glm::vec3 size = b.AABBMax - b.AABBMin;
    const float scale = 0.5f * std::max(size.x, std::max(size.y, size.z));
    const float vis = std::clamp(d / std::max(scale, 1e-4f), -1.0f, 1.0f);

    glm::vec3 col;
    switch (vizMode) {
        case 1: col = ColourHeatmap(vis);   break;
        case 2: col = ColourGrayscale(vis); break;
        case 3: col = ColourSignedBW(vis);  break;
        case 4: {
            const glm::vec3 g = SDFGradientCpu(b, xi, yi, zi);
            col = ColourGradientMag(glm::length(g));
            break;
        }
        default: col = ColourSignedRGB(vis); break;
    }

    auto pack = [](float c) {
        const float v = std::clamp(c, 0.0f, 1.0f);
        return static_cast<uint32_t>(v * 255.0f + 0.5f);
    };
    return (0xFFu << 24) | (pack(col.b) << 16) | (pack(col.g) << 8) | pack(col.r);
}

void RegenerateSlice(const VulkanContext& ctx, SdfBakerState& s) {
    if (!s.ResidentBakedCopy) return;
    const uint32_t sz = s.SliceResolution;
    s.SliceRGBA.assign(sz * sz, 0xFF202020u);
    const int vizMode = std::clamp(s.VizModeChoice, 0, 4);
    for (uint32_t y = 0; y < sz; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(sz - 1);
        for (uint32_t x = 0; x < sz; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(sz - 1);
            s.SliceRGBA[y * sz + x] = SDFSliceColour(*s.ResidentBakedCopy, u, v,
                                                    s.SlicePlaneY01, vizMode);
        }
    }
    if (s.SlicePreviewImage) {
        UploadPreviewPixels(ctx, s.SlicePreviewImage, sz, s.SliceRGBA.data());
    }
    s.SliceDirty = false;
}

void EnsurePreviewTexture(const VulkanContext& ctx, SdfBakerState& s) {
    if (s.SlicePreviewImage != VK_NULL_HANDLE) return;
    if (!CreatePreviewImage(ctx, s.SliceResolution,
                            s.SlicePreviewImage, s.SlicePreviewMemory,
                            s.SlicePreviewView)) {
        RS_LOG_ERROR("SdfBaker: preview image creation failed");
        return;
    }
    // Upload an initial grey so the ImGui::Image draws something on the first
    // frame even before a bake lands.
    std::vector<uint32_t> grey(s.SliceResolution * s.SliceResolution, 0xFF202020u);
    UploadPreviewPixels(ctx, s.SlicePreviewImage, s.SliceResolution, grey.data());

    VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(
        s.SlicePreviewView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    s.SlicePreviewTex = reinterpret_cast<ImTextureID>(ds);
}

void KickBake(SdfBakerState& s, const VulkanContext& ctx,
              const Scene& scene, MeshHandle mesh, uint32_t resolution, int algorithmChoice) {
    if (s.BakeInFlight) return;
    const GpuMesh* gpuMesh = SceneMeshes(scene).Get(mesh);
    if (!gpuMesh) return;

    s.BakeInFlight = true;
    s.Progress     = std::make_shared<BakeProgress>();
    s.PendingUpload= std::make_shared<BakedSDF>();
    s.WorkerDone.store(false, std::memory_order_release);

    // Worker captures by value where possible. The VulkanContext is read-only
    // (we only call vkMapMemory inside the worker on host-visible memory,
    // which is safe — vkMapMemory + vkUnmapMemory don't require external
    // synchronisation when the memory is HOST_VISIBLE | HOST_COHERENT).
    std::shared_ptr<BakeProgress> progress = s.Progress;
    std::shared_ptr<BakedSDF>     pending  = s.PendingUpload;
    std::atomic<bool>*            done     = &s.WorkerDone;
    const VulkanContext* ctxPtr = &ctx;
    const GpuMesh* meshPtr = gpuMesh;

    s.Worker = std::thread([progress, pending, done, ctxPtr, meshPtr, resolution, algorithmChoice]() {
        *pending = MeshSDFBakerBake(*ctxPtr, *meshPtr, resolution, progress.get(), algorithmChoice);
        done->store(true, std::memory_order_release);
    });
}

void RefreshStatsFromBaked(SdfBakerState& s, const BakedSDF& baked,
                           bool fromCache, double durationMs, uint32_t triCount,
                           const std::string& diskPath) {
    s.Stats.Valid          = true;
    s.Stats.FromCache      = fromCache;
    s.Stats.Resolution     = baked.Resolution;
    s.Stats.VoxelCount     = static_cast<uint64_t>(baked.Voxels.size());
    s.Stats.CpuBytes       = s.Stats.VoxelCount * sizeof(int16_t);
    s.Stats.GpuBytes       = s.Stats.VoxelCount * sizeof(int16_t);  // R16_SNORM
    s.Stats.DiskBytes      = diskPath.empty() ? 0u : QueryFileSize(diskPath.c_str());
    s.Stats.BakeDurationMs = durationMs;
    s.Stats.TriangleCount  = triCount;
    s.Stats.MaxDist        = baked.MaxDist;
    s.Stats.LastSavedPath  = diskPath;
}

void FinaliseBake(SdfBakerState& s, const VulkanContext& ctx, GlobalSDF& globalSdf,
                  bool& outResidencyChanged) {
    if (!s.WorkerDone.load(std::memory_order_acquire)) return;
    if (s.Worker.joinable()) s.Worker.join();

    s.BakeInFlight = false;
    s.WorkerDone.store(false, std::memory_order_release);

    if (!s.PendingUpload || s.PendingUpload->Voxels.empty()) {
        RS_LOG_INFO("SdfBaker: bake produced no voxels (cancelled or empty)");
        s.PendingUpload.reset();
        s.Progress.reset();
        return;
    }

    const double durationMs = s.Progress
        ? s.Progress->DurationMs.load(std::memory_order_relaxed) : 0.0;
    const uint32_t triCount = s.Progress
        ? s.Progress->TriangleCount.load(std::memory_order_relaxed) : 0u;

    const ResidentSDF r = GlobalSDFUploadBaked(globalSdf, ctx, s.TargetMesh,
                                               *s.PendingUpload, /*fromCache*/ false);
    std::string savedPath;
    if (r.Image == VK_NULL_HANDLE) {
        RS_LOG_ERROR("SdfBaker: upload failed");
    } else {
        outResidencyChanged = true;
        s.ResidentBakedCopy = s.PendingUpload;
        s.SliceDirty = true;
        // Persist to disk using the panel's editable output path.
        SeedOutputBuffersFromMesh(s);
        const std::string outPath = ComposeOutputPath(s, s.PendingUpload->Resolution);
        EnsureDirRecursive(s.OutputDirBuf.data());
        if (SDFCacheSave(outPath.c_str(), *s.PendingUpload)) {
            savedPath = outPath;
        }
    }

    RefreshStatsFromBaked(s, *s.PendingUpload, /*fromCache*/ false,
                          durationMs, triCount, savedPath);

    // Sparse VDB-style follow-on. Build from the dense voxels in memory (no
    // re-walking the BVH), save to .rsdfvdb, and upload to the sparse residency
    // table alongside the dense one.
    if (s.UseSparse && r.Image != VK_NULL_HANDLE) {
        const uint32_t brickSize = BrickSizePx(s.BrickSizeChoice);
        BakedSparseSDF sparse = SDFCacheBuildSparse(*s.PendingUpload, brickSize);
        if (sparse.Resolution != 0) {
            const std::string sparsePath = ComposeSparseOutputPath(s,
                sparse.Resolution, sparse.BrickSize);
            EnsureDirRecursive(s.OutputDirBuf.data());
            const bool sparseSaved = SDFCacheSaveSparse(sparsePath.c_str(), sparse);
            const ResidentSparseSDF rs = GlobalSDFUploadSparse(globalSdf, ctx,
                                                               s.TargetMesh, sparse,
                                                               /*fromCache*/ false);
            s.Stats.SparseValid           = (rs.IndexBuffer != VK_NULL_HANDLE);
            s.Stats.SparseBrickSize       = sparse.BrickSize;
            s.Stats.SparseOccupiedBricks  = sparse.OccupiedBrickCount;
            s.Stats.SparseTotalBricks     = static_cast<uint32_t>(sparse.BrickIndex.size());
            s.Stats.SparseGpuBytes        = rs.IndexBytes + rs.PoolBytes;
            s.Stats.SparseDiskBytes       = sparseSaved
                ? QueryFileSize(sparsePath.c_str()) : 0u;
            s.Stats.SparseLastSavedPath   = sparseSaved ? sparsePath : std::string{};
        }
    }

    s.PendingUpload.reset();
    s.Progress.reset();
}

} // namespace

uint32_t SdfBakerResolutionPx(const SdfBakerState& s) {
    const int idx = std::clamp(s.ResolutionChoice, 0,
                               static_cast<int>(IM_ARRAYSIZE(kResolutions)) - 1);
    return static_cast<uint32_t>(kResolutions[idx]);
}

bool SdfBakerInitialize(SdfBakerState& s, const VulkanContext& ctx) {
    EnsurePreviewTexture(ctx, s);
    return true;
}

void SdfBakerTerminate(SdfBakerState& s, const VulkanContext& ctx) {
    if (s.Worker.joinable()) {
        if (s.Progress) s.Progress->Cancel.store(true, std::memory_order_release);
        s.Worker.join();
    }
    vkDeviceWaitIdle(ctx.Device);
    if (s.SlicePreviewTex) {
        ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(s.SlicePreviewTex));
        s.SlicePreviewTex = 0;
    }
    if (s.SlicePreviewView)   vkDestroyImageView(ctx.Device, s.SlicePreviewView,   nullptr);
    if (s.SlicePreviewImage)  vkDestroyImage    (ctx.Device, s.SlicePreviewImage,  nullptr);
    if (s.SlicePreviewMemory) vkFreeMemory      (ctx.Device, s.SlicePreviewMemory, nullptr);
    s.SlicePreviewView   = VK_NULL_HANDLE;
    s.SlicePreviewImage  = VK_NULL_HANDLE;
    s.SlicePreviewMemory = VK_NULL_HANDLE;
    s.ResidentBakedCopy.reset();
    s.PendingUpload.reset();
    s.Progress.reset();
}

bool SdfBakerPanelDrawAndPump(SdfBakerState& s,
                              const VulkanContext& ctx,
                              GlobalSDF& globalSdf,
                              const Scene& scene) {
    bool residencyChanged = false;

    EnsurePreviewTexture(ctx, s);

    // Finalise any in-flight bake first so the panel reflects the new state.
    if (s.BakeInFlight) FinaliseBake(s, ctx, globalSdf, residencyChanged);

    if (s.SliceDirty && s.ResidentBakedCopy) RegenerateSlice(ctx, s);

    ImGui::SetNextWindowPos (ImVec2(640.0f, 360.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 520.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("SDF Baker")) {
        ImGui::End();
        return residencyChanged;
    }

    // Target & current residency state -----------------------------------
    ImGui::TextUnformatted("Target");
    ImGui::Separator();
    if (s.TargetMesh == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f),
                           "No mesh selected.");
    } else {
        ImGui::Text("Mesh handle: %u", s.TargetMesh);
        if (s.TargetSourcePath) ImGui::Text("Source: %s", s.TargetSourcePath);
        const ResidentSDF* resident = GlobalSDFGet(globalSdf, s.TargetMesh);
        if (resident) {
            ImGui::Text("Resident: %u^3 %s", resident->Resolution,
                        resident->FromCache ? "(from cache)" : "(fresh bake)");
            ImGui::Text("AABB: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)",
                        resident->AABBMin.x, resident->AABBMin.y, resident->AABBMin.z,
                        resident->AABBMax.x, resident->AABBMax.y, resident->AABBMax.z);
            ImGui::Text("MaxDist: %.3f", resident->MaxDist);
        } else {
            ImGui::TextDisabled("Not resident — click Bake.");
        }
    }

    // Algorithm + resolution ---------------------------------------------
    ImGui::Separator();
    ImGui::TextUnformatted("Settings");
    ImGui::Combo("Resolution", &s.ResolutionChoice, kResolutionLabels,
                 IM_ARRAYSIZE(kResolutionLabels));
    if (s.ResolutionChoice >= 4) {
        // 256^3 and above — surface bake cost so the user isn't surprised.
        const uint32_t r = SdfBakerResolutionPx(s);
        const uint64_t voxels = static_cast<uint64_t>(r) * r * r;
        const uint64_t vramMB = (voxels * 2u) / (1024u * 1024u);
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f),
                           "Warning: %u^3 = %llu voxels (%llu MB VRAM); bake may take minutes.",
                           r, static_cast<unsigned long long>(voxels),
                           static_cast<unsigned long long>(vramMB));
    }
    if (ImGui::BeginCombo("Algorithm", kAlgorithmLabels[s.AlgorithmChoice])) {
        for (int i = 0; i < IM_ARRAYSIZE(kAlgorithmLabels); ++i) {
            const bool enabled = (i == 0 || i == 3);
            ImGui::BeginDisabled(!enabled);
            if (ImGui::Selectable(kAlgorithmLabels[i], i == s.AlgorithmChoice)) {
                if (enabled) s.AlgorithmChoice = i;
            }
            ImGui::EndDisabled();
            if (!enabled && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Slated for Phase 16 polish. Exact BVH baker"
                                  " is the v1 default (UE5 Mesh DF style).");
            }
        }
        ImGui::EndCombo();
    }

    // Visualization ------------------------------------------------------
    // Drives both the slice preview (CPU rasterised) and the main viewport's
    // SDFSlice mode (GPU shader). One source of truth — flipping the combo
    // updates both this frame.
    if (ImGui::Combo("Visualization", &s.VizModeChoice, kVizModeLabels,
                     IM_ARRAYSIZE(kVizModeLabels))) {
        s.SliceDirty = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Signed RGB: red=inside, blue=outside, green iso.\n"
                          "Heatmap: |d| → blue→cyan→green→yellow→red.\n"
                          "Grayscale: |d| → grey ramp.\n"
                          "Signed B/W: inside black, outside white.\n"
                          "Gradient mag: |∇SDF| (should be ≈1 for clean bakes).");
    }

    // Sparse VDB-style storage -------------------------------------------
    ImGui::Separator();
    ImGui::TextUnformatted("Sparse storage (VDB-style)");
    ImGui::Checkbox("Build sparse .rsdfvdb alongside dense .rsdf", &s.UseSparse);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Hand-rolled 2-level brick container.\n"
                          "Empty bricks compress to sentinel indices (no payload).\n"
                          "Typical compression on shaderBall_256: 4-7x.");
    }
    ImGui::BeginDisabled(!s.UseSparse);
    ImGui::Combo("Brick size", &s.BrickSizeChoice, kBrickSizeLabels,
                 IM_ARRAYSIZE(kBrickSizeLabels));
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("4^3: finer sparsity, more index overhead. Good for thin geometry.\n"
                          "8^3: NanoVDB default. Balanced.\n"
                          "16^3: coarser sparsity, less indirection. Better for blobby meshes.");
    }
    ImGui::EndDisabled();

    // Output target ------------------------------------------------------
    SeedOutputBuffersFromMesh(s);
    ImGui::Separator();
    ImGui::TextUnformatted("Output");
    ImGui::InputText("Folder", s.OutputDirBuf.data(),
                     s.OutputDirBuf.size());
    ImGui::InputText("Name",   s.OutputNameBuf.data(),
                     s.OutputNameBuf.size());
    const uint32_t plannedRes = SdfBakerResolutionPx(s);
    const std::string plannedPath = ComposeOutputPath(s, plannedRes);
    ImGui::TextDisabled("→ %s", plannedPath.c_str());
    if (s.UseSparse) {
        const uint32_t bs = BrickSizePx(s.BrickSizeChoice);
        const std::string sparsePath = ComposeSparseOutputPath(s, plannedRes, bs);
        ImGui::TextDisabled("→ %s", sparsePath.c_str());
    }

    // Action buttons -----------------------------------------------------
    ImGui::Separator();
    const bool canAct = (s.TargetMesh != 0) && !s.BakeInFlight;
    ImGui::BeginDisabled(!canAct);
    if (ImGui::Button("Bake")) {
        KickBake(s, ctx, scene, s.TargetMesh, SdfBakerResolutionPx(s), s.AlgorithmChoice);
    }
    ImGui::SameLine();
    if (ImGui::Button("Re-bake")) {
        if (GlobalSDFGet(globalSdf, s.TargetMesh)) {
            GlobalSDFEvict(globalSdf, ctx, s.TargetMesh);
            residencyChanged = true;
            s.ResidentBakedCopy.reset();
        }
        KickBake(s, ctx, scene, s.TargetMesh, SdfBakerResolutionPx(s), s.AlgorithmChoice);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        // Prefer the panel's editable output path; fall back to the legacy
        // Cache/SDF/<basename>_<res>.rsdf derive if that file is missing.
        const uint32_t res = SdfBakerResolutionPx(s);
        const std::string userPath = ComposeOutputPath(s, res);
        std::string loadPath = userPath;
        struct _stat64 probe{};
        if (_stat64(userPath.c_str(), &probe) != 0 && s.TargetSourcePath) {
            loadPath = SDFCacheDerivePath(s.TargetSourcePath, res);
        }
        auto baked = std::make_shared<BakedSDF>();
        if (SDFCacheLoad(loadPath.c_str(), *baked)) {
            const ResidentSDF up = GlobalSDFUploadBaked(globalSdf, ctx,
                                                       s.TargetMesh, *baked,
                                                       /*fromCache*/ true);
            if (up.Image != VK_NULL_HANDLE) {
                residencyChanged = true;
                s.ResidentBakedCopy = baked;
                s.SliceDirty = true;
                RefreshStatsFromBaked(s, *baked, /*fromCache*/ true,
                                      /*durationMs*/ 0.0,
                                      /*triCount*/ 0u, loadPath);
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As")) {
        // Write the currently-resident CPU copy to the panel's output path on
        // demand. Useful when re-baking under a different name without
        // re-running the BVH bake.
        if (s.ResidentBakedCopy) {
            SeedOutputBuffersFromMesh(s);
            const std::string outPath = ComposeOutputPath(s,
                s.ResidentBakedCopy->Resolution);
            EnsureDirRecursive(s.OutputDirBuf.data());
            if (SDFCacheSave(outPath.c_str(), *s.ResidentBakedCopy)) {
                s.Stats.LastSavedPath = outPath;
                s.Stats.DiskBytes     = QueryFileSize(outPath.c_str());
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Evict")) {
        if (GlobalSDFGet(globalSdf, s.TargetMesh)) {
            GlobalSDFEvict(globalSdf, ctx, s.TargetMesh);
            residencyChanged = true;
            s.ResidentBakedCopy.reset();
            s.SliceDirty = true;
            s.Stats = SdfBakerState::LastBakeStats{};
        }
    }
    ImGui::EndDisabled();

    if (s.BakeInFlight) {
        ImGui::SameLine();
        if (ImGui::Button("Cancel") && s.Progress) {
            s.Progress->Cancel.store(true, std::memory_order_release);
        }
    }

    // Progress bar -------------------------------------------------------
    if (s.BakeInFlight && s.Progress) {
        const float frac = s.Progress->Fraction.load(std::memory_order_relaxed);
        ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f));
    }

    // Slice preview ------------------------------------------------------
    ImGui::Separator();
    ImGui::Checkbox("Show slice preview", &s.ShowSlicePreview);
    if (ImGui::SliderFloat("Plane Y (0..1)", &s.SlicePlaneY01, 0.0f, 1.0f)) {
        s.SliceDirty = true;
    }
    if (s.ShowSlicePreview && s.SlicePreviewTex) {
        if (!s.ResidentBakedCopy) {
            ImGui::TextDisabled("Preview is empty until a bake completes.");
        }
        ImGui::Image(s.SlicePreviewTex, ImVec2(256.0f, 256.0f));
    }

    ImGui::TextDisabled("Tip: the main viewport's Debug → \"SDF Slice\" mode\n"
                        "uses the same plane Y, viz mode and GPU 3D texture.");

    // Stats --------------------------------------------------------------
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!s.Stats.Valid) {
            ImGui::TextDisabled("No bake yet — Bake / Load to populate.");
        } else {
            char buf[64];
            ImGui::Text("Resolution   : %u^3 (%llu voxels)",
                        s.Stats.Resolution,
                        static_cast<unsigned long long>(s.Stats.VoxelCount));
            FormatBytes(s.Stats.CpuBytes, buf, sizeof(buf));
            ImGui::Text("CPU RAM      : %s", buf);
            FormatBytes(s.Stats.GpuBytes, buf, sizeof(buf));
            ImGui::Text("GPU VRAM     : %s (R16_SNORM 3D)", buf);
            if (s.Stats.DiskBytes > 0) {
                FormatBytes(s.Stats.DiskBytes, buf, sizeof(buf));
                ImGui::Text("On disk      : %s", buf);
                ImGui::TextDisabled("%s", s.Stats.LastSavedPath.c_str());
            } else {
                ImGui::TextDisabled("On disk      : (not saved yet)");
            }
            if (s.Stats.FromCache) {
                ImGui::TextDisabled("Bake time    : (loaded from cache)");
            } else {
                ImGui::Text("Bake time    : %.1f ms", s.Stats.BakeDurationMs);
            }
            if (s.Stats.TriangleCount > 0) {
                ImGui::Text("Triangles    : %u", s.Stats.TriangleCount);
            }
            ImGui::Text("MaxDist      : %.4f", s.Stats.MaxDist);
            const double q = (s.Stats.MaxDist > 0.0f)
                ? static_cast<double>(s.Stats.MaxDist) / 32767.0 : 0.0;
            ImGui::Text("R16 step     : %.6f world units", q);

            if (s.Stats.SparseValid) {
                ImGui::Separator();
                ImGui::TextUnformatted("Sparse (.rsdfvdb)");
                ImGui::Text("Brick size   : %u^3", s.Stats.SparseBrickSize);
                ImGui::Text("Bricks       : %u / %u occupied (%.1f%%)",
                            s.Stats.SparseOccupiedBricks,
                            s.Stats.SparseTotalBricks,
                            s.Stats.SparseTotalBricks
                                ? (100.0 * s.Stats.SparseOccupiedBricks
                                   / double(s.Stats.SparseTotalBricks))
                                : 0.0);
                FormatBytes(s.Stats.SparseGpuBytes, buf, sizeof(buf));
                ImGui::Text("Sparse VRAM  : %s (idx+pool SSBOs)", buf);
                if (s.Stats.SparseDiskBytes > 0) {
                    FormatBytes(s.Stats.SparseDiskBytes, buf, sizeof(buf));
                    ImGui::Text("On disk      : %s", buf);
                    ImGui::TextDisabled("%s", s.Stats.SparseLastSavedPath.c_str());
                }
                if (s.Stats.GpuBytes > 0 && s.Stats.SparseGpuBytes > 0) {
                    const double ratio = double(s.Stats.GpuBytes)
                                       / double(s.Stats.SparseGpuBytes);
                    ImGui::Text("Compression  : %.2fx vs dense R16 3D image",
                                ratio);
                }
            }
        }
    }

    ImGui::End();
    return residencyChanged;
}

SDFSliceState SdfBakerComposeSliceState(const SdfBakerState& s,
                                        const GlobalSDF& globalSdf) {
    SDFSliceState out{};
    out.Sampler = globalSdf.Sampler;
    const ResidentSDF* r = GlobalSDFGet(globalSdf, s.TargetMesh);
    if (r) {
        out.View    = r->View;
        out.AABBMin = r->AABBMin;
        out.AABBMax = r->AABBMax;
        out.MaxDist = r->MaxDist;
        out.HasSDF  = true;
    } else {
        out.View    = globalSdf.DummyView;
        out.AABBMin = glm::vec3(0.0f);
        out.AABBMax = glm::vec3(1.0f);
        out.MaxDist = 1.0f;
        out.HasSDF  = false;
    }
    const float t = std::clamp(s.SlicePlaneY01, 0.0f, 1.0f);
    out.PlaneY = glm::mix(out.AABBMin.y, out.AABBMax.y, t);
    out.VizMode = static_cast<SDFVizMode>(std::clamp(s.VizModeChoice, 0, 4));
    return out;
}

} // namespace RS
