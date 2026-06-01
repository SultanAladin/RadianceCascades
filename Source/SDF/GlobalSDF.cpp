// Source/SDF/GlobalSDF.cpp — Phase 12
#include "SDF/GlobalSDF.h"
#include "Core/Logger.h"

#include <cstring>

namespace RS {

namespace {

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

bool Create3DImage(const VulkanContext& ctx, uint32_t res,
                   VkImage& outImage, VkDeviceMemory& outMemory,
                   VkImageView& outView) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType     = VK_IMAGE_TYPE_3D;
    ici.format        = VK_FORMAT_R16_SNORM;
    ici.extent        = { res, res, res };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.Device, &ici, nullptr, &outImage) != VK_SUCCESS) {
        RS_LOG_ERROR("GlobalSDF: vkCreateImage(3D) failed (res=%u)", res);
        return false;
    }
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
    vci.viewType         = VK_IMAGE_VIEW_TYPE_3D;
    vci.format           = VK_FORMAT_R16_SNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(ctx.Device, &vci, nullptr, &outView) == VK_SUCCESS;
}

bool CreateHostBuffer(const VulkanContext& ctx, VkDeviceSize bytes,
                      VkBufferUsageFlags usage,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bytes;
    bci.usage       = usage;
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

// One-shot command buffer for the staging upload. Allocates a transient cmd
// buffer from the main pool, records the staging→image copy, submits, waits
// idle, frees. Keeps Phase 12 from threading the live frame's cmd buffer with
// a layout-juggle that has to coordinate with PickingRecordCopy etc.
void SubmitOneShot(const VulkanContext& ctx,
                   VkBuffer staging, VkImage dst,
                   uint32_t res) {
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool        = ctx.CommandPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.Device, &cai, &cmd) != VK_SUCCESS) return;

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // UNDEFINED → TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image               = dst;
    toDst.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toDst.srcAccessMask       = 0;
    toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toDst);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { res, res, res };
    vkCmdCopyBufferToImage(cmd, staging, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier toRead = toDst;
    toRead.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toRead);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(ctx.GraphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.GraphicsQueue);

    vkFreeCommandBuffers(ctx.Device, ctx.CommandPool, 1, &cmd);
}

void DestroyResident(VkDevice device, ResidentSDF& r) {
    if (r.View)   vkDestroyImageView(device, r.View,   nullptr);
    if (r.Image)  vkDestroyImage    (device, r.Image,  nullptr);
    if (r.Memory) vkFreeMemory      (device, r.Memory, nullptr);
    r = ResidentSDF{};
}

void DestroyResidentSparse(VkDevice device, ResidentSparseSDF& r) {
    if (r.IndexBuffer) vkDestroyBuffer(device, r.IndexBuffer, nullptr);
    if (r.IndexMemory) vkFreeMemory   (device, r.IndexMemory, nullptr);
    if (r.PoolBuffer)  vkDestroyBuffer(device, r.PoolBuffer,  nullptr);
    if (r.PoolMemory)  vkFreeMemory   (device, r.PoolMemory,  nullptr);
    r = ResidentSparseSDF{};
}

// Create a device-local SSBO + matching staging buffer, copy via one-shot.
// Returns true on success. Used by sparse upload for both index + pool buffers.
bool CreateDeviceSSBO(const VulkanContext& ctx, VkDeviceSize bytes,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = bytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.Device, &bci, nullptr, &outBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.Device, outBuf, &req);
    const int memType = FindMemoryType(ctx.PhysicalDevice, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType < 0) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memType);
    if (vkAllocateMemory(ctx.Device, &mai, nullptr, &outMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(ctx.Device, outBuf, outMem, 0);
    return true;
}

void SubmitBufferCopy(const VulkanContext& ctx, VkBuffer src, VkBuffer dst,
                      VkDeviceSize bytes) {
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool        = ctx.CommandPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.Device, &cai, &cmd) != VK_SUCCESS) return;

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkBufferCopy region{};
    region.size = bytes;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    // Make the upload visible to compute + fragment reads.
    VkBufferMemoryBarrier bmb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bmb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = dst;
    bmb.size                = bytes;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(ctx.GraphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.GraphicsQueue);
    vkFreeCommandBuffers(ctx.Device, ctx.CommandPool, 1, &cmd);
}

uint64_t HashResidency(MeshHandle mesh, uint32_t res,
                       const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(b[i]);
            h *= 1099511628211ull;
        }
    };
    mix(&mesh,    sizeof(mesh));
    mix(&res,     sizeof(res));
    mix(&aabbMin, sizeof(aabbMin));
    mix(&aabbMax, sizeof(aabbMax));
    return h;
}

} // namespace

bool GlobalSDFInitialize(GlobalSDF& g, const VulkanContext& ctx) {
    if (g.Initialized) return true;

    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod       = 0.0f;
    sci.maxLod       = 0.0f;
    if (vkCreateSampler(ctx.Device, &sci, nullptr, &g.Sampler) != VK_SUCCESS) {
        RS_LOG_ERROR("GlobalSDF: sampler creation failed");
        return false;
    }

    // 1x1x1 fallback so descriptor sets bind something valid before any bake.
    if (!Create3DImage(ctx, 1, g.DummyImage, g.DummyMemory, g.DummyView)) {
        RS_LOG_ERROR("GlobalSDF: dummy 1^3 image failed");
        return false;
    }
    // Upload +1 (always-outside) into the dummy.
    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    if (!CreateHostBuffer(ctx, sizeof(int16_t),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMem)) return false;
    void* mapped = nullptr;
    vkMapMemory(ctx.Device, stagingMem, 0, sizeof(int16_t), 0, &mapped);
    int16_t v = 32767;
    std::memcpy(mapped, &v, sizeof(v));
    vkUnmapMemory(ctx.Device, stagingMem);
    SubmitOneShot(ctx, staging, g.DummyImage, 1);
    vkDestroyBuffer(ctx.Device, staging,    nullptr);
    vkFreeMemory   (ctx.Device, stagingMem, nullptr);

    g.Initialized = true;
    RS_LOG_INFO("GlobalSDF: ready (dummy 1^3 bound)");
    return true;
}

void GlobalSDFTerminate(GlobalSDF& g, const VulkanContext& ctx) {
    if (!g.Initialized) return;
    vkDeviceWaitIdle(ctx.Device);
    for (auto& kv : g.Resident) DestroyResident(ctx.Device, kv.second);
    g.Resident.clear();
    for (auto& kv : g.ResidentSparse) DestroyResidentSparse(ctx.Device, kv.second);
    g.ResidentSparse.clear();

    if (g.DummyView)   vkDestroyImageView(ctx.Device, g.DummyView,   nullptr);
    if (g.DummyImage)  vkDestroyImage    (ctx.Device, g.DummyImage,  nullptr);
    if (g.DummyMemory) vkFreeMemory      (ctx.Device, g.DummyMemory, nullptr);
    if (g.Sampler)     vkDestroySampler  (ctx.Device, g.Sampler,     nullptr);
    g = GlobalSDF{};
}

const ResidentSDF* GlobalSDFGet(const GlobalSDF& g, MeshHandle mesh) {
    auto it = g.Resident.find(mesh);
    if (it == g.Resident.end()) return nullptr;
    return &it->second;
}

void GlobalSDFEvict(GlobalSDF& g, const VulkanContext& ctx, MeshHandle mesh) {
    auto it = g.Resident.find(mesh);
    if (it == g.Resident.end()) return;
    vkDeviceWaitIdle(ctx.Device);
    DestroyResident(ctx.Device, it->second);
    g.Resident.erase(it);
    RS_LOG_INFO("GlobalSDF: evicted mesh %u", mesh);
}

ResidentSDF GlobalSDFUploadBaked(GlobalSDF& g, const VulkanContext& ctx,
                                 MeshHandle mesh, const BakedSDF& baked,
                                 bool fromCache) {
    if (!g.Initialized || mesh == 0 || baked.Resolution == 0 || baked.Voxels.empty()) {
        return ResidentSDF{};
    }

    // If the same mesh already has a residency, evict the previous one.
    auto existing = g.Resident.find(mesh);
    if (existing != g.Resident.end()) {
        vkDeviceWaitIdle(ctx.Device);
        DestroyResident(ctx.Device, existing->second);
        g.Resident.erase(existing);
    }

    ResidentSDF r{};
    r.Resolution = baked.Resolution;
    r.AABBMin    = baked.AABBMin;
    r.AABBMax    = baked.AABBMax;
    r.MaxDist    = baked.MaxDist;
    r.Hash       = HashResidency(mesh, baked.Resolution, baked.AABBMin, baked.AABBMax);
    r.FromCache  = fromCache;

    if (!Create3DImage(ctx, baked.Resolution, r.Image, r.Memory, r.View)) {
        RS_LOG_ERROR("GlobalSDF: 3D image alloc failed for mesh %u", mesh);
        return ResidentSDF{};
    }

    // Staging copy.
    const VkDeviceSize bytes = sizeof(int16_t) * baked.Voxels.size();
    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    if (!CreateHostBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMem)) {
        DestroyResident(ctx.Device, r);
        return ResidentSDF{};
    }
    void* mapped = nullptr;
    vkMapMemory(ctx.Device, stagingMem, 0, bytes, 0, &mapped);
    std::memcpy(mapped, baked.Voxels.data(), static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.Device, stagingMem);

    SubmitOneShot(ctx, staging, r.Image, baked.Resolution);

    vkDestroyBuffer(ctx.Device, staging,    nullptr);
    vkFreeMemory   (ctx.Device, stagingMem, nullptr);

    g.Resident.emplace(mesh, r);
    RS_LOG_INFO("GlobalSDF: uploaded mesh %u (%u^3, %.1f KB, %s)",
                mesh, baked.Resolution, bytes / 1024.0,
                fromCache ? "from cache" : "fresh bake");
    return r;
}

bool GlobalSDFTryLoadFromCache(GlobalSDF& g, const VulkanContext& ctx,
                               MeshHandle mesh, const char* sourcePath,
                               uint32_t resolution) {
    if (!g.Initialized) return false;
    const std::string path = SDFCacheDerivePath(sourcePath, resolution);
    BakedSDF baked;
    if (!SDFCacheLoad(path.c_str(), baked)) return false;
    const ResidentSDF r = GlobalSDFUploadBaked(g, ctx, mesh, baked, /*fromCache*/ true);
    return r.Image != VK_NULL_HANDLE;
}

bool GlobalSDFSaveResidentToCache(const GlobalSDF& g, MeshHandle mesh,
                                  const char* sourcePath) {
    // Caller holds the BakedSDF on its end; we only have the GPU image. So this
    // function expects the bake path to be Save = post-bake on the worker side.
    // For "save the currently-resident SDF to disk" we need to read it back —
    // not implemented v1 because the bake worker always saves immediately
    // after producing the BakedSDF. Returning false signals "use the bake
    // path's save instead".
    (void)g; (void)mesh; (void)sourcePath;
    return false;
}

// ---- sparse VDB-style residency --------------------------------------------

const ResidentSparseSDF* GlobalSDFGetSparse(const GlobalSDF& g, MeshHandle mesh) {
    auto it = g.ResidentSparse.find(mesh);
    if (it == g.ResidentSparse.end()) return nullptr;
    return &it->second;
}

void GlobalSDFEvictSparse(GlobalSDF& g, const VulkanContext& ctx, MeshHandle mesh) {
    auto it = g.ResidentSparse.find(mesh);
    if (it == g.ResidentSparse.end()) return;
    vkDeviceWaitIdle(ctx.Device);
    DestroyResidentSparse(ctx.Device, it->second);
    g.ResidentSparse.erase(it);
    RS_LOG_INFO("GlobalSDF: evicted sparse mesh %u", mesh);
}

ResidentSparseSDF GlobalSDFUploadSparse(GlobalSDF& g, const VulkanContext& ctx,
                                        MeshHandle mesh,
                                        const BakedSparseSDF& baked,
                                        bool fromCache) {
    if (!g.Initialized || mesh == 0 || baked.Resolution == 0 ||
        baked.BrickIndex.empty()) {
        return ResidentSparseSDF{};
    }

    auto existing = g.ResidentSparse.find(mesh);
    if (existing != g.ResidentSparse.end()) {
        vkDeviceWaitIdle(ctx.Device);
        DestroyResidentSparse(ctx.Device, existing->second);
        g.ResidentSparse.erase(existing);
    }

    ResidentSparseSDF r{};
    r.AABBMin            = baked.AABBMin;
    r.AABBMax            = baked.AABBMax;
    r.Resolution         = baked.Resolution;
    r.BrickSize          = baked.BrickSize;
    r.BrickGrid          = baked.BrickGrid;
    r.OccupiedBrickCount = baked.OccupiedBrickCount;
    r.MaxDist            = baked.MaxDist;
    r.FromCache          = fromCache;

    const VkDeviceSize indexBytes = sizeof(uint32_t) * baked.BrickIndex.size();
    const VkDeviceSize poolBytes  = baked.BrickPool.empty()
        ? sizeof(int16_t)   // 0-byte SSBOs are invalid; one int16 stub keeps Vulkan happy
        : sizeof(int16_t) * baked.BrickPool.size();
    r.IndexBytes = indexBytes;
    r.PoolBytes  = poolBytes;

    if (!CreateDeviceSSBO(ctx, indexBytes, r.IndexBuffer, r.IndexMemory)) {
        RS_LOG_ERROR("GlobalSDF: sparse index SSBO alloc failed (%llu B)",
                     static_cast<unsigned long long>(indexBytes));
        return ResidentSparseSDF{};
    }
    if (!CreateDeviceSSBO(ctx, poolBytes, r.PoolBuffer, r.PoolMemory)) {
        RS_LOG_ERROR("GlobalSDF: sparse pool SSBO alloc failed (%llu B)",
                     static_cast<unsigned long long>(poolBytes));
        DestroyResidentSparse(ctx.Device, r);
        return ResidentSparseSDF{};
    }

    // Staging copies.
    auto stage = [&](const void* src, VkDeviceSize bytes, VkBuffer dst) -> bool {
        VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        if (!CreateHostBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging, stagingMem)) return false;
        void* mapped = nullptr;
        vkMapMemory(ctx.Device, stagingMem, 0, bytes, 0, &mapped);
        if (src) std::memcpy(mapped, src, static_cast<size_t>(bytes));
        else     std::memset(mapped, 0, static_cast<size_t>(bytes));   // stub case
        vkUnmapMemory(ctx.Device, stagingMem);
        SubmitBufferCopy(ctx, staging, dst, bytes);
        vkDestroyBuffer(ctx.Device, staging,    nullptr);
        vkFreeMemory   (ctx.Device, stagingMem, nullptr);
        return true;
    };

    if (!stage(baked.BrickIndex.data(), indexBytes, r.IndexBuffer)) {
        DestroyResidentSparse(ctx.Device, r);
        return ResidentSparseSDF{};
    }
    if (!stage(baked.BrickPool.empty() ? nullptr : baked.BrickPool.data(),
               poolBytes, r.PoolBuffer)) {
        DestroyResidentSparse(ctx.Device, r);
        return ResidentSparseSDF{};
    }

    g.ResidentSparse.emplace(mesh, r);
    RS_LOG_INFO("GlobalSDF: uploaded sparse mesh %u (%u^3 b%u, %u/%u bricks, idx=%llu KB, pool=%llu KB, %s)",
                mesh, baked.Resolution, baked.BrickSize,
                baked.OccupiedBrickCount,
                static_cast<uint32_t>(baked.BrickIndex.size()),
                static_cast<unsigned long long>(indexBytes / 1024),
                static_cast<unsigned long long>(poolBytes / 1024),
                fromCache ? "from cache" : "fresh bake");
    return r;
}

bool GlobalSDFTryLoadSparseFromCache(GlobalSDF& g, const VulkanContext& ctx,
                                     MeshHandle mesh, const char* sourcePath,
                                     uint32_t resolution, uint32_t brickSize) {
    if (!g.Initialized || mesh == 0 || sourcePath == nullptr) return false;
    const std::string path = SDFCacheDeriveSparsePath(sourcePath, resolution, brickSize);
    BakedSparseSDF baked;
    if (!SDFCacheLoadSparse(path.c_str(), baked)) return false;
    const ResidentSparseSDF r = GlobalSDFUploadSparse(g, ctx, mesh, baked,
                                                      /*fromCache*/ true);
    return r.IndexBuffer != VK_NULL_HANDLE;
}

} // namespace RS
