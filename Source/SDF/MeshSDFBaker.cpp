// Source/SDF/MeshSDFBaker.cpp — Phase 12
// Brute-force exact mesh-to-SDF: median-split AABB BVH + Eberly point-to-tri
// closed-form distance + closest-tri face-normal sign. Parallel over Z slices.
#include "SDF/MeshSDFBaker.h"
#include "Scene/ObjLoader.h"
#include "Core/Logger.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace RS {

namespace {

// ---- triangle BVH ----------------------------------------------------------

struct Tri {
    glm::vec3 V0, V1, V2;
    glm::vec3 Centroid;
    glm::vec3 Normal;       // face normal (un-normalised area-weighted) → normalised once
    glm::vec3 AABBMin, AABBMax;
};

struct BvhNode {
    glm::vec3 AABBMin;
    glm::vec3 AABBMax;
    int32_t   FirstTri = 0;   // leaf: first tri index in m_OrderedTris; internal: -1
    int32_t   TriCount = 0;   // leaf: count; internal: 0
    int32_t   LeftChild  = -1;
    int32_t   RightChild = -1;
};

class TriBVH {
public:
    bool Build(const std::vector<Tri>& tris) {
        if (tris.empty()) return false;
        m_Tris = tris;
        m_OrderedIndices.resize(m_Tris.size());
        for (size_t i = 0; i < m_Tris.size(); ++i) m_OrderedIndices[i] = static_cast<int>(i);
        m_Nodes.reserve(m_Tris.size() * 2);

        std::vector<int> work = m_OrderedIndices;
        m_OrderedIndices.clear();
        BuildRecursive(work.data(), work.data() + work.size());
        return true;
    }

    // Returns squared distance to nearest triangle + closest-point and the
    // index of the winning triangle.
    void ClosestPoint(const glm::vec3& p,
                      float& outDistSq,
                      glm::vec3& outClosest,
                      int& outTriIndex) const {
        outDistSq   = std::numeric_limits<float>::max();
        outClosest  = glm::vec3(0.0f);
        outTriIndex = -1;
        if (m_Nodes.empty()) return;
        DescendClosest(0, p, outDistSq, outClosest, outTriIndex);
    }

    const Tri& GetTri(int index) const { return m_Tris[index]; }

private:
    static glm::vec3 AABBMinV(const glm::vec3& a, const glm::vec3& b) { return glm::min(a, b); }
    static glm::vec3 AABBMaxV(const glm::vec3& a, const glm::vec3& b) { return glm::max(a, b); }

    int BuildRecursive(int* first, int* last) {
        const int node = static_cast<int>(m_Nodes.size());
        m_Nodes.emplace_back();
        BvhNode& n = m_Nodes.back();

        // Compute node AABB + centroid AABB
        glm::vec3 aMin( std::numeric_limits<float>::max());
        glm::vec3 aMax(-std::numeric_limits<float>::max());
        glm::vec3 cMin = aMin, cMax = aMax;
        for (int* it = first; it != last; ++it) {
            const Tri& t = m_Tris[*it];
            aMin = AABBMinV(aMin, t.AABBMin); aMax = AABBMaxV(aMax, t.AABBMax);
            cMin = AABBMinV(cMin, t.Centroid); cMax = AABBMaxV(cMax, t.Centroid);
        }
        n.AABBMin = aMin; n.AABBMax = aMax;

        const int count = static_cast<int>(last - first);
        constexpr int kLeafThreshold = 8;
        if (count <= kLeafThreshold) {
            n.FirstTri = static_cast<int>(m_OrderedIndices.size());
            n.TriCount = count;
            for (int* it = first; it != last; ++it) m_OrderedIndices.push_back(*it);
            return node;
        }

        // Split along longest centroid axis
        const glm::vec3 cExtent = cMax - cMin;
        int axis = 0;
        if (cExtent.y > cExtent.x) axis = 1;
        if (cExtent.z > cExtent[axis]) axis = 2;

        // Degenerate: all centroids coincide → make a leaf
        if (cExtent[axis] <= 0.0f) {
            n.FirstTri = static_cast<int>(m_OrderedIndices.size());
            n.TriCount = count;
            for (int* it = first; it != last; ++it) m_OrderedIndices.push_back(*it);
            return node;
        }

        int* mid = first + count / 2;
        std::nth_element(first, mid, last, [this, axis](int a, int b) {
            return m_Tris[a].Centroid[axis] < m_Tris[b].Centroid[axis];
        });

        // Recurse — careful: emplace_back may invalidate `n`; use the node index.
        const int leftChild  = BuildRecursive(first, mid);
        const int rightChild = BuildRecursive(mid,   last);
        m_Nodes[node].LeftChild  = leftChild;
        m_Nodes[node].RightChild = rightChild;
        m_Nodes[node].FirstTri   = -1;
        m_Nodes[node].TriCount   = 0;
        return node;
    }

    static float PointAABBDistSq(const glm::vec3& p,
                                 const glm::vec3& mn, const glm::vec3& mx) {
        float d2 = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const float v = p[i];
            if      (v < mn[i]) { const float d = mn[i] - v; d2 += d * d; }
            else if (v > mx[i]) { const float d = v - mx[i]; d2 += d * d; }
        }
        return d2;
    }


public:
    // Eberly closest-point on triangle (regions 0-6).
    static glm::vec3 ClosestPointOnTriangle(const glm::vec3& p,
                                            const glm::vec3& a,
                                            const glm::vec3& b,
                                            const glm::vec3& c) {
        const glm::vec3 ab = b - a;
        const glm::vec3 ac = c - a;
        const glm::vec3 ap = p - a;

        const float d1 = glm::dot(ab, ap);
        const float d2 = glm::dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return a;

        const glm::vec3 bp = p - b;
        const float d3 = glm::dot(ab, bp);
        const float d4 = glm::dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) return b;

        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float v = d1 / (d1 - d3);
            return a + v * ab;
        }

        const glm::vec3 cp = p - c;
        const float d5 = glm::dot(ab, cp);
        const float d6 = glm::dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) return c;

        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float w = d2 / (d2 - d6);
            return a + w * ac;
        }

        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b);
        }

        const float denom = 1.0f / (va + vb + vc);
        const float v = vb * denom;
        const float w = vc * denom;
        return a + ab * v + ac * w;
    }

private:
    void DescendClosest(int nodeIdx, const glm::vec3& p,
                        float& bestSq, glm::vec3& bestPt, int& bestTri) const {
        const BvhNode& n = m_Nodes[nodeIdx];
        const float nodeSq = PointAABBDistSq(p, n.AABBMin, n.AABBMax);
        if (nodeSq >= bestSq) return;

        if (n.LeftChild < 0) {
            // Leaf — brute test
            for (int k = 0; k < n.TriCount; ++k) {
                const int triIdx = m_OrderedIndices[n.FirstTri + k];
                const Tri& t = m_Tris[triIdx];
                const glm::vec3 cp = ClosestPointOnTriangle(p, t.V0, t.V1, t.V2);
                const glm::vec3 d  = p - cp;
                const float dsq    = glm::dot(d, d);
                if (dsq < bestSq) {
                    bestSq  = dsq;
                    bestPt  = cp;
                    bestTri = triIdx;
                }
            }
            return;
        }

        // Recurse into the closer child first for better pruning
        const float leftSq  = PointAABBDistSq(p, m_Nodes[n.LeftChild ].AABBMin,
                                                  m_Nodes[n.LeftChild ].AABBMax);
        const float rightSq = PointAABBDistSq(p, m_Nodes[n.RightChild].AABBMin,
                                                  m_Nodes[n.RightChild].AABBMax);
        if (leftSq < rightSq) {
            if (leftSq  < bestSq) DescendClosest(n.LeftChild , p, bestSq, bestPt, bestTri);
            if (rightSq < bestSq) DescendClosest(n.RightChild, p, bestSq, bestPt, bestTri);
        } else {
            if (rightSq < bestSq) DescendClosest(n.RightChild, p, bestSq, bestPt, bestTri);
            if (leftSq  < bestSq) DescendClosest(n.LeftChild , p, bestSq, bestPt, bestTri);
        }
    }

    std::vector<Tri>     m_Tris;
    std::vector<BvhNode> m_Nodes;
    std::vector<int>     m_OrderedIndices;
};

// ---- mesh data access ------------------------------------------------------
// Map the host-visible vertex + index buffers, copy to local std::vectors so
// the GPU buffers can be safely unmapped before we kick the parallel bake.
bool ReadMeshCpuData(const VulkanContext& ctx, const GpuMesh& mesh,
                     std::vector<ParsedVertex>& outVerts,
                     std::vector<uint32_t>&     outIndices) {
    if (mesh.VertexBuffer == VK_NULL_HANDLE || mesh.IndexBuffer == VK_NULL_HANDLE) {
        return false;
    }
    const VkDeviceSize vBytes = sizeof(ParsedVertex) * mesh.VertexCount;
    const VkDeviceSize iBytes = sizeof(uint32_t)     * mesh.IndexCount;

    void* mapped = nullptr;
    if (vkMapMemory(ctx.Device, mesh.VertexMemory, 0, vBytes, 0, &mapped) != VK_SUCCESS) {
        return false;
    }
    outVerts.resize(mesh.VertexCount);
    std::memcpy(outVerts.data(), mapped, static_cast<size_t>(vBytes));
    vkUnmapMemory(ctx.Device, mesh.VertexMemory);

    if (vkMapMemory(ctx.Device, mesh.IndexMemory, 0, iBytes, 0, &mapped) != VK_SUCCESS) {
        return false;
    }
    outIndices.resize(mesh.IndexCount);
    std::memcpy(outIndices.data(), mapped, static_cast<size_t>(iBytes));
    vkUnmapMemory(ctx.Device, mesh.IndexMemory);
    return true;
}

// ---- shared bake helpers (Phase 15a refactor) ------------------------------

struct SdfBounds {
    glm::vec3 AABBMin;
    glm::vec3 AABBMax;
    glm::vec3 VoxelSize;
    float     MaxDist;
};

// Build per-triangle records from a parsed vertex + index stream. Drops
// degenerate (zero-area) triangles. Returns true if at least one triangle made
// it through.
bool BuildTris(const std::vector<ParsedVertex>& verts,
               const std::vector<uint32_t>&     indices,
               std::vector<Tri>&                outTris) {
    const size_t triCount = indices.size() / 3;
    outTris.clear();
    outTris.reserve(triCount);
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = indices[t * 3 + 0];
        const uint32_t i1 = indices[t * 3 + 1];
        const uint32_t i2 = indices[t * 3 + 2];
        if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;

        Tri tri{};
        tri.V0 = verts[i0].Position;
        tri.V1 = verts[i1].Position;
        tri.V2 = verts[i2].Position;
        const glm::vec3 e1 = tri.V1 - tri.V0;
        const glm::vec3 e2 = tri.V2 - tri.V0;
        glm::vec3 n  = glm::cross(e1, e2);
        const float nl = glm::length(n);
        if (nl < 1e-20f) continue;        // degenerate triangle
        tri.Normal   = n / nl;
        tri.Centroid = (tri.V0 + tri.V1 + tri.V2) * (1.0f / 3.0f);
        tri.AABBMin  = glm::min(glm::min(tri.V0, tri.V1), tri.V2);
        tri.AABBMax  = glm::max(glm::max(tri.V0, tri.V1), tri.V2);
        outTris.push_back(tri);
    }
    return !outTris.empty();
}

// Pad the mesh AABB by 5% so the SDF has gradient slack outside the surface;
// pick MaxDist = 1.5 × box diagonal so even the far corner sits unsaturated in
// the R16_SNORM band.
SdfBounds ComputeSdfBounds(const GpuMesh& mesh, uint32_t resolution) {
    SdfBounds b{};
    const glm::vec3 aabbCenter = 0.5f * (mesh.AABBMin + mesh.AABBMax);
    const glm::vec3 aabbHalf   = 0.5f * (mesh.AABBMax - mesh.AABBMin);
    const glm::vec3 padHalf    = aabbHalf * 1.05f + glm::vec3(1e-3f);
    b.AABBMin   = aabbCenter - padHalf;
    b.AABBMax   = aabbCenter + padHalf;
    const glm::vec3 boxSize = b.AABBMax - b.AABBMin;
    b.VoxelSize = boxSize / static_cast<float>(resolution);
    b.MaxDist   = 1.5f * glm::length(boxSize);
    return b;
}

// Pulls mesh CPU data, builds tris, builds BVH, computes bounds. Publishes the
// triangle count via `progress` once the BVH is ready (matches the dense
// path's existing point of publication so progress observers behave the same).
// Returns false on read failure / empty mesh / BVH build failure.
bool PrepareBake(const VulkanContext& ctx,
                 const GpuMesh&       mesh,
                 uint32_t             resolution,
                 BakeProgress*        progress,
                 std::vector<Tri>&    outTris,
                 TriBVH&              outBvh,
                 SdfBounds&           outBounds) {
    if (mesh.IndexCount < 3) {
        RS_LOG_ERROR("MeshSDFBaker: mesh has no triangles");
        return false;
    }
    std::vector<ParsedVertex> verts;
    std::vector<uint32_t>     indices;
    if (!ReadMeshCpuData(ctx, mesh, verts, indices)) {
        RS_LOG_ERROR("MeshSDFBaker: failed to read mesh CPU data");
        return false;
    }
    if (!BuildTris(verts, indices, outTris)) {
        RS_LOG_ERROR("MeshSDFBaker: all triangles degenerate");
        return false;
    }
    if (!outBvh.Build(outTris)) {
        RS_LOG_ERROR("MeshSDFBaker: BVH build failed");
        return false;
    }
    if (progress) {
        progress->TriangleCount.store(static_cast<uint32_t>(outTris.size()),
                                      std::memory_order_relaxed);
    }
    outBounds = ComputeSdfBounds(mesh, resolution);
    return true;
}

} // namespace

BakedSDF MeshSDFBakerBake(const VulkanContext& ctx,
                          const GpuMesh&       mesh,
                          uint32_t             resolution,
                          BakeProgress*        progress,
                          int                  algorithmChoice) {
    const auto bakeStart = std::chrono::steady_clock::now();
    BakedSDF result{};
    if (resolution == 0 || resolution > 256) {
        RS_LOG_ERROR("MeshSDFBaker: invalid resolution %u", resolution);
        return result;
    }

    // 1-4. Shared prep: mesh CPU read + tri build + BVH + bounds.
    std::vector<Tri> tris;
    TriBVH bvh;
    SdfBounds bounds{};
    if (!PrepareBake(ctx, mesh, resolution, progress, tris, bvh, bounds)) {
        return result;
    }

    result.AABBMin    = bounds.AABBMin;
    result.AABBMax    = bounds.AABBMax;
    result.Resolution = resolution;
    result.MaxDist    = bounds.MaxDist;
    const glm::vec3 voxelSize = bounds.VoxelSize;

    const size_t voxelCount = static_cast<size_t>(resolution) *
                              static_cast<size_t>(resolution) *
                              static_cast<size_t>(resolution);
    result.Voxels.assign(voxelCount, 0);

    if (algorithmChoice == 0) {
        // 5. Parallel bake over Z slices (Brute-force).
        const unsigned hwc = std::thread::hardware_concurrency();
        const unsigned workerCount = hwc == 0 ? 4u : std::min(hwc, 16u);
        std::atomic<uint32_t> nextZ { 0 };
        std::atomic<uint32_t> doneZ { 0 };

        auto worker = [&]() {
            const uint32_t res = resolution;
            const glm::vec3 origin = result.AABBMin + 0.5f * voxelSize;
            while (true) {
                if (progress && progress->Cancel.load(std::memory_order_relaxed)) return;
                const uint32_t z = nextZ.fetch_add(1, std::memory_order_relaxed);
                if (z >= res) return;
                for (uint32_t y = 0; y < res; ++y) {
                    for (uint32_t x = 0; x < res; ++x) {
                        const glm::vec3 p = origin + glm::vec3(
                            voxelSize.x * static_cast<float>(x),
                            voxelSize.y * static_cast<float>(y),
                            voxelSize.z * static_cast<float>(z));

                        float dsq; glm::vec3 cp; int triIdx;
                        bvh.ClosestPoint(p, dsq, cp, triIdx);
                        const float dist = std::sqrt(dsq);
                        float signedDist = dist;
                        if (triIdx >= 0) {
                            const glm::vec3& n = bvh.GetTri(triIdx).Normal;
                            if (glm::dot(p - cp, n) < 0.0f) signedDist = -dist;
                        }

                        // R16_SNORM encode
                        const float t = std::max(-1.0f, std::min(1.0f,
                            signedDist / result.MaxDist));
                        const int q = static_cast<int>(std::lround(t * 32767.0f));
                        const size_t idx = (static_cast<size_t>(z) * res + y) * res + x;
                        result.Voxels[idx] = static_cast<int16_t>(q);
                    }
                }
                const uint32_t finished = doneZ.fetch_add(1, std::memory_order_relaxed) + 1;
                if (progress) {
                    progress->Fraction.store(static_cast<float>(finished) /
                                             static_cast<float>(res),
                                             std::memory_order_relaxed);
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(workerCount);
        for (unsigned i = 0; i < workerCount; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    } else if (algorithmChoice == 3) {
        // Fast Sweeping Method (FSM)
        const uint32_t res = resolution;
        const size_t voxelCount = static_cast<size_t>(res) * res * res;
        std::vector<float> distSq(voxelCount, std::numeric_limits<float>::max());
        std::vector<int> closestTri(voxelCount, -1);
        
        auto getIdx = [res](int x, int y, int z) -> size_t {
            return (static_cast<size_t>(z) * res + y) * res + x;
        };

        const glm::vec3 origin = result.AABBMin + 0.5f * voxelSize;

        // Step 1: Narrow band initialization
        for (size_t t = 0; t < tris.size(); ++t) {
            if (progress && progress->Cancel.load(std::memory_order_relaxed)) return result;
            const Tri& tri = tris[t];
            glm::vec3 triMin = (tri.AABBMin - result.AABBMin) / voxelSize;
            glm::vec3 triMax = (tri.AABBMax - result.AABBMin) / voxelSize;
            int minX = std::max(0, static_cast<int>(std::floor(triMin.x)) - 1);
            int minY = std::max(0, static_cast<int>(std::floor(triMin.y)) - 1);
            int minZ = std::max(0, static_cast<int>(std::floor(triMin.z)) - 1);
            int maxX = std::min(static_cast<int>(res) - 1, static_cast<int>(std::ceil(triMax.x)) + 1);
            int maxY = std::min(static_cast<int>(res) - 1, static_cast<int>(std::ceil(triMax.y)) + 1);
            int maxZ = std::min(static_cast<int>(res) - 1, static_cast<int>(std::ceil(triMax.z)) + 1);

            for (int z = minZ; z <= maxZ; ++z) {
                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        glm::vec3 p = origin + glm::vec3(x * voxelSize.x, y * voxelSize.y, z * voxelSize.z);
                        glm::vec3 cp = TriBVH::ClosestPointOnTriangle(p, tri.V0, tri.V1, tri.V2);
                        float d2 = glm::dot(p - cp, p - cp);
                        size_t idx = getIdx(x, y, z);
                        if (d2 < distSq[idx]) {
                            distSq[idx] = d2;
                            closestTri[idx] = static_cast<int>(t);
                        }
                    }
                }
            }
        }

        // Step 2: Sweeping passes
        for (int pass = 0; pass < 8; ++pass) {
            if (progress && progress->Cancel.load(std::memory_order_relaxed)) return result;
            int dirX = (pass & 1) ? -1 : 1;
            int dirY = (pass & 2) ? -1 : 1;
            int dirZ = (pass & 4) ? -1 : 1;
            int startX = (dirX > 0) ? 0 : res - 1;
            int startY = (dirY > 0) ? 0 : res - 1;
            int startZ = (dirZ > 0) ? 0 : res - 1;

            for (int z = startZ; z >= 0 && z < static_cast<int>(res); z += dirZ) {
                for (int y = startY; y >= 0 && y < static_cast<int>(res); y += dirY) {
                    for (int x = startX; x >= 0 && x < static_cast<int>(res); x += dirX) {
                        size_t idx = getIdx(x, y, z);
                        glm::vec3 p = origin + glm::vec3(x * voxelSize.x, y * voxelSize.y, z * voxelSize.z);

                        // Check 3 neighbors that have already been updated in this sweep
                        int nx = x - dirX;
                        int ny = y - dirY;
                        int nz = z - dirZ;
                        
                        auto testNeighbor = [&](int nx, int ny, int nz) {
                            if (nx >= 0 && nx < static_cast<int>(res) &&
                                ny >= 0 && ny < static_cast<int>(res) &&
                                nz >= 0 && nz < static_cast<int>(res)) {
                                size_t nIdx = getIdx(nx, ny, nz);
                                int nTri = closestTri[nIdx];
                                if (nTri >= 0) {
                                    const Tri& tri = tris[nTri];
                                    glm::vec3 cp = TriBVH::ClosestPointOnTriangle(p, tri.V0, tri.V1, tri.V2);
                                    float d2 = glm::dot(p - cp, p - cp);
                                    if (d2 < distSq[idx]) {
                                        distSq[idx] = d2;
                                        closestTri[idx] = nTri;
                                    }
                                }
                            }
                        };

                        testNeighbor(nx, y, z);
                        testNeighbor(x, ny, z);
                        testNeighbor(x, y, nz);
                    }
                }
            }
            if (progress) progress->Fraction.store((pass + 1) / 9.0f, std::memory_order_relaxed);
        }

        // Step 3: Sign evaluation and output
        for (int z = 0; z < static_cast<int>(res); ++z) {
            for (int y = 0; y < static_cast<int>(res); ++y) {
                for (int x = 0; x < static_cast<int>(res); ++x) {
                    size_t idx = getIdx(x, y, z);
                    float signedDist = result.MaxDist; // Default to far outside
                    int triIdx = closestTri[idx];
                    
                    if (triIdx >= 0) {
                        const Tri& tri = tris[triIdx];
                        glm::vec3 p = origin + glm::vec3(x * voxelSize.x, y * voxelSize.y, z * voxelSize.z);
                        glm::vec3 cp = TriBVH::ClosestPointOnTriangle(p, tri.V0, tri.V1, tri.V2);
                        float dist = std::sqrt(distSq[idx]);
                        signedDist = dist;
                        if (glm::dot(p - cp, tri.Normal) < 0.0f) signedDist = -dist;
                    }

                    // R16_SNORM encode
                    const float t = std::max(-1.0f, std::min(1.0f, signedDist / result.MaxDist));
                    const int q = static_cast<int>(std::lround(t * 32767.0f));
                    result.Voxels[idx] = static_cast<int16_t>(q);
                }
            }
        }
        if (progress) progress->Fraction.store(1.0f, std::memory_order_relaxed);
    }

    if (progress && progress->Cancel.load(std::memory_order_relaxed)) {
        RS_LOG_INFO("MeshSDFBaker: bake cancelled (res %u)", resolution);
        result = BakedSDF{};
        return result;
    }

    if (progress) progress->Fraction.store(1.0f, std::memory_order_relaxed);
    const auto bakeEnd = std::chrono::steady_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(
        bakeEnd - bakeStart).count();
    if (progress) progress->DurationMs.store(durationMs, std::memory_order_relaxed);
    RS_LOG_INFO("MeshSDFBaker: baked %u^3 SDF over %zu triangles (maxDist=%.3f, %.1f ms)",
                resolution, tris.size(), result.MaxDist, durationMs);
    return result;
}

// ---- Phase 15a: brick-native sparse baker ----------------------------------
// Produces BakedSparseSDF directly. Peak RAM scales with occupiedBricks ×
// brickSize^3, not res^3. Matches what SDFCacheBuildSparse(dense, brickSize)
// would produce on a finished dense bake (same classify threshold, same sign
// rule, same trailing-brick clamp).
BakedSparseSDF MeshSDFBakerBakeSparse(const VulkanContext& ctx,
                                      const GpuMesh&       mesh,
                                      uint32_t             resolution,
                                      uint32_t             brickSize,
                                      BakeProgress*        progress,
                                      int                  algorithmChoice) {
    const auto bakeStart = std::chrono::steady_clock::now();
    BakedSparseSDF result{};
    if (resolution == 0 || resolution > 1024) {
        RS_LOG_ERROR("MeshSDFBakerSparse: invalid resolution %u", resolution);
        return result;
    }
    if (brickSize != 4u && brickSize != 8u && brickSize != 16u) {
        RS_LOG_ERROR("MeshSDFBakerSparse: invalid brickSize %u (must be 4/8/16)", brickSize);
        return result;
    }
    if (algorithmChoice != 0) {
        RS_LOG_WARN("MeshSDFBakerSparse: algorithmChoice=%d not implemented (FSM is Phase 15b); falling back to BVH",
                    algorithmChoice);
    }

    std::vector<Tri> tris;
    TriBVH bvh;
    SdfBounds bounds{};
    if (!PrepareBake(ctx, mesh, resolution, progress, tris, bvh, bounds)) {
        return result;
    }

    result.AABBMin    = bounds.AABBMin;
    result.AABBMax    = bounds.AABBMax;
    result.Resolution = resolution;
    result.MaxDist    = bounds.MaxDist;
    result.BrickSize  = brickSize;
    result.BrickGrid  = (resolution + brickSize - 1u) / brickSize;

    const uint32_t bg      = result.BrickGrid;
    const uint32_t bs      = brickSize;
    const uint32_t bsCubed = bs * bs * bs;
    const size_t   indexCount = static_cast<size_t>(bg) * bg * bg;
    result.BrickIndex.assign(indexCount, kSparseBrickIndexAllOutside);

    // Same one-SNORM-step classify threshold as SDFCacheBuildSparse → identical
    // mixed/sentinel partition. A brick is one-sided iff every voxel-centre has
    // |d| > threshold. Use the BVH's exact closest-point query at the brick
    // centre + the triangle-inequality bound:
    //
    //   for any voxel-centre v in the brick: dist(v, mesh) ≥ centreDist − halfDiag
    //
    // where halfDiag = half the brick's voxel-centre diagonal. So we can
    // classify the brick as one-sided when:
    //
    //   centreDist > threshold + halfDiag
    //
    // (AABB-vs-tri-AABB at BVH leaves is too loose — long thin tris have AABBs
    // that intersect distant bricks, collapsing the bound to ~0 and forcing
    // every brick into the mixed bucket.)
    const float threshold = (1.0f / 32767.0f) * result.MaxDist;

    const glm::vec3 voxelSize = bounds.VoxelSize;
    const glm::vec3 origin    = bounds.AABBMin + 0.5f * voxelSize;

    // Per-worker accumulation. Each worker collects mixed bricks into a local
    // pool + a map of (flatBrickIdx → localOffsetInBricks). Sentinels are
    // written directly into the shared BrickIndex (distinct slots, race-free).
    struct WorkerOut {
        std::vector<int16_t>                       LocalPool;       // contiguous bs^3 chunks
        std::vector<std::pair<uint32_t, uint32_t>> LocalMap;        // (flatBrickIdx, localChunkIndex)
    };

    const unsigned hwc = std::thread::hardware_concurrency();
    const unsigned workerCount = hwc == 0 ? 4u : std::min(hwc, 16u);
    std::vector<WorkerOut> workerOut(workerCount);

    std::atomic<uint32_t> nextBrick { 0 };
    std::atomic<uint32_t> doneBricks { 0 };
    const uint32_t totalBricks = static_cast<uint32_t>(indexCount);

    auto worker = [&](unsigned widx) {
        WorkerOut& wo = workerOut[widx];
        while (true) {
            if (progress && progress->Cancel.load(std::memory_order_relaxed)) return;
            const uint32_t bi = nextBrick.fetch_add(1, std::memory_order_relaxed);
            if (bi >= totalBricks) return;

            // Unpack brick coord. BrickIndex layout = (bz * bg + by) * bg + bx.
            const uint32_t bx = bi % bg;
            const uint32_t by = (bi / bg) % bg;
            const uint32_t bz = bi / (bg * bg);

            // Voxel range in dense grid (inclusive lo, exclusive hi). Trailing
            // brick may straddle the resolution edge.
            const uint32_t x0 = bx * bs, x1 = std::min(x0 + bs, resolution);
            const uint32_t y0 = by * bs, y1 = std::min(y0 + bs, resolution);
            const uint32_t z0 = bz * bs, z1 = std::min(z0 + bs, resolution);

            // Voxel-centre extents inside the brick. Use the full bs span (not
            // the clamped x1/y1/z1) so the classify gate stays consistent
            // across every brick; the inner loop handles the clamp for partial
            // trailing bricks via the (sx,sy,sz) coordinate clamp below.
            const glm::vec3 brickMin = origin + glm::vec3(
                voxelSize.x * static_cast<float>(x0),
                voxelSize.y * static_cast<float>(y0),
                voxelSize.z * static_cast<float>(z0));
            const glm::vec3 brickMax = origin + glm::vec3(
                voxelSize.x * static_cast<float>(x0 + bs - 1u),
                voxelSize.y * static_cast<float>(y0 + bs - 1u),
                voxelSize.z * static_cast<float>(z0 + bs - 1u));
            const glm::vec3 centre   = 0.5f * (brickMin + brickMax);
            const float     halfDiag = 0.5f * glm::length(brickMax - brickMin);

            // One BVH query at the brick centre serves both classify and sign.
            float centreDsq; glm::vec3 centreCp; int centreTri;
            bvh.ClosestPoint(centre, centreDsq, centreCp, centreTri);
            const float centreDist = std::sqrt(centreDsq);

            // Sign at brick centre (same rule as the per-voxel inner loop).
            bool centreInside = false;
            if (centreTri >= 0) {
                const glm::vec3& n = bvh.GetTri(centreTri).Normal;
                if (glm::dot(centre - centreCp, n) < 0.0f) centreInside = true;
            }

            if (centreDist > threshold + halfDiag) {
                // Triangle inequality guarantees every voxel-centre in the
                // brick saturates: dist(v) ≥ centreDist − halfDiag > threshold.
                result.BrickIndex[bi] = centreInside ? kSparseBrickIndexAllInside
                                                     : kSparseBrickIndexAllOutside;
            } else {
                // Mixed — allocate a chunk in this worker's local pool.
                const uint32_t localChunk = static_cast<uint32_t>(wo.LocalMap.size());
                wo.LocalMap.emplace_back(bi, localChunk);
                const size_t poolBase = wo.LocalPool.size();
                wo.LocalPool.resize(poolBase + bsCubed);

                for (uint32_t lz = 0; lz < bs; ++lz) {
                for (uint32_t ly = 0; ly < bs; ++ly) {
                for (uint32_t lx = 0; lx < bs; ++lx) {
                    // Sample coord with trailing-brick clamp (matches
                    // SDFCacheBuildSparse lines 239-241): voxels past the
                    // dense resolution edge clamp to the last in-range coord.
                    const uint32_t sx = std::min(x0 + lx, resolution - 1u);
                    const uint32_t sy = std::min(y0 + ly, resolution - 1u);
                    const uint32_t sz = std::min(z0 + lz, resolution - 1u);
                    const glm::vec3 p = origin + glm::vec3(
                        voxelSize.x * static_cast<float>(sx),
                        voxelSize.y * static_cast<float>(sy),
                        voxelSize.z * static_cast<float>(sz));

                    float dsq; glm::vec3 cp; int triIdx;
                    bvh.ClosestPoint(p, dsq, cp, triIdx);
                    const float dist = std::sqrt(dsq);
                    float signedDist = dist;
                    if (triIdx >= 0) {
                        const glm::vec3& n = bvh.GetTri(triIdx).Normal;
                        if (glm::dot(p - cp, n) < 0.0f) signedDist = -dist;
                    }
                    const float t = std::max(-1.0f, std::min(1.0f,
                        signedDist / result.MaxDist));
                    const int q = static_cast<int>(std::lround(t * 32767.0f));
                    const size_t localIdx = (static_cast<size_t>(lz) * bs + ly) * bs + lx;
                    wo.LocalPool[poolBase + localIdx] = static_cast<int16_t>(q);
                }}}
            }

            const uint32_t finished = doneBricks.fetch_add(1, std::memory_order_relaxed) + 1;
            if (progress && totalBricks > 0) {
                progress->Fraction.store(static_cast<float>(finished) /
                                         static_cast<float>(totalBricks),
                                         std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workerCount);
    for (unsigned i = 0; i < workerCount; ++i) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();

    if (progress && progress->Cancel.load(std::memory_order_relaxed)) {
        RS_LOG_INFO("MeshSDFBakerSparse: bake cancelled (res %u^3, brick %u)",
                    resolution, brickSize);
        return BakedSparseSDF{};
    }

    // After-join concat: sort all (flatBrickIdx, payload) pairs by flatBrickIdx
    // so BrickIndex offsets follow the converter's (bz,by,bx) row-major order
    // — gives byte-parity with SDFCacheBuildSparse output, which simplifies
    // verification (see Phase 15a plan §7).
    struct Pending {
        uint32_t FlatIdx;
        unsigned Worker;
        uint32_t LocalChunk;
    };
    size_t mixedCount = 0;
    for (const auto& wo : workerOut) mixedCount += wo.LocalMap.size();

    std::vector<Pending> pending;
    pending.reserve(mixedCount);
    for (unsigned w = 0; w < workerCount; ++w) {
        for (const auto& m : workerOut[w].LocalMap) {
            pending.push_back({ m.first, w, m.second });
        }
    }
    std::sort(pending.begin(), pending.end(),
              [](const Pending& a, const Pending& b) { return a.FlatIdx < b.FlatIdx; });

    result.BrickPool.resize(static_cast<size_t>(mixedCount) * bsCubed);
    for (uint32_t poolIdx = 0; poolIdx < pending.size(); ++poolIdx) {
        const Pending& p = pending[poolIdx];
        result.BrickIndex[p.FlatIdx] = poolIdx;
        const WorkerOut& wo = workerOut[p.Worker];
        const size_t srcBase = static_cast<size_t>(p.LocalChunk) * bsCubed;
        const size_t dstBase = static_cast<size_t>(poolIdx) * bsCubed;
        std::memcpy(result.BrickPool.data() + dstBase,
                    wo.LocalPool.data()     + srcBase,
                    bsCubed * sizeof(int16_t));
    }
    result.OccupiedBrickCount = static_cast<uint32_t>(pending.size());

    if (progress) progress->Fraction.store(1.0f, std::memory_order_relaxed);
    const auto bakeEnd = std::chrono::steady_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(
        bakeEnd - bakeStart).count();
    if (progress) progress->DurationMs.store(durationMs, std::memory_order_relaxed);
    RS_LOG_INFO("MeshSDFBakerSparse: baked %u^3 @ b%u over %zu tris (occ=%u/%zu, %.1f ms)",
                resolution, brickSize, tris.size(),
                result.OccupiedBrickCount, indexCount, durationMs);
    return result;
}

} // namespace RS
