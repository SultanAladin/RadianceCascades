// Source/Scene/SceneInternal.h — Phase 5
// Project-internal hooks for hosts that need to drive the Scene facade with a
// VulkanContext (mesh upload happens lazily during InscribeMesh). NOT exposed
// through Include/RS/ — only the standalone exe and future internal callers
// reach for this.
#pragma once

#include "RS/Scene.h"
#include "Core/VulkanContext.h"
#include "Scene/MeshRegistry.h"
#include "Scene/InstanceRegistry.h"

namespace RS {

// Bind the VulkanContext the Scene uses for buffer uploads. Must be called
// before any InscribeMesh/InstanceRegistry operation that touches Vulkan.
void SceneAttachVulkan(Scene& scene, const VulkanContext& ctx);

// Drop the Vulkan bindings + free all uploaded mesh buffers. Call before the
// VulkanContext itself shuts down.
void SceneDetachVulkan(Scene& scene);

// Read access into the internal registries — needed by the Phase 5 ForwardPass
// so it can iterate live instances + look up their meshes.
const MeshRegistry&     SceneMeshes   (const Scene& scene);
const InstanceRegistry& SceneInstances(const Scene& scene);

// Mutable access — Phase 9's Materials panel needs to call BindMaterial on
// behalf of the user without owning a separate InstanceRegistry. Same Scene&
// the caller already holds onto.
InstanceRegistry&       SceneInstancesMut(Scene& scene);

} // namespace RS
