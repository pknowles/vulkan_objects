// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <assert.h>
#include <limits>
#include <span>
#include <vko/adapters.hpp>
#include <vko/allocator.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>

namespace vko {

namespace simple {

template <class InstanceCommands>
VkPhysicalDeviceRayTracingPipelinePropertiesKHR
rayTracingPipelineProperties(const InstanceCommands& vk, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR result{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        .pNext = nullptr,
        .shaderGroupHandleSize              = std::numeric_limits<uint32_t>::max(),
        .maxRayRecursionDepth               = std::numeric_limits<uint32_t>::max(),
        .maxShaderGroupStride               = std::numeric_limits<uint32_t>::max(),
        .shaderGroupBaseAlignment           = std::numeric_limits<uint32_t>::max(),
        .shaderGroupHandleCaptureReplaySize = std::numeric_limits<uint32_t>::max(),
        .maxRayDispatchInvocationCount      = std::numeric_limits<uint32_t>::max(),
        .shaderGroupHandleAlignment         = std::numeric_limits<uint32_t>::max(),
        .maxRayHitAttributeSize             = std::numeric_limits<uint32_t>::max(),
    };
    VkPhysicalDeviceProperties2 prop2{.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                      .pNext      = &result,
                                      .properties = {}};
    vk.vkGetPhysicalDeviceProperties2(physicalDevice, &prop2);
    return result;
}

class HitGroupHandles {
public:
    template <device_and_commands DeviceAndCommands>
    HitGroupHandles(
        const DeviceAndCommands&                               device,
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties,
        const VkPipeline& pipeline, size_t groupCount)
        : m_handleSize(rayTracingPipelineProperties.shaderGroupHandleSize) {
        m_handles.resize(m_handleSize * groupCount);
        check(device.vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount,
                                                          m_handles.size(), m_handles.data()));
    }

    std::span<const std::byte> operator[](size_t index) const {
        return {m_handles.data() + index * m_handleSize, m_handleSize};
    }

    class Iterator {
    public:
        Iterator(const HitGroupHandles* parent, size_t index)
            : m_parent(parent)
            , m_index(index) {}
        bool      operator!=(const Iterator& other) const { return m_index != other.m_index; }
        Iterator& operator++() {
            ++m_index;
            return *this;
        }
        std::span<const std::byte> operator*() { return (*m_parent)[m_index]; }

    private:
        const HitGroupHandles* m_parent;
        size_t                 m_index;
    };

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, size()); }
    size_t   size() const { return m_handles.size() / m_handleSize; }

private:
    std::vector<std::byte> m_handles;
    size_t                 m_handleSize;
};

struct StridedOffsetRegion {
    VkDeviceSize                    offset;
    VkDeviceSize                    stride;
    VkDeviceSize                    size;
    VkStridedDeviceAddressRegionKHR atAddress(const VkDeviceAddress address) const {
        return VkStridedDeviceAddressRegionKHR{
            .deviceAddress = address + offset,
            .stride        = stride,
            .size          = size,
        };
    }
};

// Assumes HitGroupHandleRange are ranges of std::span<const std::byte>, each
// exactly shaderGroupHandleSize bytes long
template <class Allocator = vma::Allocator>
struct ShaderBindingTablesStaging {
    template <device_and_commands DeviceAndCommands, std::ranges::input_range HitGroupHandleRange>
    ShaderBindingTablesStaging(
        Allocator& allocator, DeviceAndCommands& device,
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties,
        HitGroupHandleRange&& raygenTable, HitGroupHandleRange&& missTable,
        HitGroupHandleRange&& hitTable, HitGroupHandleRange&& callableTable)
        : tables(device,
                 (std::ranges::size(raygenTable) + std::ranges::size(missTable) +
                  std::ranges::size(hitTable) + std::ranges::size(callableTable)) *
                         rayTracingPipelineProperties.shaderGroupHandleSize +
                     3 * rayTracingPipelineProperties.shaderGroupBaseAlignment,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 allocator) {
        assert(rayTracingPipelineProperties.shaderGroupBaseAlignment >=
               rayTracingPipelineProperties.shaderGroupHandleAlignment);
        auto tablesData = tables.map();
        auto next       = tablesData.begin();
        auto alignCopy  = [handleSize = rayTracingPipelineProperties.shaderGroupHandleSize,
                          align      = rayTracingPipelineProperties.shaderGroupBaseAlignment,
                          begin      = next](const auto& source, auto next,
                                        std::optional<StridedOffsetRegion>& storeOffset) {
            auto offset        = std::distance(begin, next);
            auto offsetAligned = (offset + align - 1) & ~(align - 1);
            next               = next + (offsetAligned - offset);
            auto before        = next;
            for (std::span<const std::byte> handle : source) {
                assert(handle.size() == handleSize);
                next = std::ranges::copy(handle, next).out;
            }
            storeOffset = StridedOffsetRegion{
                 .offset = VkDeviceSize(std::distance(begin, before)),
                 .stride = handleSize,
                 .size   = VkDeviceSize(std::distance(before, next)),
            };
            return next;
        };
        if (!std::ranges::empty(raygenTable))
            next = alignCopy(raygenTable, next, raygenTableOffset);
        if (!std::ranges::empty(missTable))
            next = alignCopy(missTable, next, missTableOffset);
        if (!std::ranges::empty(hitTable))
            next = alignCopy(hitTable, next, hitTableOffset);
        if (!std::ranges::empty(callableTable))
            next = alignCopy(callableTable, next, callableTableOffset);
        assert(size_t(std::distance(tablesData.begin(), next)) <= tablesData.size());
        assert(raygenTableOffset);

        // VUID-vkCmdTraceRaysKHR-size-04023 should always be a raygen shader
        assert(raygenTableOffset->stride == raygenTableOffset->size);
    }

    BoundBuffer<std::byte>             tables;
    std::optional<StridedOffsetRegion> raygenTableOffset;
    std::optional<StridedOffsetRegion> missTableOffset;
    std::optional<StridedOffsetRegion> hitTableOffset;
    std::optional<StridedOffsetRegion> callableTableOffset;
};

template <class Allocator = vma::Allocator>
struct ShaderBindingTables {
    template <device_and_commands DeviceAndCommands, class StagingAllocator = vma::Allocator>
    ShaderBindingTables(const DeviceAndCommands& device, VkCommandPool pool, VkQueue queue,
                        ShaderBindingTablesStaging<StagingAllocator>&& staging,
                        Allocator&                                     allocator)
        : tables(device, staging.tables.size(),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator)
        , raygenTableOffset(staging.raygenTableOffset
                                ? staging.raygenTableOffset->atAddress(tables.address(device))
                                : VkStridedDeviceAddressRegionKHR{})
        , missTableOffset(staging.missTableOffset
                              ? staging.missTableOffset->atAddress(tables.address(device))
                              : VkStridedDeviceAddressRegionKHR{})
        , hitTableOffset(staging.hitTableOffset
                             ? staging.hitTableOffset->atAddress(tables.address(device))
                             : VkStridedDeviceAddressRegionKHR{})
        , callableTableOffset(staging.callableTableOffset
                                  ? staging.callableTableOffset->atAddress(tables.address(device))
                                  : VkStridedDeviceAddressRegionKHR{}) {
        vko::simple::ImmediateCommandBuffer cmd(device, pool, queue);
        VkBufferCopy                        bufferCopy{
                                   .srcOffset = 0,
                                   .dstOffset = 0,
                                   .size      = staging.tables.size(),
        };
        device.vkCmdCopyBuffer(cmd, staging.tables, tables, 1, &bufferCopy);
    }
    BoundBuffer<std::byte>          tables;
    VkStridedDeviceAddressRegionKHR raygenTableOffset;
    VkStridedDeviceAddressRegionKHR missTableOffset;
    VkStridedDeviceAddressRegionKHR hitTableOffset;
    VkStridedDeviceAddressRegionKHR callableTableOffset;
};

template <class PushConstants, size_t MaxRecursionDepth>
class RayTracingPipeline {
public:
    template <device_and_commands DeviceAndCommands>
    RayTracingPipeline(const DeviceAndCommands&               device,
                       std::span<const VkDescriptorSetLayout> descriptorSetLayouts,
                       VkShaderModule raygen, VkShaderModule anyHit, VkShaderModule closestHit,
                       VkShaderModule miss)
        : RayTracingPipeline(
              device, descriptorSetLayouts,
              std::to_array({
                  VkPipelineShaderStageCreateInfo{
                      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .pNext               = nullptr,
                      .flags               = 0,
                      .stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                      .module              = raygen,
                      .pName               = "main",
                      .pSpecializationInfo = nullptr,
                  },
                  VkPipelineShaderStageCreateInfo{
                      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .pNext               = nullptr,
                      .flags               = 0,
                      .stage               = VK_SHADER_STAGE_MISS_BIT_KHR,
                      .module              = miss,
                      .pName               = "main",
                      .pSpecializationInfo = nullptr,
                  },
                  VkPipelineShaderStageCreateInfo{
                      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .pNext               = nullptr,
                      .flags               = 0,
                      .stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                      .module              = closestHit,
                      .pName               = "main",
                      .pSpecializationInfo = nullptr,
                  },
                  VkPipelineShaderStageCreateInfo{
                      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .pNext               = nullptr,
                      .flags               = 0,
                      .stage               = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                      .module              = anyHit,
                      .pName               = "main",
                      .pSpecializationInfo = nullptr,
                  },
              }),
              std::to_array({
                  VkRayTracingShaderGroupCreateInfoKHR{
                      .sType         = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                      .pNext         = nullptr,
                      .type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                      .generalShader = 0 /* "raygen" shaderStages[0] */,
                      .closestHitShader                = VK_SHADER_UNUSED_KHR,
                      .anyHitShader                    = VK_SHADER_UNUSED_KHR,
                      .intersectionShader              = VK_SHADER_UNUSED_KHR,
                      .pShaderGroupCaptureReplayHandle = nullptr,
                  },
                  VkRayTracingShaderGroupCreateInfoKHR{
                      .sType         = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                      .pNext         = nullptr,
                      .type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                      .generalShader = 1 /* "miss" shaderStages[1] */,
                      .closestHitShader                = VK_SHADER_UNUSED_KHR,
                      .anyHitShader                    = VK_SHADER_UNUSED_KHR,
                      .intersectionShader              = VK_SHADER_UNUSED_KHR,
                      .pShaderGroupCaptureReplayHandle = nullptr,
                  },
                  VkRayTracingShaderGroupCreateInfoKHR{
                      .sType         = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                      .pNext         = nullptr,
                      .type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                      .generalShader = VK_SHADER_UNUSED_KHR,
                      .closestHitShader                = 2 /* "any hit" shaderStages[2] */,
                      .anyHitShader                    = 3 /* "closest hit" shaderStages[3] */,
                      .intersectionShader              = VK_SHADER_UNUSED_KHR,
                      .pShaderGroupCaptureReplayHandle = nullptr,
                  },
              })) {}

    template <device_and_commands DeviceAndCommands>
    RayTracingPipeline(const DeviceAndCommands&                              device,
                       std::span<const VkDescriptorSetLayout>                descriptorSetLayouts,
                       std::span<const VkPipelineShaderStageCreateInfo>      stages,
                       std::span<const VkRayTracingShaderGroupCreateInfoKHR> groups)
        : m_pipelineLayout(device,
                           VkPipelineLayoutCreateInfo{
                               .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                               .pNext          = nullptr,
                               .flags          = 0,
                               .setLayoutCount = uint32_t(descriptorSetLayouts.size()),
                               .pSetLayouts    = descriptorSetLayouts.data(),
                               .pushConstantRangeCount = 1,
                               .pPushConstantRanges    = tmpPtr(VkPushConstantRange{
                                   VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants)}),
                           })
        , m_pipeline(device, VkRayTracingPipelineCreateInfoKHR{
                                 .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                                 .pNext = nullptr,
                                 .flags = 0,
                                 .stageCount = static_cast<uint32_t>(stages.size()),
                                 .pStages    = stages.data(),
                                 .groupCount = static_cast<uint32_t>(groups.size()),
                                 .pGroups    = groups.data(),
                                 .maxPipelineRayRecursionDepth = MaxRecursionDepth,
                                 .pLibraryInfo                 = nullptr,
                                 .pLibraryInterface            = nullptr,
                                 .pDynamicState                = nullptr,
                                 .layout                       = m_pipelineLayout,
                                 .basePipelineHandle           = VK_NULL_HANDLE,
                                 .basePipelineIndex            = 0,
                             }) {}

    VkPipelineLayout layout() const { return m_pipelineLayout; }
    operator VkPipeline() const { return m_pipeline; }

private:
    PipelineLayout        m_pipelineLayout;
    RayTracingPipelineKHR m_pipeline;
};

} // namespace simple

} // namespace vko