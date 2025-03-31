// Copyright (c) 2025 Pyarelal Knowles, MIT License
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Much of this file is based on code from this NVIDIA nvpro sample.
// https://github.com/nvpro-samples/vk_raytrace_displacement/blob/main/src/raytracing_vk.hpp
//
// This file provides RAII-inspired C++ objects to facilitate creating and using
// a VkAccelerationStructureKHR object. This better demonstrates object
// dependencies and order of operations with more compile-time validation.
// Once a rt::BuiltAS is created it should be valid and
// usable in a raytracing pipeline. To do so, create...
//
// - vko::as::Input (optional utility container)
//   - E.g. using: vko::as::createBlasInput() for triangle geometry
//   - E.g. using: vko::as::createTlasInput() for BLAS instances
// - vko::as::Sizes
// - vko::as::AS
//
// Finally, rt::BuiltAS provides a .object() and .address().
#pragma once

#include "vulkan/vulkan_core.h"
#include <assert.h>
#include <vko/array.hpp>
#include <vko/handles.hpp>

namespace vko {

namespace as {

// Optional utility object to group source data for building and updating
// acceleration structures.
struct Input {
    VkAccelerationStructureTypeKHR                        type;
    VkBuildAccelerationStructureFlagsKHR                  flags;
    std::vector<VkAccelerationStructureGeometryKHR>       geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
};

// VkAccelerationStructureBuildSizesInfoKHR wrapper, a dependency of the main
// AS.
class Sizes {
public:
    template <class DeviceAndCommands>
    Sizes(const DeviceAndCommands& device, const Input& input)
        : Sizes(device, input.type, input.flags, input.geometries, input.rangeInfos) {}
    template <class DeviceAndCommands>
    Sizes(const DeviceAndCommands& device, VkAccelerationStructureTypeKHR type,
          VkBuildAccelerationStructureFlagsKHR                      flags,
          std::span<const VkAccelerationStructureGeometryKHR>       geometries,
          std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos)
        : m_sizeInfo{
              .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
              .pNext = nullptr,
              .accelerationStructureSize = ~VkDeviceSize(0),
              .updateScratchSize         = ~VkDeviceSize(0),
              .buildScratchSize          = ~VkDeviceSize(0),
          } {
        assert(geometries.size() == rangeInfos.size());
        VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .pNext = nullptr,
            .type  = type,
            .flags = flags,
            .mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = VK_NULL_HANDLE,
            .dstAccelerationStructure = VK_NULL_HANDLE,
            .geometryCount            = static_cast<uint32_t>(geometries.size()),
            .pGeometries              = geometries.data(),
            .ppGeometries             = nullptr,
            .scratchData              = {},
        };
        std::vector<uint32_t> primitiveCounts(rangeInfos.size());
        std::transform(rangeInfos.begin(), rangeInfos.end(), primitiveCounts.begin(),
                       [](const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
                           return rangeInfo.primitiveCount;
                       });
        device.vkGetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo,
            primitiveCounts.data(), &m_sizeInfo);
    }
    const VkAccelerationStructureBuildSizesInfoKHR& operator*() const { return m_sizeInfo; }
    VkAccelerationStructureBuildSizesInfoKHR&       operator*() { return m_sizeInfo; }
    const VkAccelerationStructureBuildSizesInfoKHR* operator->() const { return &m_sizeInfo; }
    VkAccelerationStructureBuildSizesInfoKHR*       operator->() { return &m_sizeInfo; }

private:
    VkAccelerationStructureBuildSizesInfoKHR m_sizeInfo;
};

// VkAccelerationStructureKHR wrapper including a Buffer that holds backing
// memory for the vulkan object itself and the built acceleration structure.
// This can be a top or bottom level acceleration structure depending on the
// 'type' passed to the constructor. To use the acceleration structure it must
// first be given to BuiltAS.
class AS {
public:
    template <class DeviceAndCommands, class Allocator = vma::Allocator>
    AS(const DeviceAndCommands& device, VkAccelerationStructureTypeKHR type,
       const VkAccelerationStructureBuildSizesInfoKHR& size,
       VkAccelerationStructureCreateFlagsKHR flags, Allocator& allocator)
        : m_type(type)
        , m_size(size)
        , m_buffer(device, m_size.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator)
        , m_accelerationStructure(
              device, VkAccelerationStructureCreateInfoKHR{
                          .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                          .pNext         = nullptr,
                          .createFlags   = flags,
                          .buffer        = m_buffer,
                          .offset        = 0,
                          .size          = m_size.accelerationStructureSize,
                          .type          = m_type,
                          .deviceAddress = 0,
                      }) {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .pNext = nullptr,
            .accelerationStructure = m_accelerationStructure,
        };
        m_address = device.vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);
    }
    const VkAccelerationStructureTypeKHR&           type() const { return m_type; }
    const VkAccelerationStructureBuildSizesInfoKHR& sizes() const { return m_size; }
    VkAccelerationStructureKHR object() const { return m_accelerationStructure; }
    const VkDeviceAddress&     address() const { return m_address; }
    operator VkAccelerationStructureKHR() const { return m_accelerationStructure; }

private:
    VkAccelerationStructureTypeKHR           m_type;
    VkAccelerationStructureBuildSizesInfoKHR m_size;
    Array<std::byte>                         m_buffer;
    AccelerationStructureKHR                 m_accelerationStructure;
    VkDeviceAddress                          m_address;
};

template <class DeviceAndCommands>
void cmdBuild(const DeviceAndCommands& device, VkCommandBuffer cmd, const AS& accelerationStructure,
              const Input& input, bool update, Array<std::byte>& scratchBuffer) {
    cmdBuild(device, cmd, accelerationStructure, input.flags, input.geometries, input.rangeInfos,
             update, scratchBuffer);
}

template <class DeviceAndCommands>
void cmdBuild(const DeviceAndCommands& device, VkCommandBuffer cmd, const AS& accelerationStructure,
              VkBuildAccelerationStructureFlagsKHR                      flags,
              std::span<const VkAccelerationStructureGeometryKHR>       geometries,
              std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos, bool update,
              Array<std::byte>& scratchBuffer) {
    assert(geometries.size() == rangeInfos.size());
    assert(!update || !!(flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
    VkBuildAccelerationStructureModeKHR mode  = update
                                                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                                                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    auto&                               sizes = accelerationStructure.sizes();
    assert(scratchBuffer.size() >= (update ? sizes.updateScratchSize : sizes.buildScratchSize));
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = accelerationStructure.type(),
        .flags = flags,
        .mode  = mode,
        .srcAccelerationStructure = update ? accelerationStructure.object() : VK_NULL_HANDLE,
        .dstAccelerationStructure = accelerationStructure.object(),
        .geometryCount            = static_cast<uint32_t>(geometries.size()),
        .pGeometries              = geometries.data(),
        .ppGeometries             = nullptr,
        .scratchData              = {.deviceAddress = scratchBuffer.address(device)},
    };
    auto rangeInfosPtr = rangeInfos.data();
    device.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildGeometryInfo, &rangeInfosPtr);
}

// Optional utility call to fill a Input with instances for
// a top level acceleration structure build and update.
Input createTlasInput(uint32_t instanceCount, VkDeviceAddress instanceBufferAddress,
                      VkBuildAccelerationStructureFlagsKHR flags) {
    return Input{
        .type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = flags,
        .geometries{
            VkAccelerationStructureGeometryKHR{
                .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext        = nullptr,
                .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                .geometry =
                    VkAccelerationStructureGeometryDataKHR{
                        .instances =
                            VkAccelerationStructureGeometryInstancesDataKHR{
                                .sType =
                                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                .pNext           = nullptr,
                                .arrayOfPointers = VK_FALSE,
                                .data            = {instanceBufferAddress},
                            },
                    },
                .flags = 0,
            },
        },
        .rangeInfos{
            VkAccelerationStructureBuildRangeInfoKHR{
                .primitiveCount  = instanceCount,
                .primitiveOffset = 0,
                .firstVertex     = 0,
                .transformOffset = 0,
            },
        },
    };
}

// Essentially just VkAccelerationStructureGeometryTrianglesDataKHR with some
// defaults and geometryFlags.
struct SimpleGeometryInput {
    uint32_t           triangleCount;
    uint32_t           maxVertex;
    VkDeviceAddress    indexAddress;
    VkDeviceAddress    vertexAddress;
    VkDeviceSize       vertexStride = sizeof(float) * 3;
    VkFormat           vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    VkIndexType        indexType    = VK_INDEX_TYPE_UINT32;
    VkGeometryFlagsKHR geometryFlags =
        VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
};

// Optional utility call to fill a Input with triangle
// geometry for a bottom level acceleration structure build and update.
Input createBlasInput(std::span<const SimpleGeometryInput> simpleInputs,
                      VkBuildAccelerationStructureFlagsKHR accelerationStructureFlags) {
    Input result{
        .type       = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags      = accelerationStructureFlags,
        .geometries = {},
        .rangeInfos = {},
    };
    for (const auto& simpleInput : simpleInputs) {
        result.geometries.emplace_back(VkAccelerationStructureGeometryKHR{
            .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .pNext        = nullptr,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry =
                VkAccelerationStructureGeometryDataKHR{
                    .triangles =
                        VkAccelerationStructureGeometryTrianglesDataKHR{
                            .sType =
                                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                            .pNext         = nullptr,
                            .vertexFormat  = simpleInput.vertexFormat,
                            .vertexData    = {simpleInput.vertexAddress},
                            .vertexStride  = simpleInput.vertexStride,
                            .maxVertex     = simpleInput.maxVertex,
                            .indexType     = simpleInput.indexType,
                            .indexData     = {simpleInput.indexAddress},
                            .transformData = {0},
                        },
                },
            .flags = simpleInput.geometryFlags,
        });
        result.rangeInfos.emplace_back(VkAccelerationStructureBuildRangeInfoKHR{
            .primitiveCount  = simpleInput.triangleCount,
            .primitiveOffset = 0,
            .firstVertex     = 0,
            .transformOffset = 0,
        });
    }
    return result;
}

}; // namespace as

} // namespace vko
