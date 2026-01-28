// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/allocator.hpp>
#include <vko/handles.hpp>

namespace vko {

template <class Allocator = vma::Allocator>
class BoundImage {
public:
    using Allocation = typename Allocator::AllocationType;

    template <device_and_commands DeviceAndCommands, class AllocationCreateInfo>
    BoundImage(const DeviceAndCommands& vk, const VkImageCreateInfo& createInfo,
               const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : BoundImage(vk, vk, createInfo, allocationCreateInfo, allocator) {}

    template <device_commands DeviceCommands, class AllocationCreateInfo>
    BoundImage(const DeviceCommands& deviceCommands, VkDevice device,
               const VkImageCreateInfo&    createInfo,
               const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : m_image(deviceCommands, device, createInfo)
        , m_allocation(allocator.create(m_image, allocationCreateInfo))
        , m_extent(createInfo.extent)
        , m_mipLevels(createInfo.mipLevels)
        , m_arrayLayers(createInfo.arrayLayers) {}

    operator VkImage() const { return m_image; }
    const VkImage* ptr() const { return m_image.ptr(); }
    VkImage        object() const { return m_image; }
    VkExtent3D     extent() const { return m_extent; }
    uint32_t       mipLevels() const { return m_mipLevels; }
    uint32_t       arrayLayers() const { return m_arrayLayers; }

private:
    Image      m_image;
    Allocation m_allocation;
    VkExtent3D m_extent      = {0u, 0u, 0u};
    uint32_t   m_mipLevels   = 0u;
    uint32_t   m_arrayLayers = 0u;
    // Is a VkFormat required or should that be tracked separately? Tracking
    // size properties matches BoundBuffer/std::vector. The format would be
    // compile time for symmetry but that wouldn't fit typical use cases.
};

} // namespace vko
