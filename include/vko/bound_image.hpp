// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <span>
#include <vko/allocator.hpp>
#include <vko/handles.hpp>

namespace vko {

template <class Allocator = vma::Allocator>
class BoundImage {
public:
    using Allocation = typename Allocator::AllocationType;

    template <class DeviceAndCommands, class AllocationCreateInfo>
    BoundImage(const DeviceAndCommands& vk, const VkImageCreateInfo& createInfo,
               const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : BoundImage(vk, vk, createInfo, allocationCreateInfo, allocator) {}

    template <class DeviceCommands, class AllocationCreateInfo>
    BoundImage(const DeviceCommands& deviceCommands, VkDevice device,
               const VkImageCreateInfo&    createInfo,
               const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : m_image(deviceCommands, device, createInfo)
        , m_allocation(allocator.create(m_image, allocationCreateInfo)) {}

    operator VkImage() const { return m_image; }
    const VkImage* ptr() const { return m_image.ptr(); }
    VkImage        object() const { return m_image; }

private:
    Image      m_image;
    Allocation m_allocation;
};

} // namespace vko
