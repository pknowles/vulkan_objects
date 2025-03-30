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

    template <class AllocationCreateInfo, class DeviceCommands>
    BoundImage(Allocator& allocator, const VkImageCreateInfo& createInfo, VkDevice device,
               const AllocationCreateInfo& allocationCreateInfo,
               const DeviceCommands&       deviceCommands)
        : m_image(createInfo, device, deviceCommands)
        , m_allocation(allocator.create(m_image, allocationCreateInfo)) {}

    operator VkImage() const { return m_image; }
    const VkImage* ptr() const { return m_image.ptr(); }

private:
    ImageOnly  m_image;
    Allocation m_allocation;
};

} // namespace vko
