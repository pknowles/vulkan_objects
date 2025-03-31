// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <span>
#include <vko/allocator.hpp>
#include <vko/handles.hpp>

namespace vko {

template <class T, class Allocator>
class BufferMapping {
public:
    using Allocation = typename Allocator::AllocationType;
    using Map        = typename Allocator::MapType;
    BufferMapping(const Allocation& allocation, VkDeviceSize elementCount)
        : m_map(allocation.map())
        , m_size(elementCount) {}
    T*     begin() const { return reinterpret_cast<T*>(m_map.data()); }
    T*     end() const { return reinterpret_cast<T*>(m_map.data()) + m_size; }
    T*     data() const { return reinterpret_cast<T*>(m_map.data()); }
    size_t size() const { return m_size; }
    T&     operator[](size_t index) const { return data()[index]; }

private:
    Map          m_map;
    VkDeviceSize m_size;
};

template <class T, class Allocator = vma::Allocator>
class BoundBuffer {
public:
    using Allocation = typename Allocator::AllocationType;
    template <class DeviceAndCommands, class AllocationCreateInfo>
    BoundBuffer(const DeviceAndCommands& device, VkDeviceSize elementCount,
                VkBufferUsageFlags usage, const AllocationCreateInfo& allocationCreateInfo,
                Allocator& allocator)
        : BoundBuffer(device, device, elementCount, nullptr, 0, usage, VK_SHARING_MODE_EXCLUSIVE,
                      {}, allocationCreateInfo, allocator) {}

    template <class DeviceCommands, class AllocationCreateInfo>
    BoundBuffer(const DeviceCommands& vk, VkDevice device, VkDeviceSize elementCount,
                const void* pNext, VkBufferCreateFlags flags, VkBufferUsageFlags usage,
                VkSharingMode sharingMode, std::span<const uint32_t> queueFamilyIndices,
                const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : m_size(elementCount)
        , m_buffer(vk, device,
                   VkBufferCreateInfo{
                       .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                       .pNext                 = pNext,
                       .flags                 = flags,
                       .size                  = sizeof(T) * elementCount,
                       .usage                 = usage,
                       .sharingMode           = sharingMode,
                       .queueFamilyIndexCount = uint32_t(queueFamilyIndices.size()),
                       .pQueueFamilyIndices =
                           queueFamilyIndices.empty() ? nullptr : queueFamilyIndices.data(),
                   })
        , m_allocation(allocator.create(m_buffer, allocationCreateInfo)) {}

    operator VkBuffer() const { return m_buffer; }
    const VkBuffer*             ptr() const { return m_buffer.ptr(); }
    VkDeviceSize                size() const { return m_size; }
    BufferMapping<T, Allocator> map() const { return {m_allocation, m_size}; }

    template <class DeviceAndCommands>
    VkDeviceAddress address(const DeviceAndCommands& device) const {
        VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                              .pNext = nullptr,
                                              .buffer = m_buffer};
        return device.vkGetBufferDeviceAddress(device, &addressInfo);
    }

private:
    VkDeviceSize m_size;
    Buffer       m_buffer;
    Allocation   m_allocation;
};

} // namespace vko
