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
    std::span<T> span() const { return std::span{data(), size()}; }

private:
    Map          m_map;
    VkDeviceSize m_size;
};

// It's common to download and inspect buffers when debugging and annoying to
// have to set the TRANSFER_SRC bit manually.
// TODO: ideally move to user layer
inline constexpr VkBufferUsageFlags debugUsageFlags() {
#if !defined(NDEBUG)
    return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
#else
    return 0;
#endif
}

template <class T, class Allocator = vma::Allocator>
class BoundBuffer {
public:
    using Allocation = typename Allocator::AllocationType;
    template <device_and_commands DeviceAndCommands, class AllocationCreateInfo>
    BoundBuffer(const DeviceAndCommands& device, VkDeviceSize elementCount,
                VkBufferUsageFlags usage, const AllocationCreateInfo& allocationCreateInfo,
                Allocator& allocator)
        : BoundBuffer(device, device, elementCount, nullptr, 0, usage, VK_SHARING_MODE_EXCLUSIVE,
                      {}, allocationCreateInfo, allocator) {}

    template <device_commands DeviceCommands, class AllocationCreateInfo>
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
                       .usage                 = usage | debugUsageFlags(),
                       .sharingMode           = sharingMode,
                       .queueFamilyIndexCount = uint32_t(queueFamilyIndices.size()),
                       .pQueueFamilyIndices =
                           queueFamilyIndices.empty() ? nullptr : queueFamilyIndices.data(),
                   })
        , m_allocation(allocator.create(m_buffer, allocationCreateInfo)) {}

    operator VkBuffer() const { return m_buffer; }
    const VkBuffer*             ptr() const { return m_buffer.ptr(); }
    VkBuffer                    object() const { return m_buffer.object(); }
    VkDeviceSize                size() const { return m_size; }
    VkDeviceSize                sizeBytes() const { return m_size * sizeof(T); }
    BufferMapping<T, Allocator> map() const { return {m_allocation, m_size}; }

    template <device_and_commands DeviceAndCommands>
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

// Wrapper to cache the device address. Assumes VK_KHR_buffer_device_address is
// available (core in vulkan 1.2).
template <class T, class Allocator = vma::Allocator>
class DeviceBuffer {
public:
    template <device_and_commands DeviceAndCommands, class AllocationCreateInfo>
    DeviceBuffer(const DeviceAndCommands& device, VkDeviceSize elementCount,
                 VkBufferUsageFlags usage, const AllocationCreateInfo& allocationCreateInfo,
                 Allocator& allocator)
        : DeviceBuffer(device, device, elementCount, nullptr, 0, usage, VK_SHARING_MODE_EXCLUSIVE,
                       {}, allocationCreateInfo, allocator) {}

    template <device_commands DeviceCommands, class AllocationCreateInfo>
    DeviceBuffer(const DeviceCommands& vk, VkDevice device, VkDeviceSize elementCount,
                 const void* pNext, VkBufferCreateFlags flags, VkBufferUsageFlags usage,
                 VkSharingMode sharingMode, std::span<const uint32_t> queueFamilyIndices,
                 const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : m_buffer(vk, device, elementCount, pNext, flags,
                   usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | debugUsageFlags(),
                   sharingMode, queueFamilyIndices,
                   allocationCreateInfo | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator)
        , m_address(vk.vkGetBufferDeviceAddress(
              device, vko::tmpPtr(VkBufferDeviceAddressInfo{
                          .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                          .pNext  = nullptr,
                          .buffer = m_buffer}))) {}
    operator VkBuffer() const { return m_buffer; }
    const VkBuffer*             ptr() const { return m_buffer.ptr(); }
    VkBuffer                    object() const { return m_buffer.object(); }
    VkDeviceSize                size() const { return m_buffer.size(); }
    VkDeviceSize                sizeBytes() const { return m_buffer.sizeBytes(); }
    BufferMapping<T, Allocator> map() const { return m_buffer.map(); }
    VkDeviceAddress             address() const { return m_address; }

private:
    // member, not parent, to avoid accidental cast (classes are not virtual)
    BoundBuffer<T, Allocator> m_buffer;
    VkDeviceAddress           m_address;
};

} // namespace vko
