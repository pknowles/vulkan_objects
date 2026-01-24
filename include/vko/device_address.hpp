// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <ostream>
#include <vulkan/vulkan_core.h>

namespace vko {

template <class T>
concept buffer = requires(T t) {
    { static_cast<VkBuffer>(t) } -> std::convertible_to<VkBuffer>;
} && std::is_trivially_destructible_v<typename T::ValueType>;

template <class Container>
using container_view_t =
    std::conditional_t<std::is_const_v<std::remove_reference<Container>>,
                       const typename std::remove_reference_t<Container>::ValueType,
                       typename std::remove_reference_t<Container>::ValueType>;

// Typed wrapper for VkDeviceAddress with element-based arithmetic. Works with
// any buffer type providing .address() and ValueType. This might be used for
// shader interop, where a uniform buffer or push constant can declare a
// DeviceAddress<T> and the C++ compiler will verify the user provides an
// address of the correct type.
template <class T>
class DeviceAddress {
public:
    using ValueType                     = T;
    DeviceAddress()                     = delete;
    DeviceAddress(const DeviceAddress&) = default;
    DeviceAddress(DeviceAddress&&)      = default;

    template <buffer Buffer>
        requires std::same_as<container_view_t<Buffer>, T> && requires(const Buffer& b) {
            { b.address() } -> std::convertible_to<VkDeviceAddress>;
        }
    explicit DeviceAddress(const Buffer& buffer)
        : m_address(buffer.address()) {}

    template <buffer Buffer, class DeviceAndCommands>
        requires std::same_as<container_view_t<Buffer>, T> &&
                 requires(const Buffer& b, const DeviceAndCommands& d) {
                     { b.address(d) } -> std::convertible_to<VkDeviceAddress>;
                 }
    explicit DeviceAddress(const Buffer& buffer, const DeviceAndCommands& device)
        : m_address(buffer.address(device)) {}

    // Non-type-safe constructor from raw address
    explicit DeviceAddress(VkDeviceAddress raw)
        : m_address(raw) {}

    // Implicit const conversion
    DeviceAddress(const DeviceAddress<std::remove_const_t<T>>& other)
        requires std::is_const_v<T>
        : m_address(other.raw()) {}

    VkDeviceAddress raw() const { return m_address; }

    int64_t operator-(const DeviceAddress& other) const {
        int64_t diff = static_cast<int64_t>(m_address) - static_cast<int64_t>(other.m_address);
        assert(diff % sizeof(T) == 0 && "Address difference not aligned to element size");
        return diff / sizeof(T);
    }

    DeviceAddress operator+(int64_t elementOffset) const {
        return DeviceAddress(m_address + elementOffset * sizeof(T));
    }

    DeviceAddress& operator+=(int64_t elementOffset) {
        m_address += elementOffset * sizeof(T);
        return *this;
    }

    DeviceAddress operator-(int64_t elementOffset) const {
        return DeviceAddress(m_address - elementOffset * sizeof(T));
    }

    DeviceAddress& operator-=(int64_t elementOffset) {
        m_address -= elementOffset * sizeof(T);
        return *this;
    }

    auto operator<=>(const DeviceAddress&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const DeviceAddress& addr) {
        return os << std::hex << addr.m_address << std::dec;
    }

private:
    VkDeviceAddress m_address = std::numeric_limits<VkDeviceAddress>::max();
};

// Reinterpret cast for device addresses
// Similar to reinterpret_cast for pointers but for device addresses.
// Only allowed when types have the same size.
template <class T, class U>
    requires(sizeof(T) == sizeof(U))
DeviceAddress<T> reinterpretCast(DeviceAddress<U> address) {
    return DeviceAddress<T>(address.raw());
}

// Typed span for device memory - analogous to std::span
//
// Non-owning view of a contiguous region of device memory. Works with any
// buffer type providing .address(), .size() and ValueType.
template <class T>
class DeviceSpan {
public:
    using ValueType = T;
    using Address   = DeviceAddress<T>;

    DeviceSpan()                  = default;
    DeviceSpan(const DeviceSpan&) = default;
    DeviceSpan(DeviceSpan&&)      = default;

    DeviceSpan(DeviceAddress<T> address, VkDeviceSize elementCount)
        : m_address(address)
        , m_size(elementCount) {}

    template <buffer Buffer>
        requires std::same_as<container_view_t<Buffer>, T>
    explicit DeviceSpan(const Buffer& buffer)
        : m_address(buffer)
        , m_size(buffer.size()) {}

    template <class Buffer, class DeviceAndCommands>
        requires requires(const Buffer& b, const DeviceAndCommands& d) {
            { b.address(d) } -> std::convertible_to<VkDeviceAddress>;
            { b.size() } -> std::convertible_to<VkDeviceSize>;
            typename Buffer::ValueType;
            requires std::same_as<typename Buffer::ValueType, T>;
        }
    explicit DeviceSpan(const Buffer& buffer, const DeviceAndCommands& device)
        : m_address(buffer, device)
        , m_size(buffer.size()) {}

    // Implicit const conversion
    DeviceSpan(const DeviceSpan<std::remove_const_t<T>>& other)
        requires std::is_const_v<T>
        : m_address(other.data())
        , m_size(other.size()) {}

    DeviceAddress<T> data() const { return m_address; }
    VkDeviceSize     size() const { return m_size; }
    VkDeviceSize     sizeBytes() const { return m_size * sizeof(T); }
    bool             empty() const { return m_size == 0; }

    DeviceSpan subspan(VkDeviceSize offset, VkDeviceSize count) const {
        assert(offset <= m_size && "DeviceSpan::subspan offset out of bounds");
        assert(offset + count <= m_size && "DeviceSpan::subspan range out of bounds");
        return DeviceSpan(m_address + offset, count);
    }

    DeviceSpan subspan(VkDeviceSize offset) const {
        assert(offset <= m_size && "DeviceSpan::subspan offset out of bounds");
        return DeviceSpan(m_address + offset, m_size - offset);
    }

private:
    DeviceAddress<T> m_address = DeviceAddress<T>(VkDeviceAddress(0));
    VkDeviceSize     m_size    = 0;
};

// Typed wrapper for VkBuffer + byte offset with element-based arithmetic.
// Analogous to DeviceAddress but for buffers without device addresses. This
// models a pattern in Vulkan where APIs take a buffer and an offset rather than
// an absolute address. For example vkCmdCopyBuffer or vkCmdBindVertexBuffers.
//
// [rant...] Note that a limitation with the whole buffer+offset approach that
// Vulkan took is once you convert it to an address you can't go back and get
// the buffer+offset againt, which makes viewing structures in GPU memory
// impossible without workarounds like VK_NV_copy_memory_indirect to inspect
// memory directly.
template <class T>
class BufferAddress {
public:
    using ValueType                     = T;
    BufferAddress()                     = delete;
    BufferAddress(const BufferAddress&) = default;
    BufferAddress(BufferAddress&&)      = default;

    template <buffer Buffer>
        requires std::same_as<container_view_t<Buffer>, T>
    BufferAddress(const Buffer& buffer, VkDeviceSize byteOffset = 0)
        : m_buffer(static_cast<VkBuffer>(buffer))
        , m_byteOffset(byteOffset) {}

    // Non-type-safe constructor from raw vulkan buffer and byte offset
    explicit BufferAddress(VkBuffer buffer, VkDeviceSize byteOffset = 0)
        : m_buffer(buffer)
        , m_byteOffset(byteOffset) {}

    // Implicit const conversion
    BufferAddress(const BufferAddress<std::remove_const_t<T>>& other)
        requires std::is_const_v<T>
        : m_buffer(other.buffer())
        , m_byteOffset(other.byteOffset()) {}

    VkBuffer     buffer() const { return m_buffer; }
    VkDeviceSize byteOffset() const { return m_byteOffset; }

    // Shortcut for converting to an address. If you have a vko::DeviceBuffer,
    // better to use that directly.
    template <device_and_commands DeviceAndCommands>
    DeviceAddress<T> address(const DeviceAndCommands& device) const {
        VkBufferDeviceAddressInfo addressInfo{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext  = nullptr,
            .buffer = m_buffer,
        };
        auto address = device.vkGetBufferDeviceAddress(device, &addressInfo);
        return DeviceAddress<T>(address + m_byteOffset);
    }

    int64_t operator-(const BufferAddress& other) const {
        assert(m_buffer == other.m_buffer &&
               "Cannot subtract BufferAddresses from different buffers");
        int64_t diff =
            static_cast<int64_t>(m_byteOffset) - static_cast<int64_t>(other.m_byteOffset);
        assert(diff % sizeof(T) == 0 && "Offset difference not aligned to element size");
        return diff / sizeof(T);
    }

    BufferAddress operator+(int64_t elementOffset) const {
        return BufferAddress(m_buffer, m_byteOffset + elementOffset * sizeof(T));
    }

    BufferAddress& operator+=(int64_t elementOffset) {
        m_byteOffset += elementOffset * sizeof(T);
        return *this;
    }

    BufferAddress operator-(int64_t elementOffset) const {
        return BufferAddress(m_buffer, m_byteOffset - elementOffset * sizeof(T));
    }

    BufferAddress& operator-=(int64_t elementOffset) {
        m_byteOffset -= elementOffset * sizeof(T);
        return *this;
    }

    auto operator<=>(const BufferAddress&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const BufferAddress& addr) {
        return os << "VkBuffer(" << addr.m_buffer << ")+0x" << std::hex << addr.m_byteOffset
                  << std::dec;
    }

private:
    VkBuffer     m_buffer     = VK_NULL_HANDLE;
    VkDeviceSize m_byteOffset = std::numeric_limits<VkDeviceSize>::max();
};

// Reinterpret cast for buffer addresses
// Similar to reinterpret_cast for pointers but for buffer addresses.
// Only allowed when types have the same size.
template <class T, class U>
    requires(sizeof(T) == sizeof(U))
BufferAddress<T> reinterpretCast(BufferAddress<U> address) {
    return BufferAddress<T>(address.buffer(), address.byteOffset());
}

// Typed span for buffer memory - analogous to std::span
//
// Non-owning view of a contiguous region of buffer memory. Works with any
// buffer type providing VkBuffer conversion, .size() and ValueType.
template <class T>
class BufferSpan {
public:
    using ValueType = T;
    using Address   = BufferAddress<T>;

    BufferSpan()                  = default;
    BufferSpan(const BufferSpan&) = default;
    BufferSpan(BufferSpan&&)      = default;

    BufferSpan(BufferAddress<T> address, VkDeviceSize elementCount)
        : m_address(address)
        , m_size(elementCount) {}

    template <class Buffer>
        requires std::is_lvalue_reference_v<Buffer> && buffer<std::remove_reference_t<Buffer>> &&
                     std::same_as<container_view_t<Buffer>, T>
    BufferSpan(Buffer&& buffer)
        : m_address(buffer)
        , m_size(buffer.size()) {}

    // Implicit const conversion
    BufferSpan(const BufferSpan<std::remove_const_t<T>>& other)
        requires std::is_const_v<T>
        : m_address(other.data())
        , m_size(other.size()) {}

    BufferAddress<T> data() const { return m_address; }
    VkBuffer         buffer() const { return m_address.buffer(); }
    VkDeviceSize     offset() const { return m_address.byteOffset(); }
    VkDeviceSize     size() const { return m_size; }
    VkDeviceSize     sizeBytes() const { return m_size * sizeof(T); }
    bool             empty() const { return m_size == 0; }

    BufferSpan subspan(VkDeviceSize offset, VkDeviceSize count) const {
        assert(offset <= m_size && "BufferSpan::subspan offset out of bounds");
        assert(offset + count <= m_size && "BufferSpan::subspan range out of bounds");
        return BufferSpan(m_address + offset, count);
    }

    BufferSpan subspan(VkDeviceSize offset) const {
        assert(offset <= m_size && "BufferSpan::subspan offset out of bounds");
        return BufferSpan(m_address + offset, m_size - offset);
    }

private:
    BufferAddress<T> m_address = BufferAddress<T>(VK_NULL_HANDLE, 0);
    VkDeviceSize     m_size    = 0;
};

// Deduction guide: always deduce const ValueType for read-only spans. Honestly
// doesn't make much difference because we can't directly read the memory, but
// doesn't hurt to be accurate.
template <class Buffer>
BufferSpan(Buffer&&) -> BufferSpan<container_view_t<Buffer>>;

// Possible helpers (currently disabled), not sure if we want these
#if 0
template <class T>
T* translateOffset(DeviceAddress<T> offsetAddress, void* hostBase) {
    if (!hostBase)
        return nullptr;
    return reinterpret_cast<T*>(static_cast<std::byte*>(hostBase) + offsetAddress.raw());
}

template <class T>
DeviceAddress<T> translateOffset(DeviceAddress<T> offsetAddress, VkDeviceAddress deviceBase) {
    if (deviceBase == 0)
        return DeviceAddress<T>(0);
    return DeviceAddress<T>(offsetAddress.raw() + deviceBase);
}

template <class T>
DeviceAddress<T> translateOffset(DeviceAddress<T> offsetAddress, DeviceAddress<T> deviceBase) {
    return translateOffset(offsetAddress, deviceBase.raw());
}
#endif

// Type-safe and bounds-checked vkCmdCopyBuffer using BufferSpan.
// Usage: copyBuffer(device, cmd, BufferSpan(srcBuf).subspan(10, 5),
//                                BufferSpan(dstBuf).subspan(20, 5));
template <device_and_commands DeviceAndCommands, class SrcT, class DstT>
    requires std::is_trivially_assignable_v<DstT&, SrcT>
void copyBuffer(const DeviceAndCommands& device, VkCommandBuffer cmd, const BufferSpan<SrcT>& src,
                const BufferSpan<DstT>& dst) {
    if (src.size() != dst.size())
        throw std::out_of_range("source and destination buffer spans must have the same size");
    if (src.size() == 0)
        return;

    VkBufferCopy region{
        .srcOffset = src.offset(), .dstOffset = dst.offset(), .size = src.sizeBytes()};
    device.vkCmdCopyBuffer(cmd, src.buffer(), dst.buffer(), 1, &region);
}

} // namespace vko
