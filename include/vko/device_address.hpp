// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <ostream>
#include <vulkan/vulkan_core.h>

namespace vko {

// Type-safe wrapper for VkDeviceAddress with element-based arithmetic.
// Works with any buffer type providing .address() and ValueType.
// T can be any type (only sizeof(T) is used).
template <class T>
class DeviceAddress {
public:
    using ValueType = T;
    DeviceAddress() = delete;
    
    template <class Buffer>
        requires requires(const Buffer& b) { 
            { b.address() } -> std::convertible_to<VkDeviceAddress>;
            typename Buffer::ValueType;
            requires std::same_as<typename Buffer::ValueType, T>;
        }
    DeviceAddress(const Buffer& buffer)
        : m_address(buffer.address()) {}
    
    template <class Buffer, class DeviceAndCommands>
        requires requires(const Buffer& b, const DeviceAndCommands& d) { 
            { b.address(d) } -> std::convertible_to<VkDeviceAddress>;
            typename Buffer::ValueType;
            requires std::same_as<typename Buffer::ValueType, T>;
        }
    DeviceAddress(const Buffer& buffer, const DeviceAndCommands& device)
        : m_address(buffer.address(device)) {}
    
    explicit DeviceAddress(VkDeviceAddress raw) : m_address(raw) {}
    
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

// Type-safe span for device memory - analogous to std::span
//
// Non-owning view of a contiguous region of device memory.
// Works with any buffer type providing .address(), .size() and ValueType.
template <class T>
class DeviceSpan {
public:
    using ValueType = T;
    using Address = DeviceAddress<T>;
    
    DeviceSpan() = default;
    
    DeviceSpan(DeviceAddress<T> address, VkDeviceSize elementCount)
        : m_address(address), m_size(elementCount) {}
    
    template <class Buffer>
        requires requires(const Buffer& b) { 
            { b.address() } -> std::convertible_to<VkDeviceAddress>;
            { b.size() } -> std::convertible_to<VkDeviceSize>;
            typename Buffer::ValueType;
            requires std::same_as<typename Buffer::ValueType, T>;
        }
    DeviceSpan(const Buffer& buffer)
        : m_address(buffer), m_size(buffer.size()) {}
    
    template <class Buffer, class DeviceAndCommands>
        requires requires(const Buffer& b, const DeviceAndCommands& d) { 
            { b.address(d) } -> std::convertible_to<VkDeviceAddress>;
            { b.size() } -> std::convertible_to<VkDeviceSize>;
            typename Buffer::ValueType;
            requires std::same_as<typename Buffer::ValueType, T>;
        }
    DeviceSpan(const Buffer& buffer, const DeviceAndCommands& device)
        : m_address(buffer, device), m_size(buffer.size()) {}
    
    DeviceAddress<T> data() const { return m_address; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceSize sizeBytes() const { return m_size * sizeof(T); }
    bool empty() const { return m_size == 0; }
    
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
    VkDeviceSize m_size = 0;
};

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

} // namespace vko
