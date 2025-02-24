// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <string_view>
#include <vko/adaptors.hpp>
#include <vko/exceptions.hpp>
#include <vko/functions.hpp>
#include <vulkan/vulkan_core.h>

namespace vko
{

template<class Handle> struct handle_traits;

template <class T, class CreateFunc, class CreateInfo> struct CreateHandle;

// Argument pack for destroying a handle. Similar to an object keeping a
// reference to its allocator. std::function could work too, but I have a
// premature optimization hunch this could be cheaper (naturally, untested).
template <class T>
struct DestroyFunc{
    // In most cases the parent is the first parameter of the destroy function
    using Parent = handle_traits<T>::destroy_first_param;

    DestroyFunc(const handle_traits<T>::table& table, T, Parent parent)
        : destroy(table.handle_traits<T>::table_destroy),
          parent(parent) {}
    void operator()(T handle) const { destroy(parent, handle, nullptr); }

    // Copy the function pointer rather than take a reference to the function
    // table itself and risk a dangling pointer
    handle_traits<T>::destroy_t destroy;

    // The owning object. This could be a weak_ptr to be extra safe, but the
    // overhead makes it a step too far.
    Parent parent;
};

template <class T, class CreateFunc, class CreateInfo>
class Handle {
public:
    #if 0
    template <class Functions, class... Args>
    Handle(const Functions& functions, const CreateInfo& createInfo, const Args&... args)
        : m_handle(CreateHandle<T, CreateFunc, CreateInfo, Functions, Args...>()(
              functions, createInfo, args...)),
          m_destroy(
              DestroyFunc<T>(functions, m_handle /* sigh; needed for instance and device */, args...)) {}
    #endif
    template <class ParentAndFunctions, class ...Args>
    Handle(const ParentAndFunctions& parentAndFunctions, const CreateInfo& createInfo,  const Args&... args)
        : m_handle(CreateHandle<T, CreateFunc, CreateInfo>()(parentAndFunctions, createInfo, parentAndFunctions, args...)),
          m_destroy(DestroyFunc<T>(parentAndFunctions, m_handle /* sigh; needed for instance and device */, parentAndFunctions, args...)) {}
    ~Handle() { destroy(); }
    Handle(const Handle& other) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(other.m_handle)
        , m_destroy(other.m_destroy) {
        other.m_handle = VK_NULL_HANDLE;
    }
    Handle& operator=(const Handle& other) = delete;
    Handle& operator=(Handle&& other) {
        destroy();
        m_destroy = other.m_destroy;
        m_handle = other.m_handle;
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator T() const { return m_handle; }
    explicit operator bool() const { return m_handle != VK_NULL_HANDLE; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle);
    }
    T              m_handle = VK_NULL_HANDLE;
    DestroyFunc<T> m_destroy;
};

template <>
struct CreateHandle<VkInstance, PFN_vkCreateInstance, VkInstanceCreateInfo> {
    template <class Functions>
    VkInstance operator()(const Functions& vk, const VkInstanceCreateInfo& createInfo) {
        VkInstance handle;
        check(vk.vkCreateInstance(&createInfo, nullptr, &handle));
        return handle;
    }
};

template <>
struct DestroyFunc<VkInstance> {
    DestroyFunc(const GlobalCommands& vk, VkInstance handle)
        : destroy(reinterpret_cast<PFN_vkDestroyInstance>(
              vk.vkGetInstanceProcAddr(handle, "vkDestroyInstance"))) {
        if (!destroy)
            throw Exception("Driver's vkGetInstanceProcAddr(vkDestroyInstance) returned null");
    }
    void                  operator()(VkInstance handle) const { destroy(handle, nullptr); }
    PFN_vkDestroyInstance destroy;
};

// Special case VkInstance
// For whatever reason, vulkan breaks create/destroy symmetry here. The destroy
// function must be loaded in InstanceCommands, but we don't have access to that
// here. Instead, we re-load the function per object (assuming there won't be
// many). The real fix would be in the vulkan spec.
using InstanceHandle = Handle<VkInstance, PFN_vkDestroyDevice, VkInstanceCreateInfo>;

// Convenience class to combine the instance handle and its function pointers
class Instance : public InstanceHandle, public InstanceCommands {
public:
    Instance(const GlobalCommands& vk, const VkInstanceCreateInfo& createInfo)
        : Instance(InstanceHandle(vk, createInfo), vk.vkGetInstanceProcAddr) {}
    Instance(InstanceHandle&& handle, PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr)
        : InstanceHandle(std::move(handle)),
          InstanceCommands(*this, vkGetInstanceProcAddr) {}
};

template <>
struct CreateHandle<VkDevice, PFN_vkCreateInstance, VkDeviceCreateInfo> {
    template <class Functions>
    VkDevice operator()(const Functions& vk, VkPhysicalDevice physicalDevice,
                        const VkDeviceCreateInfo& createInfo) {
        VkDevice handle;
        check(vk.vkCreateDevice(physicalDevice, &createInfo, nullptr, &handle));
        return handle;
    }
};

template <>
struct DestroyFunc<VkDevice> {
    DestroyFunc(const Instance& vk, VkPhysicalDevice, VkDevice handle)
        : destroy(reinterpret_cast<PFN_vkDestroyDevice>(
              vk.vkGetDeviceProcAddr(handle, "vkDestroyDevice")))
    {
        if (!destroy)
            throw Exception("Driver's vkGetDeviceProcAddr(vkDestroyDevice) returned null");
    }
    PFN_vkDestroyDevice destroy;
};

// Special case VkDevice
// For whatever reason, vulkan breaks create/destroy symmetry here. The destroy
// function must be loaded in InstanceCommands, but we don't have access to that
// here. Instead, we re-load the function per object (assuming there won't be
// many). The real fix would be in the vulkan spec.
using DeviceHandle = Handle<VkDevice, PFN_vkDestroyDevice, VkDeviceCreateInfo>;

// Convenience class to combine the device handle and its function pointers
class Device : public DeviceHandle, public DeviceCommands {
public:
    Device(DeviceHandle&& handle, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr)
        : DeviceHandle(std::move(handle)),
          DeviceCommands(*this, vkGetDeviceProcAddr) {}
};

} // namespace vko
