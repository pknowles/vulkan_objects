// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/gen_handles.hpp>

namespace vko
{

template <>
struct CreateHandle<VkInstance, PFN_vkCreateInstance, VkInstanceCreateInfo> {
    template <class Functions>
    VkInstance operator()(const VkInstanceCreateInfo& createInfo, const Functions& vk) {
        VkInstance handle;
        check(vk.vkCreateInstance(&createInfo, nullptr, &handle));
        return handle;
    }
};

template <>
struct DestroyFunc<VkInstance> {
    template <class Functions>
    DestroyFunc(VkInstance handle, const Functions& vk)
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
using InstanceHandle = Handle<VkInstance, PFN_vkCreateInstance, VkInstanceCreateInfo>;

// Convenience class to combine the instance handle and its function pointers
class Instance : public InstanceHandle, public InstanceCommands {
public:
    Instance(const VkInstanceCreateInfo& createInfo, const GlobalCommands& vk)
        : Instance(InstanceHandle(createInfo, vk), vk.vkGetInstanceProcAddr) {}
    Instance(InstanceHandle&& handle, PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr)
        : InstanceHandle(std::move(handle)),
          InstanceCommands(*this, vkGetInstanceProcAddr) {}
};

template <>
struct CreateHandle<VkDevice, PFN_vkCreateDevice, VkDeviceCreateInfo> {
    template <class Functions>
    VkDevice operator()(const VkDeviceCreateInfo& createInfo, const Functions& vk,
                        VkPhysicalDevice physicalDevice) {
        VkDevice handle;
        check(vk.vkCreateDevice(physicalDevice, &createInfo, nullptr, &handle));
        return handle;
    }
};

template <>
struct DestroyFunc<VkDevice> {
    DestroyFunc(VkDevice handle, const InstanceCommands& vk, VkPhysicalDevice /* fake "parent", not needed for destruction */)
        : destroy(reinterpret_cast<PFN_vkDestroyDevice>(
              vk.vkGetDeviceProcAddr(handle, "vkDestroyDevice")))
    {
        if (!destroy)
            throw Exception("Driver's vkGetDeviceProcAddr(vkDestroyDevice) returned null");
    }
    void                  operator()(VkDevice handle) const { destroy(handle, nullptr); }
    PFN_vkDestroyDevice destroy;
};

// Special case VkDevice
// For whatever reason, vulkan breaks create/destroy symmetry here. The destroy
// function must be loaded in InstanceCommands, but we don't have access to that
// here. Instead, we re-load the function per object (assuming there won't be
// many). The real fix would be in the vulkan spec.
using DeviceHandle = Handle<VkDevice, PFN_vkCreateDevice, VkDeviceCreateInfo>;

// Convenience class to combine the device handle and its function pointers
class Device : public DeviceHandle, public DeviceCommands {
public:
    Device(const VkDeviceCreateInfo& createInfo, const InstanceCommands& vk,
           VkPhysicalDevice physicalDevice)
        : Device(DeviceHandle(createInfo, vk, physicalDevice), vk.vkGetDeviceProcAddr) {}
    Device(DeviceHandle&& handle, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr)
        : DeviceHandle(std::move(handle)),
          DeviceCommands(*this, vkGetDeviceProcAddr) {}
};

} // namespace vko
