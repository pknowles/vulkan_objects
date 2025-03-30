// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <span>
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
    VkDevice operator()(const VkDeviceCreateInfo& createInfo, VkPhysicalDevice physicalDevice,
                        const Functions& vk) {
        VkDevice handle;
        check(vk.vkCreateDevice(physicalDevice, &createInfo, nullptr, &handle));
        return handle;
    }
};

template <>
struct DestroyFunc<VkDevice> {
    DestroyFunc(VkDevice handle, VkPhysicalDevice /* fake "parent", not needed for destruction */,
                const InstanceCommands& vk)
        : destroy(reinterpret_cast<PFN_vkDestroyDevice>(
              vk.vkGetDeviceProcAddr(handle, "vkDestroyDevice"))) {
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
        : Device(DeviceHandle(createInfo, physicalDevice, vk), vk.vkGetDeviceProcAddr) {}
    Device(DeviceHandle&& handle, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr)
        : DeviceHandle(std::move(handle)),
          DeviceCommands(*this, vkGetDeviceProcAddr) {}
};

template <>
struct CreateHandleVector<VkCommandBuffer, PFN_vkAllocateCommandBuffers,
                          VkCommandBufferAllocateInfo> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    std::vector<VkCommandBuffer> operator()(const VkCommandBufferAllocateInfo& createInfo,
                                            const FunctionsAndParent&          vk) {
        return (*this)(createInfo, vk, vk);
    }
    template <class Functions>
    std::vector<VkCommandBuffer> operator()(const VkCommandBufferAllocateInfo& createInfo,
                                            VkDevice device, const Functions& vk) {
        std::vector<VkCommandBuffer> handles(createInfo.commandBufferCount);
        check(vk.vkAllocateCommandBuffers(device, &createInfo, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkCommandBuffer> {
    DestroyVectorFunc(const VkCommandBufferAllocateInfo& createInfo, VkDevice device,
                      const DeviceCommands& vk)
        : destroy(vk.vkFreeCommandBuffers)
        , device(device)
        , commandPool(createInfo.commandPool) {}
    void operator()(const std::vector<VkCommandBuffer>& handles) const {
        destroy(device, commandPool, uint32_t(handles.size()), handles.data());
    }
    PFN_vkFreeCommandBuffers destroy;
    VkDevice                 device;
    VkCommandPool            commandPool;
};

// An array of VkCommandBuffer. Exposes an array directly because that's what
// the API provides.
using CommandBuffers =
    HandleVector<VkCommandBuffer, PFN_vkAllocateCommandBuffers, VkCommandBufferAllocateInfo>;

// Utility to expose CommandBuffers as a single VkCommandBuffer
// TODO: special case to avoid the std::vector heap allocation
class CommandBuffer {
public:
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    CommandBuffer(const void* pNext, VkCommandPool commandPool, VkCommandBufferLevel level,
                  const FunctionsAndParent& vk)
        : CommandBuffer(pNext, commandPool, level, vk, vk) {}
    template <class Functions>
    CommandBuffer(const void* pNext, VkCommandPool commandPool, VkCommandBufferLevel level,
                  VkDevice device, const Functions& vk)
        : m_commandBuffers(
              VkCommandBufferAllocateInfo{
                  .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                  .pNext              = pNext,
                  .commandPool        = commandPool,
                  .level              = level,
                  .commandBufferCount = 1,
              },
              device, vk) {}
    operator VkCommandBuffer() const { return m_commandBuffers[0]; }
    const VkCommandBuffer* ptr() const { return &m_commandBuffers[0]; }

private:
    CommandBuffers m_commandBuffers;
};

// An array of VkShaderEXT. Exposes an array directly because that's what the
// API provides.
using ShadersEXT =
    HandleVector<VkShaderEXT, PFN_vkCreateShadersEXT, std::span<VkShaderCreateInfoEXT>>;

template <>
struct CreateHandleVector<VkShaderEXT, PFN_vkCreateShadersEXT, std::span<VkShaderCreateInfoEXT>> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    std::vector<VkShaderEXT> operator()(const std::span<VkShaderCreateInfoEXT>& createInfo,
                                        const FunctionsAndParent&               vk) {
        return (*this)(createInfo, vk, vk);
    }
    template <class Functions>
    std::vector<VkShaderEXT> operator()(const std::span<VkShaderCreateInfoEXT>& createInfo,
                                        VkDevice device, const Functions& vk) {
        std::vector<VkShaderEXT> handles(createInfo.size());
        check(vk.vkCreateShadersEXT(device, uint32_t(createInfo.size()), createInfo.data(), nullptr,
                                    handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkShaderEXT> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    DestroyVectorFunc(const std::span<VkShaderCreateInfoEXT>& createInfo,
                      const FunctionsAndParent&               vk)
        : DestroyVectorFunc(createInfo, vk, vk) {}
    DestroyVectorFunc(const std::span<VkShaderCreateInfoEXT>&, VkDevice device,
                      const DeviceCommands& vk)
        : destroy(vk.vkDestroyShaderEXT)
        , device(device) {}
    void operator()(const std::vector<VkShaderEXT>& handles) const {
        for (auto handle : handles)
            destroy(device, handle, nullptr);
    }
    PFN_vkDestroyShaderEXT destroy;
    VkDevice               device;
};

// An array of RayTracingPipelinesKHR. Exposes an array directly because that's what the
// API provides.
using RayTracingPipelinesKHR = HandleVector<VkPipeline, PFN_vkCreateRayTracingPipelinesKHR,
                                            std::span<const VkRayTracingPipelineCreateInfoKHR>>;

template <>
struct CreateHandleVector<VkPipeline, PFN_vkCreateRayTracingPipelinesKHR,
                          std::span<const VkRayTracingPipelineCreateInfoKHR>> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    std::vector<VkPipeline>
    operator()(const std::span<const VkRayTracingPipelineCreateInfoKHR>& createInfo,
               const FunctionsAndParent&                                 vk) {
        return (*this)(createInfo, vk, vk);
    }
    template <class Functions>
    std::vector<VkPipeline>
    operator()(const std::span<const VkRayTracingPipelineCreateInfoKHR>& createInfo,
               VkDevice device, const Functions& vk) {
        std::vector<VkPipeline> handles(createInfo.size());
        check(vk.vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                uint32_t(createInfo.size()), createInfo.data(),
                                                nullptr, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkPipeline> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    DestroyVectorFunc(const std::span<const VkRayTracingPipelineCreateInfoKHR>& createInfo,
                      const FunctionsAndParent&                                 vk)
        : DestroyVectorFunc(createInfo, vk, vk) {}

    template <class DeviceCommands>
    DestroyVectorFunc(const std::span<const VkRayTracingPipelineCreateInfoKHR>&, VkDevice device,
                      const DeviceCommands& vk)
        : destroy(vk.vkDestroyPipeline)
        , device(device) {}
    void operator()(const std::vector<VkPipeline>& handles) const {
        for (auto handle : handles)
            destroy(device, handle, nullptr);
    }
    PFN_vkDestroyPipeline destroy;
    VkDevice              device;
};

// Utility to expose RayTracingPipelinesKHR as a single VkPipeline
// TODO: special case to avoid the std::vector heap allocation
class RayTracingPipelineKHR {
public:
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    RayTracingPipelineKHR(const VkRayTracingPipelineCreateInfoKHR& createInfo,
                          const FunctionsAndParent&                vk)
        : RayTracingPipelineKHR(createInfo, vk, vk) {}
    template <class Functions>
    RayTracingPipelineKHR(const VkRayTracingPipelineCreateInfoKHR& createInfo, VkDevice device,
                          const Functions& vk)
        : m_pipelines(std::span{&createInfo, 1}, device, vk) {}
    operator VkPipeline() const { return m_pipelines[0]; }
    const VkPipeline* ptr() const { return &m_pipelines[0]; }

private:
    RayTracingPipelinesKHR m_pipelines;
};

// An array of RayTracingPipelinesKHR. Exposes an array directly because that's what the
// API provides.
using DescriptorSets =
    HandleVector<VkDescriptorSet, PFN_vkAllocateDescriptorSets, VkDescriptorSetAllocateInfo>;

template <>
struct CreateHandleVector<VkDescriptorSet, PFN_vkAllocateDescriptorSets,
                          VkDescriptorSetAllocateInfo> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    std::vector<VkDescriptorSet> operator()(const VkDescriptorSetAllocateInfo& createInfo,
                                            const FunctionsAndParent&          vk) {
        return (*this)(createInfo, vk, vk);
    }
    template <class Functions>
    std::vector<VkDescriptorSet> operator()(const VkDescriptorSetAllocateInfo& createInfo,
                                            VkDevice device, const Functions& vk) {
        std::vector<VkDescriptorSet> handles(createInfo.descriptorSetCount);
        check(vk.vkAllocateDescriptorSets(device, &createInfo, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkDescriptorSet> {
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    DestroyVectorFunc(const VkDescriptorSetAllocateInfo& allocateInfo, const FunctionsAndParent& vk)
        : DestroyVectorFunc(allocateInfo, vk, vk) {}

    template <class DeviceCommands>
    DestroyVectorFunc(const VkDescriptorSetAllocateInfo& allocateInfo, VkDevice device,
                      const DeviceCommands& vk)
        : destroy(vk.vkFreeDescriptorSets)
        , device(device)
        , descriptorPool(allocateInfo.descriptorPool) {}
    void operator()(const std::vector<VkDescriptorSet>& handles) const {
        destroy(device, descriptorPool, uint32_t(handles.size()), handles.data());
    }
    PFN_vkFreeDescriptorSets destroy;
    VkDevice                 device;
    VkDescriptorPool         descriptorPool;
};

class DescriptorSet {
public:
    template <class FunctionsAndParent>
        requires std::constructible_from<VkDevice, FunctionsAndParent>
    DescriptorSet(const void* pNext, VkDescriptorPool descriptorPool,
                  VkDescriptorSetLayout setLayout, const FunctionsAndParent& vk)
        : DescriptorSet(pNext, descriptorPool, setLayout, vk, vk) {}
    template <class Functions>
    DescriptorSet(const void* pNext, VkDescriptorPool descriptorPool,
                  VkDescriptorSetLayout setLayout, VkDevice device, const Functions& vk)
        : m_descriptorSets(
              VkDescriptorSetAllocateInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .pNext = pNext,
                                          .descriptorPool     = descriptorPool,
                                          .descriptorSetCount = 1,
                                          .pSetLayouts        = &setLayout},
              device, vk) {}
    operator VkDescriptorSet() const { return m_descriptorSets[0]; }
    const VkDescriptorSet* ptr() const { return &m_descriptorSets[0]; }

private:
    DescriptorSets m_descriptorSets;
};

} // namespace vko
