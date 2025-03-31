// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <span>
#include <vko/gen_handles.hpp>

namespace vko
{

// Special case handle for VkInstance
class InstanceHandle {
public:
    using CreateInfo = VkInstanceCreateInfo;

    template <class GlobalCommands>
    InstanceHandle(const GlobalCommands& vk, const VkInstanceCreateInfo& createInfo) {
        check(vk.vkCreateInstance(&createInfo, nullptr, &m_handle));

        m_destroy = reinterpret_cast<PFN_vkDestroyInstance>(
            vk.vkGetInstanceProcAddr(m_handle, "vkDestroyInstance"));

        // WARNING: this leaks the VkInstance. IMO this is a spec bug. Should
        // not have to have a valid instance before loading the destroy
        // function.
        if (!m_destroy)
            throw Exception("Driver's vkGetInstanceProcAddr(vkDestroyInstance) returned null");
    }

    ~InstanceHandle() { destroy(); }
    InstanceHandle(const InstanceHandle& other) = delete;
    InstanceHandle(InstanceHandle&& other) noexcept
        : m_handle(std::move(other.m_handle))
        , m_destroy(std::move(other.m_destroy)) {
        other.m_handle = VK_NULL_HANDLE;
    }
    InstanceHandle& operator=(const InstanceHandle& other) = delete;
    InstanceHandle& operator=(InstanceHandle&& other) {
        destroy();
        m_destroy      = std::move(other.m_destroy);
        m_handle       = std::move(other.m_handle);
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator VkInstance() const { return m_handle; }
    explicit          operator bool() const { return m_handle != VK_NULL_HANDLE; }
    const VkInstance* ptr() const { return &m_handle; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle, nullptr);
    }
    VkInstance            m_handle = VK_NULL_HANDLE;
    PFN_vkDestroyInstance m_destroy;
};

// Convenience class to combine the instance handle and its function pointers
class Instance : public InstanceHandle, public InstanceCommands {
public:
    Instance(const GlobalCommands& vk, const VkInstanceCreateInfo& createInfo)
        : Instance(InstanceHandle(vk, createInfo), vk.vkGetInstanceProcAddr) {}
    Instance(InstanceHandle&& handle, PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr)
        : InstanceHandle(std::move(handle)),
          InstanceCommands(*this, vkGetInstanceProcAddr) {}
};

// Special case handle for VkDevice
class DeviceHandle {
public:
    using CreateInfo = VkDeviceCreateInfo;

    template <class InstanceCommands>
    DeviceHandle(const InstanceCommands& vk, VkPhysicalDevice physicalDevice,
                 const VkDeviceCreateInfo& createInfo) {
        check(vk.vkCreateDevice(physicalDevice, &createInfo, nullptr, &m_handle));

        m_destroy = reinterpret_cast<PFN_vkDestroyDevice>(
            vk.vkGetDeviceProcAddr(m_handle, "vkDestroyDevice"));

        // WARNING: this leaks the VkDevice. IMO this is a spec bug. Should not
        // have to have a valid device before loading the destroy function.
        if (!m_destroy)
            throw Exception("Driver's vkGetDeviceProcAddr(vkDestroyDevice) returned null");
    }

    ~DeviceHandle() { destroy(); }
    DeviceHandle(const DeviceHandle& other) = delete;
    DeviceHandle(DeviceHandle&& other) noexcept
        : m_handle(std::move(other.m_handle))
        , m_destroy(std::move(other.m_destroy)) {
        other.m_handle = VK_NULL_HANDLE;
    }
    DeviceHandle& operator=(const DeviceHandle& other) = delete;
    DeviceHandle& operator=(DeviceHandle&& other) {
        destroy();
        m_destroy      = std::move(other.m_destroy);
        m_handle       = std::move(other.m_handle);
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator VkDevice() const { return m_handle; }
    explicit        operator bool() const { return m_handle != VK_NULL_HANDLE; }
    const VkDevice* ptr() const { return &m_handle; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle, nullptr);
    }
    VkDevice            m_handle = VK_NULL_HANDLE;
    PFN_vkDestroyDevice m_destroy;
};

// Convenience class to combine the device handle and its function pointers
class Device : public DeviceHandle, public DeviceCommands {
public:
    Device(const InstanceCommands& vk, VkPhysicalDevice physicalDevice,
           const VkDeviceCreateInfo& createInfo)
        : Device(DeviceHandle(vk, physicalDevice, createInfo), vk.vkGetDeviceProcAddr) {}
    Device(DeviceHandle&& handle, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr)
        : DeviceHandle(std::move(handle)),
          DeviceCommands(*this, vkGetDeviceProcAddr) {}
};

template <>
struct CreateHandleVector<VkCommandBuffer, PFN_vkAllocateCommandBuffers> {
    using CreateInfo = VkCommandBufferAllocateInfo;

    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    std::vector<VkCommandBuffer> operator()(const DeviceAndCommands&           vk,
                                            const VkCommandBufferAllocateInfo& createInfo) {
        return (*this)(vk, vk, createInfo);
    }
    template <class DeviceCommands>
    std::vector<VkCommandBuffer> operator()(const DeviceCommands& vk, VkDevice device,
                                            const VkCommandBufferAllocateInfo& createInfo) {
        std::vector<VkCommandBuffer> handles(createInfo.commandBufferCount);
        check(vk.vkAllocateCommandBuffers(device, &createInfo, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkCommandBuffer> {
    DestroyVectorFunc(const DeviceCommands& vk, VkDevice device,
                      const VkCommandBufferAllocateInfo& createInfo)
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
using CommandBuffers = HandleVector<VkCommandBuffer, PFN_vkAllocateCommandBuffers>;

// Utility to expose CommandBuffers as a single VkCommandBuffer
// TODO: special case to avoid the std::vector heap allocation
class CommandBuffer {
public:
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    CommandBuffer(const DeviceAndCommands& vk, const void* pNext, VkCommandPool commandPool,
                  VkCommandBufferLevel level)
        : CommandBuffer(vk, vk, pNext, commandPool, level) {}
    template <class Commands>
    CommandBuffer(const Commands& vk, VkDevice device, const void* pNext, VkCommandPool commandPool,
                  VkCommandBufferLevel level)
        : m_commandBuffers(vk, device,
                           VkCommandBufferAllocateInfo{
                               .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                               .pNext              = pNext,
                               .commandPool        = commandPool,
                               .level              = level,
                               .commandBufferCount = 1,
                           }) {}
    operator VkCommandBuffer() const { return m_commandBuffers[0]; }
    const VkCommandBuffer* ptr() const { return &m_commandBuffers[0]; }

private:
    CommandBuffers m_commandBuffers;
};

// An array of VkShaderEXT. Exposes an array directly because that's what the
// API provides.
using ShadersEXT = HandleVector<VkShaderEXT, PFN_vkCreateShadersEXT>;

template <>
struct CreateHandleVector<VkShaderEXT, PFN_vkCreateShadersEXT> {
    using CreateInfo = std::span<const VkShaderCreateInfoEXT>;

    template <class DeviceAndCommands>
    // requires std::constructible_from<VkDevice, DeviceAndCommands>
    std::vector<VkShaderEXT> operator()(const DeviceAndCommands&               vk,
                                        std::span<const VkShaderCreateInfoEXT> createInfo) {
        return (*this)(vk, vk, createInfo);
    }
    template <class DeviceCommands>
    std::vector<VkShaderEXT> operator()(const DeviceCommands& vk, VkDevice device,
                                        std::span<const VkShaderCreateInfoEXT> createInfo) {
        std::vector<VkShaderEXT> handles(createInfo.size());
        check(vk.vkCreateShadersEXT(device, uint32_t(createInfo.size()), createInfo.data(), nullptr,
                                    handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkShaderEXT> {
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    DestroyVectorFunc(const DeviceAndCommands&               vk,
                      std::span<const VkShaderCreateInfoEXT> createInfo)
        : DestroyVectorFunc(vk, vk, createInfo) {}
    DestroyVectorFunc(const DeviceCommands& vk, VkDevice device,
                      const std::span<const VkShaderCreateInfoEXT>&)
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
using RayTracingPipelinesKHR = HandleVector<VkPipeline, PFN_vkCreateRayTracingPipelinesKHR>;

template <>
struct CreateHandleVector<VkPipeline, PFN_vkCreateRayTracingPipelinesKHR> {
    using CreateInfo = std::span<const VkRayTracingPipelineCreateInfoKHR>;

    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    std::vector<VkPipeline>
    operator()(const DeviceAndCommands&                           vk,
               std::span<const VkRayTracingPipelineCreateInfoKHR> createInfo) {
        return (*this)(vk, vk, createInfo);
    }
    template <class Commands>
    std::vector<VkPipeline>
    operator()(const Commands& vk, VkDevice device,
               std::span<const VkRayTracingPipelineCreateInfoKHR> createInfo) {
        std::vector<VkPipeline> handles(createInfo.size());
        check(vk.vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                uint32_t(createInfo.size()), createInfo.data(),
                                                nullptr, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkPipeline> {
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    DestroyVectorFunc(const DeviceAndCommands&                           vk,
                      std::span<const VkRayTracingPipelineCreateInfoKHR> createInfo)
        : DestroyVectorFunc(vk, vk, createInfo) {}

    template <class DeviceCommands>
    DestroyVectorFunc(const DeviceCommands& vk, VkDevice device,
                      std::span<const VkRayTracingPipelineCreateInfoKHR>)
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
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    RayTracingPipelineKHR(const DeviceAndCommands&                 vk,
                          const VkRayTracingPipelineCreateInfoKHR& createInfo)
        : RayTracingPipelineKHR(vk, vk, createInfo) {}
    template <class DeviceCommands>
    RayTracingPipelineKHR(const DeviceCommands& vk, VkDevice device,
                          const VkRayTracingPipelineCreateInfoKHR& createInfo)
        : m_pipelines(vk, device, std::span{&createInfo, 1}) {}
    operator VkPipeline() const { return m_pipelines[0]; }
    const VkPipeline* ptr() const { return &m_pipelines[0]; }

private:
    RayTracingPipelinesKHR m_pipelines;
};

// An array of RayTracingPipelinesKHR. Exposes an array directly because that's what the
// API provides.
using DescriptorSets = HandleVector<VkDescriptorSet, PFN_vkAllocateDescriptorSets>;

template <>
struct CreateHandleVector<VkDescriptorSet, PFN_vkAllocateDescriptorSets> {
    using CreateInfo = VkDescriptorSetAllocateInfo;

    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    std::vector<VkDescriptorSet> operator()(const DeviceAndCommands&           vk,
                                            const VkDescriptorSetAllocateInfo& createInfo) {
        return (*this)(vk, vk, createInfo);
    }
    template <class DeviceCommands>
    std::vector<VkDescriptorSet> operator()(const DeviceCommands& vk, VkDevice device,
                                            const VkDescriptorSetAllocateInfo& createInfo) {
        std::vector<VkDescriptorSet> handles(createInfo.descriptorSetCount);
        check(vk.vkAllocateDescriptorSets(device, &createInfo, handles.data()));
        return handles;
    }
};

template <>
struct DestroyVectorFunc<VkDescriptorSet> {
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    DestroyVectorFunc(const DeviceAndCommands& vk, const VkDescriptorSetAllocateInfo& allocateInfo)
        : DestroyVectorFunc(vk, vk, allocateInfo) {}

    template <class DeviceCommands>
    DestroyVectorFunc(const DeviceCommands& vk, VkDevice device,
                      const VkDescriptorSetAllocateInfo& allocateInfo)
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
    template <class DeviceAndCommands>
        requires std::constructible_from<VkDevice, DeviceAndCommands>
    DescriptorSet(const DeviceAndCommands& vk, const void* pNext, VkDescriptorPool descriptorPool,
                  VkDescriptorSetLayout setLayout)
        : DescriptorSet(vk, vk, pNext, descriptorPool, setLayout) {}
    template <class DeviceCommands>
    DescriptorSet(const DeviceCommands& vk, VkDevice device, const void* pNext,
                  VkDescriptorPool descriptorPool, VkDescriptorSetLayout setLayout)
        : m_descriptorSets(
              vk, device,
              VkDescriptorSetAllocateInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .pNext = pNext,
                                          .descriptorPool     = descriptorPool,
                                          .descriptorSetCount = 1,
                                          .pSetLayouts        = &setLayout}) {}
    operator VkDescriptorSet() const { return m_descriptorSets[0]; }
    const VkDescriptorSet* ptr() const { return &m_descriptorSets[0]; }

private:
    DescriptorSets m_descriptorSets;
};

} // namespace vko
