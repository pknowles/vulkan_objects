// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <memory>
#include <vk_mem_alloc.h>
#include <vko/exceptions.hpp>

namespace vko {

namespace vma {

class Map {
public:
    Map(VmaAllocator allocator, VmaAllocation allocation)
        : m_allocator(allocator)
        , m_allocation(allocation) {
        check(vmaMapMemory(m_allocator, m_allocation, &m_data));
    }
    ~Map() { free(); }
    Map(const Map& other) = delete;
    Map(Map&& other) noexcept
        : m_allocator(other.m_allocator)
        , m_allocation(other.m_allocation)
        , m_data(other.m_data) {
        other.m_allocation = nullptr;
    }
    Map& operator=(const Map& other) = delete;
    Map& operator=(Map&& other) noexcept {
        free();
        m_allocator        = other.m_allocator;
        m_allocation       = other.m_allocation;
        m_data             = other.m_data;
        other.m_allocation = nullptr;
        return *this;
    }
    void* data() const { return m_data; }
    void  invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
        vmaInvalidateAllocation(m_allocator, m_allocation, offset, size);
    }
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
        vmaFlushAllocation(m_allocator, m_allocation, offset, size);
    }

private:
    void free() {
        if (m_allocation) {
            vmaUnmapMemory(m_allocator, m_allocation);
        }
    }
    VmaAllocator  m_allocator  = nullptr; // non-owning
    VmaAllocation m_allocation = nullptr; // non-owning
    void*         m_data;
};

class Allocation {
public:
    Allocation() = delete;
    Allocation(VmaAllocator allocator, VmaAllocation&& allocation,
               const VmaAllocationInfo&) noexcept
        : m_allocator(allocator)
        , m_allocation(allocation) {}
    Allocation(const Allocation& other) = delete;
    Allocation(Allocation&& other) noexcept
        : m_allocator(other.m_allocator)
        , m_allocation(other.m_allocation) {
        other.m_allocation = nullptr;
    }
    Allocation& operator=(const Allocation& other) = delete;
    Allocation& operator=(Allocation&& other) noexcept {
        free();
        m_allocator        = other.m_allocator;
        m_allocation       = other.m_allocation;
        other.m_allocation = nullptr;
        return *this;
    }
    ~Allocation() { free(); }
    Map map() const { return Map(m_allocator, m_allocation); }

private:
    void free() {
        if (m_allocation)
            vmaFreeMemory(m_allocator, m_allocation);
    }

    VmaAllocator  m_allocator  = nullptr; // non-owning
    VmaAllocation m_allocation = nullptr;
};

class Allocator {
public:
    using AllocationType = Allocation;
    using MapType        = Map;

    template <class GlobalFunctions, class InstanceAndCommands>
    Allocator(const GlobalFunctions &globalFunctions, const InstanceAndCommands &instance,
              VkPhysicalDevice physicalDevice, VkDevice device, uint32_t vulkanApiVersion,
              VkBuildAccelerationStructureFlagsKHR flags)
        : Allocator(globalFunctions.vkGetInstanceProcAddr, instance.vkGetDeviceProcAddr, instance, physicalDevice, device, vulkanApiVersion, flags) {}

    Allocator(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr,
              PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr, VkInstance instance,
              VkPhysicalDevice physicalDevice, VkDevice device, uint32_t vulkanApiVersion,
              VkBuildAccelerationStructureFlagsKHR flags)
        : Allocator(vkGetInstanceProcAddr, vkGetDeviceProcAddr,
                    VmaAllocatorCreateInfo{
                        .flags = flags,
                        .physicalDevice = physicalDevice,
                        .device = device,
                        .preferredLargeHeapBlockSize = 0,
                        .pAllocationCallbacks = nullptr,
                        .pDeviceMemoryCallbacks = nullptr,
                        .pHeapSizeLimit = nullptr,
                        .pVulkanFunctions = nullptr,
                        .instance = instance,
                        .vulkanApiVersion = vulkanApiVersion,
                        .pTypeExternalMemoryHandleTypes = nullptr,
                    })
    {
    }

    Allocator(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr,
              PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr, const VmaAllocatorCreateInfo &createInfo)
    {

        if (createInfo.pVulkanFunctions)
            throw Exception("Custom VmaAllocatorCreateInfo::pVulkanFunctions not supported");

        m_vulkanFunctions                        = std::make_unique<VmaVulkanFunctions>();
        m_vulkanFunctions->vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        m_vulkanFunctions->vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

        auto createInfoMut             = createInfo;
        createInfoMut.pVulkanFunctions = m_vulkanFunctions.get();

        check(vmaCreateAllocator(&createInfoMut, &m_allocator));
    }

    Allocator(const Allocator& other) = delete;
    Allocator(Allocator&& other) noexcept
        : m_vulkanFunctions(std::move(other.m_vulkanFunctions))
        , m_allocator(other.m_allocator) {
        other.m_allocator = nullptr;
    }
    Allocator& operator=(const Allocator& other) = delete;
    Allocator& operator=(Allocator&& other) noexcept {
        free();
        m_vulkanFunctions = std::move(other.m_vulkanFunctions);
        m_allocator       = other.m_allocator;
        other.m_allocator = nullptr;
        return *this;
    }
    ~Allocator() { free(); }

    Allocation create(const VkMemoryRequirements& memoryRequirements,
                      VkMemoryPropertyFlags       memoryPropertyFlags) {
        return create(memoryRequirements, allocationCreateInfo(memoryPropertyFlags));
    }

    Allocation create(const VkMemoryRequirements&    memoryRequirements,
                      const VmaAllocationCreateInfo& createInfo) {
        VmaAllocation     allocation;
        VmaAllocationInfo allocationInfo;
        check(vmaAllocateMemory(m_allocator, &memoryRequirements, &createInfo, &allocation,
                                &allocationInfo));
        return {m_allocator, std::move(allocation), allocationInfo};
    }

    Allocation create(VkBuffer buffer, VkMemoryPropertyFlags memoryPropertyFlags) {
        return create(buffer, allocationCreateInfo(memoryPropertyFlags));
    }

    Allocation create(VkBuffer buffer, const VmaAllocationCreateInfo& createInfo) {
        VmaAllocation     allocation;
        VmaAllocationInfo allocationInfo;
        check(vmaAllocateMemoryForBuffer(m_allocator, buffer, &createInfo, &allocation,
                                         &allocationInfo));

        // Binding is a side effect of create(). This is unexpected but
        // currently saves returning the VkDeviceMemory and offset.
        // TODO: probably better to pass these to the caller and not hide features
        check(vmaBindBufferMemory(m_allocator, allocation, buffer));
        return {m_allocator, std::move(allocation), allocationInfo};
    }

    Allocation create(VkImage image, VkMemoryPropertyFlags memoryPropertyFlags) {
        return create(image, allocationCreateInfo(memoryPropertyFlags));
    }

    Allocation create(VkImage image, const VmaAllocationCreateInfo& createInfo) {
        VmaAllocation     allocation;
        VmaAllocationInfo allocationInfo;
        check(vmaAllocateMemoryForImage(m_allocator, image, &createInfo, &allocation,
                                        &allocationInfo));

        // Binding is a side effect of create(). This is unexpected but
        // currently saves returning the VkDeviceMemory and offset.
        // TODO: probably better to pass these to the caller and not hide features
        check(vmaBindImageMemory(m_allocator, allocation, image));
        return {m_allocator, std::move(allocation), allocationInfo};
    }

    operator VmaAllocator() const { return m_allocator; }

private:
    void free() {
        if (m_allocator)
            vmaDestroyAllocator(m_allocator);
    }

    VmaAllocationCreateInfo allocationCreateInfo(VkMemoryPropertyFlags memoryPropertyFlags) const {
        // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
        // WARNING: many flags are outright ignored when
        // VMA_MEMORY_USAGE_UNKNOWN is used and memory properties are given
        // directly.
        VmaAllocationCreateFlags vmaFlags = 0;
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
            vmaFlags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
        // VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
        return {
            .flags          = vmaFlags,
            .usage          = VMA_MEMORY_USAGE_UNKNOWN,
            .requiredFlags  = memoryPropertyFlags,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = nullptr,
            .priority       = 0.0f,
        };
    }

    std::unique_ptr<VmaVulkanFunctions> m_vulkanFunctions;
    VmaAllocator                        m_allocator = nullptr;
};

} // namespace vma

} // namespace vko
