// Copyright (c) 2025 Pyarelal Knowles, MIT License

#pragma once

#include <algorithm>
#include <debugbreak.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <test_context_create_info.hpp>
#include <vko/adapters.hpp>
#include <vko/command_recording.hpp>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>
#include <vko/handles.hpp>
#include <vko/shortcuts.hpp>
#include <vulkan/vulkan_core.h>

// Forward declarations from context_create_info.hpp
bool physicalDeviceSuitable(const vko::Instance& instance, VkPhysicalDevice physicalDevice);
bool queueSuitable(const VkQueueFamilyProperties& properties);

// Helper to find a queue family with specific flags
inline std::optional<uint32_t> findQueueFamily(
    const std::vector<VkQueueFamilyProperties>& props, 
    VkQueueFlags requiredFlags,
    std::optional<uint32_t> excludeIndex = std::nullopt) {
    for (size_t i = 0; i < props.size(); ++i) {
        if (excludeIndex.has_value() && i == excludeIndex.value())
            continue;
        if ((props[i].queueFlags & requiredFlags) == requiredFlags) {
            return uint32_t(i);
        }
    }
    return std::nullopt;
}

struct Context {
    vko::VulkanLibrary  library;
    vko::GlobalCommands globalCommands;
    vko::Instance       instance;
#if VULKAN_OBJECTS_HAS_VVL
    vko::DebugMessenger debugMessenger;
#endif
    VkPhysicalDevice           physicalDevice    = VK_NULL_HANDLE;
    uint32_t                   queueFamilyIndex  = 0;
    std::optional<uint32_t>    queueFamilyIndex2;
    std::vector<const char*>   optionalExtensions;
    vko::Device                device;
    vko::vma::Allocator        allocator;

    Context()
        : library()
        , globalCommands(library.loader())
        , instance(globalCommands, TestInstanceCreateInfo())
#if VULKAN_OBJECTS_HAS_VVL
        , debugMessenger(instance,
                         [](VkDebugUtilsMessageSeverityFlagBitsEXT severityBits,
                            VkDebugUtilsMessageTypeFlagsEXT,
                            const VkDebugUtilsMessengerCallbackDataEXT& callbackData) -> bool {
                             std::cout << callbackData.pMessage << std::endl;
                             VkFlags breakOnSeverity =
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                             if ((severityBits & breakOnSeverity) != 0) {
                                 debug_break();
                             }
                             return false;
                         })
#endif
        , physicalDevice([this]() {
            auto physicalDevices = vko::toVector(instance.vkEnumeratePhysicalDevices, instance);
            auto it = std::ranges::find_if(physicalDevices, [this](VkPhysicalDevice pd) {
                return physicalDeviceSuitable(instance, pd);
            });
            if (it == physicalDevices.end()) {
                throw std::runtime_error("No suitable physical device");
            }
            return *it;
        }())
        , queueFamilyIndex(findQueueFamily(
            vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice),
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT).value())
        , queueFamilyIndex2(findQueueFamily(
            vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice),
            VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
            queueFamilyIndex))
        // Detect and enable optional vendor-specific extensions if available
        // Note: a real app would prioritize the physical device supporting desired extensions
        , optionalExtensions([this]() {
            std::vector<const char*> extensions;
            auto availableExts = vko::toVector(instance.vkEnumerateDeviceExtensionProperties, 
                                              physicalDevice, nullptr);
            if (std::ranges::any_of(availableExts, [](const VkExtensionProperties& p) {
                return std::string_view(p.extensionName) == VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME;
            })) {
                extensions.push_back(VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME);
            }
            return extensions;
        }())
        , device(instance, physicalDevice,
                 queueFamilyIndex2.has_value()
                     ? TestDeviceCreateInfo(queueFamilyIndex, queueFamilyIndex2.value(), optionalExtensions)
                     : TestDeviceCreateInfo(queueFamilyIndex, optionalExtensions))
        , allocator(globalCommands, instance, physicalDevice, device, VK_API_VERSION_1_4, 0) {}

    // Helper to create a command pool for testing
    vko::CommandPool createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndex,
        };
        return vko::CommandPool(device, poolInfo);
    }

    // Begin recording a command buffer for testing
    vko::simple::RecordingCommandBuffer beginRecording(VkCommandPool pool) {
        return vko::simple::RecordingCommandBuffer(
            device,
            vko::CommandBuffer(device, device, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY),
            VkCommandBufferBeginInfo{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext            = nullptr,
                .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr,
            });
    }
};

class UnitTestFixture : public ::testing::Test {
protected:
    static inline std::unique_ptr<Context> ctx;
    static void                            SetUpTestSuite() { ctx = std::make_unique<Context>(); }
    static void                            TearDownTestSuite() { ctx.reset(); }
};