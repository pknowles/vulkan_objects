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

struct Context {
    vko::VulkanLibrary  library;
    vko::GlobalCommands globalCommands;
    vko::Instance       instance;
#if VULKAN_OBJECTS_HAS_VVL
    vko::DebugMessenger debugMessenger;
#endif
    VkPhysicalDevice    physicalDevice   = VK_NULL_HANDLE;
    uint32_t            queueFamilyIndex = 0;
    vko::Device         device;
    vko::vma::Allocator allocator;

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
        , queueFamilyIndex([this]() {
            auto props =
                vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice);
            auto it = std::ranges::find_if(props, queueSuitable);
            if (it == props.end()) {
                throw std::runtime_error("No suitable queue family");
            }
            return uint32_t(std::distance(props.begin(), it));
        }())
        , device(instance, physicalDevice, TestDeviceCreateInfo(queueFamilyIndex))
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

    // Helper to record an empty command buffer (for testing timeline operations)
    vko::simple::RecordingCommandBuffer recordEmptyCommandBuffer(VkCommandPool pool) {
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