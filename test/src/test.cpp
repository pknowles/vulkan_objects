// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <vko/adaptors.hpp>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>
#include <vko/handles.hpp>
#include <vulkan/vulkan_core.h>

// Dangerous internal pointers encapsulated in a non-copyable non-movable
// app-specific struct
struct TestInstanceCreateInfo {
    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "vulkan_objects test application",
        .applicationVersion = 0,
        .pEngineName = nullptr,
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instanceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };
    operator VkInstanceCreateInfo&() { return instanceCreateInfo; }
    TestInstanceCreateInfo() = default;
    TestInstanceCreateInfo(const TestInstanceCreateInfo& other) = delete;
    TestInstanceCreateInfo operator=(const TestInstanceCreateInfo& other) = delete;
};

struct TestDeviceCreateInfo {
    std::array<const char*, 2>              deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                                VK_EXT_SHADER_OBJECT_EXTENSION_NAME};
    float                                   queuePriority = 1.0f;
    VkDeviceQueueCreateInfo                 queueCreateInfo;
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature;
    VkDeviceCreateInfo                      deviceCreateInfo;
    TestDeviceCreateInfo(uint32_t queueFamilyIndex)
        : queueCreateInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                          .pNext = nullptr,
                          .flags = 0,
                          .queueFamilyIndex = queueFamilyIndex,
                          .queueCount = 1,
                          .pQueuePriorities = &queuePriority},
          shaderObjectFeature{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
                              .pNext = nullptr,
                              .shaderObject = VK_TRUE},
          deviceCreateInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &shaderObjectFeature,
                           .flags = 0,
                           .queueCreateInfoCount = 1U,
                           .pQueueCreateInfos = &queueCreateInfo,
                           .enabledLayerCount = 0,
                           .ppEnabledLayerNames = 0,
                           .enabledExtensionCount = uint32_t(deviceExtensions.size()),
                           .ppEnabledExtensionNames = deviceExtensions.data(),
                           .pEnabledFeatures = nullptr} {}
    operator VkDeviceCreateInfo&() { return deviceCreateInfo; }
    TestDeviceCreateInfo(const TestDeviceCreateInfo& other) = delete;
    TestDeviceCreateInfo operator=(const TestDeviceCreateInfo& other) = delete;
};

TEST(Integration, InitHappyPath) {
    vko::VulkanLibrary           library;
    vko::GlobalCommands          globalCommands(library.loader());
    std::optional<vko::Instance> another;
    vko::Instance instance(vko::InstanceHandle(TestInstanceCreateInfo(), globalCommands),
                           library.loader());

    // Pick a VkPhysicalDevice
    std::vector<VkPhysicalDevice> physicalDevices =
        vko::toVector(instance.vkEnumeratePhysicalDevices, instance);
    auto physicalDeviceIt =
        std::ranges::find_if(physicalDevices, [&](VkPhysicalDevice physicalDevice) -> bool {
            VkPhysicalDeviceFeatures2 features2 =
                vko::get(instance.vkGetPhysicalDeviceFeatures2, physicalDevice);
            bool anyHasAll = std::ranges::any_of(
                vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice),
                [](const VkQueueFamilyProperties& properties) {
                    return (properties.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                                                     VK_QUEUE_TRANSFER_BIT)) != 0;
                });
            // Don't really care. Purely for demonstration
            return features2.features.shaderInt64 == true && anyHasAll;
        });
    ASSERT_NE(physicalDeviceIt, physicalDevices.end());
    VkPhysicalDevice physicalDevice = *physicalDeviceIt;

    // Pick a single VkQueue family
    // TODO: reuse results from above call
    std::vector<VkQueueFamilyProperties> queueProperties =
        vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice);
    auto queuePropertiesIt =
        std::ranges::find_if(queueProperties, [](const VkQueueFamilyProperties& properties) {
            return (properties.queueFlags &
                    (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) != 0;
        });
    ASSERT_NE(queuePropertiesIt, queueProperties.end());
    uint32_t queueFamilyIndex(std::distance(queueProperties.begin(), queuePropertiesIt));

    // Create a VkDevice
    vko::Device device(TestDeviceCreateInfo(queueFamilyIndex), instance, physicalDevice);

    VkQueue queue = vko::get(device.vkGetDeviceQueue, device, queueFamilyIndex, 0);

    // Test the first device call
    device.vkQueueWaitIdle(queue);

    // Create the first non-instance/device object
    VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = queueFamilyIndex,
    };
    vko::CommandPool commandPool(commandPoolCreateInfo, device);
}
