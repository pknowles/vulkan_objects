// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <algorithm>
#include <array>
#include <debugbreak.h>
#include <vko/adapters.hpp>
#include <vko/array.hpp>
#include <vko/command_recording.hpp>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>
#include <vko/glfw_objects.hpp>
#include <vko/handles.hpp>
#include <vko/image.hpp>
#include <vko/swapchain.hpp>
#include <vko/timeline_queue.hpp>
#include <vulkan/vulkan_core.h>

#ifdef VK_USE_PLATFORM_XLIB_KHR
    #pragma push_macro("None")
    #pragma push_macro("Bool")
    #undef None
    #undef Bool
#endif
#include <gtest/gtest.h>
#ifdef VK_USE_PLATFORM_XLIB_KHR
    #pragma pop_macro("Bool")
    #pragma pop_macro("None")
#endif

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
            return features2.features.shaderInt64 == VK_TRUE && anyHasAll;
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
    uint32_t queueFamilyIndex = uint32_t(std::distance(queueProperties.begin(), queuePropertiesIt));

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

struct WindowInstanceCreateInfo {
    std::vector<const char*>   requiredLayers;
    std::array<const char*, 3> requiredExtensions;
    VkApplicationInfo          applicationInfo;
    const VkBool32               verboseValue = true;
    const VkLayerSettingEXT      layerSetting = {"VK_LAYER_KHRONOS_validation", "validate_sync",
                                                 VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &verboseValue};
    VkLayerSettingsCreateInfoEXT layerSettingsCreateInfo = {
        VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT, nullptr, 1, &layerSetting};
    VkInstanceCreateInfo instanceCreateInfo;
    WindowInstanceCreateInfo(vko::glfw::PlatformSupport support)
        : requiredLayers{"VK_LAYER_KHRONOS_validation"} // only when !defined(NDEBUG) in a real app
        , requiredExtensions{VK_KHR_SURFACE_EXTENSION_NAME,
                             vko::glfw::platformSurfaceExtension(support),
                             VK_EXT_DEBUG_UTILS_EXTENSION_NAME}
        , applicationInfo{.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .pNext              = nullptr,
                          .pApplicationName   = "vulkan_objects test application",
                          .applicationVersion = 0,
                          .pEngineName        = nullptr,
                          .engineVersion      = 0,
                          .apiVersion         = VK_API_VERSION_1_4}
        , instanceCreateInfo{.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                             .pNext                   = &layerSettingsCreateInfo,
                             .flags                   = 0,
                             .pApplicationInfo        = &applicationInfo,
                             .enabledLayerCount       = uint32_t(requiredLayers.size()),
                             .ppEnabledLayerNames     = requiredLayers.data(),
                             .enabledExtensionCount   = uint32_t(requiredExtensions.size()),
                             .ppEnabledExtensionNames = requiredExtensions.data()} {}
    operator VkInstanceCreateInfo&() { return instanceCreateInfo; }
    WindowInstanceCreateInfo(const WindowInstanceCreateInfo& other) = delete;
    WindowInstanceCreateInfo operator=(const WindowInstanceCreateInfo& other) = delete;
};

TEST(Integration, WindowSystemIntegration) {
    vko::VulkanLibrary  library;
    vko::GlobalCommands globalCommands(library.loader());
    auto                instanceExtensions =
        vko::toVector(globalCommands.vkEnumerateInstanceExtensionProperties, nullptr);
    auto instanceLayers = vko::toVector(globalCommands.vkEnumerateInstanceLayerProperties);
    EXPECT_NE(std::ranges::find_if(instanceLayers,
                                   [](const VkLayerProperties& layer) {
                                       return std::string_view(layer.layerName) ==
                                              "VK_LAYER_KHRONOS_validation";
                                   }),
              instanceLayers.end());
    vko::glfw::PlatformSupport platformSupport(instanceExtensions);
    vko::glfw::ScopedInit glfwInit;
    vko::Instance instance(vko::InstanceHandle(WindowInstanceCreateInfo(platformSupport), globalCommands),
                           library.loader());

    vko::DebugUtilsMessengerEXT debugMessenger(
        VkDebugUtilsMessengerCreateInfoEXT{
            .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext           = nullptr,
            .flags           = 0,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severityBits,
                                  VkDebugUtilsMessageTypeFlagsEXT,
                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                  void*) -> VkBool32 {
                std::cout << pCallbackData->pMessage << std::endl;
                VkFlags breakOnSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                if ((severityBits & breakOnSeverity) != 0) {
                    debug_break();
                }
                return VK_FALSE;
            },
            .pUserData = nullptr},
        instance);

    std::vector<VkPhysicalDevice> physicalDevices =
        vko::toVector(instance.vkEnumeratePhysicalDevices, instance);
    auto physicalDeviceIt =
        std::ranges::find_if(physicalDevices, [&](VkPhysicalDevice physicalDevice) -> bool {
            uint32_t queueFamilyIndex = 0; // TODO: search more than just the first?
            return vko::glfw::physicalDevicePresentationSupport(instance, platformSupport, physicalDevice,
                                                                queueFamilyIndex);
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
    uint32_t queueFamilyIndex = uint32_t(std::distance(queueProperties.begin(), queuePropertiesIt));

    // Create a VkDevice
    vko::Device device(TestDeviceCreateInfo(queueFamilyIndex), instance, physicalDevice);

    // vko::simple::TimelineQueue queue(queueFamilyIndex, 0, device);
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

    vko::glfw::Window     window = vko::glfw::createWindow(800, 600, "Vulkan Window");
    vko::glfw::SurfaceKHR surface(platformSupport, window.get(), instance);
    auto                  surfaceFormats =
        vko::toVector(instance.vkGetPhysicalDeviceSurfaceFormatsKHR, physicalDevice, surface);
    auto surfaceFormatIt =
        std::ranges::find_if(surfaceFormats, [](const VkSurfaceFormatKHR& format) {
            return format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
    ASSERT_NE(surfaceFormatIt, surfaceFormats.end());
    VkSurfaceFormatKHR surfaceFormat = *surfaceFormatIt;
    auto               surfacePresentModes =
        vko::toVector(instance.vkGetPhysicalDeviceSurfacePresentModesKHR, physicalDevice, surface);
    constexpr VkPresentModeKHR preferredPresentMode[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                                         VK_PRESENT_MODE_FIFO_LATEST_READY_EXT,
                                                         VK_PRESENT_MODE_FIFO_KHR};
    auto                       surfacePresentModeIt   = std::ranges::find_if(
        preferredPresentMode, [&surfacePresentModes](const VkPresentModeKHR& mode) {
            return std::ranges::count(surfacePresentModes, mode) > 0;
        });
    ASSERT_NE(surfacePresentModeIt, std::end(preferredPresentMode));
    VkPresentModeKHR surfacePresentMode = *surfacePresentModeIt;

    vko::vma::Allocator  allocator(globalCommands, instance, physicalDevice, device,
                                   VK_API_VERSION_1_4);
    vko::Array<uint32_t> imageData(
        allocator, 1024 * 1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, device);
    uint32_t pixelIndex = 0;
    for (uint32_t& pixel : imageData.map()) {
        uint32_t x = pixelIndex % 1024;
        uint32_t y = pixelIndex / 1024;
        pixel      = (((x ^ y) & 8) != 0) ? 0xFF000000U : 0xFFFFFFFFU;
        ++pixelIndex;
    }
    vko::BoundImage image(allocator,
                          VkImageCreateInfo{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                            .pNext       = nullptr,
                                            .flags       = 0,
                                            .imageType   = VK_IMAGE_TYPE_2D,
                                            .format      = VK_FORMAT_R8G8B8A8_UNORM,
                                            .extent      = {1024, 1024, 1},
                                            .mipLevels   = 1,
                                            .arrayLayers = 1,
                                            .samples     = VK_SAMPLE_COUNT_1_BIT,
                                            .tiling      = VK_IMAGE_TILING_OPTIMAL,
                                            .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                                            .queueFamilyIndexCount = 0,
                                            .pQueueFamilyIndices   = nullptr,
                                            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED},
                          device, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, device);
    {
        vko::simple::ImmediateCommandBuffer cmd(commandPool, queue, device);
        VkBufferImageCopy                   region{
                              .bufferOffset      = 0,
                              .bufferRowLength   = 0,
                              .bufferImageHeight = 0,
                              .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageOffset       = {0, 0, 0},
                              .imageExtent       = {1024, 1024, 1},
        };
        {
            VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                              nullptr,
                                              0U,
                                              VK_ACCESS_TRANSFER_WRITE_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              0U,
                                              0U,
                                              image,
                                              {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U}};
            device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U,
                                        nullptr, 1U, &imageBarrier);
        }
        device.vkCmdCopyBufferToImage(cmd, imageData, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      1, &region);
        {
            VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                              nullptr,
                                              VK_ACCESS_TRANSFER_WRITE_BIT,
                                              VK_ACCESS_TRANSFER_READ_BIT,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              VK_IMAGE_LAYOUT_GENERAL,
                                              0U,
                                              0U,
                                              image,
                                              {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U}};
            device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U,
                                        nullptr, 1U, &imageBarrier);
        }
    }

    for (;;) {
        int width, height;
        glfwGetWindowSize(window.get(), &width, &height);
        vko::simple::Swapchain swapchain{surface,
                                         surfaceFormat,
                                         VkExtent2D{uint32_t(width), uint32_t(height)},
                                         queueFamilyIndex,
                                         surfacePresentMode,
                                         VK_NULL_HANDLE,
                                         device};

        auto [imageIndex, reuseImageSemaphore] = swapchain.acquire(0ULL, device);
        VkSemaphore renderingFinished          = swapchain.renderFinishedSemaphores[imageIndex];

        {
            vko::simple::ImmediateCommandBuffer cmd(commandPool, queue, device);
            cmd.addWait(reuseImageSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            cmd.addSignal(renderingFinished);
            vko::simple::clearSwapchainImage(
                cmd, swapchain.images[imageIndex],
                swapchain.presented[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkClearColorValue{.float32 = {1.0f, 1.0f, 0.0f, 1.0f}}, device);

            VkImageCopy region{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .srcOffset      = {0, 0, 0},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .dstOffset      = {0, 0, 0},
                .extent = {uint32_t(std::min(1024, width)), uint32_t(std::min(1024, height)), 1},
            };
            device.vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, swapchain.images[imageIndex],
                                  VK_IMAGE_LAYOUT_GENERAL, 1, &region);

            {
                VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                  nullptr,
                                                  VK_ACCESS_TRANSFER_WRITE_BIT,
                                                  0U,
                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                                  0U,
                                                  0U,
                                                  swapchain.images[imageIndex],
                                                  {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U}};
                device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0U, 0U,
                                            nullptr, 0U, nullptr, 1U, &imageBarrier);
            }
        }

        swapchain.present(queue, imageIndex, renderingFinished, device);
        device.vkQueueWaitIdle(queue);
        break;
    }
}
