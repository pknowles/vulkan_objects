// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <algorithm>
#include <array>
#include <debugbreak.h>
#include <iostream>
#include <type_traits>
#include <vko/acceleration_structures.hpp>
#include <vko/adapters.hpp>
#include <vko/allocator.hpp>
#include <vko/bindings.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/bound_image.hpp>
#include <vko/command_recording.hpp>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>
#include <vko/glfw_objects.hpp>
#include <vko/handles.hpp>
#include <vko/ray_tracing.hpp>
#include <vko/shortcuts.hpp>
#include <vko/slang_compiler.hpp>
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

VkBool32 debugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severityBits,
                              VkDebugUtilsMessageTypeFlagsEXT,
                              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
    std::cout << pCallbackData->pMessage << std::endl;
    VkFlags breakOnSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if ((severityBits & breakOnSeverity) != 0) {
        debug_break();
    }
    return VK_FALSE;
}

// Dangerous internal pointers encapsulated in a non-copyable non-movable
// app-specific struct
struct TestInstanceCreateInfo {
    static constexpr const char* requiredExtensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
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
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .pApplicationInfo        = &applicationInfo,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = uint32_t(std::size(requiredExtensions)),
        .ppEnabledExtensionNames = requiredExtensions,
    };
    operator VkInstanceCreateInfo&() { return instanceCreateInfo; }
    TestInstanceCreateInfo() = default;
    TestInstanceCreateInfo(const TestInstanceCreateInfo& other) = delete;
    TestInstanceCreateInfo operator=(const TestInstanceCreateInfo& other) = delete;
};

struct TestDeviceCreateInfo {
    std::array<const char*, 2>              deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                                VK_EXT_SHADER_OBJECT_EXTENSION_NAME};
    float                                   queuePriority    = 1.0f;
    VkDeviceQueueCreateInfo                 queueCreateInfo;
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature = {
        .sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        .pNext        = nullptr,
        .shaderObject = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features vulkan13Feature = {
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext              = &shaderObjectFeature,
        .robustImageAccess  = VK_FALSE,
        .inlineUniformBlock = VK_FALSE,
        .descriptorBindingInlineUniformBlockUpdateAfterBind = VK_FALSE,
        .pipelineCreationCacheControl                       = VK_FALSE,
        .privateData                                        = VK_FALSE,
        .shaderDemoteToHelperInvocation                     = VK_FALSE,
        .shaderTerminateInvocation                          = VK_FALSE,
        .subgroupSizeControl                                = VK_FALSE,
        .computeFullSubgroups                               = VK_FALSE,
        .synchronization2                                   = VK_FALSE,
        .textureCompressionASTC_HDR                         = VK_FALSE,
        .shaderZeroInitializeWorkgroupMemory                = VK_FALSE,
        .dynamicRendering        = VK_TRUE, // yay no default initializers
        .shaderIntegerDotProduct = VK_FALSE,
        .maintenance4            = VK_FALSE,
    };
    VkDeviceCreateInfo deviceCreateInfo;
    TestDeviceCreateInfo(uint32_t queueFamilyIndex)
        : queueCreateInfo{.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                          .pNext            = nullptr,
                          .flags            = 0,
                          .queueFamilyIndex = queueFamilyIndex,
                          .queueCount       = 1,
                          .pQueuePriorities = &queuePriority}
        , deviceCreateInfo{.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext                   = &vulkan13Feature,
                           .flags                   = 0,
                           .queueCreateInfoCount    = 1U,
                           .pQueueCreateInfos       = &queueCreateInfo,
                           .enabledLayerCount       = 0,
                           .ppEnabledLayerNames     = nullptr,
                           .enabledExtensionCount   = uint32_t(deviceExtensions.size()),
                           .ppEnabledExtensionNames = deviceExtensions.data(),
                           .pEnabledFeatures        = nullptr} {}
    operator VkDeviceCreateInfo&() { return deviceCreateInfo; }
    TestDeviceCreateInfo(const TestDeviceCreateInfo& other) = delete;
    TestDeviceCreateInfo operator=(const TestDeviceCreateInfo& other) = delete;
};

TEST(Integration, InitHappyPath) {
    vko::VulkanLibrary           library;
    vko::GlobalCommands          globalCommands(library.loader());
    vko::Instance                instance(globalCommands, TestInstanceCreateInfo());
    vko::SimpleDebugMessenger    debugMessenger(instance, debugMessageCallback);

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
    vko::Device device(instance, physicalDevice, TestDeviceCreateInfo(queueFamilyIndex));

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
    vko::CommandPool commandPool(device, commandPoolCreateInfo);
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
    vko::Instance              instance(globalCommands, WindowInstanceCreateInfo(platformSupport));
    vko::SimpleDebugMessenger  debugMessenger(instance, debugMessageCallback);

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
    vko::Device device(instance, physicalDevice, TestDeviceCreateInfo(queueFamilyIndex));

    // vko::simple::TimelineQueue queue(device, queueFamilyIndex, 0);
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
    vko::CommandPool commandPool(device, commandPoolCreateInfo);

    vko::glfw::Window     window = vko::glfw::createWindow(800, 600, "Vulkan Window");
    vko::glfw::SurfaceKHR surface(instance, platformSupport, window.get());
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
                                   VK_API_VERSION_1_4, 0);
    VkExtent3D                 imageExtent = {800U, 600U, 1U};
    vko::BoundBuffer<uint32_t> imageData(
        device, imageExtent.width * imageExtent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, allocator);
    uint32_t pixelIndex = 0;
    for (uint32_t& pixel : imageData.map()) {
        uint32_t x = pixelIndex % imageExtent.width;
        uint32_t y = pixelIndex / imageExtent.width;
        pixel      = (((x ^ y) & 8) != 0) ? 0xFF000000U : 0xFFFFFFFFU;
        ++pixelIndex;
    }
    vko::BoundImage image(device,
                          VkImageCreateInfo{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                            .pNext       = nullptr,
                                            .flags       = 0,
                                            .imageType   = VK_IMAGE_TYPE_2D,
                                            .format      = VK_FORMAT_R8G8B8A8_UNORM,
                                            .extent      = imageExtent,
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
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    {
        vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
        VkBufferImageCopy                   region{
                              .bufferOffset      = 0,
                              .bufferRowLength   = 0,
                              .bufferImageHeight = 0,
                              .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageOffset       = {0, 0, 0},
                              .imageExtent       = imageExtent,
        };
        vko::cmdImageBarrier(device, cmd, image,
                             {
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0U,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                             },
                             {
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             });
        device.vkCmdCopyBufferToImage(cmd, imageData, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      1, &region);
        vko::cmdImageBarrier(device, cmd, image,
                             {
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             },
                             {
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT,
                                 VK_IMAGE_LAYOUT_GENERAL,
                             });
    }

    vko::slang::GlobalSession globalSession;
    slang::TargetDesc         targets[]     = {{
                    .format  = SLANG_SPIRV,
                    .profile = globalSession->findProfile("spirv_1_6"),
    }};
    const char*               searchPaths[] = {"test/shaders"};
    vko::slang::Session       session(globalSession,
                                      slang::SessionDesc{
                                          .targets         = targets,
                                          .targetCount     = SlangInt(std::size(targets)),
                                          .searchPaths     = searchPaths,
                                          .searchPathCount = SlangInt(std::size(searchPaths)),

                                });
    vko::slang::Module        rasterTriangleModule(session, "rasterTriangle");
    ::slang::IComponentType*  entrypoints[] = {
        vko::slang::EntryPoint(rasterTriangleModule, "vsMain"),
        vko::slang::EntryPoint(rasterTriangleModule, "psMain"),
    };
    vko::slang::Composition rasterTriangleComposition(session, entrypoints);
    vko::slang::Program     rasterTriangleProgram(rasterTriangleComposition);
    vko::slang::Code        vsCode(rasterTriangleProgram, 0, 0);
    vko::slang::Code        psCode(rasterTriangleProgram, 1, 0);
    VkShaderCreateInfoEXT   rasterTriangleShaderInfos[]{
        {
              .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
              .pNext                  = nullptr,
              .flags                  = 0,
              .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
              .nextStage              = 0,
              .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
              .codeSize               = uint32_t(vsCode.size()),
              .pCode                  = vsCode.data(),
              .pName                  = "main",
              .setLayoutCount         = 0,
              .pSetLayouts            = nullptr,
              .pushConstantRangeCount = 0,
              .pPushConstantRanges    = nullptr,
              .pSpecializationInfo    = nullptr,
        },
        {
              .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
              .pNext                  = nullptr,
              .flags                  = 0,
              .stage                  = VK_SHADER_STAGE_FRAGMENT_BIT,
              .nextStage              = 0,
              .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
              .codeSize               = uint32_t(psCode.size()),
              .pCode                  = psCode.data(),
              .pName                  = "main",
              .setLayoutCount         = 0,
              .pSetLayouts            = nullptr,
              .pushConstantRangeCount = 0,
              .pPushConstantRanges    = nullptr,
              .pSpecializationInfo    = nullptr,
        },
    };
    vko::ShadersEXT       rasterTriangleShaders(device, device, rasterTriangleShaderInfos);
    VkShaderStageFlagBits rasterTriangleShadersStages[] = {VK_SHADER_STAGE_VERTEX_BIT,
                                                           VK_SHADER_STAGE_FRAGMENT_BIT};

    for (;;) {
        int width, height;
        glfwGetWindowSize(window.get(), &width, &height);
        vko::simple::Swapchain swapchain{
            device,           surface,
            surfaceFormat,    VkExtent2D{uint32_t(width), uint32_t(height)},
            queueFamilyIndex, surfacePresentMode,
            VK_NULL_HANDLE,
        };

        auto [imageIndex, reuseImageSemaphore] = swapchain.acquire(device, 0ULL);
        VkSemaphore renderingFinished          = swapchain.renderFinishedSemaphores[imageIndex];

        {
            vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
            cmd.addWait(reuseImageSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            cmd.addSignal(renderingFinished);
            vko::simple::clearSwapchainImage(
                device, cmd, swapchain.images[imageIndex],
                swapchain.presented[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkClearColorValue{.float32 = {1.0f, 1.0f, 0.0f, 1.0f}});

            VkImageCopy region{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .srcOffset      = {0, 0, 0},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .dstOffset      = {0, 0, 0},
                .extent         = {std::min(imageExtent.width, uint32_t(width)),
                                   std::min(imageExtent.height, uint32_t(height)), 1U},
            };
            device.vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, swapchain.images[imageIndex],
                                  VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            vko::cmdImageBarrier(
                device, cmd, swapchain.images[imageIndex],
                {
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL,
                },
                {
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });

            VkRenderingAttachmentInfo renderingAttachmentInfo{
                .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext              = nullptr,
                .imageView          = swapchain.imageViews[imageIndex],
                .imageLayout        = VK_IMAGE_LAYOUT_GENERAL,
                .resolveMode        = VK_RESOLVE_MODE_NONE,
                .resolveImageView   = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp             = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue         = {},
            };
            VkRenderingInfo renderingInfo{
                .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext                = nullptr,
                .flags                = 0,
                .renderArea           = {{0U, 0U}, {uint32_t(width), uint32_t(height)}},
                .layerCount           = 1U,
                .viewMask             = 0,
                .colorAttachmentCount = 1U,
                .pColorAttachments    = &renderingAttachmentInfo,
                .pDepthAttachment     = nullptr,
                .pStencilAttachment   = nullptr,
            };
            device.vkCmdBeginRendering(cmd, &renderingInfo);
            device.vkCmdBindShadersEXT(cmd, 2U, rasterTriangleShadersStages,
                                       rasterTriangleShaders.data());
            vko::cmdDynamicRenderingDefaults(device, cmd, uint32_t(width), uint32_t(height));
            device.vkCmdDraw(cmd, 3U, 1U, 0U, 0U);
            device.vkCmdEndRendering(cmd);
            vko::cmdImageBarrier(device, cmd, swapchain.images[imageIndex],
                                 {
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 },
                                 {
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     0U,
                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 });
        }

        swapchain.present(device, queue, imageIndex, renderingFinished);
        device.vkQueueWaitIdle(queue);
        break;
    }
}

struct RayTracingDeviceCreateInfo {
    static constexpr auto                       deviceExtensions = std::to_array({
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
    });
    float                                   queuePriority    = 1.0f;
    VkDeviceQueueCreateInfo                 queueCreateInfo;
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeature = {
        .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .pNext               = nullptr,
        .bufferDeviceAddress = VK_TRUE,
        .bufferDeviceAddressCaptureReplay = VK_FALSE,
        .bufferDeviceAddressMultiDevice   = VK_FALSE,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext              = &bufferDeviceAddressFeature,
        .rayTracingPipeline = VK_TRUE,
        .rayTracingPipelineShaderGroupHandleCaptureReplay      = VK_FALSE,
        .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
        .rayTracingPipelineTraceRaysIndirect                   = VK_FALSE,
        .rayTraversalPrimitiveCulling                          = VK_FALSE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeature = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &rayTracingPipeline,
        .accelerationStructure                                 = VK_TRUE,
        .accelerationStructureCaptureReplay                    = VK_FALSE,
        .accelerationStructureIndirectBuild                    = VK_FALSE,
        .accelerationStructureHostCommands                     = VK_FALSE,
        .descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
    };
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature = {
        .sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        .pNext        = &accelerationStructureFeature,
        .shaderObject = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features vulkan13Feature = {
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext              = &shaderObjectFeature,
        .robustImageAccess  = VK_FALSE,
        .inlineUniformBlock = VK_FALSE,
        .descriptorBindingInlineUniformBlockUpdateAfterBind = VK_FALSE,
        .pipelineCreationCacheControl                       = VK_FALSE,
        .privateData                                        = VK_FALSE,
        .shaderDemoteToHelperInvocation                     = VK_FALSE,
        .shaderTerminateInvocation                          = VK_FALSE,
        .subgroupSizeControl                                = VK_FALSE,
        .computeFullSubgroups                               = VK_FALSE,
        .synchronization2                                   = VK_FALSE,
        .textureCompressionASTC_HDR                         = VK_FALSE,
        .shaderZeroInitializeWorkgroupMemory                = VK_FALSE,
        .dynamicRendering        = VK_TRUE, // yay no default initializers
        .shaderIntegerDotProduct = VK_FALSE,
        .maintenance4            = VK_FALSE,
    };
    VkDeviceCreateInfo deviceCreateInfo;
    RayTracingDeviceCreateInfo(uint32_t queueFamilyIndex)
        : queueCreateInfo{.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                          .pNext            = nullptr,
                          .flags            = 0,
                          .queueFamilyIndex = queueFamilyIndex,
                          .queueCount       = 1,
                          .pQueuePriorities = &queuePriority}
        , deviceCreateInfo{.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext                   = &vulkan13Feature,
                           .flags                   = 0,
                           .queueCreateInfoCount    = 1U,
                           .pQueueCreateInfos       = &queueCreateInfo,
                           .enabledLayerCount       = 0,
                           .ppEnabledLayerNames     = 0,
                           .enabledExtensionCount   = uint32_t(deviceExtensions.size()),
                           .ppEnabledExtensionNames = deviceExtensions.data(),
                           .pEnabledFeatures        = nullptr} {}
    operator VkDeviceCreateInfo&() { return deviceCreateInfo; }
    RayTracingDeviceCreateInfo(const RayTracingDeviceCreateInfo& other) = delete;
    RayTracingDeviceCreateInfo operator=(const RayTracingDeviceCreateInfo& other) = delete;
};

TEST(Integration, HelloTriangleRayTracing) {
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
    vko::Instance              instance(globalCommands, WindowInstanceCreateInfo(platformSupport));
    vko::SimpleDebugMessenger  debugMessenger(instance, debugMessageCallback);

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
    vko::Device device(instance, physicalDevice, RayTracingDeviceCreateInfo(queueFamilyIndex));

    // vko::simple::TimelineQueue queue(device, queueFamilyIndex, 0);
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
    vko::CommandPool commandPool(device, commandPoolCreateInfo);

    vko::glfw::Window     window = vko::glfw::createWindow(800, 600, "Vulkan Window");
    vko::glfw::SurfaceKHR surface(instance, platformSupport, window.get());
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
                                   VK_API_VERSION_1_4,
                                   VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
    VkExtent3D           imageExtent = {800U, 600U, 1U};
    vko::BoundImage      image(device,
                               VkImageCreateInfo{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                                 .pNext       = nullptr,
                                                 .flags       = 0,
                                                 .imageType   = VK_IMAGE_TYPE_2D,
                                                 .format      = VK_FORMAT_B8G8R8A8_UNORM,
                                                 .extent      = imageExtent,
                                                 .mipLevels   = 1,
                                                 .arrayLayers = 1,
                                                 .samples     = VK_SAMPLE_COUNT_1_BIT,
                                                 .tiling      = VK_IMAGE_TILING_OPTIMAL,
                                                 .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                     VK_IMAGE_USAGE_STORAGE_BIT,
                                                 .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                                                 .queueFamilyIndexCount = 0,
                                                 .pQueueFamilyIndices   = nullptr,
                                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED},
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    vko::ImageView       imageView(device, VkImageViewCreateInfo{
                                               .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                               .pNext    = nullptr,
                                               .flags    = 0,
                                               .image    = image,
                                               .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                               .format   = VK_FORMAT_B8G8R8A8_UNORM,
                                               .components = VkComponentMapping{},
                                               .subresourceRange =
                                             VkImageSubresourceRange{
                                                       .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                                       .baseMipLevel   = 0,
                                                       .levelCount     = 1,
                                                       .baseArrayLayer = 0,
                                                       .layerCount     = 1,
                                             },
                                     });
    {
        vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
        vko::cmdImageBarrier(device, cmd, image,
                             {
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                             },
                             {
                                 VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_GENERAL,
                             });
    }

    vko::BoundBuffer<uint32_t> triangles = uploadImmediate<uint32_t>(
        allocator, commandPool, queue, device, std::to_array({0U, 1U, 2U, 0U, 2U, 3U}),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    vko::BoundBuffer<float> vertices = uploadImmediate<float>(
        allocator, commandPool, queue, device,
        std::to_array({-1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f}),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    std::vector<vko::as::SimpleGeometryInput> simpleGeometryInputs{
        vko::as::SimpleGeometryInput{
            .triangleCount = static_cast<uint32_t>(triangles.size()),
            .maxVertex =
                static_cast<uint32_t>(vertices.size()) - 1, // Max. index one less than count
            .indexAddress  = triangles.address(device),
            .vertexAddress = vertices.address(device),
            .vertexStride  = sizeof(float) * 3,
        },
    };
    vko::as::Input blasInput = vko::as::createBlasInput(
        simpleGeometryInputs, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);
    vko::as::Sizes                 blasSizes(device, blasInput);
    vko::as::AccelerationStructure blas(device, blasInput.type, *blasSizes, 0, allocator);
    VkTransformMatrixKHR identity{.matrix = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}};
    vko::BoundBuffer<VkAccelerationStructureInstanceKHR> instances =
        uploadImmediate<VkAccelerationStructureInstanceKHR>(
            allocator, commandPool, queue, device,
            std::to_array({VkAccelerationStructureInstanceKHR{
                .transform                              = identity,
                .instanceCustomIndex                    = 0U,
                .mask                                   = 0xFFU,
                .instanceShaderBindingTableRecordOffset = 0U,
                .flags                                  = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
                .accelerationStructureReference         = blas.address(),
            }}),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    vko::as::Input tlasInput =
        vko::as::createTlasInput(uint32_t(instances.size()), instances.address(device),
                                 VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);
    vko::as::Sizes                 tlasSizes(device, tlasInput);
    vko::as::AccelerationStructure tlas(device, tlasInput.type, *tlasSizes, 0, allocator);
    vko::BoundBuffer<std::byte>    scratch(
        device, std::max(blasSizes->buildScratchSize, tlasSizes->buildScratchSize),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    {
        vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
        vko::as::cmdBuild(device, cmd, blas, blasInput, false, scratch);
        vko::cmdMemoryBarrier(device, cmd,
                              {VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR},
                              {VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR});
        vko::as::cmdBuild(device, cmd, tlas, tlasInput, false, scratch);
        vko::cmdMemoryBarrier(device, cmd,
                              {VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR},
                              {VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
    }

    vko::slang::GlobalSession globalSession;
    slang::TargetDesc         targets[]     = {{
                    .format  = SLANG_SPIRV,
                    .profile = globalSession->findProfile("spirv_1_6"),
    }};
    const char*               searchPaths[] = {"test/shaders"};
    vko::slang::Session       session(globalSession,
                                      slang::SessionDesc{
                                          .targets         = targets,
                                          .targetCount     = SlangInt(std::size(targets)),
                                          .searchPaths     = searchPaths,
                                          .searchPathCount = SlangInt(std::size(searchPaths)),

                                });
    vko::slang::Module       raytraceModule(session, "raytrace");
    ::slang::IComponentType* raytraceEntrypoints[] = {
        vko::slang::EntryPoint(raytraceModule, "rayGenMain"),
        vko::slang::EntryPoint(raytraceModule, "anyHitMain"),
        vko::slang::EntryPoint(raytraceModule, "closestHitMain"),
        vko::slang::EntryPoint(raytraceModule, "missMain"),
    };
    vko::slang::Composition raytraceComposition(session, raytraceEntrypoints);
    vko::slang::Program     raytraceProgram(raytraceComposition);
    vko::slang::Code        rayGenCode(raytraceProgram, 0, 0);
    vko::slang::Code        anyHitCode(raytraceProgram, 1, 0);
    vko::slang::Code        closestHitCode(raytraceProgram, 2, 0);
    vko::slang::Code        missCode(raytraceProgram, 3, 0);
    auto                    makeModule = [&](vko::slang::Code code) {
        return vko::ShaderModule(
            device,
            VkShaderModuleCreateInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                        .pNext    = nullptr,
                                                        .flags    = 0,
                                                        .codeSize = uint32_t(code.size()),
                                                        .pCode = reinterpret_cast<const uint32_t*>(code.data())});
    };
    vko::ShaderModule rayGen     = makeModule(rayGenCode);
    vko::ShaderModule anyHit     = makeModule(anyHitCode);
    vko::ShaderModule closestHit = makeModule(closestHitCode);
    vko::ShaderModule miss       = makeModule(missCode);

    struct RtPushConstants {
        VkExtent3D imageSize;
    };
    vko::BindingsAndFlags bindings{
        {VkDescriptorSetLayoutBinding{.binding = 0,
                                      .descriptorType =
                                          VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                      .descriptorCount    = 1,
                                      .stageFlags         = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                      .pImmutableSamplers = nullptr},
         VkDescriptorSetLayoutBinding{.binding            = 1,
                                      .descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .descriptorCount    = 1,
                                      .stageFlags         = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                      .pImmutableSamplers = nullptr}},
        {0, 0}};
    vko::SingleDescriptorSet                            descriptorSet(device, bindings, 0,
                                                                      VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
    vko::simple::RayTracingPipeline<RtPushConstants, 4> rtPipeline(
        device, std::to_array({static_cast<VkDescriptorSetLayout>(descriptorSet.layout)}), rayGen,
        anyHit, closestHit, miss);
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties =
        vko::simple::rayTracingPipelineProperties(instance, physicalDevice);
    vko::simple::HitGroupHandles     hitGroupHandles(device, rtPipelineProperties, rtPipeline, 3);
    vko::simple::ShaderBindingTables sbt(
        device, commandPool, queue,
        vko::simple::ShaderBindingTablesStaging(
            allocator, device, rtPipelineProperties, {hitGroupHandles[0]}, {hitGroupHandles[1]},
            {hitGroupHandles[2]}, std::initializer_list<std::span<const std::byte>>{}),
        allocator);

    for (;;) {
        int width, height;
        glfwGetWindowSize(window.get(), &width, &height);
        vko::simple::Swapchain swapchain{
            device,           surface,
            surfaceFormat,    VkExtent2D{uint32_t(width), uint32_t(height)},
            queueFamilyIndex, surfacePresentMode,
            VK_NULL_HANDLE};

        vko::WriteDescriptorSetBuilder writes;
        writes.push_back<VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR>(
            descriptorSet.set, bindings.bindings[0], 0, tlas);
        writes.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
            descriptorSet.set, bindings.bindings[1], 0,
            VkDescriptorImageInfo{.sampler     = VK_NULL_HANDLE,
                                  .imageView   = imageView,
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
        device.vkUpdateDescriptorSets(device, writes.writes().size(), writes.writes().data(), 0U,
                                      nullptr);
        // device.vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        // rtPipeline.layout(), 0U, writes.writes().size(), writes.writes().data());

        auto [imageIndex, reuseImageSemaphore] = swapchain.acquire(device, 0ULL);
        VkSemaphore renderingFinished          = swapchain.renderFinishedSemaphores[imageIndex];

        {
            vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
            cmd.addWait(reuseImageSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            cmd.addSignal(renderingFinished);
            vko::simple::clearSwapchainImage(
                device, cmd, swapchain.images[imageIndex],
                swapchain.presented[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkClearColorValue{.float32 = {1.0f, 1.0f, 0.0f, 1.0f}});

            RtPushConstants pushConstant{imageExtent};
            device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
            device.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                           rtPipeline.layout(), 0, 1U, descriptorSet.set.ptr(), 0,
                                           nullptr);
            device.vkCmdPushConstants(cmd, rtPipeline.layout(), VK_SHADER_STAGE_ALL, 0,
                                      sizeof(pushConstant), &pushConstant);
            device.vkCmdTraceRaysKHR(cmd, &sbt.raygenTableOffset, &sbt.missTableOffset,
                                     &sbt.hitTableOffset, &sbt.callableTableOffset,
                                     imageExtent.width, imageExtent.height, 1);
            vko::cmdImageBarrier(device, cmd, image,
                                 {
                                     VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                 },
                                 {
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_ACCESS_TRANSFER_READ_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                 });

            VkImageCopy region{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .srcOffset      = {0, 0, 0},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .dstOffset      = {0, 0, 0},
                .extent         = {std::min(imageExtent.width, uint32_t(width)),
                                   std::min(imageExtent.height, uint32_t(height)), 1U},
            };
            device.vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, swapchain.images[imageIndex],
                                  VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            vko::cmdImageBarrier(device, cmd, swapchain.images[imageIndex],
                                 {
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                 },
                                 {
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     0U,
                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 });
        }

        swapchain.present(device, queue, imageIndex, renderingFinished);
        device.vkQueueWaitIdle(queue);
        break;
    }
}
