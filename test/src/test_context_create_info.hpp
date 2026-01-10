// Copyright (c) 2025 Pyarelal Knowles, MIT License

#pragma once

#include <array>
#include <string_view>
#include <vko/adapters.hpp>
#include <vko/glfw_objects.hpp>
#include <vko/handles.hpp>
#include <vulkan/vulkan_core.h>
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

// Combined vulkan features struct initialized with required flags set.
// Non-copyable so the pNext chain remains valid.
struct TestDeviceFeatures {
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
    bool hasAll(const TestDeviceFeatures& required) {
        // For demonstration. Ideally this code would be generated for vulkan
        // feature structs
        if (required.vulkan13Feature.dynamicRendering && !vulkan13Feature.dynamicRendering)
            return false;
        if (required.shaderObjectFeature.shaderObject && !shaderObjectFeature.shaderObject)
            return false;
        // ...
        return true;
    }
    TestDeviceFeatures()                                          = default;
    TestDeviceFeatures(const TestDeviceFeatures& other)           = delete;
    TestDeviceFeatures operator=(const TestDeviceFeatures& other) = delete;
    void*              pNext() { return &vulkan13Feature; }
};

struct TestDeviceCreateInfo {
    std::vector<const char*>             deviceExtensions;
    float                                queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    TestDeviceFeatures                   features;
    VkDeviceCreateInfo                   deviceCreateInfo;
    
    template <std::ranges::range QueueFamilyIndices>
    TestDeviceCreateInfo(const QueueFamilyIndices& queueFamilyIndices,
                         std::span<const char* const> optionalExtensions = {})
        : deviceExtensions([&optionalExtensions]() {
            std::vector<const char*> exts{VK_KHR_SWAPCHAIN_EXTENSION_NAME, 
                                          VK_EXT_SHADER_OBJECT_EXTENSION_NAME};
            exts.insert(exts.end(), optionalExtensions.begin(), optionalExtensions.end());
            return exts;
        }())
        , queueCreateInfos(makeQueueCreateInfos(queueFamilyIndices))
        , deviceCreateInfo{
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                   = features.pNext(),
            .flags                   = 0,
            .queueCreateInfoCount    = uint32_t(queueCreateInfos.size()),
            .pQueueCreateInfos       = queueCreateInfos.data(),
            .enabledLayerCount       = 0,
            .ppEnabledLayerNames     = nullptr,
            .enabledExtensionCount   = uint32_t(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures        = nullptr
        } {}
    
    TestDeviceCreateInfo(uint32_t queueFamilyIndex,
                         std::span<const char* const> optionalExtensions = {})
        : TestDeviceCreateInfo(std::array{queueFamilyIndex}, optionalExtensions) {}
    
    TestDeviceCreateInfo(uint32_t queueFamilyIndex1, uint32_t queueFamilyIndex2,
                         std::span<const char* const> optionalExtensions = {})
        : TestDeviceCreateInfo(std::array{queueFamilyIndex1, queueFamilyIndex2}, optionalExtensions) {}
    
    operator VkDeviceCreateInfo&() { return deviceCreateInfo; }
    TestDeviceCreateInfo(const TestDeviceCreateInfo& other) = delete;
    TestDeviceCreateInfo operator=(const TestDeviceCreateInfo& other) = delete;

private:
    template <std::ranges::range QueueFamilyIndices>
    std::vector<VkDeviceQueueCreateInfo> makeQueueCreateInfos(const QueueFamilyIndices& indices) {
        std::vector<VkDeviceQueueCreateInfo> infos;
        infos.reserve(std::ranges::size(indices));
        for (uint32_t familyIndex : indices) {
            infos.push_back(VkDeviceQueueCreateInfo{
                .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext            = nullptr,
                .flags            = 0,
                .queueFamilyIndex = familyIndex,
                .queueCount       = 1,
                .pQueuePriorities = &queuePriority
            });
        }
        return infos;
    }
};

inline bool queueSuitable(const VkQueueFamilyProperties& properties) {
    VkQueueFlags requiredFlags =
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    return (properties.queueFlags & requiredFlags) == requiredFlags;
};

inline bool physicalDeviceSuitable(const vko::Instance& instance, VkPhysicalDevice physicalDevice) {
    // Check device has required features
    TestDeviceFeatures hasFeatures; // Reuse the required features pNext chain for querying
    VkPhysicalDeviceFeatures2 features2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                        .pNext = hasFeatures.pNext(),
                                        .features = {}};
    instance.vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    if (features2.features.shaderInt64 != VK_TRUE)
        return false; // Don't really care. Purely for demonstration
    if (!hasFeatures.hasAll(TestDeviceFeatures{}))
        return false; // TODO: move features2 into TestDeviceFeatures
    if (!std::ranges::any_of(
            vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice),
            queueSuitable))
        return false;

    // Check device has required extensions
    auto extensions =
        vko::toVector(instance.vkEnumerateDeviceExtensionProperties, physicalDevice, nullptr);
    for (auto required : TestDeviceCreateInfo(0).deviceExtensions)
        if (std::ranges::none_of(extensions, [&required](const VkExtensionProperties& p) {
                return std::string_view(p.extensionName) == required;
            }))
            return false;

    // Check device has required properties
    // TODO: could return a priority and choose the highest
    VkPhysicalDeviceProperties properties =
        vko::get(instance.vkGetPhysicalDeviceProperties, physicalDevice);
    if (false && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
        return false; // For demonstration
    return true;
};

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
