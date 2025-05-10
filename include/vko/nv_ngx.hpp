// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

// Assumes https://github.com/NVIDIA/DLSS is made available by the build system
// See the readme at https://github.com/nvpro-samples/vk_denoise_dlssrr

#include <codecvt>
#include <locale>
#include <memory>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_vk.h>

// *sigh* must be after nvsdk_ngx_helpers_vk.h
#include <nvsdk_ngx_helpers_dlssd_vk.h>

/*
    // Example usage:

    // Create the instance and device with the required extensions
    // Make sure to set NVSDK_NGX_FeatureCommonInfo::PathListInfo as this is
    // where NGX goes looking for DLLs implementing its features.
    vko::ngx::requiredInstanceExtensions(NVSDK_NGX_FeatureDiscoveryInfo{...})
    ...
    vko::ngx::requiredDeviceExtensions(instance, physicalDevice,
                                       NVSDK_NGX_FeatureDiscoveryInfo{...})

    // Global init :( *shakes fist*
    vko::ngx::ScopedInit ngx(ngxApplicationId, ngxApplicationPath, instance, physicalDevice, device,
                             globalCommands.vkGetInstanceProcAddr, instance.vkGetDeviceProcAddr,
                             &ngxCommonInfo)

    // Dependent on output window size/resize
    vko::ngx::CapabilityParameter ngxParameter;
    vko::ngx::OptimalSettings dlssOptimal(ngxParameter, outputSize.x, outputSize.y,
   NVSDK_NGX_PerfQuality_Value_MaxQuality);
    ... resize G-buffer to {dlssOptimal.renderOptimalWidth, dlssOptimal.renderOptimalHeight}
    vko::ngx::RayReconstruction dlssrr(device, cmd, 1u, 1u, ngxParameter,
   NVSDK_NGX_DLSSD_Create_Params{...});

    // Upscale and denoise
    glm::mat4 viewRowMajor       = glm::transpose(viewMatrix);
    glm::mat4 projectionRowMajor = glm::transpose(projectionMatrix);
    // NOTE: Make sure these have VK_IMAGE_USAGE_SAMPLED_BIT!
    NVSDK_NGX_Resource_VK colorOut = NVSDK_NGX_Create_ImageView_Resource_VK(...);
    NVSDK_NGX_Resource_VK color    = NVSDK_NGX_Create_ImageView_Resource_VK(...);
    ...
    NVSDK_NGX_VK_DLSSD_Eval_Params dlssdEvalParams{};
    dlssdEvalParams.pInOutput                        = &colorOut;
    dlssdEvalParams.pInColor                         = &color;
    dlssdEvalParams.pInDiffuseAlbedo                 = &albedo;
    dlssdEvalParams.pInSpecularAlbedo                = &specularAlbedo;
    dlssdEvalParams.pInSpecularHitDistance           = &specularHitDistance;
    dlssdEvalParams.pInNormals                       = &normalRoughness;
    dlssdEvalParams.pInDepth                         = &linearDepth;
    dlssdEvalParams.pInMotionVectors                 = &motionVector;
    dlssdEvalParams.pInRoughness                     = &normalRoughness;
    dlssdEvalParams.InJitterOffsetX                  = -jitter.x;
    dlssdEvalParams.InJitterOffsetY                  = -jitter.y;
    dlssdEvalParams.InMVScaleX                       = 1.0f;
    dlssdEvalParams.InMVScaleY                       = 1.0f;
    dlssdEvalParams.InRenderSubrectDimensions.Width  = inputSize.x;
    dlssdEvalParams.InRenderSubrectDimensions.Height = inputSize.y;
    dlssdEvalParams.pInWorldToViewMatrix             = glm::value_ptr(viewRowMajor);
    dlssdEvalParams.pInViewToClipMatrix              = glm::value_ptr(projectionRowMajor);
    dlssdEvalParams.InReset                          = frameIndex <= 1;
    dlssrr->evaluate(commandBuffer, ngxParameter, dlssdEvalParams);
*/

namespace vko {
namespace ngx {

// template <NVSDK_NGX_Result result>
class ResultException : public Exception {
public:
    ResultException(NVSDK_NGX_Result result)
        : Exception("NGX error: " + std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(
                                        GetNGXResultAsString(result))) {}
};

inline void check(NVSDK_NGX_Result result) {
    if (NVSDK_NGX_FAILED(result)) {
        throw ResultException(result);
    }
}

// Straight init wrapper and error handling
class ScopedInit {
public:
    ScopedInit(unsigned long long applicationId, const std::wstring& applicationDataPath,
               VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
               PFN_vkGetInstanceProcAddr          vkGetInstanceProcAddr,
               PFN_vkGetDeviceProcAddr            vkGetDeviceProcAddr,
               const NVSDK_NGX_FeatureCommonInfo* featureInfo = nullptr,
               NVSDK_NGX_Version                  sdkVersion  = NVSDK_NGX_Version_API)
        : m_device(device) {
        check(NVSDK_NGX_VULKAN_Init(applicationId, applicationDataPath.c_str(), instance,
                                    physicalDevice, device, vkGetInstanceProcAddr,
                                    vkGetDeviceProcAddr, featureInfo, sdkVersion));
    }
    ScopedInit(const std::string& projectId, NVSDK_NGX_EngineType engineType,
               const std::string& engineVersion, const std::wstring& applicationDataPath,
               VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
               PFN_vkGetInstanceProcAddr          vkGetInstanceProcAddr,
               PFN_vkGetDeviceProcAddr            vkGetDeviceProcAddr,
               const NVSDK_NGX_FeatureCommonInfo* featureInfo = nullptr,
               NVSDK_NGX_Version                  sdkVersion  = NVSDK_NGX_Version_API)
        : m_device(device) {
        check(NVSDK_NGX_VULKAN_Init_with_ProjectID(
            projectId.c_str(), engineType, engineVersion.c_str(), applicationDataPath.c_str(),
            instance, physicalDevice, device, vkGetInstanceProcAddr, vkGetDeviceProcAddr,
            featureInfo, sdkVersion));
    }
    ~ScopedInit() { check(NVSDK_NGX_VULKAN_Shutdown1(m_device)); }
    ScopedInit(const ScopedInit& other)           = delete;
    ScopedInit operator=(const ScopedInit& other) = delete;

private:
    VkDevice m_device = VK_NULL_HANDLE;
};

namespace {
struct ParameterDeleter {
    void operator()(NVSDK_NGX_Parameter* p) const { NVSDK_NGX_VULKAN_DestroyParameters(p); }
};

std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter> capabilityParameter() {
    NVSDK_NGX_Parameter* parameter = nullptr;
    check(NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameter));
    return std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter>{parameter};
}
} // anonymous namespace

struct CapabilityParameter : std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter> {
    CapabilityParameter()
        : std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter>(capabilityParameter()) {}
};

template <class T>
T get(const NVSDK_NGX_Parameter& parameter, const char* name) {
    T result;
    check(parameter.Get(name, &result));
    return result;
}

inline std::span<const VkExtensionProperties>
requiredInstanceExtensions(const NVSDK_NGX_FeatureDiscoveryInfo& featureDiscoveryInfo) {
    uint32_t               extensionCount = 0;
    VkExtensionProperties* extensions     = nullptr;
    check(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&featureDiscoveryInfo,
                                                                   &extensionCount, &extensions));
    return {extensions, extensionCount};
}

inline std::span<const VkExtensionProperties>
requiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                         const NVSDK_NGX_FeatureDiscoveryInfo& featureDiscoveryInfo) {
    uint32_t               extensionCount = 0;
    VkExtensionProperties* extensions     = nullptr;
    check(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance, physicalDevice, &featureDiscoveryInfo, &extensionCount, &extensions));
    return {extensions, extensionCount};
}

struct OptimalSettings {
    OptimalSettings(NVSDK_NGX_Parameter& parameter, unsigned int selectedWidth,
                    unsigned int selectedHeight, NVSDK_NGX_PerfQuality_Value qualityValue) {
        check(NGX_DLSSD_GET_OPTIMAL_SETTINGS(
            &parameter, selectedWidth, selectedHeight, qualityValue, &renderOptimalWidth,
            &renderOptimalHeight, &renderMaxWidth, &renderMaxHeight, &renderMinWidth,
            &renderMinHeight, &sharpness));
    }
    unsigned int renderOptimalWidth;
    unsigned int renderOptimalHeight;
    unsigned int renderMaxWidth;
    unsigned int renderMaxHeight;
    unsigned int renderMinWidth;
    unsigned int renderMinHeight;
    float        sharpness;
};

template <auto CreateFunc>
class Handle {
public:
    template <class... Args>
    Handle(Args&&... args)
        : m_handle(CreateFunc(std::forward<Args>(args)...)) {}
    ~Handle() { destroy(); }
    Handle(const Handle&)            = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(other.m_handle) {
        other.m_handle = nullptr;
    }
    Handle& operator=(Handle&& other) noexcept {
        destroy();
        m_handle       = other.m_handle;
        other.m_handle = nullptr;
        return *this;
    }
    operator NVSDK_NGX_Handle*() const { return m_handle; }

private:
    void destroy() {
        if (m_handle)
            NVSDK_NGX_VULKAN_ReleaseFeature(m_handle);
    }
    NVSDK_NGX_Handle* m_handle = nullptr;
};

namespace {

inline NVSDK_NGX_Handle* makeRayReconstruction(
    VkDevice device, VkCommandBuffer commandBuffer, unsigned int creationNodeMask,
    unsigned int visibilityNodeMask, NVSDK_NGX_Parameter& parameter,
    NVSDK_NGX_DLSSD_Create_Params&
        dlssDCreateParams) { // To NGX: dlssDCreateParams should be able to be const
    NVSDK_NGX_Handle* handle = nullptr;
    check(NGX_VULKAN_CREATE_DLSSD_EXT1(device, commandBuffer, creationNodeMask, visibilityNodeMask,
                                       &handle, &parameter, &dlssDCreateParams));
    return handle;
}

// Example/demonstration
inline void assertRayReconstructionSupported(NVSDK_NGX_Parameter& parameter) {
    auto minMajor =
        get<unsigned>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor);
    auto minMinor =
        get<unsigned>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor);
    if (get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver) != 0)
        throw Exception("NGX Super Sampling Denoising needs a driver update: Min. version " +
                        std::to_string(minMajor) + "." + std::to_string(minMinor));
    if (get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available) == 0)
        throw Exception("NGX Super Sampling Denoising is not available");
    if (get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult) == 0)
        throw Exception("NGX Super Sampling Denoising FeatureInitResult was 0");
}

} // anonymous namespace

class RayReconstruction : public Handle<makeRayReconstruction> {
public:
    using Handle<makeRayReconstruction>::Handle;
    void evaluate(VkCommandBuffer commandBuffer, NVSDK_NGX_Parameter& parameter,
                  NVSDK_NGX_VK_DLSSD_Eval_Params&
                      evalParams) { // To NGX: evalParams should be able to be const
        check(NGX_VULKAN_EVALUATE_DLSSD_EXT(commandBuffer, *this, &parameter, &evalParams));
    }
};

} // namespace ngx
} // namespace vko
