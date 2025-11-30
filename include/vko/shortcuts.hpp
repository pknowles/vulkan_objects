// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/bound_buffer.hpp>
#include <vko/bound_image.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>

namespace vko {

// Guess the image aspect mask from format
// Returns sensible defaults for all standard formats
inline VkImageAspectFlags guessAspect(VkFormat format) {
    switch (format) {
    // Depth-only formats
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    
    // Stencil-only formats
    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    
    // Combined depth-stencil formats (return both aspects)
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    
    // All other formats are color
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

template <device_and_commands DeviceAndCommands = Device>
void cmdDynamicRenderingDefaults(const DeviceAndCommands& device, VkCommandBuffer cmd,
                                 uint32_t width, uint32_t height) {
    VkViewport            viewport{0.0F, 0.0F, float(width), float(height), 0.0F, 1.0F};
    VkRect2D              scissor{{0, 0}, {width, height}};
    VkSampleMask          sampleMask      = 0xFU;
    VkBool32              blendEnabled    = VK_FALSE;
    VkColorComponentFlags colorComponents = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    device.vkCmdSetVertexInputEXT(cmd, 0U, nullptr, 0U, nullptr);
    device.vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    device.vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
    device.vkCmdSetViewportWithCount(cmd, 1U, &viewport);
    device.vkCmdSetScissorWithCount(cmd, 1U, &scissor);
    device.vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
    device.vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
    device.vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    device.vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
    device.vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
    device.vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
    device.vkCmdSetFrontFace(cmd, VK_FRONT_FACE_CLOCKWISE);
    device.vkCmdSetDepthTestEnable(cmd, VK_FALSE);
    device.vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
    device.vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
    device.vkCmdSetStencilTestEnable(cmd, VK_FALSE);
    device.vkCmdSetColorBlendEnableEXT(cmd, 0U, 1U, &blendEnabled);
    device.vkCmdSetColorWriteMaskEXT(cmd, 0U, 1U, &colorComponents);
}

struct ImageAccess {
    VkPipelineStageFlags stage  = 0U;
    VkAccessFlags        access = 0U;
    VkImageLayout        layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

template <device_and_commands DeviceAndCommands = Device>
void cmdImageBarrier(const DeviceAndCommands& device, VkCommandBuffer cmd, VkImage image,
                     ImageAccess src, ImageAccess dst,
                     VkImageAspectFlagBits aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier imageBarrier{.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                      .pNext               = nullptr,
                                      .srcAccessMask       = src.access,
                                      .dstAccessMask       = dst.access,
                                      .oldLayout           = src.layout,
                                      .newLayout           = dst.layout,
                                      .srcQueueFamilyIndex = 0U,
                                      .dstQueueFamilyIndex = 0U,
                                      .image               = image,
                                      .subresourceRange{aspect, 0U, 1U, 0U, 1U}};
    device.vkCmdPipelineBarrier(cmd, src.stage, dst.stage, 0U, 0U, nullptr, 0U, nullptr, 1U,
                                &imageBarrier);
}

struct MemoryAccess {
    VkPipelineStageFlags stage  = 0U;
    VkAccessFlags        access = 0U;
};

template <device_and_commands DeviceAndCommands = Device>
void cmdMemoryBarrier(const DeviceAndCommands& device, VkCommandBuffer cmd, MemoryAccess src,
                      MemoryAccess dst) {
    VkMemoryBarrier memoryBarrier{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .pNext         = nullptr,
                                  .srcAccessMask = src.access,
                                  .dstAccessMask = dst.access};
    device.vkCmdPipelineBarrier(cmd, src.stage, dst.stage, 0U, 1U, &memoryBarrier, 0U, nullptr, 0U,
                                nullptr);
}

// Debug messenger with a global callback (not using the user data pointer)
struct GlobalDebugMessenger {
    template <instance_and_commands InstanceAndCommands = Instance>
    GlobalDebugMessenger(const InstanceAndCommands&           vk,
                         PFN_vkDebugUtilsMessengerCallbackEXT callback)
        : messenger(vk, VkDebugUtilsMessengerCreateInfoEXT{
                            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                            .pNext = nullptr,
                            .flags = 0,
                            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                            .pfnUserCallback = callback,
                            .pUserData       = nullptr}) {}
    vko::DebugUtilsMessengerEXT messenger;
};

struct DebugMessenger {
    using Callback = std::function<bool(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                           const VkDebugUtilsMessengerCallbackDataEXT&)>;

    template <instance_and_commands InstanceAndCommands = Instance, class Fn = Callback>
    DebugMessenger(const InstanceAndCommands& vk, Fn&& callback)
        : callback(std::forward<Fn>(callback))
        , messenger(vk, VkDebugUtilsMessengerCreateInfoEXT{
                            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                            .pNext = nullptr,
                            .flags = 0,
                            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                            .pfnUserCallback = debugMessageCallback,
                            .pUserData       = &callback}) {}
    vko::DebugUtilsMessengerEXT                         messenger;
    Callback callback;
    static VkBool32 debugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                         VkDebugUtilsMessageTypeFlagsEXT        messageTypes,
                                         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                         void*                                       pUserData) {
        return (*reinterpret_cast<Callback*>(pUserData))(messageSeverity, messageTypes,
                                                                   *pCallbackData)
                   ? VK_TRUE
                   : VK_FALSE;
    }
};

// Assumes the caller will add a pipeline barrier to make the vkCmdCopyBuffer()
// visible to the next user of the buffer
template <class T, vko::device_and_commands DeviceAndCommands = Device,
          class Allocator = vko::vma::Allocator>
DeviceBuffer<T> uploadImmediate(Allocator& allocator, VkCommandPool pool, VkQueue queue,
                                const DeviceAndCommands&       device,
                                std::span<std::add_const_t<T>> data, VkBufferUsageFlags usage) {
    BoundBuffer<T> staging(
        device, data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, allocator);
    DeviceBuffer<T> result(device, data.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        std::ranges::copy(data, staging.map().begin());
        VkBufferCopy bufferCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size      = data.size() * sizeof(T),
        };
        device.vkCmdCopyBuffer(cmd, staging, result, 1, &bufferCopy);
    }
    return result;
}

// Assumes the caller has already added a pipeline barrier to make src input
// data visible to vkCmdCopyBuffer()
template <class T, vko::device_and_commands DeviceAndCommands = Device,
          class Allocator = vko::vma::Allocator>
BoundBuffer<T> downloadImmediate(Allocator& allocator, VkCommandPool pool, VkQueue queue,
                                 const DeviceAndCommands& device, const DeviceBuffer<T>& src) {
    BoundBuffer<T> staging(
        device, src.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, allocator);
    {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        VkBufferCopy                   bufferCopy{
                              .srcOffset = 0,
                              .dstOffset = 0,
                              .size      = src.size() * sizeof(T),
        };
        device.vkCmdCopyBuffer(cmd, src, staging, 1, &bufferCopy);
        cmdMemoryBarrier(device, cmd,
                         {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
                         {VK_PIPELINE_STAGE_HOST_BIT,
                          VK_ACCESS_HOST_READ_BIT}); // not sure if host bits do anything
    }
    return staging;
}

template <class Allocator = vko::vma::Allocator>
struct ViewedImage {
    template <device_and_commands DeviceAndCommands, class AllocationCreateInfo>
    ViewedImage(const DeviceAndCommands& device, const VkImageCreateInfo imageCreateInfo,
                const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : image(device, device, imageCreateInfo, allocationCreateInfo, allocator)
        , view(device, device,
               VkImageViewCreateInfo{
                   .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                   .pNext    = nullptr,
                   .flags    = 0,
                   .image    = image,
                   .viewType = VK_IMAGE_VIEW_TYPE_2D, // TODO: compute from createInfo.imageType
                   .format     = imageCreateInfo.format,
                   .components = VkComponentMapping{},
                   .subresourceRange =
                       VkImageSubresourceRange{
                           .aspectMask     = guessAspect(imageCreateInfo.format),
                           .baseMipLevel   = 0,
                           .levelCount     = 1,
                           .baseArrayLayer = 0,
                           .layerCount     = 1,
                       },
               }) {}
    BoundImage<Allocator> image;
    ImageView             view;
    operator VkImage() const { return image; } // typing "image.image" looks weird
};

// Creates a 2D Image and View with very general usage bits. Single layer, no
// mipmapping.
template <device_and_commands DeviceAndCommands = Device, class Allocator = vko::vma::Allocator>
ViewedImage<Allocator> makeImage(const DeviceAndCommands& device, VkExtent3D imageExtent,
                                 VkFormat format, Allocator& allocator) {
    return ViewedImage{
        device,
        VkImageCreateInfo{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .pNext       = nullptr,
                          .flags       = 0,
                          .imageType   = VK_IMAGE_TYPE_2D,
                          .format      = format,
                          .extent      = imageExtent,
                          .mipLevels   = 1,
                          .arrayLayers = 1,
                          .samples     = VK_SAMPLE_COUNT_1_BIT,
                          .tiling      = VK_IMAGE_TILING_OPTIMAL,
                          .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                          .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                          .queueFamilyIndexCount = 0,
                          .pQueueFamilyIndices   = nullptr,
                          .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED},
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator};
}

// Remember to enable VK_EXT_DEBUG_UTILS_EXTENSION_NAME
template<device_and_commands Device, class Handle>
void setName(const Device& device, Handle handle, const std::string& name) {
    VkDebugUtilsObjectNameInfoEXT objectNameInfo{
        .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .pNext        = nullptr,
        .objectType   = handle_traits<Handle>::type_enum,
        .objectHandle = reinterpret_cast<uint64_t>(handle),
        .pObjectName  = name.c_str(),
    };
    device.vkSetDebugUtilsObjectNameEXT(device, &objectNameInfo);
}

} // namespace vko
