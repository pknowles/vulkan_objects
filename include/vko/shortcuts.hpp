// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/bound_buffer.hpp>
#include <vko/bound_image.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>

namespace vko {

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
template <instance_and_commands InstanceAndCommands = Instance>
struct SimpleDebugMessenger {
    SimpleDebugMessenger(const InstanceAndCommands&           vk,
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

template <class T, vko::device_and_commands DeviceAndCommands = Device,
          class Allocator = vko::vma::Allocator>
BoundBuffer<T> uploadImmediate(Allocator& allocator, VkCommandPool pool, VkQueue queue,
                               const DeviceAndCommands& device, std::span<std::add_const_t<T>> data,
                               VkBufferUsageFlags usage) {
    BoundBuffer<T> staging(
        device, data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, allocator);
    BoundBuffer<T> result(device, data.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
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

template <class Allocator = vko::vma::Allocator>
struct ViewedImage {
    template <device_and_commands DeviceAndCommands, class AllocationCreateInfo>
    ViewedImage(const DeviceAndCommands& device, const VkImageCreateInfo createInfo,
                const AllocationCreateInfo& allocationCreateInfo, Allocator& allocator)
        : image(device, device, createInfo, allocationCreateInfo, allocator)
        , view(device, device,
               VkImageViewCreateInfo{
                   .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                   .pNext    = nullptr,
                   .flags    = 0,
                   .image    = image,
                   .viewType = VK_IMAGE_VIEW_TYPE_2D, // TODO: should be able to compute
                                                      // this based on createInfo.imageType
                   .format     = createInfo.format,
                   .components = VkComponentMapping{},
                   .subresourceRange =
                       VkImageSubresourceRange{
                           .aspectMask =
                               VK_IMAGE_ASPECT_COLOR_BIT, // TODO: should be able to compute
                                                          // this based on the format
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

template <device_and_commands DeviceAndCommands = Device, class Allocator = vko::vma::Allocator>
ViewedImage<Allocator> makeImage(const DeviceAndCommands& device, VkExtent3D imageExtent,
                                 VkFormat format, Allocator& allocator) {
    return ViewedImage{device,
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
                                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                  VK_IMAGE_USAGE_STORAGE_BIT,
                                         .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                                         .queueFamilyIndexCount = 0,
                                         .pQueueFamilyIndices   = nullptr,
                                         .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED},
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator};
}

} // namespace vko
