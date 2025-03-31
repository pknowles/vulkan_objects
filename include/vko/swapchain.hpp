// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <limits>
#include <vko/adapters.hpp>
#include <vko/handles.hpp>

namespace vko {

namespace simple {

// Swapchain with images and views created
struct Swapchain {
    // Simple/default create info. Use the other constructor to override
    template <class DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, VkSurfaceKHR surface,
              VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
              VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : Swapchain(device,
                    VkSwapchainCreateInfoKHR{
                        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                        .pNext            = nullptr,
                        .flags            = 0,
                        .surface          = surface,
                        .minImageCount    = 2,
                        .imageFormat      = surfaceFormat.format,
                        .imageColorSpace  = surfaceFormat.colorSpace,
                        .imageExtent      = extent,
                        .imageArrayLayers = 1,
                        .imageUsage =
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
                        .queueFamilyIndexCount = 1,
                        .pQueueFamilyIndices   = &queueFamilyIndex,
                        .preTransform          = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                        .compositeAlpha =
                            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, // (enum type, not uint bitfield)
                        .presentMode  = presentMode,
                        .clipped      = VK_FALSE,
                        .oldSwapchain = oldSwapchain,
                    }) {}

    template <class DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : swapchain(device, createInfo)
        , acquiredImageSemaphore(
              device,
              {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = nullptr, .flags = 0})
        , images(vko::toVector(device.vkGetSwapchainImagesKHR, device, swapchain)) {
        imageViews.reserve(images.size());
        presented.resize(images.size(), false);
        renderFinishedSemaphores.reserve(images.size());
        for (auto& image : images) {
            imageViews.emplace_back(device, VkImageViewCreateInfo{
                                                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                .pNext = nullptr,
                                                .flags = 0,
                                                .image = image,
                                                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                                                .format     = createInfo.imageFormat,
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
            renderFinishedSemaphores.emplace_back(
                device, VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0});
        }
    }
    operator VkSwapchainKHR() const { return swapchain; }

    std::pair<uint32_t, VkSemaphore> acquire(const Device& device, uint64_t timeout) {
        // TODO: alternative for exceptions? In particular VK_TIMEOUT,
        // VK_NOT_READY or VK_SUBOPTIMAL_KHR might be common and not
        // "exceptional"
        auto imageIndex = std::numeric_limits<uint32_t>::max();
        check(device.vkAcquireNextImageKHR(device, swapchain, timeout, acquiredImageSemaphore,
                                           VK_NULL_HANDLE, &imageIndex));
        return {imageIndex, acquiredImageSemaphore};
    }

    struct SyncedImage {
        VkImage     image;
        VkImageView imageView;
        VkSemaphore acquiredImageSemaphore;
        VkSemaphore renderFinishedSemaphore;
    };

    SyncedImage operator[](uint32_t index) const {
        return {
            images[index],
            imageViews[index],
            acquiredImageSemaphore,
            renderFinishedSemaphores[index],
        };
    }

    template <class DeviceAndCommands>
    void present(const DeviceAndCommands& device, VkQueue queue, uint32_t index,
                 VkSemaphore renderFinished) {
        VkPresentInfoKHR presentInfo{
            .sType{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR},
            .pNext{nullptr},
            .waitSemaphoreCount{1U},
            .pWaitSemaphores{&renderFinished},
            .swapchainCount{1U},
            .pSwapchains{swapchain.ptr()},
            .pImageIndices{&index},
            .pResults{nullptr},
        };
        check(device.vkQueuePresentKHR(queue, &presentInfo));
        presented[index] = true;
    }

    SwapchainKHR           swapchain;
    Semaphore              acquiredImageSemaphore;
    std::vector<VkImage>   images; // non-owning
    std::vector<ImageView> imageViews;
    std::vector<Semaphore> renderFinishedSemaphores;
    std::vector<bool>      presented; // first-use flag, implying VK_IMAGE_LAYOUT_UNDEFINED
};

template <class DeviceAndCommands>
inline void clearSwapchainImage(const DeviceAndCommands& device, VkCommandBuffer commandBuffer,
                                VkImage image, VkImageLayout srcLayout, VkImageLayout dstLayout,
                                VkAccessFlags dstAccess, VkPipelineStageFlags dstStage,
                                VkClearColorValue clearColorValue) {
    VkImageMemoryBarrier imagePresentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                             nullptr,
                                             0U,
                                             VK_ACCESS_TRANSFER_WRITE_BIT,
                                             srcLayout,
                                             VK_IMAGE_LAYOUT_GENERAL,
                                             0U,
                                             0U,
                                             image,
                                             {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U}};
    device.vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U,
                                &imagePresentBarrier);

    VkImageSubresourceRange subresourceRange{.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                             .baseMipLevel   = 0,
                                             .levelCount     = 1,
                                             .baseArrayLayer = 0,
                                             .layerCount     = 1};
    device.vkCmdClearColorImage(commandBuffer, image, VK_IMAGE_LAYOUT_GENERAL, &clearColorValue, 1,
                                &subresourceRange);

    VkImageMemoryBarrier imageAttachmentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                nullptr,
                                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                                dstAccess,
                                                VK_IMAGE_LAYOUT_GENERAL,
                                                dstLayout,
                                                0U,
                                                0U,
                                                image,
                                                {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U}};
    device.vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0U, 0U,
                                nullptr, 0U, nullptr, 1U, &imageAttachmentBarrier);
}

} // namespace simple

} // namespace vko