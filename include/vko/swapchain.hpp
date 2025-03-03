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
    Swapchain(VkSurfaceKHR surface, VkExtent2D extent, uint32_t queueFamilyIndex,
              VkSwapchainKHR oldSwapchain, const Device& device)
        : Swapchain(
              VkSwapchainCreateInfoKHR{
                  .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                  .pNext                 = nullptr,
                  .flags                 = 0,
                  .surface               = surface,
                  .minImageCount         = 3,
                  .imageFormat           = VK_FORMAT_R8G8B8A8_UNORM,
                  .imageColorSpace       = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                  .imageExtent           = extent,
                  .imageArrayLayers      = 1,
                  .imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                  .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
                  .queueFamilyIndexCount = 1,
                  .pQueueFamilyIndices   = &queueFamilyIndex,
                  .preTransform          = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                  .compositeAlpha =
                      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, // (enum type, not uint bitfield)
                  .presentMode  = VK_PRESENT_MODE_MAILBOX_KHR,
                  .clipped      = VK_FALSE,
                  .oldSwapchain = oldSwapchain,
              },
              device) {}

    Swapchain(const VkSwapchainCreateInfoKHR& createInfo, const Device& device)
        : swapchain(createInfo, device)
        , acquiredImageSemaphore(
              {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = nullptr, .flags = 0},
              device)
        , images(vko::toVector(device.vkGetSwapchainImagesKHR, device, swapchain)) {
        imageViews.reserve(images.size());
        renderFinishedSemaphores.reserve(images.size());
        for (auto& image : images) {
            imageViews.emplace_back(
                VkImageViewCreateInfo{
                    .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext      = nullptr,
                    .flags      = 0,
                    .image      = image,
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
                },
                device);
            renderFinishedSemaphores.emplace_back(
                VkSemaphoreCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = nullptr, .flags = 0},
                device);
        }
    }
    operator VkSwapchainKHR() const { return swapchain; }

    std::pair<uint32_t, VkSemaphore> acquire(uint64_t timeout, const Device& device) {
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

    void present(VkQueue queue, uint32_t index, VkSemaphore renderFinished, const Device& device) {
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
    }

    SwapchainKHR           swapchain;
    Semaphore              acquiredImageSemaphore;
    std::vector<VkImage>   images; // non-owning
    std::vector<ImageView> imageViews;
    std::vector<Semaphore> renderFinishedSemaphores;
};

} // namespace simple

} // namespace vko