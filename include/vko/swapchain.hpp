// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <limits>
#include <queue>
#include <vko/adapters.hpp>
#include <vko/handles.hpp>
#include <vko/pnext_chain.hpp>

namespace vko {

namespace simple {

// Utility: pNext modifier for VK_KHR_present_id
// Usage: chainPNext(nullptr, withPresentId(id), [&](const void* pNext) { swapchain.present(...,
// pNext); })
inline auto withPresentId(uint64_t presentId) {
    return [presentId](auto&& cont, const void* pNext) mutable {
        VkPresentIdKHR info{
            .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .pNext          = pNext,
            .swapchainCount = 1U,
            .pPresentIds    = &presentId,
        };
        return cont(&info);
    };
}

// Pool of binary semaphores with fence-tracked reuse for acquire/present
// cycles. Avoids creating new semaphores by tracking which ones are safe to
// reuse via fences. Typical usage: acquire semaphore before
// vkAcquireNextImageKHR, another for render finish, then endBatch with the
// present fence to mark both reclaimable when fence signals. You will want a
// separate mechanism to limit the number of frames in-flight or this pool could
// grow indefinitely.
// TODO: add a pending batch limit at which acquire() would block?
//
// There's a fundamental flaw in Vulkan's binary semaphore model for swapchains:
// they can't be reused until the GPU has finished waiting on them. This is
// frustrating because we can set up pipelines where we will only reuse them
// after the GPU has finished waiting on them, but the Vulkan spec says we can't
// even queue up API calls to reuse them. This pool is a workaround.
class BinarySemaphorePool {
public:
    template <device_and_commands DeviceAndCommands>
    VkSemaphore acquire(const DeviceAndCommands& device) {
        currentBatch.push_back(reuseOrMakeSemaphore(device));
        return currentBatch.back();
    }

    void endBatch(Fence&& fence) {
        // Always store the fence, even if no semaphores were acquired this batch
        // The fence tracks when the present completes, independent of semaphore lifecycle
        inFlight.push({std::move(currentBatch), std::move(fence)});
        currentBatch = std::move(emptyVector); // Reuse capacity
    }

    size_t size() const { return totalSemaphores; }

private:
    template <device_and_commands DeviceAndCommands>
    Semaphore reuseOrMakeSemaphore(const DeviceAndCommands& device) {
        if (available.empty() && !inFlight.empty()) {
            auto& [batch, fence] = inFlight.front();
            VkResult result      = device.vkGetFenceStatus(device, fence);
            if (result == VK_SUCCESS) {
                available = std::move(batch); // Recycle vector capacity
                inFlight.pop();
            } else if (result == VK_NOT_READY) {
                // Not ready yet, nothing after it will be either
            } else {
                check(result);
            }
        }

        if (available.empty()) {
            ++totalSemaphores;
            return Semaphore(device,
                             VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                                   .pNext = nullptr,
                                                   .flags = 0});
        } else {
            Semaphore recycledSemaphore = std::move(available.back());
            available.pop_back();

            // Allow recycling the vector heap allocation
            if (available.empty()) {
                emptyVector = std::move(available);
            }
            return recycledSemaphore;
        }
    }

    using Semaphores = std::vector<Semaphore>;
    std::queue<std::pair<Semaphores, Fence>> inFlight; // Still in use
    Semaphores currentBatch;        // Semaphores that do not have a completion fence yet
    Semaphores available;           // Ready for reuse
    Semaphores emptyVector;         // Allows recycling the vector heap allocation
    size_t     totalSemaphores = 0; // Tracking the total number of semaphores created
};

// Swapchain images and views - requires caller to provide semaphores for acquire/present
struct SwapchainImages {
    // Simple/default create info. Use the other constructor to override
    template <device_and_commands DeviceAndCommands>
    SwapchainImages(const DeviceAndCommands& device, VkSurfaceKHR surface,
                    VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
                    VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : SwapchainImages(
              device, VkSwapchainCreateInfoKHR{
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

    template <device_and_commands DeviceAndCommands>
    SwapchainImages(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : swapchain(device, createInfo)
        , images(vko::toVector(device.vkGetSwapchainImagesKHR, device, swapchain)) {
        imageViews.reserve(images.size());
        presented.resize(images.size(), false);
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
        }
    }

    operator VkSwapchainKHR() const { return swapchain; }

    // Acquire next swapchain image using provided semaphore
    // User is responsible for managing semaphore lifecycle and reuse
    template <device_and_commands DeviceAndCommands>
    uint32_t acquire(const DeviceAndCommands& device, VkSemaphore acquireSemaphore,
                     uint64_t timeout) {
        auto imageIndex = std::numeric_limits<uint32_t>::max();
        check(device.vkAcquireNextImageKHR(device, swapchain, timeout, acquireSemaphore,
                                           VK_NULL_HANDLE, &imageIndex));
        return imageIndex;
    }

    // Low-level present with extension chain (e.g. VK_KHR_present_id, VK_GOOGLE_display_timing)
    // Use withPresentId() and chainPNext() for composable pNext chains
    template <device_and_commands DeviceAndCommands>
    void present(const DeviceAndCommands& device, VkQueue queue, uint32_t index,
                 VkSemaphore renderFinished, const void* presentInfoPNext) {
        VkPresentInfoKHR presentInfo{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = presentInfoPNext,
            .waitSemaphoreCount = 1U,
            .pWaitSemaphores    = &renderFinished,
            .swapchainCount     = 1U,
            .pSwapchains        = swapchain.ptr(),
            .pImageIndices      = &index,
            .pResults           = nullptr,
        };

        check(device.vkQueuePresentKHR(queue, &presentInfo));
        presented[index] = true;
    }

    SwapchainKHR           swapchain;
    std::vector<VkImage>   images; // non-owning
    std::vector<ImageView> imageViews;
    std::vector<bool>      presented; // first-use flag, implying VK_IMAGE_LAYOUT_UNDEFINED
};

// Managed swapchain with automatic semaphore pooling
// Provides the same interface as the old Swapchain but handles semaphore lifecycle automatically
struct Swapchain : public SwapchainImages {
    template <device_and_commands DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, VkSurfaceKHR surface,
              VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
              VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : SwapchainImages(device, surface, surfaceFormat, extent, queueFamilyIndex, presentMode,
                          oldSwapchain) {}

    template <device_and_commands DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : SwapchainImages(device, createInfo) {}

    operator VkSwapchainKHR() const { return SwapchainImages::swapchain; }

    // Acquire with automatic semaphore management
    // Returns imageIndex and acquire semaphore (user must wait on this semaphore in their submit)
    template <device_and_commands DeviceAndCommands>
    std::pair<uint32_t, VkSemaphore> acquire(const DeviceAndCommands& device, uint64_t timeout) {
        VkSemaphore acquireSem = semaphorePool.acquire(device);
        uint32_t    imageIndex = SwapchainImages::acquire(device, acquireSem, timeout);
        return {imageIndex, acquireSem};
    }

    // Get a render-finished semaphore for the user to signal when rendering completes
    template <device_and_commands DeviceAndCommands>
    VkSemaphore getRenderSemaphore(const DeviceAndCommands& device) {
        return semaphorePool.acquire(device);
    }

    // Present with automatic fence tracking for semaphore reuse
    // The fence signals when both acquire and render semaphores can be reused
    // User pNext is chained with VkSwapchainPresentFenceInfoEXT
    // Use withPresentId() with chainPNext() to add VK_KHR_present_id
    template <device_and_commands DeviceAndCommands>
    void present(const DeviceAndCommands& device, VkQueue queue, uint32_t index,
                 VkSemaphore renderFinished, const void* userPNext) {
        Fence presentFence(device, VkFenceCreateInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                                     .pNext = nullptr,
                                                     .flags = 0});

        VkSwapchainPresentFenceInfoEXT fenceInfo{
            .sType          = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
            .pNext          = userPNext,
            .swapchainCount = 1U,
            .pFences        = presentFence.ptr(),
        };

        try {
            SwapchainImages::present(device, queue, index, renderFinished, &fenceInfo);
        } catch (const ResultException<VK_ERROR_OUT_OF_DATE_KHR>& e) {
            // Vulkan, or maybe just the validation layer, has still consumed
            // the fence even when out of date is returned. This simple solution
            // of waiting on it avoids the error.
            device.vkWaitForFences(device, 1U, presentFence.ptr(), VK_TRUE,
                                   std::numeric_limits<uint64_t>::max());
            semaphorePool.endBatch(std::move(presentFence));
            throw e;
        }

        semaphorePool.endBatch(std::move(presentFence));
    }

    BinarySemaphorePool semaphorePool;
};

template <device_and_commands DeviceAndCommands>
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