// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <limits>
#include <queue>
#include <vko/adapters.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>
#include <vko/pnext_chain.hpp>

namespace vko {

// Utility: pNext modifier for VK_KHR_present_id
// Usage: chainPNext(nullptr, withPresentId(id), [&](const void* pNext) {
// swapchain.present(..., pNext); })
inline auto withPresentId(uint64_t presentId) {
    return [presentId](auto&& cont, const void* pNext) mutable {
        VkPresentIdKHR presentIdInfo{
            .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .pNext          = pNext,
            .swapchainCount = 1U,
            .pPresentIds    = &presentId,
        };
        return cont(&presentIdInfo);
    };
}

// Utility: present fence pNext modifier, requires VK_KHR_swapchain_maintenance1
// Usage: chainPNext(nullptr, withPresentFence(id), [&](const void* pNext) {
// swapchain.present(..., pNext); })
inline auto withPresentFence(VkFence fence) {
    return [fence](auto&& cont, const void* pNext) mutable {
        VkSwapchainPresentFenceInfoEXT presentFenceInfo{
            .sType          = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
            .pNext          = pNext,
            .swapchainCount = 1U,
            .pFences        = &fence,
        };
        return cont(&presentFenceInfo);
    };
}

// An owning Fence that waits on destruction. The cost is duplicating the
// m_vkWaitForFences function pointer for each fence and an unconditional wait
// on destruction even if known to be ready.
class WaitingFence {
public:
    template <device_and_commands DeviceAndCommands>
    WaitingFence(const DeviceAndCommands& device, const VkFenceCreateInfo& createInfo)
        : m_fence(device, createInfo)
        , m_vkWaitForFences(device.vkWaitForFences) {}
    template <device_and_commands DeviceAndCommands>
    explicit WaitingFence(const DeviceAndCommands& device, Fence&& fence)
        : m_fence(std::move(fence))
        , m_vkWaitForFences(device.vkWaitForFences) {}
    ~WaitingFence() {
        if (m_fence.engaged())
            m_vkWaitForFences(m_fence.parent(), 1u, m_fence.ptr(), VK_TRUE,
                              std::numeric_limits<uint64_t>::max());
    }
    WaitingFence(WaitingFence&& other) noexcept = default;
    WaitingFence& operator=(WaitingFence&& other) noexcept {
        if (m_fence.engaged())
            m_vkWaitForFences(m_fence.parent(), 1u, m_fence.ptr(), VK_TRUE,
                              std::numeric_limits<uint64_t>::max());
        m_fence           = std::move(other.m_fence);
        m_vkWaitForFences = other.m_vkWaitForFences;
        return *this;
    }

    operator VkFence() const& { return m_fence; }
    operator VkFence() && = delete;
    bool    engaged() const { return m_fence.engaged(); }
    VkFence object() const& { return m_fence.object(); } // useful to be explicit for type deduction
    const VkFence* ptr() const& { return m_fence.ptr(); }
    VkDevice       parent() const { return m_fence.parent(); }

private:
    Fence               m_fence;
    PFN_vkWaitForFences m_vkWaitForFences = nullptr;
};

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
//

// Due to the use of WaitingFence, this object will block on destruction until
// all semaphores are free.
class BinarySemaphorePool {
public:
    template <device_and_commands DeviceAndCommands>
    VkSemaphore acquire(const DeviceAndCommands& device) {
        currentBatch.push_back(reuseOrMakeSemaphore(device));
        return currentBatch.back();
    }

    void endBatch(WaitingFence&& fence) {
        // Always store the fence, even if no semaphores were acquired this batch
        // The fence tracks when the present completes, independent of semaphore lifecycle
        inFlight.emplace(std::move(currentBatch), std::move(fence));
        currentBatch = std::move(emptyVector); // Reuse capacity
    }

    size_t size() const { return totalSemaphores; }

    // Wait for all semaphores to become ready. Returns true if all batches are
    // ready.
    template <device_and_commands DeviceAndCommands>
    bool tryWait(const DeviceAndCommands& device) {
        while (!inFlight.empty()) {
            if (!reuseOne(device))
                return false;
        }
        return true;
    }

private:
    template <device_and_commands DeviceAndCommands>
    bool reuseOne(const DeviceAndCommands& device) {
        assert(!inFlight.empty());
        auto& [batch, fence] = inFlight.front();
        VkResult result      = device.vkGetFenceStatus(device, fence);
        if (result == VK_SUCCESS) {
            if (available.empty()) {
                available = std::move(batch); // Recycle vector capacity
            } else {
                std::ranges::move(batch, std::back_inserter(available));
            }
            inFlight.pop();
            return true;
        } else if (result == VK_NOT_READY) {
            // Not ready yet, nothing after it will be either
        } else {
            check(result);
        }
        return false;
    }

    template <device_and_commands DeviceAndCommands>
    Semaphore reuseOrMakeSemaphore(const DeviceAndCommands& device) {
        if (available.empty() && !inFlight.empty()) {
            reuseOne(device);
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
    std::queue<std::pair<Semaphores, WaitingFence>> inFlight; // Still in use
    Semaphores currentBatch;        // Semaphores that do not have a completion fence yet
    Semaphores available;           // Ready for reuse
    Semaphores emptyVector;         // Allows recycling the vector heap allocation
    size_t     totalSemaphores = 0; // Tracking the total number of semaphores created
};

template <typename R, typename T>
concept random_access_range_of =
    std::ranges::random_access_range<R> && std::same_as<std::ranges::range_value_t<R>, T>;

template <typename T>
concept swapchain = requires(const T& swapchain) {
    { static_cast<VkSwapchainKHR>(swapchain) } -> std::same_as<VkSwapchainKHR>;
    requires random_access_range_of<decltype(swapchain.images()), VkImage>;
    requires random_access_range_of<decltype(swapchain.imageViews()), ImageView>;
    requires random_access_range_of<decltype(swapchain.presented()), bool>;
    { swapchain.engaged() } -> std::same_as<bool>;
};

// Swapchain images and views - requires caller to provide semaphores for acquire/present
class SwapchainImages {
public:
    // Simple/default create info. Use the other constructor to override.
    // minImageCount should come from VkSurfaceCapabilitiesKHR. Beware its
    // current extent may be a special 0xFFFFFFFF value to indicate the user
    // must choose the extent.
    template <device_and_commands DeviceAndCommands>
    SwapchainImages(const DeviceAndCommands& device, VkSurfaceKHR surface, uint32_t minImageCount,
                    VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
                    VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : SwapchainImages(
              device, VkSwapchainCreateInfoKHR{
                          .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                          .pNext            = nullptr,
                          .flags            = 0,
                          .surface          = surface,
                          .minImageCount    = minImageCount,
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

    // User-provided VkSwapchainCreateInfoKHR
    template <device_and_commands DeviceAndCommands>
    SwapchainImages(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : m_swapchain(device, createInfo)
        , m_images(toVector(device.vkGetSwapchainImagesKHR, device, m_swapchain)) {
        m_imageViews.reserve(m_images.size());
        m_presented.resize(m_images.size(), false);
        for (auto& image : m_images) {
            m_imageViews.emplace_back(device, VkImageViewCreateInfo{
                                                  .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                  .pNext = nullptr,
                                                  .flags = 0,
                                                  .image = image,
                                                  .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                                                  .format     = createInfo.imageFormat,
                                                  .components = VkComponentMapping{},
                                                  .subresourceRange =
                                                      VkImageSubresourceRange{
                                                          .aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT,
                                                          .baseMipLevel = 0,
                                                          .levelCount   = 1,
                                                          .baseArrayLayer = 0,
                                                          .layerCount     = 1,
                                                      },
                                              });
        }
    }

    // Acquire next swapchain image using provided semaphore
    // User is responsible for managing semaphore lifecycle and reuse
    template <device_and_commands DeviceAndCommands>
    uint32_t acquire(const DeviceAndCommands& device, VkSemaphore acquireSemaphore,
                     uint64_t timeout) {
        auto imageIndex = std::numeric_limits<uint32_t>::max();
        check(device.vkAcquireNextImageKHR(device, m_swapchain, timeout, acquireSemaphore,
                                           VK_NULL_HANDLE, &imageIndex));
        return imageIndex;
    }

    // Present with a single render-finished semaphore. User is responsible for
    // managing semaphore lifecycle and reuse
    template <device_and_commands DeviceAndCommands>
    void present(const DeviceAndCommands& device, VkQueue queue, uint32_t index,
                 VkSemaphore renderFinished, const void* presentInfoPNext) {
        VkPresentInfoKHR presentInfo{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = presentInfoPNext,
            .waitSemaphoreCount = 1U,
            .pWaitSemaphores    = &renderFinished,
            .swapchainCount     = 1U,
            .pSwapchains        = m_swapchain.ptr(),
            .pImageIndices      = &index,
            .pResults           = nullptr,
        };

        check(device.vkQueuePresentKHR(queue, &presentInfo));
        m_presented[index] = true;
    }

    operator VkSwapchainKHR() const { return m_swapchain; }
    const std::vector<VkImage>&   images() const { return m_images; }
    const std::vector<ImageView>& imageViews() const { return m_imageViews; }
    const std::vector<bool>&      presented() const { return m_presented; }
    bool                          engaged() const { return m_swapchain.engaged(); }

private:
    SwapchainKHR           m_swapchain;
    std::vector<VkImage>   m_images; // non-owning
    std::vector<ImageView> m_imageViews;
    std::vector<bool>      m_presented; // first-use flag, implying VK_IMAGE_LAYOUT_UNDEFINED
};
static_assert(swapchain<SwapchainImages>);

// Combines SwapchainImages with a BinarySemaphorePool to provide semaphores for
// acquire and present.
class Swapchain {
public:
    template <device_and_commands DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, VkSurfaceKHR surface, uint32_t minImageCount,
              VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
              VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : m_swapchain(device, surface, minImageCount, surfaceFormat, extent, queueFamilyIndex,
                      presentMode, oldSwapchain) {}

    template <device_and_commands DeviceAndCommands>
    Swapchain(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : m_swapchain(device, createInfo) {}

    // Acquire with automatic semaphore management
    // Returns imageIndex and acquire semaphore (user must wait on this semaphore in their submit)
    template <device_and_commands DeviceAndCommands>
    std::pair<uint32_t, VkSemaphore> acquire(const DeviceAndCommands& device, uint64_t timeout) {
        VkSemaphore acquireSem = m_semaphorePool.acquire(device);
        uint32_t    imageIndex = m_swapchain.acquire(device, acquireSem, timeout);
        return {imageIndex, acquireSem};
    }

    // Get a render-finished semaphore for the user to signal when rendering completes
    template <device_and_commands DeviceAndCommands>
    VkSemaphore getRenderSemaphore(const DeviceAndCommands& device) {
        return m_semaphorePool.acquire(device);
    }

    // Present with automatic fence tracking for semaphore reuse. Requires
    // VK_KHR_swapchain_maintenance1. The fence signals when both acquire and
    // render semaphores can be reused User pNext is chained with
    // VkSwapchainPresentFenceInfoEXT Use withPresentId() with chainPNext() to
    // add VK_KHR_present_id
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
            m_swapchain.present(device, queue, index, renderFinished, &fenceInfo);
        } catch (const ResultException<VK_ERROR_OUT_OF_DATE_KHR>& e) {
            // Vulkan, or maybe just the validation layer, has still consumed
            // the fence even when out of date is returned. This simple solution
            // of waiting on it avoids the error.
            device.vkWaitForFences(device, 1U, presentFence.ptr(), VK_TRUE,
                                   std::numeric_limits<uint64_t>::max());
            m_semaphorePool.endBatch(WaitingFence(device, std::move(presentFence)));
            throw e;
        }

        m_semaphorePool.endBatch(WaitingFence(device, std::move(presentFence)));
    }

    operator VkSwapchainKHR() const { return m_swapchain; }
    const std::vector<VkImage>&   images() const { return m_swapchain.images(); }
    const std::vector<ImageView>& imageViews() const { return m_swapchain.imageViews(); }
    const std::vector<bool>&      presented() const { return m_swapchain.presented(); }
    bool                          engaged() const { return m_swapchain.engaged(); }
    BinarySemaphorePool&          semaphorePool() { return m_semaphorePool; }
    const BinarySemaphorePool&    semaphorePool() const { return m_semaphorePool; }

private:
    SwapchainImages     m_swapchain;
    BinarySemaphorePool m_semaphorePool;
};
static_assert(swapchain<Swapchain>);

// A Swapchain that provides a basic way to limit the number of in-flight frames
// to reduce latency. Requires VK_KHR_present_id and VK_KHR_present_wait to be
// enabled. The user should call waitForPresentIds(1) before acquire (and before
// polling for input) each frame. This is included in a swapchain as the IDs are
// owned by the swapchain. I.e. you can't wait on an old ID with a new
// swapchain.
template <swapchain SwapchainType = Swapchain>
class LimitedSwapchain {
public:
    template <device_and_commands DeviceAndCommands>
    LimitedSwapchain(const DeviceAndCommands& device, VkSurfaceKHR surface, uint32_t minImageCount,
                     VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent, uint32_t queueFamilyIndex,
                     VkPresentModeKHR presentMode, VkSwapchainKHR oldSwapchain)
        : m_swapchain(device, surface, minImageCount, surfaceFormat, extent, queueFamilyIndex,
                      presentMode, oldSwapchain) {}

    template <device_and_commands DeviceAndCommands>
    LimitedSwapchain(const DeviceAndCommands& device, const VkSwapchainCreateInfoKHR& createInfo)
        : m_swapchain(device, createInfo) {}

    template <device_and_commands DeviceAndCommands>
    std::pair<uint32_t, VkSemaphore> acquire(const DeviceAndCommands& device, uint64_t timeout) {
        return m_swapchain.acquire(device, timeout);
    }
    template <device_and_commands DeviceAndCommands>
    VkSemaphore getRenderSemaphore(const DeviceAndCommands& device) {
        return m_swapchain.getRenderSemaphore(device);
    }
    template <device_and_commands DeviceAndCommands>
    void present(const DeviceAndCommands& device, VkQueue queue, uint32_t index,
                 VkSemaphore renderFinished, const void* userPNext) {
        VkPresentIdKHR presentId{
            .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .pNext          = userPNext,
            .swapchainCount = 1U,
            .pPresentIds    = &m_nextPresentId,
        };
        m_swapchain.present(device, queue, index, renderFinished, &presentId);
        m_pendingPresentIds.push(m_nextPresentId);
        m_nextPresentId++;
    }

    // Limits pending frames to at most maxPending
    template <device_and_commands DeviceAndCommands>
    void waitForPresentIds(const DeviceAndCommands& device, size_t maxPending = 1) {
        while (m_pendingPresentIds.size() > maxPending) {
            check(device.vkWaitForPresentKHR(device, m_swapchain, m_pendingPresentIds.front(),
                                             std::numeric_limits<uint64_t>::max()));
            m_pendingPresentIds.pop();
        }
    }

    operator VkSwapchainKHR() const { return m_swapchain; }
    const std::vector<VkImage>&   images() const { return m_swapchain.images(); }
    const std::vector<ImageView>& imageViews() const { return m_swapchain.imageViews(); }
    const std::vector<bool>&      presented() const { return m_swapchain.presented(); }
    bool                          engaged() const { return m_swapchain.engaged(); }
    BinarySemaphorePool&          semaphorePool() { return m_swapchain.semaphorePool(); }
    const BinarySemaphorePool&    semaphorePool() const { return m_swapchain.semaphorePool(); }

private:
    SwapchainType        m_swapchain;
    std::queue<uint64_t> m_pendingPresentIds;
    uint64_t             m_nextPresentId    = 1;
    size_t               m_maxPendingFrames = 1;
};
static_assert(swapchain<LimitedSwapchain<>>);

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

// Swapchain container that handles extent fallback and facilitates recreation
// Assumes the backend implicitly waits for presents on destruction due to the
// BinarySemaphorePool member and its use of WaitingFence.
template <swapchain SwapchainType = LimitedSwapchain<>>
class RecreatingSwapchain {
public:
    struct Config {
        VkSurfaceKHR       surface          = VK_NULL_HANDLE;
        VkSurfaceFormatKHR surfaceFormat    = {};
        uint32_t           queueFamilyIndex = 0;
        VkPresentModeKHR   presentMode      = {};
        VkExtent2D         fallbackExtent   = {}; // for Wayland
        uint32_t minImageCount = 0; // Optional: constrain VkSurfaceCapabilitiesKHR::minImageCount
    };

    template <instance_and_commands InstanceAndCommands, device_and_commands DeviceAndCommands>
    RecreatingSwapchain(const InstanceAndCommands& instance, const DeviceAndCommands& device,
                        VkPhysicalDevice physicalDevice, const Config& config)
        : RecreatingSwapchain(device,
                              get(instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
                                  physicalDevice, config.surface),
                              config) {}

    template <device_and_commands DeviceAndCommands>
    RecreatingSwapchain(const DeviceAndCommands&        device,
                        const VkSurfaceCapabilitiesKHR& capabilities, const Config& config)
        : m_vkQueueWaitIdle(device.vkQueueWaitIdle)
        , m_extent(chooseExtent(capabilities, config))
        , m_swapchain(createSwapchain(device, m_extent, capabilities, config, VK_NULL_HANDLE)) {}

    template <instance_and_commands InstanceAndCommands, device_and_commands DeviceAndCommands>
    void recreate(const InstanceAndCommands& instance, const DeviceAndCommands& device,
                  VkPhysicalDevice physicalDevice, const Config& config) {
        recreate(
            device,
            get(instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR, physicalDevice, config.surface),
            config);
    }

    template <device_and_commands DeviceAndCommands>
    void recreate(const DeviceAndCommands& device, const VkSurfaceCapabilitiesKHR& capabilities,
                  const Config& config) {
        // Destroy the old swapchain immediately. Note this requires a stall,
        // waiting for all in-flight presents to finish. I've tried keeping the
        // swapchain alive to avoid stalling, and just hit
        // VK_ERROR_SURFACE_LOST_KHR. Yes, this does break exception safety - if
        // the recreate fails, we are left with a moved-from m_swapchain.
        SwapchainType(std::move(m_swapchain));

        // Create the new swapchain
        VkExtent2D newExtent = chooseExtent(capabilities, config);
        m_swapchain = createSwapchain(device, newExtent, capabilities, config, m_swapchain);
        m_extent    = newExtent;
    }

    // Provide the chosen extent, after fallback handling
    VkExtent2D extent() const { return m_extent; }

    // Direct swapchain access. Simpler than implementing and plumbing through
    // its interface
    SwapchainType&       swapchain() { return m_swapchain; }
    const SwapchainType& swapchain() const { return m_swapchain; }

private:
    //void freeOldSwapchains()
    //{
    //    // Call the old swapchain poll-to-destroy functions
    //    while(!m_oldSwapchains.empty() && m_oldSwapchains.front()())
    //    {
    //        m_oldSwapchains.pop();
    //    }
    //}

    static VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                   const Config&                   config) {
        // The surface may pass uint max as a special value "indicating that the
        // surface size will be determined by the extent of a swapchain
        // targeting the surface". I.e. we get to choose. The currentExtent may
        // also be zero if the window is invalid/minimized.
        if (capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max() ||
            capabilities.currentExtent.height == std::numeric_limits<uint32_t>::max() ||
            capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0) {
            return config.fallbackExtent;
        } else {
            return capabilities.currentExtent;
        }
    }

    template <device_and_commands DeviceAndCommands>
    static SwapchainType createSwapchain(const DeviceAndCommands& device, const VkExtent2D& extent,
                                         const VkSurfaceCapabilitiesKHR& capabilities,
                                         const Config& config, VkSwapchainKHR oldSwapchain) {

        // Pass old swapchain handle to allow driver to reuse resources during resize
        return SwapchainType{device,
                             config.surface,
                             std::clamp(config.minImageCount, capabilities.minImageCount,
                                        capabilities.maxImageCount),
                             config.surfaceFormat,
                             extent,
                             config.queueFamilyIndex,
                             config.presentMode,
                             oldSwapchain};
    }

    PFN_vkQueueWaitIdle m_vkQueueWaitIdle;
    VkExtent2D          m_extent = {};
    SwapchainType       m_swapchain;
    //std::queue<std::function<bool()>> m_oldSwapchains;
};

// Standalone helper: acquire -> render -> present with automatic OUT_OF_DATE
// handling.
//
// Acquires the next swapchain image, calls the user's render function to record
// commands, submits with synchronization, and presents the result.
//
// If VK_ERROR_OUT_OF_DATE_KHR occurs (at acquire or present), the nullopt is
// returned and the caller should recreate the swapchain and retry.
//
// RenderFn signature: void(CyclingCommandBufferType& cmd, VkImage image,
// VkImageView imageView, VkImageLayout initialLayout)
//
// Returns: Timeline semaphore value on success, nullopt if OUT_OF_DATE
template <swapchain SwapchainType, class CyclingCommandBufferType, typename RenderFn>
std::optional<SemaphoreValue> tryPresentFrame(SwapchainType&            swapchain,
                                              CyclingCommandBufferType& cmd, RenderFn&& renderFn) {
    auto& device = cmd.device();
    auto& queue  = cmd.queue();
    try {
        // Acquire next swapchain image
        auto [imageIndex, acquireSem] = swapchain.acquire(device, UINT64_MAX);

        // Don't start rendering to the image until this semaphore is signalled
        cmd.waitOnNextSubmit(VkSemaphoreSubmitInfo{
            .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext       = nullptr,
            .semaphore   = acquireSem,
            .value       = 0,
            .stageMask   = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .deviceIndex = 0,
        });

        // Determine initial layout: PRESENT_SRC if previously presented, else UNDEFINED
        VkImageLayout initialLayout = swapchain.presented()[imageIndex]
                                          ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                          : VK_IMAGE_LAYOUT_UNDEFINED;

        // Call user's render function to record commands
        renderFn(cmd, swapchain.images()[imageIndex], swapchain.imageViews()[imageIndex],
                 initialLayout);

        // Semaphore for presenting after rendering finished
        VkSemaphore renderFinishedSemaphore = swapchain.getRenderSemaphore(device);
        std::array  extraSignals            = {VkSemaphoreSubmitInfo{
                        .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                        .pNext       = nullptr,
                        .semaphore   = renderFinishedSemaphore,
                        .value       = 0,
                        .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .deviceIndex = 0,
        }};

        // Submit with wait/signal semaphores
        SemaphoreValue renderFinished =
            cmd.submit({}, extraSignals, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

        // Present the rendered image
        swapchain.present(device, queue, imageIndex, renderFinishedSemaphore, nullptr);

        return renderFinished;
    } catch (const ResultException<VK_ERROR_OUT_OF_DATE_KHR>&) {
        return std::nullopt;
    }
}

} // namespace vko
