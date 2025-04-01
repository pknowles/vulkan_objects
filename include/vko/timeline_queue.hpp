// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <concepts>
#include <future>
#include <ranges>
#include <vector>
#include <vko/adapters.hpp>
#include <vko/exceptions.hpp>
#include <vko/handles.hpp>

namespace vko {

namespace simple {

class TimelineSemaphore {
public:
    template <device_and_commands DeviceAndCommands>
    TimelineSemaphore(const DeviceAndCommands& device, uint64_t initialValue)
        : m_semaphore(device, VkSemaphoreCreateInfo{
                                  .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                  .pNext = tmpNext(VkSemaphoreTypeCreateInfo{
                                      .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                      .pNext         = nullptr,
                                      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                      .initialValue  = initialValue}),
                                  .flags = 0}) {}
    operator VkSemaphore() const { return m_semaphore; }

private:
    // DANGER: pointer to temporary
    static const VkSemaphoreTypeCreateInfo* tmpNext(const VkSemaphoreTypeCreateInfo& info) {
        return &info;
    }
    Semaphore m_semaphore;
};

struct SemaphoreValue {
    VkSemaphore           semaphore = VK_NULL_HANDLE;
    uint64_t              value     = 0xffffffffffffffffull;
    VkPipelineStageFlags2 stageMask;
    VkSemaphoreSubmitInfo submitInfo(uint32_t deviceIndex) const {
        return {.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = semaphore,
                .value       = value,
                .stageMask   = stageMask,
                .deviceIndex = deviceIndex};
    }
};

class TimelineQueue;

struct SynchronizedCommandBuffer {

    // TODO: can this be avoided?
    friend class TimelineQueue;

    SynchronizedCommandBuffer(CommandBuffer&& cmd)
        : commandBuffer(std::move(cmd)) {}
    operator VkCommandBuffer() const { return commandBuffer; }

    CommandBuffer                                   commandBuffer;
    std::vector<std::shared_future<SemaphoreValue>> waitSemaphores;
    std::optional<VkPipelineStageFlags2> stageMask; // XXX how can I enforce setting this?
    uint32_t                             deviceIndex;

    std::shared_future<SemaphoreValue> getSignalSemaphoreValue() {
        if (!signalSemaphoreValueFuture.valid())
            signalSemaphoreValueFuture = signalSemaphoreValue.get_future();
        return signalSemaphoreValueFuture;
    }

    bool setSignalSemaphoreValue(const SemaphoreValue& semaphoreValue) {
        if (signalSemaphoreValueFuture.valid()) {
            signalSemaphoreValue.set_value(semaphoreValue);
            return true;
        }
        return false;
    }

private:
    std::promise<SemaphoreValue>       signalSemaphoreValue;
    std::shared_future<SemaphoreValue> signalSemaphoreValueFuture;
};

struct Timeline {
    Timeline(const Device& device)
        : semaphore(device, 0) {}
    TimelineSemaphore semaphore;
    uint64_t          value = 0;
};

class TimelineQueue {
public:
    TimelineQueue(uint32_t familyIndex, uint32_t queueIndex, const Device& device)
        : TimelineQueue(vko::get(device.vkGetDeviceQueue, device, familyIndex, queueIndex),
                        device) {}
    TimelineQueue(VkQueue queue, const Device& device)
        : m_queue(queue)
        , m_timeline(device)
        , vkQueueSubmit2(device.vkQueueSubmit2) {}

    template <class Range = std::initializer_list<SynchronizedCommandBuffer*>>
        requires std::convertible_to<std::ranges::range_value_t<Range>, SynchronizedCommandBuffer*>
    void submit(Range&& commandBuffers) {
        std::lock_guard lk(m_mutex);
        ++m_timeline.value;

        constexpr uint32_t deviceIndex = 0;

        // Count first to avoid reallocation, which would break any pointers
        size_t signalInfosCount = 0;
        size_t waitInfosCount   = 0;
        for (auto& commandBuffer : commandBuffers) {
            if (commandBuffer->signalSemaphoreValueFuture.valid())
                ++signalInfosCount;
            waitInfosCount += commandBuffer->waitSemaphores.size();
        }
        std::vector<VkSemaphoreSubmitInfo> signalInfos;
        signalInfos.reserve(signalInfosCount);
        std::vector<VkSemaphoreSubmitInfo> waitInfos;
        waitInfos.reserve(waitInfosCount);
        std::vector<VkCommandBufferSubmitInfo> commandBufferInfos;
        commandBufferInfos.reserve(commandBuffers.size());
        for (SynchronizedCommandBuffer* commandBuffer : commandBuffers) {
            SemaphoreValue signalSemaphoreValue{
                .semaphore = m_timeline.semaphore,
                .value     = m_timeline.value,
                .stageMask = commandBuffer->stageMask.value(), // TODO: may throw
            };
            if (commandBuffer->setSignalSemaphoreValue(signalSemaphoreValue)) {
                signalInfos.push_back(signalSemaphoreValue.submitInfo(deviceIndex));
            }
            for (auto& waitSemaphore : commandBuffer->waitSemaphores) {
                waitInfos.push_back(waitSemaphore.get().submitInfo(deviceIndex));
            }
            commandBufferInfos.push_back(VkCommandBufferSubmitInfo{
                .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext         = nullptr,
                .commandBuffer = *commandBuffer,
                .deviceMask    = 0, // all
            });
        }
        VkSubmitInfo2 submitInfo{
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext                    = nullptr,
            .flags                    = 0,
            .waitSemaphoreInfoCount   = uint32_t(waitInfos.size()),
            .pWaitSemaphoreInfos      = waitInfos.data(),
            .commandBufferInfoCount   = uint32_t(commandBufferInfos.size()),
            .pCommandBufferInfos      = commandBufferInfos.data(),
            .signalSemaphoreInfoCount = uint32_t(signalInfos.size()),
            .pSignalSemaphoreInfos    = signalInfos.data(),
        };
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    }

    operator VkQueue() const { return m_queue; }

private:
    VkQueue            m_queue;
    Timeline           m_timeline;
    std::mutex         m_mutex;
    PFN_vkQueueSubmit2 vkQueueSubmit2;
};

} // namespace simple

} // namespace vko
