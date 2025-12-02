// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <concepts>
#include <future>
#include <ranges>
#include <type_traits>
#include <vector>
#include <vko/adapters.hpp>
#include <vko/exceptions.hpp>
#include <vko/handles.hpp>

namespace vko {

class TimelineSubmitCancel : std::exception {
public:
    const char* what() const noexcept override {
        return "SynchronizedCommandBuffer was never submitted to a TimelineQueue";
    }
};

class TimelineSemaphore : public Semaphore {
public:
    template <device_and_commands DeviceAndCommands>
    TimelineSemaphore(const DeviceAndCommands& device, uint64_t initialValue)
        : Semaphore(device,
                    VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                          .pNext = tmpPtr(VkSemaphoreTypeCreateInfo{
                                              .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                              .initialValue  = initialValue}),
                                          .flags = 0}) {}
};

// Helper function to wait for a timeline semaphore and return true iff the
// semaphore is signaled. Returns false on timeout.
template <device_and_commands DeviceAndCommands>
bool waitTimelineSemaphore(DeviceAndCommands& device, VkSemaphore semaphore, uint64_t value,
                           uint64_t timeout) {
    VkSemaphoreWaitInfo waitInfo{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &semaphore,
        .pValues        = &value,
    };
    VkResult r = device.vkWaitSemaphores(device, &waitInfo, timeout);
    if (r == VK_TIMEOUT)
        return false;
    check(r);
    return true;
}

// A timeline semaphore future value, so the event can be shared before the
// value is known. Must outlive the TimelineSemaphore it was created with.
// Alternative name: 'TimelinePoint'?
struct SemaphoreValue {
    using Clock = std::chrono::steady_clock;

    SemaphoreValue() = delete;
    SemaphoreValue(const TimelineSemaphore& semaphore, const std::shared_future<uint64_t>& value)
        : semaphore(semaphore)
        , value(value) {}
    SemaphoreValue(const TimelineSemaphore& semaphore, std::shared_future<uint64_t>&& value)
        : semaphore(semaphore)
        , value(std::move(value)) {}

    // Wait for the semaphore to be signaled. Returns false on cancellation.
    template <vko::device_and_commands DeviceAndCommands>
    bool wait(DeviceAndCommands& device) const {
        assert(semaphore != VK_NULL_HANDLE);
        assert(value.valid());
        try {
            value.wait();
            return waitTimelineSemaphore(device, semaphore, value.get(),
                                         std::numeric_limits<uint64_t>::max());
        } catch (TimelineSubmitCancel&) {
            return false;
        }
    }

    // Wait for the semaphore to be signaled, up to the given duration. Returns
    // false on timeout or cancellation.
    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        uint64_t remainingNs = 0;
        if (duration.count() <= 0) {
            if (!hasValue())
                return false;
        } else {
            const auto begin = Clock::now();
            try {
                if (value.wait_until(begin + duration) == std::future_status::timeout)
                    return false;
            } catch (TimelineSubmitCancel&) {
                return false;
            }
            auto remaining = std::max(begin + duration - Clock::now(), Clock::duration::zero());
            remainingNs = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
        }
        return waitTimelineSemaphore(device, semaphore, value.get(), remainingNs);
    }

    // Wait for the semaphore to be signaled, up to the given time point. Returns
    // false on timeout or cancellation.
    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(DeviceAndCommands&                       device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        // First wait for the submission value to become known, up to the deadline.
        try {
            if (value.wait_until(deadline) == std::future_status::timeout)
                return false;
        } catch (TimelineSubmitCancel&) {
            return false;
        }

        const auto remaining = std::max(deadline - Clock::now(), Clock::duration::zero());
        uint64_t   remainingNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
        return waitTimelineSemaphore(device, semaphore, value.get(), remainingNs);
    }

    // Checks the semaphore value status, but not its signalled status. This is
    // typically used to check if a TimelineQueue submission has been made.
    bool hasValue() const {
        try {
            return value.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
        } catch (TimelineSubmitCancel&) {
            return false;
        }
    }

    // Checks the semaphore signalled status.
    template <vko::device_and_commands DeviceAndCommands>
    bool isSignaled(DeviceAndCommands& device) const {
        return waitFor(device, std::chrono::seconds(0));
    }

    // Returns a VkSemaphoreSubmitInfo for waiting on this semaphore. Will until
    // the semaphore value is known.
    // NOTE: may throw TimelineSubmitCancel, e.g. if the app exits while building
    // a command buffer.
    [[nodiscard]] VkSemaphoreSubmitInfo waitSemaphoreInfo(VkPipelineStageFlags2 stageMask,
                                                          uint32_t              deviceIndex = 0) {
        assert(semaphore != VK_NULL_HANDLE);
        return {.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = semaphore,
                .value       = value.get(),
                .stageMask   = stageMask,
                .deviceIndex = deviceIndex};
    }

    // TODO: there's an argument for making this a
    // std::shared_ptr<TimelineSemaphore> instead. It allows the SemaphoreValue
    // to own its own semaphore. E.g. to mark it as already signalled.
    VkSemaphore                  semaphore = VK_NULL_HANDLE;
    std::shared_future<uint64_t> value;
};

// Generic submit wrapper
template <device_and_commands           DeviceAndCommands,
          std::ranges::contiguous_range Range = std::initializer_list<VkSubmitInfo2>>
    requires std::same_as<std::ranges::range_value_t<Range>, VkSubmitInfo2>
void submit(DeviceAndCommands& device, VkQueue queue, Range&& submitInfos,
            VkFence fence = VK_NULL_HANDLE) {
    check(device.vkQueueSubmit2(queue, uint32_t(std::ranges::size(submitInfos)),
                                std::ranges::data(submitInfos), fence));
}

template <std::ranges::contiguous_range WaitRange = std::initializer_list<VkSemaphoreSubmitInfo>,
          std::ranges::contiguous_range CmdRange = std::initializer_list<VkCommandBufferSubmitInfo>,
          std::ranges::contiguous_range SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<WaitRange>>,
                          VkSemaphoreSubmitInfo> &&
             std::same_as<std::remove_cv_t<std::ranges::range_value_t<CmdRange>>,
                          VkCommandBufferSubmitInfo> &&
             std::same_as<std::remove_cv_t<std::ranges::range_value_t<SignalRange>>,
                          VkSemaphoreSubmitInfo>
VkSubmitInfo2 makeSubmitInfo(WaitRange&& waitInfos, CmdRange&& commandBufferInfos,
                             SignalRange&& signalInfos) {
    return {
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags                    = 0,
        .waitSemaphoreInfoCount   = uint32_t(std::ranges::size(waitInfos)),
        .pWaitSemaphoreInfos      = std::ranges::data(waitInfos),
        .commandBufferInfoCount   = uint32_t(std::ranges::size(commandBufferInfos)),
        .pCommandBufferInfos      = std::ranges::data(commandBufferInfos),
        .signalSemaphoreInfoCount = uint32_t(std::ranges::size(signalInfos)),
        .pSignalSemaphoreInfos    = std::ranges::data(signalInfos),
    };
}

// Submit with a single VkSubmitInfo2
template <std::ranges::contiguous_range WaitRange = std::initializer_list<VkSemaphoreSubmitInfo>,
          std::ranges::contiguous_range CmdRange = std::initializer_list<VkCommandBufferSubmitInfo>,
          std::ranges::contiguous_range SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>,
          device_and_commands           DeviceAndCommands>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<WaitRange>>,
                          VkSemaphoreSubmitInfo> &&
             std::same_as<std::remove_cv_t<std::ranges::range_value_t<CmdRange>>,
                          VkCommandBufferSubmitInfo> &&
             std::same_as<std::remove_cv_t<std::ranges::range_value_t<SignalRange>>,
                          VkSemaphoreSubmitInfo>
void submit(DeviceAndCommands& device, VkQueue queue, WaitRange&& waitInfos,
            CmdRange&& commandBufferInfos, SignalRange&& signalInfos) {
    submit(device, queue,
           {makeSubmitInfo(std::forward<WaitRange>(waitInfos),
                           std::forward<CmdRange>(commandBufferInfos),
                           std::forward<SignalRange>(signalInfos))});
}

// Shortcut for a single command buffer on all devices
template <std::ranges::contiguous_range WaitRange   = std::initializer_list<VkSemaphoreSubmitInfo>,
          std::ranges::contiguous_range SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>,
          device_and_commands           DeviceAndCommands>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<WaitRange>>,
                          VkSemaphoreSubmitInfo> &&
             std::same_as<std::remove_cv_t<std::ranges::range_value_t<SignalRange>>,
                          VkSemaphoreSubmitInfo>
void submit(DeviceAndCommands& device, VkQueue queue, WaitRange&& waitInfos,
            VkCommandBuffer commandBuffer, SignalRange&& signalInfos) {
    VkCommandBufferSubmitInfo commandBufferSubmitInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = commandBuffer,
        .deviceMask    = 0, // all
    };
    submit(device, queue, waitInfos, std::span(&commandBufferSubmitInfo, 1), signalInfos);
}

class TimelineSubmission {
    std::vector<VkSemaphoreSubmitInfo>     waits;
    std::vector<VkCommandBufferSubmitInfo> commands;
    std::vector<VkSemaphoreSubmitInfo>     signals;
    SemaphoreValue                         signalValue;
};

// A promise semaphore value and a shared future. Useful for sharing copies of a
// SemaphoreValue-to-be for a future submission.
struct SubmitPromise {
    SubmitPromise(const TimelineSemaphore& semaphore)
        : m_signalFuture{semaphore, m_promiseValue.get_future()} {}
    SubmitPromise(const SubmitPromise& other)            = delete;
    SubmitPromise& operator=(const SubmitPromise& other) = delete;
    SubmitPromise(SubmitPromise&& other) noexcept
        : m_promiseValue(std::move(other.m_promiseValue))
        , m_signalFuture(std::move(other.m_signalFuture))
        , m_hasValue(other.m_hasValue) {
        other.m_hasValue = false;
    }
    SubmitPromise& operator=(SubmitPromise&& other) noexcept {
        if (m_hasValue)
            m_promiseValue.set_exception(std::make_exception_ptr(TimelineSubmitCancel()));
        m_promiseValue  = std::move(other.m_promiseValue);
        m_signalFuture  = std::move(other.m_signalFuture);
        m_hasValue       = other.m_hasValue;
        other.m_hasValue = false;
        return *this;
    }
    ~SubmitPromise() {
        if (m_hasValue)
            m_promiseValue.set_exception(std::make_exception_ptr(TimelineSubmitCancel()));
    }
    void setValue(uint64_t value) {
        m_promiseValue.set_value(value);
        m_hasValue = false;
    }

    SemaphoreValue futureValue() const { return m_signalFuture; }

private:
    std::promise<uint64_t> m_promiseValue;
    SemaphoreValue         m_signalFuture;
    bool                   m_hasValue = true;
};

// A VkQueue with a timeline semaphore that tracks submission. Not thread safe.
// Use only for tracking submissions to a single device in a device group. It is
// just a ConcurrentTimelineQueue a single internal SubmitPromise.
class TimelineQueue {
public:
    template <device_and_commands DeviceAndCommands>
    TimelineQueue(const DeviceAndCommands& device, uint32_t queueFamilyIndex, uint32_t queueIndex,
                  uint64_t initialValue = 0, uint32_t deviceIndex = 0)
        : m_queue(vko::get(device.vkGetDeviceQueue, device, queueFamilyIndex, queueIndex))
        , m_familyIndex(queueFamilyIndex)
        , m_semaphore(device, initialValue)
        , m_nextValue(initialValue + 1)
        , m_deviceIndex(deviceIndex)
        , m_submitPromise{m_semaphore} {}

    // Returns a SemaphoreValue for the next submission. Be careful something
    // else doesn't submit in the meantime. If this is unknown, use a
    // ConcurrentTimelineQueue.
    SemaphoreValue nextSubmitSemaphore() const { return m_submitPromise.futureValue(); }

    [[nodiscard]] VkSemaphoreSubmitInfo
    signalInfoAndAdvance(VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        VkSemaphoreSubmitInfo timelineSignalInfo = {.sType =
                                                        VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                                    .pNext       = nullptr,
                                                    .semaphore   = m_semaphore,
                                                    .value       = m_nextValue++,
                                                    .stageMask   = timelineSemaphoreStageMask,
                                                    .deviceIndex = m_deviceIndex};
        m_submitPromise.setValue(timelineSignalInfo.value);
        m_submitPromise = SubmitPromise(m_semaphore);
        return timelineSignalInfo;
    }

    // Shortcut for submitting a single command buffer to the timeline queue
    template <device_and_commands DeviceAndCommands,
              std::ranges::contiguous_range WaitRange = std::initializer_list<VkSemaphoreSubmitInfo>>
        requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<WaitRange>>,
                              VkSemaphoreSubmitInfo>
    void submit(DeviceAndCommands& device, WaitRange&& waitInfos,
                VkCommandBuffer commandBuffer, VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        auto                      timelineSignalInfo = signalInfoAndAdvance(timelineSemaphoreStageMask);
        VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext         = nullptr,
            .commandBuffer = commandBuffer,
            .deviceMask    = 1u << m_deviceIndex,
        };
        vko::submit(device, m_queue, std::forward<WaitRange>(waitInfos), {commandBufferSubmitInfo}, {timelineSignalInfo});
    }

    // Like above, but with additional user-provided signal infos.
    template <device_and_commands DeviceAndCommands,
              std::ranges::contiguous_range WaitRange = std::initializer_list<VkSemaphoreSubmitInfo>,
              std::ranges::contiguous_range SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>>
        requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<WaitRange>>,
                              VkSemaphoreSubmitInfo> &&
                 std::same_as<std::remove_cv_t<std::ranges::range_value_t<SignalRange>>,
                              VkSemaphoreSubmitInfo>
    void submit(DeviceAndCommands& device, WaitRange&& waitInfos,
                VkCommandBuffer commandBuffer, VkPipelineStageFlags2 timelineSemaphoreStageMask,
                SignalRange&& extraSignalInfos) {
        auto                      timelineSignalInfo = signalInfoAndAdvance(timelineSemaphoreStageMask);
        VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext         = nullptr,
            .commandBuffer = commandBuffer,
            .deviceMask    = 1u << m_deviceIndex,
        };
        std::vector<VkSemaphoreSubmitInfo> signals;
        signals.reserve(std::ranges::size(extraSignalInfos) + 1);
        signals.insert(signals.end(), std::ranges::begin(extraSignalInfos), std::ranges::end(extraSignalInfos));
        signals.push_back(timelineSignalInfo);
        vko::submit(device, m_queue, std::forward<WaitRange>(waitInfos), {commandBufferSubmitInfo}, signals);
    }

    // Not used internaly, but convenient for e.g. creating a VkCommandPool for
    // submissions to this queue.
    uint32_t familyIndex() const { return m_familyIndex; }

    uint32_t deviceIndex() const { return m_deviceIndex; }

    operator VkQueue() const { return m_queue; }
    const VkQueue* ptr() const { return &m_queue; }

private:
    VkQueue           m_queue;
    uint32_t          m_familyIndex = 0;
    TimelineSemaphore m_semaphore;
    uint64_t          m_nextValue   = 0;
    uint32_t          m_deviceIndex = 0;
    SubmitPromise     m_submitPromise;
};

// A thread safe version of SerialTimelineQueue. This may also be useful in a
// single-threaded case where submission order may differ from command building
// order.
class ConcurrentTimelineQueue {
public:
    static_assert(!std::is_copy_constructible_v<SubmitPromise>);

    template <device_and_commands DeviceAndCommands>
    ConcurrentTimelineQueue(const DeviceAndCommands& device, uint32_t queueFamilyIndex,
                            uint32_t queueIndex, uint64_t initialValue = 0,
                            uint32_t deviceIndex = 0)
        : m_queue(vko::get(device.vkGetDeviceQueue, device, queueFamilyIndex, queueIndex))
        , m_familyIndex(queueFamilyIndex)
        , m_semaphore(device, initialValue)
        , m_nextValue(initialValue + 1)
        , m_deviceIndex(deviceIndex) {}

    // Create a promise with a shared future SemaphoreValue. The SemaphoreValue
    // can be given to users that must wait for a submission to complete. Then
    // std::move the SubmitPromise back to the TimelineQueue when submitting for
    // the value to be set. This separation is necessary if multiple threads
    // intend to submit to the same queue - the semaphore value cannot be
    // predicted in advance.
    SubmitPromise submitPromise() const { return SubmitPromise(m_semaphore); }

    // Consumes a SubmitPromise and produces a VkSemaphoreSubmitInfo for
    // submission. The user must guarantee a queue submit in the callback. This
    // allows the caller to compose their own complicated submissions on the
    // stack. I.e. an alternative could be std::vector to append to a
    // caller-provided submission. The callback allows the mutex to combine
    // m_nextValue++ and the queue submit.
    template <class Fn>
    void withNextTimelineInfo(SubmitPromise&& submitPromise, VkPipelineStageFlags2 stageMask,
                              Fn&& fn) {
        std::lock_guard       lk(m_mutex);
        VkSemaphoreSubmitInfo timelineSignalInfo = {.sType =
                                                        VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                                    .pNext       = nullptr,
                                                    .semaphore   = m_semaphore,
                                                    .value       = m_nextValue++,
                                                    .stageMask   = stageMask,
                                                    .deviceIndex = m_deviceIndex};
        submitPromise.setValue(timelineSignalInfo.value);
        fn(m_queue, timelineSignalInfo);
    }

    // Shortcut for submitting a single command buffer to the timeline queue
    template <device_and_commands DeviceAndCommands>
    void submit(DeviceAndCommands& device, std::span<VkSemaphoreSubmitInfo> waitInfos,
                VkCommandBuffer commandBuffer, SubmitPromise&& submitPromises,
                VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        withNextTimelineInfo(std::move(submitPromises), timelineSemaphoreStageMask,
                             [&](VkQueue queue, VkSemaphoreSubmitInfo timelineSignalInfo) {
                                 VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
                                     .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                     .pNext         = nullptr,
                                     .commandBuffer = commandBuffer,
                                     .deviceMask    = 1u << m_deviceIndex,
                                 };
                                 submit(device, queue, waitInfos, {commandBufferSubmitInfo},
                                        {timelineSignalInfo});
                             });
    }

    // Like above, but with additional user-provided signal infos.
    template <device_and_commands DeviceAndCommands>
    void submit(DeviceAndCommands& device, std::span<VkSemaphoreSubmitInfo> waitInfos,
                VkCommandBuffer commandBuffer, SubmitPromise&& submitPromises,
                VkPipelineStageFlags2            timelineSemaphoreStageMask,
                std::span<VkSemaphoreSubmitInfo> extraSignalInfos) {
        withNextTimelineInfo(
            std::move(submitPromises), timelineSemaphoreStageMask,
            [&](VkQueue queue, VkSemaphoreSubmitInfo timelineSignalInfo) {
                VkCommandBufferSubmitInfo commandBufferSubmitInfo = {
                    .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                    .pNext         = nullptr,
                    .commandBuffer = commandBuffer,
                    .deviceMask    = 1u << m_deviceIndex,
                };

                std::vector<VkSemaphoreSubmitInfo> signals;
                signals.reserve(extraSignalInfos.size() + 1);
                signals.insert(signals.end(), extraSignalInfos.begin(), extraSignalInfos.end());
                signals.push_back(timelineSignalInfo);
                submit(device, queue, waitInfos, {commandBufferSubmitInfo}, signals);
            });
    }

    // Not used internaly, but convenient for e.g. creating a VkCommandPool for
    // submissions to this queue.
    uint32_t familyIndex() const { return m_familyIndex; }

    uint32_t deviceIndex() const { return m_deviceIndex; }

    operator VkQueue() const { return m_queue; }
    const VkQueue* ptr() const { return &m_queue; }

private:
    VkQueue           m_queue;
    uint32_t          m_familyIndex = 0;
    TimelineSemaphore m_semaphore;
    uint64_t          m_nextValue   = 0;
    uint32_t          m_deviceIndex = 0;
    std::mutex        m_mutex;
};

} // namespace vko
