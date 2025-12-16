// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <cassert>
#include <chrono>
#include <concepts>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>
#include <vko/adapters.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/exceptions.hpp>
#include <vko/handles.hpp>
#include <vulkan/vulkan_core.h>

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

template <device_and_commands DeviceAndCommands>
void waitTimelineSemaphore(DeviceAndCommands& device, VkSemaphore semaphore, uint64_t value) {
    VkSemaphoreWaitInfo waitInfo{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &semaphore,
        .pValues        = &value,
    };
    check(device.vkWaitSemaphores(device, &waitInfo, std::numeric_limits<uint64_t>::max()));
}

// A timeline semaphore future value, so the event can be shared before the
// value is known. Must outlive the TimelineSemaphore it was created with.
// Alternative name: 'TimelinePoint'?
class SemaphoreValue {
public:
    using Clock = std::chrono::steady_clock;

    SemaphoreValue() = delete;
    SemaphoreValue(const TimelineSemaphore& semaphore, const std::shared_future<uint64_t>& value)
        : semaphore(semaphore)
        , value(value) {}

    SemaphoreValue(const TimelineSemaphore& semaphore, std::shared_future<uint64_t>&& value)
        : semaphore(semaphore)
        , value(std::move(value)) {}

    // Since we cache the signalled state we can provide a special constructor
    // for an already-signalled semaphore.
    struct already_signalled_t {};
    SemaphoreValue(already_signalled_t)
        : signalledCache(true) {}
    static SemaphoreValue makeSignalled() { return SemaphoreValue(already_signalled_t{}); }

    // Wait for the semaphore to be signaled. Returns false on cancellation.
    template <vko::device_and_commands DeviceAndCommands>
    void wait(DeviceAndCommands& device) const {
        if(signalledCache)
            return;
        assert(semaphore != VK_NULL_HANDLE);
        assert(value.valid());
        waitTimelineSemaphore(device, semaphore, value.get()); // May throw TimelineSubmitCancel
        signalledCache = true;
    }

    // Wait for the semaphore to be signaled. Returns false on cancellation. Use
    // isSignaled() to poll instead of waiting.
    template <vko::device_and_commands DeviceAndCommands>
    bool tryWait(DeviceAndCommands& device) const {
        if(signalledCache)
            return true;
        try {
            wait(device);
        } catch (TimelineSubmitCancel&) {
            return false;
        }
        signalledCache = true;
        return true;
    }

    // Wait for the semaphore to be signaled, up to the given duration. Returns
    // false on timeout or cancellation.
    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        if(signalledCache)
            return true;
        assert(semaphore != VK_NULL_HANDLE);
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
        return signalledCache = waitTimelineSemaphore(device, semaphore, value.get(), remainingNs);
    }

    // Wait for the semaphore to be signaled, up to the given time point. Returns
    // false on timeout or cancellation.
    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(DeviceAndCommands&                       device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        if(signalledCache)
            return true;
        assert(semaphore != VK_NULL_HANDLE);
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
        return signalledCache = waitTimelineSemaphore(device, semaphore, value.get(), remainingNs);
    }

    // Checks the semaphore value status, but not its signalled status. This is
    // typically used to check if a TimelineQueue submission has been made.
    bool hasValue() const {
        if (signalledCache)
            return true;
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

private:
    // TODO: there's an argument for making this a
    // std::shared_ptr<TimelineSemaphore> instead. It allows the SemaphoreValue
    // to own its own semaphore. E.g. to mark it as already signalled.
    VkSemaphore                  semaphore = VK_NULL_HANDLE;
    std::shared_future<uint64_t> value;
    mutable bool                 signalledCache = false; // caches semaphore status
};

template <class T>
class TimelineFuture {
public:
    // Not like a std::future - we often have the object handle available
    // immediately, we just shouldn't access it until the future is ready.
    template <class U, class SV>
    TimelineFuture(SV&& semaphore, U&& value)
        : m_semaphore(std::forward<SV>(semaphore))
        , m_value(std::forward<U>(value)) {}

    template <vko::device_and_commands DeviceAndCommands>
    T& get(const DeviceAndCommands& device) {
        m_semaphore.wait(device);
        return m_value;
    }

    template <vko::device_and_commands DeviceAndCommands>
    const T& get(const DeviceAndCommands& device) const {
        m_semaphore.wait(device);
        return m_value;
    }

    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(const DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        return m_semaphore.waitFor(device, duration);
    }

    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(const DeviceAndCommands& device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        return m_semaphore.waitUntil(device, deadline);
    }

    template <vko::device_and_commands DeviceAndCommands>
    bool ready(const DeviceAndCommands& device) const { return m_semaphore.isSignaled(device); }

    const SemaphoreValue& semaphore() const { return m_semaphore; }

private:
    SemaphoreValue m_semaphore;
    T              m_value;
};

// For when we actually don't have a value until the semaphore is signaled. More
// like a real std::future, except the value is cached.
template <class T>
class LazyTimelineFuture {
public:
    template <class SV, class Fn>
    LazyTimelineFuture(SV&& semaphore, Fn&& producer)
        : m_semaphore(std::forward<SV>(semaphore))
        , m_producer(std::forward<Fn>(producer)) {}

    template <vko::device_and_commands DeviceAndCommands>
    T& get(const DeviceAndCommands& device) {
        if (!m_value) {
            m_semaphore.wait(device);
            m_value = m_producer();  // Lazy evaluation
        }
        return *m_value;
    }

    template <vko::device_and_commands DeviceAndCommands>
    const T& get(const DeviceAndCommands& device) const {
        if (!m_value) {
            m_semaphore.wait(device);
            m_value = m_producer();  // Lazy evaluation
        }
        return *m_value;
    }

    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(const DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        return m_semaphore.waitFor(device, duration);
    }

    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(const DeviceAndCommands& device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        return m_semaphore.waitUntil(device, deadline);
    }

    template <vko::device_and_commands DeviceAndCommands>
    bool ready(const DeviceAndCommands& device) const { return m_semaphore.isSignaled(device); }

    const SemaphoreValue& semaphore() const { return m_semaphore; }

private:
    SemaphoreValue           m_semaphore;
    std::function<T()>       m_producer;
    mutable std::optional<T> m_value;
};


// A shared version for a forced evaluation.
template <class T>
class SharedLazyTimelineFuture {
public:
    template <class SV, class Fn>
    SharedLazyTimelineFuture(SV&& semaphore, Fn&& producer)
        : m_state(std::make_shared<State>(std::forward<SV>(semaphore), std::forward<Fn>(producer))) {}

    template <vko::device_and_commands DeviceAndCommands>
    T& get(const DeviceAndCommands& device) {
        if (!m_state->producer) {
            throw TimelineSubmitCancel();
        }
        if (!m_state->value) {
            m_state->semaphore.wait(device);
            m_state->value = m_state->producer();  // Lazy evaluation
        }
        return *m_state->value;
    }

    template <vko::device_and_commands DeviceAndCommands>
    const T& get(const DeviceAndCommands& device) const {
        if (!m_state->producer) {
            throw TimelineSubmitCancel();
        }
        if (!m_state->value) {
            m_state->semaphore.wait(device);
            m_state->value = m_state->producer();  // Lazy evaluation
        }
        return *m_state->value;
    }

    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(const DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        return m_state.semaphore.waitFor(device, duration);
    }

    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(const DeviceAndCommands& device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        return m_state.semaphore.waitUntil(device, deadline);
    }

    template <vko::device_and_commands DeviceAndCommands>
    bool ready(const DeviceAndCommands& device) const { return m_state->semaphore.isSignaled(device); }

    const SemaphoreValue& semaphore() const { return m_state->semaphore; }

    // Returns a type erased function that will produce the value if it's not
    // already produced.
    template <device_and_commands DeviceAndCommands>
    std::function<void(bool)> evaluator(const DeviceAndCommands& device) const {
        return [state = m_state, &device](bool call) {
            if (call) {
                if (!state->value) {
                    state->semaphore.wait(device);
                    state->value = state->producer();
                }
            } else {
                state->producer = nullptr;
            }
        };
    }

    // Returns a type erased function that will produce the value if it's not
    // already produced and the shared future still exists.
    template<device_and_commands DeviceAndCommands>
    std::function<void(bool)> weakEvaluator(const DeviceAndCommands& device) const {
        return [weak = std::weak_ptr<State>(m_state), &device](bool call) {
            if (auto state = weak.lock()) {
                if (call) {
                    if (!state->value) {
                        state->semaphore.wait(device);
                        state->value = state->producer();
                    }
                } else {
                    state->producer = nullptr;
                }
            }
        };
    }

private:
    struct State {
        std::function<T()>       producer;
        mutable std::optional<T> value;
        SemaphoreValue           semaphore; // already shared, but whatever
    };

    std::shared_ptr<State> m_state;
};

// Generic queue that tracks completion with a semaphore for each item. It is
// assumed that semaphores will complete in the order they were pushed.
// TODO: heavy use of callbacks. would views be nicer or good in addition?
template <typename T>
class CompletionQueue {
public:
    CompletionQueue() = default;

    // Push an entry with its completion semaphore
    void push_back(T&& entry, SemaphoreValue semaphore) {
        m_entries.push_back({std::move(entry), std::move(semaphore)});
    }

    // Allow direct semaphore access so user can wait with timeouts
    const SemaphoreValue& frontSemaphore() const { return m_entries.front().semaphore; }

    void pop_front() { m_entries.pop_front(); }

    // Block until the front/oldest entry is ready and return a reference to it.
    template <device_and_commands DeviceAndCommands>
    T& front(const DeviceAndCommands& device) {
        frontSemaphore().wait(device);
        return m_entries.front().value;
    }

    // Checks all entries from the front and calls the callback for those that
    // are ready
    template <device_and_commands DeviceAndCommands, class Fn>
        requires std::invocable<Fn, T&>
    size_t visitReady(const DeviceAndCommands& device, Fn&& callback) {
        auto next = m_entries.begin();
        while (next != m_entries.end() && next->semaphore.isSignaled(device)) {
            if constexpr (std::same_as<std::invoke_result_t<Fn, T&>, bool>) {
                if(!callback(next->value))
                    break;
            } else {
                callback(next->value);
            }
            ++next;
        }
        return std::distance(m_entries.begin(), next);
    }

    // Same as visitReady, but also removes the ready entries from the queue
    template <device_and_commands DeviceAndCommands, class Fn>
        requires std::invocable<Fn, T&>
    size_t consumeReady(const DeviceAndCommands& device, Fn&& callback) {
        auto next = m_entries.begin();
        while (next != m_entries.end() && next->semaphore.isSignaled(device)) {
            if constexpr (std::same_as<std::invoke_result_t<Fn, T&>, bool>) {
                if(!callback(next->value))
                    break;
            } else {
                callback(next->value);
            }
            ++next;
        }
        size_t processed = std::distance(m_entries.begin(), next);
        m_entries.erase(m_entries.begin(), next);
        return processed;
    }

    // Same as visitReady, but removes the ready entries from the queue as long
    // there are at least keepReadyCount ready entries after it. Returns the
    // number of entries removed.
    template <device_and_commands DeviceAndCommands>
    size_t visitAndConsumeReadyWithMargin(const DeviceAndCommands& device, size_t keepReadyCount,
                                          std::function<void(T&)> callback) {
        auto next   = m_entries.begin();
        auto remove = next;
        while (next != m_entries.end() && next->semaphore.isSignaled(device)) {
            callback(next->value);
            ++next;
            if (keepReadyCount > 0) {
                --keepReadyCount;
            } else {
                ++remove;
            }
        }
        size_t removed = std::distance(m_entries.begin(), remove);
        m_entries.erase(m_entries.begin(), remove);
        return removed;
    }

    // Waits for all entries to be ready
    template <device_and_commands DeviceAndCommands>
    void wait(const DeviceAndCommands& device) {
        if (!m_entries.empty()) {
            // Assumes waiting on the newest semaphore guarantees all older
            // semaphores are also signaled
            m_entries.back().semaphore.wait(device);
        }
    }

    // Waits for all entries to be ready and calls the callback for each and
    // clears the queue
    template <device_and_commands DeviceAndCommands, class Fn>
        requires std::invocable<Fn, T&>
    void waitAndConsume(const DeviceAndCommands& device, Fn&& callback) {
        if (!m_entries.empty()) {
            // Assumes waiting on the newest semaphore guarantees all older
            // semaphores are also signaled
            m_entries.back().semaphore.wait(device);
            for (auto& entry : m_entries) {
                callback(entry.value);
            }
            m_entries.clear();
        }
    }

    // Access to entries without checking if they are ready
    void visitAll(std::function<void(T&)> callback) {
        for (auto& entry : m_entries) {
            callback(entry.value);
        }
    }

    size_t size() const { return m_entries.size(); }
    bool empty() const { return m_entries.empty(); }

private:
    struct Entry {
        T              value;
        SemaphoreValue semaphore;
    };

    std::deque<Entry> m_entries;
};

// Generic submit wrapper
template <device_and_commands           DeviceAndCommands,
          std::ranges::contiguous_range Range = std::initializer_list<VkSubmitInfo2>>
    requires std::same_as<std::ranges::range_value_t<Range>, VkSubmitInfo2>
void submit(const DeviceAndCommands& device, VkQueue queue, Range&& submitInfos,
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
void submit(const DeviceAndCommands& device, VkQueue queue, WaitRange&& waitInfos,
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
void submit(const DeviceAndCommands& device, VkQueue queue, WaitRange&& waitInfos,
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
    void submit(const DeviceAndCommands& device, WaitRange&& waitInfos,
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
    void submit(const DeviceAndCommands& device, WaitRange&& waitInfos,
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

