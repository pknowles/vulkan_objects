// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vector>
#include <vko/exceptions.hpp>
#include <vko/handles.hpp>
#include <vko/timeline_queue.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

namespace simple {

// Creates a primary command buffer from a command pool and calls the provided
// function between begin/end recording. It is expected the user
// vkResetCommandPool() the command pool periodically as the returned command
// buffer is raw and does not free itself.
template <device_commands DeviceCommands, class Fn>
VkCommandBuffer recordCommands(const DeviceCommands& vk, VkCommandPool commandPool, Fn&& fn) {
    VkCommandBufferAllocateInfo allocateInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer;
    check(vk.vkAllocateCommandBuffers(vk, &allocateInfo, &commandBuffer));
    VkCommandBufferBeginInfo beginInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    check(vk.vkBeginCommandBuffer(commandBuffer, &beginInfo));
    fn(commandBuffer);
    check(vk.vkEndCommandBuffer(commandBuffer));
    return commandBuffer;
}

// TODO: maybe remove. I love the compile-time state this provides, but for it
// to actually be useful the vkCmd*() commands would need to only accept the
// recording command buffer
class RecordingCommandBuffer {
public:
    template <device_commands DeviceCommands>
    RecordingCommandBuffer(const DeviceCommands& vk, CommandBuffer&& commandBuffer,
                           const VkCommandBufferBeginInfo& beginInfo)
        : m_commandBuffer(std::move(commandBuffer))
        , vkEndCommandBuffer(vk.vkEndCommandBuffer) {
        check(vk.vkBeginCommandBuffer(m_commandBuffer, &beginInfo));
    }
    operator VkCommandBuffer() const& { return m_commandBuffer; }
    operator VkCommandBuffer() && = delete;
    bool            engaged() const { return m_commandBuffer.engaged(); }
    CommandBuffer&& end() {
        assert(engaged());
        if (engaged())
            check(vkEndCommandBuffer(m_commandBuffer));
        return std::move(m_commandBuffer);
    }

private:
    CommandBuffer          m_commandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
};

class ImmediateCommandBuffer {
public:
    template <device_and_commands DeviceAndCommands>
    ImmediateCommandBuffer(const DeviceAndCommands& vk, VkCommandPool commandPool, VkQueue queue)
        : ImmediateCommandBuffer(vk, vk, commandPool, queue) {}
    template <device_commands DeviceCommands>
    ImmediateCommandBuffer(const DeviceCommands& vk, VkDevice device, VkCommandPool commandPool,
                           VkQueue queue)
        : m_commandBuffer(
              vk, CommandBuffer(vk, device, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY),
              VkCommandBufferBeginInfo{
                  .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                  .pNext            = nullptr,
                  .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                  .pInheritanceInfo = nullptr,
              })
        , m_queue(queue)
        , vkQueueSubmit(vk.vkQueueSubmit)
        , vkQueueWaitIdle(vk.vkQueueWaitIdle) {}
    ImmediateCommandBuffer(ImmediateCommandBuffer&& other) noexcept = default;
    ImmediateCommandBuffer& operator=(ImmediateCommandBuffer&& other) noexcept
    {
        submit();
        m_commandBuffer = std::move(other.m_commandBuffer);
        m_queue = other.m_queue;
        vkQueueSubmit = other.vkQueueSubmit;
        vkQueueWaitIdle = other.vkQueueWaitIdle;
        return *this;
    }
    ~ImmediateCommandBuffer() {
        submit();
    }
    void addWait(VkSemaphore semaphore, VkPipelineStageFlags stageMask) {
        m_waitSemaphores.push_back(semaphore);
        m_waitSemaphoreStageMasks.push_back(stageMask);
    }
    void addSignal(VkSemaphore semaphore) { m_signalSemaphores.push_back(semaphore); }
    operator VkCommandBuffer() const { return m_commandBuffer; }
    void submit() {
        if (m_commandBuffer.engaged()) {
            CommandBuffer cmd(m_commandBuffer.end());
            VkSubmitInfo  submitInfo{
                 .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                 .pNext                = nullptr,
                 .waitSemaphoreCount   = uint32_t(m_waitSemaphores.size()),
                 .pWaitSemaphores      = m_waitSemaphores.data(),
                 .pWaitDstStageMask    = m_waitSemaphoreStageMasks.data(),
                 .commandBufferCount   = 1U,
                 .pCommandBuffers      = cmd.ptr(),
                 .signalSemaphoreCount = uint32_t(m_signalSemaphores.size()),
                 .pSignalSemaphores    = m_signalSemaphores.data(),
            };
            vko::check(vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE));
            vko::check(vkQueueWaitIdle(m_queue));
        }
    }

private:
    RecordingCommandBuffer            m_commandBuffer;
    VkQueue                           m_queue;
    std::vector<VkSemaphore>          m_waitSemaphores;
    std::vector<VkPipelineStageFlags> m_waitSemaphoreStageMasks;
    std::vector<VkSemaphore>          m_signalSemaphores;
    PFN_vkQueueSubmit                 vkQueueSubmit;
    PFN_vkQueueWaitIdle               vkQueueWaitIdle;
};

} // namespace simple

// Pairs a recording command buffer with a submit promise. It is common to want
// to know when commands in a command buffer finish. Note that this doesn't make
// sense to use with command buffers that are submitted multiple times.
struct TimelineCommandBuffer {
    simple::RecordingCommandBuffer cmd;
    SubmitPromise                  promise;

    // Implicit conversion to VkCommandBuffer for convenience
    operator VkCommandBuffer() const { return cmd; }
};

// Manages command buffer lifecycle with automatic recycling of completed buffers.
// The destructor waits for all in-flight command buffers to complete, ensuring
// safe cleanup even if GPU work is still executing.
// Note: A CyclingCommandBufferRef version could be created to allow sharing
// a CommandPool across multiple command buffer instances, following the same
// pattern as StagingStreamRef/StagingStream.
template <device_and_commands DeviceAndCommands = Device, class Queue = TimelineQueue>
class CyclingCommandBuffer {
public:
    using DeviceAndCommandsType = DeviceAndCommands;
    using QueueType             = Queue;

    CyclingCommandBuffer(const DeviceAndCommands& device, Queue& queue)
        : m_device(device)
        , m_queue(queue)
        , m_commandPool(device, VkCommandPoolCreateInfo{
                                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .pNext = nullptr,
                                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = queue.familyIndex(),
                                }) {}

    CyclingCommandBuffer(CyclingCommandBuffer&&) = default;
    CyclingCommandBuffer& operator=(CyclingCommandBuffer&& other) noexcept {
        tryWait();
        m_device       = other.m_device;
        m_queue        = other.m_queue;
        m_commandPool  = std::move(other.m_commandPool);
        m_inFlightCmds = std::move(other.m_inFlightCmds);
        m_current      = std::move(other.m_current);
        return *this;
    }

    ~CyclingCommandBuffer() { std::ignore = tryWait(); }

    // Manual submission interface.
    SemaphoreValue submit(VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        return submit({}, {}, timelineSemaphoreStageMask);
    }

    // TODO: to be really generic and match the Vulkan API, I think we want a "Submission" object
    template <class WaitRange   = std::initializer_list<VkSemaphoreSubmitInfo>,
              class SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>>
    SemaphoreValue submit(WaitRange&& waitInfos, SignalRange&& signalInfos,
                          VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        if (!m_current) {
            // No current command buffer. We can't return makeSignalled()
            // because the user may be relying on sequential submissions
            // completing in order. There may also be waits/signals to insert
            // into the queue. Sadly, it's more robust to create an empty command buffer.
            std::ignore = commandBuffer();
        }

        SemaphoreValue result = m_current->promise.futureValue();
        CommandBuffer  cmd(m_current->cmd.end());

        if (m_pendingWaits.empty()) {
            m_queue.get().submit(m_device.get(), std::forward<WaitRange>(waitInfos), cmd,
                                 std::move(m_current->promise), timelineSemaphoreStageMask,
                                 std::forward<SignalRange>(signalInfos));
        } else {
            m_pendingWaits.insert(m_pendingWaits.begin(), std::begin(waitInfos),
                                  std::end(waitInfos));
            m_queue.get().submit(m_device.get(), m_pendingWaits, cmd, std::move(m_current->promise),
                                 timelineSemaphoreStageMask,
                                 std::forward<SignalRange>(signalInfos));
            m_pendingWaits.clear();
        }
        m_current.reset();

        m_inFlightCmds.push_back({std::move(cmd), result});
        return result;
    }

    void submitAndWait(VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        submit(timelineSemaphoreStageMask).wait(m_device.get());
    }

    // TODO: maybe limit access via a callback? LLMs love to store the result
    // and it's really dangerous due to it cycling.
    const TimelineCommandBuffer& commandBuffer() {
        if (!m_current) {
            m_current =
                TimelineCommandBuffer{simple::RecordingCommandBuffer(
                                          m_device.get(), reuseOrMakeCommandBuffer(),
                                          VkCommandBufferBeginInfo{
                                              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .pNext = nullptr,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr,
                                          }),
                                      m_queue.get().submitPromise()};
        }
        return *m_current;
    }

    // Check to see if commandBuffer() was called since the last submit. Handy
    // to skip operations if no commands were recorded.
    bool hasCurrent() const { return m_current.has_value(); }

    // Implicit conversion to VkCommandBuffer for convenience
    // WARNING: Do NOT store the result as VkCommandBuffer - the handle can change after submit!
    // Always use the CyclingCommandBuffer reference directly, which will fetch
    // the current handle on each use via this conversion operator.
    operator VkCommandBuffer() { return commandBuffer(); }

    // Convenience helper to get the semaphore value for the current command buffer's next submit.
    SemaphoreValue nextSubmitSemaphore() { return commandBuffer().promise.futureValue(); }

    // Appends the given semaphore wait to the next submission. The caller may
    // require the next submission to happen after some GPU semaphore but may
    // not control when the submission happens. An equivalent call to add a
    // semaphore to signal is not expected to be needed as the user would call
    // submit() at the time.
    void waitOnNextSubmit(const VkSemaphoreSubmitInfo& submitWaitInfo) {
        m_pendingWaits.push_back(submitWaitInfo);
    }

    const DeviceAndCommands& device() const { return m_device.get(); }
    Queue&                   queue() { return m_queue.get(); }

private:
    struct InFlightCmd {
        CommandBuffer  commandBuffer;
        SemaphoreValue readySemaphore;
    };

    CommandBuffer reuseOrMakeCommandBuffer() {
        // Try to reuse a command buffer from the in-flight queue
        if (!m_inFlightCmds.empty() &&
            m_inFlightCmds.front().readySemaphore.isSignaled(m_device.get())) {
            auto result = std::move(m_inFlightCmds.front().commandBuffer);
            m_inFlightCmds.pop_front();

            // If there's a relatively long queue of ready command buffers, free
            // some up. Leave at least one of the ready ones to avoid frequent
            // allocations/deallocations.
            if (m_inFlightCmds.size() >= 2 &&
                m_inFlightCmds.front().readySemaphore.isSignaled(m_device.get())) {
                while (
                    m_inFlightCmds.size() >= 2 &&
                    std::next(m_inFlightCmds.begin())->readySemaphore.isSignaled(m_device.get())) {
                    m_inFlightCmds.pop_front();
                }
            }

            m_device.get().vkResetCommandBuffer(result, 0);
            return result;
        }

        // Else, create a new command buffer
        return CommandBuffer(m_device.get(), nullptr, m_commandPool,
                             VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    }

    // Wait for all in-flight command buffers to complete before destroying the command pool
    bool tryWait() const {
        bool result = true;
        for (auto& inFlight : m_inFlightCmds) {
            // tryWait returns false if cancelled/never submitted, but won't throw
            result = inFlight.readySemaphore.tryWait(m_device.get()) && result;
        }
        return result;
    }

    std::reference_wrapper<const DeviceAndCommands> m_device;
    std::reference_wrapper<Queue>                   m_queue;
    CommandPool                                     m_commandPool;
    std::deque<InFlightCmd>                         m_inFlightCmds;
    std::optional<TimelineCommandBuffer>            m_current;
    std::vector<VkSemaphoreSubmitInfo>              m_pendingWaits;
};

} // namespace vko
