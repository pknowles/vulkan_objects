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
    // explicit operator bool() const & { return static_cast<bool>(m_commandBuffer); }
    bool            engaged() const { return m_commandBuffer.engaged(); }
    CommandBuffer&& end() {
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
    ~ImmediateCommandBuffer() {
        if (static_cast<bool>(m_commandBuffer)) {
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
            vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_queue);
        }
    }
    void addWait(VkSemaphore semaphore, VkPipelineStageFlags stageMask) {
        m_waitSemaphores.push_back(semaphore);
        m_waitSemaphoreStageMasks.push_back(stageMask);
    }
    void addSignal(VkSemaphore semaphore) { m_signalSemaphores.push_back(semaphore); }
    operator VkCommandBuffer() const { return m_commandBuffer; }

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

} // namespace vko
