// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <vko/handles.hpp>
#include <vko/timeline_queue.hpp>

namespace vko {

namespace simple {

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
    operator VkCommandBuffer() const { return m_commandBuffer; }
    explicit operator bool() const { return static_cast<bool>(m_commandBuffer); }
    CommandBuffer&& end() {
        check(vkEndCommandBuffer(m_commandBuffer));
        return std::move(m_commandBuffer);
    }

private:
    CommandBuffer          m_commandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
};

template <class Queue = VkQueue>
class ImmediateCommandBuffer;

template <>
class ImmediateCommandBuffer<VkQueue> {
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

template <device_and_commands DeviceAndCommands>
ImmediateCommandBuffer(const DeviceAndCommands& vk, VkCommandPool commandPool,
                       VkQueue queue) -> ImmediateCommandBuffer<VkQueue>;

template <>
class ImmediateCommandBuffer<TimelineQueue> {
public:
    template <device_and_commands DeviceAndCommands>
    ImmediateCommandBuffer(const DeviceAndCommands& vk, VkCommandPool commandPool,
                           TimelineQueue& queue)
        : ImmediateCommandBuffer(vk, vk, commandPool, queue) {}
    template <class Commands>
    ImmediateCommandBuffer(VkCommandPool commandPool, TimelineQueue& queue, VkDevice device,
                           const Commands& vk)
        : m_commandBuffer(
              CommandBuffer(nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, device, vk),
              VkCommandBufferBeginInfo{
                  .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                  .pNext            = nullptr,
                  .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                  .pInheritanceInfo = nullptr,
              },
              vk)
        , m_queue(queue)
        , vkQueueWaitIdle(vk.vkQueueWaitIdle) {}
    ~ImmediateCommandBuffer() {
        SynchronizedCommandBuffer readyCmd(m_commandBuffer.end());
        // TODO: stageMask??
        readyCmd.stageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        m_queue.submit({&readyCmd});
        vkQueueWaitIdle(m_queue);
    }
    operator VkCommandBuffer() const { return m_commandBuffer; }

private:
    RecordingCommandBuffer m_commandBuffer;
    TimelineQueue&         m_queue;
    PFN_vkQueueWaitIdle    vkQueueWaitIdle;
};

template <device_and_commands DeviceAndCommands>
ImmediateCommandBuffer(const DeviceAndCommands& vk, VkCommandPool commandPool,
                       TimelineQueue queue) -> ImmediateCommandBuffer<TimelineQueue>;

} // namespace simple

} // namespace vko
