// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "vulkan/vulkan_core.h"
#include <vko/handles.hpp>

namespace vko {

namespace simple {

// TODO: maybe remove. I love the compile-time state this provides, but for it
// to actually be useful the vkCmd*() commands would need to only accept the
// recording command buffer
class RecordingCommandBuffer {
public:
    template <class Functions>
    RecordingCommandBuffer(CommandBuffer&& commandBuffer, const VkCommandBufferBeginInfo& beginInfo,
                           const Functions& vk)
        : m_commandBuffer(std::move(commandBuffer))
        , vkEndCommandBuffer(vk.vkEndCommandBuffer) {
        check(vk.vkBeginCommandBuffer(m_commandBuffer, &beginInfo));
    }
    operator VkCommandBuffer() const { return m_commandBuffer; }
    operator bool() const { return static_cast<bool>(m_commandBuffer); }
    CommandBuffer&& end() {
        check(vkEndCommandBuffer(m_commandBuffer));
        return std::move(m_commandBuffer);
    }

private:
    CommandBuffer          m_commandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
};

class ImmediateCommandBuffer {
public:
    template <class FunctionsAndParent>
    ImmediateCommandBuffer(VkCommandPool commandPool, VkQueue queue, const FunctionsAndParent& vk)
        : ImmediateCommandBuffer(commandPool, queue, vk, vk) {}
    template <class Functions>
    ImmediateCommandBuffer(VkCommandPool commandPool, VkQueue queue, VkDevice device,
                           const Functions& vk)
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
        , vkQueueSubmit(vk.vkQueueSubmit)
        , vkQueueWaitIdle(vk.vkQueueWaitIdle) {}
    ~ImmediateCommandBuffer() {
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

} // namespace vko
