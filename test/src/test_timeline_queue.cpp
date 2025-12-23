// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <chrono>
#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/timeline_queue.hpp>

TEST_F(UnitTestFixture, TimelineQueue_BasicConstruction) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    
    // Test accessor methods
    EXPECT_EQ(queue.familyIndex(), ctx->queueFamilyIndex);
    EXPECT_EQ(queue.deviceIndex(), 0u);
    EXPECT_NE(static_cast<VkQueue>(queue), VK_NULL_HANDLE);
    EXPECT_NE(queue.ptr(), nullptr);
    
    ctx->device.vkQueueWaitIdle(queue);
}

TEST_F(UnitTestFixture, TimelineQueue_NextSubmitSemaphore) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0, 100);
    
    // Get the next semaphore value before submitting
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    
    // The value should not be set yet (hasValue should return false)
    EXPECT_FALSE(nextValue.hasValue());
}

TEST_F(UnitTestFixture, TimelineQueue_SignalInfoAndAdvance) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0, 50);
    
    // Get signal info and verify it advances
    VkSemaphoreSubmitInfo info1 = queue.signalInfoAndAdvance(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    EXPECT_EQ(info1.sType, VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
    EXPECT_EQ(info1.value, 51u);
    EXPECT_EQ(info1.stageMask, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    EXPECT_EQ(info1.deviceIndex, 0u);
    EXPECT_NE(info1.semaphore, VK_NULL_HANDLE);
    
    // Next one should have incremented value
    VkSemaphoreSubmitInfo info2 = queue.signalInfoAndAdvance(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(info2.value, 52u);
    EXPECT_EQ(info2.stageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(info2.semaphore, info1.semaphore); // Same semaphore
}

TEST_F(UnitTestFixture, TimelineQueue_SimpleSubmit) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    // Test the raw submit API - record and end command buffer manually
    vko::simple::RecordingCommandBuffer recording = ctx->beginRecording(commandPool);
    VkCommandBuffer cmdBuffer = recording.end();
    
    // Submit using the queue's .submit() method
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // Wait for completion
    ctx->device.vkQueueWaitIdle(queue);
}

TEST_F(UnitTestFixture, TimelineQueue_MultipleSubmissions) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    // Submit multiple command buffers
    constexpr int numSubmissions = 5;
    for (int i = 0; i < numSubmissions; ++i) {
        VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                     cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
    
    // Wait for all to complete
    ctx->device.vkQueueWaitIdle(queue);
}

TEST_F(UnitTestFixture, SemaphoreValue_AlreadySignalled) {
    auto signalled = vko::SemaphoreValue::makeSignalled();
    
    // Should already be ready without any GPU work
    EXPECT_TRUE(signalled.isSignaled(ctx->device));
}

TEST_F(UnitTestFixture, SemaphoreValue_WaitOnSubmission) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    // Get the semaphore value for the next submission
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    EXPECT_FALSE(nextValue.hasValue()); // Not submitted yet
    
    // Submit a command buffer
    VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // Now the value should be set
    EXPECT_TRUE(nextValue.hasValue());
    
    // Wait for it
    nextValue.wait(ctx->device);
    
    // Should be signaled now
    EXPECT_TRUE(nextValue.isSignaled(ctx->device));
}

TEST_F(UnitTestFixture, SemaphoreValue_TryWait) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // TryWait should succeed (no cancellation)
    EXPECT_TRUE(nextValue.tryWait(ctx->device));
}

TEST_F(UnitTestFixture, SemaphoreValue_WaitFor) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // WaitFor with a reasonable timeout should succeed for simple empty commands
    EXPECT_TRUE(nextValue.waitFor(ctx->device, std::chrono::seconds(5)));
}

TEST_F(UnitTestFixture, SemaphoreValue_WaitUntil) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // WaitUntil with future deadline should succeed
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    EXPECT_TRUE(nextValue.waitUntil(ctx->device, deadline));
}

TEST_F(UnitTestFixture, SemaphoreValue_WaitSemaphoreInfo) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    vko::SemaphoreValue nextValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // Get a VkSemaphoreSubmitInfo for waiting on this value
    VkSemaphoreSubmitInfo waitInfo = nextValue.waitSemaphoreInfo(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
    EXPECT_EQ(waitInfo.sType, VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
    EXPECT_NE(waitInfo.semaphore, VK_NULL_HANDLE);
    EXPECT_GT(waitInfo.value, 0u);
    EXPECT_EQ(waitInfo.stageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(waitInfo.deviceIndex, 1u);
    
    // Wait to complete
    nextValue.wait(ctx->device);
}

TEST_F(UnitTestFixture, TimelineQueue_SubmitWithDependency) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();
    
    // First submission
    vko::SemaphoreValue firstValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer1 = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, 
                 cmdBuffer1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // Second submission that waits on the first
    VkSemaphoreSubmitInfo waitInfo = firstValue.waitSemaphoreInfo(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    vko::SemaphoreValue secondValue = queue.nextSubmitSemaphore();
    VkCommandBuffer cmdBuffer2 = ctx->beginRecording(commandPool).end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{waitInfo}, 
                 cmdBuffer2, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    
    // Wait for the second one, which implies the first also completed
    secondValue.wait(ctx->device);
    EXPECT_TRUE(firstValue.isSignaled(ctx->device));
    EXPECT_TRUE(secondValue.isSignaled(ctx->device));
}
