// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vector>
#include <vko/bound_buffer.hpp>
#include <vko/query_pool.hpp>
#include <vko/timeline_queue.hpp>

namespace {
// Helper struct to hold buffers for GPU work (must stay alive until GPU completes)
struct DummyWork {
    vko::BoundBuffer<uint8_t> buf1;
    vko::BoundBuffer<uint8_t> buf2;

    DummyWork(const vko::Device& device, vko::vma::Allocator& allocator)
        : buf1(device, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               allocator)
        , buf2(device, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               allocator) {}

    void record(const vko::Device& device, VkCommandBuffer cmd) {
        VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = 4096};
        device.vkCmdCopyBuffer(cmd, buf1, buf2, 1, &region);
    }
};
} // namespace

// Use-case: Frame-based GPU timestamp profiling (non-shared)
// Tests the complete workflow of timing GPU operations across a simulated frame.
// This is the most common use case for query pools in real applications.
TEST_F(UnitTestFixture, QueryBatch_TimestampProfiling_NonShared) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";
    ASSERT_GT(props.limits.timestampPeriod, 0.0f) << "Invalid timestampPeriod";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    // Create a non-shared timestamp batch builder
    vko::TimestampQueryBatchBuilder<> builder;

    auto recording = ctx->beginRecording(commandPool);

    // Create buffers that must outlive GPU work
    DummyWork dummyWork(ctx->device, ctx->allocator);

    // Write timestamps - returns QueryHandle (lightweight index)
    auto handleBegin = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                              VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
    dummyWork.record(ctx->device, recording);
    auto handleEnd = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

    auto               submitSemaphore = queue.nextSubmitSemaphore();
    vko::CommandBuffer cmdBuffer       = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    // Complete the builder into a batch
    auto batch = complete(std::move(builder), submitSemaphore);

    // Get results - blocks until GPU completes
    uint64_t timestampBegin = batch.get(ctx->device, handleBegin);
    uint64_t timestampEnd   = batch.get(ctx->device, handleEnd);

    // Verify timestamps are reasonable
    EXPECT_GT(timestampEnd, timestampBegin);

    // Convert to nanoseconds using timestampPeriod
    double elapsedNs = (timestampEnd - timestampBegin) * props.limits.timestampPeriod;
    EXPECT_GT(elapsedNs, 0.0);
    EXPECT_LT(elapsedNs, 1e9); // Should be less than 1 second

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Frame-based GPU timestamp profiling (shared)
// Tests SharedQuery handles that survive batch destruction.
TEST_F(UnitTestFixture, QueryBatch_TimestampProfiling_Shared) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    // Create a shared timestamp batch builder
    vko::SharedTimestampQueryBatchBuilder<> builder;

    auto recording = ctx->beginRecording(commandPool);

    // Create buffers that must outlive GPU work
    DummyWork dummyWork(ctx->device, ctx->allocator);

    // Write timestamps - returns SharedQuery (holds shared_ptr to batch state)
    auto sharedBegin = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                              VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
    dummyWork.record(ctx->device, recording);
    auto sharedEnd = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

    auto               submitSemaphore = queue.nextSubmitSemaphore();
    vko::CommandBuffer cmdBuffer       = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    // Complete the builder
    auto batch = complete(std::move(builder), submitSemaphore);

    // Get results via SharedQuery handles
    uint64_t timestampBegin = sharedBegin.get(ctx->device);
    uint64_t timestampEnd   = sharedEnd.get(ctx->device);

    EXPECT_GT(timestampEnd, timestampBegin);

    double elapsedNs = (timestampEnd - timestampBegin) * props.limits.timestampPeriod;
    EXPECT_GT(elapsedNs, 0.0);
    EXPECT_LT(elapsedNs, 1e9);

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Occlusion query with RAII scoped query (non-shared)
// Tests begin/end query pattern for measuring visible samples.
TEST_F(UnitTestFixture, ScopedQuery_Occlusion_NonShared) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    vko::OcclusionQueryBatchBuilder<> builder;

    auto recording = ctx->beginRecording(commandPool);

    vko::QueryHandle handle;
    {
        vko::ScopedQuery<uint64_t, decltype(builder)> scopedQuery(ctx->device, recording, builder);
        handle = scopedQuery.handle();
        // In a real app, draw calls would go here
        // The query automatically ends when scopedQuery goes out of scope
    }

    auto               submitSemaphore = queue.nextSubmitSemaphore();
    vko::CommandBuffer cmdBuffer       = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    auto batch = complete(std::move(builder), submitSemaphore);

    // Result should be 0 (no actual rendering happened)
    uint64_t result = batch.get(ctx->device, handle);
    EXPECT_EQ(result, 0u);

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Occlusion query with RAII scoped query (shared)
TEST_F(UnitTestFixture, ScopedQuery_Occlusion_Shared) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    vko::SharedOcclusionQueryBatchBuilder<> builder;

    auto recording = ctx->beginRecording(commandPool);

    vko::SharedQuery<uint64_t, VK_QUERY_TYPE_OCCLUSION> sharedHandle = [&]() {
        vko::ScopedQuery<uint64_t, decltype(builder)> scopedQuery(ctx->device, recording, builder);
        return scopedQuery.handle();
    }();

    auto               submitSemaphore = queue.nextSubmitSemaphore();
    vko::CommandBuffer cmdBuffer       = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    auto batch = complete(std::move(builder), submitSemaphore);

    // Can get result via the SharedQuery handle directly
    uint64_t result = sharedHandle.get(ctx->device);
    EXPECT_EQ(result, 0u);

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: SharedQuery handles survive batch destruction
// Tests that shared handles remain valid after batch goes out of scope.
TEST_F(UnitTestFixture, QueryBatch_SharedHandlesSurviveBatchDestruction) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    // Store handles outside the scope where batch exists
    std::optional<vko::SharedQuery<uint64_t, VK_QUERY_TYPE_TIMESTAMP>> sharedBegin;
    std::optional<vko::SharedQuery<uint64_t, VK_QUERY_TYPE_TIMESTAMP>> sharedEnd;

    {
        vko::SharedTimestampQueryBatchBuilder<> builder;
        DummyWork                               dummyWork(ctx->device, ctx->allocator);

        auto recording = ctx->beginRecording(commandPool);

        sharedBegin = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        dummyWork.record(ctx->device, recording);
        sharedEnd = vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                           VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

        auto               submitSemaphore = queue.nextSubmitSemaphore();
        vko::CommandBuffer cmdBuffer       = recording.end();
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        // Complete and immediately destroy the batch
        auto batch = complete(std::move(builder), submitSemaphore);

        // Wait for GPU before command buffer goes out of scope
        vko::check(ctx->device.vkQueueWaitIdle(queue));
        // batch and dummyWork go out of scope here
    }

    // Handles should still work because they hold shared_ptr to state
    uint64_t timestampBegin = sharedBegin->get(ctx->device);
    uint64_t timestampEnd   = sharedEnd->get(ctx->device);

    EXPECT_GT(timestampEnd, timestampBegin);
}

// Use-case: Pool recycling with QueryPoolRecycler
// Tests that pools are properly recycled when batches complete.
TEST_F(UnitTestFixture, QueryBatch_PoolRecycling) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    // Use small pool capacity to force multiple pools
    constexpr uint32_t                            poolCapacity = 4;
    vko::TimestampQueryPoolRecycler<poolCapacity> recycler;

    // First batch - creates new pools
    std::optional<vko::CommandBuffer> cmdBuffer1;
    {
        vko::TimestampQueryBatchBuilder<poolCapacity> builder;

        auto recording = ctx->beginRecording(commandPool);

        // Allocate enough queries to need multiple pools
        for (int i = 0; i < 6; ++i) {
            vko::cmdWriteTimestamp(ctx->device, recording, builder, recycler,
                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

        auto submitSemaphore = queue.nextSubmitSemaphore();
        cmdBuffer1           = recording.end();
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, *cmdBuffer1,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        auto batch = complete(std::move(builder), submitSemaphore);

        // Recycle pools back (this waits for the semaphore)
        recycler.push(batch.recyclePools(ctx->device));
    }

    EXPECT_FALSE(recycler.empty());

    // Second batch - should reuse recycled pools
    std::optional<vko::CommandBuffer> cmdBuffer2;
    {
        vko::TimestampQueryBatchBuilder<poolCapacity> builder;
        DummyWork                                     dummyWork(ctx->device, ctx->allocator);

        auto recording = ctx->beginRecording(commandPool);

        // Allocate queries - should come from recycled pools
        auto h1 = vko::cmdWriteTimestamp(ctx->device, recording, builder, recycler,
                                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        dummyWork.record(ctx->device, recording);
        auto h2 = vko::cmdWriteTimestamp(ctx->device, recording, builder, recycler,
                                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

        auto submitSemaphore = queue.nextSubmitSemaphore();
        cmdBuffer2           = recording.end();
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, *cmdBuffer2,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        auto batch = complete(std::move(builder), submitSemaphore);

        uint64_t t1 = batch.get(ctx->device, h1);
        uint64_t t2 = batch.get(ctx->device, h2);
        EXPECT_GT(t2, t1);
    }

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Multiple frames with QueryBatchRecycler
// Tests that SharedQueryBatch objects are properly managed by the batch recycler.
TEST_F(UnitTestFixture, QueryBatch_MultiFrameRecycling) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    constexpr uint32_t                             poolCapacity = 4;
    vko::TimestampQueryBatchRecycler<poolCapacity> recycler;

    // Keep command buffers alive until GPU work completes
    std::vector<vko::CommandBuffer> cmdBuffers;

    // Simulate 3 frames
    for (int frame = 0; frame < 3; ++frame) {
        vko::SharedTimestampQueryBatchBuilder<poolCapacity> builder;

        auto recording = ctx->beginRecording(commandPool);

        // Allocate 2 queries per frame
        vko::cmdWriteTimestamp(ctx->device, recording, builder, recycler,
                               VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        vko::cmdWriteTimestamp(ctx->device, recording, builder, recycler,
                               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

        auto submitSemaphore = queue.nextSubmitSemaphore();
        cmdBuffers.push_back(recording.end());
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffers.back(),
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        auto batch = complete(std::move(builder), submitSemaphore);
        recycler.push(std::move(batch));
    }

    // Wait for all frames to complete
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Reclaim ready batches
    recycler.reclaimReady(ctx->device);

    // Should have pools available for reuse
    EXPECT_TRUE(recycler.tryPop(ctx->device).has_value());
}

// Use-case: Recycling while holding SharedQuery handles
// Tests that results are cached and accessible even after pools are recycled.
TEST_F(UnitTestFixture, QueryBatch_RecyclingWhileHoldingHandles) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    constexpr uint32_t                             poolCapacity = 4;
    vko::TimestampQueryBatchRecycler<poolCapacity> recycler;

    // Store handles from multiple batches
    std::vector<vko::SharedQuery<uint64_t, VK_QUERY_TYPE_TIMESTAMP, poolCapacity>> handles;
    std::vector<vko::CommandBuffer>                                                cmdBuffers;

    // Create several batches and push them to recycler
    for (int batchIdx = 0; batchIdx < 3; ++batchIdx) {
        vko::SharedTimestampQueryBatchBuilder<poolCapacity> builder;

        auto recording = ctx->beginRecording(commandPool);

        // Store handles for later
        handles.push_back(vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));
        handles.push_back(vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));

        auto submitSemaphore = queue.nextSubmitSemaphore();
        cmdBuffers.push_back(recording.end());
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffers.back(),
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        auto batch = complete(std::move(builder), submitSemaphore);

        // Push batch to recycler - recycler will reclaim pools when semaphore signals
        recycler.push(std::move(batch));
    }

    // Wait for GPU
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Trigger reclamation - pools are recycled but results should be cached
    recycler.reclaimReady(ctx->device);

    // Handles should still be valid - results are cached in shared state
    for (const auto& handle : handles) {
        uint64_t result = handle.get(ctx->device);
        EXPECT_GT(result, 0u);
    }
}

// Use-case: Multiple pools within single batch
// Tests that allocation correctly spans multiple pools when capacity is exceeded.
TEST_F(UnitTestFixture, QueryBatch_MultiplePools) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    // Very small pool to force multiple pools
    constexpr uint32_t                            poolCapacity = 2;
    vko::TimestampQueryBatchBuilder<poolCapacity> builder;

    auto recording = ctx->beginRecording(commandPool);

    // Allocate 5 queries - needs 3 pools with capacity 2
    std::vector<vko::QueryHandle> handles;
    for (int i = 0; i < 5; ++i) {
        handles.push_back(vko::cmdWriteTimestamp(ctx->device, recording, builder,
                                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));
    }

    auto               submitSemaphore = queue.nextSubmitSemaphore();
    vko::CommandBuffer cmdBuffer       = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    auto batch = complete(std::move(builder), submitSemaphore);

    // All queries should have valid results
    uint64_t prevTimestamp = 0;
    for (const auto& handle : handles) {
        uint64_t timestamp = batch.get(ctx->device, handle);
        EXPECT_GE(timestamp, prevTimestamp);
        prevTimestamp = timestamp;
    }

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Empty batch (no queries allocated)
TEST_F(UnitTestFixture, QueryBatch_EmptyBatch) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool         commandPool = ctx->createCommandPool();

    vko::TimestampQueryBatchBuilder<> builder;

    auto               recording       = ctx->beginRecording(commandPool);
    vko::CommandBuffer cmdBuffer       = recording.end();
    auto               submitSemaphore = queue.nextSubmitSemaphore();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{}, cmdBuffer,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    // Complete with no queries allocated - should not crash
    auto batch = complete(std::move(builder), submitSemaphore);

    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// TODO: Future test cases
//
// Additional tests to add as the implementation matures:
//
// QueryBatchBuilder / QueryBatch:
//   • Test with pipeline statistics queries (multiple results per query)
//   • Test with VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
//   • Test concurrent batch building from multiple threads (thread safety)
//   • Test behavior when device is lost
//   • Test with very large query counts (stress test)
//   • Test batch.download() explicit pre-fetch
//
// SharedQuery:
//   • Test SharedQuery copy semantics
//   • Test SharedQuery move semantics
//   • Test multiple SharedQuery handles to same query
//   • Test SharedQuery validity after batch destruction
//
// QueryPoolRecycler:
//   • Test recycler with mismatched pool capacities (should not compile)
//   • Test recycler.pop() when empty
//   • Test pushing pools from different devices (should assert/fail)
//
// QueryBatchRecycler:
//   • Test reclaimReady() with no ready batches
//   • Test pop() returning recycled pools
//   • Test mixing shared and non-shared batches (compile error expected)
//
// ScopedQuery:
//   • Test exception safety (query ends even if exception thrown)
//   • Test with VK_QUERY_CONTROL_PRECISE_BIT
//   • Test nested scoped queries (hierarchical profiling)
//   • Test moving ScopedQuery (should not be possible)
//
// Type Safety:
//   • Verify compile-time error when mixing uint32_t and uint64_t pools
//   • Test with both 32-bit and 64-bit occlusion query results
//   • Test query_recycler concept with custom recycler type
//
// Integration:
//   • Test with multiple query types in same frame
//   • Test query batches with different lifetimes overlapping
//   • Test with render pass integration
//   • Test with multiple queues (transfer + graphics)
//
// Performance:
//   • Benchmark allocation overhead vs raw vkCreateQueryPool
//   • Benchmark pool recycling latency
//   • Profile memory usage with varying pool sizes
//   • Compare shared vs non-shared handle overhead
//
// Edge Cases:
//   • Test endBatch() / complete() with no allocations between
//   • Test destroying batch with pending GPU work (should wait)
//   • Test query result reading order (verify caching works)
//   • Test recyclePools() called multiple times (should be safe)
//
