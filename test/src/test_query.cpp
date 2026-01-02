// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/query_pool.hpp>
#include <vko/timeline_queue.hpp>

// Use-case: Frame-based GPU timestamp profiling
// Tests the complete workflow of timing GPU operations across a simulated frame.
// This is the most common use case for query pools in real applications.
TEST_F(UnitTestFixture, RecyclingQueryPool_TimestampProfiling) {
    // Verify timestamps are supported
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics) 
        << "Timestamps not supported on this device";
    ASSERT_GT(props.limits.timestampPeriod, 0.0f)
        << "Invalid timestampPeriod";

    // Create a timeline queue and command pool for submissions
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();

    // Create timestamp query pool (uint64_t results)
    vko::RecyclingQueryPool<uint64_t> queries(ctx->device, VK_QUERY_TYPE_TIMESTAMP,
                                               /*queriesPerPool=*/256,
                                               /*minPools=*/2,
                                               /*maxPools=*/5);

    EXPECT_EQ(queries.poolCount(), 2u);  // Pre-allocated minimum pools
    EXPECT_EQ(queries.queriesPerPool(), 256u);
    EXPECT_EQ(queries.capacity(), 2u * 256u);

    // Simulate a frame: record timestamps for begin/end
    auto recording = ctx->beginRecording(commandPool);
    VkCommandBuffer cmd = recording;

    // Allocate queries - type-safe Query<uint64_t>
    auto queryBegin = queries.allocate();
    auto queryEnd = queries.allocate();

    // Write timestamps using the helper function
    auto futureBegin = vko::cmdWriteTimestamp(ctx->device, cmd, queryBegin,
                                               VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                               queue.nextSubmitSemaphore());

    // ... render commands would go here ...

    auto futureEnd = vko::cmdWriteTimestamp(ctx->device, cmd, queryEnd,
                                             VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                             queue.nextSubmitSemaphore());

    // End command buffer and submit
    VkCommandBuffer cmdBuffer = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{},
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    // Mark queries as recyclable once GPU is done (use semaphore from one of the futures)
    queries.endBatch(futureBegin.semaphore());

    // Get results - blocks until GPU completes
    uint64_t timestampBegin = futureBegin.get(ctx->device);
    uint64_t timestampEnd = futureEnd.get(ctx->device);

    // Verify timestamps are reasonable
    EXPECT_GT(timestampEnd, timestampBegin);
    uint64_t elapsed = timestampEnd - timestampBegin;
    
    // Convert to nanoseconds using timestampPeriod
    double elapsedNs = elapsed * props.limits.timestampPeriod;
    EXPECT_GT(elapsedNs, 0.0);
    EXPECT_LT(elapsedNs, 1e9);  // Should be less than 1 second

    // Verify type safety - ResultType should be uint64_t
    static_assert(std::same_as<decltype(queries)::ResultType, uint64_t>);
    static_assert(std::same_as<decltype(queryBegin)::ResultType, uint64_t>);
    static_assert(std::same_as<decltype(futureBegin)::ResultType, uint64_t>);

    // Wait for all work to complete before cleanup
    queries.wait();
    ctx->device.vkQueueWaitIdle(queue);
}

// Use-case: Occlusion query with RAII scoped query
// Tests begin/end query pattern for measuring visible samples.
// Common for conditional rendering and visibility determination.
TEST_F(UnitTestFixture, RecyclingQueryPool_OcclusionQuery) {
    // Create a timeline queue and command pool
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();

    // Create occlusion query pool (can use uint32_t for sample counts)
    vko::RecyclingQueryPool<uint64_t> queries(ctx->device, VK_QUERY_TYPE_OCCLUSION,
                                               /*queriesPerPool=*/128,
                                               /*minPools=*/1,
                                               /*maxPools=*/3);

    EXPECT_EQ(queries.poolCount(), 1u);
    EXPECT_EQ(queries.capacity(), 128u);

    // Record command buffer with scoped query
    auto recording = ctx->beginRecording(commandPool);

    auto query = queries.allocate();

    // Use ScopedQuery for automatic begin/end and create future
    vko::QueryResultFuture<uint64_t> future = [&]() {
        vko::ScopedQuery<uint64_t> scopedQuery(ctx->device, recording, query);
        // In a real app, draw calls would go here
        // The query automatically ends when scopedQuery goes out of scope
        return scopedQuery.future(queue.nextSubmitSemaphore());
    }(); // Immediately invoked lambda ensures scopedQuery destructor runs before recording.end()

    // Verify type safety
    static_assert(std::same_as<decltype(future)::ResultType, uint64_t>);

    // Submit and mark queries recyclable
    VkCommandBuffer cmdBuffer = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{},
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    queries.endBatch(future.semaphore());

    // Try non-blocking read first
    auto result = future.tryGet(ctx->device);
    if (!result) {
        // Not ready yet, wait for it
        result = future.get(ctx->device);
    }

    // Result should be 0 (no actual rendering happened)
    EXPECT_EQ(*result, 0u);

    // Test advanced semaphore access
    EXPECT_TRUE(future.semaphore().ready(ctx->device));

    // Cleanup
    queries.wait();
    ctx->device.vkQueueWaitIdle(queue);
}

// Use-case: Multiple frames with pool recycling
// Tests that pools are properly recycled when batches complete.
TEST_F(UnitTestFixture, RecyclingQueryPool_MultiFrameRecycling) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();

    // Small pool to force recycling
    vko::RecyclingQueryPool<uint64_t> queries(ctx->device, VK_QUERY_TYPE_TIMESTAMP,
                                               /*queriesPerPool=*/4,  // Small!
                                               /*minPools=*/1,
                                               /*maxPools=*/2);

    size_t initialPoolCount = queries.poolCount();

    // Simulate 3 frames
    for (int frame = 0; frame < 3; ++frame) {
        auto recording = ctx->beginRecording(commandPool);
        VkCommandBuffer cmd = recording;

        // Allocate 2 queries per frame
        auto q1 = queries.allocate();
        auto q2 = queries.allocate();

        auto f1 = vko::cmdWriteTimestamp(ctx->device, cmd, q1,
                              VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                              queue.nextSubmitSemaphore());
        vko::cmdWriteTimestamp(ctx->device, cmd, q2,
                              VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                              queue.nextSubmitSemaphore());

        VkCommandBuffer cmdBuffer = recording.end();
        queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{},
                     cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        queries.endBatch(f1.semaphore());
    }

    // Wait for all frames to complete
    queries.wait();
    ctx->device.vkQueueWaitIdle(queue);

    // After recycling, should be back to minimum pools
    EXPECT_EQ(queries.poolCount(), initialPoolCount);
}

// Use-case: Pool exhaustion and blocking allocation
// Tests behavior when allocating more queries than a single pool can hold.
TEST_F(UnitTestFixture, RecyclingQueryPool_PoolExpansion) {
    VkPhysicalDeviceProperties props;
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ASSERT_TRUE(props.limits.timestampComputeAndGraphics)
        << "Timestamps not supported on this device";

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::CommandPool commandPool = ctx->createCommandPool();

    // Very small pool to test expansion
    vko::RecyclingQueryPool<uint64_t> queries(ctx->device, VK_QUERY_TYPE_TIMESTAMP,
                                               /*queriesPerPool=*/2,  // Tiny!
                                               /*minPools=*/1,
                                               /*maxPools=*/3);

    EXPECT_EQ(queries.poolCount(), 1u);

    auto recording = ctx->beginRecording(commandPool);

    // Allocate more queries than fit in one pool
    std::vector<vko::Query<uint64_t>> allocatedQueries;
    for (int i = 0; i < 5; ++i) {  // More than 2 queries per pool
        allocatedQueries.push_back(queries.allocate());
    }
    
    // Write a timestamp so we have a semaphore to use for endBatch
    auto timestampFuture = vko::cmdWriteTimestamp(ctx->device, recording, allocatedQueries[0],
                                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                   queue.nextSubmitSemaphore());

    // Should have expanded to more pools
    EXPECT_GT(queries.poolCount(), 1u);
    EXPECT_LE(queries.poolCount(), 3u);  // Should not exceed maxPools

    // Cleanup
    VkCommandBuffer cmdBuffer = recording.end();
    queue.submit(ctx->device, std::initializer_list<VkSemaphoreSubmitInfo>{},
                 cmdBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    queries.endBatch(timestampFuture.semaphore());
    queries.wait();
    ctx->device.vkQueueWaitIdle(queue);
}

// ============================================================================
// TODO: Future test cases
// ============================================================================
//
// Additional tests to add as the implementation matures:
//
// RecyclingQueryPool:
//   • Test tryAllocate() returning nullopt at max capacity
//   • Test with pipeline statistics queries (multiple results per query)
//   • Test query pool reset correctness (write after reset)
//   • Test with VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
//   • Test concurrent allocation from multiple threads (thread safety)
//   • Test behavior when device is lost
//   • Test with very large query counts (stress test)
//   • Test wait() timeout behavior
//
// QueryResultFuture:
//   • Test get() with VK_QUERY_RESULT_WAIT_BIT behavior
//   • Test tryGet() returning nullopt before GPU completion
//   • Test accessing semaphore() for custom synchronization
//   • Test with custom VkQueryResultFlags combinations
//
// ScopedQuery:
//   • Test exception safety (query ends even if exception thrown)
//   • Test with VK_QUERY_CONTROL_PRECISE_BIT
//   • Test nested scoped queries (hierarchical profiling)
//   • Test moving ScopedQuery (should not be possible)
//
// Type Safety:
//   • Verify compile-time error when mixing uint32_t and uint64_t pools
//   • Test Query<T>::ResultType propagation through generic code
//   • Test with both 32-bit and 64-bit occlusion query results
//
// Integration:
//   • Test with multiple query types in same frame
//   • Test query pools with different lifetimes overlapping
//   • Test with render pass integration
//   • Test with multiple queues (transfer + graphics)
//
// Performance:
//   • Benchmark allocation overhead
//   • Benchmark pool recycling latency
//   • Profile memory usage with varying pool sizes
//
// Edge Cases:
//   • Test allocating zero queries
//   • Test endBatch() with no allocations
//   • Test multiple endBatch() calls without allocations between
//   • Test destroying pool with pending queries
//   • Test query result reading with mismatched type size
//

