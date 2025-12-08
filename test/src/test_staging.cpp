// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/allocator.hpp>
#include <vko/staging_memory.hpp>
#include <vko/timeline_queue.hpp>

TEST_F(UnitTestFixture, RecyclingStagingPool_BasicConstruction) {
    size_t minPools = 3;
    VkDeviceSize poolSize = 1 << 24; // 16MB
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator, 
                                                         minPools, /*maxPools=*/5, poolSize);
    
    // Should have minimum pools pre-allocated
    EXPECT_EQ(staging.capacity(), minPools * poolSize);
    EXPECT_EQ(staging.size(), 0u); // No buffers allocated yet
}

TEST_F(UnitTestFixture, RecyclingStagingPool_SimpleAllocation) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator, 
        /*minPools=*/2, /*maxPools=*/5, /*poolSize=*/1 << 20); // 1MB pools
    
    bool callbackInvoked = false;
    auto* buffer = staging.tryMake<uint32_t>(100, [&callbackInvoked](bool signaled) {
        callbackInvoked = true;
        EXPECT_TRUE(signaled); // Should be signaled when properly released
    });
    
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->size(), 100u);
    EXPECT_GT(staging.size(), 0u);
    
    // Write some data to verify the buffer is usable
    {
        auto mapping = buffer->map();
        for (size_t i = 0; i < 100; ++i) {
            mapping[i] = static_cast<uint32_t>(i);
        }
    } // mapping destroyed here
    
    // End the batch with a signaled semaphore
    auto semaphore = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(semaphore);
    
    // Wait should invoke callbacks
    staging.wait();
    EXPECT_TRUE(callbackInvoked);
    EXPECT_EQ(staging.size(), 0u); // All buffers released
}

TEST_F(UnitTestFixture, RecyclingStagingPool_PoolRecycling) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 16); // 64KB pools
    
    // This test never forces new pool allocations, so capacity should never
    // change
    VkDeviceSize initialCapacity = staging.capacity();
    
    // Allocate and release first batch
    auto* buffer1 = staging.tryMake<uint32_t>(1000, [](bool) {});
    ASSERT_NE(buffer1, nullptr);
    EXPECT_EQ(staging.capacity(), initialCapacity);
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    EXPECT_EQ(staging.capacity(), initialCapacity);
    staging.wait();
    EXPECT_EQ(staging.capacity(), initialCapacity);
    
    // Allocate second batch - should reuse pool
    auto* buffer2 = staging.tryMake<uint32_t>(1000, [](bool) {});
    ASSERT_NE(buffer2, nullptr);
    
    // Capacity should not have increased (recycled pool)
    EXPECT_EQ(staging.capacity(), initialCapacity);
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

TEST_F(UnitTestFixture, RecyclingStagingPool_PartialAllocation) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16); // 64KB pools
    
    // Request more than pool size - should get partial allocation
    size_t hugeSize = (1 << 20) / sizeof(uint32_t); // 1MB worth of uint32_t
    auto* buffer = staging.tryMake<uint32_t>(hugeSize, [](bool) {});
    
    ASSERT_NE(buffer, nullptr);
    EXPECT_LT(buffer->size(), hugeSize); // Should be less than requested
    EXPECT_GT(buffer->size(), 0u);       // But not zero
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

TEST_F(UnitTestFixture, RecyclingStagingPool_MultiplePoolsInBatch) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/5, /*poolSize=*/1 << 14); // 16KB pools
    
    // Allocate enough to span multiple pools
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    
    for (int i = 0; i < 10; ++i) {
        auto* buffer = staging.tryMake<uint32_t>(1000, [](bool) {}); // ~4KB each
        if (buffer) {
            buffers.push_back(buffer);
        }
    }
    
    EXPECT_GT(buffers.size(), 0u);
    EXPECT_GT(staging.capacity(), staging.poolSize()); // Should have multiple pools
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// TODO: Future test cases for more complete coverage:
// - RecyclingStagingPool_MaxPoolsReached: Test hitting maxPools limit, tryMake returns nullptr
// - RecyclingStagingPool_BlockingBehavior: Test blocking when at max pools until semaphore signals
// - RecyclingStagingPool_TryWithCallback: Test tryWith() interface with populate callback
// - RecyclingStagingPool_CallbackOnDestruct: Test destruct callbacks are called with false on early destruction
// - RecyclingStagingPool_MultipleBatchesInFlight: Test multiple batches with different semaphores
// - RecyclingStagingPool_SemaphoreNotSignaled: Test buffers stay in use until semaphore signals
// - RecyclingStagingPool_GreedyCallbackInvocation: Test callbacks invoked as soon as semaphore ready
// - RecyclingStagingPool_MemoryAlignment: Test allocations respect alignment requirements
// - RecyclingStagingPool_ExcessPoolFreeing: Test wait() frees pools beyond minPools
// - RecyclingStagingPool_MoveSemantics: Test move constructor and move assignment
// - RecyclingStagingPool_EmptyBatch: Test endBatch() with no allocations
// - RecyclingStagingPool_ZeroSizeAllocation: Test edge case of size=0
// - RecyclingStagingPool_ActualDataTransfer: Test actual upload/download with command buffers
// - StreamingStaging_BasicUsage: Test StreamingStaging wrapper
// - StreamingStaging_AutoSubmit: Test automatic submission on threshold
// - StreamingStaging_CommandBufferReuse: Test command buffer recycling
