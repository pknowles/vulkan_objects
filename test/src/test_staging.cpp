// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/allocator.hpp>
#include <vko/staging_memory.hpp>
#include <vko/timeline_queue.hpp>
#include <atomic>
#include <chrono>
#include <future>
#include <numeric>
#include <thread>

// Use-case: Basic staging pool initialization
// Verifies that the pool pre-allocates the minimum number of memory pools upfront,
// reducing allocation overhead during the first transfers.
TEST_F(UnitTestFixture, RecyclingStagingPool_BasicConstruction) {
    size_t minPools = 3;
    VkDeviceSize poolSize = 1 << 24; // 16MB
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator, 
                                                         minPools, /*maxPools=*/5, poolSize);
    
    // Should have minimum pools pre-allocated
    EXPECT_EQ(staging.capacity(), minPools * poolSize);
    EXPECT_EQ(staging.size(), 0u); // No buffers allocated yet
}

// Use-case: Single small CPU→GPU transfer with synchronization
// Tests the complete lifecycle: allocate staging buffer, write data, end batch with
// timeline semaphore, and verify the destruction callback fires when signaled.
TEST_F(UnitTestFixture, RecyclingStagingPool_SimpleAllocation) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator, 
        /*minPools=*/2, /*maxPools=*/5, /*poolSize=*/1 << 20); // 1MB pools
    
    bool callbackInvoked = false;
    auto* buffer = staging.allocateUpTo<uint32_t>(100, [&callbackInvoked](bool signaled) {
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

// Use-case: Memory efficiency through pool reuse
// Tests that completed staging buffers are recycled rather than allocating new memory,
// crucial for applications doing continuous streaming without memory growth.
TEST_F(UnitTestFixture, RecyclingStagingPool_PoolRecycling) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 16); // 64KB pools
    
    // This test never forces new pool allocations, so capacity should never
    // change
    VkDeviceSize initialCapacity = staging.capacity();
    
    // Allocate and release first batch
    auto* buffer1 = staging.allocateUpTo<uint32_t>(1000, [](bool) {});
    ASSERT_NE(buffer1, nullptr);
    EXPECT_EQ(staging.capacity(), initialCapacity);
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    EXPECT_EQ(staging.capacity(), initialCapacity);
    staging.wait();
    EXPECT_EQ(staging.capacity(), initialCapacity);
    
    // Allocate second batch - should reuse pool
    auto* buffer2 = staging.allocateUpTo<uint32_t>(1000, [](bool) {});
    ASSERT_NE(buffer2, nullptr);
    
    // Capacity should not have increased (recycled pool)
    EXPECT_EQ(staging.capacity(), initialCapacity);
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: Handling requests larger than available pool size
// When a single transfer request exceeds the pool size, the allocator should return
// a partial allocation rather than failing. Caller can loop to handle the remainder.
TEST_F(UnitTestFixture, RecyclingStagingPool_PartialAllocation) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16); // 64KB pools
    
    // Request more than pool size - should get partial allocation
    size_t hugeSize = (1 << 20) / sizeof(uint32_t); // 1MB worth of uint32_t
    auto* buffer = staging.allocateUpTo<uint32_t>(hugeSize, [](bool) {});
    
    ASSERT_NE(buffer, nullptr);
    EXPECT_LT(buffer->size(), hugeSize); // Should be less than requested
    EXPECT_GT(buffer->size(), 0u);       // But not zero
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: Batch of small transfers that collectively exceed one pool
// Tests dynamic pool allocation when a single batch requires multiple pools.
// All allocations in one batch should be tracked together for synchronization.
TEST_F(UnitTestFixture, RecyclingStagingPool_MultiplePoolsInBatch) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/5, /*poolSize=*/1 << 14); // 16KB pools
    
    // Allocate enough to span multiple pools
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    
    for (int i = 0; i < 10; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(1000, [](bool) {}); // ~4KB each
        if (buffer) {
            buffers.push_back(buffer);
        }
    }
    
    EXPECT_GT(buffers.size(), 0u);
    EXPECT_GT(staging.capacity(), staging.poolSize()); // Should have multiple pools
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: Memory budget enforcement - prevent runaway allocation
// When maxPools is reached and all pools are in use, allocateUpTo() should return nullptr
// rather than allocating more memory, allowing the application to handle backpressure.
TEST_F(UnitTestFixture, RecyclingStagingPool_MaxPoolsReached) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 12); // 4KB pools
    
    // Fill up both pools completely
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    for (int i = 0; i < 100; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(900, [](bool) {}); // ~3.6KB each
        if (buffer) {
            buffers.push_back(buffer);
        } else {
            break; // Hit the limit
        }
    }
    
    // Should have allocated at least some buffers
    EXPECT_GT(buffers.size(), 0u);
    
    // Next allocation should fail (at max pools, all in use)
    auto* failedBuffer = staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_EQ(failedBuffer, nullptr);
    
    // Clean up
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: RAII-style staging buffer with automatic cleanup
// Tests tryWith() which combines allocation, population, and destruction callback
// registration in one call - convenient for small one-off transfers.
TEST_F(UnitTestFixture, RecyclingStagingPool_TryWithCallback) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
    
    bool populateCalled = false;
    bool destructCalled = false;
    
    bool success = staging.tryWith<float>(100, 
        [&populateCalled](const vko::BoundBuffer<float, vko::vma::Allocator>& buffer) {
            populateCalled = true;
            // Verify we can write to the buffer
            auto mapping = buffer.map();
            mapping[0] = 3.14f;
        },
        [&destructCalled](bool signaled) {
            destructCalled = true;
            EXPECT_TRUE(signaled);
        });
    
    EXPECT_TRUE(success);
    EXPECT_TRUE(populateCalled);
    EXPECT_FALSE(destructCalled); // Not yet
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
    EXPECT_TRUE(destructCalled);
}

// Use-case: Cleanup of unsignaled/incomplete transfers on shutdown
// Verifies that destruction callbacks receive false when the semaphore hasn't signaled,
// allowing the application to detect and handle incomplete transfers during cleanup.
TEST_F(UnitTestFixture, RecyclingStagingPool_CallbackOnDestruct) {
    bool callback1Called = false;
    bool callback1Signaled = true; // Default to true to detect if it's set
    bool callback2Called = false;
    bool callback2Signaled = true;
    
    {
        vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
            /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
        
        // Allocate but don't end batch - should get false on destruct
        auto* buffer1 = staging.allocateUpTo<uint32_t>(100, [&](bool signaled) {
            callback1Called = true;
            callback1Signaled = signaled;
        });
        ASSERT_NE(buffer1, nullptr);
        
        // Allocate and end batch with unsignaled semaphore - should also get false
        auto* buffer2 = staging.allocateUpTo<uint32_t>(100, [&](bool signaled) {
            callback2Called = true;
            callback2Signaled = signaled;
        });
        ASSERT_NE(buffer2, nullptr);
        
        // Create an unsignaled timeline semaphore and SemaphoreValue
        vko::TimelineSemaphore sem(ctx->device, 0);
        std::promise<uint64_t> promise;
        promise.set_value(1); // Set the value but semaphore is still at 0
        staging.endBatch(vko::SemaphoreValue(sem, promise.get_future().share()));
        
        // Destructor called here - both callbacks should be called with false
    }
    
    EXPECT_TRUE(callback1Called);
    EXPECT_FALSE(callback1Signaled);
    EXPECT_TRUE(callback2Called);
    EXPECT_FALSE(callback2Signaled);
}

// Use-case: Overlapping transfers for maximum GPU utilization
// Tests multiple batches in flight simultaneously, each with independent semaphores.
// Callbacks should fire independently as each batch completes, not blocking each other.
TEST_F(UnitTestFixture, RecyclingStagingPool_MultipleBatchesInFlight) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    
    int batch1Callbacks = 0;
    int batch2Callbacks = 0;
    int batch3Callbacks = 0;
    
    // Batch 1
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch1Callbacks++; });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch1Callbacks++; });
    auto sem1 = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(sem1);
    
    // Batch 2
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch2Callbacks++; });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch2Callbacks++; });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch2Callbacks++; });
    auto sem2 = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(sem2);
    
    // Batch 3
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { if(s) batch3Callbacks++; });
    auto sem3 = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(sem3);
    
    staging.wait();
    
    EXPECT_EQ(batch1Callbacks, 2);
    EXPECT_EQ(batch2Callbacks, 3);
    EXPECT_EQ(batch3Callbacks, 1);
}

// Use-case: Asynchronous GPU work with delayed signaling
// Tests that buffers remain valid until their semaphore signals, even when new batches
// are started. Pools shouldn't be recycled until GPU work completes.
TEST_F(UnitTestFixture, RecyclingStagingPool_SemaphoreNotSignaled) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
    
    bool callbackInvoked = false;
    
    // Create timeline semaphore
    vko::TimelineSemaphore sem(ctx->device, 0);
    
    auto* buffer = staging.allocateUpTo<uint32_t>(100, [&](bool signaled) {
        callbackInvoked = true;
        EXPECT_TRUE(signaled);
    });
    ASSERT_NE(buffer, nullptr);
    
    VkDeviceSize sizeBeforeEnd = staging.size();
    EXPECT_GT(sizeBeforeEnd, 0u);
    
    // End batch with unsignaled semaphore
    std::promise<uint64_t> promise1;
    promise1.set_value(1);
    staging.endBatch(vko::SemaphoreValue(sem, promise1.get_future().share()));
    
    // Size should still include the buffer (not released yet)
    EXPECT_EQ(staging.size(), sizeBeforeEnd);
    EXPECT_FALSE(callbackInvoked);
    
    // Try to allocate more - should still work as we have pools available
    auto* buffer2 = staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_NE(buffer2, nullptr);
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    
    // Now signal the first semaphore
    VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = sem,
        .value = 1,
    };
    ctx->device.vkSignalSemaphore(ctx->device, &signalInfo);
    
    // Wait should process both batches
    staging.wait();
    EXPECT_TRUE(callbackInvoked);
    EXPECT_EQ(staging.size(), 0u);
}

// Use-case: Lazy cleanup - minimize CPU overhead during transfers
// Verifies that destruction callbacks are invoked only when pools are actually recycled
// (at the last possible moment), not eagerly during endBatch(). This maximizes the time
// resources remain valid for potential reuse and defers cleanup overhead.
TEST_F(UnitTestFixture, RecyclingStagingPool_CallbackOnRecycle) {
    // This test verifies that destroy callbacks are invoked at the last possible
    // moment: when we actually recycle a pool from a completed batch. They should
    // NOT be invoked during endBatch() or allocation attempts - only when the pool
    // is being reclaimed for reuse.
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/3, /*maxPools=*/3, /*poolSize=*/1 << 16);
    
    int callbackCount = 0;
    
    // Allocate a buffer with callback from first pool
    staging.allocateUpTo<uint32_t>(100, [&](bool s) { 
        if(s) callbackCount++; 
    });
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    
    // Callback should NOT be invoked by endBatch
    EXPECT_EQ(callbackCount, 0) << "endBatch() should not invoke callbacks";
    
    // Allocate from second pool - first pool is still not recycled
    staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_EQ(callbackCount, 0) << "Callback not yet invoked (using different pool)";
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    EXPECT_EQ(callbackCount, 0) << "endBatch() should not invoke callbacks";
    
    // Allocate from third pool - first pool still not recycled
    staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_EQ(callbackCount, 0) << "Callback not yet invoked (using third pool)";
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    EXPECT_EQ(callbackCount, 0) << "endBatch() should not invoke callbacks";
    
    // Now all 3 pools are in use. Next allocation MUST recycle first batch's pool.
    // This is when the callback gets invoked - at the last possible moment.
    staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_EQ(callbackCount, 1) << "Callback should be invoked when recycling pool";
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: GPU-friendly memory layout for optimal transfer performance
// Verifies that all allocated buffers meet Vulkan alignment requirements, ensuring
// transfers work correctly and efficiently across different GPU architectures.
TEST_F(UnitTestFixture, RecyclingStagingPool_MemoryAlignment) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
    
    // Allocate several buffers and verify they're properly aligned
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    
    for (int i = 0; i < 10; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(100, [](bool) {});
        if (buffer) {
            buffers.push_back(buffer);
            
            // Get the buffer's device address to check alignment
            VkBufferDeviceAddressInfo info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = *buffer,
            };
            VkDeviceAddress addr = ctx->device.vkGetBufferDeviceAddress(ctx->device, &info);
            
            // Should be aligned to at least the buffer's alignment requirement
            // For transfer buffers, this is typically 4 or 16 bytes
            EXPECT_EQ(addr % 4, 0u) << "Buffer " << i << " not aligned";
        }
    }
    
    EXPECT_GT(buffers.size(), 0u);
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: Memory reuse without deallocation - steady-state performance
// Tests that wait() releases buffers (size → 0) but retains pools (capacity unchanged),
// keeping memory warm for subsequent batches without expensive allocations.
TEST_F(UnitTestFixture, RecyclingStagingPool_WaitFreesBuffersNotPools) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/5, /*poolSize=*/1 << 16);
    
    VkDeviceSize initialCapacity = staging.capacity();
    
    // Allocate buffers using multiple pools
    for (int i = 0; i < 20; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(1000, [](bool) {});
        if (!buffer) break;
    }
    
    VkDeviceSize capacityAfterAlloc = staging.capacity();
    VkDeviceSize sizeAfterAlloc = staging.size();
    EXPECT_GE(capacityAfterAlloc, initialCapacity);
    EXPECT_GT(sizeAfterAlloc, 0u);
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
    
    // After wait: buffers freed (size=0) but pools remain (capacity unchanged)
    EXPECT_EQ(staging.size(), 0u);
    EXPECT_EQ(staging.capacity(), capacityAfterAlloc); // Capacity stays the same
    EXPECT_GE(staging.capacity(), initialCapacity); // At least minPools remain
}

// Use-case: Transferring ownership of staging resources (e.g., returning from factory)
// Verifies that RecyclingStagingPool can be safely moved, preserving all state including
// pending batches and callbacks, enabling flexible resource management patterns.
TEST_F(UnitTestFixture, RecyclingStagingPool_MoveSemantics) {
    bool callback1Called = false;
    bool callback2Called = false;
    
    vko::vma::RecyclingStagingPool<vko::Device> staging1(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
    
    staging1.allocateUpTo<uint32_t>(100, [&](bool s) { callback1Called = s; });
    VkDeviceSize size1 = staging1.size();
    EXPECT_GT(size1, 0u);
    
    // Move construct
    vko::vma::RecyclingStagingPool<vko::Device> staging2(std::move(staging1));
    EXPECT_EQ(staging2.size(), size1);
    
    staging2.allocateUpTo<uint32_t>(100, [&](bool s) { callback2Called = s; });
    
    // Move assign
    vko::vma::RecyclingStagingPool<vko::Device> staging3(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 16);
    staging3 = std::move(staging2);
    
    EXPECT_GT(staging3.size(), 0u);
    
    staging3.endBatch(vko::SemaphoreValue::makeSignalled());
    staging3.wait();
    
    EXPECT_TRUE(callback1Called);
    EXPECT_TRUE(callback2Called);
}

// Use-case: Handling no-op frames (e.g., no new geometry this frame)
// Tests that endBatch() with no allocations is a safe no-op, allowing applications
// to unconditionally call endBatch() every frame without special-casing empty batches.
TEST_F(UnitTestFixture, RecyclingStagingPool_EmptyBatch) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    
    VkDeviceSize initialSize = staging.size();
    VkDeviceSize initialCapacity = staging.capacity();
    
    // End a batch without any allocations
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    
    // Size and capacity should be unchanged
    EXPECT_EQ(staging.size(), initialSize);
    EXPECT_EQ(staging.capacity(), initialCapacity);
    
    // Should still be able to allocate normally
    auto* buffer = staging.allocateUpTo<uint32_t>(100, [](bool) {});
    EXPECT_NE(buffer, nullptr);
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: Blocking behavior when pools exhausted
// Tests that allocateUpTo() blocks waiting for a semaphore when all pools are in use,
// and unblocks once the semaphore is signaled, allowing pool recycling.
TEST_F(UnitTestFixture, RecyclingStagingPool_BlockingBehavior) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 12); // 4KB pools
    
    // Create timeline semaphore that starts unsignaled
    vko::TimelineSemaphore sem(ctx->device, 0);
    
    // Fill both pools completely
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    for (int i = 0; i < 100; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(900, [](bool) {}); // ~3.6KB each
        if (!buffer) break;
        buffers.push_back(buffer);
    }
    EXPECT_GT(buffers.size(), 0u);
    
    // End batch with unsignaled semaphore - pools are now all in use
    std::promise<uint64_t> promise1;
    promise1.set_value(1);
    staging.endBatch(vko::SemaphoreValue(sem, promise1.get_future().share()));
    
    // Note: allocateUpTo() will BLOCK (not return nullptr) when all pools are in use.
    // We test this blocking behavior using a thread.
    
    // Test blocking behavior: allocate in a thread, signal from main thread
    std::atomic<bool> allocationStarted{false};
    std::atomic<bool> allocationCompleted{false};
    vko::BoundBuffer<uint32_t, vko::vma::Allocator>* threadBuffer = nullptr;
    
    std::thread allocThread([&]() {
        allocationStarted = true;
        threadBuffer = staging.allocateUpTo<uint32_t>(100, [](bool) {});
        allocationCompleted = true;
    });
    
    // Wait for thread to start and begin blocking (with timeout)
    auto startTime = std::chrono::steady_clock::now();
    while (!allocationStarted && 
           std::chrono::steady_clock::now() - startTime < std::chrono::seconds(2)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(allocationStarted) << "Thread didn't start within timeout";
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Give it time to block
    
    // Should still be blocked
    EXPECT_FALSE(allocationCompleted.load());
    
    // Signal the semaphore to unblock
    VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = sem,
        .value = 1,
    };
    ctx->device.vkSignalSemaphore(ctx->device, &signalInfo);
    
    // Wait for allocation to complete with timeout
    bool joined = false;
    startTime = std::chrono::steady_clock::now();
    while (!allocationCompleted && 
           std::chrono::steady_clock::now() - startTime < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (allocationCompleted) {
        allocThread.join();
        joined = true;
    } else {
        // Thread is hung - detach it to avoid terminate() on destruction
        // This is not ideal but prevents the test suite from hanging
        allocThread.detach();
        FAIL() << "Blocking allocation didn't complete within timeout - thread may be hung";
    }
    
    if (joined) {
        EXPECT_TRUE(allocationCompleted.load());
        EXPECT_NE(threadBuffer, nullptr);
        
        staging.endBatch(vko::SemaphoreValue::makeSignalled());
        staging.wait();
    }
}

// Use-case: Precise buffer allocation near pool boundaries with alignment
// This test verifies the 2-attempt allocation strategy handles pool exhaustion correctly
// and accounts for alignment padding in m_currentPoolUsedBytes tracking. Critical for
// avoiding wasted space when pools are nearly full but have enough for small allocations.
TEST_F(UnitTestFixture, RecyclingStagingPool_PartialAllocationWithRemainder) {
    // This test verifies the 2-attempt allocation strategy handles pool exhaustion correctly
    // and accounts for alignment padding in m_currentPoolUsedBytes tracking
    
    VkDeviceSize poolSize = 1 << 16; // 64KB
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, poolSize);
    
    VkDeviceSize elementsInPool = poolSize / sizeof(uint32_t);
    
    // Query alignment requirement by creating a temporary buffer
    VkBufferCreateInfo tempBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = 1,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    vko::Buffer tempBuffer(ctx->device, tempBufferInfo);
    VkMemoryRequirements req;
    ctx->device.vkGetBufferMemoryRequirements(ctx->device, tempBuffer, &req);
    VkDeviceSize alignment = req.alignment;
    
    // Leave enough space for 1 element PLUS alignment padding to ensure
    // after align_up() there's still space for buffer2
    VkDeviceSize bytesToLeave = sizeof(uint32_t) + alignment;
    VkDeviceSize almostFullSize = (poolSize - bytesToLeave) / sizeof(uint32_t);
    
    // Step 1: Allocate almost the entire pool
    auto* buffer1 = staging.allocateUpTo<uint32_t>(almostFullSize, [](bool) {});
    ASSERT_NE(buffer1, nullptr);
    EXPECT_EQ(buffer1->size(), almostFullSize);
    
    // Step 2: Allocate 1 element - should fit in remaining space
    auto* buffer2 = staging.allocateUpTo<uint32_t>(1, [](bool) {});
    ASSERT_NE(buffer2, nullptr);
    EXPECT_EQ(buffer2->size(), 1u);
    
    // Should still be on first pool only
    EXPECT_EQ(staging.capacity(), poolSize) 
        << "Should still be on first pool only";
    
    // Step 3: Request a full pool's worth - current pool is exhausted after alignment
    // Attempt 0: Current pool has insufficient space after alignment, skips
    // Attempt 1: Gets new pool and allocates from it
    auto* buffer3 = staging.allocateUpTo<uint32_t>(elementsInPool, [](bool) {});
    ASSERT_NE(buffer3, nullptr);
    
    // Should get exactly what we requested from the fresh pool
    EXPECT_EQ(buffer3->size(), elementsInPool) 
        << "Should get full allocation from fresh pool";
    
    // Should have allocated a second pool
    EXPECT_EQ(staging.capacity(), 2 * poolSize) 
        << "Should have exactly 2 pools";
    
    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
    
    EXPECT_EQ(staging.size(), 0u);
}

// Additional test ideas for future coverage:
// - RecyclingStagingPool_ZeroSizeAllocation: Test edge case of size=0
// - RecyclingStagingPool_ActualDataTransfer: Test actual upload/download with command buffers
// - RecyclingStagingPool_AllocationFailureRecovery: Test recovery when VMA throws (pool exhaustion)

// Use-case: Uploading procedurally generated data without pre-buffering on CPU
// Tests upload() with a callback that fills each staging chunk as it's allocated,
// avoiding the need to hold entire datasets in CPU memory before transfer.
TEST_F(UnitTestFixture, StreamingStaging_UploadChunked) {
    // Setup: Create queue and staging allocator
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16); // 64KB pools
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create GPU buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 10000, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload with chunked fill - callback provides offset for each chunk
    streaming.upload(gpuBuffer, 0, 10000, 
        [](VkDeviceSize offset, auto span) {
            std::iota(span.begin(), span.end(), static_cast<int>(offset));
        });
    
    streaming.submit(); // Ensure upload completes
    ctx->device.vkQueueWaitIdle(queue);
    
    // TODO: Verify by downloading once download is implemented
    // For now, just verify it doesn't crash
}

// Use-case: Uploading large assets (textures, meshes) that exceed staging pool size
// Tests automatic chunking when transfer size > pool size. StreamingStaging should
// transparently handle partial allocations and issue multiple copy commands.
TEST_F(UnitTestFixture, StreamingStaging_UploadLarge) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create buffer larger than single pool to force chunking
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, 10000,  // 40KB worth of floats, larger than 16KB pool
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload should handle multiple chunks automatically
    streaming.upload(gpuBuffer, 0, 10000,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i) * 2.0f;
            }
        });
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
}

// Use-case: GPU→CPU data processing without storage (e.g., checksums, statistics)
// Tests downloadVisit() which processes data chunks without copying to a vector,
// ideal for streaming analytics where only aggregate results are needed.
TEST_F(UnitTestFixture, StreamingStaging_DownloadVoid) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Setup buffer with known data
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    streaming.upload(gpuBuffer, 0, 1000,
        [](VkDeviceSize offset, auto span) {
            std::iota(span.begin(), span.end(), static_cast<int>(offset));
        });
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Download with void callback - just accumulate stats, no storage
    int sum = 0;
    int count = 0;
    
    auto handle = streaming.downloadVisit(gpuBuffer, 0, 1000,
        [&sum, &count](VkDeviceSize, auto mapped) {
            // Process chunk without storing
            for (auto val : mapped) {
                sum += val;
                count++;
            }
        });
    
    streaming.submit();
    handle.wait(ctx->device);  // Wait and process all chunks
    
    // Verify we processed everything
    EXPECT_EQ(count, 1000);
    EXPECT_EQ(sum, 1000 * 999 / 2);  // Sum of 0..999
}

// Use-case: GPU readback with optional transformation (e.g., format conversion, filtering)
// Tests downloadTransform() which collects data into a vector with per-chunk processing.
// Useful for reading back GPU results (render targets, compute output) to CPU.
TEST_F(UnitTestFixture, StreamingStaging_DownloadWithTransform) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small pools to force chunking
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Upload known data
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, 5000,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    streaming.upload(gpuBuffer, 0, 5000,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i) * 2.0f;
            }
        });
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Download with transform - returns transformed data
    auto downloadFuture =
        streaming.downloadTransform<float>(gpuBuffer, 0, 5000, [](VkDeviceSize, auto input, auto output) {
            std::ranges::copy(input, output.begin()); // Identity transform
        });

    streaming.submit();
    
    // Wait and get result
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 5000);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i) * 2.0f);
    }
}

// Use-case: Simple GPU readback without transformation (most common case)
// Tests download() convenience wrapper which provides identity transform automatically.
// This is the simplest API for reading back GPU data unchanged to CPU.
TEST_F(UnitTestFixture, StreamingStaging_DownloadSimple) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Upload known data
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 2000,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    streaming.upload(gpuBuffer, 0, 2000,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<int>(offset + i) * 10;
            }
        });
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Use simple download() - no transform lambda needed
    auto downloadFuture = streaming.download(gpuBuffer, 0, 2000);
    streaming.submit();
    
    // Wait and get result
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 2000);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i) * 10);
    }
}

// Use-case: Efficient batch updates (e.g., multiple small uniform buffers per frame)
// Tests that multiple small transfers can be allocated from the same pool and submitted
// together in one command buffer, minimizing GPU synchronization overhead.
TEST_F(UnitTestFixture, StreamingStaging_MultipleBatchedTransfers) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16); // 64KB pools
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create multiple small buffers
    std::vector<vko::BoundBuffer<int>> buffers;
    for (int i = 0; i < 5; ++i) {
        buffers.emplace_back(
            ctx->device, 100,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ctx->allocator);
    }
    
    // Upload to all buffers - should all fit in one pool and batch together
    for (size_t i = 0; i < buffers.size(); ++i) {
        streaming.upload(buffers[i], 0, 100,
            [i](VkDeviceSize, auto span) {
                std::fill(span.begin(), span.end(), static_cast<int>(i * 1000));
            });
    }
    
    // Single submit should handle all transfers batched together
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Success - multiple small transfers were batched
    // (Download verification is tested separately in other tests)
}

// Use-case: Streaming huge assets (e.g., high-res textures, large models) with limited staging
// Tests that transfers much larger than total pool capacity automatically cycle pools:
// allocate → submit → wait → recycle repeatedly until complete, without manual intervention.
TEST_F(UnitTestFixture, StreamingStaging_GiantTransferImplicitCycling) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools, 48KB total
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create buffer much larger than total pool capacity (400KB >> 48KB)
    size_t largeSize = 100000; // 400KB of floats
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, largeSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload should automatically cycle pools, submit, and wait as needed
    streaming.upload(gpuBuffer, 0, largeSize,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i);
            }
        });
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Verify with download
    float sum = 0.0f;
    size_t count = 0;
    auto handle = streaming.downloadVisit(gpuBuffer, 0, largeSize,
        [&sum, &count](VkDeviceSize, auto mapped) {
            for (auto val : mapped) {
                sum += val;
                count++;
            }
        });
    
    streaming.submit();
    handle.wait(ctx->device);
    
    EXPECT_EQ(count, largeSize);
    // Sum of 0..99999 = 99999 * 100000 / 2
    float expectedSum = static_cast<float>(largeSize - 1) * static_cast<float>(largeSize) / 2.0f;
    EXPECT_FLOAT_EQ(sum, expectedSum);
}

// Use-case: Asynchronous readback for debugging/profiling without stalling rendering
// Tests that download futures remain valid even after staging pools are reused for other
// work. The original download's data must be preserved until get() is called.
TEST_F(UnitTestFixture, StreamingStaging_NonBlockingDownloadWithPoolCycling) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // 16KB pools
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create and upload initial buffer
    auto buffer1 = vko::BoundBuffer<int>(
        ctx->device, 1000,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    streaming.upload(buffer1, 0, 1000,
        [](VkDeviceSize offset, auto span) {
            std::iota(span.begin(), span.end(), static_cast<int>(offset));
        });
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Start small download but DON'T wait on it yet
    auto downloadFuture = streaming.downloadTransform<int>(
        buffer1, 0, 100,
        [](VkDeviceSize, auto input, auto output) {
            std::ranges::copy(input, output.begin());
        });
    streaming.submit();
    
    // Now do lots of other work that cycles all pools multiple times
    auto buffer2 = vko::BoundBuffer<float>(
        ctx->device, 20000, // Large enough to force pool cycling
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // This should cycle pools multiple times, but shouldn't affect our download
    streaming.upload(buffer2, 0, 20000,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i);
            }
        });
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // NOW retrieve the original download - should still work
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 100);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }
}

// Use-case: Explicit control over submission timing (e.g., synchronous readback)
// Tests that users can manually submit() even for tiny transfers that wouldn't trigger
// automatic submission, and immediately get() the result for synchronous workflows.
TEST_F(UnitTestFixture, StreamingStaging_ManualSubmitTinyDownload) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StreamingStaging streaming(queue, std::move(staging));
    
    // Create tiny buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 10, // Very small
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload
    streaming.upload(gpuBuffer, 0, 10,
        [](VkDeviceSize, auto span) {
            std::iota(span.begin(), span.end(), 42);
        });
    streaming.submit(); // Manual submit
    ctx->device.vkQueueWaitIdle(queue);
    
    // Download with manual submit
    auto downloadFuture = streaming.downloadTransform<int>(
        gpuBuffer, 0, 10,
        [](VkDeviceSize, auto input, auto output) {
            std::ranges::copy(input, output.begin());
        });
    
    streaming.submit(); // Manual submit even though it's tiny
    
    // Manual get
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 10);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], 42 + static_cast<int>(i));
    }
}

// Use-case: Error detection for incomplete transfers (forgot to submit)
// Tests that accessing a download future without submitting the batch throws
// TimelineSubmitCancel, helping catch programmer errors where submit() was forgotten.
TEST_F(UnitTestFixture, StreamingStaging_CancelOnScopeExit) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    
    // Create buffer outside the scope
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload some data first
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 16);
        vko::StreamingStaging streaming(queue, std::move(staging));
        
        streaming.upload(gpuBuffer, 0, 1000,
            [](VkDeviceSize offset, auto span) {
                std::iota(span.begin(), span.end(), static_cast<int>(offset));
            });
        streaming.submit();
        ctx->device.vkQueueWaitIdle(queue);
    }
    
    // Now start a download but let streaming go out of scope WITHOUT submit
    auto downloadFuture = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/1, /*maxPools=*/2, /*poolSize=*/1 << 16);
        vko::StreamingStaging streaming(queue, std::move(staging));
        
        auto future = streaming.downloadTransform<int>(
            gpuBuffer, 0, 1000,
            [](VkDeviceSize, auto input, auto output) {
                std::ranges::copy(input, output.begin());
            });
        
        // NO submit() - streaming destructor should cancel
        return future;
    }(); // streaming goes out of scope here
    
    // Attempting to get should throw TimelineSubmitCancel
    EXPECT_THROW({
        downloadFuture.get(ctx->device);
    }, vko::TimelineSubmitCancel);
}

// Use-case: Error detection for partially submitted downloads (subtle bugs)
// Tests that even when automatic submits occur during pool cycling, the final chunk
// still needs explicit submit(). Forgetting this should throw TimelineSubmitCancel.
TEST_F(UnitTestFixture, StreamingStaging_PartialDownloadMissingFinalSubmit) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    
    // Create large buffer
    size_t largeSize = 50000; // Large enough to cause multiple auto-submits
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, largeSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload data
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Small pools
        vko::StreamingStaging streaming(queue, std::move(staging));
        
        streaming.upload(gpuBuffer, 0, largeSize,
            [](VkDeviceSize offset, auto span) {
                for (size_t i = 0; i < span.size(); ++i) {
                    span[i] = static_cast<float>(offset + i);
                }
            });
        streaming.submit();
        ctx->device.vkQueueWaitIdle(queue);
    }
    
    // Start large download that will cause automatic submits
    auto downloadFuture = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Small pools
        vko::StreamingStaging streaming(queue, std::move(staging));
        
        // This will trigger multiple automatic submits as pools cycle
        // but won't submit the final chunk
        auto future = streaming.downloadTransform<float>(
            gpuBuffer, 0, largeSize,
            [](VkDeviceSize, auto input, auto output) {
                std::ranges::copy(input, output.begin());
            });
        
        // NO final submit() - some chunks submitted automatically, but not all
        return future;
    }(); // streaming goes out of scope
    
    // Should throw because final chunk was never submitted
    EXPECT_THROW({
        downloadFuture.get(ctx->device);
    }, vko::TimelineSubmitCancel);
}

// TODO: StreamingStaging_InterleavedUploadDownload - Test alternating uploads and downloads to verify command buffer state management
// TODO: StreamingStaging_MultipleQueueSupport - Test with transfers to different queues
// TODO: StreamingStaging_AllocationFailureRecovery - Test behavior when staging allocation fails mid-transfer
// TODO: StreamingStaging_ZeroSizeTransfer - Test edge case of size=0 upload/download
// TODO: StreamingStaging_UnalignedOffsetAndSize - Test with non-aligned buffer offsets and sizes
// TODO: StreamingStaging_CommandBufferRecycling - Verify command buffers are properly recycled and not leaked
// TODO: StreamingStaging_ConcurrentDownloads - Multiple downloads in flight with different completion times
// TODO: StreamingStaging_ExceptionSafety - Verify proper cleanup when exceptions occur during transfers
// TODO: StreamingStaging_DownloadWithPartialChunkProcessing - Test download where callback processes chunks at different rates
// TODO: StreamingStaging_MemoryPressure - Test behavior under memory pressure (all pools exhausted, waiting required)
// TODO: StreamingStaging_SubrangeTransfers - Upload/download non-contiguous subranges of a buffer
// TODO: StreamingStaging_QueueFamilyTransition - Test transfers that require queue family ownership transfer
// TODO: StreamingStaging_LargeAlignment - Test with buffers requiring large alignment (e.g., 64KB for some GPUs)
// TODO: StreamingStaging_DownloadVisitVsTransformPerformance - Compare performance characteristics
// TODO: StreamingStaging_SemaphoreChaining - Test that timeline semaphores properly chain dependencies
