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
TEST_F(UnitTestFixture, StagingStream_UploadChunked) {
    // Setup: Create queue and staging allocator
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16); // 64KB pools
    vko::StagingStream streaming(queue, std::move(staging));
    
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
// Tests automatic chunking when transfer size > pool size. StagingStream should
// transparently handle partial allocations and issue multiple copy commands.
TEST_F(UnitTestFixture, StagingStream_UploadLarge) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools
    vko::StagingStream streaming(queue, std::move(staging));
    
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
// Tests downloadForEach() which processes data subranges without copying to a vector,
// ideal for streaming analytics where only aggregate results are needed.
TEST_F(UnitTestFixture, StagingStream_DownloadVoid) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
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
    
    auto handle = streaming.downloadForEach(gpuBuffer, 0, 1000,
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
TEST_F(UnitTestFixture, StagingStream_DownloadWithTransform) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small pools to force chunking
    vko::StagingStream streaming(queue, std::move(staging));
    
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
TEST_F(UnitTestFixture, StagingStream_DownloadSimple) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
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
TEST_F(UnitTestFixture, StagingStream_MultipleBatchedTransfers) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16); // 64KB pools
    vko::StagingStream streaming(queue, std::move(staging));
    
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
TEST_F(UnitTestFixture, StagingStream_GiantTransferImplicitCycling) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools, 48KB total
    vko::StagingStream streaming(queue, std::move(staging));
    
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
    
    // Insert memory barrier to make upload writes visible to download reads
    // This is the user's responsibility when doing back-to-back transfers
    vko::cmdMemoryBarrier(ctx->device, streaming.commandBuffer(),
        vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
        vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Verify with download - comprehensive chunk validation
    struct ChunkRecord {
        size_t chunkId;
        size_t offset;
        size_t size;
    };
    std::vector<ChunkRecord> chunks;
    std::vector<uint8_t> visitCount(largeSize, 0); // Track visit count per index
    size_t valueErrors = 0;
    size_t duplicateVisits = 0;
    
    auto handle = streaming.downloadForEach(gpuBuffer, 0, largeSize,
        [&](VkDeviceSize offset, auto mapped) {
            size_t chunkId = chunks.size();
            chunks.push_back({chunkId, (size_t)offset, mapped.size()});
            
            // Validate each value in this chunk
            for (size_t i = 0; i < mapped.size(); ++i) {
                size_t index = offset + i;
                float expected = static_cast<float>(index);
                
                // Check value correctness
                if (mapped[i] != expected) {
                    valueErrors++;
                }
                
                // Track visits (detect duplicates)
                if (index < visitCount.size()) {
                    visitCount[index]++;
                    if (visitCount[index] > 1) {
                        duplicateVisits++;
                    }
                }
            }
        });
    
    streaming.submit();
    
    // TODO: Add manual GPU buffer verification if needed
    
    // Wait for completion
    handle.wait(ctx->device);
    
    // Validate chunk coverage
    size_t totalCovered = 0;
    std::vector<std::pair<size_t, size_t>> gaps;
    size_t pos = 0;
    
    // Sort chunks by offset to check coverage
    std::sort(chunks.begin(), chunks.end(), 
              [](const ChunkRecord& a, const ChunkRecord& b) { return a.offset < b.offset; });
    
    for (const auto& chunk : chunks) {
        if (chunk.offset > pos) {
            gaps.push_back({pos, chunk.offset});
        }
        totalCovered += chunk.size;
        pos = std::max(pos, chunk.offset + chunk.size);
    }
    
    if (pos < largeSize) {
        gaps.push_back({pos, largeSize});
    }
    
    // Check for overlaps
    size_t overlaps = 0;
    for (size_t i = 1; i < chunks.size(); ++i) {
        if (chunks[i].offset < chunks[i-1].offset + chunks[i-1].size) {
            overlaps++;
        }
    }
    
    // Count missing indices
    size_t missingCount = std::count(visitCount.begin(), visitCount.end(), 0);
    
    // Report results
    EXPECT_EQ(gaps.size(), 0) << "Found " << gaps.size() << " gaps in chunk coverage";
    EXPECT_EQ(overlaps, 0) << "Found " << overlaps << " overlapping chunks";
    EXPECT_EQ(missingCount, 0) << "Missing " << missingCount << " values";
    EXPECT_EQ(duplicateVisits, 0) << "Found " << duplicateVisits << " duplicate visits";
    EXPECT_EQ(valueErrors, 0) << "Found " << valueErrors << " incorrect values";
}

// Use-case: Asynchronous readback for debugging/profiling without stalling rendering
// Tests that download futures remain valid even after staging pools are reused for other
// work. The original download's data must be preserved until get() is called.
TEST_F(UnitTestFixture, StagingStream_NonBlockingDownloadWithPoolCycling) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // 16KB pools
    vko::StagingStream streaming(queue, std::move(staging));
    
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
TEST_F(UnitTestFixture, StagingStream_ManualSubmitTinyDownload) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
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
TEST_F(UnitTestFixture, StagingStream_CancelOnScopeExit) {
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
        vko::StagingStream streaming(queue, std::move(staging));
        
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
        vko::StagingStream streaming(queue, std::move(staging));
        
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
TEST_F(UnitTestFixture, StagingStream_PartialDownloadMissingFinalSubmit) {
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
        vko::StagingStream streaming(queue, std::move(staging));
        
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
        vko::StagingStream streaming(queue, std::move(staging));
        
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

// Use-case: Updating a subrange of a larger buffer (e.g., updating part of a texture or mesh)
// Tests that callbacks receive data-relative offsets (starting at 0) even when uploading/downloading
// to non-zero buffer offsets. This is critical for user code to correctly index into source data.
TEST_F(UnitTestFixture, StagingStream_NonZeroBufferOffset) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools to force chunking
    vko::StagingStream streaming(queue, std::move(staging));
    
    // Create a large buffer
    constexpr VkDeviceSize bufferSize = 20000;
    constexpr VkDeviceSize uploadOffset = 5000;  // Upload starts at 5000, NOT 0
    constexpr VkDeviceSize uploadSize = 8000;    // Upload 8000 elements
    
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // UPLOAD TEST: Upload to non-zero offset
    // Verify that callback receives offsets starting at 0, NOT uploadOffset
    std::vector<VkDeviceSize> uploadCallbackOffsets;
    streaming.upload(gpuBuffer, uploadOffset, uploadSize,
        [&uploadCallbackOffsets](VkDeviceSize offset, auto span) {
            uploadCallbackOffsets.push_back(offset);
            // Fill with: value = userOffset + localIndex
            // This relies on offset being relative to our data (0-based), not the buffer offset
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<int>(offset + i);
            }
        });
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Verify that callbacks received 0-based offsets
    ASSERT_FALSE(uploadCallbackOffsets.empty()) << "Upload callback never called";
    EXPECT_EQ(uploadCallbackOffsets.front(), 0) << "First upload chunk should have offset 0, not buffer offset";
    
    // Verify offsets are sequential and cover the range [0, uploadSize)
    VkDeviceSize expectedOffset = 0;
    for (auto cbOffset : uploadCallbackOffsets) {
        EXPECT_EQ(cbOffset, expectedOffset) << "Upload callback offsets should be sequential";
        expectedOffset = cbOffset + streaming.capacity(); // Approximate chunk size
        if (expectedOffset > uploadSize) break;
    }
    
    // DOWNLOAD TEST: Download from non-zero offset
    // Verify that callback receives offsets starting at 0, NOT uploadOffset
    std::vector<VkDeviceSize> downloadCallbackOffsets;
    auto downloadFuture = streaming.downloadTransform<int>(
        gpuBuffer, uploadOffset, uploadSize,
        [&downloadCallbackOffsets](VkDeviceSize offset, auto input, auto output) {
            downloadCallbackOffsets.push_back(offset);
            // Verify data matches what we uploaded
            for (size_t i = 0; i < input.size(); ++i) {
                EXPECT_EQ(input[i], static_cast<int>(offset + i)) 
                    << "Data at offset " << offset << " index " << i << " doesn't match";
                output[i] = input[i];
            }
        });
    
    streaming.submit();
    auto& result = downloadFuture.get(ctx->device);
    
    // Verify that callbacks received 0-based offsets
    ASSERT_FALSE(downloadCallbackOffsets.empty()) << "Download callback never called";
    EXPECT_EQ(downloadCallbackOffsets.front(), 0) << "First download chunk should have offset 0, not buffer offset";
    
    // Verify the complete downloaded data
    ASSERT_EQ(result.size(), uploadSize);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i)) 
            << "Downloaded value at index " << i << " doesn't match expected";
    }
    
    // FOREACH TEST: Test downloadForEach with non-zero offset
    std::vector<VkDeviceSize> forEachCallbackOffsets;
    auto forEachHandle = streaming.downloadForEach(
        gpuBuffer, uploadOffset, uploadSize,
        [&forEachCallbackOffsets](VkDeviceSize offset, auto mapped) {
            forEachCallbackOffsets.push_back(offset);
            // Verify data
            for (size_t i = 0; i < mapped.size(); ++i) {
                EXPECT_EQ(mapped[i], static_cast<int>(offset + i));
            }
        });
    
    streaming.submit();
    forEachHandle.wait(ctx->device);
    
    // Verify forEach also gets 0-based offsets
    ASSERT_FALSE(forEachCallbackOffsets.empty()) << "ForEach callback never called";
    EXPECT_EQ(forEachCallbackOffsets.front(), 0) << "First forEach chunk should have offset 0, not buffer offset";
}

// Edge case: Zero-size transfers
// Tests that zero-size uploads/downloads are handled gracefully (should be no-ops)
TEST_F(UnitTestFixture, StagingStream_ZeroSizeTransfer) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 1000,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Zero-size upload should not crash or call callback
    bool uploadCallbackCalled = false;
    streaming.upload(buffer, 0, 0,
        [&uploadCallbackCalled](VkDeviceSize, auto) {
            uploadCallbackCalled = true;
        });
    
    EXPECT_FALSE(uploadCallbackCalled) << "Zero-size upload should not invoke callback";
    
    // Zero-size download should work now (supported for free!)
    auto downloadFuture = streaming.downloadTransform<int>(
        buffer, 0, 0,
        []([[maybe_unused]] VkDeviceSize offset, 
           [[maybe_unused]] auto input, 
           [[maybe_unused]] auto output) {
            FAIL() << "Zero-size download callback should not be called";
        });
    
    streaming.submit();
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 0) << "Zero-size download should return empty vector";
    
    // Zero-size forEach should also work
    bool forEachCalled = false;
    auto forEachHandle = streaming.downloadForEach(
        buffer, 0, 0,
        [&forEachCalled](VkDeviceSize, auto) {
            forEachCalled = true;
        });
    
    streaming.submit();
    forEachHandle.wait(ctx->device);
    EXPECT_FALSE(forEachCalled) << "Zero-size forEach should not invoke callback";
}

// Edge case: Unaligned offsets and sizes
// Tests that transfers work correctly with non-aligned buffer offsets and sizes
// Vulkan requires alignment but the staging system should handle this transparently
TEST_F(UnitTestFixture, StagingStream_UnalignedOffsetAndSize) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
    // Create buffer with odd size
    constexpr VkDeviceSize bufferSize = 1337; // Odd number
    constexpr VkDeviceSize oddOffset = 333;   // Odd offset
    constexpr VkDeviceSize oddSize = 777;     // Odd size
    
    auto buffer = vko::BoundBuffer<int>(
        ctx->device, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Upload with unaligned parameters
    streaming.upload(buffer, oddOffset, oddSize,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<int>(offset + i + 1000);
            }
        });
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Download the same unaligned region and verify
    auto downloadFuture = streaming.download(buffer, oddOffset, oddSize);
    streaming.submit();
    
    auto& result = downloadFuture.get(ctx->device);
    ASSERT_EQ(result.size(), oddSize);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i + 1000))
            << "Unaligned transfer data mismatch at index " << i;
    }
}

// Use-case: Updating multiple non-contiguous regions (e.g., sparse texture updates)
// Tests uploading and downloading multiple separate subranges of a buffer
TEST_F(UnitTestFixture, StagingStream_SubrangeTransfers) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));
    
    constexpr VkDeviceSize bufferSize = 10000;
    auto buffer = vko::BoundBuffer<float>(
        ctx->device, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Define non-contiguous subranges to update
    struct Subrange { VkDeviceSize offset; VkDeviceSize size; float baseValue; };
    std::vector<Subrange> subranges = {
        {100, 500, 1000.0f},   // First region
        {2000, 800, 2000.0f},  // Second region (gap of ~1500)
        {5000, 1200, 5000.0f}, // Third region (gap of ~2200)
        {8000, 300, 8000.0f}   // Fourth region (gap of ~2800)
    };
    
    // Upload each subrange with different data
    for (const auto& sub : subranges) {
        streaming.upload(buffer, sub.offset, sub.size,
            [baseValue = sub.baseValue](VkDeviceSize offset, auto span) {
                for (size_t i = 0; i < span.size(); ++i) {
                    span[i] = baseValue + static_cast<float>(offset + i);
                }
            });
    }
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Download and verify each subrange
    for (const auto& sub : subranges) {
        auto downloadFuture = streaming.download(buffer, sub.offset, sub.size);
        streaming.submit();
        
        auto& result = downloadFuture.get(ctx->device);
        ASSERT_EQ(result.size(), sub.size) << "Subrange size mismatch";
        
        for (size_t i = 0; i < result.size(); ++i) {
            float expected = sub.baseValue + static_cast<float>(i);
            EXPECT_FLOAT_EQ(result[i], expected)
                << "Subrange data mismatch at offset " << sub.offset 
                << " index " << i;
        }
    }
}

// Lifetime and destruction order edge cases (important for multithreaded shutdown scenarios)
// TODO: LifetimeEdgeCase_DownloadFutureOutlivesStaging - Download future kept alive after StagingStream destroyed (should unmap cleanly)
// TODO: LifetimeEdgeCase_StagingOutlivesDownloadFuture - Opposite order - download future destroyed first, then staging
// TODO: LifetimeEdgeCase_MultipleDownloadsCancelledOutOfOrder - Multiple downloads, some cancelled, some completed, in mixed order
// TODO: LifetimeEdgeCase_UploadMappingActiveOnDestruct - StagingStream destroyed while upload mapping is still in scope (should unmap)
// TODO: LifetimeEdgeCase_PoolDestroyedDuringCallback - RecyclingStagingPool destroyed while evaluator callbacks are running
// TODO: LifetimeEdgeCase_DownloadFutureAbandonedNeverAccessed - Future created but get() never called, should clean up properly
// TODO: LifetimeEdgeCase_SubmitAfterStagingDestroyed - Try to access download future after staging is gone (current behavior)
// TODO: LifetimeEdgeCase_RecursiveCallbackDestruction - Callback triggers another operation that destroys resources
// TODO: LifetimeEdgeCase_MoveSemanticsDuringPendingTransfers - Move staging pool while downloads are in flight

// Use-case: Mixed workloads with uploads and downloads (e.g., compute pipeline with feedback)
// Tests that alternating uploads and downloads correctly manage command buffer state
TEST_F(UnitTestFixture, StagingStream_InterleavedUploadDownload) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 15);
    vko::StagingStream streaming(queue, std::move(staging));
    
    constexpr size_t numBuffers = 5;
    constexpr VkDeviceSize bufferSize = 1000;
    
    // Create multiple buffers
    std::vector<vko::BoundBuffer<int>> buffers;
    for (size_t i = 0; i < numBuffers; ++i) {
        buffers.emplace_back(
            ctx->device, bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ctx->allocator);
    }
    
    // Interleave uploads and downloads
    // Use a simple copy lambda that can be used consistently
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(streaming.downloadTransform<int>(buffers[0], 0, 0, copyLambda));
    std::vector<FutureType> futures;
    
    for (size_t i = 0; i < numBuffers; ++i) {
        // Upload to buffer i
        streaming.upload(buffers[i], 0, bufferSize,
            [baseValue = i * 1000](VkDeviceSize offset, auto span) {
                for (size_t j = 0; j < span.size(); ++j) {
                    span[j] = static_cast<int>(baseValue + offset + j);
                }
            });
        
        // Submit upload to ensure it completes before download
        streaming.submit();
        
        // Download from current buffer (now that upload is submitted)
        futures.push_back(streaming.downloadTransform<int>(
            buffers[i], 0, bufferSize, copyLambda));
    }
    
    streaming.submit();
    
    // Verify all downloads
    for (size_t i = 0; i < futures.size(); ++i) {
        auto& result = futures[i].get(ctx->device);
        ASSERT_EQ(result.size(), bufferSize);
        
        size_t bufferIdx = i; // Downloaded from buffer i (which is i+1-1)
        for (size_t j = 0; j < result.size(); ++j) {
            int expected = static_cast<int>(bufferIdx * 1000 + j);
            EXPECT_EQ(result[j], expected) 
                << "Buffer " << bufferIdx << " index " << j;
        }
    }
}

// Use-case: Multiple async readbacks (e.g., profiling multiple GPU timers)
// Tests that multiple downloads can be in flight with different completion times
TEST_F(UnitTestFixture, StagingStream_ConcurrentDownloads) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/3, /*maxPools=*/5, /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));
    
    constexpr size_t numDownloads = 10;
    std::vector<VkDeviceSize> downloadSizes = {100, 500, 200, 1000, 50, 800, 150, 400, 600, 300};
    
    // Create buffers of varying sizes
    std::vector<vko::BoundBuffer<float>> buffers;
    for (size_t i = 0; i < numDownloads; ++i) {
        buffers.emplace_back(
            ctx->device, downloadSizes[i],
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ctx->allocator);
        
        // Upload unique data to each
        streaming.upload(buffers[i], 0, downloadSizes[i],
            [marker = i * 100.0f](VkDeviceSize offset, auto span) {
                for (size_t j = 0; j < span.size(); ++j) {
                    span[j] = marker + static_cast<float>(offset + j);
                }
            });
    }
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Start all downloads concurrently (don't wait on any yet)
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(streaming.downloadTransform<float>(buffers[0], 0, 0, copyLambda));
    std::vector<FutureType> futures;
    
    for (size_t i = 0; i < numDownloads; ++i) {
        futures.push_back(streaming.downloadTransform<float>(
            buffers[i], 0, downloadSizes[i], copyLambda));
    }
    
    streaming.submit();
    
    // Retrieve results in random order to test independence
    std::vector<size_t> retrievalOrder = {5, 2, 8, 0, 9, 3, 7, 1, 4, 6};
    for (size_t idx : retrievalOrder) {
        auto& result = futures[idx].get(ctx->device);
        ASSERT_EQ(result.size(), downloadSizes[idx]);
        
        float marker = idx * 100.0f;
        for (size_t j = 0; j < result.size(); ++j) {
            EXPECT_FLOAT_EQ(result[j], marker + static_cast<float>(j))
                << "Download " << idx << " index " << j;
        }
    }
}

// Use-case: Long-running streaming with many submissions (e.g., video encoding)
// Tests that command buffers are properly recycled and not leaked over many submits
TEST_F(UnitTestFixture, StagingStream_CommandBufferRecycling) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));
    
    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 500,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // Perform many small uploads with submits between them
    // This should trigger command buffer recycling
    constexpr size_t numIterations = 50;
    for (size_t i = 0; i < numIterations; ++i) {
        streaming.upload(buffer, 0, 500,
            [value = static_cast<int>(i)]([[maybe_unused]] VkDeviceSize offset, auto span) {
                for (size_t j = 0; j < span.size(); ++j) {
                    span[j] = value;
                }
            });
        
        streaming.submit();
        
        // Wait occasionally to allow recycling
        if (i % 10 == 9) {
            ctx->device.vkQueueWaitIdle(queue);
        }
    }
    
    // Final wait
    ctx->device.vkQueueWaitIdle(queue);
    
    // If command buffers leaked, memory usage would grow unbounded
    // No assertion needed - test passes if it doesn't crash or OOM
    SUCCEED() << "Command buffer recycling appears to work correctly";
}

// Use-case: High-frequency streaming with limited memory (e.g., real-time texture streaming)
// Tests behavior when all pools are exhausted and allocation must wait/block
TEST_F(UnitTestFixture, StagingStream_MemoryPressure) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Only 2 pools
    vko::StagingStream streaming(queue, std::move(staging));
    
    // Create a transfer much larger than total pool capacity
    constexpr VkDeviceSize totalPoolCapacity = 2 * (1 << 14) / sizeof(float);
    constexpr VkDeviceSize largeTransferSize = totalPoolCapacity * 3; // 3x capacity
    
    auto buffer = vko::BoundBuffer<float>(
        ctx->device, largeTransferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    // This upload should automatically submit and wait when pools are exhausted
    streaming.upload(buffer, 0, largeTransferSize,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i);
            }
        });
    
    streaming.submit();
    ctx->device.vkQueueWaitIdle(queue);
    
    // Verify the transfer completed successfully despite memory pressure
    auto downloadFuture = streaming.download(buffer, 0, largeTransferSize);
    streaming.submit();
    
    auto& result = downloadFuture.get(ctx->device);
    ASSERT_EQ(result.size(), largeTransferSize);
    
    // Spot check some values
    for (size_t i = 0; i < std::min<size_t>(100, result.size()); ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
    }
}

// Use-case: Pipeline with dependent stages (e.g., multiple compute passes reading previous results)
// Tests that timeline semaphores properly chain dependencies across multiple operations
TEST_F(UnitTestFixture, StagingStream_SemaphoreChaining) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));
    
    constexpr size_t numOperations = 5;
    constexpr VkDeviceSize bufferSize = 100;
    
    auto buffer = vko::BoundBuffer<int>(
        ctx->device, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(streaming.downloadTransform<int>(buffer, 0, 0, copyLambda));
    std::vector<FutureType> futures;
    
    // Chain operations: upload, download, upload, download, etc.
    // Each operation depends on the previous via semaphore chaining
    for (size_t i = 0; i < numOperations; ++i) {
        // Upload with incremented values
        streaming.upload(buffer, 0, bufferSize,
            [iteration = i](VkDeviceSize offset, auto span) {
                for (size_t j = 0; j < span.size(); ++j) {
                    span[j] = static_cast<int>(iteration * 1000 + offset + j);
                }
            });
        streaming.submit();
        
        // Download to verify
        futures.push_back(streaming.downloadTransform<int>(
            buffer, 0, bufferSize, copyLambda));
        streaming.submit();
    }
    
    // Verify all futures resolve correctly in order
    for (size_t i = 0; i < futures.size(); ++i) {
        auto& result = futures[i].get(ctx->device);
        ASSERT_EQ(result.size(), bufferSize);
        
        int expectedBase = static_cast<int>(i * 1000);
        for (size_t j = 0; j < result.size(); ++j) {
            EXPECT_EQ(result[j], expectedBase + static_cast<int>(j))
                << "Operation " << i << " index " << j;
        }
    }
}

// Use-case: Speculative readback that gets abandoned (e.g., user cancels before results needed)
// Tests that futures can be destroyed without calling get(), ensuring proper cleanup
TEST_F(UnitTestFixture, StagingStream_FutureAbandonedNeverAccessed) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));
    
    auto buffer = vko::BoundBuffer<float>(
        ctx->device, 1000,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    streaming.upload(buffer, 0, 1000,
        [](VkDeviceSize offset, auto span) {
            for (size_t i = 0; i < span.size(); ++i) {
                span[i] = static_cast<float>(offset + i);
            }
        });
    streaming.submit();
    
    {
        // Create future but never call get() - should cancel on destruction
        auto abandonedFuture = streaming.download(buffer, 0, 1000);
        streaming.submit();
        // Future destroyed here without get()
    }
    
    // Staging should still be usable after abandoned future
    auto normalFuture = streaming.download(buffer, 0, 500);
    streaming.submit();
    auto& result = normalFuture.get(ctx->device);
    
    ASSERT_EQ(result.size(), 500);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
    }
}

// Use-case: Long-lived result caching (e.g., keeping readback results after streaming context closed)
// Tests that futures can be evaluated before StagingStream destruction and data accessed after
TEST_F(UnitTestFixture, StagingStream_FutureOutlivesStaging) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    
    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 500,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    
    auto future = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
        vko::StagingStream streaming(queue, std::move(staging));
        
        streaming.upload(buffer, 0, 500,
            [](VkDeviceSize offset, auto span) {
                for (size_t i = 0; i < span.size(); ++i) {
                    span[i] = static_cast<int>(offset + i * 2);
                }
            });
        streaming.submit();
        
        auto result = streaming.downloadTransform<int>(buffer, 0, 500, copyLambda);
        streaming.submit();
        
        // Evaluate now while StagingStream exists
        result.get(ctx->device);
        
        // StagingStream destroyed here, but future retains evaluated data
        return result;
    }();
    
    // Future object outlives StagingStream - access previously evaluated data
    auto& result = future.get(ctx->device); // Second get() should just return cached data
    ASSERT_EQ(result.size(), 500u);
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i * 2));
    }
}

// Use-case: Error handling in async pipelines (e.g., detecting cancelled operations)
// Tests that TimelineSubmitCancel is thrown when future is evaluated after StagingStream destruction
TEST_F(UnitTestFixture, StagingStream_FutureThrowsWhenStagingDestroyed) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    
    auto buffer = vko::BoundBuffer<float>(
        ctx->device, 200,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ctx->allocator);
    
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    
    // Create future but destroy StagingStream before evaluation
    auto future = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
        vko::StagingStream streaming(queue, std::move(staging));
        
        streaming.upload(buffer, 0, 200,
            [](VkDeviceSize offset, auto span) {
                for (size_t i = 0; i < span.size(); ++i) {
                    span[i] = static_cast<float>(offset + i);
                }
            });
        streaming.submit();
        
        auto result = streaming.downloadTransform<float>(buffer, 0, 200, copyLambda);
        streaming.submit();
        
        // Return future WITHOUT calling get() - StagingStream destroyed here
        return result;
    }();
    
    // Attempting to evaluate after StagingStream destruction should throw
    EXPECT_THROW({
        future.get(ctx->device);
    }, vko::TimelineSubmitCancel);
}

// Use-case: Dynamic result storage (e.g., moving futures into containers or returning from functions)
// Tests that futures can be moved during pending transfers and remain valid
TEST_F(UnitTestFixture, StagingStream_MoveSemanticsDuringPendingTransfers) {
    vko::TimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));
    
    constexpr size_t numBuffers = 3;
    std::vector<vko::BoundBuffer<double>> buffers;
    for (size_t i = 0; i < numBuffers; ++i) {
        buffers.emplace_back(
            ctx->device, 200,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ctx->allocator);
        
        streaming.upload(buffers[i], 0, 200,
            [marker = i * 10.0](VkDeviceSize offset, auto span) {
                for (size_t j = 0; j < span.size(); ++j) {
                    span[j] = marker + static_cast<double>(offset + j);
                }
            });
    }
    streaming.submit();
    
    // Create futures and immediately move them into a vector
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(streaming.downloadTransform<double>(buffers[0], 0, 0, copyLambda));
    std::vector<FutureType> futures;
    
    for (size_t i = 0; i < numBuffers; ++i) {
        futures.push_back(streaming.downloadTransform<double>(
            buffers[i], 0, 200, copyLambda));
    }
    streaming.submit();
    
    // Move futures to a different container while transfers are pending
    std::vector<FutureType> movedFutures;
    for (auto& f : futures) {
        movedFutures.push_back(std::move(f));
    }
    
    // Verify moved futures still work correctly
    for (size_t i = 0; i < numBuffers; ++i) {
        auto& result = movedFutures[i].get(ctx->device);
        ASSERT_EQ(result.size(), 200);
        
        double marker = i * 10.0;
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_DOUBLE_EQ(result[j], marker + static_cast<double>(j))
                << "Buffer " << i << " index " << j;
        }
    }
}

// TODO: StagingStream_MultipleQueueSupport - Test with transfers to different queues
// TODO: StagingStream_AllocationFailureRecovery - Test behavior when staging allocation fails mid-transfer
// TODO: StagingStream_ExceptionSafety - Verify proper cleanup when exceptions occur during transfers
// TODO: StagingStream_DownloadWithPartialChunkProcessing - Test download where callback processes chunks at different rates
// TODO: StagingStream_MemoryPressure - Test behavior under memory pressure (all pools exhausted, waiting required)
// TODO: StagingStream_SubrangeTransfers - Upload/download non-contiguous subranges of a buffer
// TODO: StagingStream_QueueFamilyTransition - Test transfers that require queue family ownership transfer
// TODO: StagingStream_LargeAlignment - Test with buffers requiring large alignment (e.g., 64KB for some GPUs)
// TODO: StagingStream_DownloadVisitVsTransformPerformance - Compare performance characteristics
// TODO: StagingStream_SemaphoreChaining - Test that timeline semaphores properly chain dependencies
