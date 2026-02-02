// Copyright (c) 2025 Pyarelal Knowles, MIT License

// TODO: switch to TimelineQueue
#define VULKAN_OBJECTS_ENABLE_FOOTGUNS

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>
#include <test_context_fixtures.hpp>
#include <thread>
#include <vko/allocator.hpp>
#include <vko/staging_memory.hpp>
#include <vko/timeline_queue.hpp>

// Use-case: Basic staging pool initialization
// Verifies that the pool pre-allocates the minimum number of memory pools upfront,
// reducing allocation overhead during the first transfers.
TEST_F(UnitTestFixture, RecyclingStagingPool_BasicConstruction) {
    size_t                                      minPools = 3;
    VkDeviceSize                                poolSize = 1 << 24; // 16MB
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator, minPools,
                                                        /*maxPools=*/5, poolSize);

    // Should have minimum pools pre-allocated
    EXPECT_EQ(staging.capacity(), minPools * poolSize);
    EXPECT_EQ(staging.size(), 0u); // No buffers allocated yet
}

// Use-case: Single small CPU→GPU transfer with synchronization
// Tests the complete lifecycle: allocate staging buffer, write data, end batch with
// timeline semaphore, and verify the destruction callback fires when signaled.
TEST_F(UnitTestFixture, RecyclingStagingPool_SimpleAllocation) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                        /*minPools=*/2, /*maxPools=*/5,
                                                        /*poolSize=*/1 << 20); // 1MB pools

    bool  callbackInvoked = false;
    auto* buffer          = staging.allocateUpTo<uint32_t>(100, [&callbackInvoked](bool signaled) {
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
                                                        /*minPools=*/1, /*maxPools=*/2,
                                                        /*poolSize=*/1 << 16); // 64KB pools

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
                                                        /*minPools=*/1, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16); // 64KB pools

    // Request more than pool size - should get partial allocation
    size_t hugeSize = (1 << 20) / sizeof(uint32_t); // 1MB worth of uint32_t
    auto*  buffer   = staging.allocateUpTo<uint32_t>(hugeSize, [](bool) {});

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
                                                        /*minPools=*/1, /*maxPools=*/5,
                                                        /*poolSize=*/1 << 14); // 16KB pools

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
                                                        /*minPools=*/1, /*maxPools=*/2,
                                                        /*poolSize=*/1 << 12); // 4KB pools

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
                                                        /*minPools=*/1, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16);

    bool populateCalled = false;
    bool destructCalled = false;

    bool success = staging.tryWith<float>(
        100,
        [&populateCalled](const vko::BoundBuffer<float, vko::vma::Allocator>& buffer) {
            populateCalled = true;
            // Verify we can write to the buffer
            auto mapping = buffer.map();
            mapping[0]   = 3.14f;
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
    bool callback1Called   = false;
    bool callback1Signaled = true; // Default to true to detect if it's set
    bool callback2Called   = false;
    bool callback2Signaled = true;

    {
        vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                            /*minPools=*/1, /*maxPools=*/3,
                                                            /*poolSize=*/1 << 16);

        // Allocate but don't end batch - should get false on destruct (unsubmitted work)
        auto* buffer1 = staging.allocateUpTo<uint32_t>(100, [&](bool signaled) {
            callback1Called   = true;
            callback1Signaled = signaled;
        });
        ASSERT_NE(buffer1, nullptr);

        // Allocate another in the same unsubmitted batch
        auto* buffer2 = staging.allocateUpTo<uint32_t>(100, [&](bool signaled) {
            callback2Called   = true;
            callback2Signaled = signaled;
        });
        ASSERT_NE(buffer2, nullptr);

        // Destructor called here without calling endBatch()
        // Both callbacks should be called with false (work was never submitted)
    }

    EXPECT_TRUE(callback1Called);
    EXPECT_FALSE(callback1Signaled) << "Callback should receive false for unsubmitted work";
    EXPECT_TRUE(callback2Called);
    EXPECT_FALSE(callback2Signaled) << "Callback should receive false for unsubmitted work";
}

// Use-case: Overlapping transfers for maximum GPU utilization
// Tests multiple batches in flight simultaneously, each with independent semaphores.
// Callbacks should fire independently as each batch completes, not blocking each other.
TEST_F(UnitTestFixture, RecyclingStagingPool_MultipleBatchesInFlight) {
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                        /*minPools=*/2, /*maxPools=*/4,
                                                        /*poolSize=*/1 << 16);

    int batch1Callbacks = 0;
    int batch2Callbacks = 0;
    int batch3Callbacks = 0;

    // Batch 1
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch1Callbacks++;
    });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch1Callbacks++;
    });
    auto sem1 = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(sem1);

    // Batch 2
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch2Callbacks++;
    });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch2Callbacks++;
    });
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch2Callbacks++;
    });
    auto sem2 = vko::SemaphoreValue::makeSignalled();
    staging.endBatch(sem2);

    // Batch 3
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            batch3Callbacks++;
    });
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
                                                        /*minPools=*/1, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16);

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
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext     = nullptr,
        .semaphore = sem,
        .value     = 1,
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
                                                        /*minPools=*/3, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16);

    int callbackCount = 0;

    // Allocate a buffer with callback from first pool
    staging.allocateUpTo<uint32_t>(100, [&](bool s) {
        if (s)
            callbackCount++;
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
    // Enable device addresses to verify GPU-side alignment
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                        /*minPools=*/1, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16, usage);

    // Allocate several buffers and verify they're properly aligned
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;

    for (int i = 0; i < 10; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(100, [](bool) {});
        if (buffer) {
            buffers.push_back(buffer);

            // Get the buffer's device address to check alignment
            VkBufferDeviceAddressInfo info{
                .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext  = nullptr,
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
                                                        /*minPools=*/2, /*maxPools=*/5,
                                                        /*poolSize=*/1 << 16);

    VkDeviceSize initialCapacity = staging.capacity();

    // Allocate buffers using multiple pools
    for (int i = 0; i < 20; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(1000, [](bool) {});
        if (!buffer)
            break;
    }

    VkDeviceSize capacityAfterAlloc = staging.capacity();
    VkDeviceSize sizeAfterAlloc     = staging.size();
    EXPECT_GE(capacityAfterAlloc, initialCapacity);
    EXPECT_GT(sizeAfterAlloc, 0u);

    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();

    // After wait: buffers freed (size=0) but pools remain (capacity unchanged)
    EXPECT_EQ(staging.size(), 0u);
    EXPECT_EQ(staging.capacity(), capacityAfterAlloc); // Capacity stays the same
    EXPECT_GE(staging.capacity(), initialCapacity);    // At least minPools remain
}

// Use-case: Transferring ownership of staging resources (e.g., returning from factory)
// Verifies that RecyclingStagingPool can be safely moved, preserving all state including
// pending batches and callbacks, enabling flexible resource management patterns.
TEST_F(UnitTestFixture, RecyclingStagingPool_MoveSemantics) {
    bool callback1Called = false;
    bool callback2Called = false;

    vko::vma::RecyclingStagingPool<vko::Device> staging1(ctx->device, ctx->allocator,
                                                         /*minPools=*/1, /*maxPools=*/3,
                                                         /*poolSize=*/1 << 16);

    staging1.allocateUpTo<uint32_t>(100, [&](bool s) { callback1Called = s; });
    VkDeviceSize size1 = staging1.size();
    EXPECT_GT(size1, 0u);

    // Move construct
    vko::vma::RecyclingStagingPool<vko::Device> staging2(std::move(staging1));
    EXPECT_EQ(staging2.size(), size1);

    staging2.allocateUpTo<uint32_t>(100, [&](bool s) { callback2Called = s; });

    // Move assign
    vko::vma::RecyclingStagingPool<vko::Device> staging3(ctx->device, ctx->allocator,
                                                         /*minPools=*/1, /*maxPools=*/3,
                                                         /*poolSize=*/1 << 16);
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
                                                        /*minPools=*/2, /*maxPools=*/4,
                                                        /*poolSize=*/1 << 16);

    VkDeviceSize initialSize     = staging.size();
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
                                                        /*minPools=*/1, /*maxPools=*/2,
                                                        /*poolSize=*/1 << 12); // 4KB pools

    // Create timeline semaphore that starts unsignaled
    vko::TimelineSemaphore sem(ctx->device, 0);

    // Fill both pools completely
    std::vector<vko::BoundBuffer<uint32_t, vko::vma::Allocator>*> buffers;
    for (int i = 0; i < 100; ++i) {
        auto* buffer = staging.allocateUpTo<uint32_t>(900, [](bool) {}); // ~3.6KB each
        if (!buffer)
            break;
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
    std::atomic<bool>                                allocationStarted{false};
    std::atomic<bool>                                allocationCompleted{false};
    vko::BoundBuffer<uint32_t, vko::vma::Allocator>* threadBuffer = nullptr;

    std::thread allocThread([&]() {
        allocationStarted   = true;
        threadBuffer        = staging.allocateUpTo<uint32_t>(100, [](bool) {});
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
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext     = nullptr,
        .semaphore = sem,
        .value     = 1,
    };
    ctx->device.vkSignalSemaphore(ctx->device, &signalInfo);

    // Wait for allocation to complete with timeout
    bool joined = false;
    startTime   = std::chrono::steady_clock::now();
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

    VkDeviceSize                                poolSize = 1 << 16; // 64KB
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                        /*minPools=*/1, /*maxPools=*/3, poolSize);

    VkDeviceSize elementsInPool = poolSize / sizeof(uint32_t);

    // Query alignment requirement by creating a temporary buffer
    VkBufferCreateInfo tempBufferInfo = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0,
        .size        = 1,
        .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    vko::Buffer          tempBuffer(ctx->device, tempBufferInfo);
    VkMemoryRequirements req;
    ctx->device.vkGetBufferMemoryRequirements(ctx->device, tempBuffer, &req);
    VkDeviceSize alignment = req.alignment;

    // Leave enough space for 1 element PLUS alignment padding to ensure
    // after align_up() there's still space for buffer2
    VkDeviceSize bytesToLeave   = sizeof(uint32_t) + alignment;
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
    EXPECT_EQ(staging.capacity(), poolSize) << "Should still be on first pool only";

    // Step 3: Request a full pool's worth - current pool is exhausted after alignment
    // Attempt 0: Current pool has insufficient space after alignment, skips
    // Attempt 1: Gets new pool and allocates from it
    auto* buffer3 = staging.allocateUpTo<uint32_t>(elementsInPool, [](bool) {});
    ASSERT_NE(buffer3, nullptr);

    // Should get exactly what we requested from the fresh pool
    EXPECT_EQ(buffer3->size(), elementsInPool) << "Should get full allocation from fresh pool";

    // Should have allocated a second pool
    EXPECT_EQ(staging.capacity(), 2 * poolSize) << "Should have exactly 2 pools";

    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();

    EXPECT_EQ(staging.size(), 0u);
}

// Use-case: Buffers with unusual alignment requirements
// Tests staging transfers with large alignment constraints (e.g., for specific hardware)
// Use-case: Buffers with alignment-induced padding
// Tests that staging buffer alignment causes pool padding and growth as expected
TEST_F(UnitTestFixture, RecyclingStagingPool_AlignmentPadding) {
    // 1. Query the alignment that RecyclingStagingPool will use
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBufferCreateInfo tempBufferInfo = {
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .size                  = 1,
        .usage                 = usage,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    vko::Buffer          tempBuffer(ctx->device, tempBufferInfo);
    VkMemoryRequirements req;
    ctx->device.vkGetBufferMemoryRequirements(ctx->device, tempBuffer, &req);
    VkDeviceSize alignment = req.alignment;

    // 2. Initialize pool with size that will test alignment padding
    VkDeviceSize poolSize = 1024 + std::max(VkDeviceSize(2), alignment);

    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/1, /*maxPools=*/2,
                                                               poolSize, usage);

    VkDeviceSize initialCapacity = staging.capacity();
    EXPECT_EQ(initialCapacity, poolSize);

    // 3. First allocation of half pool size
    VkDeviceSize                                     halfPool = poolSize / 2;
    vko::BoundBuffer<uint32_t, vko::vma::Allocator>* buf1 =
        staging.allocateUpTo<uint32_t>(halfPool);
    ASSERT_NE(buf1, nullptr);

    // 4. Check first allocation didn't create a new pool
    EXPECT_EQ(staging.capacity(), initialCapacity) << "First allocation should use initial pool";

    // 5. Second allocation of half pool size
    vko::BoundBuffer<uint32_t, vko::vma::Allocator>* buf2 =
        staging.allocateUpTo<uint32_t>(halfPool);
    ASSERT_NE(buf2, nullptr);

    // 6. Check if second pool was needed (depends on alignment)
    if (alignment == 1) {
        // No padding - both allocations fit in one pool
        EXPECT_EQ(staging.capacity(), initialCapacity)
            << "With alignment=1, no padding needed, should fit in one pool";
    } else {
        // Alignment padding should cause second pool allocation
        EXPECT_EQ(staging.capacity(), initialCapacity * 2)
            << "With alignment=" << alignment << ", padding should require second pool";
    }
}

// Use-case: Uploading procedurally generated data without pre-buffering on CPU
// Tests upload() with a callback that fills each staging chunk as it's allocated,
// avoiding the need to hold entire datasets in CPU memory before transfer.
TEST_F(UnitTestFixture, StagingStream_UploadChunked) {
    // Setup: Create queue and staging allocator
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16); // 64KB pools
    vko::StagingStream streaming(queue, std::move(staging));

    // Create GPU buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 10000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload with chunked fill - callback provides offset for each chunk
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    std::iota(span.begin(), span.end(), static_cast<int>(offset));
                });

    streaming.submit(); // Ensure upload completes
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // TODO: Verify by downloading once download is implemented
    // For now, just verify it doesn't crash
}

// Use-case: Uploading large assets (textures, meshes) that exceed staging pool size
// Tests automatic chunking when transfer size > pool size. StagingStream should
// transparently handle partial allocations and issue multiple copy commands.
TEST_F(UnitTestFixture, StagingStream_UploadLarge) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer larger than single pool to force chunking
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, 10000, // 40KB worth of floats, larger than 16KB pool
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload should handle multiple chunks automatically
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i) * 2.0f;
                    }
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: GPU→CPU data processing without storage (e.g., checksums, statistics)
// Tests downloadForEach() which processes data subranges without copying to a vector,
// ideal for streaming analytics where only aggregate results are needed.
TEST_F(UnitTestFixture, StagingStream_DownloadVoid) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Setup buffer with known data
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    std::iota(span.begin(), span.end(), static_cast<int>(offset));
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download with void callback using free function - just accumulate stats, no storage
    int sum   = 0;
    int count = 0;

    auto handle = vko::downloadForEach(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                                       [&sum, &count](VkDeviceSize, auto mapped) {
                                           // Process chunk without storing
                                           for (auto val : mapped) {
                                               sum += val;
                                               count++;
                                           }
                                       });

    streaming.submit();
    handle.wait(ctx->device); // Wait and process all chunks

    // Verify we processed everything
    EXPECT_EQ(count, 1000);
    EXPECT_EQ(sum, 1000 * 999 / 2); // Sum of 0..999
}

// Use-case: GPU readback with optional transformation (e.g., format conversion, filtering)
// Tests downloadTransform() which collects data into a vector with per-chunk processing.
// Useful for reading back GPU results (render targets, compute output) to CPU.
TEST_F(UnitTestFixture, StagingStream_DownloadWithTransform) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/1, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small pools to force chunking
    vko::StagingStream streaming(queue, std::move(staging));

    // Upload known data
    auto gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, 5000, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i) * 2.0f;
                    }
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download with transform using free function
    auto downloadFuture =
        vko::download<float>(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                             [](VkDeviceSize, auto input, auto output) {
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Upload known data
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 2000, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<int>(offset + i) * 10;
                    }
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Use simple download() free function - no transform lambda needed
    auto downloadFuture = vko::download(streaming, ctx->device, vko::BufferSpan(gpuBuffer));
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16); // 64KB pools
    vko::StagingStream streaming(queue, std::move(staging));

    // Create multiple small buffers
    std::vector<vko::BoundBuffer<int>> buffers;
    for (int i = 0; i < 5; ++i) {
        buffers.emplace_back(ctx->device, 100,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);
    }

    // Upload to all buffers - should all fit in one pool and batch together
    for (size_t i = 0; i < buffers.size(); ++i) {
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffers[i]),
                    [i](VkDeviceSize, auto span) {
                        std::fill(span.begin(), span.end(), static_cast<int>(i * 1000));
                    });
    }

    // Single submit should handle all transfers batched together
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Success - multiple small transfers were batched
    // (Download verification is tested separately in other tests)
}

// Use-case: Streaming huge assets (e.g., high-res textures, large models) with limited staging
// Tests that transfers much larger than total pool capacity automatically cycle pools:
// allocate → submit → wait → recycle repeatedly until complete, without manual intervention.
TEST_F(UnitTestFixture, StagingStream_GiantTransferImplicitCycling) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools, 48KB total
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer much larger than total pool capacity (400KB >> 48KB)
    size_t largeSize = 100000; // 400KB of floats
    auto   gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, largeSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload should automatically cycle pools, submit, and wait as needed
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i);
                    }
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Insert memory barrier to make upload writes visible to download reads
    // This is the user's responsibility when doing back-to-back transfers
    vko::cmdMemoryBarrier(
        ctx->device, streaming.commandBuffer(),
        vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
        vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});

    // Verify with download - comprehensive chunk validation
    struct ChunkRecord {
        size_t chunkId;
        size_t offset;
        size_t size;
    };
    std::vector<ChunkRecord> chunks;
    std::vector<uint8_t>     visitCount(largeSize, 0); // Track visit count per index
    size_t                   valueErrors     = 0;
    size_t                   duplicateVisits = 0;

    auto handle = vko::downloadForEach(
        streaming, ctx->device, vko::BufferSpan(gpuBuffer), [&](VkDeviceSize offset, auto mapped) {
            size_t chunkId = chunks.size();
            chunks.push_back({chunkId, (size_t)offset, mapped.size()});

            // Validate each value in this chunk
            for (size_t i = 0; i < mapped.size(); ++i) {
                size_t index    = offset + i;
                float  expected = static_cast<float>(index);

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
    size_t                                 totalCovered = 0;
    std::vector<std::pair<size_t, size_t>> gaps;
    size_t                                 pos = 0;

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
        if (chunks[i].offset < chunks[i - 1].offset + chunks[i - 1].size) {
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/2,
                                                               /*poolSize=*/1 << 14); // 16KB pools
    vko::StagingStream streaming(queue, std::move(staging));

    // Create and upload initial buffer
    auto buffer1 = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer1).subspan(0, 1000),
                [](VkDeviceSize offset, auto span) {
                    std::iota(span.begin(), span.end(), static_cast<int>(offset));
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Start small download but DON'T wait on it yet
    auto downloadFuture = vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(buffer1).subspan(0, 100),
        [](VkDeviceSize, auto input, auto output) { std::ranges::copy(input, output.begin()); });
    streaming.submit();

    // Now do lots of other work that cycles all pools multiple times
    auto buffer2 = vko::BoundBuffer<float>(ctx->device, 20000, // Large enough to force pool cycling
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // This should cycle pools multiple times, but shouldn't affect our download
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer2).subspan(0, 20000),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i);
                    }
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create tiny buffer
    auto gpuBuffer =
        vko::BoundBuffer<int>(ctx->device, 10, // Very small
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 10),
                [](VkDeviceSize, auto span) { std::iota(span.begin(), span.end(), 42); });
    streaming.submit(); // Manual submit
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download with manual submit
    auto downloadFuture = vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 10),
        [](VkDeviceSize, auto input, auto output) { std::ranges::copy(input, output.begin()); });

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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Create buffer outside the scope
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload some data first
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/1, /*maxPools=*/2,
                                                                   /*poolSize=*/1 << 16);
        vko::StagingStream streaming(queue, std::move(staging));

        vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 1000),
                    [](VkDeviceSize offset, auto span) {
                        std::iota(span.begin(), span.end(), static_cast<int>(offset));
                    });
        streaming.submit();
        vko::check(ctx->device.vkQueueWaitIdle(queue));
    }

    // Now start a download but let streaming go out of scope WITHOUT submit
    auto downloadFuture = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/1, /*maxPools=*/2,
                                                                   /*poolSize=*/1 << 16);
        vko::StagingStream streaming(queue, std::move(staging));

        auto future =
            vko::download<int>(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 1000),
                               [](VkDeviceSize, auto input, auto output) {
                                   std::ranges::copy(input, output.begin());
                               });

        // NO submit() - streaming destructor should cancel
        return future;
    }(); // streaming goes out of scope here

    // Attempting to get should throw TimelineSubmitCancel
    EXPECT_THROW({ downloadFuture.get(ctx->device); }, vko::TimelineSubmitCancel);
}

// Use-case: Error detection for partially submitted downloads (subtle bugs)
// Tests that even when automatic submits occur during pool cycling, the final chunk
// still needs explicit submit(). Forgetting this should throw TimelineSubmitCancel.
TEST_F(UnitTestFixture, StagingStream_PartialDownloadMissingFinalSubmit) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Create large buffer
    size_t largeSize = 50000; // Large enough to cause multiple auto-submits
    auto   gpuBuffer = vko::BoundBuffer<float>(
        ctx->device, largeSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload data
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Small pools
        vko::StagingStream streaming(queue, std::move(staging));

        vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, largeSize),
                    [](VkDeviceSize offset, auto span) {
                        for (size_t i = 0; i < span.size(); ++i) {
                            span[i] = static_cast<float>(offset + i);
                        }
                    });
        streaming.submit();
        vko::check(ctx->device.vkQueueWaitIdle(queue));
    }

    // Start large download that will cause automatic submits
    auto downloadFuture = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator,
            /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Small pools
        vko::StagingStream streaming(queue, std::move(staging));

        // This will trigger multiple automatic submits as pools cycle
        // but won't submit the final chunk
        auto future = vko::download<float>(streaming, ctx->device,
                                           vko::BufferSpan(gpuBuffer).subspan(0, largeSize),
                                           [](VkDeviceSize, auto input, auto output) {
                                               std::ranges::copy(input, output.begin());
                                           });

        // NO final submit() - some chunks submitted automatically, but not all
        return future;
    }(); // streaming goes out of scope

    // Should throw because final chunk was never submitted
    EXPECT_THROW({ downloadFuture.get(ctx->device); }, vko::TimelineSubmitCancel);
}

// Use-case: Updating a subrange of a larger buffer (e.g., updating part of a texture or mesh)
// Tests that callbacks receive data-relative offsets (starting at 0) even when
// uploading/downloading to non-zero buffer offsets. This is critical for user code to correctly
// index into source data.
TEST_F(UnitTestFixture, StagingStream_NonZeroBufferOffset) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14); // Small 16KB pools to force chunking
    vko::StagingStream streaming(queue, std::move(staging));

    // Create a large buffer
    constexpr VkDeviceSize bufferSize   = 20000;
    constexpr VkDeviceSize uploadOffset = 5000; // Upload starts at 5000, NOT 0
    constexpr VkDeviceSize uploadSize   = 8000; // Upload 8000 elements

    auto gpuBuffer =
        vko::BoundBuffer<int>(ctx->device, bufferSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // UPLOAD TEST: Upload to non-zero offset
    // Verify that callback receives offsets starting at 0, NOT uploadOffset
    std::vector<VkDeviceSize> uploadCallbackOffsets;
    vko::upload(streaming, ctx->device,
                vko::BufferSpan(gpuBuffer).subspan(uploadOffset, uploadSize),
                [&uploadCallbackOffsets](VkDeviceSize offset, auto span) {
                    uploadCallbackOffsets.push_back(offset);
                    // Fill with: value = userOffset + localIndex
                    // This relies on offset being relative to our data (0-based), not the
                    // buffer offset
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<int>(offset + i);
                    }
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Verify that callbacks received 0-based offsets
    ASSERT_FALSE(uploadCallbackOffsets.empty()) << "Upload callback never called";
    EXPECT_EQ(uploadCallbackOffsets.front(), 0)
        << "First upload chunk should have offset 0, not buffer offset";

    // Verify offsets are sequential and cover the range [0, uploadSize)
    VkDeviceSize expectedOffset = 0;
    for (auto cbOffset : uploadCallbackOffsets) {
        EXPECT_EQ(cbOffset, expectedOffset) << "Upload callback offsets should be sequential";
        expectedOffset = cbOffset + streaming.capacity(); // Approximate chunk size
        if (expectedOffset > uploadSize)
            break;
    }

    // DOWNLOAD TEST: Download from non-zero offset
    // Verify that callback receives offsets starting at 0, NOT uploadOffset
    std::vector<VkDeviceSize> downloadCallbackOffsets;
    auto                      downloadFuture = vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(uploadOffset, uploadSize),
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
    EXPECT_EQ(downloadCallbackOffsets.front(), 0)
        << "First download chunk should have offset 0, not buffer offset";

    // Verify the complete downloaded data
    ASSERT_EQ(result.size(), uploadSize);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i))
            << "Downloaded value at index " << i << " doesn't match expected";
    }

    // FOREACH TEST: Test downloadForEach with non-zero offset
    std::vector<VkDeviceSize> forEachCallbackOffsets;
    auto                      forEachHandle = vko::downloadForEach(
        streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(uploadOffset, uploadSize),
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
    EXPECT_EQ(forEachCallbackOffsets.front(), 0)
        << "First forEach chunk should have offset 0, not buffer offset";
}

// Edge case: Zero-size transfers
// Tests that zero-size uploads/downloads are handled gracefully (should be no-ops)
TEST_F(UnitTestFixture, StagingStream_ZeroSizeTransfer) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Zero-size upload should not crash or call callback
    bool uploadCallbackCalled = false;
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 0),
                [&uploadCallbackCalled](VkDeviceSize, auto) { uploadCallbackCalled = true; });

    EXPECT_FALSE(uploadCallbackCalled) << "Zero-size upload should not invoke callback";

    // Zero-size download should work now (supported for free!)
    auto downloadFuture =
        vko::download<int>(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 0),
                           []([[maybe_unused]] VkDeviceSize offset, [[maybe_unused]] auto input,
                              [[maybe_unused]] auto output) {
                               FAIL() << "Zero-size download callback should not be called";
                           });

    streaming.submit();
    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 0) << "Zero-size download should return empty vector";

    // Zero-size forEach should also work
    bool forEachCalled = false;
    auto forEachHandle =
        vko::downloadForEach(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 0),
                             [&forEachCalled](VkDeviceSize, auto) { forEachCalled = true; });

    streaming.submit();
    forEachHandle.wait(ctx->device);
    EXPECT_FALSE(forEachCalled) << "Zero-size forEach should not invoke callback";
}

// Edge case: Unaligned offsets and sizes
// Tests that transfers work correctly with non-aligned buffer offsets and sizes
// Vulkan requires alignment but the staging system should handle this transparently
TEST_F(UnitTestFixture, StagingStream_UnalignedOffsetAndSize) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer with odd size
    constexpr VkDeviceSize bufferSize = 1337; // Odd number
    constexpr VkDeviceSize oddOffset  = 333;  // Odd offset
    constexpr VkDeviceSize oddSize    = 777;  // Odd size

    auto buffer =
        vko::BoundBuffer<int>(ctx->device, bufferSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload with unaligned parameters
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(oddOffset, oddSize),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<int>(offset + i + 1000);
                    }
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download the same unaligned region and verify
    auto downloadFuture =
        vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(oddOffset, oddSize));
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr VkDeviceSize bufferSize = 10000;
    auto                   buffer =
        vko::BoundBuffer<float>(ctx->device, bufferSize,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Define non-contiguous subranges to update
    struct Subrange {
        VkDeviceSize offset;
        VkDeviceSize size;
        float        baseValue;
    };
    std::vector<Subrange> subranges = {
        {100, 500, 1000.0f},   // First region
        {2000, 800, 2000.0f},  // Second region (gap of ~1500)
        {5000, 1200, 5000.0f}, // Third region (gap of ~2200)
        {8000, 300, 8000.0f}   // Fourth region (gap of ~2800)
    };

    // Upload each subrange with different data
    for (const auto& sub : subranges) {
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(sub.offset, sub.size),
                    [baseValue = sub.baseValue](VkDeviceSize offset, auto span) {
                        for (size_t i = 0; i < span.size(); ++i) {
                            span[i] = baseValue + static_cast<float>(offset + i);
                        }
                    });
    }

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download and verify each subrange
    for (const auto& sub : subranges) {
        auto downloadFuture = vko::download(streaming, ctx->device,
                                            vko::BufferSpan(buffer).subspan(sub.offset, sub.size));
        streaming.submit();

        auto& result = downloadFuture.get(ctx->device);
        ASSERT_EQ(result.size(), sub.size) << "Subrange size mismatch";

        for (size_t i = 0; i < result.size(); ++i) {
            float expected = sub.baseValue + static_cast<float>(i);
            EXPECT_FLOAT_EQ(result[i], expected)
                << "Subrange data mismatch at offset " << sub.offset << " index " << i;
        }
    }
}

// Use-case: Cleanup during shutdown (e.g., some async reads complete, others are abandoned)
// Tests that multiple futures can be cancelled or completed in any order without issues
TEST_F(UnitTestFixture, StagingStream_MultipleDownloadsCancelledOutOfOrder) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t                   numDownloads = 5;
    std::vector<vko::BoundBuffer<int>> buffers;

    // Upload data to all buffers
    for (size_t i = 0; i < numDownloads; ++i) {
        buffers.emplace_back(ctx->device, 100,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

        vko::upload(streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, 100),
                    [marker = static_cast<int>(i * 100)](VkDeviceSize offset, auto span) {
                        for (size_t j = 0; j < span.size(); ++j) {
                            span[j] = marker + static_cast<int>(offset + j);
                        }
                    });
    }
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Start downloads
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(buffers[0]).subspan(0, 0), copyLambda));
    std::vector<FutureType> futures;

    for (size_t i = 0; i < numDownloads; ++i) {
        futures.push_back(vko::download<int>(
            streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, 100), copyLambda));
    }
    streaming.submit();

    // Complete some, abandon others in mixed order
    // Complete 0, 2, 4; abandon 1, 3
    for (size_t i : {0, 2, 4}) {
        auto& result = futures[i].get(ctx->device);
        ASSERT_EQ(result.size(), 100u);

        int marker = static_cast<int>(i * 100);
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_EQ(result[j], marker + static_cast<int>(j))
                << "Download " << i << " index " << j;
        }
    }

    // Futures 1 and 3 are destroyed without calling get() - should clean up gracefully
}

// Use-case: Robust error handling (e.g., user callback throws during processing)
// Tests that exceptions in user callbacks are propagated correctly without leaking resources
TEST_F(UnitTestFixture, StagingStream_ExceptionInUserCallback) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<float>(
        ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload should complete even if callback throws
    bool exceptionCaught = false;
    try {
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 500),
                    [](VkDeviceSize, auto span) {
                        // Fill first half, then throw
                        for (size_t i = 0; i < span.size() / 2; ++i) {
                            span[i] = static_cast<float>(i);
                        }
                        throw std::runtime_error("User callback exception");
                    });
    } catch (const std::runtime_error& e) {
        exceptionCaught = true;
        EXPECT_STREQ(e.what(), "User callback exception");
    }

    EXPECT_TRUE(exceptionCaught) << "Exception should have been thrown";

    // StagingStream should still be usable after exception
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i);
                    }
                });
    streaming.submit();

    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), 100u);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
    }
}

// Lifetime and destruction order edge cases (important for multithreaded shutdown scenarios)
// TODO: LifetimeEdgeCase_UploadMappingActiveOnDestruct - StagingStream destroyed while upload
// mapping is still in scope (should unmap)

// Use-case: Mixed workloads with uploads and downloads (e.g., compute pipeline with feedback)
// Tests that alternating uploads and downloads correctly manage command buffer state
TEST_F(UnitTestFixture, StagingStream_InterleavedUploadDownload) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 15);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t       numBuffers = 5;
    constexpr VkDeviceSize bufferSize = 1000;

    // Create multiple buffers
    std::vector<vko::BoundBuffer<int>> buffers;
    for (size_t i = 0; i < numBuffers; ++i) {
        buffers.emplace_back(ctx->device, bufferSize,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);
    }

    // Interleave uploads and downloads
    // Use a simple copy lambda that can be used consistently
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(buffers[0]).subspan(0, 0), copyLambda));
    std::vector<FutureType> futures;

    for (size_t i = 0; i < numBuffers; ++i) {
        // Upload to buffer i
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, bufferSize),
                    [baseValue = i * 1000](VkDeviceSize offset, auto span) {
                        for (size_t j = 0; j < span.size(); ++j) {
                            span[j] = static_cast<int>(baseValue + offset + j);
                        }
                    });

        // Submit upload to ensure it completes before download
        streaming.submit();

        // Download from current buffer (now that upload is submitted)
        futures.push_back(vko::download<int>(streaming, ctx->device,
                                             vko::BufferSpan(buffers[i]).subspan(0, bufferSize),
                                             copyLambda));
    }

    streaming.submit();

    // Verify all downloads
    for (size_t i = 0; i < futures.size(); ++i) {
        auto& result = futures[i].get(ctx->device);
        ASSERT_EQ(result.size(), bufferSize);

        size_t bufferIdx = i; // Downloaded from buffer i (which is i+1-1)
        for (size_t j = 0; j < result.size(); ++j) {
            int expected = static_cast<int>(bufferIdx * 1000 + j);
            EXPECT_EQ(result[j], expected) << "Buffer " << bufferIdx << " index " << j;
        }
    }
}

#define SKIP_STAGING_UPLOADS 0 // Set to 1 to replace uploads with vkCmdFill (isolate download bugs)
#define UPLOAD_EACH_ITERATION 0

inline uint32_t humanValueHash(size_t iteration, size_t download, VkDeviceSize offset, size_t i) {
#if !UPLOAD_EACH_ITERATION
    // If uploaded only once on the first iteration
    iteration = 0;
#endif

#if SKIP_STAGING_UPLOADS
    (void)offset;
    (void)i;
    return (uint32_t(iteration) << 24) | (uint32_t(download) << 16);
#else
    return static_cast<uint32_t>(iteration * 10000000 + download * 100000 + offset + i);
#endif
}

// Use-case: Multiple async readbacks (e.g., profiling multiple GPU timers)
// Tests that multiple downloads can be in flight with different completion times
// Loops internally to accumulate state and increase failure rate
// Uses a background thread to inject delays on the GPU queue to widen race windows
TEST_F(UnitTestFixture, StagingStream_ConcurrentDownloads) {
    constexpr VkDeviceSize downloadSize     = 1697; // counts of uint32_t
    constexpr size_t       numDownloads     = 20;
    constexpr size_t       poolCycleCount   = 3;
    constexpr size_t       maxPools         = 10;
    constexpr size_t       numIterations    = 20; // Fewer iterations since delays make it slower
    constexpr VkDeviceSize bytesPerDownload = downloadSize * sizeof(uint32_t);
    constexpr VkDeviceSize poolSize =
        (bytesPerDownload * numDownloads) / (maxPools * poolCycleCount) + 2957;
    ASSERT_GT(poolSize, 0);
    ASSERT_LT(poolSize * maxPools + downloadSize * numDownloads, 1 << 30); // 1GB total

    constexpr auto sleepAfterChunkDownload = std::chrono::microseconds(100);
    constexpr auto sleepBetweenGpuDelay    = std::chrono::microseconds(100);
    constexpr auto gpuDelaySleep           = std::chrono::microseconds(10);
    constexpr auto callbackRecheckDelay    = std::chrono::microseconds(100);

    // Shared locking queue - guards all submits with internal mutex
    using LockingTimelineQueue = vko::LockingQueue<vko::TimelineQueue>;
    LockingTimelineQueue sharedQueue(vko::TimelineQueue(ctx->device, ctx->queueFamilyIndex, 0));

    // Background thread that injects delays on the GPU queue
    std::atomic<bool> stopDelayThread{false};
    std::jthread      delayThread([&](std::stop_token stopToken) {
        // Thread-local semaphore and counter - no cross-thread coordination needed
        vko::TimelineSemaphore delaySemaphore(ctx->device, 0);
        uint64_t               delayCounter = 0;

        // Use the shared locking queue
        vko::CyclingCommandBuffer<vko::Device, LockingTimelineQueue> cmdBuf(ctx->device,
                                                                                 sharedQueue);

        while (!stopToken.stop_requested() && !stopDelayThread.load()) {
            // Brief pause so delays aren't continuous
            std::this_thread::sleep_for(sleepBetweenGpuDelay);

            uint64_t waitValue = ++delayCounter;

            // Empty command buffer - just need to submit something that waits
            std::ignore = cmdBuf.commandBuffer();

            // Submit waiting on our delay semaphore (queue mutex handled internally)
            VkSemaphoreSubmitInfo waitSemaphore{
                     .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                     .pNext       = nullptr,
                     .semaphore   = delaySemaphore,
                     .value       = waitValue,
                     .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     .deviceIndex = 0,
            };

            cmdBuf.submit(std::array{waitSemaphore}, std::array<VkSemaphoreSubmitInfo, 0>{},
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            // Brief delay before releasing the GPU
            std::this_thread::sleep_for(gpuDelaySleep);

            // Signal the semaphore to release the GPU
            VkSemaphoreSignalInfo signalInfo{
                     .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                     .pNext     = nullptr,
                     .semaphore = delaySemaphore,
                     .value     = waitValue,
            };
            ctx->device.vkSignalSemaphore(ctx->device, &signalInfo);
        }
        // CyclingCommandBuffer destructor waits for in-flight work
    });

#if 1
    stopDelayThread.store(true);
    delayThread.join();
#endif

// Main thread staging - also uses the shared locking queue
#if 1
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/3, maxPools, poolSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
#else
    auto staging = vko::DedicatedStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
#endif
    using CmdBufType = vko::CyclingCommandBuffer<vko::Device, LockingTimelineQueue>;
    CmdBufType                                           cmdBuf(ctx->device, sharedQueue);
    vko::StagingStreamRef<decltype(staging), CmdBufType> streaming(cmdBuf, staging);

    // Create buffers once, reuse across iterations
    std::vector<vko::BoundBuffer<uint32_t>> buffers;
    for (size_t i = 0; i < numDownloads; ++i) {
        buffers.emplace_back(ctx->device, downloadSize,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);
    }

    auto copyLambda = [sleepAfterChunkDownload](VkDeviceSize, auto input, auto output) {
        std::this_thread::sleep_for(sleepAfterChunkDownload);
        std::ranges::copy(input, output.begin());

        // Danger! UB. Clobber the mapped input data to detect stale data on
        // reuse. May just crash with some vulkan implementations that map
        // read-only memory.
        std::span mutableInput(const_cast<uint32_t*>(input.data()), input.size());
        std::ranges::fill(mutableInput, 0xFFFFFFFF);
    };

    for (size_t iter = 0; iter < numIterations; ++iter) {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        if (UPLOAD_EACH_ITERATION || iter == 0) {
#if SKIP_STAGING_UPLOADS
            // Use vkCmdFill instead of staging uploads to isolate download-only bugs
            // This bypasses the staging upload path entirely
            {
                for (size_t i = 0; i < numDownloads; ++i) {
                    // Fill with a simple pattern: (iter << 24) | (i << 16) | (offset_word)
                    // Note: vkCmdFillBuffer fills with a single 32-bit value, so we use a simpler pattern
                    uint32_t fillValue = (uint32_t(iter) << 24) | (uint32_t(i) << 16);
                    ctx->device.vkCmdFillBuffer(streaming.commandBuffer(), buffers[i], 0,
                                                VK_WHOLE_SIZE, fillValue);
                }
                streaming.submit();
                streaming.wait();
            }

            vko::cmdMemoryBarrier(
                ctx->device, streaming.commandBuffer(),
                vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
                vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
#else
            // Upload unique data pattern for this iteration
            for (size_t i = 0; i < numDownloads; ++i) {
                vko::upload(streaming, ctx->device,
                            vko::BufferSpan(buffers[i]).subspan(0, downloadSize),
                            [iter, i](VkDeviceSize offset, auto span) {
                                for (size_t j = 0; j < span.size(); ++j) {
                                    span[j] = humanValueHash(iter, i, offset, j);
                                }
                            });
            }
#endif
            [[maybe_unused]] auto uploadComplete = streaming.submit();
            EXPECT_EQ(streaming.unsubmittedTransfers(), 0);

#if 1
            uploadComplete.wait(ctx->device);
#else
            streaming.wait();
            EXPECT_EQ(streaming.pendingTransfers(), 0);
#endif

            vko::cmdMemoryBarrier(
                ctx->device, streaming.commandBuffer(),
                vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
                vko::MemoryAccess{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
        }

        auto verifyResults = [&iter](size_t offset, std::span<const uint32_t> result,
                                     size_t downloadIdx) {
            std::vector<uint32_t> expected(result.size());
            for (size_t j = 0; j < result.size(); ++j) {
                expected[j] = humanValueHash(iter, downloadIdx, offset, j);
            }
            EXPECT_THAT(result, testing::Pointwise(testing::Eq(), expected))
                << " downloadIdx=" << downloadIdx << " offset=" << offset;
        };

        using TransformFutureType = decltype(vko::download<uint32_t>(
            streaming, ctx->device, vko::BufferSpan(buffers[0]).subspan(0, 0), copyLambda));
        using ForEachFutureType   = decltype(vko::downloadForEach(
            streaming, ctx->device, vko::BufferSpan(buffers[0]).subspan(0, 0),
            std::function<void(VkDeviceSize, std::span<const uint32_t>)>()));

        // Start all downloads concurrently. Alternating between Transform and
        // ForEach to test both paths.
        std::vector<std::pair<size_t, TransformFutureType>> futures;
        std::vector<ForEachFutureType>                      forEachFutures;
        size_t                                              forEachCountSubmitted = 0;
        size_t                                              forEachCountCompleted = 0;
        size_t                                              forEachBytesCompleted = 0;
        for (size_t i = 0; i < numDownloads; ++i) {
            if (i % 2 == 0) {
                futures.push_back(
                    {i, vko::download<uint32_t>(
                            streaming, ctx->device,
                            vko::BufferSpan(buffers[i]).subspan(0, downloadSize), copyLambda)});
            } else {
                bool keepFuture = (i / 2) % 2 == 0;
                forEachCountSubmitted++;
                auto future = vko::downloadForEach(
                    streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, downloadSize),
                    std::function([&sharedQueue, &callbackRecheckDelay, &verifyResults, iter,
                                   downloadIdx = i, &forEachCountCompleted, &forEachBytesCompleted](
                                      VkDeviceSize offset, std::span<const uint32_t> result) {
                        SCOPED_TRACE("foreach download");
                        // Only count the first chunk
                        if (offset == 0) {
                            forEachCountCompleted++;
                        }
                        forEachBytesCompleted += result.size();

// Debug: check if the memory changes while we're in the callback
#if 1
                        std::span<volatile const uint32_t> volatileResult(result);
                        std::vector<uint32_t> before(volatileResult.begin(), volatileResult.end());
                        std::this_thread::sleep_for(callbackRecheckDelay);
    #if 0
                        sharedQueue.wait(ctx->device);
    #endif
                        bool anyNotExpected = false;
                        //printf("Callback download %zu offset=%zu mapping=%p\n", downloadIdx, offset, (const void*)result.data());
                        for (size_t i = 0; i < result.size(); ++i) {
                            uint32_t expected = humanValueHash(iter, downloadIdx, offset, i);
                            if (expected != before[i]) {
                                anyNotExpected = true;
                                ADD_FAILURE() << " offset=" << offset << " expected=" << expected
                                              << " got=" << before[i];
                            }
    #if 1
                            EXPECT_EQ(before[i], volatileResult[i])
                                << " offset=" << offset << " index=" << i
                                << " expected=" << expected;
    #else
                            EXPECT_EQ(expected, before[i])
                                << " offset=" << offset << " index=" << i;
                            EXPECT_EQ(expected, volatileResult[i])
                                << " offset=" << offset << " index=" << i;
                            if (expected != before[i] || expected != volatileResult[i]) {
                                break;
                            }
    #endif
                        }
                        EXPECT_FALSE(anyNotExpected)
                            << " downloadIdx=" << downloadIdx << " offset=" << offset;
#else
                        verifyResults(offset /* partial download */, result, downloadIdx);
#endif

                        // Danger! UB. Clobber the mapped input data to detect stale data on
                        // reuse. May just crash with some vulkan implementations that map
                        // read-only memory.
                        std::span mutableResult(const_cast<uint32_t*>(result.data()),
                                                result.size());
                        std::ranges::fill(mutableResult, 0xFFFFFFFF);
                    }));
                // Save every other future to verify foreach still works if the
                // future is destroyed before the download is complete.
                if (keepFuture) {
                    forEachFutures.push_back(std::move(future));
                }
            }
        }

        streaming.submit();
        EXPECT_EQ(streaming.unsubmittedTransfers(), 0);

        // Verify all downloads to a std::vector
        {
            SCOPED_TRACE("transform downloads");
            for (auto& [idx, future] : futures) {
                auto& result = future.get(ctx->device);
                ASSERT_EQ(result.size(), downloadSize);
                verifyResults(0 /* verifying the entire download */, result, idx);
            }
        }

        // Call all the foreach callbacks
        streaming.wait();
        EXPECT_EQ(streaming.pendingTransfers(), 0);
        EXPECT_EQ(forEachCountSubmitted, forEachCountCompleted);
        EXPECT_EQ(downloadSize * forEachCountSubmitted, forEachBytesCompleted);
    }

    // Stop delay thread - it will finish its current iteration and exit
    if (delayThread.joinable()) {
        stopDelayThread.store(true);
        delayThread.join();
    }

    // Wait for all GPU work to complete before destroying resources
    vko::check(ctx->device.vkQueueWaitIdle(sharedQueue));
}

// Debug test: Stress test with tiny pools to catch race conditions
// Uses downloadForEach and poisons staging memory after read to identify stale data sources
TEST_F(UnitTestFixture, StagingStream_TinyPoolStressTest) {
    constexpr VkDeviceSize poolSize = 16; // 16 bytes = 4 floats per pool

    // Test with progressively more pools to catch issues at different recycling pressures
    for (size_t maxPools : {1, 2, 3, 10}) {
        SCOPED_TRACE("maxPools = " + std::to_string(maxPools));

        vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
        auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
            ctx->device, ctx->allocator, /*minPools=*/1, maxPools, poolSize);
        vko::StagingStream streaming(queue, std::move(staging));

        // Create a buffer larger than total pool capacity to force chunking and recycling
        constexpr VkDeviceSize bufferSize =
            32; // 32 floats = 128 bytes = 8 chunks with 16-byte pools
        auto buffer = vko::BoundBuffer<float>(ctx->device, bufferSize,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

        // Upload known data pattern
        float uploadMarker = static_cast<float>(maxPools * 1000);
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize),
                    [uploadMarker](VkDeviceSize offset, auto span) {
                        for (size_t j = 0; j < span.size(); ++j) {
                            span[j] = uploadMarker + static_cast<float>(offset + j);
                        }
                    });
        streaming.submit();
        vko::check(ctx->device.vkQueueWaitIdle(queue));

        // Download and verify using downloadForEach, with poison values written after read
        size_t              chunkIndex = 0;
        std::vector<size_t> chunkOffsets;
        std::vector<size_t> chunkSizes;
        size_t              errorCount = 0;

        auto handle = vko::downloadForEach(
            streaming, ctx->device, vko::BufferSpan(buffer),
            [&, uploadMarker](VkDeviceSize offset, std::span<const float> mapped) {
                size_t thisChunk = chunkIndex++;
                chunkOffsets.push_back(static_cast<size_t>(offset));
                chunkSizes.push_back(mapped.size());

                // Verify each value in this chunk
                for (size_t j = 0; j < mapped.size(); ++j) {
                    float expected = uploadMarker + static_cast<float>(offset + j);
                    if (mapped[j] != expected) {
                        if (errorCount < 10) { // Limit output
                            ADD_FAILURE() << "maxPools=" << maxPools << " chunk=" << thisChunk
                                          << " offset=" << offset << " index=" << j
                                          << " expected=" << expected << " got=" << mapped[j];
                        }
                        errorCount++;
                    }
                }

                // POISON: Write unique values to detect stale staging buffer reads
                // Encodes: iteration (maxPools), chunk index, and position within chunk
                auto mutableSpan = const_cast<float*>(mapped.data());
                for (size_t j = 0; j < mapped.size(); ++j) {
                    // Poison pattern: negative value encoding maxPools, chunk, and index
                    // Format: -(maxPools * 10000 + chunk * 100 + j)
                    mutableSpan[j] = -static_cast<float>(maxPools * 10000 + thisChunk * 100 + j);
                }
            });

        streaming.submit();
        handle.wait(ctx->device);

        EXPECT_EQ(errorCount, 0) << "maxPools=" << maxPools << " had " << errorCount << " errors";
        EXPECT_GT(chunkIndex, 0) << "No chunks were processed";

        // Log chunk info for debugging
        if (::testing::Test::HasFailure()) {
            std::cout << "Chunk layout for maxPools=" << maxPools << ":\n";
            for (size_t i = 0; i < chunkOffsets.size(); ++i) {
                std::cout << "  chunk " << i << ": offset=" << chunkOffsets[i]
                          << " size=" << chunkSizes[i] << "\n";
            }
        }
    }
}

// Use-case: Long-running streaming with many submissions (e.g., video encoding)
// Tests that command buffers are properly recycled and not leaked over many submits
TEST_F(UnitTestFixture, StagingStream_CommandBufferRecycling) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<int>(ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Perform many small uploads with submits between them
    // This should trigger command buffer recycling
    constexpr size_t numIterations = 50;
    for (size_t i = 0; i < numIterations; ++i) {
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 500),
                    [value = static_cast<int>(i)]([[maybe_unused]] VkDeviceSize offset, auto span) {
                        for (size_t j = 0; j < span.size(); ++j) {
                            span[j] = value;
                        }
                    });

        streaming.submit();

        // Wait occasionally to allow recycling
        if (i % 10 == 9) {
            vko::check(ctx->device.vkQueueWaitIdle(queue));
        }
    }

    // Final wait
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // If command buffers leaked, memory usage would grow unbounded
    // No assertion needed - test passes if it doesn't crash or OOM
    SUCCEED() << "Command buffer recycling appears to work correctly";
}

// Use-case: High-frequency streaming with limited memory (e.g., real-time texture streaming)
// Tests behavior when all pools are exhausted and allocation must wait/block
TEST_F(UnitTestFixture, StagingStream_MemoryPressure) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/2, /*poolSize=*/1 << 14); // Only 2 pools
    vko::StagingStream streaming(queue, std::move(staging));

    // Create a transfer much larger than total pool capacity
    constexpr VkDeviceSize totalPoolCapacity = 2 * (1 << 14) / sizeof(float);
    constexpr VkDeviceSize largeTransferSize = totalPoolCapacity * 3; // 3x capacity

    auto buffer =
        vko::BoundBuffer<float>(ctx->device, largeTransferSize,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // This upload should automatically submit and wait when pools are exhausted
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, largeTransferSize),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i);
                    }
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Verify the transfer completed successfully despite memory pressure
    auto downloadFuture = vko::download(streaming, ctx->device,
                                        vko::BufferSpan(buffer).subspan(0, largeTransferSize));
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t       numOperations = 5;
    constexpr VkDeviceSize bufferSize    = 100;

    auto buffer =
        vko::BoundBuffer<int>(ctx->device, bufferSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };
    using FutureType = decltype(vko::download<int>(
        streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 0), copyLambda));
    std::vector<FutureType> futures;

    // Chain operations: upload, download, upload, download, etc.
    // Each operation depends on the previous via semaphore chaining
    for (size_t i = 0; i < numOperations; ++i) {
        // Upload with incremented values
        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize),
                    [iteration = i](VkDeviceSize offset, auto span) {
                        for (size_t j = 0; j < span.size(); ++j) {
                            span[j] = static_cast<int>(iteration * 1000 + offset + j);
                        }
                    });
        streaming.submit();

        // Download to verify
        futures.push_back(vko::download<int>(
            streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize), copyLambda));
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
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<float>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 1000),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<float>(offset + i);
                    }
                });
    streaming.submit();

    {
        // Create future but never call get() - should cancel on destruction
        auto abandonedFuture =
            vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 1000));
        streaming.submit();
        // Future destroyed here without get()
    }

    // Staging should still be usable after abandoned future
    auto normalFuture =
        vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 500));
    streaming.submit();
    auto& result = normalFuture.get(ctx->device);

    ASSERT_EQ(result.size(), 500);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
    }
}

// Use-case: Long-lived result caching (e.g., keeping readback results after streaming context
// closed) Tests that futures can be evaluated before StagingStream destruction and data accessed
// after
TEST_F(UnitTestFixture, StagingStream_FutureOutlivesStaging) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };

    auto future = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/2, /*maxPools=*/3,
                                                                   /*poolSize=*/1 << 14);
        vko::StagingStream streaming(queue, std::move(staging));

        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 500),
                    [](VkDeviceSize offset, auto span) {
                        for (size_t i = 0; i < span.size(); ++i) {
                            span[i] = static_cast<int>(offset + i * 2);
                        }
                    });
        streaming.submit();

        auto result = vko::download<int>(streaming, ctx->device,
                                         vko::BufferSpan(buffer).subspan(0, 500), copyLambda);
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

// Use-case: Futures remain valid after staging destruction when work was properly submitted
// Tests that submitted GPU work completes successfully even after StagingStream destructs,
// thanks to proper destructor synchronization
TEST_F(UnitTestFixture, StagingStream_FutureCompletesAfterStagingDestroyed) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    auto buffer = vko::BoundBuffer<float>(
        ctx->device, 200, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };

    // Create future and destroy StagingStream before evaluation
    auto future = [&]() {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/2, /*maxPools=*/3,
                                                                   /*poolSize=*/1 << 14);
        vko::StagingStream streaming(queue, std::move(staging));

        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 200),
                    [](VkDeviceSize offset, auto span) {
                        for (size_t i = 0; i < span.size(); ++i) {
                            span[i] = static_cast<float>(offset + i);
                        }
                    });
        streaming.submit();

        auto result = vko::download<float>(streaming, ctx->device,
                                           vko::BufferSpan(buffer).subspan(0, 200), copyLambda);
        streaming.submit();

        // StagingStream destructor waits for submitted work, so future remains valid
        return result;
    }(); // streaming destroyed here, but work was submitted so it waits

    // Future should work fine because destructor waited for GPU work
    EXPECT_NO_THROW({
        auto& data = future.get(ctx->device);
        ASSERT_EQ(data.size(), 200u);
        // Verify data is correct
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
        }
    });
}

// Use-case: Dynamic result storage (e.g., moving futures into containers or returning from
// functions) Tests that futures can be moved during pending transfers and remain valid
TEST_F(UnitTestFixture, StagingStream_MoveSemanticsDuringPendingTransfers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/3,
                                                               /*poolSize=*/1 << 14);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t                      numBuffers = 3;
    std::vector<vko::BoundBuffer<double>> buffers;
    for (size_t i = 0; i < numBuffers; ++i) {
        buffers.emplace_back(ctx->device, 200,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

        vko::upload(streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, 200),
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
    using FutureType = decltype(vko::download<double>(
        streaming, ctx->device, vko::BufferSpan(buffers[0]).subspan(0, 0), copyLambda));
    std::vector<FutureType> futures;

    for (size_t i = 0; i < numBuffers; ++i) {
        futures.push_back(vko::download<double>(
            streaming, ctx->device, vko::BufferSpan(buffers[i]).subspan(0, 200), copyLambda));
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

// Use-case: End-to-end data transfer with actual GPU operations
// Tests the full pipeline: allocate staging, copy to device buffer, copy back, verify data
TEST_F(UnitTestFixture, RecyclingStagingPool_ActualDataTransfer) {
    vko::SerialTimelineQueue                    queue(ctx->device, ctx->queueFamilyIndex, 0);
    vko::vma::RecyclingStagingPool<vko::Device> staging(ctx->device, ctx->allocator,
                                                        /*minPools=*/1, /*maxPools=*/3,
                                                        /*poolSize=*/1 << 16);

    constexpr size_t      numElements = 256;
    std::vector<uint32_t> sourceData(numElements);
    std::iota(sourceData.begin(), sourceData.end(), 0u);

    // Create device-local buffer
    vko::BoundBuffer<uint32_t> deviceBuffer(ctx->device, numElements,
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload via staging buffer
    auto* uploadStaging = staging.allocateUpTo<uint32_t>(numElements, [](bool) {});
    ASSERT_NE(uploadStaging, nullptr);
    ASSERT_EQ(uploadStaging->size(), numElements);

    {
        auto mapped = uploadStaging->map();
        std::ranges::copy(sourceData, mapped.begin());
    }

    // Record and submit copy command
    auto pool = ctx->createCommandPool();
    auto cmd  = ctx->beginRecording(pool);

    vko::copyBuffer(ctx->device, cmd, vko::BufferSpan(*uploadStaging),
                    vko::BufferSpan(deviceBuffer));

    auto               uploadSemaphore     = queue.nextSubmitSemaphore();
    vko::CommandBuffer uploadCommandBuffer = cmd.end();
    queue.submit(ctx->device, {}, uploadCommandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    staging.endBatch(uploadSemaphore);
    uploadSemaphore.wait(ctx->device);
    staging.wait();

    // Download to verify
    auto* downloadStaging = staging.allocateUpTo<uint32_t>(numElements, [](bool) {});
    ASSERT_NE(downloadStaging, nullptr);

    cmd = ctx->beginRecording(pool);
    vko::copyBuffer(ctx->device, cmd, vko::BufferSpan(deviceBuffer),
                    vko::BufferSpan(*downloadStaging));

    auto               downloadSemaphore     = queue.nextSubmitSemaphore();
    vko::CommandBuffer downloadCommandBuffer = cmd.end();
    queue.submit(ctx->device, {}, downloadCommandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    staging.endBatch(downloadSemaphore);
    downloadSemaphore.wait(ctx->device);

    // Verify data
    {
        auto mapped = downloadStaging->map();
        for (size_t i = 0; i < numElements; ++i) {
            EXPECT_EQ(mapped[i], sourceData[i]) << "Mismatch at index " << i;
        }
    }

    staging.wait();
}

// Mock staging allocator that always fails allocations
namespace {
template <class DeviceAndCommandsType>
struct FailingStagingAllocator {
    using DeviceAndCommands                      = DeviceAndCommandsType;
    using Allocator                              = vko::vma::Allocator;
    static constexpr bool AllocateAlwaysSucceeds = false;
    static constexpr bool AllocateAlwaysFull     = false;

    FailingStagingAllocator(const DeviceAndCommands& device, Allocator&,
                            VkBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : m_device(device) {}

    FailingStagingAllocator(FailingStagingAllocator&&)            = default;
    FailingStagingAllocator& operator=(FailingStagingAllocator&&) = default;

    // Always fail to allocate
    template <class T>
    bool tryWith(size_t, std::function<void(const vko::BoundBuffer<T, Allocator>&)>) {
        return false;
    }

    // Overload with per-buffer callback - also fails
    template <class T>
    bool tryWith(size_t, std::function<void(const vko::BoundBuffer<T, Allocator>&)>,
                 std::function<void(bool)>) {
        return false;
    }

    // allocateUpTo - returns nullptr on failure
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t) {
        return nullptr;
    }

    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t, std::function<void(bool)>) {
        return nullptr;
    }

    // allocateSingleAndUpTo - returns nullopt on failure
    template <class TSingle, class TUpTo>
    std::optional<
        std::pair<vko::BoundBuffer<TSingle, Allocator>&, vko::BoundBuffer<TUpTo, Allocator>&>>
    allocateSingleAndUpTo(VkDeviceSize) {
        return std::nullopt;
    }

    void                     registerBatchCallback(std::function<void(bool)>) {}
    void                     endBatch(vko::SemaphoreValue) {}
    void                     poll() {}
    void                     wait() {}
    VkDeviceSize             capacity() const { return 0; }
    VkDeviceSize             size() const { return 0; }
    const DeviceAndCommands& device() const { return m_device; }

private:
    const DeviceAndCommands& m_device;
};
} // namespace
static_assert(vko::staging_allocator<FailingStagingAllocator<vko::Device>>);

// Use-case: Persistent allocation failure detection
// Tests that StagingStream detects and throws when allocation persistently fails
TEST_F(UnitTestFixture, StagingStream_PersistentAllocationFailureThrows) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Create a staging allocator that always fails
    FailingStagingAllocator<vko::Device> failingStaging(ctx->device, ctx->allocator,
                                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    vko::StagingStream streaming(queue, std::move(failingStaging));

    // Create a test buffer
    vko::BoundBuffer<uint32_t> buffer(
        ctx->device, 100, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Try to upload - should throw after submit/retry fails
    EXPECT_THROW(
        {
            vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100),
                        [](VkDeviceSize, auto span) {
                            for (size_t i = 0; i < span.size(); ++i) {
                                span[i] = static_cast<uint32_t>(i);
                            }
                        });
        },
        std::runtime_error);
}

// Use-case: Multiple queue families for parallel transfer operations
// Tests simultaneous staging operations on different queue families
TEST_F(UnitTestFixture, StagingStream_MultipleQueueSupport) {
    if (!ctx->queueFamilyIndex2.has_value()) {
        GTEST_SKIP() << "Test requires two queue families";
    }

    vko::SerialTimelineQueue queue1(ctx->device, ctx->queueFamilyIndex, 0);
    vko::SerialTimelineQueue queue2(ctx->device, ctx->queueFamilyIndex2.value(), 0);

    auto staging1 = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/1, /*maxPools=*/3,
                                                                /*poolSize=*/1 << 16);
    auto staging2 = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/1, /*maxPools=*/3,
                                                                /*poolSize=*/1 << 16);

    vko::StagingStream stream1(queue1, std::move(staging1));
    vko::StagingStream stream2(queue2, std::move(staging2));

    constexpr size_t bufferSize = 512;

    // Create buffers for each queue
    vko::BoundBuffer<float> buffer1(ctx->device, bufferSize,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::BoundBuffer<float> buffer2(ctx->device, bufferSize,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload on both queues simultaneously
    vko::upload(stream1, ctx->device, vko::BufferSpan(buffer1), [](VkDeviceSize offset, auto span) {
        for (size_t i = 0; i < span.size(); ++i) {
            span[i] = static_cast<float>(offset + i) * 10.0f;
        }
    });

    vko::upload(stream2, ctx->device, vko::BufferSpan(buffer2), [](VkDeviceSize offset, auto span) {
        for (size_t i = 0; i < span.size(); ++i) {
            span[i] = static_cast<float>(offset + i) * 20.0f;
        }
    });

    stream1.submit();
    stream2.submit();

    // Download and verify from both queues
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };

    auto future1 = vko::download<float>(stream1, ctx->device, vko::BufferSpan(buffer1), copyLambda);
    auto future2 = vko::download<float>(stream2, ctx->device, vko::BufferSpan(buffer2), copyLambda);

    stream1.submit();
    stream2.submit();

    auto& result1 = future1.get(ctx->device);
    auto& result2 = future2.get(ctx->device);

    ASSERT_EQ(result1.size(), bufferSize);
    ASSERT_EQ(result2.size(), bufferSize);

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(result1[i], static_cast<float>(i) * 10.0f);
        EXPECT_FLOAT_EQ(result2[i], static_cast<float>(i) * 20.0f);
    }
}

// Use-case: Queue family ownership transfer for cross-queue resource sharing
// Tests buffer ownership transfer between queue families with proper barriers
TEST_F(UnitTestFixture, StagingStream_QueueFamilyTransition) {
    if (!ctx->queueFamilyIndex2.has_value()) {
        GTEST_SKIP() << "Test requires two queue families";
    }

    vko::SerialTimelineQueue queue1(ctx->device, ctx->queueFamilyIndex, 0);
    vko::SerialTimelineQueue queue2(ctx->device, ctx->queueFamilyIndex2.value(), 0);

    auto staging1 = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/1, /*maxPools=*/2,
                                                                /*poolSize=*/1 << 16);
    auto staging2 = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/1, /*maxPools=*/2,
                                                                /*poolSize=*/1 << 16);

    vko::StagingStream stream1(queue1, std::move(staging1));
    vko::StagingStream stream2(queue2, std::move(staging2));

    constexpr size_t bufferSize = 256;

    // Create buffer that will be transferred between queues
    vko::BoundBuffer<uint32_t> sharedBuffer(ctx->device, bufferSize,
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload using queue1
    vko::upload(stream1, ctx->device, vko::BufferSpan(sharedBuffer),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<uint32_t>(offset + i + 1000);
                    }
                });

    auto uploadComplete = stream1.submit();
    uploadComplete.wait(ctx->device);

    // Transfer ownership from queue1 to queue2 with barrier
    auto pool1 = ctx->createCommandPool();
    auto cmd1  = ctx->beginRecording(pool1);

    VkBufferMemoryBarrier releaseBarrier{.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         .pNext         = nullptr,
                                         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                         .dstAccessMask = 0,
                                         .srcQueueFamilyIndex = ctx->queueFamilyIndex,
                                         .dstQueueFamilyIndex = ctx->queueFamilyIndex2.value(),
                                         .buffer              = sharedBuffer,
                                         .offset              = 0,
                                         .size                = VK_WHOLE_SIZE};

    ctx->device.vkCmdPipelineBarrier(cmd1, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1,
                                     &releaseBarrier, 0, nullptr);

    auto               releaseSemaphore     = queue1.nextSubmitSemaphore();
    vko::CommandBuffer releaseCommandBuffer = cmd1.end();
    queue1.submit(ctx->device, {}, releaseCommandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    releaseSemaphore.wait(ctx->device);

    // Acquire ownership on queue2
    auto pool2Pool = vko::CommandPool(
        ctx->device,
        VkCommandPoolCreateInfo{.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                .pNext            = nullptr,
                                .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                .queueFamilyIndex = ctx->queueFamilyIndex2.value()});

    auto cmd2 = vko::simple::RecordingCommandBuffer(
        ctx->device,
        vko::CommandBuffer(ctx->device, ctx->device, nullptr, pool2Pool,
                           VK_COMMAND_BUFFER_LEVEL_PRIMARY),
        VkCommandBufferBeginInfo{.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .pNext            = nullptr,
                                 .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                 .pInheritanceInfo = nullptr});

    VkBufferMemoryBarrier acquireBarrier{.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         .pNext         = nullptr,
                                         .srcAccessMask = 0,
                                         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                         .srcQueueFamilyIndex = ctx->queueFamilyIndex,
                                         .dstQueueFamilyIndex = ctx->queueFamilyIndex2.value(),
                                         .buffer              = sharedBuffer,
                                         .offset              = 0,
                                         .size                = VK_WHOLE_SIZE};

    ctx->device.vkCmdPipelineBarrier(cmd2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                                     &acquireBarrier, 0, nullptr);

    auto               acquireSemaphore     = queue2.nextSubmitSemaphore();
    vko::CommandBuffer acquireCommandBuffer = cmd2.end();
    queue2.submit(ctx->device, {}, acquireCommandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    acquireSemaphore.wait(ctx->device);

    // Download using queue2
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };

    auto future =
        vko::download<uint32_t>(stream2, ctx->device, vko::BufferSpan(sharedBuffer), copyLambda);
    stream2.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), bufferSize);

    // Verify data survived the queue transfer
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(result[i], static_cast<uint32_t>(i + 1000))
            << "Data corruption at index " << i << " after queue transfer";
    }
}

// Use-case: Atomic paired allocation succeeds with exact fit
// Tests that withSingleAndStagingBuffer allocates both buffers successfully
TEST_F(UnitTestFixture, StagingStream_AtomicPairAllocation_ExactFit) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Small pool to test basic functionality
    constexpr size_t dataSize     = 1000;
    constexpr size_t tinyPoolSize = 4096; // Generous headroom for alignment and VMA overhead

    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2,
                                                               tinyPoolSize);
    vko::StagingStream streaming(queue, std::move(staging));

    int callCount = 0;

    streaming.withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::byte>(
        dataSize,
        [&](VkCommandBuffer, auto& cmd, auto& data,
            VkDeviceSize) -> std::optional<std::function<void(bool)>> {
            callCount++;
            EXPECT_EQ(cmd.size(), 1u);
            EXPECT_GT(data.size(), 0u);
            EXPECT_LE(data.size(), dataSize);
            return std::nullopt;
        });

    EXPECT_GT(callCount, 0);

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Atomic paired allocation chunks when data exceeds pool size
// Tests automatic chunking and retry logic
TEST_F(UnitTestFixture, StagingStream_AtomicPairAllocation_Chunking) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    constexpr size_t       poolSize = 2048;
    constexpr VkDeviceSize dataSize = 5000; // Larger than a single pool to force chunking

    auto staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2, poolSize);
    vko::StagingStream streaming(queue, std::move(staging));

    int          callCount      = 0;
    VkDeviceSize totalProcessed = 0;

    streaming.withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::byte>(
        dataSize,
        [&](VkCommandBuffer, auto& cmd, auto& data,
            VkDeviceSize offset) -> std::optional<std::function<void(bool)>> {
            callCount++;
            EXPECT_EQ(cmd.size(), 1u);
            EXPECT_EQ(offset, totalProcessed);
            totalProcessed += data.size();
            return std::nullopt;
        });

    EXPECT_GT(callCount, 1) << "Should have chunked into multiple calls";
    EXPECT_EQ(totalProcessed, dataSize);

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Atomic paired allocation userOffset increments correctly
// Tests that userOffset parameter tracks progress through the data
TEST_F(UnitTestFixture, StagingStream_AtomicPairAllocation_OffsetTracking) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    constexpr size_t       poolSize  = 4096;
    constexpr VkDeviceSize largeSize = 8000; // Larger than single pool to force chunking

    auto staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2, poolSize);
    vko::StagingStream streaming(queue, std::move(staging));

    std::vector<VkDeviceSize> offsets;

    streaming.withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::byte>(
        largeSize,
        [&](VkCommandBuffer, auto&, auto&,
            VkDeviceSize offset) -> std::optional<std::function<void(bool)>> {
            offsets.push_back(offset);
            return std::nullopt;
        });

    EXPECT_FALSE(offsets.empty());
    EXPECT_EQ(offsets[0], 0u) << "First chunk should start at 0";

    // Verify offsets are sequential
    for (size_t i = 1; i < offsets.size(); ++i) {
        EXPECT_GT(offsets[i], offsets[i - 1]) << "Offsets should increment";
    }

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Test download(DeviceSpan) with real data using VK_NV_copy_memory_indirect
TEST_F(UnitTestFixture, StagingStream_Download_DeviceSpan) {
    // Check if VK_NV_copy_memory_indirect is supported
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    constexpr size_t dataSize = 1000;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 42);

    // Create device buffer with shader device address support
    auto buffer = vko::BoundBuffer<int>(ctx->device, dataSize,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 1, 3, 16384,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    // Upload test data
    vko::upload(streaming, ctx->device, testData, vko::BufferSpan(buffer));
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Test 1: Download full buffer using DeviceSpan constructor
    {
        vko::DeviceAddress<int>    deviceAddr(buffer, ctx->device);
        vko::DeviceSpan<const int> deviceSpan(vko::DeviceAddress<const int>(deviceAddr.raw()),
                                              buffer.size());

        auto future = vko::download(streaming, ctx->device, deviceSpan);
        streaming.submit();

        auto& result = future.get(ctx->device);

        ASSERT_EQ(result.size(), dataSize);
        for (size_t i = 0; i < dataSize; ++i) {
            EXPECT_EQ(result[i], testData[i]) << "Mismatch at index " << i;
        }
    }

    // Test 2: Download subspan with offset to verify offset handling
    {
        constexpr size_t offset = 100;
        constexpr size_t count  = 200;

        vko::DeviceAddress<int>    addr(buffer, ctx->device);
        vko::DeviceSpan<const int> fullSpan(vko::DeviceAddress<const int>(addr.raw()),
                                            buffer.size());
        vko::DeviceSpan<const int> subSpan = fullSpan.subspan(offset, count);

        auto future = vko::download(streaming, ctx->device, subSpan);
        streaming.submit();

        auto& result = future.get(ctx->device);

        ASSERT_EQ(result.size(), count);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(result[i], testData[offset + i]) << "Mismatch at index " << i;
        }
    }
}

// Test upload(DeviceSpan) with callback using VK_NV_copy_memory_indirect
TEST_F(UnitTestFixture, StagingStream_Upload_DeviceSpan_Callback) {
    // Check if VK_NV_copy_memory_indirect is supported
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    constexpr size_t dataSize = 1000;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 42);

    // Create device buffer with shader device address support
    auto buffer =
        vko::BoundBuffer<int>(ctx->device, dataSize,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 1, 3, 16384,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    // Upload using callback overload
    vko::DeviceAddress<int> deviceAddr(buffer, ctx->device);
    vko::DeviceSpan<int>    deviceSpan(deviceAddr, buffer.size());

    vko::upload<int>(streaming, ctx->device, deviceSpan, dataSize,
                     [&](VkDeviceSize offset, std::span<int> mapped) {
                         std::copy_n(testData.begin() + offset, mapped.size(), mapped.begin());
                     });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Verify by downloading the data back
    auto future =
        vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, dataSize));
    streaming.submit();

    auto& result = future.get(ctx->device);

    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], testData[i]) << "Mismatch at index " << i;
    }
}

// Test upload(DeviceSpan) with range overload using VK_NV_copy_memory_indirect
TEST_F(UnitTestFixture, StagingStream_Upload_DeviceSpan_Range) {
    // Check if VK_NV_copy_memory_indirect is supported
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    constexpr size_t dataSize = 500;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 100);

    // Create device buffer with shader device address support
    auto buffer =
        vko::BoundBuffer<int>(ctx->device, dataSize,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 1, 3, 16384,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    // Upload using range overload
    vko::DeviceAddress<int> deviceAddr(buffer, ctx->device);
    vko::DeviceSpan<int>    deviceSpan(deviceAddr, buffer.size());

    vko::upload(streaming, ctx->device, testData, deviceSpan);
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Verify by downloading the data back
    auto future =
        vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, dataSize));
    streaming.submit();

    auto& result = future.get(ctx->device);

    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], testData[i]) << "Mismatch at index " << i;
    }
}

// Test upload() with range (not callback) to typed buffer
TEST_F(UnitTestFixture, StagingStream_UploadRange) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 3, 16384);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize = 300;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 42);

    auto buffer = vko::BoundBuffer<int>(
        ctx->device, dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Use free function with range
    vko::upload(streaming, ctx->device, testData, vko::BufferSpan(buffer));
    streaming.submit();

    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], testData[i]);
    }
}

// Test upload() that creates a new buffer via callback
TEST_F(UnitTestFixture, StagingStream_UploadCreateBuffer_Callback) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 3, 16384);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize = 200;

    // Create buffer and upload via callback in one call
    auto buffer = vko::upload<int>(
        streaming, ctx->device, ctx->allocator, dataSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        [](VkDeviceSize offset, std::span<int> mapped) {
            std::iota(mapped.begin(), mapped.end(), static_cast<int>(offset) + 100);
        });
    streaming.submit();

    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i + 100));
    }
}

// Test upload() that creates a new buffer from range
TEST_F(UnitTestFixture, StagingStream_UploadCreateBuffer_Range) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 3, 16384);
    vko::StagingStream streaming(queue, std::move(staging));

    std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Create buffer and upload from range in one call
    auto buffer = vko::upload(streaming, ctx->device, ctx->allocator, testData,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    streaming.submit();

    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), testData.size());
    for (size_t i = 0; i < testData.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], testData[i]);
    }
}

// Test download() with transform from DeviceSpan
// NOTE: Current implementation requires DstT == DeviceSpan element type
TEST_F(UnitTestFixture, StagingStream_Download_DeviceSpan_Transform) {
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 1, 3, 16384,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize = 100;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 10);

    auto buffer =
        vko::BoundBuffer<int>(ctx->device, dataSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, testData, vko::BufferSpan(buffer));
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Create DeviceSpan and download with transform
    vko::DeviceAddress<int>    addr(buffer, ctx->device);
    vko::DeviceSpan<const int> span(vko::DeviceAddress<const int>(addr.raw()), buffer.size());

    // Transform: multiply each value by 3
    auto future =
        vko::download<int>(streaming, ctx->device, span,
                           [](VkDeviceSize, std::span<const int> input, std::span<int> output) {
                               for (size_t i = 0; i < input.size(); ++i) {
                                   output[i] = input[i] * 3;
                               }
                           });
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], (static_cast<int>(i) + 10) * 3);
    }
}

// Test downloadForEach() from DeviceSpan
TEST_F(UnitTestFixture, StagingStream_DownloadForEach_DeviceSpan) {
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, 1, 3, 16384,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize = 150;
    std::vector<int> testData(dataSize);
    std::iota(testData.begin(), testData.end(), 1);

    auto buffer =
        vko::BoundBuffer<int>(ctx->device, dataSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, testData, vko::BufferSpan(buffer));
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Create DeviceSpan and downloadForEach
    vko::DeviceAddress<int>    addr(buffer, ctx->device);
    vko::DeviceSpan<const int> span(vko::DeviceAddress<const int>(addr.raw()), buffer.size());

    int  count  = 0;
    int  maxVal = 0;
    auto handle = vko::downloadForEach<const int>(
        streaming, ctx->device, span, [&count, &maxVal](VkDeviceSize, std::span<const int> data) {
            for (int val : data) {
                count++;
                maxVal = std::max(maxVal, val);
            }
        });
    streaming.submit();
    handle.wait(ctx->device);

    EXPECT_EQ(count, static_cast<int>(dataSize));
    EXPECT_EQ(maxVal, static_cast<int>(dataSize)); // max value is 150
}

// Test upload() with BufferSpan and callback
TEST_F(UnitTestFixture, StagingStream_UploadBufferSpan_Callback) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 3, 16384);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize = 128;
    auto             buffer   = vko::BoundBuffer<std::byte>(
        ctx->device, dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Use upload free function with BufferSpan
    vko::BufferSpan<std::byte> bufSpan(buffer);
    vko::upload(streaming, ctx->device, bufSpan,
                [](VkDeviceSize offset, std::span<std::byte> mapped) {
                    for (size_t i = 0; i < mapped.size(); ++i) {
                        mapped[i] = static_cast<std::byte>((offset + i) & 0xFF);
                    }
                });
    streaming.submit();

    // Verify by downloading
    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], static_cast<std::byte>(i & 0xFF));
    }
}

// Test upload() with BufferSpan and range
TEST_F(UnitTestFixture, StagingStream_UploadBufferSpan_Range) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 3, 16384);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t       dataSize = 64;
    std::vector<std::byte> testData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        testData[i] = static_cast<std::byte>(i * 2);
    }

    auto buffer = vko::BoundBuffer<std::byte>(
        ctx->device, dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Use upload free function with BufferSpan and range
    vko::BufferSpan<std::byte> bufSpan(buffer);
    vko::upload(streaming, ctx->device, testData, bufSpan);
    streaming.submit();

    // Verify by downloading
    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(buffer));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], testData[i]);
    }
}

// Verify that cancellation works correctly in two scenarios:
// 1. Legitimate cancellation - submit() called but streaming destroyed before future evaluated
// 2. Missing submit() - user forgets to call submit(), should throw TimelineSubmitCancel
TEST_F(UnitTestFixture, StagingStream_CancellationBehavior) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    auto buffer = vko::BoundBuffer<int>(
        ctx->device, 100, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Case 1: submit() IS called - cancellation should work cleanly
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/1, /*maxPools=*/2,
                                                                   /*poolSize=*/1 << 16);
        vko::StagingStream streaming(queue, std::move(staging));

        vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100),
                    [](VkDeviceSize offset, auto span) {
                        std::iota(span.begin(), span.end(), static_cast<int>(offset));
                    });
        streaming.submit();
        vko::check(ctx->device.vkQueueWaitIdle(queue));

        auto future =
            vko::download<int>(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100),
                               [](VkDeviceSize, auto input, auto output) {
                                   std::ranges::copy(input, output.begin());
                               });
        streaming.submit();

        // Destroy streaming before calling get() - legitimate cancellation
    }

    // Case 2: submit() is NOT called - should throw TimelineSubmitCancel
    {
        auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                   /*minPools=*/1, /*maxPools=*/2,
                                                                   /*poolSize=*/1 << 16);
        vko::StagingStream streaming(queue, std::move(staging));

        auto future =
            vko::download<int>(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, 100),
                               [](VkDeviceSize, auto input, auto output) {
                                   std::ranges::copy(input, output.begin());
                               });

        // NO submit() - user forgot!
        // Streaming destructs, cancels all pending futures
    }
    // Note: We can't easily detect missing submit() with an assert because
    // the semaphore value is allocated before submit (from nextSubmitSemaphore).
    // The TimelineSubmitCancel exception is the mechanism that catches this.
}

// Regression test: Destructor must wait for in-flight GPU work before destroying buffers
// Tests that rapid creation/destruction of staging pools with submitted work doesn't cause
// device loss or data corruption from use-after-free on GPU. If destructors don't wait,
// staging buffers get freed while GPU is still reading from them, causing corruption or crash.
TEST_F(UnitTestFixture, StagingStream_DestructorWaitsForInFlightWork) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Use larger buffers to increase GPU operation duration
    constexpr size_t bufferSize = 5000;
    auto             buffer =
        vko::BoundBuffer<int>(ctx->device, bufferSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Lambda for downloading data
    auto copyLambda = [](VkDeviceSize, auto input, auto output) {
        std::ranges::copy(input, output.begin());
    };

    // Step-by-step type deduction for future storage
    // Fn deduces as a reference because copyLambda is an lvalue
    using Fn = decltype(copyLambda)&;
    // Allocator type used by RecyclingStagingPool
    using Alloc = vko::vma::Allocator;
    // Exact future type returned by downloadTransform when passed an lvalue callable
    using Future = vko::DownloadTransformFuture<Fn, int, int, Alloc>;

    // Store futures to create overlapping GPU work
    std::vector<Future> futures;

    // Rapidly create/destroy multiple staging pools with overlapping GPU work
    for (int iteration = 0; iteration < 10; ++iteration) {
        auto future = [&]() {
            auto staging = vko::vma::RecyclingStagingPool<vko::Device>(
                ctx->device, ctx->allocator,
                /*minPools=*/2, /*maxPools=*/3, /*poolSize=*/1 << 14);
            vko::StagingStream streaming(queue, std::move(staging));

            // Upload unique data for this iteration
            vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize),
                        [iter = iteration](VkDeviceSize offset, auto span) {
                            for (size_t i = 0; i < span.size(); ++i) {
                                span[i] = static_cast<int>(iter * 100000 + offset + i);
                            }
                        });
            streaming.submit();

            // Download to verify data integrity later
            auto result = vko::download<int>(
                streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize), copyLambda);
            streaming.submit();

            // CRITICAL: Return future while staging pool destructs here with GPU work in-flight
            // Without proper synchronization, staging buffers are freed while GPU reads them
            return result;
        }(); // staging destructs here - MUST wait for GPU

        futures.push_back(std::move(future));
    }

    // Verify all futures complete successfully with correct data
    // If destructors didn't wait, we'd see corrupted data or device loss
    for (size_t iteration = 0; iteration < futures.size(); ++iteration) {
        auto& data = futures[iteration].get(ctx->device);
        ASSERT_EQ(data.size(), bufferSize);

        // Verify several values to detect corruption
        int expectedBase = static_cast<int>(iteration * 100000);
        EXPECT_EQ(data[0], expectedBase) << "Iteration " << iteration << " corrupted at index 0";
        EXPECT_EQ(data[100], expectedBase + 100)
            << "Iteration " << iteration << " corrupted at index 100";
        EXPECT_EQ(data[bufferSize - 1], expectedBase + bufferSize - 1)
            << "Iteration " << iteration << " corrupted at last index";
    }

    // Ensure all GPU work is complete before test ends
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Non-owning reference to staging resources
// Tests StagingStreamRef which holds references to separately-owned command buffer
// and staging allocator. This allows flexible sharing and eliminates lifetime issues
// since the ref has a trivial destructor.
TEST_F(UnitTestFixture, StagingStreamRef_BasicUploadDownload) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::CyclingCommandBuffer<vko::Device> cmdBuffer(ctx->device, queue);

    // Create a reference to the owned resources (following std::atomic_ref pattern)
    vko::StagingStreamRef<vko::vma::RecyclingStagingPool<vko::Device>> streamingRef(cmdBuffer,
                                                                                    staging);

    // Create GPU buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload via ref
    vko::upload(streamingRef, ctx->device, vko::BufferSpan(gpuBuffer),
                [](VkDeviceSize offset, auto span) {
                    std::iota(span.begin(), span.end(), static_cast<int>(offset));
                });
    streamingRef.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download via ref
    auto downloadFuture = vko::download(streamingRef, ctx->device, vko::BufferSpan(gpuBuffer));
    streamingRef.submit();

    auto& result = downloadFuture.get(ctx->device);
    EXPECT_EQ(result.size(), 1000);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }

    // Wait for all GPU work to complete before test cleanup
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: Multiple refs can share same resources
// Demonstrates that multiple StagingStreamRef instances can reference the same
// command buffer and staging allocator, useful for passing to different functions
// or subsystems without ownership transfer.
TEST_F(UnitTestFixture, StagingStreamRef_SharedResources) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::CyclingCommandBuffer<vko::Device> cmdBuffer(ctx->device, queue);

    // Create two refs to the same resources
    vko::StagingStreamRef<vko::vma::RecyclingStagingPool<vko::Device>> ref1(cmdBuffer, staging);
    vko::StagingStreamRef<vko::vma::RecyclingStagingPool<vko::Device>> ref2(cmdBuffer, staging);

    // Create two buffers
    auto buffer1 = vko::BoundBuffer<int>(
        ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    auto buffer2 = vko::BoundBuffer<int>(
        ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload using ref1
    vko::upload(
        ref1, ctx->device, vko::BufferSpan(buffer1),
        [](VkDeviceSize /*offset*/, auto span) { std::fill(span.begin(), span.end(), 100); });

    // Upload using ref2 (shares same command buffer and staging)
    vko::upload(
        ref2, ctx->device, vko::BufferSpan(buffer2),
        [](VkDeviceSize /*offset*/, auto span) { std::fill(span.begin(), span.end(), 200); });

    // Both uploads in same batch, submitted together
    ref1.submit();

    // Verify both resources are shared (same capacity)
    EXPECT_EQ(ref1.capacity(), ref2.capacity());
    EXPECT_GT(ref1.capacity(), 0u);

    // Wait for all GPU work to complete before test cleanup
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: BufferSpan with suballocation offset (e.g., accessing packed structs)
// Tests download() with BufferSpan using byte offset
TEST_F(UnitTestFixture, StagingStream_DownloadBufferSpanWithOffset) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer with header + data
    constexpr size_t headerSize = 16;   // bytes
    constexpr size_t dataSize   = 1000; // floats
    constexpr size_t totalSize  = headerSize + dataSize * sizeof(float);

    std::vector<float> sourceData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        sourceData[i] = static_cast<float>(i) * 2.5f;
    }

    auto buffer = vko::BoundBuffer<std::byte>(
        ctx->device, totalSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload header (zeros) + float data using BufferSpan
    vko::BufferSpan<std::byte> bufSpan(buffer);
    vko::upload(streaming, ctx->device, bufSpan,
                [&sourceData, headerSize](VkDeviceSize offset, std::span<std::byte> mapped) {
                    if (offset < headerSize) {
                        // Header region - write zeros
                        size_t headerBytes = std::min(mapped.size(), headerSize - offset);
                        std::memset(mapped.data(), 0, headerBytes);
                        if (mapped.size() > headerBytes) {
                            // Remainder is data
                            std::memcpy(mapped.data() + headerBytes,
                                        reinterpret_cast<const std::byte*>(sourceData.data()),
                                        mapped.size() - headerBytes);
                        }
                    } else {
                        // Data region
                        size_t dataOffset = offset - headerSize;
                        std::memcpy(mapped.data(),
                                    reinterpret_cast<const std::byte*>(sourceData.data()) +
                                        dataOffset,
                                    mapped.size());
                    }
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download just the data region using BufferSpan with byte offset
    vko::BufferAddress<const float> dataAddr(static_cast<VkBuffer>(buffer), headerSize);
    vko::BufferSpan<const float>    dataSpan(dataAddr, dataSize);
    auto                            future = vko::download(streaming, ctx->device, dataSpan);
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_FLOAT_EQ(result[i], sourceData[i]) << "Mismatch at index " << i;
    }
}

// Use-case: Type conversion with transform (e.g., format conversion or scaling)
// Tests download<DstT> with actual type conversion and transformation
TEST_F(UnitTestFixture, StagingStream_DownloadWithActualTypeConversion) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer with int32 data
    constexpr size_t     dataSize = 500;
    std::vector<int32_t> sourceData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        sourceData[i] = static_cast<int32_t>(i * 100);
    }

    auto buffer = vko::BoundBuffer<int32_t>(
        ctx->device, dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, sourceData, vko::BufferSpan(buffer));
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download with type conversion: int32_t → float (with scaling)
    auto future = vko::download<float>(
        streaming, ctx->device, vko::BufferSpan(buffer),
        [](VkDeviceSize, std::span<const int32_t> input, std::span<float> output) {
            // Convert int32_t to float with scaling
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = static_cast<float>(input[i]) / 100.0f;
            }
        });
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i)) << "Mismatch at index " << i;
    }
}

// Use-case: Type conversion with DeviceAddress (e.g., format conversion on GPU-visible memory)
// Tests download from DeviceAddress with type conversion
TEST_F(UnitTestFixture, StagingStream_DownloadDeviceAddressWithTypeConversion) {
    if (!ctx->device.vkCmdCopyMemoryIndirectNV) {
        GTEST_SKIP() << "VK_NV_copy_memory_indirect not available";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator, /*minPools=*/2, /*maxPools=*/4, /*poolSize=*/1 << 16,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create buffer with uint32_t data
    constexpr size_t      dataSize = 300;
    std::vector<uint32_t> sourceData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        sourceData[i] = static_cast<uint32_t>(i + 1000);
    }

    auto buffer = vko::BoundBuffer<uint32_t>(ctx->device, dataSize,
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    vko::upload(streaming, ctx->device, sourceData, vko::BufferSpan(buffer));
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Create DeviceSpan and download with type conversion: uint32_t → int64_t
    vko::DeviceAddress<uint32_t>    addr(buffer, ctx->device);
    vko::DeviceSpan<const uint32_t> span(vko::DeviceAddress<const uint32_t>(addr.raw()),
                                         buffer.size());

    auto future = vko::download<int64_t>(
        streaming, ctx->device, span,
        [](VkDeviceSize, std::span<const uint32_t> input, std::span<int64_t> output) {
            // Convert uint32_t to int64_t with offset
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = static_cast<int64_t>(input[i]) - 1000;
            }
        });
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        EXPECT_EQ(result[i], static_cast<int64_t>(i)) << "Mismatch at index " << i;
    }
}

// Test alignment edge cases with small primes for pool and buffer sizes
// This catches issues where align_up/align_down calculations don't match VMA's behavior
TEST_F(UnitTestFixture, RecyclingStagingPool_PrimeAlignmentEdgeCases) {
    // Small primes for pool sizes (bytes) - deliberately not powers of 2
    constexpr VkDeviceSize poolSizes[] = {17, 31, 37, 61, 67, 127};

    // Small primes for element counts
    constexpr size_t elemCounts[] = {1, 2, 3, 5, 7, 11, 13, 17, 19, 23};

    // Minimum allocation with typical 16-byte alignment is 16 bytes
    constexpr VkDeviceSize minPoolForOneFloat = 16;

    for (VkDeviceSize poolSize : poolSizes) {
        for (size_t elemCount : elemCounts) {
            SCOPED_TRACE("poolSize=" + std::to_string(poolSize) +
                         " elemCount=" + std::to_string(elemCount));

            // Fresh staging for each combo, single pool to test alignment at boundaries
            vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
            auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
                ctx->device, ctx->allocator, /*minPools=*/1, /*maxPools=*/1, poolSize);

            // Allocate same-size buffers until pool is exhausted
            size_t allocCount = 0;
            while (true) {
                auto* buffer = staging.allocateUpTo<float>(elemCount);
                if (!buffer)
                    break; // Pool exhausted

                allocCount++;

                // Verify allocation size is valid
                EXPECT_GT(buffer->size(), 0);
                EXPECT_LE(buffer->size(), elemCount);
            }

            // With sufficient pool size and minimum element count, must allocate
            if (poolSize >= minPoolForOneFloat && elemCount == 1) {
                EXPECT_GT(allocCount, 0)
                    << "Pool of " << poolSize << " bytes should fit at least one float";
            }
        }
    }
}

// Test makeTmpBufferPair alignment edge cases specifically
// For each (poolSize, upToSize) combo, allocate same-size buffers until pool is exhausted
// This ensures we hit the alignment boundary for that specific size
TEST_F(UnitTestFixture, RecyclingStagingPool_PairAllocationPrimeEdgeCases) {
    // Pool sizes chosen to create tight alignment scenarios
    constexpr VkDeviceSize poolSizes[] = {33, 47, 63, 65, 97};
    // Small primes for upTo element counts
    constexpr size_t upToSizes[] = {1, 2, 3, 5, 7, 11, 13};

    // Minimum pair allocation with 16-byte alignment: 16 (uint32_t) + 16 (float) = 32 bytes
    constexpr VkDeviceSize minPoolForOnePair = 32;

    for (VkDeviceSize poolSize : poolSizes) {
        for (size_t upToSize : upToSizes) {
            SCOPED_TRACE("poolSize=" + std::to_string(poolSize) +
                         " upToSize=" + std::to_string(upToSize));

            vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
            auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
                ctx->device, ctx->allocator, /*minPools=*/1, /*maxPools=*/1, poolSize);

            // Allocate same-size buffers until pool is exhausted
            size_t allocCount = 0;
            while (true) {
                auto result = staging.allocateSingleAndUpTo<uint32_t, float>(upToSize);
                if (!result.has_value())
                    break; // Pool exhausted

                auto& [single, upTo] = *result;
                allocCount++;

                // Verify allocation sizes are valid
                EXPECT_EQ(single.size(), 1);
                EXPECT_GT(upTo.size(), 0);
                EXPECT_LE(upTo.size(), upToSize);
            }

            // With sufficient pool size and minimum upTo count, must allocate
            if (poolSize >= minPoolForOnePair && upToSize == 1) {
                EXPECT_GT(allocCount, 0)
                    << "Pool of " << poolSize << " bytes should fit at least one pair";
            }
        }
    }
}

// Use-case: DedicatedStagingPool with allocateSingleAndUpTo
// Tests that the method always succeeds and allocates full requested size
TEST_F(UnitTestFixture, DedicatedStagingPool_AllocateSingleAndUpTo_Basic) {
    vko::DedicatedStagingPool<vko::Device, vko::vma::Allocator> staging(ctx->device,
                                                                        ctx->allocator);

    constexpr size_t upToSize = 100;
    auto             result   = staging.allocateSingleAndUpTo<uint32_t, float>(upToSize);

    ASSERT_TRUE(result.has_value()) << "DedicatedStagingPool should always succeed";

    auto& [single, upTo] = *result;

    // Verify single element buffer
    EXPECT_EQ(single.size(), 1u);

    // DedicatedStagingPool always allocates full size (AllocateAlwaysFull = true)
    EXPECT_EQ(upTo.size(), upToSize);

    // Verify buffers are usable
    single.map()[0] = 42;
    EXPECT_EQ(single.map()[0], 42u);

    for (size_t i = 0; i < upTo.size(); ++i) {
        upTo.map()[i] = static_cast<float>(i);
    }
    EXPECT_EQ(upTo.map()[50], 50.0f);

    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();
}

// Use-case: StagingStream with DedicatedStagingPool backend
// Tests that StagingStream works correctly with DedicatedStagingPool,
// useful for one-off large transfers during initialization.
TEST_F(UnitTestFixture, StagingStream_DedicatedStagingPool_BasicUpload) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::DedicatedStagingPool<vko::Device, vko::vma::Allocator>(ctx->device, ctx->allocator);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create GPU buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload with callback
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 1000),
                [](VkDeviceSize offset, auto span) {
                    std::iota(span.begin(), span.end(), static_cast<int>(offset));
                });

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: StagingStream with DedicatedStagingPool using withSingleAndStagingBuffer
// Tests atomic paired allocation for indirect copy commands with dedicated staging
TEST_F(UnitTestFixture, StagingStream_DedicatedStagingPool_AtomicPairAllocation) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::DedicatedStagingPool<vko::Device, vko::vma::Allocator>(ctx->device, ctx->allocator);
    vko::StagingStream streaming(queue, std::move(staging));

    constexpr size_t dataSize  = 1000;
    int              callCount = 0;

    streaming.withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::byte>(
        dataSize,
        [&](VkCommandBuffer, auto& cmd, auto& data,
            VkDeviceSize) -> std::optional<std::function<void(bool)>> {
            callCount++;
            EXPECT_EQ(cmd.size(), 1u);
            // DedicatedStagingPool always allocates full size
            EXPECT_EQ(data.size(), dataSize);
            return std::nullopt;
        });

    // DedicatedStagingPool always succeeds with full allocation, so only one call
    EXPECT_EQ(callCount, 1);

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));
}

// Use-case: StagingStream with DedicatedStagingPool download roundtrip
// Verifies complete upload/download cycle with dedicated staging
TEST_F(UnitTestFixture, StagingStream_DedicatedStagingPool_UploadDownloadRoundtrip) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::DedicatedStagingPool<vko::Device, vko::vma::Allocator>(ctx->device, ctx->allocator);
    vko::StagingStream streaming(queue, std::move(staging));

    // Create GPU buffer
    auto gpuBuffer = vko::BoundBuffer<int>(
        ctx->device, 500, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Upload known pattern: value at index i = i * 2
    vko::upload(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 500),
                [](VkDeviceSize offset, auto span) {
                    for (size_t i = 0; i < span.size(); ++i) {
                        span[i] = static_cast<int>((offset + i) * 2);
                    }
                });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Download and verify
    auto future = vko::download(streaming, ctx->device, vko::BufferSpan(gpuBuffer).subspan(0, 500));
    streaming.submit();

    auto& result = future.get(ctx->device);
    ASSERT_EQ(result.size(), 500u);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i * 2)) << "Mismatch at index " << i;
    }
}

// Use-case: Multiple sequential allocateSingleAndUpTo calls
// Tests that DedicatedStagingPool handles multiple paired allocations
TEST_F(UnitTestFixture, DedicatedStagingPool_AllocateSingleAndUpTo_Multiple) {
    vko::DedicatedStagingPool<vko::Device, vko::vma::Allocator> staging(ctx->device,
                                                                        ctx->allocator);

    // Allocate multiple pairs
    std::vector<std::pair<uint32_t, size_t>> pairs; // (single value, upTo size)

    for (size_t i = 0; i < 5; ++i) {
        size_t upToSize = 10 + i * 10;
        auto   result   = staging.allocateSingleAndUpTo<uint32_t, float>(upToSize);
        ASSERT_TRUE(result.has_value());

        auto& [single, upTo] = *result;
        single.map()[0]      = static_cast<uint32_t>(i);
        EXPECT_EQ(upTo.size(), upToSize);

        pairs.emplace_back(i, upToSize);
    }

    // Verify size accounting
    EXPECT_GT(staging.size(), 0u);

    staging.endBatch(vko::SemaphoreValue::makeSignalled());
    staging.wait();

    // After wait, size should return to 0
    EXPECT_EQ(staging.size(), 0u);
}

// Verify that dropping a DownloadForEach future does NOT immediately invoke the callback.
// The callback's State survives in the staging pool - it's not called until batch processing.
TEST_F(UnitTestFixture, StagingStream_DroppedForEachFutureDoesNotImmediatelyInvoke) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2, 4096);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<int>(ctx->device, 100, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    std::atomic<size_t> callbackCount{0};

    // Start download and IMMEDIATELY drop the future
    {
        auto future =
            vko::downloadForEach(streaming, ctx->device, vko::BufferSpan(buffer),
                                 [&](VkDeviceSize, std::span<const int>) { callbackCount++; });
    }

    // Callback should NOT have been invoked - dropping the future doesn't trigger it
    EXPECT_EQ(callbackCount.load(), 0u) << "Callback invoked prematurely when future dropped";

    // Submit and wait - still shouldn't invoke (sem.wait doesn't process staging callbacks)
    streaming.submit().wait(ctx->device);

    EXPECT_EQ(callbackCount.load(), 0u) << "Callback invoked by sem.wait()";

    // poll() should invoke the callback (GPU work is complete)
    streaming.poll();
    EXPECT_EQ(callbackCount.load(), 1u) << "Callback NOT invoked by stream.poll()";
}

// Verify stream.wait() invokes callbacks for dropped futures
TEST_F(UnitTestFixture, StagingStream_WaitInvokesDroppedForEachCallback) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2, 4096);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<int>(ctx->device, 100, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    std::atomic<size_t> callbackCount{0};

    // Start download and drop the future
    {
        auto future =
            vko::downloadForEach(streaming, ctx->device, vko::BufferSpan(buffer),
                                 [&](VkDeviceSize, std::span<const int>) { callbackCount++; });
    }

    streaming.submit();

    EXPECT_EQ(callbackCount.load(), 0u) << "Callback invoked before wait()";

    // wait() should block until complete AND invoke callbacks
    streaming.wait();
    EXPECT_EQ(callbackCount.load(), 1u) << "Callback NOT invoked by stream.wait()";
}

// Verify future.wait() invokes its own callback
TEST_F(UnitTestFixture, StagingStream_FutureWaitInvokesCallback) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator, 1, 2, 4096);
    vko::StagingStream streaming(queue, std::move(staging));

    auto buffer = vko::BoundBuffer<int>(ctx->device, 100, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    std::atomic<size_t> callbackCount{0};

    auto future =
        vko::downloadForEach(streaming, ctx->device, vko::BufferSpan(buffer),
                             [&](VkDeviceSize, std::span<const int>) { callbackCount++; });

    streaming.submit();

    EXPECT_EQ(callbackCount.load(), 0u) << "Callback invoked before future.wait()";

    // future.wait(device) should invoke the callback
    future.wait(ctx->device);
    EXPECT_EQ(callbackCount.load(), 1u) << "Callback NOT invoked by future.wait()";
}

// Comprehensive test for all free-function upload/download overloads with non-zero offsets.
// Uses small pool size to force chunking, verifying offset handling across chunk boundaries.
TEST_F(UnitTestFixture, FreeFunctionOverloads_OffsetHandling) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    // Small pool size (64 bytes = 16 ints) to force chunking, many pools for speed
    // Include SHADER_DEVICE_ADDRESS_BIT for DeviceSpan operations using indirect copy
    constexpr VkDeviceSize poolSize = 64;
    auto                   staging  = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/50, poolSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream streaming(queue, std::move(staging));

    // Buffer layout: 100 elements, we'll write to different regions
    constexpr size_t bufferSize = 100;
    constexpr int    sentinel   = -1;

    auto buffer =
        vko::BoundBuffer<int>(ctx->device, bufferSize,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Pre-fill buffer with sentinel value
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize),
                [=](VkDeviceSize, auto span) { std::ranges::fill(span, sentinel); });
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // Test regions: each overload writes to a separate 10-element region
    // Region 0: offset 0-9   (upload: buffer+offset+size+fn)
    // Region 1: offset 10-19 (upload: buffer+range+offset)
    // Region 2: offset 20-29 (upload: BufferSpan+fn with subspan)
    // Region 3: offset 30-39 (upload: BufferSpan+range with subspan)
    // Region 4: offset 40-49 (upload: DeviceSpan+size+fn)
    // Region 5: offset 50-59 (upload: DeviceSpan+range)
    // Region 6: offset 60-69 (upload: allocator+size+fn - separate buffer)
    // Region 7: offset 70-79 (upload: allocator+range - separate buffer)
    // Remaining 80-99: stays sentinel for boundary checks

    constexpr size_t regionSize = 10;

    // Marker values: region N gets values N*100 + index
    auto makeMarker = [](int region, size_t idx) { return region * 100 + static_cast<int>(idx); };

    // ===== UPLOAD TESTS =====

    // 1. upload(stream, device, BufferSpan<T>, fn) - region 0
    vko::upload(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, regionSize),
                [&](VkDeviceSize off, std::span<int> span) {
                    for (size_t i = 0; i < span.size(); ++i)
                        span[i] = makeMarker(0, off + i);
                });

    // 2. upload(stream, device, BufferSpan<T>, range) - region 1
    std::vector<int> region1Data(regionSize);
    for (size_t i = 0; i < regionSize; ++i)
        region1Data[i] = makeMarker(1, i);
    vko::upload(streaming, ctx->device, region1Data,
                vko::BufferSpan(buffer).subspan(10, regionSize));

    // 3. upload(stream, device, BufferSpan<T>, fn) with subspan offset
    vko::BufferSpan<int> bufSpan2 = vko::BufferSpan(buffer).subspan(20, regionSize);
    vko::upload(streaming, ctx->device, bufSpan2, [&](VkDeviceSize off, std::span<int> span) {
        for (size_t i = 0; i < span.size(); ++i)
            span[i] = makeMarker(2, off + i);
    });

    // 4. upload(stream, device, BufferSpan<T>, range) with subspan offset
    std::vector<int> region3Data(regionSize);
    for (size_t i = 0; i < regionSize; ++i)
        region3Data[i] = makeMarker(3, i);
    vko::BufferSpan<int> bufSpan3 = vko::BufferSpan(buffer).subspan(30, regionSize);
    vko::upload(streaming, ctx->device, region3Data, bufSpan3);

    // Submit uploads so far
    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // DeviceSpan overloads require VK_NV_copy_memory_indirect
    bool hasIndirectCopy = ctx->device.vkCmdCopyMemoryIndirectNV != nullptr;

    if (hasIndirectCopy) {
        // 5. upload(stream, device, DeviceSpan, size, fn) with subspan offset
        vko::DeviceAddress<int> bufAddr(buffer, ctx->device);
        vko::DeviceSpan<int>    devSpan4 =
            vko::DeviceSpan<int>(bufAddr, bufferSize).subspan(40, regionSize);
        vko::upload(streaming, ctx->device, devSpan4, VkDeviceSize{regionSize},
                    [&](VkDeviceSize off, std::span<int> span) {
                        for (size_t i = 0; i < span.size(); ++i)
                            span[i] = makeMarker(4, off + i);
                    });

        // 6. upload(stream, device, DeviceSpan, range) with subspan offset
        std::vector<int> region5Data(regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            region5Data[i] = makeMarker(5, i);
        vko::DeviceSpan<int> devSpan5 =
            vko::DeviceSpan<int>(bufAddr, bufferSize).subspan(50, regionSize);
        vko::upload(streaming, ctx->device, region5Data, devSpan5);

        streaming.submit();
        vko::check(ctx->device.vkQueueWaitIdle(queue));
    }

    // 7 & 8: Allocator overloads create new buffers, test separately
    std::vector<int> region6Data(regionSize);
    for (size_t i = 0; i < regionSize; ++i)
        region6Data[i] = makeMarker(6, i);
    auto allocBuffer6 =
        vko::upload<int>(streaming, ctx->device, ctx->allocator, VkDeviceSize{regionSize},
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         [&](VkDeviceSize off, std::span<int> span) {
                             for (size_t i = 0; i < span.size(); ++i)
                                 span[i] = makeMarker(6, off + i);
                         });

    std::vector<int> region7Data(regionSize);
    for (size_t i = 0; i < regionSize; ++i)
        region7Data[i] = makeMarker(7, i);
    auto allocBuffer7 =
        vko::upload(streaming, ctx->device, ctx->allocator, region7Data,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    streaming.submit();
    vko::check(ctx->device.vkQueueWaitIdle(queue));

    // ===== VERIFY UPLOADS via raw download =====
    auto verifyFuture =
        vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, bufferSize));
    streaming.submit();
    auto& fullData = verifyFuture.get(ctx->device);

    // Check region 0
    for (size_t i = 0; i < regionSize; ++i)
        EXPECT_EQ(fullData[0 + i], makeMarker(0, i)) << "Region 0 mismatch at " << i;

    // Check region 1
    for (size_t i = 0; i < regionSize; ++i)
        EXPECT_EQ(fullData[10 + i], makeMarker(1, i)) << "Region 1 mismatch at " << i;

    // Check region 2
    for (size_t i = 0; i < regionSize; ++i)
        EXPECT_EQ(fullData[20 + i], makeMarker(2, i)) << "Region 2 mismatch at " << i;

    // Check region 3
    for (size_t i = 0; i < regionSize; ++i)
        EXPECT_EQ(fullData[30 + i], makeMarker(3, i)) << "Region 3 mismatch at " << i;

    if (hasIndirectCopy) {
        // Check region 4
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(fullData[40 + i], makeMarker(4, i)) << "Region 4 mismatch at " << i;

        // Check region 5
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(fullData[50 + i], makeMarker(5, i)) << "Region 5 mismatch at " << i;
    }

    // Check sentinel regions (60-99, or 40-99 if no indirect copy)
    size_t sentinelStart = hasIndirectCopy ? 60 : 40;
    for (size_t i = sentinelStart; i < bufferSize; ++i)
        EXPECT_EQ(fullData[i], sentinel) << "Sentinel corrupted at " << i;

    // Verify allocator buffers
    auto alloc6Future = vko::download(streaming, ctx->device, vko::BufferSpan(allocBuffer6));
    auto alloc7Future = vko::download(streaming, ctx->device, vko::BufferSpan(allocBuffer7));
    streaming.submit();

    auto& alloc6Data = alloc6Future.get(ctx->device);
    auto& alloc7Data = alloc7Future.get(ctx->device);
    for (size_t i = 0; i < regionSize; ++i) {
        EXPECT_EQ(alloc6Data[i], makeMarker(6, i)) << "Alloc buffer 6 mismatch at " << i;
        EXPECT_EQ(alloc7Data[i], makeMarker(7, i)) << "Alloc buffer 7 mismatch at " << i;
    }

    // ===== DOWNLOAD OVERLOAD TESTS =====
    // Use region 0 data (values 0-9) for download tests

    // download(stream, device, BufferSpan<T>) - identity
    {
        auto future =
            vko::download(streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, regionSize));
        streaming.submit();
        auto& result = future.get(ctx->device);
        ASSERT_EQ(result.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(result[i], makeMarker(0, i)) << "download(BufferSpan) mismatch at " << i;
    }

    // download<DstT>(stream, device, BufferSpan<SrcT>, fn) - transform
    {
        auto future = vko::download<float>(
            streaming, ctx->device, vko::BufferSpan(buffer).subspan(0, regionSize),
            [](VkDeviceSize, std::span<const int> in, std::span<float> out) {
                for (size_t i = 0; i < in.size(); ++i)
                    out[i] = static_cast<float>(in[i]) + 0.5f;
            });
        streaming.submit();
        auto& result = future.get(ctx->device);
        ASSERT_EQ(result.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_FLOAT_EQ(result[i], makeMarker(0, i) + 0.5f)
                << "download<DstT>(BufferSpan,fn) mismatch at " << i;
    }

    // 11. download(stream, device, BufferSpan<T>) - identity with subspan offset
    {
        vko::BufferSpan<const int> srcSpan = vko::BufferSpan(buffer).subspan(10, regionSize);
        auto                       future  = vko::download(streaming, ctx->device, srcSpan);
        streaming.submit();
        auto& result = future.get(ctx->device);
        ASSERT_EQ(result.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(result[i], makeMarker(1, i)) << "download(BufferSpan) mismatch at " << i;
    }

    // 12. download<DstT>(stream, device, BufferSpan<SrcT>, fn) - transform with subspan
    {
        vko::BufferSpan<const int> srcSpan = vko::BufferSpan(buffer).subspan(20, regionSize);
        auto                       future =
            vko::download<float>(streaming, ctx->device, srcSpan,
                                 [](VkDeviceSize, std::span<const int> in, std::span<float> out) {
                                     for (size_t i = 0; i < in.size(); ++i)
                                         out[i] = static_cast<float>(in[i]) + 0.25f;
                                 });
        streaming.submit();
        auto& result = future.get(ctx->device);
        ASSERT_EQ(result.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_FLOAT_EQ(result[i], makeMarker(2, i) + 0.25f)
                << "download<DstT>(BufferSpan,fn) mismatch at " << i;
    }

    // downloadForEach(stream, device, BufferSpan<T>, fn) - streaming with subspan
    {
        std::vector<int> collected;
        auto             handle = vko::downloadForEach(streaming, ctx->device,
                                                       vko::BufferSpan(buffer).subspan(30, regionSize),
                                                       [&](VkDeviceSize, std::span<const int> data) {
                                               for (size_t i = 0; i < data.size(); ++i)
                                                   collected.push_back(data[i]);
                                           });
        streaming.submit();
        handle.wait(ctx->device);
        ASSERT_EQ(collected.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(collected[i], makeMarker(3, i))
                << "downloadForEach(BufferSpan) mismatch at " << i;
    }

    // 16. downloadForEach(stream, device, BufferSpan<T>, fn) - streaming with subspan
    {
        vko::BufferSpan<const int> srcSpan = vko::BufferSpan(buffer).subspan(0, regionSize);
        std::vector<int>           collected;
        auto                       handle = vko::downloadForEach(streaming, ctx->device, srcSpan,
                                                                 [&](VkDeviceSize, std::span<const int> data) {
                                               for (size_t i = 0; i < data.size(); ++i)
                                                   collected.push_back(data[i]);
                                           });
        streaming.submit();
        handle.wait(ctx->device);
        ASSERT_EQ(collected.size(), regionSize);
        for (size_t i = 0; i < regionSize; ++i)
            EXPECT_EQ(collected[i], makeMarker(0, i))
                << "downloadForEach(BufferSpan) mismatch at " << i;
    }

    if (hasIndirectCopy) {
        vko::DeviceAddress<int> bufAddr(buffer, ctx->device);

        // 13. download(stream, device, DeviceSpan) - identity with subspan
        {
            vko::DeviceSpan<const int> srcSpan =
                vko::DeviceSpan<const int>(vko::DeviceAddress<const int>(bufAddr.raw()), bufferSize)
                    .subspan(40, regionSize);
            auto future = vko::download(streaming, ctx->device, srcSpan);
            streaming.submit();
            auto& result = future.get(ctx->device);
            ASSERT_EQ(result.size(), regionSize);
            for (size_t i = 0; i < regionSize; ++i)
                EXPECT_EQ(result[i], makeMarker(4, i)) << "download(DeviceSpan) mismatch at " << i;
        }

        // 14. download<DstT>(stream, device, DeviceSpan, fn) - transform with subspan
        {
            vko::DeviceSpan<const int> srcSpan =
                vko::DeviceSpan<const int>(vko::DeviceAddress<const int>(bufAddr.raw()), bufferSize)
                    .subspan(50, regionSize);
            auto future = vko::download<float>(
                streaming, ctx->device, srcSpan,
                [](VkDeviceSize, std::span<const int> in, std::span<float> out) {
                    for (size_t i = 0; i < in.size(); ++i)
                        out[i] = static_cast<float>(in[i]) + 0.75f;
                });
            streaming.submit();
            auto& result = future.get(ctx->device);
            ASSERT_EQ(result.size(), regionSize);
            for (size_t i = 0; i < regionSize; ++i)
                EXPECT_FLOAT_EQ(result[i], makeMarker(5, i) + 0.75f)
                    << "download<DstT>(DeviceSpan,fn) mismatch at " << i;
        }

        // 17. downloadForEach(stream, device, DeviceSpan, fn) - streaming with subspan
        {
            vko::DeviceSpan<const int> srcSpan =
                vko::DeviceSpan<const int>(vko::DeviceAddress<const int>(bufAddr.raw()), bufferSize)
                    .subspan(40, regionSize);
            std::vector<int> collected;
            auto             handle = vko::downloadForEach(streaming, ctx->device, srcSpan,
                                                           [&](VkDeviceSize, std::span<const int> data) {
                                                   for (size_t i = 0; i < data.size(); ++i)
                                                       collected.push_back(data[i]);
                                               });
            streaming.submit();
            handle.wait(ctx->device);
            ASSERT_EQ(collected.size(), regionSize);
            for (size_t i = 0; i < regionSize; ++i)
                EXPECT_EQ(collected[i], makeMarker(4, i))
                    << "downloadForEach(DeviceSpan) mismatch at " << i;
        }
    }
}
