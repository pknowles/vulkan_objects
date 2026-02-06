// Copyright (c) 2026 Pyarelal Knowles, MIT License
// Minimal reproduction case for StagingStream_ConcurrentDownloads data corruption
//
// =============================================================================
// DEBUGGING PROGRESS
// =============================================================================
//
// RULED OUT:
// - VMA pool reuse mechanics: This MWE passes 100% with identical allocation patterns
// - Alignment calculations: Chunk sizes match exactly between MWE and original
// - Race conditions: Single-threaded, vkQueueWaitIdle after every copy
// - Semaphore ordering: Tracker verifies semaphores signaled before markProcessed
// - Range tracking: Tracker verifies exact byte ranges match track↔processed
// - Staging buffer lifetime: Tracker catches premature free
// - Staging buffer reuse: Tracker catches reuse before markProcessed
// - Mapping pointer lifetime: Tracker verifies mapping valid when accessed
// - VMA pointer aliasing: Tracker catches mapping same pointer without unmap
//
// WHAT MWE DOESN'T HAVE (remaining suspects):
// - Upload phase before downloads: Original uploads then downloads, pools have history
// - StagingStream's future/callback machinery: downloadTransform, downloadForEach
// - Lambda/closure captures: Something captured by & that changes?
//
// KEY CLUE:
// Corrupted data comes from OTHER chunks of OTHER downloads, SAME iteration.
// Values are often offset by 4096 bytes (page size!) which is suspicious given
// download sizes are weird primes (6763 elements = 27052 bytes).
//
// NEXT STEPS:
// - Try MWE with upload phase first (match original test's pool history)
// - Add verifyUserRangeFromHostPtr() to tracker for callback verification
// - Try original test with uploads replaced by vkCmdFill (isolate download bugs)
//
// =============================================================================

#include <cstring>
#include <deque>
#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/allocator.hpp>

// Helper to track non-contiguous bad ranges and print a summary
struct BadRangeTracker {
    struct Range {
        size_t start, end;
    };
    std::vector<Range> ranges;
    size_t             firstBadIdx   = 0;
    uint32_t           firstExpected = 0;
    uint32_t           firstGot      = 0;

    void add(size_t idx, uint32_t expected, uint32_t got) {
        if (ranges.empty()) {
            firstBadIdx   = idx;
            firstExpected = expected;
            firstGot      = got;
        }
        // Try to extend last range
        if (!ranges.empty() && ranges.back().end == idx) {
            ranges.back().end = idx + 1;
            return;
        }
        // Start new range
        ranges.push_back({idx, idx + 1});
    }

    bool empty() const { return ranges.empty(); }

    size_t totalBad() const {
        size_t total = 0;
        for (const auto& r : ranges)
            total += r.end - r.start;
        return total;
    }

    std::string summary() const {
        std::string s;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (i > 0)
                s += ", ";
            if (i >= 5) {
                s += "...";
                break;
            } // Limit output
            s += "[" + std::to_string(ranges[i].start) + "," + std::to_string(ranges[i].end) + ")";
        }
        return s;
    }

    std::string firstValue() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "idx=%zu expected=0x%08X got=0x%08X", firstBadIdx, firstExpected,
                 firstGot);
        return buf;
    }
};

// Test parameters - EXACTLY matching StagingStream_ConcurrentDownloads
constexpr VkDeviceSize downloadSize     = 1697; // elements, not bytes!
constexpr size_t       numDownloads     = 20;
constexpr size_t       poolCycleCount   = 3;
constexpr size_t       maxPools         = 10;
constexpr size_t       numIterations    = 20;
constexpr VkDeviceSize bytesPerDownload = downloadSize * sizeof(uint32_t); // 27052 bytes
// Pool size: EXACTLY as original (uses element count, not bytes - quirky but matches)
constexpr VkDeviceSize poolSize =
    (bytesPerDownload * numDownloads) / (maxPools * poolCycleCount) + 2957; // 7465 bytes

// ALIGNMENT OVERRIDE: Set to 0 to use driver-reported alignment, or a power of 2 to force higher
// Try: 64 (cache line), 256, 4096 (page), etc. to see if bug disappears
constexpr VkDeviceSize ALIGNMENT_OVERRIDE = 0;

// Hash function matching original test (with iteration=0 since we only upload once)
inline uint32_t humanValueHash(size_t iteration, size_t download, VkDeviceSize offset, size_t i) {
    iteration = 0; // Match original test: only upload on first iteration
    return static_cast<uint32_t>(iteration * 10000000 + download * 100000 + offset + i);
}

// Device address range tracker to detect VMA memory aliasing
struct DeviceAddressRangeTracker {
    struct Range {
        VkDeviceMemory memory{};
        VkDeviceSize   offset{};
        VkDeviceSize   size{};
        VkBuffer       buffer{}; // For debug output
    };

    std::vector<Range> activeRanges;

    void add(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkBuffer buffer) {
        VkDeviceSize newEnd = offset + size;

        // Check for overlaps with existing ranges in same memory
        for (const auto& r : activeRanges) {
            if (r.memory != memory)
                continue;

            VkDeviceSize existingEnd = r.offset + r.size;
            // Overlap if: new.start < existing.end AND new.end > existing.start
            if (offset < existingEnd && newEnd > r.offset) {
                printf("BUG: Device memory OVERLAP detected!\n");
                printf("    New:      [%zu, %zu) buffer=%p\n", static_cast<size_t>(offset),
                       static_cast<size_t>(newEnd), static_cast<void*>(buffer));
                printf("    Existing: [%zu, %zu) buffer=%p\n", static_cast<size_t>(r.offset),
                       static_cast<size_t>(existingEnd), static_cast<void*>(r.buffer));
                throw std::runtime_error("Device memory overlap detected");
            }
        }

        activeRanges.push_back({memory, offset, size, buffer});
    }

    void remove(VkBuffer buffer) {
        auto it = std::find_if(activeRanges.begin(), activeRanges.end(),
                               [buffer](const Range& r) { return r.buffer == buffer; });
        if (it != activeRanges.end()) {
            activeRanges.erase(it);
        }
    }

    void clear() { activeRanges.clear(); }
};

// Finally figured out this was my GPU - maybe VRAM, cache, trasnfer, MB even?
TEST_F(UnitTestFixture, DISABLED_StagingPoolReuseBugReproVMA) {
    VmaAllocator vma = ctx->allocator;

    auto    cmdPool = ctx->createCommandPool();
    VkQueue queue{};
    ctx->device.vkGetDeviceQueue(ctx->device, ctx->queueFamilyIndex, 0, &queue);

    // Create VMA pools with LINEAR algorithm - use a deque to match queue behavior
    std::deque<VmaPool> pools;

    VkBufferCreateInfo sampleBufferInfo = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0,
        .size        = poolSize,
        .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    VmaAllocationCreateInfo sampleAllocInfo = {
        .flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage         = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = VK_NULL_HANDLE,
        .pUserData      = nullptr,
        .priority       = 0.0f,
    };
    uint32_t memTypeIndex = 0;
    ASSERT_EQ(VK_SUCCESS, vmaFindMemoryTypeIndexForBufferInfo(vma, &sampleBufferInfo,
                                                              &sampleAllocInfo, &memTypeIndex));

    // Query alignment requirement (same as RecyclingStagingPool)
    VkDeviceSize alignment = 1;
    {
        VkBufferCreateInfo tempInfo = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .size        = 1,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VkBuffer tempBuffer{};
        ASSERT_EQ(VK_SUCCESS,
                  ctx->device.vkCreateBuffer(ctx->device, &tempInfo, nullptr, &tempBuffer));
        VkMemoryRequirements req{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, tempBuffer, &req);
        alignment = req.alignment;
        if constexpr (ALIGNMENT_OVERRIDE > 0) {
            alignment = std::max(alignment, ALIGNMENT_OVERRIDE);
        }
        ctx->device.vkDestroyBuffer(ctx->device, tempBuffer, nullptr);
    }

    // Query physical device limits for copy alignment
    VkPhysicalDeviceProperties props{};
    ctx->instance.vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    VkDeviceSize optimalCopyAlign = props.limits.optimalBufferCopyOffsetAlignment;
    (void)props.limits.nonCoherentAtomSize; // Available if needed for debugging

    auto alignUp = [](VkDeviceSize value, VkDeviceSize align) {
        return (value + align - 1) / align * align;
    };

    // Full memory barrier hammer - sprinkle EVERYWHERE to rule out sync issues
    auto memoryBarrierHammer = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier2 barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext         = nullptr,
            .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        };
        VkDependencyInfo depInfo = {
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext                    = nullptr,
            .dependencyFlags          = 0,
            .memoryBarrierCount       = 1,
            .pMemoryBarriers          = &barrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers    = nullptr,
            .imageMemoryBarrierCount  = 0,
            .pImageMemoryBarriers     = nullptr,
        };
        ctx->device.vkCmdPipelineBarrier2(cmd, &depInfo);
    };

    for (size_t i = 0; i < maxPools; ++i) {
        VmaPoolCreateInfo poolCreateInfo = {
            .memoryTypeIndex        = memTypeIndex,
            .flags                  = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,
            .blockSize              = poolSize,
            .minBlockCount          = 1,
            .maxBlockCount          = 1,
            .priority               = 0.0f,
            .minAllocationAlignment = 0,
            .pMemoryAllocateNext    = nullptr,
        };
        VmaPool pool;
        ASSERT_EQ(VK_SUCCESS, vmaCreatePool(vma, &poolCreateInfo, &pool));
        pools.push_back(pool);
    }

    // Create GPU buffers (device-local)
    struct GpuBuffer {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };
    std::vector<GpuBuffer> gpuBuffers(numDownloads);

    for (size_t i = 0; i < numDownloads; ++i) {
        VkBufferCreateInfo bufferInfo = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .size        = bytesPerDownload,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VmaAllocationCreateInfo allocInfo = {
            .flags          = 0,
            .usage          = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = nullptr,
            .priority       = 0.0f,
        };
        ASSERT_EQ(VK_SUCCESS, vmaCreateBuffer(vma, &bufferInfo, &allocInfo, &gpuBuffers[i].buffer,
                                              &gpuBuffers[i].allocation, nullptr));
    }

    // Staging buffer wrapper
    struct StagingBuffer {
        VkBuffer          buffer      = VK_NULL_HANDLE;
        VmaAllocation     allocation  = nullptr;
        VmaAllocationInfo allocInfo   = {};
        size_t            downloadIdx = 0;
        VkDeviceSize      srcOffset   = 0;
        VkDeviceSize      size        = 0;
    };

    // Track pool usage (LINEAR algorithm - bump allocator)
    std::vector<VkDeviceSize> poolUsedBytes(maxPools, 0);

    // Pending staging buffers awaiting verification
    std::vector<StagingBuffer> pending;

    // Track device memory ranges to detect VMA aliasing
    DeviceAddressRangeTracker deviceRangeTracker;

    // Run iterations
    size_t currentPoolIdx = 0;

    // Lambda to free staging buffers without verification (for uploads)
    auto freeUploadPending = [&](std::vector<StagingBuffer>& uploadPending) {
        // Sanity check: verify staging buffer mappings are correct BEFORE freeing
        for (const auto& staging : uploadPending) {
            const volatile uint32_t* volatileData =
                static_cast<const volatile uint32_t*>(staging.allocInfo.pMappedData);
            size_t       numElements   = staging.size / sizeof(uint32_t);
            VkDeviceSize elementOffset = staging.srcOffset / sizeof(uint32_t);

            for (size_t j = 0; j < numElements; ++j) {
                uint32_t expected = humanValueHash(0, staging.downloadIdx, elementOffset, j);
                uint32_t actual   = volatileData[j];
                if (actual != expected) {
                    ADD_FAILURE() << "Upload staging buffer corrupted!"
                                  << " buf=" << staging.downloadIdx
                                  << " srcOffset=" << staging.srcOffset << " offset=" << j
                                  << " expected=" << expected << " got=" << actual;
                    break;
                }
            }
        }

        for (auto& staging : uploadPending) {
            deviceRangeTracker.remove(staging.buffer);
            vmaDestroyBuffer(vma, staging.buffer, staging.allocation);
        }
        uploadPending.clear();
        std::fill(poolUsedBytes.begin(), poolUsedBytes.end(), 0);
        std::reverse(pools.begin(), pools.end());
        currentPoolIdx = 0;
    };
    for (size_t iter = 0; iter < numIterations; ++iter) {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        // Upload GPU buffers using staging (only on first iteration, matching original)
        if (iter == 0) {
            std::vector<StagingBuffer> uploadPending;
            std::vector<VkDeviceSize>  uploadProgress(numDownloads, 0);

            for (size_t di = 0; di < numDownloads; ++di) {
                while (uploadProgress[di] < bytesPerDownload) {
                    VkDeviceSize remaining = bytesPerDownload - uploadProgress[di];

                    VkDeviceSize alignedUsed = alignUp(poolUsedBytes[currentPoolIdx], alignment);
                    VkDeviceSize available = (alignedUsed < poolSize) ? poolSize - alignedUsed : 0;
                    VkDeviceSize maxAllocatable = (available / alignment) * alignment;

                    if (maxAllocatable == 0) {
                        currentPoolIdx++;
                        if (currentPoolIdx >= pools.size()) {
                            freeUploadPending(uploadPending);
                        }
                        continue;
                    }

                    VkDeviceSize chunkSize = std::min(remaining, maxAllocatable);
                    chunkSize              = (chunkSize / 4) * 4;

                    if (chunkSize == 0) {
                        currentPoolIdx++;
                        if (currentPoolIdx >= pools.size()) {
                            freeUploadPending(uploadPending);
                        }
                        continue;
                    }

                    VkBufferCreateInfo stagingInfo = {
                        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .pNext                 = nullptr,
                        .flags                 = 0,
                        .size                  = chunkSize,
                        .usage                 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                        .queueFamilyIndexCount = 0,
                        .pQueueFamilyIndices   = nullptr,
                    };
                    VmaAllocationCreateInfo stagingAllocInfo = {
                        .flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                        .usage         = VMA_MEMORY_USAGE_UNKNOWN,
                        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        .preferredFlags = 0,
                        .memoryTypeBits = 0,
                        .pool           = pools[currentPoolIdx],
                        .pUserData      = nullptr,
                        .priority       = 0.0f,
                    };

                    StagingBuffer staging;
                    staging.downloadIdx = di;
                    staging.srcOffset   = uploadProgress[di];
                    staging.size        = chunkSize;

                    VkResult    result{};
                    std::string tag;
                    {
                        result =
                            vmaCreateBuffer(vma, &stagingInfo, &stagingAllocInfo, &staging.buffer,
                                            &staging.allocation, &staging.allocInfo);
                        if (result == VK_SUCCESS) {
                            // Track device memory range for overlap detection
                            deviceRangeTracker.add(staging.allocInfo.deviceMemory,
                                                   staging.allocInfo.offset, chunkSize,
                                                   staging.buffer);
                        }
                    }

                    if (result != VK_SUCCESS) {
                        currentPoolIdx++;
                        if (currentPoolIdx >= pools.size()) {
                            freeUploadPending(uploadPending);
                        }
                        continue;
                    }

                    poolUsedBytes[currentPoolIdx] =
                        alignUp(poolUsedBytes[currentPoolIdx], alignment) +
                        alignUp(chunkSize, alignment);

                    // Fill staging buffer with humanValueHash data
                    auto*        data = static_cast<uint32_t*>(staging.allocInfo.pMappedData);
                    size_t       numElements   = chunkSize / sizeof(uint32_t);
                    VkDeviceSize elementOffset = uploadProgress[di] / sizeof(uint32_t);
                    for (size_t j = 0; j < numElements; ++j) {
                        data[j] = humanValueHash(iter, di, elementOffset, j);
                    }

                    // Explicit flush even though we requested HOST_COHERENT
                    vmaFlushAllocation(vma, staging.allocation, 0, VK_WHOLE_SIZE);

                    uploadProgress[di] += chunkSize;

                    // Record copy, submit, wait
                    {
                        auto recording = ctx->beginRecording(cmdPool);

                        // Barrier: ensure host writes to staging buffer are visible to transfer
                        VkMemoryBarrier2 preBarrier = {
                            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                            .pNext         = nullptr,
                            .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                            .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                            .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                        };
                        VkDependencyInfo preDepInfo = {
                            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .pNext                    = nullptr,
                            .dependencyFlags          = 0,
                            .memoryBarrierCount       = 1,
                            .pMemoryBarriers          = &preBarrier,
                            .bufferMemoryBarrierCount = 0,
                            .pBufferMemoryBarriers    = nullptr,
                            .imageMemoryBarrierCount  = 0,
                            .pImageMemoryBarriers     = nullptr,
                        };
                        ctx->device.vkCmdPipelineBarrier2(recording, &preDepInfo);
                        memoryBarrierHammer(recording);

                        VkBufferCopy copyRegion = {
                            .srcOffset = 0,
                            .dstOffset = staging.srcOffset,
                            .size      = chunkSize,
                        };

                        // Check alignment
                        if (staging.srcOffset % optimalCopyAlign != 0) {
                            throw std::runtime_error("UPLOAD: dstOffset " +
                                                     std::to_string(staging.srcOffset) +
                                                     " not aligned to optimalCopyAlign " +
                                                     std::to_string(optimalCopyAlign));
                        }
                        if (staging.allocInfo.offset % alignment != 0) {
                            throw std::runtime_error("UPLOAD: staging buffer VMA offset " +
                                                     std::to_string(staging.allocInfo.offset) +
                                                     " not aligned to buffer alignment " +
                                                     std::to_string(alignment));
                        }

                        ctx->device.vkCmdCopyBuffer(recording, staging.buffer,
                                                    gpuBuffers[di].buffer, 1, &copyRegion);
                        memoryBarrierHammer(recording);

                        vko::CommandBuffer cmdBuf     = recording.end();
                        VkCommandBuffer    cmd        = cmdBuf;
                        VkSubmitInfo       submitInfo = {
                                  .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                  .pNext                = nullptr,
                                  .waitSemaphoreCount   = 0,
                                  .pWaitSemaphores      = nullptr,
                                  .pWaitDstStageMask    = nullptr,
                                  .commandBufferCount   = 1,
                                  .pCommandBuffers      = &cmd,
                                  .signalSemaphoreCount = 0,
                                  .pSignalSemaphores    = nullptr,
                        };
                        ASSERT_EQ(VK_SUCCESS,
                                  ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                        ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
                    }

                    uploadPending.push_back(staging);
                }
            }

            // Free remaining upload staging buffers
            if (!uploadPending.empty()) {
                freeUploadPending(uploadPending);
            }

            // Memory barrier after uploads
            {
                auto             recording = ctx->beginRecording(cmdPool);
                VkMemoryBarrier2 barrier   = {
                      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                      .pNext         = nullptr,
                      .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                };
                VkDependencyInfo depInfo = {
                    .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .pNext                    = nullptr,
                    .dependencyFlags          = 0,
                    .memoryBarrierCount       = 1,
                    .pMemoryBarriers          = &barrier,
                    .bufferMemoryBarrierCount = 0,
                    .pBufferMemoryBarriers    = nullptr,
                    .imageMemoryBarrierCount  = 0,
                    .pImageMemoryBarriers     = nullptr,
                };
                ctx->device.vkCmdPipelineBarrier2(recording, &depInfo);

                vko::CommandBuffer cmdBuf     = recording.end();
                VkCommandBuffer    cmd        = cmdBuf;
                VkSubmitInfo       submitInfo = {
                          .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext                = nullptr,
                          .waitSemaphoreCount   = 0,
                          .pWaitSemaphores      = nullptr,
                          .pWaitDstStageMask    = nullptr,
                          .commandBufferCount   = 1,
                          .pCommandBuffers      = &cmd,
                          .signalSemaphoreCount = 0,
                          .pSignalSemaphores    = nullptr,
                };
                ASSERT_EQ(VK_SUCCESS,
                          ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
            }
        }

        // Track how much of each download we've copied
        std::vector<VkDeviceSize> downloadProgress(numDownloads, 0);

        // Lambda to verify and free all pending staging buffers
        auto verifyAndFreePending = [&]() {
            for (auto& staging : pending) {
                // Explicit invalidate even though we requested HOST_COHERENT
                vmaInvalidateAllocation(vma, staging.allocation, 0, VK_WHOLE_SIZE);

                auto*        data          = static_cast<uint32_t*>(staging.allocInfo.pMappedData);
                size_t       numElements   = staging.size / sizeof(uint32_t);
                VkDeviceSize elementOffset = staging.srcOffset / sizeof(uint32_t);

                for (size_t j = 0; j < numElements; ++j) {
                    uint32_t expected = humanValueHash(iter, staging.downloadIdx, elementOffset, j);
                    if (data[j] != expected) {
                        ADD_FAILURE()
                            << "Data mismatch at iter=" << iter << " buf=" << staging.downloadIdx
                            << " srcOffset=" << staging.srcOffset << " offset=" << j
                            << " expected=" << expected << " got=" << data[j];
                        break;
                    }
                }

                deviceRangeTracker.remove(staging.buffer);
                vmaDestroyBuffer(vma, staging.buffer, staging.allocation);
            }
            pending.clear();

            // Reset pool usage - LINEAR pools reset when empty
            std::fill(poolUsedBytes.begin(), poolUsedBytes.end(), 0);

            // Reverse pool order (queue behavior - pop from front, push to back)
            // After a cycle, the pools end up in reversed order
            std::reverse(pools.begin(), pools.end());
            currentPoolIdx = 0;
        };

        // Iterate through downloads, chunking them across pools
        for (size_t di = 0; di < numDownloads; ++di) {
            while (downloadProgress[di] < bytesPerDownload) {
                VkDeviceSize remaining = bytesPerDownload - downloadProgress[di];

                // Try to allocate from current pool
                VkDeviceSize alignedUsed    = alignUp(poolUsedBytes[currentPoolIdx], alignment);
                VkDeviceSize available      = (alignedUsed < poolSize) ? poolSize - alignedUsed : 0;
                VkDeviceSize maxAllocatable = (available / alignment) * alignment;

                if (maxAllocatable == 0) {
                    // Current pool is full, try next pool
                    currentPoolIdx++;

                    if (currentPoolIdx >= pools.size()) {
                        // All pools full - verify pending, free them, reset pools
                        verifyAndFreePending();
                    }
                    continue;
                }

                VkDeviceSize chunkSize = std::min(remaining, maxAllocatable);
                chunkSize              = (chunkSize / 4) * 4; // Align to uint32_t

                if (chunkSize == 0) {
                    currentPoolIdx++;
                    if (currentPoolIdx >= pools.size()) {
                        verifyAndFreePending();
                    }
                    continue;
                }

                // Allocate staging buffer
                VkBufferCreateInfo stagingInfo = {
                    .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext                 = nullptr,
                    .flags                 = 0,
                    .size                  = chunkSize,
                    .usage                 = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices   = nullptr,
                };
                VmaAllocationCreateInfo stagingAllocInfo = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                    .usage = VMA_MEMORY_USAGE_UNKNOWN,
                    .requiredFlags =
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .preferredFlags = 0,
                    .memoryTypeBits = 0,
                    .pool           = pools[currentPoolIdx],
                    .pUserData      = nullptr,
                    .priority       = 0.0f,
                };

                StagingBuffer staging;
                staging.downloadIdx = di;
                staging.srcOffset   = downloadProgress[di];
                staging.size        = chunkSize;

                VkResult    result{};
                std::string downloadTag;
                {
                    result = vmaCreateBuffer(vma, &stagingInfo, &stagingAllocInfo, &staging.buffer,
                                             &staging.allocation, &staging.allocInfo);
                    if (result == VK_SUCCESS) {
                        // Track device memory range for overlap detection
                        deviceRangeTracker.add(staging.allocInfo.deviceMemory,
                                               staging.allocInfo.offset, chunkSize, staging.buffer);
                    }
                }

                if (result != VK_SUCCESS) {
                    // Allocation failed, try next pool
                    currentPoolIdx++;
                    if (currentPoolIdx >= pools.size()) {
                        verifyAndFreePending();
                    }
                    continue;
                }

                poolUsedBytes[currentPoolIdx] = alignUp(poolUsedBytes[currentPoolIdx], alignment) +
                                                alignUp(chunkSize, alignment);
                downloadProgress[di] += chunkSize;

                // Record copy, submit, wait immediately (simpler, same outcome)
                {
                    auto recording = ctx->beginRecording(cmdPool);

                    memoryBarrierHammer(recording);

                    VkBufferCopy copyRegion = {
                        .srcOffset = staging.srcOffset,
                        .dstOffset = 0,
                        .size      = chunkSize,
                    };

                    // Check alignment
                    if (staging.srcOffset % optimalCopyAlign != 0) {
                        throw std::runtime_error(
                            "DOWNLOAD: srcOffset " + std::to_string(staging.srcOffset) +
                            " not aligned to optimalCopyAlign " + std::to_string(optimalCopyAlign));
                    }
                    if (staging.allocInfo.offset % alignment != 0) {
                        throw std::runtime_error("DOWNLOAD: staging buffer VMA offset " +
                                                 std::to_string(staging.allocInfo.offset) +
                                                 " not aligned to buffer alignment " +
                                                 std::to_string(alignment));
                    }

                    ctx->device.vkCmdCopyBuffer(recording, gpuBuffers[di].buffer, staging.buffer, 1,
                                                &copyRegion);
                    memoryBarrierHammer(recording);

                    VkMemoryBarrier2 barrier = {
                        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                        .pNext         = nullptr,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                    };
                    VkDependencyInfo depInfo = {
                        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                        .pNext                    = nullptr,
                        .dependencyFlags          = 0,
                        .memoryBarrierCount       = 1,
                        .pMemoryBarriers          = &barrier,
                        .bufferMemoryBarrierCount = 0,
                        .pBufferMemoryBarriers    = nullptr,
                        .imageMemoryBarrierCount  = 0,
                        .pImageMemoryBarriers     = nullptr,
                    };
                    ctx->device.vkCmdPipelineBarrier2(recording, &depInfo);

                    vko::CommandBuffer cmdBuf     = recording.end();
                    VkCommandBuffer    cmd        = cmdBuf;
                    VkSubmitInfo       submitInfo = {
                              .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .pNext                = nullptr,
                              .waitSemaphoreCount   = 0,
                              .pWaitSemaphores      = nullptr,
                              .pWaitDstStageMask    = nullptr,
                              .commandBufferCount   = 1,
                              .pCommandBuffers      = &cmd,
                              .signalSemaphoreCount = 0,
                              .pSignalSemaphores    = nullptr,
                    };
                    ASSERT_EQ(VK_SUCCESS,
                              ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                    ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
                }

                // Sanity check: verify download staging buffer immediately after GPU copy
                {
                    vmaInvalidateAllocation(vma, staging.allocation, 0, VK_WHOLE_SIZE);
                    const volatile uint32_t* volatileData =
                        static_cast<const volatile uint32_t*>(staging.allocInfo.pMappedData);
                    size_t       numElements   = staging.size / sizeof(uint32_t);
                    VkDeviceSize elementOffset = staging.srcOffset / sizeof(uint32_t);

                    for (size_t j = 0; j < numElements; ++j) {
                        uint32_t expected =
                            humanValueHash(iter, staging.downloadIdx, elementOffset, j);
                        uint32_t actual = volatileData[j];
                        if (actual != expected) {
                            ADD_FAILURE() << "Download staging buffer wrong immediately after copy!"
                                          << " iter=" << iter << " buf=" << staging.downloadIdx
                                          << " srcOffset=" << staging.srcOffset << " offset=" << j
                                          << " expected=" << expected << " got=" << actual;
                            break;
                        }
                    }
                }

                pending.push_back(staging);
            }
        }

        // Verify any remaining pending buffers at end of iteration
        if (!pending.empty()) {
            verifyAndFreePending();
        }
    }

    // Cleanup
    for (auto& gpu : gpuBuffers) {
        vmaDestroyBuffer(vma, gpu.buffer, gpu.allocation);
    }
    for (auto pool : pools) {
        vmaDestroyPool(vma, pool);
    }
}

// =============================================================================
// Same test but bypassing VMA - direct Vulkan memory allocation
// =============================================================================

// Finally figured out this was my GPU - maybe VRAM, cache, trasnfer, MB even?
TEST_F(UnitTestFixture, DISABLED_StagingPoolReuseBugReproVKOnly) {
    auto    cmdPool = ctx->createCommandPool();
    VkQueue queue{};
    ctx->device.vkGetDeviceQueue(ctx->device, ctx->queueFamilyIndex, 0, &queue);

    // Find memory type for host-visible, host-coherent staging memory
    VkPhysicalDeviceMemoryProperties memProps{};
    ctx->instance.vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);

    uint32_t stagingMemTypeIndex = UINT32_MAX;
    uint32_t deviceMemTypeIndex  = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
        if ((flags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (stagingMemTypeIndex == UINT32_MAX) {
                stagingMemTypeIndex = i;
            }
        }
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            if (deviceMemTypeIndex == UINT32_MAX) {
                deviceMemTypeIndex = i;
            }
        }
    }
    ASSERT_NE(stagingMemTypeIndex, UINT32_MAX) << "No host-visible coherent memory type found";
    ASSERT_NE(deviceMemTypeIndex, UINT32_MAX) << "No device-local memory type found";

    // Query buffer alignment
    VkDeviceSize alignment = 1;
    {
        VkBufferCreateInfo tempInfo = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .size        = 1,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VkBuffer tempBuffer{};
        ASSERT_EQ(VK_SUCCESS,
                  ctx->device.vkCreateBuffer(ctx->device, &tempInfo, nullptr, &tempBuffer));
        VkMemoryRequirements req{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, tempBuffer, &req);
        alignment = req.alignment;
        if constexpr (ALIGNMENT_OVERRIDE > 0) {
            alignment = std::max(alignment, ALIGNMENT_OVERRIDE);
        }
        ctx->device.vkDestroyBuffer(ctx->device, tempBuffer, nullptr);
    }

    auto alignUp = [](VkDeviceSize value, VkDeviceSize align) {
        return (value + align - 1) / align * align;
    };

    // Full memory barrier hammer - sprinkle EVERYWHERE to rule out sync issues
    auto memoryBarrierHammer = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier2 barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext         = nullptr,
            .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        };
        VkDependencyInfo depInfo = {
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext                    = nullptr,
            .dependencyFlags          = 0,
            .memoryBarrierCount       = 1,
            .pMemoryBarriers          = &barrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers    = nullptr,
            .imageMemoryBarrierCount  = 0,
            .pImageMemoryBarriers     = nullptr,
        };
        ctx->device.vkCmdPipelineBarrier2(cmd, &depInfo);
    };

    // Dedicated staging buffer for verification (completely separate from pools)
    struct VerifyStaging {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*          mapped = nullptr;
        VkDeviceSize   size   = 0;
    };

    auto createVerifyStaging = [&](VkDeviceSize size) -> VerifyStaging {
        VerifyStaging vs;
        vs.size = size;

        VkBufferCreateInfo bufInfo = {
            .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext                 = nullptr,
            .flags                 = 0,
            .size                  = size,
            .usage                 = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        EXPECT_EQ(VK_SUCCESS,
                  ctx->device.vkCreateBuffer(ctx->device, &bufInfo, nullptr, &vs.buffer));

        VkMemoryRequirements memReq{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, vs.buffer, &memReq);

        VkMemoryAllocateInfo allocInfo = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext           = nullptr,
            .allocationSize  = memReq.size,
            .memoryTypeIndex = stagingMemTypeIndex,
        };
        EXPECT_EQ(VK_SUCCESS,
                  ctx->device.vkAllocateMemory(ctx->device, &allocInfo, nullptr, &vs.memory));
        EXPECT_EQ(VK_SUCCESS, ctx->device.vkBindBufferMemory(ctx->device, vs.buffer, vs.memory, 0));
        EXPECT_EQ(VK_SUCCESS,
                  ctx->device.vkMapMemory(ctx->device, vs.memory, 0, VK_WHOLE_SIZE, 0, &vs.mapped));
        return vs;
    };

    auto destroyVerifyStaging = [&](VerifyStaging& vs) {
        ctx->device.vkUnmapMemory(ctx->device, vs.memory);
        ctx->device.vkDestroyBuffer(ctx->device, vs.buffer, nullptr);
        ctx->device.vkFreeMemory(ctx->device, vs.memory, nullptr);
        vs = {};
    };

    // Allocate pool memory blocks directly (no VMA)
    struct MemoryPool {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*          mapped = nullptr;
        VkDeviceSize   used   = 0;
    };
    std::deque<MemoryPool> pools(maxPools);

// Align pool size to our alignment (especially important with ALIGNMENT_OVERRIDE)
#if ALIGNMENT_OVERRIDE > 0
    VkDeviceSize alignedPoolSize = alignUp(poolSize, alignment);
#else
    VkDeviceSize alignedPoolSize = poolSize;
#endif

    for (size_t i = 0; i < maxPools; ++i) {
        VkMemoryAllocateInfo allocInfo = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext           = nullptr,
            .allocationSize  = alignedPoolSize,
            .memoryTypeIndex = stagingMemTypeIndex,
        };
        ASSERT_EQ(VK_SUCCESS,
                  ctx->device.vkAllocateMemory(ctx->device, &allocInfo, nullptr, &pools[i].memory));
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkMapMemory(ctx->device, pools[i].memory, 0,
                                                      VK_WHOLE_SIZE, 0, &pools[i].mapped));
    }

    // Allocate GPU buffers (device-local) - use VMA for simplicity here
    VmaAllocator vma = ctx->allocator;
    struct GpuBuffer {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };
    std::vector<GpuBuffer> gpuBuffers(numDownloads);

    for (size_t i = 0; i < numDownloads; ++i) {
        VkBufferCreateInfo bufferInfo = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .size        = alignUp(bytesPerDownload, alignment),
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VmaAllocationCreateInfo allocInfo = {
            .flags          = 0,
            .usage          = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = nullptr,
            .priority       = 0.0f,
        };
        ASSERT_EQ(VK_SUCCESS, vmaCreateBuffer(vma, &bufferInfo, &allocInfo, &gpuBuffers[i].buffer,
                                              &gpuBuffers[i].allocation, nullptr));
    }

    // Damage check staging buffer - for monitoring corruption after uploads
    VerifyStaging damageCheckStaging = createVerifyStaging(bytesPerDownload);

    // Helper to verify entire GPU buffer matches expected data after each upload
    auto verifyBufferState = [&](size_t bufIdx, const std::vector<uint32_t>& expectedData,
                                 const char* context) {
        // Download entire GPU buffer
        {
            auto recording = ctx->beginRecording(cmdPool);
            memoryBarrierHammer(recording);
            VkBufferCopy copyRegion = {.srcOffset = 0, .dstOffset = 0, .size = bytesPerDownload};
            ctx->device.vkCmdCopyBuffer(recording, gpuBuffers[bufIdx].buffer,
                                        damageCheckStaging.buffer, 1, &copyRegion);
            memoryBarrierHammer(recording);
            VkMemoryBarrier2 barrier = {
                .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext         = nullptr,
                .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
            };
            VkDependencyInfo depInfo = {
                .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext                    = nullptr,
                .dependencyFlags          = 0,
                .memoryBarrierCount       = 1,
                .pMemoryBarriers          = &barrier,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers    = nullptr,
                .imageMemoryBarrierCount  = 0,
                .pImageMemoryBarriers     = nullptr,
            };
            memoryBarrierHammer(recording);
            ctx->device.vkCmdPipelineBarrier2(recording, &depInfo);
            memoryBarrierHammer(recording);
            vko::CommandBuffer cmdBuf     = recording.end();
            VkCommandBuffer    cmd        = cmdBuf;
            VkSubmitInfo       submitInfo = {
                      .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .pNext                = nullptr,
                      .waitSemaphoreCount   = 0,
                      .pWaitSemaphores      = nullptr,
                      .pWaitDstStageMask    = nullptr,
                      .commandBufferCount   = 1,
                      .pCommandBuffers      = &cmd,
                      .signalSemaphoreCount = 0,
                      .pSignalSemaphores    = nullptr,
            };
            ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
        }

        // Compare entire buffer against expected
        auto*           data = static_cast<uint32_t*>(damageCheckStaging.mapped);
        BadRangeTracker bad;
        for (size_t j = 0; j < downloadSize; ++j) {
            if (data[j] != expectedData[j]) {
                bad.add(j, expectedData[j], data[j]);
            }
        }

        if (!bad.empty()) {
            ADD_FAILURE() << "BUFFER MISMATCH " << context << " buf=" << bufIdx
                          << " bad=" << bad.totalBad() << "/" << downloadSize
                          << " ranges=" << bad.summary() << " first: " << bad.firstValue();
        }
    };

    // Staging buffer wrapper (no VMA)
    struct StagingBuffer {
        VkBuffer     buffer      = VK_NULL_HANDLE;
        size_t       poolIdx     = 0;
        VkDeviceSize poolOffset  = 0; // Offset within pool memory
        size_t       downloadIdx = 0;
        VkDeviceSize srcOffset   = 0; // Offset within GPU buffer being transferred
        VkDeviceSize size        = 0;
    };

    std::vector<VkDeviceSize>  poolUsedBytes(maxPools, 0);
    std::vector<StagingBuffer> pending;
    size_t                     currentPoolIdx = 0;

    // Track device memory ranges to detect overlapping allocations
    DeviceAddressRangeTracker noVmaRangeTracker;

    // Suballocate a staging buffer from our pools
    auto allocateStagingBuffer = [&](VkDeviceSize size) -> std::optional<StagingBuffer> {
        VkDeviceSize alignedUsed = alignUp(poolUsedBytes[currentPoolIdx], alignment);
        VkDeviceSize available =
            (alignedUsed < alignedPoolSize) ? alignedPoolSize - alignedUsed : 0;

        if (available < size) {
            return std::nullopt;
        }

        // Create VkBuffer
        VkBufferCreateInfo bufferInfo = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .size        = size,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };

        StagingBuffer staging;
        staging.poolIdx    = currentPoolIdx;
        staging.poolOffset = alignedUsed;
        staging.size       = size;

        VkResult result =
            ctx->device.vkCreateBuffer(ctx->device, &bufferInfo, nullptr, &staging.buffer);
        if (result != VK_SUCCESS) {
            return std::nullopt;
        }

        // Check memory requirements
        VkMemoryRequirements memReq{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, staging.buffer, &memReq);

        // Verify our pool memory is compatible
        if ((memReq.memoryTypeBits & (1 << stagingMemTypeIndex)) == 0) {
            throw std::runtime_error("Pool memory type not compatible with staging buffer!");
        }
        if (alignedUsed % memReq.alignment != 0) {
            throw std::runtime_error("Pool offset " + std::to_string(alignedUsed) +
                                     " not aligned to buffer requirement " +
                                     std::to_string(memReq.alignment));
        }
        // Note: memReq.size may be larger than buffer size due to alignment padding
        // We must check the ALIGNED size fits, not just memReq.size
        VkDeviceSize alignedMemSize = alignUp(memReq.size, alignment);
        if (alignedMemSize > available) {
            ctx->device.vkDestroyBuffer(ctx->device, staging.buffer, nullptr);
            return std::nullopt; // Not enough space for aligned memory requirement
        }

        // Bind to pool memory at offset
        result = ctx->device.vkBindBufferMemory(ctx->device, staging.buffer,
                                                pools[currentPoolIdx].memory, alignedUsed);
        if (result != VK_SUCCESS) {
            ctx->device.vkDestroyBuffer(ctx->device, staging.buffer, nullptr);
            return std::nullopt;
        }

        // Track memory usage
        VkDeviceSize newUsed     = alignedUsed + alignedMemSize;
        #if 0
        static bool  printedOnce = false;
        if (!printedOnce) {
            // After first allocation at 0, next would need to start at newUsed
            size_t canFitSecond = (newUsed + memReq.size <= alignedPoolSize) ? 1 : 0;
            printf("DirectMemory: alignment=%zu alignedPoolSize=%zu memReq.size=%zu newUsed=%zu "
                   "(1st fits, 2nd %s)\n",
                   (size_t)alignment, (size_t)alignedPoolSize, (size_t)memReq.size, (size_t)newUsed,
                   canFitSecond ? "fits" : "NO");
            printedOnce = true;
        }
        #endif
        poolUsedBytes[currentPoolIdx] = newUsed;
        return staging;
    };

    // Get mapped pointer for a staging buffer
    auto getStagingPtr = [&](const StagingBuffer& staging) -> void* {
        return static_cast<char*>(pools[staging.poolIdx].mapped) + staging.poolOffset;
    };

    // Track actual transfers
    size_t totalUploads = 0, totalDownloads = 0;

    // Fill all GPU buffers with sentinel value before uploads
    constexpr uint32_t SENTINEL = 0xDEADBEEF;
    {
        auto recording = ctx->beginRecording(cmdPool);
        for (size_t i = 0; i < numDownloads; ++i) {
            ctx->device.vkCmdFillBuffer(recording, gpuBuffers[i].buffer, 0, VK_WHOLE_SIZE,
                                        SENTINEL);
        }
        vko::CommandBuffer cmdBuf     = recording.end();
        VkCommandBuffer    cmd        = cmdBuf;
        VkSubmitInfo       submitInfo = {
                  .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                  .pNext                = nullptr,
                  .waitSemaphoreCount   = 0,
                  .pWaitSemaphores      = nullptr,
                  .pWaitDstStageMask    = nullptr,
                  .commandBufferCount   = 1,
                  .pCommandBuffers      = &cmd,
                  .signalSemaphoreCount = 0,
                  .pSignalSemaphores    = nullptr,
        };
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
    }

    // Track expected state for each GPU buffer (sentinel or uploaded data)
    std::vector<std::vector<uint32_t>> expectedGpuData(numDownloads);
    for (size_t i = 0; i < numDownloads; ++i) {
        expectedGpuData[i].resize(downloadSize, SENTINEL);
    }

    // Run iterations
    for (size_t iter = 0; iter < numIterations; ++iter) {
        // Upload phase (first iteration only, like original test)
        if (iter == 0) {
            std::vector<VkDeviceSize>  uploadProgress(numDownloads, 0);
            std::vector<StagingBuffer> uploadPending;

            auto freeUploadPending = [&]() {
                for (auto& staging : uploadPending) {
                    ctx->device.vkDestroyBuffer(ctx->device, staging.buffer, nullptr);
                }
                uploadPending.clear();
                std::fill(poolUsedBytes.begin(), poolUsedBytes.end(), 0);
                std::reverse(pools.begin(), pools.end());
                currentPoolIdx = 0;
            };

            for (size_t di = 0; di < numDownloads; ++di) {
                while (uploadProgress[di] < bytesPerDownload) {
                    VkDeviceSize remaining = bytesPerDownload - uploadProgress[di];
                    VkDeviceSize chunkSize = std::min(remaining, poolSize);
                    chunkSize              = (chunkSize / 4) * 4;

                    auto maybeStagingOpt = allocateStagingBuffer(chunkSize);
                    if (!maybeStagingOpt) {
                        currentPoolIdx++;
                        if (currentPoolIdx >= pools.size()) {
                            freeUploadPending();
                        }
                        continue;
                    }

                    StagingBuffer staging = *maybeStagingOpt;
                    staging.downloadIdx   = di;
                    staging.srcOffset     = uploadProgress[di];

                    // Fill staging buffer
                    auto*        data          = static_cast<uint32_t*>(getStagingPtr(staging));
                    size_t       numElements   = staging.size / sizeof(uint32_t);
                    VkDeviceSize elementOffset = staging.srcOffset / sizeof(uint32_t);
                    for (size_t j = 0; j < numElements; ++j) {
                        data[j] = humanValueHash(iter, di, elementOffset, j);
                    }

                    // VERIFY: Check staging buffer content right after writing, before any GPU ops
                    {
                        BadRangeTracker    stagingBad;
                        volatile uint32_t* vdata =
                            static_cast<volatile uint32_t*>(getStagingPtr(staging));
                        for (size_t j = 0; j < numElements; ++j) {
                            uint32_t expected = humanValueHash(iter, di, elementOffset, j);
                            if (vdata[j] != expected) {
                                stagingBad.add(j, expected, vdata[j]);
                            }
                        }
                        if (!stagingBad.empty()) {
                            ADD_FAILURE() << "STAGING BUFFER CORRUPTED before copy! buf=" << di
                                          << " chunk_offset=" << staging.srcOffset
                                          << " bad=" << stagingBad.totalBad() << "/" << numElements
                                          << " ranges=" << stagingBad.summary()
                                          << " first: " << stagingBad.firstValue();
                        }
                    }

                    uploadProgress[di] += staging.size;

                    // Record and submit upload copy
                    {
                        auto recording = ctx->beginRecording(cmdPool);

                        // HOST_WRITE -> TRANSFER_READ barrier for staging buffer
                        VkMemoryBarrier2 uploadBarrier = {
                            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                            .pNext         = nullptr,
                            .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                            .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                            .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                        };
                        VkDependencyInfo uploadDepInfo = {
                            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .pNext                    = nullptr,
                            .dependencyFlags          = 0,
                            .memoryBarrierCount       = 1,
                            .pMemoryBarriers          = &uploadBarrier,
                            .bufferMemoryBarrierCount = 0,
                            .pBufferMemoryBarriers    = nullptr,
                            .imageMemoryBarrierCount  = 0,
                            .pImageMemoryBarriers     = nullptr,
                        };
                        memoryBarrierHammer(recording);
                        ctx->device.vkCmdPipelineBarrier2(recording, &uploadDepInfo);
                        memoryBarrierHammer(recording);

                        VkBufferCopy copyRegion = {
                            .srcOffset = 0,
                            .dstOffset = staging.srcOffset,
                            .size      = staging.size,
                        };
                        memoryBarrierHammer(recording);
                        ctx->device.vkCmdCopyBuffer(recording, staging.buffer,
                                                    gpuBuffers[di].buffer, 1, &copyRegion);
                        memoryBarrierHammer(recording);

                        vko::CommandBuffer cmdBuf     = recording.end();
                        VkCommandBuffer    cmd        = cmdBuf;
                        VkSubmitInfo       submitInfo = {
                                  .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                  .pNext                = nullptr,
                                  .waitSemaphoreCount   = 0,
                                  .pWaitSemaphores      = nullptr,
                                  .pWaitDstStageMask    = nullptr,
                                  .commandBufferCount   = 1,
                                  .pCommandBuffers      = &cmd,
                                  .signalSemaphoreCount = 0,
                                  .pSignalSemaphores    = nullptr,
                        };
                        ASSERT_EQ(VK_SUCCESS,
                                  ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                        ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
                    }

                    // Update expected data for this upload chunk
                    size_t startElem = staging.srcOffset / sizeof(uint32_t);
                    size_t numElems  = staging.size / sizeof(uint32_t);
                    for (size_t j = 0; j < numElems; ++j) {
                        expectedGpuData[di][startElem + j] = humanValueHash(iter, di, startElem, j);
                    }

                    // Verify buffer state immediately after this upload
                    verifyBufferState(di, expectedGpuData[di], "after upload");

                    uploadPending.push_back(staging);
                    totalUploads++;
                }
            }
            freeUploadPending();

            // SANITY CHECK: Verify uploads worked by downloading and checking GPU buffer content
            for (size_t di = 0; di < numDownloads; ++di) {
                VerifyStaging vs = createVerifyStaging(bytesPerDownload);

                // Download entire GPU buffer to dedicated staging
                {
                    auto recording = ctx->beginRecording(cmdPool);
                    memoryBarrierHammer(recording);
                    VkBufferCopy copyRegion = {
                        .srcOffset = 0, .dstOffset = 0, .size = bytesPerDownload};
                    ctx->device.vkCmdCopyBuffer(recording, gpuBuffers[di].buffer, vs.buffer, 1,
                                                &copyRegion);
                    memoryBarrierHammer(recording);
                    VkMemoryBarrier2 barrier = {
                        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                        .pNext         = nullptr,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                    };
                    VkDependencyInfo depInfo = {
                        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                        .pNext                    = nullptr,
                        .dependencyFlags          = 0,
                        .memoryBarrierCount       = 1,
                        .pMemoryBarriers          = &barrier,
                        .bufferMemoryBarrierCount = 0,
                        .pBufferMemoryBarriers    = nullptr,
                        .imageMemoryBarrierCount  = 0,
                        .pImageMemoryBarriers     = nullptr,
                    };
                    memoryBarrierHammer(recording);
                    ctx->device.vkCmdPipelineBarrier2(recording, &depInfo);
                    memoryBarrierHammer(recording);
                    vko::CommandBuffer cmdBuf     = recording.end();
                    VkCommandBuffer    cmd        = cmdBuf;
                    VkSubmitInfo       submitInfo = {
                              .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .pNext                = nullptr,
                              .waitSemaphoreCount   = 0,
                              .pWaitSemaphores      = nullptr,
                              .pWaitDstStageMask    = nullptr,
                              .commandBufferCount   = 1,
                              .pCommandBuffers      = &cmd,
                              .signalSemaphoreCount = 0,
                              .pSignalSemaphores    = nullptr,
                    };
                    ASSERT_EQ(VK_SUCCESS,
                              ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                    ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
                }

                // Verify content from dedicated staging memory
                auto*           verifyData = static_cast<uint32_t*>(vs.mapped);
                BadRangeTracker bad;
                for (size_t j = 0; j < downloadSize; ++j) {
                    uint32_t expected = humanValueHash(0, di, 0, j);
                    if (verifyData[j] != expected) {
                        bad.add(j, expected, verifyData[j]);
                    }
                }
                if (!bad.empty()) {
                    ADD_FAILURE() << "DirectMemory UPLOAD VERIFY FAILED: buf=" << di
                                  << " bad=" << bad.totalBad() << "/" << downloadSize
                                  << " ranges=" << bad.summary() << " first: " << bad.firstValue();
                }

                destroyVerifyStaging(vs);
            }
        }

        // Download phase
        std::vector<VkDeviceSize> downloadProgress(numDownloads, 0);

        auto verifyAndFreePending = [&]() {
            for (const auto& staging : pending) {
                auto*        data          = static_cast<uint32_t*>(getStagingPtr(staging));
                size_t       numElements   = staging.size / sizeof(uint32_t);
                VkDeviceSize elementOffset = staging.srcOffset / sizeof(uint32_t);

                BadRangeTracker bad;
                for (size_t j = 0; j < numElements; ++j) {
                    uint32_t expected = humanValueHash(iter, staging.downloadIdx, elementOffset, j);
                    if (data[j] != expected) {
                        bad.add(j, expected, data[j]);
                    }
                }
                if (!bad.empty()) {
                    ADD_FAILURE() << "DirectMemory DOWNLOAD FAILED: iter=" << iter
                                  << " buf=" << staging.downloadIdx
                                  << " chunk_offset=" << staging.srcOffset
                                  << " bad=" << bad.totalBad() << "/" << numElements
                                  << " ranges=" << bad.summary() << " first: " << bad.firstValue();
                }
                ctx->device.vkDestroyBuffer(ctx->device, staging.buffer, nullptr);
            }
            pending.clear();
            std::fill(poolUsedBytes.begin(), poolUsedBytes.end(), 0);
            std::reverse(pools.begin(), pools.end());
            currentPoolIdx = 0;
        };

        for (size_t di = 0; di < numDownloads; ++di) {
            while (downloadProgress[di] < bytesPerDownload) {
                VkDeviceSize remaining = bytesPerDownload - downloadProgress[di];
                VkDeviceSize chunkSize = std::min(remaining, poolSize);
                chunkSize              = (chunkSize / 4) * 4;

                auto maybeStagingOpt = allocateStagingBuffer(chunkSize);
                if (!maybeStagingOpt) {
                    currentPoolIdx++;
                    if (currentPoolIdx >= pools.size()) {
                        verifyAndFreePending();
                    }
                    continue;
                }

                StagingBuffer staging = *maybeStagingOpt;
                staging.downloadIdx   = di;
                staging.srcOffset     = downloadProgress[di];
                downloadProgress[di] += staging.size;

                // Record and submit download copy
                {
                    auto recording = ctx->beginRecording(cmdPool);
                    memoryBarrierHammer(recording);
                    VkBufferCopy copyRegion = {
                        .srcOffset = staging.srcOffset,
                        .dstOffset = 0,
                        .size      = staging.size,
                    };
                    ctx->device.vkCmdCopyBuffer(recording, gpuBuffers[di].buffer, staging.buffer, 1,
                                                &copyRegion);
                    memoryBarrierHammer(recording);

                    VkMemoryBarrier2 barrier = {
                        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                        .pNext         = nullptr,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                    };
                    VkDependencyInfo depInfo = {
                        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                        .pNext                    = nullptr,
                        .dependencyFlags          = 0,
                        .memoryBarrierCount       = 1,
                        .pMemoryBarriers          = &barrier,
                        .bufferMemoryBarrierCount = 0,
                        .pBufferMemoryBarriers    = nullptr,
                        .imageMemoryBarrierCount  = 0,
                        .pImageMemoryBarriers     = nullptr,
                    };
                    memoryBarrierHammer(recording);
                    ctx->device.vkCmdPipelineBarrier2(recording, &depInfo);
                    memoryBarrierHammer(recording);

                    vko::CommandBuffer cmdBuf     = recording.end();
                    VkCommandBuffer    cmd        = cmdBuf;
                    VkSubmitInfo       submitInfo = {
                              .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .pNext                = nullptr,
                              .waitSemaphoreCount   = 0,
                              .pWaitSemaphores      = nullptr,
                              .pWaitDstStageMask    = nullptr,
                              .commandBufferCount   = 1,
                              .pCommandBuffers      = &cmd,
                              .signalSemaphoreCount = 0,
                              .pSignalSemaphores    = nullptr,
                    };
                    ASSERT_EQ(VK_SUCCESS,
                              ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                    ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
                }

                pending.push_back(staging);
                totalDownloads++;
            }
        }

        if (!pending.empty()) {
            verifyAndFreePending();
        }
    }

    // Verify we actually did transfers
    //printf("DirectMemory: totalUploads=%zu totalDownloads=%zu\n", totalUploads, totalDownloads);
    ASSERT_GT(totalUploads, 0) << "No uploads happened - test is invalid!";
    ASSERT_GT(totalDownloads, 0) << "No downloads happened - test is invalid!";

    // Cleanup
    destroyVerifyStaging(damageCheckStaging);
    for (auto& gpu : gpuBuffers) {
        vmaDestroyBuffer(vma, gpu.buffer, gpu.allocation);
    }
    for (auto& pool : pools) {
        ctx->device.vkUnmapMemory(ctx->device, pool.memory);
        ctx->device.vkFreeMemory(ctx->device, pool.memory, nullptr);
    }
}

// =============================================================================
// MINIMAL SINGLE-BUFFER TEST
// =============================================================================
// Simplest possible reproduction: one upload staging buffer, one GPU buffer,
// one download staging buffer. No pools, no reuse.

// Toggle: use exact buffer size for memory allocation instead of poolSize
#define USE_EXACT_BUFFER_SIZE_FOR_MEMORY 1 // Use poolSize (larger allocation)

// Buffer size options for minimal test:
// 0 = original size (6788 bytes, crosses page boundary at 4096)
// 1 = single page (4096 bytes, no page crossing)
// 2 = two full pages (8192 bytes, crosses at aligned boundary)
#define MINIMAL_BUFFER_SIZE_OPTION 0

// Finally figured out this was my GPU - maybe VRAM, cache, trasnfer, MB even?
TEST_F(UnitTestFixture, DISABLED_StagingPoolReuseBugReproMinimal) {
    auto    cmdPool = ctx->createCommandPool();
    VkQueue queue{};
    ctx->device.vkGetDeviceQueue(ctx->device, ctx->queueFamilyIndex, 0, &queue);

#if MINIMAL_BUFFER_SIZE_OPTION == 0
    constexpr VkDeviceSize bufferSize = bytesPerDownload; // 6788 bytes - crosses page
#elif MINIMAL_BUFFER_SIZE_OPTION == 1
    constexpr VkDeviceSize bufferSize = 4096; // Single page, no crossing
#elif MINIMAL_BUFFER_SIZE_OPTION == 2
    constexpr VkDeviceSize bufferSize = 8192; // Two full pages
#endif

    // Find memory types
    VkPhysicalDeviceMemoryProperties memProps{};
    ctx->instance.vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);

    uint32_t stagingMemTypeIndex = UINT32_MAX;
    uint32_t deviceMemTypeIndex  = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (stagingMemTypeIndex == UINT32_MAX)
                stagingMemTypeIndex = i;
        }
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            if (deviceMemTypeIndex == UINT32_MAX)
                deviceMemTypeIndex = i;
        }
    }
    ASSERT_NE(stagingMemTypeIndex, UINT32_MAX);
    ASSERT_NE(deviceMemTypeIndex, UINT32_MAX);

    // Track alignments for pass/fail analysis
    std::map<size_t, size_t> passAlignments; // alignment -> count
    std::map<size_t, size_t> failAlignments;
    // upload device, gpu device, download device, upload host, passed
    std::vector<std::tuple<VkDeviceAddress, VkDeviceAddress, VkDeviceAddress, uintptr_t, bool>>
        allResults;

    for (uint32_t allocIter = 0; allocIter < 100; ++allocIter) {

        // Create upload staging buffer (with device address for tracking)
        VkBufferCreateInfo stagingBufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size  = bufferSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VkBuffer uploadStagingBuf{};
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkCreateBuffer(ctx->device, &stagingBufInfo, nullptr,
                                                         &uploadStagingBuf));

        VkMemoryRequirements uploadMemReq{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, uploadStagingBuf, &uploadMemReq);

// Print memory requirements on first iteration
#if 0
        if (allocIter == 0) {
            printf("Minimal test memory requirements:\n");
            printf("  bufferSize requested: %zu bytes\n", (size_t)bufferSize);
            printf("  uploadMemReq.size:    %zu bytes\n", (size_t)uploadMemReq.size);
            printf("  uploadMemReq.alignment: %zu bytes\n", (size_t)uploadMemReq.alignment);
            printf("  uploadMemReq.memoryTypeBits: 0x%x\n", uploadMemReq.memoryTypeBits);
            printf("  stagingMemTypeIndex: %u (bit set: %s)\n", stagingMemTypeIndex,
                   (uploadMemReq.memoryTypeBits & (1 << stagingMemTypeIndex)) ? "YES" : "NO!");
            printf("  Page boundary at byte 4096 = element %zu\n", 4096 / sizeof(uint32_t));
            printf("  Buffer spans pages: [0, %zu) and [4096, %zu)\n", 
                   std::min((size_t)4096, (size_t)bufferSize), (size_t)bufferSize);
        }
#endif

        // Verify memory type is compatible with buffer
        ASSERT_TRUE(uploadMemReq.memoryTypeBits & (1 << stagingMemTypeIndex))
            << "stagingMemTypeIndex " << stagingMemTypeIndex << " not in memoryTypeBits 0x"
            << std::hex << uploadMemReq.memoryTypeBits;

#if !USE_EXACT_BUFFER_SIZE_FOR_MEMORY
        ASSERT_GE(poolSize, uploadMemReq.size); // Only needed if using poolSize for allocation
#endif
        // Need device address flag for vkGetBufferDeviceAddress
        VkMemoryAllocateFlagsInfo allocFlags = {
            .sType      = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext      = nullptr,
            .flags      = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0,
        };
        VkMemoryAllocateInfo uploadMemAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &allocFlags,
#if USE_EXACT_BUFFER_SIZE_FOR_MEMORY
            .allocationSize = uploadMemReq.size, // Exact driver-requested size
#else
            .allocationSize = poolSize, // Larger pool-like allocation
#endif
            .memoryTypeIndex = stagingMemTypeIndex,
        };
        VkDeviceMemory uploadStagingMem{};
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkAllocateMemory(ctx->device, &uploadMemAllocInfo,
                                                           nullptr, &uploadStagingMem));
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkBindBufferMemory(ctx->device, uploadStagingBuf,
                                                             uploadStagingMem, 0));

        // Get device address for tracking
        VkBufferDeviceAddressInfo addrInfo = {
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext  = nullptr,
            .buffer = uploadStagingBuf,
        };
        VkDeviceAddress uploadDeviceAddr =
            ctx->device.vkGetBufferDeviceAddress(ctx->device, &addrInfo);

        void* uploadMapped = nullptr;
        uint32_t staticAnalysisIgnorer = 42;
        uploadMapped = (void*)&staticAnalysisIgnorer;
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkMapMemory(ctx->device, uploadStagingMem, 0,
                                                      VK_WHOLE_SIZE, 0, &uploadMapped));

        // Create GPU buffer (device local)
        VkBufferCreateInfo gpuBufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size  = bufferSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VkBuffer gpuBuf{};
        ASSERT_EQ(VK_SUCCESS,
                  ctx->device.vkCreateBuffer(ctx->device, &gpuBufInfo, nullptr, &gpuBuf));

        VkMemoryRequirements gpuMemReq{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, gpuBuf, &gpuMemReq);

#if 0
        if (allocIter == 0) {
            printf("  gpuMemReq.size:       %zu bytes\n", (size_t)gpuMemReq.size);
            printf("  gpuMemReq.alignment:  %zu bytes\n", (size_t)gpuMemReq.alignment);
        }
#endif

        VkMemoryAllocateInfo gpuMemAllocInfo = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext           = &allocFlags,
            .allocationSize  = gpuMemReq.size,
            .memoryTypeIndex = deviceMemTypeIndex,
        };
        VkDeviceMemory gpuMem{};
        ASSERT_EQ(VK_SUCCESS,
                  ctx->device.vkAllocateMemory(ctx->device, &gpuMemAllocInfo, nullptr, &gpuMem));
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkBindBufferMemory(ctx->device, gpuBuf, gpuMem, 0));

        addrInfo.buffer = gpuBuf;
        VkDeviceAddress gpuDeviceAddr =
            ctx->device.vkGetBufferDeviceAddress(ctx->device, &addrInfo);

        // Create download staging buffer (for verification)
        stagingBufInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer downloadStagingBuf{};
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkCreateBuffer(ctx->device, &stagingBufInfo, nullptr,
                                                         &downloadStagingBuf));

        VkMemoryRequirements downloadMemReq{};
        ctx->device.vkGetBufferMemoryRequirements(ctx->device, downloadStagingBuf, &downloadMemReq);

        VkMemoryAllocateInfo downloadMemAllocInfo = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext           = &allocFlags,
            .allocationSize  = downloadMemReq.size,
            .memoryTypeIndex = stagingMemTypeIndex,
        };
        VkDeviceMemory downloadStagingMem{};
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkAllocateMemory(ctx->device, &downloadMemAllocInfo,
                                                           nullptr, &downloadStagingMem));
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkBindBufferMemory(ctx->device, downloadStagingBuf,
                                                             downloadStagingMem, 0));

        addrInfo.buffer = downloadStagingBuf;
        VkDeviceAddress downloadDeviceAddr =
            ctx->device.vkGetBufferDeviceAddress(ctx->device, &addrInfo);

        void* downloadMapped = nullptr;
        downloadMapped = (void*)&staticAnalysisIgnorer;
        ASSERT_EQ(VK_SUCCESS, ctx->device.vkMapMemory(ctx->device, downloadStagingMem, 0,
                                                      VK_WHOLE_SIZE, 0, &downloadMapped));

        // Run test multiple iterations
        constexpr size_t numIterationsLocal = 10;
        size_t           numElements   = bufferSize / sizeof(uint32_t);

        for (size_t iter = 0; iter < numIterationsLocal; ++iter) {
            // Fill upload staging buffer with test pattern
            auto* uploadData = static_cast<uint32_t*>(uploadMapped);
            for (size_t j = 0; j < numElements; ++j) {
                uploadData[j] = humanValueHash(iter, 0, 0, j);
            }

            // Verify staging buffer content (paranoia check)
            volatile uint32_t* vUpload = static_cast<volatile uint32_t*>(uploadMapped);
            for (size_t j = 0; j < numElements; ++j) {
                uint32_t expected = humanValueHash(iter, 0, 0, j);
                ASSERT_EQ(vUpload[j], expected)
                    << "Upload staging corrupted at iter=" << iter << " j=" << j;
            }

            // Upload: staging -> GPU
            {
                auto recording = ctx->beginRecording(cmdPool);

                // HOST_WRITE -> TRANSFER_READ barrier
                VkMemoryBarrier2 barrier1 = {
                    .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext         = nullptr,
                    .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                    .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                };
                VkDependencyInfo dep1 = {
                    .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .pNext                    = nullptr,
                    .dependencyFlags          = 0,
                    .memoryBarrierCount       = 1,
                    .pMemoryBarriers          = &barrier1,
                    .bufferMemoryBarrierCount = 0,
                    .pBufferMemoryBarriers    = nullptr,
                    .imageMemoryBarrierCount  = 0,
                    .pImageMemoryBarriers     = nullptr,
                };
                ctx->device.vkCmdPipelineBarrier2(recording, &dep1);

                VkBufferCopy uploadCopy = {.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
                ctx->device.vkCmdCopyBuffer(recording, uploadStagingBuf, gpuBuf, 1, &uploadCopy);

                // TRANSFER_WRITE -> TRANSFER_READ barrier
                VkMemoryBarrier2 barrier2 = {
                    .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext         = nullptr,
                    .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                };
                VkDependencyInfo dep2 = {
                    .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .pNext                    = nullptr,
                    .dependencyFlags          = 0,
                    .memoryBarrierCount       = 1,
                    .pMemoryBarriers          = &barrier2,
                    .bufferMemoryBarrierCount = 0,
                    .pBufferMemoryBarriers    = nullptr,
                    .imageMemoryBarrierCount  = 0,
                    .pImageMemoryBarriers     = nullptr,
                };
                ctx->device.vkCmdPipelineBarrier2(recording, &dep2);

                // Download: GPU -> verify staging
                VkBufferCopy downloadCopy = {.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
                ctx->device.vkCmdCopyBuffer(recording, gpuBuf, downloadStagingBuf, 1,
                                            &downloadCopy);

                // TRANSFER_WRITE -> HOST_READ barrier
                VkMemoryBarrier2 barrier3 = {
                    .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext         = nullptr,
                    .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                };
                VkDependencyInfo dep3 = {
                    .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .pNext                    = nullptr,
                    .dependencyFlags          = 0,
                    .memoryBarrierCount       = 1,
                    .pMemoryBarriers          = &barrier3,
                    .bufferMemoryBarrierCount = 0,
                    .pBufferMemoryBarriers    = nullptr,
                    .imageMemoryBarrierCount  = 0,
                    .pImageMemoryBarriers     = nullptr,
                };
                ctx->device.vkCmdPipelineBarrier2(recording, &dep3);

                vko::CommandBuffer cmdBuf     = recording.end();
                VkCommandBuffer    cmd        = cmdBuf;
                VkSubmitInfo       submitInfo = {
                          .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext                = nullptr,
                          .waitSemaphoreCount   = 0,
                          .pWaitSemaphores      = nullptr,
                          .pWaitDstStageMask    = nullptr,
                          .commandBufferCount   = 1,
                          .pCommandBuffers      = &cmd,
                          .signalSemaphoreCount = 0,
                          .pSignalSemaphores    = nullptr,
                };
                ASSERT_EQ(VK_SUCCESS,
                          ctx->device.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
                ASSERT_EQ(VK_SUCCESS, ctx->device.vkQueueWaitIdle(queue));
            }

            // Verify downloaded data
            auto*           downloadData = static_cast<volatile uint32_t*>(downloadMapped);
            BadRangeTracker bad;
            for (size_t j = 0; j < numElements; ++j) {
                uint32_t expected = humanValueHash(iter, 0, 0, j);
                if (downloadData[j] != expected) {
                    bad.add(j, expected, downloadData[j]);
                }
            }

            bool thisFailed = !bad.empty();
            if (thisFailed) {
                ADD_FAILURE() << "MINIMAL TEST FAILED: iter=" << iter << " bad=" << bad.totalBad()
                              << "/" << numElements << " ranges=" << bad.summary()
                              << " first: " << bad.firstValue();

                // Calculate alignment (lowest set bit)
                auto getAlignment = [](VkDeviceAddress addr) -> size_t {
                    if (addr == 0)
                        return 0;
                    size_t align = 1;
                    while ((addr & align) == 0)
                        align <<= 1;
                    return align;
                };

                failAlignments[getAlignment(uploadDeviceAddr)]++;
                allResults.emplace_back(uploadDeviceAddr, gpuDeviceAddr, downloadDeviceAddr,
                                        reinterpret_cast<uintptr_t>(uploadMapped), false);
                break; // Stop on first failure
            }
        }

        // Record pass case (only if we didn't break early due to failure)
        bool hadFailure = false;
        for (auto& r : allResults)
            if (!std::get<4>(r))
                hadFailure = true;
        if (!hadFailure) {
            auto getAlignment = [](VkDeviceAddress addr) -> size_t {
                if (addr == 0)
                    return 0;
                size_t align = 1;
                while ((addr & align) == 0)
                    align <<= 1;
                return align;
            };
            passAlignments[getAlignment(uploadDeviceAddr)]++;
            allResults.emplace_back(uploadDeviceAddr, gpuDeviceAddr, downloadDeviceAddr,
                                    reinterpret_cast<uintptr_t>(uploadMapped), true);
        }

        // Cleanup
        ctx->device.vkUnmapMemory(ctx->device, uploadStagingMem);
        ctx->device.vkUnmapMemory(ctx->device, downloadStagingMem);
        ctx->device.vkDestroyBuffer(ctx->device, uploadStagingBuf, nullptr);
        ctx->device.vkDestroyBuffer(ctx->device, gpuBuf, nullptr);
        ctx->device.vkDestroyBuffer(ctx->device, downloadStagingBuf, nullptr);
        ctx->device.vkFreeMemory(ctx->device, uploadStagingMem, nullptr);
        ctx->device.vkFreeMemory(ctx->device, gpuMem, nullptr);
        ctx->device.vkFreeMemory(ctx->device, downloadStagingMem, nullptr);
    }

// Print alignment histogram
#if 0
    printf("\n=== Device Address Alignment Analysis ===\n");
    if (passAlignments.size() > 0) {
        printf("PASS alignments (upload staging device addr):\n");
        for (auto& [align, count] : passAlignments) {
            printf("  %8zu-byte aligned: %zu\n", align, count);
        }
    }
    if (failAlignments.size() > 0) {
        printf("FAIL alignments (upload staging device addr):\n");
        for (auto& [align, count] : failAlignments) {
            printf("  %8zu-byte aligned: %zu\n", align, count);
        }
    }
    
    // Print actual addresses for first few pass/fail
    size_t passCount = 0, failCount = 0;
    printf("\nSample addresses (device + host):\n");
    for (auto& [upload, gpu, download, hostPtr, passed] : allResults) {
        if (passed && passCount < 5) {
            printf("  PASS: devUpload=0x%lx (%%4096=%lu) devGpu=0x%lx hostUpload=0x%lx\n", 
                   (unsigned long)upload, (unsigned long)(upload % 4096),
                   (unsigned long)gpu, (unsigned long)hostPtr);
            passCount++;
        } else if (!passed && failCount < 5) {
            printf("  FAIL: devUpload=0x%lx (%%4096=%lu) devGpu=0x%lx hostUpload=0x%lx\n",
                   (unsigned long)upload, (unsigned long)(upload % 4096),
                   (unsigned long)gpu, (unsigned long)hostPtr);
            failCount++;
        }
    }
#endif
}
