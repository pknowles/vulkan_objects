// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <functional>
#include <stdexcept>
#include <vko/allocator.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>
#include <vko/shortcuts.hpp>
#include <vko/timeline_queue.hpp>
#include <vko/unique_any.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

template <class T>
concept staging_allocator =
    allocator<typename T::Allocator> && device_and_commands<typename T::DeviceAndCommands> &&
    std::same_as<decltype(T::TryCanFail), const bool> &&
    requires(T t, size_t size, const SemaphoreValue& releaseSemaphore,
             // Callback for using the temporary staging buffer
             std::function<void(const BoundBuffer<int, typename T::Allocator>&)> populate,
             // Callback just before the buffer is destroyed. Passed true if the
             // buffer's semaphore is signalled. False if the staging allocator
             // is destroyed before endBatch() is called.
             std::function<void(bool)> destruct) {
        // Try to allocate a staging buffer up to the given size. A partial
        // allocation is allowed and multiple staging buffers may be needed to
        // complete the staging operation. If this returns false, the staging
        // allocator is full and the caller should call endBatch() to allow
        // cycling old buffers. This call may block until a buffer is ready,
        // depending on the implementation strategy. Note that the backing
        // allocator can always throw too.
        { t.template tryWith<int>(size, populate, destruct) } -> std::same_as<bool>;
        // Same as tryWith(), but returns a pointer to the allocated buffer.
        // Returns nullptr if allocation fails.
        {
            t.template tryMake<int>(size, destruct)
        } -> std::same_as<BoundBuffer<int, typename T::Allocator>*>;
        // Provide a ready-semaphore for all buffers allocated since the last
        // call to endBatch(). Buffers cannot be destroyed until the
        // semaphore is signalled.
        { t.endBatch(releaseSemaphore) } -> std::same_as<void>;
        // Total bytes currently in use by staging buffers (current + unreleased batches)
        { t.size() } -> std::same_as<VkDeviceSize>;
        // Total allocated staging memory (including recyclable/idle memory)
        { t.capacity() } -> std::same_as<VkDeviceSize>;
        { t.device() } -> std::same_as<const typename T::DeviceAndCommands&>;
    };

// Upload generated data. The user provides a function that writes to the mapped
// staging buffer.
template <staging_allocator Staging, class Fn, buffer DstBuffer>
    requires (!Staging::TryCanFail)
void upload(Staging& staging, VkCommandBuffer cmd, const DstBuffer& dstBuffer, VkDeviceSize offset,
            VkDeviceSize size, Fn&& fn) {
    auto* stagingBuf = staging.template tryMake<typename DstBuffer::ValueType>(size, [](bool) {});
    assert(stagingBuf);
    fn(stagingBuf->map().span());
    copyBuffer<typename Staging::DeviceAndCommands, decltype(*stagingBuf), DstBuffer>(
        staging.device(), cmd, *stagingBuf, offset, dstBuffer, offset, size);
}

// Upload a range of existing data directly
template <staging_allocator Staging, std::ranges::contiguous_range SrcRange, buffer DstBuffer>
    requires(!Staging::TryCanFail) &&
            std::is_trivially_assignable_v<typename DstBuffer::ValueType,
                                           typename std::ranges::range_value_t<SrcRange>>
void upload(Staging& staging, VkCommandBuffer cmd, SrcRange&& data, const DstBuffer& dstBuffer) {
    using T          = std::remove_cv_t<std::ranges::range_value_t<SrcRange>>;
    auto* stagingBuf = staging.template tryMake<T>(std::ranges::size(data), [](bool) {});
    assert(stagingBuf);
    std::ranges::copy(data, stagingBuf->map().begin());
    copyBuffer<typename Staging::DeviceAndCommands, decltype(*stagingBuf), DstBuffer>(
        staging.device(), cmd, *stagingBuf, 0, dstBuffer, 0, std::ranges::size(data));
}

// Download data to a staging buffer. The caller is responsible for get()ing the
// result before the staging buffer gets released.
template <staging_allocator Staging, buffer SrcBuffer, class SV, class Fn>
    requires (!Staging::TryCanFail)
LazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>
download(Staging& staging, VkCommandBuffer cmd, SV&& submitPromise, const SrcBuffer& srcBuffer,
         VkDeviceSize offset, VkDeviceSize size, Fn&& fn) {
    auto* stagingBuf = staging.template tryMake<typename SrcBuffer::ValueType>(size, [](bool) {});
    assert(stagingBuf);
    copyBuffer<typename Staging::DeviceAndCommands, SrcBuffer, decltype(*stagingBuf)>(
        staging.device(), cmd, srcBuffer, offset, *stagingBuf, 0, size);
    return LazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>(
        std::forward<SV>(submitPromise),
        [mapping = stagingBuf->map(), fn = std::forward<Fn>(fn)]() { return fn(mapping.span()); });
}

// Staging buffer allocator that allocates individual staging buffers and
// maintains ownership of them. The caller must keep the DedicatedStaging object
// alive until all buffers are no longer in use, i.e. synchronize with the GPU
// so any command buffers referencing them have finished execution. Use this
// when making intermittent and large one-off transfers such as during
// initialization.
template <device_and_commands DeviceAndCommandsType = Device,
          allocator           AllocatorType         = vko::vma::Allocator>
struct DedicatedStaging {
public:
    using DeviceAndCommands          = DeviceAndCommandsType;
    using Allocator                  = AllocatorType;
    static constexpr bool TryCanFail = false;
    DedicatedStaging(const DeviceAndCommands& device, Allocator& allocator,
                     VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : m_device(device)
        , m_allocator(allocator)
        , m_bufferUsageFlags(usage) {}

    ~DedicatedStaging() {
        // Call unreleased destruct callbacks with false
        for (auto& callback : m_current.destroyCallbacks) {
            callback(false);
        }
        // Call released but uncompleted destruct callbacks with false
        m_released.visitAll([](Batch& batch) {
            for (auto& callback : batch.destroyCallbacks) {
                callback(false);
            }
        });
    }

    template <class T>
    bool tryWith(size_t size, std::function<void(const vko::BoundBuffer<T, Allocator>&)> populate,
                 std::function<void(bool)> destruct) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            m_device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += ptr->sizeBytes();
        m_totalSize += ptr->sizeBytes();
        populate(*ptr);
        m_current.buffers.emplace_back(std::move(any));
        m_current.destroyCallbacks.emplace_back(std::move(destruct));
        return true; // Always succeeds with full allocation
    }

    // Returns a non-owning pointer to a temporary buffer to use for staging
    // memory
    template <class T>
    vko::BoundBuffer<T, Allocator>* tryMake(size_t size,
                                             std::function<void(bool)> destruct) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            m_device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += ptr->sizeBytes();
        m_totalSize += ptr->sizeBytes();
        m_current.buffers.emplace_back(std::move(any));
        m_current.destroyCallbacks.emplace_back(std::move(destruct));
        return ptr; // Always succeeds with full allocation
    }

    void endBatch(SemaphoreValue releaseSemaphore) {
        // Move current batch to completion queue
        m_released.push_back(std::move(m_current), releaseSemaphore);
        m_current = {};

        // Greedily check for completed semaphores and call callbacks
        checkAndInvokeCompletedCallbacks();
    }

    void wait() {
        // Wait for all and invoke callbacks
        m_released.waitAndConsume(m_device.get(), [this](Batch& batch) {
            for (auto& callback : batch.destroyCallbacks) {
                callback(true);
            }
            m_totalSize -= batch.size;
        });
    }

    VkDeviceSize size() const { return m_totalSize; }
    VkDeviceSize capacity() const { return size(); } // Same as size for DedicatedStaging
    const DeviceAndCommands& device() const { return m_device.get(); }

private:
    void checkAndInvokeCompletedCallbacks() {
        // Greedily invoke and cleanup completed batches
        m_released.consumeReady(m_device.get(), [this](Batch& batch) {
            for (auto& callback : batch.destroyCallbacks) {
                callback(true);
            }
            m_totalSize -= batch.size;
        });
    }

    struct Batch {
        std::vector<UniqueAny>                  buffers;
        std::vector<std::function<void(bool)>> destroyCallbacks;
        VkDeviceSize                            size = 0;
    };

    Batch                                           m_current;
    CompletionQueue<Batch>                          m_released;
    std::reference_wrapper<const DeviceAndCommands> m_device;
    std::reference_wrapper<Allocator>               m_allocator;
    VkDeviceSize                                    m_totalSize        = 0;
    VkBufferUsageFlags                              m_bufferUsageFlags = 0;
};
static_assert(staging_allocator<DedicatedStaging<Device, vko::vma::Allocator>>);

namespace vma {

// Align a value up to the nearest multiple of alignment
constexpr VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Staging buffer allocator that provides temporary buffers from cyclical memory
// pools. The caller must call endBatch() with a SemaphoreValue that will
// be signaled when the buffers are no longer in use so that their pool can be
// reused. Use this when you have many small transfers to make on a regular
// basis to avoid frequent allocations/deallocations.
template <device_and_commands DeviceAndCommandsType>
class RecyclingStagingPool {
public:
    using DeviceAndCommands        = DeviceAndCommandsType;
    using Allocator                = vma::Allocator;
    static constexpr bool TryCanFail = true;

    RecyclingStagingPool(
        const DeviceAndCommands& device, Allocator& allocator, size_t minPools = 3,
        std::optional<size_t> maxPools      = 5 /* explicit std::nullopt for unlimited */,
        VkDeviceSize          poolSizeBytes = 1 << 24 /* 16MB * 5 = 80MB */,
        VkBufferUsageFlags    usage         = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : m_device(device)
        , m_allocator(allocator)
        , m_poolSize(poolSizeBytes)
        , m_bufferUsageFlags(usage)
        , m_maxPools(maxPools.value_or(0))
        , m_minPools(minPools) {
        
        // Query alignment requirement for this usage
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
        vko::Buffer tempBuffer(device, tempBufferInfo);
        VkMemoryRequirements req;
        device.vkGetBufferMemoryRequirements(device, tempBuffer, &req);
        m_alignment = req.alignment;

        // Pre-allocate minimum pools as already-signaled batches
        for (size_t i = 0; i < m_minPools; ++i) {
            PoolBatch batch;
            batch.pools.push_back(makePool());
            m_inUse.push_back(std::move(batch), SemaphoreValue::makeSignalled());
            m_totalPoolBytes += m_poolSize;
            ++m_totalPoolCount;
        }
    }

    RecyclingStagingPool(const RecyclingStagingPool& other) = delete;
    RecyclingStagingPool& operator=(const RecyclingStagingPool& other) = delete;
    RecyclingStagingPool& operator=(RecyclingStagingPool&& other) noexcept {
        // Notify all remaining handlers they're never going to get called
        for(auto& callback : m_current.destroyCallbacks) {
            callback(false);
        }
        m_inUse.visitAll([](PoolBatch& batch) {
            for(auto& callback : batch.destroyCallbacks) {
                callback(false);
            }
        });
        m_device = other.m_device;
        m_allocator = other.m_allocator;
        m_poolSize = other.m_poolSize;
        m_bufferUsageFlags = other.m_bufferUsageFlags;
        m_maxPools = other.m_maxPools;
        m_minPools = other.m_minPools;
        m_inUse = std::move(other.m_inUse);
        m_current = std::move(other.m_current);
        m_totalPoolBytes = other.m_totalPoolBytes;
        m_totalPoolCount = other.m_totalPoolCount;
        m_totalBufferBytes = other.m_totalBufferBytes;
        m_alignment = other.m_alignment;
        m_currentPoolUsedBytes = other.m_currentPoolUsedBytes;
        return *this;
    }
    RecyclingStagingPool(RecyclingStagingPool&& other) noexcept
        : m_device(other.m_device)
        , m_allocator(other.m_allocator)
        , m_poolSize(other.m_poolSize)
        , m_totalPoolBytes(other.m_totalPoolBytes)
        , m_totalPoolCount(other.m_totalPoolCount)
        , m_totalBufferBytes(other.m_totalBufferBytes)
        , m_current(std::move(other.m_current))
        , m_inUse(std::move(other.m_inUse))
        , m_bufferUsageFlags(other.m_bufferUsageFlags)
        , m_maxPools(other.m_maxPools)
        , m_minPools(other.m_minPools)
        , m_alignment(other.m_alignment)
        , m_currentPoolUsedBytes(other.m_currentPoolUsedBytes) {}
    ~RecyclingStagingPool() {
        // Notify all remaining handlers they're never going to get called
        for(auto& callback : m_current.destroyCallbacks) {
            callback(false);
        }
        m_inUse.visitAll([](PoolBatch& batch) {
            for(auto& callback : batch.destroyCallbacks) {
                callback(false);
            }
        });
    }

    template <class T>
    vko::BoundBuffer<T, Allocator>* tryMake(size_t size,
                                             std::function<void(bool)> destruct) {
        auto result = makeTmpBuffer<T>(size);
        if (result) {
            m_current.destroyCallbacks.push_back(std::move(destruct));
        }
        return result;
    }

    template <class T>
    bool tryWith(size_t size, std::function<void(const BoundBuffer<T, Allocator>&)> populate,
                 std::function<void(bool)> destruct) {
        auto result = makeTmpBuffer<T>(size);
        if (result) {
            populate(*result);
            m_current.destroyCallbacks.push_back(std::move(destruct));
            return true;
        }
        return false;
    }

    // Mark all buffers in the 'current' pools as free to be recycled once the
    // reuseSemaphore is signaled
    void endBatch(SemaphoreValue reuseSemaphore) {
        // Greedily invoke ready callbacks of previous batches that have been
        // signalled.
        invokeReadyCallbacks();

        // Only move batch to m_inUse if it has pools (RAII: no empty batches)
        if (!m_current.pools.empty()) {
            m_inUse.push_back(std::move(m_current), reuseSemaphore);
        }
        assert(m_current.buffers.empty());
        assert(m_current.destroyCallbacks.empty());
        m_current = {};
    }

    // Wait for all batches to finish
    void wait() {
        // Wait for all batches
        m_inUse.wait(m_device.get());
        
        // Process ALL ready batches (visitAll since they're all ready after wait())
        // Clean up buffers and invoke callbacks, then remove empty batches
        m_inUse.visitAll([this](PoolBatch& batch) {
            // Invoke callbacks and clear buffers
            for (auto& callback : batch.destroyCallbacks) {
                callback(true);
            }
            batch.destroyCallbacks.clear();
            batch.buffers.clear();
            m_totalBufferBytes -= batch.bufferBytes;
            batch.bufferBytes = 0;
        });
        
        freeExcessPools();
    }

    VkDeviceSize capacity() const { return m_totalPoolBytes; }
    VkDeviceSize size() const { return m_totalBufferBytes; }
    const DeviceAndCommands& device() const { return m_device.get(); }

    // Size of a single pool. Best to submit and endBatch() before size()
    // reaches this value.
    VkDeviceSize poolSize() const { return m_poolSize; }

private:

    struct PoolBatch {
        std::vector<vma::Pool>                  pools;
        std::vector<UniqueAny>                  buffers;
        std::vector<std::function<void(bool)>> destroyCallbacks;
        VkDeviceSize                            bufferBytes = 0;
    };

    bool hasCurrentPool() const {
        return !m_current.pools.empty();
    }

    vma::Pool& currentPool() {
        return m_current.pools.back();
    }

    vma::Pool makePool() {
        VkBufferCreateInfo sampleBufferCreateInfo = {
            .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext                 = nullptr,
            .flags                 = 0,
            .size                  = m_poolSize,
            .usage                 = m_bufferUsageFlags,
            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };
        VmaAllocationCreateInfo sampleAllocCreateInfo = {
            .flags = vma::defaultAllocationCreateFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            .usage = VMA_MEMORY_USAGE_UNKNOWN,  // Use legacy mode with only requiredFlags (supports both upload/download)
            .requiredFlags =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = nullptr,
            .priority       = 0.0f,
        };
        uint32_t memTypeIndex = 0;
        check(vmaFindMemoryTypeIndexForBufferInfo(m_allocator.get(), &sampleBufferCreateInfo,
                                                  &sampleAllocCreateInfo, &memTypeIndex));

        VmaPoolCreateInfo poolCreateInfo = {
            .memoryTypeIndex        = memTypeIndex,
            .flags                  = 0,
            .blockSize              = m_poolSize,
            .minBlockCount          = 1,
            .maxBlockCount          = 1,
            .priority               = 0.0f,
            .minAllocationAlignment = 0,
            .pMemoryAllocateNext    = nullptr,
        };
        return vma::Pool(m_allocator.get(), poolCreateInfo);
    }

    template <class T>
    BoundBuffer<T, Allocator>* makeTmpBuffer(VkDeviceSize size) {
        // Ensure we have a pool with space for at least one element
        if (!ensureCurrentPoolHasSpace(sizeof(T))) {
            return nullptr; // No pools available
        }

        // Calculate how much we can allocate from current pool
        VkDeviceSize availableBytes = m_poolSize - m_currentPoolUsedBytes;
        VkDeviceSize trySize = std::min(size, availableBytes / static_cast<VkDeviceSize>(sizeof(T)));
        assert(trySize > 0); // ensureCurrentPoolHasSpace should guarantee this

        // Allocate from current pool - if this fails, it's a real VMA error
        auto buffer = allocateFromCurrentPool<T>(trySize);
        
        auto [any, ptr] = toUniqueAnyWithPtr(std::move(buffer));
        m_current.buffers.emplace_back(std::move(any));
        return ptr;
    }

    // Ensures we have a pool with space for at least one element of given size
    bool ensureCurrentPoolHasSpace(size_t elementSize) {
        // Greedily invoke ready callbacks
        invokeReadyCallbacks();

        // Check if current pool has space
        if (hasCurrentPool() && (m_poolSize - m_currentPoolUsedBytes) >= elementSize) {
            return true;
        }

        // Need a new pool
        return addPoolToCurrentBatch();
    }

    // Adds a pool to current batch (recycled if available, otherwise new)
    // Returns false if at max capacity and no pools are ready
    bool addPoolToCurrentBatch() {
        // Try to recycle from ready batches (non-blocking)
        while (!m_inUse.empty() && m_inUse.frontSemaphore().isSignaled(m_device.get())) {
            auto& front = m_inUse.front(m_device.get());
            
            // Cleanup: invoke callbacks and clear buffers
            m_totalBufferBytes -= front.bufferBytes;
            front.buffers.clear();
            for (auto& callback : front.destroyCallbacks) {
                callback(true);
            }
            front.destroyCallbacks.clear();
            front.bufferBytes = 0;
            
            // Skip empty batches (should not exist, but defensive)
            if (front.pools.empty()) {
                m_inUse.pop_front();
                continue;
            }
            
            // Recycle one pool
            recyclePool(front);
            
            // Remove batch if it's now empty (no empty batches allowed)
            if (front.pools.empty()) {
                m_inUse.pop_front();
            }
            
            return true;
        }
        
        // Allocate new if below max
        if (m_maxPools == 0 || m_totalPoolCount < m_maxPools) {
            m_current.pools.push_back(makePool());
            m_totalPoolBytes += m_poolSize;
            ++m_totalPoolCount;
            m_currentPoolUsedBytes = 0;
            return true;
        }
        
        // At max, block for a pool
        while (!m_inUse.empty()) {
            auto& front = m_inUse.front(m_device.get()); // BLOCKS
            
            // Cleanup: invoke callbacks and clear buffers
            m_totalBufferBytes -= front.bufferBytes;
            front.buffers.clear();
            for (auto& callback : front.destroyCallbacks) {
                callback(true);
            }
            front.destroyCallbacks.clear();
            front.bufferBytes = 0;
            
            // Skip empty batches (should not exist, but defensive)
            if (front.pools.empty()) {
                m_inUse.pop_front();
                continue;
            }
            
            // Recycle one pool
            recyclePool(front);
            
            // Remove batch if it's now empty (no empty batches allowed)
            if (front.pools.empty()) {
                m_inUse.pop_front();
            }
            
            return true;
        }
        
        return false;
    }

    // Allocates a buffer from the current pool and tracks usage
    // Assumes hasCurrentPool() is true and size calculation is correct
    template <class T>
    BoundBuffer<T, Allocator> allocateFromCurrentPool(VkDeviceSize size) {
        assert(hasCurrentPool());
        
        auto buffer = BoundBuffer<T, Allocator>(
            m_device.get(), size, m_bufferUsageFlags,
            vma::allocationCreateInfo(currentPool(),
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            m_allocator.get());

        // Track usage
        VkDeviceSize allocatedBytes = buffer.sizeBytes();
        m_current.bufferBytes += allocatedBytes;
        m_totalBufferBytes += allocatedBytes;
        m_currentPoolUsedBytes = align_up(m_currentPoolUsedBytes + allocatedBytes, m_alignment);

        return buffer;
    }

    void invokeReadyCallbacks() {
        // Greedily check released pools and invoke callbacks if ready
        m_inUse.visitReady(m_device.get(), [](PoolBatch& batch) {
            for (auto& callback : batch.destroyCallbacks) {
                callback(true);
            }
            batch.destroyCallbacks.clear();
            return true;
        });
    }

    // Takes a pool from batch, adds to current.
    // Precondition: batch.pools is not empty
    // Postcondition: batch.pools may become empty (caller should check and remove)
    void recyclePool(PoolBatch& batch) {
        assert(!batch.pools.empty());
        
        m_current.pools.push_back(std::move(batch.pools.back()));
        batch.pools.pop_back();
        
        // Reset usage tracker for the new pool
        m_currentPoolUsedBytes = 0;
        
        // Reuse the batch's vector allocations if it's now empty
        if (batch.pools.empty() && m_current.buffers.empty()) {
            m_current.buffers.swap(batch.buffers);
        }
    }

    void freeExcessPools() {
        invokeReadyCallbacks();
        
        // Free excess pools from ready batches
        m_inUse.consumeReady(m_device.get(), [this](PoolBatch& batch) {
            // Free pools from this batch until we're at minPools
            while (m_totalPoolCount > m_minPools && !batch.pools.empty()) {
                m_totalPoolBytes -= m_poolSize;
                batch.pools.pop_back();
                --m_totalPoolCount;
            }
            
            // Consume empty batches immediately (RAII: no empty batches)
            return batch.pools.empty();
        });
    }

    std::reference_wrapper<const DeviceAndCommands> m_device;
    std::reference_wrapper<Allocator>               m_allocator;
    VkDeviceSize                                    m_poolSize = 0;
    VkDeviceSize                                    m_totalPoolBytes = 0;
    size_t                                          m_totalPoolCount = 0;
    VkDeviceSize                                    m_totalBufferBytes = 0;
    PoolBatch                                       m_current;
    CompletionQueue<PoolBatch>                      m_inUse;
    VkBufferUsageFlags                              m_bufferUsageFlags = 0;
    size_t                                          m_maxPools = 0;
    size_t                                          m_minPools = 0;
    VkDeviceSize                                    m_alignment = 0;
    VkDeviceSize                                    m_currentPoolUsedBytes = 0;
};
static_assert(staging_allocator<RecyclingStagingPool<Device>>);

} // namespace vma

// Staging wrapper that also holds a queue and command buffer for
// staging/streaming memory and automatically submits command buffers as needed
// to reduce staging memory usage.
template<class StagingAllocator>
class StreamingStaging {
public:
    using DeviceAndCommands = typename StagingAllocator::DeviceAndCommands;
    StreamingStaging(const TimelineQueue& queue, StagingAllocator&& staging, VkDeviceSize submitThresholdBytes)
        : m_queue(queue)
        , m_staging(std::move(staging))
        , m_commandPool(m_staging.device(), VkCommandPoolCreateInfo{
                                    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .pNext            = nullptr,
                                    .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = queue.familyIndex(),
                                })
        , m_submitThresholdBytes(submitThresholdBytes) {}

    template <class T>
    bool with(size_t size,
              std::function<void(const vko::BoundBuffer<T, typename StagingAllocator::Allocator>&, VkCommandBuffer)>
                  func) {
        return m_staging.template tryWith<T>(
            size, [&](const vko::BoundBuffer<T, typename StagingAllocator::Allocator>& tmpBuffer) {
                func(tmpBuffer, commandBuffer());
            }, [](bool) {});
    }

    template <buffer DstBuffer, class Fn>
    bool upload(const DstBuffer& dstBuffer, VkDeviceSize offset, VkDeviceSize size, Fn&& fn) {
        auto* stagingBuf = m_staging.template tryMake<typename DstBuffer::ValueType>(size, [](bool) {});
        if (!stagingBuf)
            return false;
        fn(stagingBuf->map().span());
        copyBuffer<DeviceAndCommands, decltype(*stagingBuf), DstBuffer>(
            m_staging.device(), commandBuffer(), *stagingBuf, 0, dstBuffer, offset, size);

        // TODO: do before allocation? account for alignment? callback if
        // m_staging actually overflows?
        if (m_staging.size() > m_submitThresholdBytes) {
            submit();
        }
        return true;
    }

    // NOTE: callback may be called even if the return value is destroyed
    // TODO: void return specialization?
    template <buffer SrcBuffer, class Fn>
    std::optional<SharedLazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>> download(const SrcBuffer& srcBuffer, VkDeviceSize offset, VkDeviceSize size, Fn&& fn) {
        SharedLazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>> result(
            m_queue.get().nextSubmitSemaphore(),
            [fn=std::forward<Fn>(fn)]() { return fn(std::span<const typename SrcBuffer::ValueType>{}); });
        
        auto* stagingBuf = m_staging.template tryMake<typename SrcBuffer::ValueType>(
            size, result.evaluator(m_staging.device()));
        if (!stagingBuf)
            return std::nullopt;
            
        copyBuffer<DeviceAndCommands, SrcBuffer, decltype(*stagingBuf)>(
            m_staging.device(), commandBuffer(), srcBuffer, offset, *stagingBuf, 0, size);
        
        // Update the result with the actual mapping
        result = SharedLazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>(
            m_queue.get().nextSubmitSemaphore(),
            [mapping = stagingBuf->map(), fn=std::forward<Fn>(fn)]() { return fn(mapping.span()); });

        // TODO: do before allocation? account for alignment? callback if
        // m_staging actually overflows?
        if (m_staging.size() > m_submitThresholdBytes) {
            submit();
        }
        
        return result;
    }

    // Manual submission interface.
    void submit()
    {
        if (m_currentCmd) {
            CommandBuffer cmd(m_currentCmd->end());
            m_currentCmd.reset();
            auto semaphoreValue = m_queue.get().nextSubmitSemaphore();
            m_queue.get().submit(m_staging.device(), {}, cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT /* TODO: is this right? */);
            m_inFlightCmds.push_back({std::move(cmd), std::move(semaphoreValue)});
            m_staging.endBatch(semaphoreValue);
        }
    }

private:
    struct InFlightCmd {
        CommandBuffer  commandBuffer;
        SemaphoreValue readySemaphore;
    };

    VkCommandBuffer commandBuffer() {
        if (!m_currentCmd) {
            m_currentCmd.emplace(m_staging.device(), reuseOrMakeCommandBuffer(),
                                 VkCommandBufferBeginInfo{
                                     .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .pNext = nullptr,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                     .pInheritanceInfo = nullptr,
                                 });
        }
        return *m_currentCmd;
    }

    CommandBuffer reuseOrMakeCommandBuffer() {
        // Try to reuse a command buffer from the in-flight queue
        if (!m_inFlightCmds.empty() && m_inFlightCmds.front().readySemaphore.wait(m_queue.get(), 0)) {
            auto result = std::move(m_inFlightCmds.front().commandBuffer);
            m_inFlightCmds.pop_front();

            // If there's a relatively long queue of ready command buffers, free
            // some up. Leave at least one of the ready ones to avoid frequent
            // allocations/deallocations.
            if (m_inFlightCmds.size() >= 2 && m_inFlightCmds.front().readySemaphore.wait(m_queue.get(), 0)) {
                while (m_inFlightCmds.size() >= 2 &&
                    std::next(m_inFlightCmds.begin())->readySemaphore.wait(m_queue.get(), 0)) {
                    m_inFlightCmds.pop_front();
                }
            }

            m_staging.device().vkResetCommandBuffer(result, 0);
            return result;
        }

        // Else, create a new command buffer
        return CommandBuffer(m_staging.device(), nullptr, m_commandPool,
                             VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    }

    std::reference_wrapper<const TimelineQueue>   m_queue;
    StagingAllocator                              m_staging;
    CommandPool                                   m_commandPool;
    std::deque<InFlightCmd>                       m_inFlightCmds;
    std::optional<simple::RecordingCommandBuffer> m_currentCmd;
    VkDeviceSize                                  m_submitThresholdBytes = 0;
};

template <class T, class Fn, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
void uploadMapped(
    StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd, Fn&& fn,
    vko::DeviceBuffer<T, Allocator>& dstBuffer,
    size_t dstOffsetElements = 0u, std::optional<VkDeviceSize> size = std::nullopt) {
    if (dstOffsetElements > dstBuffer.size() ||
        (size && dstOffsetElements + *size > dstBuffer.size()))
        throw std::out_of_range("destination buffer is too small");
    if (dstOffsetElements == dstBuffer.size() || (size && *size == 0))
        throw std::runtime_error("uploading zero elements");
    VkDeviceSize subspanSize = size.value_or(dstBuffer.size() - dstOffsetElements);
    if (!staging.template tryWith<T>(
        subspanSize, [&](const vko::BoundBuffer<T, Allocator>& buffer) {
            fn(buffer.map().span());
            VkBufferCopy bufferCopy{
                .srcOffset = 0,
                .dstOffset = dstOffsetElements * sizeof(T),
                .size      = subspanSize * sizeof(T),
            };
            device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
        }, [](bool) {})) {
        throw std::runtime_error("Failed to allocate staging buffer");
    }
}

template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
void uploadTo(
    StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd, Range&& data,
    vko::DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<Range>>, Allocator>& dstBuffer,
    size_t dstOffsetElements = 0u) {
    using T = std::remove_cv_t<std::ranges::range_value_t<Range>>;
    if (std::ranges::empty(data))
        throw std::runtime_error("uploading empty data");
    if (dstOffsetElements + std::ranges::size(data) > dstBuffer.size())
        throw std::out_of_range("destination buffer is too small");
    if (!staging.template tryWith<T>(std::ranges::size(data),
                             [&](const vko::BoundBuffer<T, Allocator>& buffer) {
                                 std::ranges::copy(data, buffer.map().begin());
                                 VkBufferCopy bufferCopy{
                                     .srcOffset = 0,
                                     .dstOffset = dstOffsetElements * sizeof(T),
                                     .size      = std::ranges::size(data) * sizeof(T),
                                 };
                                 device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
                             }, [](bool) {})) {
        throw std::runtime_error("Failed to allocate staging buffer");
    }
}

// Non type-safe upload. Avoid where possible. The returned BufferMapping must
// not outlive the staging memory.
template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<Range>>, std::byte>
void
uploadToBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd, Range&& data,
              VkBuffer dstBuffer, VkDeviceSize dstOffset) {
    if (std::ranges::empty(data))
        throw std::runtime_error("uploading empty data");
    if (!staging.template tryWith<std::byte>(
        std::ranges::size(data), [&](const vko::BoundBuffer<std::byte, typename StagingAllocator::Allocator>& buffer) {
            std::ranges::copy(data, buffer.map().begin());
            VkBufferCopy bufferCopy{
                .srcOffset = 0,
                .dstOffset = dstOffset,
                .size      = std::ranges::size(data),
            };
            device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
        }, [](bool) {})) {
        throw std::runtime_error("Failed to allocate staging buffer");
    }
}

template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<Range>>, Allocator>
upload(StagingAllocator& staging, const DeviceAndCommands& device, Allocator& allocator,
       VkCommandBuffer cmd, Range&& data,
       VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    using T = std::remove_cv_t<std::ranges::range_value_t<Range>>;
    if (std::ranges::empty(data))
        throw std::runtime_error("uploading empty data");
    vko::DeviceBuffer<T, Allocator> result(device, std::ranges::size(data), usage,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    uploadTo(staging, device, cmd, data, result);
    return result;
}

template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
void immediateUploadTo(StagingAllocator& staging, const DeviceAndCommands& device,
                       VkCommandPool pool, VkQueue queue, Range&& data,
                       const vko::DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<Range>>,
                                               Allocator>& dstBuffer,
                       size_t                              dstOffsetElements = 0u) {
    vko::simple::ImmediateCommandBuffer cmd(device, pool, queue);
    uploadTo(staging, device, cmd, std::forward<Range>(data), dstBuffer, dstOffsetElements);
}

template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<Range>>, Allocator>
immediateUpload(StagingAllocator& staging, const DeviceAndCommands& device, Allocator& allocator,
                VkCommandPool pool, VkQueue queue, Range&& data,
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    using T = std::remove_cv_t<std::ranges::range_value_t<Range>>;
    if (std::ranges::empty(data))
        throw std::runtime_error("uploading empty data");
    vko::DeviceBuffer<T, Allocator> result(device, std::ranges::size(data), usage,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    immediateUploadTo(staging, device, pool, queue, std::forward<Range>(data), result);
    return result;
}

template <class Fn, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
void uploadMappedBytes(
    StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd, Fn&& fn,
    VkBuffer dstBuffer,
    size_t offset, VkDeviceSize size) {
    if (!staging.template tryWith<std::byte>(
        size, [&](const vko::BoundBuffer<std::byte, Allocator>& buffer) {
            fn(buffer.map().span());
            VkBufferCopy bufferCopy{
                .srcOffset = 0,
                .dstOffset = offset,
                .size      = size,
            };
            device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
        }, [](bool) {})) {
        throw std::runtime_error("Failed to allocate staging buffer");
    }
}

// TODO: this is totally not safe, but I've already learned this structure and
// not sure how to do it better yet
#if 1
// DANGER: The returned BufferMapping is populated once the command buffer is
// submitted and has completed execution. It does not include the staging offset
// or size. It must not outlive the staging memory. The caller is responsible
// for inserting the appropriate memory barriers.
template <class T, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
BufferMapping<T, Allocator>
download(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
         vko::DeviceBuffer<T, Allocator>& srcBuffer, VkDeviceSize srcOffsetElements = 0u, std::optional<VkDeviceSize> srcSize = std::nullopt) {
    if (srcOffsetElements >= srcBuffer.size())
        throw std::out_of_range("source offset is beyond buffer size");
    VkDeviceSize copySize = srcSize.value_or(srcBuffer.size() - srcOffsetElements);
    auto* buffer = staging.template tryMake<T>(copySize, [](bool) {});
    if (!buffer)
        throw std::runtime_error("Failed to allocate staging buffer");
    VkBufferCopy bufferCopy{
        .srcOffset = srcOffsetElements * sizeof(T),
        .dstOffset = 0,
        .size      = copySize * sizeof(T),
    };
    device.vkCmdCopyBuffer(cmd, srcBuffer, *buffer, 1, &bufferCopy);
    return buffer->map();
}
#endif

#if 1
// Non type-safe download. Avoid where possible.
// DANGER: returns a mapping for the entire buffer, not the offset/size that was copied
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
BufferMapping<std::byte, Allocator>
downloadBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
              VkBuffer srcBuffer, VkDeviceSize offset, size_t size) {
    auto* buffer = staging.template tryMake<std::byte>(size, [](bool) {});
    if (!buffer)
        throw std::runtime_error("Failed to allocate staging buffer");
    VkBufferCopy bufferCopy{.srcOffset = offset, .dstOffset = 0, .size = size};
    device.vkCmdCopyBuffer(cmd, srcBuffer, *buffer, 1, &bufferCopy);
    return buffer->map();
}
#endif

template <std::ranges::contiguous_range DstRange, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
    requires(!std::is_const_v<std::ranges::range_value_t<DstRange>>)
void immediateDownloadTo(
    StagingAllocator& staging, const DeviceAndCommands& device, VkCommandPool pool, VkQueue queue,
    vko::DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<DstRange>>, Allocator>& srcBuffer,
    size_t srcOffsetElements, DstRange&& dstRange,
    std::optional<MemoryAccess> srcAccess = std::nullopt) {
    if (srcOffsetElements + std::ranges::size(dstRange) > srcBuffer.size())
        throw std::out_of_range("source buffer is too small for requested download");
    using T = std::remove_cv_t<std::ranges::range_value_t<DstRange>>;
    BufferMapping<T, Allocator> mapping = [&]() {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        if (srcAccess)
            cmdMemoryBarrier(device, cmd, *srcAccess,
                             {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
        return download<T>(staging, device, cmd, srcBuffer, srcOffsetElements, std::ranges::size(dstRange));
        // ImmediateCommandBuffer destructor is called here - submits and waits
    }();
    // After this point, the GPU copy should be complete and visible
    std::ranges::copy(mapping, std::ranges::begin(dstRange));
}

template <class T, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
std::vector<T> immediateDownload(StagingAllocator& staging, const DeviceAndCommands& device,
                                 VkCommandPool pool, VkQueue queue,
                                 vko::DeviceBuffer<T, Allocator>& srcBuffer,
                                 size_t                           srcOffsetElements = 0u,
                                 std::optional<MemoryAccess>      srcAccess = std::nullopt) {
    std::vector<T> result(srcBuffer.size());
    immediateDownloadTo(staging, device, pool, queue, srcBuffer, srcOffsetElements, result, srcAccess);
    return result;
}

template <std::ranges::contiguous_range DstRange, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
void immediateDownloadToBytes(StagingAllocator& staging, const DeviceAndCommands& device,
                              VkCommandPool pool, VkQueue queue, VkBuffer buffer,
                              VkDeviceSize offset, VkDeviceSize size, DstRange&& dstRange,
                              std::optional<MemoryAccess> srcAccess = std::nullopt) {
    if (size > std::ranges::size(dstRange))
        throw std::out_of_range("destination range is too small");
    auto mapping = [&]() {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        if (srcAccess)
            cmdMemoryBarrier(device, cmd, *srcAccess,
                             {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
        return downloadBytes(staging, device, cmd, buffer, offset, size);
        // ImmediateCommandBuffer destructor is called here - submits and waits
    }();
    // After this point, the GPU copy should be complete and visible
    std::ranges::copy(mapping, std::ranges::begin(dstRange));
}

template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
std::vector<std::byte>
immediateDownloadBytes(StagingAllocator& staging, const DeviceAndCommands& device,
                       VkCommandPool pool, VkQueue queue, VkBuffer buffer, VkDeviceSize offset,
                       VkDeviceSize size, std::optional<MemoryAccess> srcAccess = std::nullopt) {
    std::vector<std::byte> result(size);
    immediateDownloadToBytes(staging, device, pool, queue, buffer, offset, size, result, srcAccess);
    return result;
}

#if 1
// Non type-safe download. Avoid where possible. Downloads data from a device
// address using VkCmdCopyMemoryIndirectNV. Requires VK_NV_copy_memory_indirect
// to be enabled and the staging allocator must include
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
BufferMapping<std::byte, typename StagingAllocator::Allocator>
downloadBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
              VkDeviceAddress address, VkDeviceSize size) {
    // Temporary host visible staging buffer
    auto* dstBuffer = staging.template tryMake<std::byte>(size, [](bool) {});
    if (!dstBuffer)
        throw std::runtime_error("Failed to allocate staging buffer");

    // Temporary buffer to hold the VkCopyMemoryIndirectCommandNV
    auto* cmdBuffer = staging.template tryMake<VkCopyMemoryIndirectCommandNV>(1, [](bool) {});
    if (!cmdBuffer)
        throw std::runtime_error("Failed to allocate staging buffer");
    VkDeviceAddress dstAddress = device.vkGetBufferDeviceAddress(
        device, tmpPtr(VkBufferDeviceAddressInfo{
                    .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .pNext  = nullptr,
                    .buffer = *dstBuffer,
                }));
    VkDeviceAddress cmdAddress = device.vkGetBufferDeviceAddress(
        device, tmpPtr(VkBufferDeviceAddressInfo{
                    .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .pNext  = nullptr,
                    .buffer = *cmdBuffer,
                }));
    cmdBuffer->map()[0] = VkCopyMemoryIndirectCommandNV{
        .srcAddress = address,
        .dstAddress = dstAddress,
        .size       = size,
    };

    device.vkCmdCopyMemoryIndirectNV(cmd, cmdAddress, 1,
                                     static_cast<uint32_t>(cmdBuffer->sizeBytes()));
    return dstBuffer->map();
}
#endif

// Non type-safe download. Avoid where possible.
template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<Range>>, std::byte>
void immediateDownloadBytesTo(StagingAllocator& staging, const DeviceAndCommands& device,
                              VkCommandPool pool, VkQueue queue, VkDeviceAddress address,
                              VkDeviceSize size, Range&& dstRange, std::optional<MemoryAccess> srcAccess = std::nullopt) {
    auto mapping = [&] {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        if (srcAccess)
            cmdMemoryBarrier(device, cmd, *srcAccess,
                             {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});
        return downloadBytes(staging, device, cmd, address, size);
    }();
    std::ranges::copy(mapping, std::ranges::begin(dstRange));
}

// Non type-safe download. Avoid where possible.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
std::vector<std::byte> immediateDownloadBytes(StagingAllocator&        staging,
                                              const DeviceAndCommands& device, VkCommandPool pool,
                                              VkQueue queue, VkDeviceAddress address,
                                              VkDeviceSize size, std::optional<MemoryAccess> srcAccess = std::nullopt) {
    std::vector<std::byte> result(size);
    immediateDownloadBytesTo(staging, device, pool, queue, address, size, result, srcAccess);
    return result;
}

} // namespace vko
