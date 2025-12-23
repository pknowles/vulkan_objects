// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include "timeline_queue.hpp"
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vko/adapters.hpp>
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
    std::same_as<decltype(T::AllocateAlwaysSucceeds), const bool> &&
    std::same_as<decltype(T::AllocateAlwaysFull), const bool> &&
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
            t.template allocateUpTo<int>(size, destruct)
        } -> std::same_as<BoundBuffer<int, typename T::Allocator>*>;
        // Simpler overloads without per-buffer callbacks
        { t.template tryWith<int>(size, populate) } -> std::same_as<bool>;
        { t.template allocateUpTo<int>(size) } -> std::same_as<BoundBuffer<int, typename T::Allocator>*>;
        // Register a callback for the current batch
        { t.registerBatchCallback(destruct) } -> std::same_as<void>;
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
    requires (Staging::AllocateAlwaysSucceeds)
void upload(Staging& staging, VkCommandBuffer cmd, const DstBuffer& dstBuffer, VkDeviceSize offset,
            VkDeviceSize size, Fn&& fn) {
    auto* stagingBuf = staging.template allocateUpTo<typename DstBuffer::ValueType>(size);
    assert(stagingBuf);
    fn(stagingBuf->map().span());
    copyBuffer<typename Staging::DeviceAndCommands, decltype(*stagingBuf), DstBuffer>(
        staging.device(), cmd, *stagingBuf, offset, dstBuffer, offset, size);
}

// Upload a range of existing data directly
template <staging_allocator Staging, std::ranges::contiguous_range SrcRange, buffer DstBuffer>
    requires(Staging::AllocateAlwaysSucceeds) &&
            std::is_trivially_assignable_v<typename DstBuffer::ValueType,
                                           typename std::ranges::range_value_t<SrcRange>>
void upload(Staging& staging, VkCommandBuffer cmd, SrcRange&& data, const DstBuffer& dstBuffer) {
    using T          = std::remove_cv_t<std::ranges::range_value_t<SrcRange>>;
    auto* stagingBuf = staging.template allocateUpTo<T>(std::ranges::size(data));
    assert(stagingBuf);
    std::ranges::copy(data, stagingBuf->map().begin());
    copyBuffer<typename Staging::DeviceAndCommands, decltype(*stagingBuf), DstBuffer>(
        staging.device(), cmd, *stagingBuf, 0, dstBuffer, 0, std::ranges::size(data));
}

// Download data to a staging buffer. The caller is responsible for get()ing the
// result before the staging buffer gets released.
template <staging_allocator Staging, buffer SrcBuffer, class SV, class Fn>
    requires (Staging::AllocateAlwaysSucceeds)
LazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>
download(Staging& staging, VkCommandBuffer cmd, SV&& submitPromise, const SrcBuffer& srcBuffer,
         VkDeviceSize offset, VkDeviceSize size, Fn&& fn) {
    auto* stagingBuf = staging.template allocateUpTo<typename SrcBuffer::ValueType>(size);
    assert(stagingBuf);
    copyBuffer<typename Staging::DeviceAndCommands, SrcBuffer, decltype(*stagingBuf)>(
        staging.device(), cmd, srcBuffer, offset, *stagingBuf, 0, size);
    return LazyTimelineFuture<std::invoke_result_t<Fn, std::span<const typename SrcBuffer::ValueType>>>(
        std::forward<SV>(submitPromise),
        [mapping = stagingBuf->map(), fn = std::forward<Fn>(fn)]() { return fn(mapping.span()); });
}

// Staging buffer allocator that allocates individual staging buffers and
// maintains ownership of them. The caller must keep the DedicatedStagingPool object
// alive until all buffers are no longer in use, i.e. synchronize with the GPU
// so any command buffers referencing them have finished execution. Use this
// when making intermittent and large one-off transfers such as during
// initialization.
template <device_and_commands DeviceAndCommandsType = Device,
          allocator           AllocatorType         = vko::vma::Allocator>
struct DedicatedStagingPool {
public:
    using DeviceAndCommands          = DeviceAndCommandsType;
    using Allocator                  = AllocatorType;
    static constexpr bool AllocateAlwaysSucceeds = true;
    static constexpr bool AllocateAlwaysFull = true;
    DedicatedStagingPool(const DeviceAndCommands& device, Allocator& allocator,
                     VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : m_device(device)
        , m_allocator(allocator)
        , m_bufferUsageFlags(usage) {}

    DedicatedStagingPool(DedicatedStagingPool&&) = default;
    DedicatedStagingPool& operator=(DedicatedStagingPool&&) = default;

    ~DedicatedStagingPool() {
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

    // Primary implementation without per-buffer callback
    template <class T>
    bool tryWith(size_t size, std::function<void(const vko::BoundBuffer<T, Allocator>&)> populate) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            m_device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += ptr->sizeBytes();
        m_totalSize += ptr->sizeBytes();
        populate(*ptr);
        m_current.buffers.emplace_back(std::move(any));
        return true; // Always succeeds with full allocation
    }

    // Overload with per-buffer callback
    template <class T>
    bool tryWith(size_t size, std::function<void(const vko::BoundBuffer<T, Allocator>&)> populate,
                 std::function<void(bool)> destruct) {
        bool result = tryWith<T>(size, populate);
        if (result) {
            m_current.destroyCallbacks.emplace_back(std::move(destruct));
        }
        return result;
    }

    // Primary implementation: returns a non-owning pointer to a temporary buffer
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            m_device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += ptr->sizeBytes();
        m_totalSize += ptr->sizeBytes();
        m_current.buffers.emplace_back(std::move(any));
        return ptr; // Always succeeds with full allocation
    }

    // Overload with per-buffer callback
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size,
                                             std::function<void(bool)> destruct) {
        auto* ptr = allocateUpTo<T>(size);
        if (ptr) {
            m_current.destroyCallbacks.emplace_back(std::move(destruct));
        }
        return ptr;
    }

    // Register a callback for the current batch
    void registerBatchCallback(std::function<void(bool)> callback) {
        m_current.destroyCallbacks.emplace_back(std::move(callback));
    }

    void endBatch(SemaphoreValue releaseSemaphore) {
        // Move current batch to completion queue
        m_released.push_back(std::move(m_current), releaseSemaphore);
        m_current = {};
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
    VkDeviceSize capacity() const { return size(); } // Same as size for DedicatedStagingPool
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
static_assert(staging_allocator<DedicatedStagingPool<Device, vko::vma::Allocator>>);
static_assert(std::is_move_constructible_v<DedicatedStagingPool<Device, vko::vma::Allocator>>);
static_assert(std::is_move_assignable_v<DedicatedStagingPool<Device, vko::vma::Allocator>>);

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
    static constexpr bool AllocateAlwaysSucceeds = false;
    static constexpr bool AllocateAlwaysFull = false;

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
        destroyBuffers(m_current, false);
        m_inUse.visitAll([this](PoolBatch& batch) {
            destroyBuffers(batch, false);
        });
    }

    // Primary implementation without per-buffer callback
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size) {
        return makeTmpBuffer<T>(size);
    }

    // Overload with per-buffer callback
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size,
                                             std::function<void(bool)> destruct) {
        auto result = makeTmpBuffer<T>(size);
        if (result) {
            m_current.destroyCallbacks.push_back(std::move(destruct));
        }
        return result;
    }

    // Primary implementation without per-buffer callback
    template <class T>
    bool tryWith(size_t size, std::function<void(const BoundBuffer<T, Allocator>&)> populate) {
        auto result = makeTmpBuffer<T>(size);
        if (result) {
            populate(*result);
            return true;
        }
        return false;
    }

    // Overload with per-buffer callback
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

    // Register a callback for the current batch
    void registerBatchCallback(std::function<void(bool)> callback) {
        m_current.destroyCallbacks.push_back(std::move(callback));
    }

    // Mark all buffers in the 'current' pools as free to be recycled once the
    // reuseSemaphore is signaled
    void endBatch(SemaphoreValue reuseSemaphore) {
        // Only move batch to m_inUse if it has pools (RAII: no empty batches)
        if (!m_current.pools.empty()) {
            m_inUse.push_back(std::move(m_current), reuseSemaphore);
            m_current = {};
        }
        assert(m_current.buffers.empty());
        assert(m_current.destroyCallbacks.empty());
    }

    // Wait for all batches to finish
    void wait() {
        // Wait for all batches
        m_inUse.wait(m_device.get());
        
        // Process ALL ready batches (visitAll since they're all ready after wait())
        m_inUse.visitAll([this](PoolBatch& batch) {
            destroyBuffers(batch, true);
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

    // Invoke callbacks and free buffers from a batch
    // Callbacks are tied to buffer lifetime, so these always happen together
    void destroyBuffers(PoolBatch& batch, bool buffersReady) {
        // Invoke callbacks BEFORE clearing buffers
        // This allows download futures to evaluate chunks while buffers are still alive
        for (auto& callback : batch.destroyCallbacks) {
            callback(buffersReady);
        }
        batch.destroyCallbacks.clear();
        
        // Now safe to destroy buffers (mappings have been evaluated/cleared)
        m_totalBufferBytes -= batch.bufferBytes;
        batch.buffers.clear();
        batch.bufferBytes = 0;
    }

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
        // Try up to 2 times: once with current pool, once with a fresh pool
        for (int attempt = 0; attempt < 2; ++attempt) {
            // On second attempt, try to get a new pool
            if (attempt > 0) {
                if (!addPoolToCurrentBatch()) {
                    return nullptr; // Can't allocate a new pool
                }
            }
            
            // If we have a pool, try to allocate
            if (hasCurrentPool()) {
                VkDeviceSize availableBytes = m_poolSize - m_currentPoolUsedBytes;
                VkDeviceSize trySize = availableBytes / static_cast<VkDeviceSize>(sizeof(T));
                
                if (trySize > 0) {
                    trySize = std::min(size, trySize);
                    
                    // Allocate from current pool
                    // Note: This can still fail if our tracking is wrong due to VMA's
                    // internal fragmentation. In that case, we'll try with a fresh pool.
                    try {
                        auto buffer = allocateFromCurrentPool<T>(trySize);
                        auto [any, ptr] = toUniqueAnyWithPtr(std::move(buffer));
                        m_current.buffers.emplace_back(std::move(any));
                        return ptr;
                    } catch (const ResultException<VK_ERROR_OUT_OF_DEVICE_MEMORY>&) {
                        // VMA allocation failed despite our tracking saying we have space.
                        // This can happen due to internal fragmentation or alignment issues.
                        // Mark this pool as full and try again with a new pool.
                        m_currentPoolUsedBytes = m_poolSize;
                        if (attempt == 1) {
                            throw; // Already tried with fresh pool, propagate error
                        }
                        // Continue to attempt 1, which will get a new pool
                    }
                }
            }
        }
        
        return nullptr;
    }

    // Adds a pool to current batch (recycled if available, otherwise new)
    // Returns false if at max capacity and no pools are ready
    bool addPoolToCurrentBatch() {
        // Try to recycle from ready batches (non-blocking)
        while (!m_inUse.empty() && m_inUse.frontSemaphore().isSignaled(m_device.get())) {
            auto& front = m_inUse.front(m_device.get());
            destroyBuffers(front, true);
            
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
            destroyBuffers(front, true);
            
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
        m_inUse.visitReady(m_device.get(), [this](PoolBatch& batch) {
            destroyBuffers(batch, true);
            return true;
        });
    }

    // Takes a pool from batch, adds to current.
    // Precondition: batch.pools is not empty
    // Postcondition: batch.pools may become empty (caller should check and remove)
    void recyclePool(PoolBatch& batch) {
        assert(!batch.pools.empty());
        
        // Reuse the batch's vector allocations when possible
        if(m_current.buffers.empty() && batch.buffers.capacity() > m_current.buffers.capacity()) {
            m_current.buffers.swap(batch.buffers);
        }
        if(m_current.destroyCallbacks.empty() && batch.destroyCallbacks.capacity() > m_current.destroyCallbacks.capacity()) {
            m_current.destroyCallbacks.swap(batch.destroyCallbacks);
        }
        
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
        // Free excess pools from ready batches
        m_inUse.consumeReady(m_device.get(), [this](PoolBatch& batch) {
            destroyBuffers(batch, true);
            
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
static_assert(std::is_move_constructible_v<RecyclingStagingPool<Device>>);
static_assert(std::is_move_assignable_v<RecyclingStagingPool<Device>>);

} // namespace vma

// Structures for DownloadFuture
template <class T, class Allocator>
struct FutureSubrange {
    VkDeviceSize offset;
    std::optional<BufferMapping<T, Allocator>> mapping; // nullopt = evaluated or cancelled
    
    bool isEvaluated() const { return !mapping.has_value(); }
};

namespace {
// Internal State structures for DownloadFuture
// Base State template (for HasOutput = false, foreach case - no output vector)
template <class Fn, class T, class Allocator, bool HasOutput>
struct DownloadFutureState {
    std::vector<FutureSubrange<T, Allocator>> subranges;
    Fn producer;
    std::optional<SemaphoreValue> semaphore;  // Set by DownloadFuture constructor
    
    DownloadFutureState(Fn&& fn)
        : producer(std::forward<Fn>(fn)) {}
    
    // No cancellation for foreach - it just doesn't call the producer
    bool isCancelled() const { return false; }
    void cancel() { assert(!"Most likely a missing staging.submit()"); /* no-op for foreach */ }
};

// Specialization for HasOutput = true (transform case - with output vector)
template <class Fn, class T, class Allocator>
struct DownloadFutureState<Fn, T, Allocator, true> {
    std::vector<FutureSubrange<T, Allocator>> subranges;
    Fn producer;
    std::vector<T> output;
    std::optional<SemaphoreValue> semaphore;  // Set by DownloadFuture constructor
    
    DownloadFutureState(Fn&& fn, size_t outputSize)
        : producer(std::forward<Fn>(fn)), output(outputSize) {}
    
    // Cancelled = output cleared but subranges existed (not just zero-size)
    bool isCancelled() const { return output.empty() && !subranges.empty(); }
    void cancel() {
        assert(!"Most likely a missing staging.submit()");
        output.clear();
    }
};
} // anonymous namespace

// Builder for DownloadFuture - used internally by StagingStream during download allocation
template <class Fn, class T, class Allocator, bool HasOutput>
class DownloadFutureBuilder {
public:
    using Subrange = FutureSubrange<T, Allocator>;
    using State = DownloadFutureState<Fn, T, Allocator, HasOutput>;
    
    // Constructor for transform case (HasOutput = true)
    DownloadFutureBuilder(Fn&& producer, size_t outputSize) requires HasOutput
        : m_state(std::make_shared<State>(std::forward<Fn>(producer), outputSize)) {}
    
    // Constructor for foreach case (HasOutput = false)
    DownloadFutureBuilder(Fn&& producer) requires (!HasOutput)
        : m_state(std::make_shared<State>(std::forward<Fn>(producer))) {}
    
    // Add a subrange and return callback for StagingStream to register with staging allocator
    std::function<void(bool)> addSubrange(VkDeviceSize offset, 
                                          BufferMapping<T, Allocator>&& mapping) {
        size_t idx = m_state->subranges.size();
        m_state->subranges.push_back({offset, std::move(mapping)});
        
        // Return callback for StagingStream to register with staging allocator
        // Captures shared State so it outlives the builder
        if constexpr (HasOutput) {
            return [state = m_state, idx](bool ready) {
                auto& sub = state->subranges[idx];
                if (sub.isEvaluated()) {
                    return; // Already evaluated
                }
                
                if (ready && !state->isCancelled()) {
                    // GPU work complete and not cancelled: evaluate this subrange
                    auto outputSpan = std::span(state->output).subspan(
                        sub.offset, sub.mapping->span().size());
                    state->producer(sub.offset, sub.mapping->span(), outputSpan);
                } else if (!ready) {
                    // Cancellation requested: clear output
                    // Can happen before semaphore is set (e.g., staging pool destruction)
                    state->cancel();
                }
                // Clear mapping (whether evaluated or cancelled)
                sub.mapping.reset();
            };
        } else {
            return [state = m_state, idx](bool ready) {
                auto& sub = state->subranges[idx];
                if (sub.isEvaluated()) {
                    return; // Already evaluated
                }
                
                if (ready) {
                    // GPU work complete: evaluate this subrange
                    state->producer(sub.offset, sub.mapping->span());
                } else {
                    // Cancellation requested: clear output
                    // Can happen before semaphore is set (e.g., staging pool destruction)
                    state->cancel();
                }
                // Clear mapping (whether evaluated or cancelled)
                sub.mapping.reset();
            };
        }
    }

    // Make state accessible to DownloadFuture for construction
    std::shared_ptr<State> m_state;
};

// Unified download future that handles both transform (with output) and foreach (no output) cases
template <class Fn, class T, class Allocator, bool HasOutput>
class DownloadFuture {
public:
    using Subrange = FutureSubrange<T, Allocator>;
    using State = DownloadFutureState<Fn, T, Allocator, HasOutput>;

    // Constructor for transform case (HasOutput = true)
    template <bool H = HasOutput> requires H
    DownloadFuture(DownloadFutureBuilder<Fn, T, Allocator, true>&& builder,
                   SemaphoreValue finalSemaphore)
        : m_state(std::move(builder.m_state)) {
        assert(!m_state->semaphore.has_value() && "Semaphore already set");
        m_state->semaphore = std::move(finalSemaphore);
    }

    // Constructor for foreach case (HasOutput = false)
    template <bool H = HasOutput> requires (!H)
    DownloadFuture(DownloadFutureBuilder<Fn, T, Allocator, false>&& builder,
                   SemaphoreValue finalSemaphore)
        : m_state(std::move(builder.m_state)) {
        assert(!m_state->semaphore.has_value() && "Semaphore already set");
        m_state->semaphore = std::move(finalSemaphore);
    }

    // For transform case: get the output vector, evaluating subranges if needed
    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput> requires H
    std::vector<T>& get(const DeviceAndCommands& device) {
        // isCancelled() checks: output empty AND subranges exist (not zero-size)
        if (m_state->isCancelled()) {
            throw TimelineSubmitCancel();
        }
        m_state->semaphore->wait(device);
        // Evaluate all subranges that haven't been evaluated yet
        for (auto& subrange : m_state->subranges) {
            if (!subrange.isEvaluated()) {
                auto outputSpan = std::span(m_state->output).subspan(
                    subrange.offset, subrange.mapping->span().size());
                m_state->producer(subrange.offset, subrange.mapping->span(), outputSpan);
                subrange.mapping.reset(); // Clear mapping after evaluation (unmaps buffer)
            }
        }
        return m_state->output;
    }

    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput> requires H
    const std::vector<T>& get(const DeviceAndCommands& device) const {
        return const_cast<DownloadFuture*>(this)->get(device);
    }

    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(const DeviceAndCommands& device, std::chrono::duration<Rep, Period> duration) const {
        return m_state->semaphore->waitFor(device, duration);
    }

    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(const DeviceAndCommands& device,
                std::chrono::time_point<Clock, Duration> deadline) const {
        return m_state->semaphore->waitUntil(device, deadline);
    }

    template <vko::device_and_commands DeviceAndCommands>
    bool ready(const DeviceAndCommands& device) const { return m_state->semaphore->isSignaled(device); }

    const SemaphoreValue& semaphore() const { return *m_state->semaphore; }

    // For foreach case: wait and call function on all subranges
    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput> requires (!H)
    void wait(const DeviceAndCommands& device) {
        m_state->semaphore->wait(device);
        // Call function on all subranges that haven't been evaluated yet
        for (auto& subrange : m_state->subranges) {
            if (!subrange.isEvaluated()) {
                m_state->producer(subrange.offset, subrange.mapping->span());
                subrange.mapping.reset(); // Clear mapping after evaluation (unmaps buffer)
            }
        }
    }


private:
    std::shared_ptr<State> m_state;
};

// Type aliases for convenience
template <class Fn, class T, class Allocator>
using DownloadTransformFuture = DownloadFuture<Fn, T, Allocator, true>;

template <class Fn, class T, class Allocator>
using DownloadForEachHandle = DownloadFuture<Fn, T, Allocator, false>;

template <device_and_commands DeviceAndCommands>
class CyclingCommandBuffer {
public:
    CyclingCommandBuffer(const DeviceAndCommands& device, TimelineQueue& queue)
        : m_device(device)
        , m_queue(queue)
        , m_commandPool(device, VkCommandPoolCreateInfo{
                                    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .pNext            = nullptr,
                                    .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                    .queueFamilyIndex = queue.familyIndex(),
                                }) {}


    // Manual submission interface.
    template <device_and_commands DeviceAndCommands>
    SemaphoreValue submit() {
        std::array<VkSemaphoreSubmitInfo, 0> noWaits{};
        std::array<VkSemaphoreSubmitInfo, 0> noSignals{};
        return submit(noWaits, noSignals);
    }

    // TODO: to be really generic and match the Vulkan API, I think we want a "Submission" object
    template <typename WaitRange, typename SignalRange>
    SemaphoreValue submit(WaitRange&& waitInfos, SignalRange&& signalInfos, VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        SemaphoreValue semaphoreValue = m_queue.get().nextSubmitSemaphore();
        if (m_currentCmd) {
            CommandBuffer cmd(m_currentCmd->end());
            m_currentCmd.reset();
            m_queue.get().submit(m_device.get(), std::forward<WaitRange>(waitInfos), cmd,
                                    timelineSemaphoreStageMask,
                                    std::forward<SignalRange>(signalInfos));
            m_inFlightCmds.push_back({std::move(cmd), semaphoreValue});
        }
        return semaphoreValue;
    }

    // TODO: maybe limit access via a callback? LLMs love to store the result
    // and it's really dangerous due to it cycling.
    VkCommandBuffer commandBuffer() {
        if (!m_currentCmd) {
            m_currentCmd.emplace(m_device.get(), reuseOrMakeCommandBuffer(),
                                    VkCommandBufferBeginInfo{
                                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .pNext = nullptr,
                                        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                        .pInheritanceInfo = nullptr,
                                    });
        }
        return *m_currentCmd;
    }

    // Check to see if commandBuffer() was called since the last submit. Handy
    // to skip operations if no commands were recorded.
    bool hasCurrent() const { return m_currentCmd.has_value(); }

    // TODO: maybe limit access via a callback? LLMs love to store the result
    // and it's really dangerous due to it cycling.
    operator VkCommandBuffer() { return commandBuffer(); }

    SemaphoreValue nextSubmitSemaphore() const { return m_queue.get().nextSubmitSemaphore(); }

private:
    struct InFlightCmd {
        CommandBuffer  commandBuffer;
        SemaphoreValue readySemaphore;
    };

    CommandBuffer reuseOrMakeCommandBuffer() {
        // Try to reuse a command buffer from the in-flight queue
        if (!m_inFlightCmds.empty() &&
            m_inFlightCmds.front().readySemaphore.isSignaled(m_device.get())) {
            auto result = std::move(m_inFlightCmds.front().commandBuffer);
            m_inFlightCmds.pop_front();

            // If there's a relatively long queue of ready command buffers, free
            // some up. Leave at least one of the ready ones to avoid frequent
            // allocations/deallocations.
            if (m_inFlightCmds.size() >= 2 &&
                m_inFlightCmds.front().readySemaphore.isSignaled(m_device.get())) {
                while (m_inFlightCmds.size() >= 2 &&
                       std::next(m_inFlightCmds.begin())
                           ->readySemaphore.isSignaled(m_device.get())) {
                    m_inFlightCmds.pop_front();
                }
            }

            m_device.get().vkResetCommandBuffer(result, 0);
            return result;
        }

        // Else, create a new command buffer
        return CommandBuffer(m_device.get(), nullptr, m_commandPool,
                             VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    }
    std::reference_wrapper<const DeviceAndCommands> m_device;
    std::reference_wrapper<TimelineQueue>         m_queue;
    CommandPool                                   m_commandPool;
    std::deque<InFlightCmd>                       m_inFlightCmds;
    std::optional<simple::RecordingCommandBuffer> m_currentCmd;
};

// Staging wrapper that also holds a queue and command buffer for
// staging/streaming memory and automatically submits command buffers as needed
// to reduce staging memory usage.
template <class StagingAllocator>
class StagingStream {
public:
    using DeviceAndCommands = typename StagingAllocator::DeviceAndCommands;
    using Allocator         = typename StagingAllocator::Allocator;
    StagingStream(TimelineQueue& queue, StagingAllocator&& staging)
        : 
        m_commandBuffer(staging.device(), queue)
        , m_staging(std::move(staging))
        {}

    template <buffer DstBuffer, class Fn>
    void upload(const DstBuffer& dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dstSize, Fn&& fn) {
        using T = typename DstBuffer::ValueType;
        VkDeviceSize userOffset = 0;
        while (dstSize > 0) {
            auto& stagingBuf = allocateUpTo<T>(dstSize);
            fn(userOffset, stagingBuf.map().span());
            copyBuffer<DeviceAndCommands, std::remove_reference_t<decltype(stagingBuf)>,
                       DstBuffer>(m_staging.device(), commandBuffer(), stagingBuf, 0,
                                  dstBuffer, dstOffset, stagingBuf.size());

            // The staging buffer may not necessarily be the full size
            // requested. Loop until the entire range is uploaded.
            userOffset += stagingBuf.size();
            dstOffset += stagingBuf.size();
            dstSize -= stagingBuf.size();
        }
    }

    // Overload for raw VkBuffer (byte-oriented upload)
    template <class Fn>
    void upload(VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dstSize, Fn&& fn) {
        VkDeviceSize userOffset = 0;
        while (dstSize > 0) {
            auto& stagingBuf = allocateUpTo<std::byte>(dstSize);
            fn(userOffset, stagingBuf.map().span());
            VkBufferCopy copyRegion{
                .srcOffset = 0,
                .dstOffset = dstOffset,
                .size      = stagingBuf.size(),
            };
            m_staging.device().vkCmdCopyBuffer(commandBuffer(), stagingBuf, dstBuffer, 1, &copyRegion);

            // The staging buffer may not necessarily be the full size
            // requested. Loop until the entire range is uploaded.
            userOffset += stagingBuf.size();
            dstOffset += stagingBuf.size();
            dstSize -= stagingBuf.size();
        }
    }

    // TODO: should this return a future as well to be safer?
    template <std::ranges::input_range SrcRange, buffer DstBuffer>
    void upload(const DstBuffer& dstBuffer, SrcRange&& srcRange, VkDeviceSize dstOffset,
                VkDeviceSize size) {
        upload(dstBuffer, dstOffset, size,
               [begin = std::ranges::begin(srcRange), next = std::ranges::begin(srcRange),
                end = std::ranges::end(srcRange)](
                   [[maybe_unused]] VkDeviceSize            offset,
                   std::span<typename DstBuffer::ValueType> mapped) mutable {
                   assert(ptrdiff_t(offset) == std::distance(begin, next));
                   next = std::ranges::copy(next, next + mapped.size(), mapped.begin()).in;
               });
    }

    // Download to a vector via a transform function that operates in batches
    // 
    // IMPORTANT: User is responsible for ensuring appropriate memory barriers are in place
    // between GPU writes to srcBuffer and the download operation. The staging buffers are
    // mapped HOST_VISIBLE | HOST_COHERENT, so no explicit cache invalidation is needed for
    // the staging memory itself, but srcBuffer must be properly synchronized.
    //
    // NOTE: callback may be called even if the return value is destroyed
    template <class DstT, buffer SrcBuffer, class Fn>
    DownloadTransformFuture<Fn, DstT, Allocator>
    downloadTransform(const SrcBuffer& srcBuffer, VkDeviceSize srcOffset, VkDeviceSize srcSize, Fn&& fn) {
        using T = typename SrcBuffer::ValueType;
        
        // Create builder (does NOT have final semaphore yet)
        DownloadFutureBuilder<Fn, DstT, Allocator, true> builder(std::forward<Fn>(fn), srcSize);

        VkDeviceSize userOffset = 0;
        while (srcSize > 0) {
            auto& stagingBuf = allocateUpTo<T>(srcSize);
            copyBuffer<DeviceAndCommands, SrcBuffer, std::remove_reference_t<decltype(stagingBuf)>>(
                m_staging.device(), commandBuffer(), srcBuffer, srcOffset, stagingBuf, 0, stagingBuf.size());

            // Create mapping now (safe: mapping before GPU copy completion is fine)
            // It's perfectly fine to create the mapping before the copy is complete or even submitted too :)
            // The mapping just establishes CPU-side access; actual data read happens
            // later in the callback after GPU work completes.
            auto callback = builder.addSubrange(userOffset, stagingBuf.map());
            m_staging.registerBatchCallback(std::move(callback));

            // The staging buffer may not necessarily be the full size
            // requested. Loop until the entire range is uploaded.
            userOffset += stagingBuf.size();
            srcOffset += stagingBuf.size();
            srcSize -= stagingBuf.size();
        }

        // NOW we know the final semaphore - construct future from builder
        // For zero-size transfers (no subranges), use already-signaled semaphore
        // Otherwise use semaphore that signals when the LAST subrange's GPU copy completes
        SemaphoreValue finalSemaphore = builder.m_state->subranges.empty() 
            ? SemaphoreValue::makeSignalled()
            : m_commandBuffer.nextSubmitSemaphore();
        return DownloadTransformFuture<Fn, DstT, Allocator>{std::move(builder), finalSemaphore};
    }

    // Regular download function with an identity transform to a vector
    template <buffer SrcBuffer>
    auto download(const SrcBuffer& srcBuffer, VkDeviceSize offset, VkDeviceSize size) {
        using T = typename SrcBuffer::ValueType;
        return downloadTransform<T, SrcBuffer>(
            srcBuffer, offset, size,
            [](VkDeviceSize /* offset */, std::span<const T> input, std::span<T> output) {
                std::ranges::copy(input, output.begin());
            });
    }

    // Call the given function on each batch of the download
    template <buffer SrcBuffer, class Fn>
    DownloadForEachHandle<Fn, typename SrcBuffer::ValueType, Allocator>
    downloadForEach(const SrcBuffer& srcBuffer, VkDeviceSize srcOffset, VkDeviceSize srcSize, Fn&& fn) {
        using T = typename SrcBuffer::ValueType;
        
        // Create builder (does NOT have final semaphore yet)
        DownloadFutureBuilder<Fn, T, Allocator, false> builder(std::forward<Fn>(fn));

        VkDeviceSize userOffset = 0;
        while (srcSize > 0) {
            auto& stagingBuf = allocateUpTo<T>(srcSize);
            copyBuffer<DeviceAndCommands, SrcBuffer, std::remove_reference_t<decltype(stagingBuf)>>(
                m_staging.device(), commandBuffer(), srcBuffer, srcOffset, stagingBuf, 0, stagingBuf.size());

            // Create mapping now (safe: mapping before GPU copy completion is fine)
            // It's perfectly fine to create the mapping before the copy is complete or even submitted too :)
            auto finalMapping = stagingBuf.map();
            
            auto callback = builder.addSubrange(userOffset, std::move(finalMapping));
            m_staging.registerBatchCallback(std::move(callback));

            // The staging buffer may not necessarily be the full size
            // requested. Loop until the entire range is uploaded.
            userOffset += stagingBuf.size();
            srcOffset += stagingBuf.size();
            srcSize -= stagingBuf.size();
        }
        
        // NOW we know the final semaphore - construct future from builder
        // For zero-size transfers (no subranges), use already-signaled semaphore
        // Otherwise use semaphore that signals when the LAST subrange's GPU copy completes
        SemaphoreValue finalSemaphore = builder.m_state->subranges.empty()
            ? SemaphoreValue::makeSignalled()
            : m_commandBuffer.nextSubmitSemaphore();
        return DownloadForEachHandle<Fn, T, Allocator>{std::move(builder), finalSemaphore};
    }

    // Manual submission interface.
    SemaphoreValue submit() {
        std::array<VkSemaphoreSubmitInfo, 0> noWaits{};
        std::array<VkSemaphoreSubmitInfo, 0> noSignals{};
        return submit(noWaits, noSignals);
    }

    // TODO: to be really generic and match the Vulkan API, I think we want a "Submission" object
    template <typename WaitRange, typename SignalRange>
    SemaphoreValue submit(WaitRange&& waitInfos, SignalRange&& signalInfos) {
        bool           hasCommands    = m_commandBuffer.hasCurrent();
        SemaphoreValue semaphoreValue = m_commandBuffer.submit(
            std::forward<WaitRange>(waitInfos), std::forward<SignalRange>(signalInfos),
            VK_PIPELINE_STAGE_TRANSFER_BIT /* TODO: is this right? */);
        if (hasCommands) {
            m_staging.endBatch(semaphoreValue);
            m_justSubmitted = true;
        }
        return semaphoreValue;
    }

    // Get total allocated staging memory size in bytes
    VkDeviceSize size() const { return m_staging.size(); }
    
    // Get total staging memory capacity in bytes
    VkDeviceSize capacity() const { return m_staging.capacity(); }

    CyclingCommandBuffer<DeviceAndCommands>& commandBuffer() { return m_commandBuffer; }
private:
    
    // Accessor for testing
    StagingAllocator& staging() { return m_staging; }
    const StagingAllocator& staging() const { return m_staging; }

    CyclingCommandBuffer<DeviceAndCommands>       m_commandBuffer;
    StagingAllocator                              m_staging;
    bool                                          m_justSubmitted = true;
    
    // Helper to allocate from staging or submit and retry once Guarantees
    // return of valid buffer or throws on persistent failure, indicating a bug
    // with the backing allocator.
    template <class T>
    BoundBuffer<T, Allocator>& allocateUpTo(VkDeviceSize size) {
        while (true) {
            auto* buf = m_staging.template allocateUpTo<T>(size);
            if (buf) {
                m_justSubmitted = false;
                return *buf;  // Success
            }
            
            if constexpr (StagingAllocator::AllocateAlwaysSucceeds) {
                // std::unreachable()
                throw std::runtime_error("AllocateAlwaysSucceeds but allocation failed");
            } else {
                if (m_justSubmitted) {
                    throw std::runtime_error("Staging allocation failed even after submit");
                }

                submit();
                // Retry
            }
        }
    }
};

// Helper: Create a device buffer and upload data to it in one call
// Usage: auto buffer = vko::upload(staging, device, allocator, data, usageFlags);
template <std::ranges::sized_range SrcRange, typename StagingT, device_and_commands DeviceAndCommands, 
          allocator AllocatorT>
auto upload(StagingT& staging, const DeviceAndCommands& device, AllocatorT& allocator, 
            SrcRange&& data, VkBufferUsageFlags usage, 
            VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    using T = std::remove_cv_t<std::ranges::range_value_t<SrcRange>>;
    vko::DeviceBuffer<T, AllocatorT> buffer(device, std::ranges::size(data), usage, 
                                            vko::vma::allocationCreateInfo(memoryProperties), allocator);
    staging.upload(buffer, std::forward<SrcRange>(data), 0, std::ranges::size(data));
    return buffer;
}

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
        })) {
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
                             })) {
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
        })) {
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
        })) {
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
    auto* buffer = staging.template allocateUpTo<T>(copySize);
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
    auto* buffer = staging.template allocateUpTo<std::byte>(size);
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
    auto* dstBuffer = staging.template allocateUpTo<std::byte>(size);
    if (!dstBuffer)
        throw std::runtime_error("Failed to allocate staging buffer");

    // Temporary buffer to hold the VkCopyMemoryIndirectCommandNV
    auto* cmdBuffer = staging.template allocateUpTo<VkCopyMemoryIndirectCommandNV>(1);
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
