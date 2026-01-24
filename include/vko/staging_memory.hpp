// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

// Staging memory utilities for GPU data transfer. Quick reference:
//
// Upload overloads:
//   upload(stream, device, BufferSpan<T>, fn)            - callback populates staging
//   upload(stream, device, range, BufferSpan<T>)         - range to buffer span
//   upload<T>(stream, device, alloc, size, usage, fn)    - creates buffer via callback
//   upload(stream, device, allocator, range, usage)      - creates buffer from range
//
// Download overloads (all return futures):
//   download(stream, device, BufferSpan<T>)              - identity to vector
//   download<DstT>(stream, device, BufferSpan<SrcT>, fn) - transform to vector
//   downloadForEach(stream, device, BufferSpan<T>, fn)   - streaming callback
//
// DeviceSpan overloads (require VK_NV_copy_memory_indirect; staging pool and buffers
// need VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, or use DeviceBuffer):
//   upload(stream, device, DeviceSpan, size, fn)         - callback to device address
//   upload(stream, device, range, DeviceSpan)            - range to device address
//   download(stream, device, DeviceSpan)                 - identity from address
//   download<DstT>(stream, device, DeviceSpan, fn)       - transform from address
//   downloadForEach(stream, device, DeviceSpan, fn)      - streaming from address
//
// Use std::span(hostData).subspan(offset, size) and
// BufferSpan(buf).subspan(offset, size) to transfer partial ranges.
//
// NOTE: must call stream.submit() before future.get()/wait()!
//
// IMPORTANT: User is responsible for gpu side memory barriers. Staging buffers
// use HOST_VISIBLE | HOST_COHERENT so no host side barriers are needed. I.e.:
// - upload() -> barrier -> GPU op
// - GPU op -> barrier -> download()
//
// Beware *ForEach() variants: The callback is stored on a shared state object
// between BOTH the returned handle and the stream. If you discard the returned
// future handle, the callback still gets called but this is only guaranteed
// after stream.wait(). Even if the GPU operations are complete, the callback
// will not be called until an explicit poll(), wait(), or the staging stream
// tries to recycle the staging allocation.
//
// auto future = vko::download(stream, device, BufferSpan(buffer), [](VkDeviceSize chunkOffset, std::span<const T> data) {
//     ... do something with your data
// });
// future.wait(device);              // BUG: deadlock as no submit was called
// auto semaphore = stream.submit(); // begin transfer work
// semaphore.wait(device);           // does not call callback
// stream.poll();                    // OK: non-blocking, calls callback because we waited on the semaphore
// stream.wait();                    // OK: waits and calls callback, if not already called
// future.wait(device);              // OK: waits and calls callback, if not already called

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vko/adapters.hpp>
#include <vko/allocator.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/command_recording.hpp>
#include <vko/device_address.hpp>
#include <vko/handles.hpp>
#include <vko/shortcuts.hpp>
#include <vko/timeline_queue.hpp>
#include <vko/unique_any.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

// Helper concept to check if a function can be called with compatible span types
// This is more lenient than exact type matching, allowing const/non-const variations
template <class Fn, class T>
concept upload_callback = requires(Fn&& fn, VkDeviceSize offset, std::span<T> data) {
    { fn(offset, data) } -> std::same_as<void>;
};

// Concept to validate transform callback signatures for download operations
// Transform from SrcT to DstT: fn(offset, input_span<SrcT>, output_span<DstT>) -> void
template <class Fn, class SrcT, class DstT>
concept download_transform_callback =
    requires(Fn&& fn, VkDeviceSize offset, std::span<SrcT> input, std::span<DstT> output) {
        { fn(offset, input, output) } -> std::same_as<void>;
    };

// Concept to validate foreach callback signatures for downloadForEach operations
template <class Fn, class T>
concept download_foreach_callback = requires(Fn&& fn, VkDeviceSize offset, std::span<T> data) {
    { fn(offset, data) } -> std::same_as<void>;
};

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
        {
            t.template allocateUpTo<int>(size)
        } -> std::same_as<BoundBuffer<int, typename T::Allocator>*>;
        // Atomically allocate a single element + up-to buffer pair
        {
            t.template allocateSingleAndUpTo<int, int>(size)
        } -> std::same_as<std::optional<std::pair<BoundBuffer<int, typename T::Allocator>&,
                                                  BoundBuffer<int, typename T::Allocator>&>>>;
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
        // Non-blocking: process callbacks for batches that are already complete
        { t.poll() } -> std::same_as<void>;
        // Blocking: wait for all batches to complete and invoke their callbacks
        { t.wait() } -> std::same_as<void>;
    };

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
    using DeviceAndCommands                      = DeviceAndCommandsType;
    using Allocator                              = AllocatorType;
    static constexpr bool AllocateAlwaysSucceeds = true;
    static constexpr bool AllocateAlwaysFull     = true;
    DedicatedStagingPool(const DeviceAndCommands& device, Allocator& allocator,
                         VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : m_device(device)
        , m_allocator(allocator)
        , m_bufferUsageFlags(usage) {}

    DedicatedStagingPool(DedicatedStagingPool&&) = default;
    DedicatedStagingPool& operator=(DedicatedStagingPool&& other) noexcept {
        finalizeAllWork();
        m_device           = other.m_device;
        m_allocator        = other.m_allocator;
        m_bufferUsageFlags = other.m_bufferUsageFlags;
        m_current          = std::move(other.m_current);
        m_released         = std::move(other.m_released);
        m_totalSize        = other.m_totalSize;
        return *this;
    }

    ~DedicatedStagingPool() { finalizeAllWork(); }

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
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size, std::function<void(bool)> destruct) {
        auto* ptr = allocateUpTo<T>(size);
        if (ptr) {
            m_current.destroyCallbacks.emplace_back(std::move(destruct));
        }
        return ptr;
    }

    // Atomically allocate one single element + up-to buffer
    // Always succeeds for DedicatedStagingPool (creates new dedicated allocations)
    template <class TSingle, class TUpTo>
    std::optional<std::pair<BoundBuffer<TSingle, Allocator>&, BoundBuffer<TUpTo, Allocator>&>>
    allocateSingleAndUpTo(VkDeviceSize upToSize) {
        // Allocate single element buffer
        auto [singleAny, singlePtr] = makeUniqueAny<vko::BoundBuffer<TSingle, Allocator>>(
            m_device.get(), 1, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += singlePtr->sizeBytes();
        m_totalSize += singlePtr->sizeBytes();
        m_current.buffers.emplace_back(std::move(singleAny));

        // Allocate up-to buffer (full size, since DedicatedStagingPool always allocates full)
        auto [upToAny, upToPtr] = makeUniqueAny<vko::BoundBuffer<TUpTo, Allocator>>(
            m_device.get(), upToSize, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_allocator.get());
        m_current.size += upToPtr->sizeBytes();
        m_totalSize += upToPtr->sizeBytes();
        m_current.buffers.emplace_back(std::move(upToAny));

        return std::pair{std::ref(*singlePtr), std::ref(*upToPtr)};
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

    // Non-blocking: process callbacks for batches that are already complete
    void poll() { checkAndInvokeCompletedCallbacks(); }

    // Blocking: wait for all batches to complete and invoke their callbacks
    void wait() {
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
    size_t                   unsubmittedTransfers() const { return m_current.buffers.size(); }
    size_t                   pendingTransfers() const {
        size_t result = m_released.size();
        m_released.visitAll([&result](const Batch& batch, const SemaphoreValue&) {
            result += batch.buffers.size();
        });
        return result;
    }

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

    void finalizeAllWork() {
        // Cancel unsubmitted batches (never started, so cancel callbacks)
        for (auto& callback : m_current.destroyCallbacks) {
            callback(false);
        }

        // Wait for submitted batches to finish, then destroy them
        m_released.waitAndConsume(m_device.get(), [this](Batch& batch) {
            for (auto& callback : batch.destroyCallbacks) {
                callback(true);
            }
        });
    }

    struct Batch {
        std::vector<UniqueAny>                 buffers;
        std::vector<std::function<void(bool)>> destroyCallbacks;
        VkDeviceSize                           size = 0;
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

// Align a value down to the nearest multiple of alignment
constexpr VkDeviceSize align_down(VkDeviceSize value, VkDeviceSize alignment) {
    return value & ~(alignment - 1);
}

// Staging buffer allocator that provides temporary buffers from cyclical memory
// pools. The caller must call endBatch() with a SemaphoreValue that will
// be signaled when the buffers are no longer in use so that their pool can be
// reused. Use this when you have many small transfers to make on a regular
// basis to avoid frequent allocations/deallocations.
// TODO: would we ever want to allow staging buffers to be recycled after
// waiting on a semaphore? Like a swapchain where the GPU can just keep
// transferring.
template <device_and_commands DeviceAndCommandsType = vko::Device>
class RecyclingStagingPool {
public:
    using DeviceAndCommands                      = DeviceAndCommandsType;
    using Allocator                              = vma::Allocator;
    static constexpr bool AllocateAlwaysSucceeds = false;
    static constexpr bool AllocateAlwaysFull     = false;

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
        vko::Buffer          tempBuffer(device, tempBufferInfo);
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

    RecyclingStagingPool(const RecyclingStagingPool& other)            = delete;
    RecyclingStagingPool& operator=(const RecyclingStagingPool& other) = delete;
    RecyclingStagingPool& operator=(RecyclingStagingPool&& other) noexcept {
        finalizeAllWork();
        m_device               = other.m_device;
        m_allocator            = other.m_allocator;
        m_poolSize             = other.m_poolSize;
        m_bufferUsageFlags     = other.m_bufferUsageFlags;
        m_maxPools             = other.m_maxPools;
        m_minPools             = other.m_minPools;
        m_inUse                = std::move(other.m_inUse);
        m_current              = std::move(other.m_current);
        m_totalPoolBytes       = other.m_totalPoolBytes;
        m_totalPoolCount       = other.m_totalPoolCount;
        m_totalBufferBytes     = other.m_totalBufferBytes;
        m_alignment            = other.m_alignment;
        m_currentPoolUsedBytes = other.m_currentPoolUsedBytes;
        return *this;
    }
    RecyclingStagingPool(RecyclingStagingPool&& other) noexcept = default;
    ~RecyclingStagingPool() { finalizeAllWork(); }

    // Primary implementation without per-buffer callback
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size) {
        return makeTmpBuffer<T>(size);
    }

    // Overload with per-buffer callback
    template <class T>
    vko::BoundBuffer<T, Allocator>* allocateUpTo(size_t size, std::function<void(bool)> destruct) {
        auto result = makeTmpBuffer<T>(size);
        if (result) {
            m_current.destroyCallbacks.push_back(std::move(destruct));
        }
        return result;
    }

    // Atomically allocate one single element + up-to buffer in same pool
    // Returns nullopt if cannot fit both (caller should endBatch/recycle)
    template <class TSingle, class TUpTo>
    std::optional<std::pair<BoundBuffer<TSingle, Allocator>&, BoundBuffer<TUpTo, Allocator>&>>
    allocateSingleAndUpTo(VkDeviceSize upToSize) {
        return makeTmpBufferPair<TSingle, TUpTo>(upToSize);
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
        m_inUse.push_back(std::move(m_current), reuseSemaphore);
        m_current = {};
        assert(m_current.buffers.empty());
        assert(m_current.destroyCallbacks.empty());
    }

    // Non-blocking: process callbacks for batches that are already complete
    void poll() { invokeReadyCallbacks(); }

    // Blocking: wait for all batches to complete and invoke their callbacks
    void wait() {
        // Wait for all batches
        m_inUse.wait(m_device.get());

        // Process ALL ready batches (visitAll since they're all ready after
        // wait()). This destroys only the buffers, not the pools.
        m_inUse.visitAll(
            [this](PoolBatch& batch, SemaphoreValue&) { destroyBuffers(batch, true); });

        // Free only excess pools in m_inUse
        freeExcessPools();
    }

    VkDeviceSize             capacity() const { return m_totalPoolBytes; }
    VkDeviceSize             size() const { return m_totalBufferBytes; }
    const DeviceAndCommands& device() const { return m_device.get(); }

    // Size of a single pool. Best to submit and endBatch() before size()
    // reaches this value.
    VkDeviceSize poolSize() const { return m_poolSize; }

    size_t unsubmittedTransfers() const { return m_current.buffers.size(); }
    size_t pendingTransfers() const {
        size_t result = 0;
        m_inUse.visitAll([&result](const PoolBatch& batch, const SemaphoreValue&) {
            result += batch.buffers.size();
        });
        return result;
    }

private:
    struct PoolBatch {
        std::vector<vma::Pool>                 pools;
        std::vector<UniqueAny>                 buffers;
        std::vector<std::function<void(bool)>> destroyCallbacks;
        VkDeviceSize                           bufferBytes = 0;
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

    void finalizeAllWork() {
        // Cancel unsubmitted batches (never started, so cancel callbacks)
        destroyBuffers(m_current, false);

        // Wait for submitted batches to finish, then destroy their buffers
        m_inUse.waitAndConsume(m_device.get(),
                               [this](PoolBatch& batch) { destroyBuffers(batch, true); });
    }

    bool hasCurrentPool() const { return !m_current.pools.empty(); }

    vma::Pool& currentPool() { return m_current.pools.back(); }

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
            .usage = VMA_MEMORY_USAGE_UNKNOWN, // Use legacy mode with only requiredFlags (supports
                                               // both upload/download)
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
            .flags                  = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,
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
                VkDeviceSize alignedOffset = align_up(m_currentPoolUsedBytes, m_alignment);
                VkDeviceSize availableBytes =
                    alignedOffset < m_poolSize ? m_poolSize - alignedOffset : 0;
                // Vulkan rounds allocation size up to alignment, so we can only use
                // a multiple of alignment bytes from the pool
                VkDeviceSize usableBytes = align_down(availableBytes, m_alignment);
                VkDeviceSize trySize     = usableBytes / static_cast<VkDeviceSize>(sizeof(T));

                if (trySize > 0) {
                    trySize = std::min(size, trySize);

                    // Allocate from current pool
                    // Note: This can still fail if our tracking is wrong due to VMA's
                    // internal fragmentation. In that case, we'll try with a fresh pool.
                    try {
                        auto buffer     = allocateFromCurrentPool<T>(trySize);
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

    // Atomically allocate single + up-to buffer pair (following makeTmpBuffer pattern)
    template <class TSingle, class TUpTo>
    std::optional<std::pair<BoundBuffer<TSingle, Allocator>&, BoundBuffer<TUpTo, Allocator>&>>
    makeTmpBufferPair(VkDeviceSize upToSize) {
        // Try up to 2 times: once with current pool, once with a fresh pool
        for (int attempt = 0; attempt < 2; ++attempt) {
            // On second attempt, try to get a new pool
            if (attempt > 0) {
                if (!addPoolToCurrentBatch()) {
                    return std::nullopt; // Can't allocate a new pool
                }
            }

            // If we have a pool, try to allocate both
            if (hasCurrentPool()) {
                // Align the predicted pool usage before allocation
                VkDeviceSize alignedOffset = align_up(m_currentPoolUsedBytes, m_alignment);

                // Check if pool is already full after alignment
                if (alignedOffset >= m_poolSize) {
                    continue; // Pool is full after alignment
                }

                // Vulkan rounds allocation size up to alignment, so usable space
                // must be a multiple of alignment
                VkDeviceSize usable = align_down(m_poolSize - alignedOffset, m_alignment);

                // Space needed for 1 single element (what Vulkan will actually require)
                VkDeviceSize singleReqSize = align_up(sizeof(TSingle), m_alignment);
                if (usable < singleReqSize)
                    continue;

                // Predict remaining space after single allocation
                VkDeviceSize afterSingle = usable - singleReqSize;
                VkDeviceSize upToElems   = std::min(upToSize, afterSingle / sizeof(TUpTo));
                if (upToElems == 0)
                    continue;

                try {
                    auto singleBuffer           = allocateFromCurrentPool<TSingle>(1);
                    auto [singleAny, singlePtr] = toUniqueAnyWithPtr(std::move(singleBuffer));
                    m_current.buffers.emplace_back(std::move(singleAny));

                    auto upToBuffer         = allocateFromCurrentPool<TUpTo>(upToElems);
                    auto [upToAny, upToPtr] = toUniqueAnyWithPtr(std::move(upToBuffer));
                    m_current.buffers.emplace_back(std::move(upToAny));

                    return std::pair{std::ref(*singlePtr), std::ref(*upToPtr)};
                } catch (const ResultException<VK_ERROR_OUT_OF_DEVICE_MEMORY>&) {
                    // Our prediction was wrong - mark pool as full and try with fresh pool
                    m_currentPoolUsedBytes = m_poolSize;
                    if (attempt == 1) {
                        throw; // Already tried with fresh pool, propagate error
                    }
                    // Continue to attempt 1 which will get a fresh pool
                }
            }
        }

        return std::nullopt;
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

        // Align the predicted usage before allocating
        m_currentPoolUsedBytes = align_up(m_currentPoolUsedBytes, m_alignment);

        auto buffer = BoundBuffer<T, Allocator>(
            m_device.get(), size, m_bufferUsageFlags,
            vma::allocationCreateInfo(currentPool(), VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            m_allocator.get());

        // Track usage - add actual allocated size
        VkDeviceSize allocatedBytes = buffer.sizeBytes();
        m_current.bufferBytes += allocatedBytes;
        m_totalBufferBytes += allocatedBytes;
        m_currentPoolUsedBytes += allocatedBytes;

        // Assert we never exceed pool size - if this fires, our tracking is wrong
        assert(m_currentPoolUsedBytes <= m_poolSize && "Pool usage tracking exceeded pool size");

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
        if (m_current.buffers.empty() && batch.buffers.capacity() > m_current.buffers.capacity()) {
            m_current.buffers.swap(batch.buffers);
        }
        if (m_current.destroyCallbacks.empty() &&
            batch.destroyCallbacks.capacity() > m_current.destroyCallbacks.capacity()) {
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
    VkDeviceSize                                    m_poolSize         = 0;
    VkDeviceSize                                    m_totalPoolBytes   = 0;
    size_t                                          m_totalPoolCount   = 0;
    VkDeviceSize                                    m_totalBufferBytes = 0;
    PoolBatch                                       m_current;
    CompletionQueue<PoolBatch>                      m_inUse;
    VkBufferUsageFlags                              m_bufferUsageFlags     = 0;
    size_t                                          m_maxPools             = 0;
    size_t                                          m_minPools             = 0;
    VkDeviceSize                                    m_alignment            = 0;
    VkDeviceSize                                    m_currentPoolUsedBytes = 0;
};
static_assert(staging_allocator<RecyclingStagingPool<Device>>);
static_assert(std::is_move_constructible_v<RecyclingStagingPool<Device>>);
static_assert(std::is_move_assignable_v<RecyclingStagingPool<Device>>);

} // namespace vma

// Structures for DownloadFuture
template <class StagingT, class Allocator>
struct FutureSubrange {
    VkDeviceSize                                      offset;
    std::optional<BufferMapping<StagingT, Allocator>> mapping; // nullopt = evaluated or cancelled

    bool isEvaluated() const { return !mapping.has_value(); }
};

// Internal State structures for DownloadFuture
// Base State template (for HasOutput = false, foreach case - no output vector)
template <class Fn, class StagingT, class OutputT, class Allocator, bool HasOutput>
struct DownloadFutureState {
    std::vector<FutureSubrange<StagingT, Allocator>> subranges;
    Fn                                               producer;
    std::optional<SemaphoreValue>                    semaphore; // Set by DownloadFuture constructor

    DownloadFutureState(Fn&& fn)
        : producer(std::forward<Fn>(fn)) {}

    // No cancellation for foreach - it just doesn't call the producer
    bool isCancelled() const { return false; }
    void cancel() {
        // Cancellation can happen for various reasons:
        // - User forgot to call submit() (bug)
        // - Staging pool destroyed before future evaluated (intentional)
        // We can't reliably distinguish these cases
        // assert(!"Most likely a missing staging.submit()");
        /* no-op for foreach */
    }
};

// Specialization for HasOutput = true (transform case - with output vector)
template <class Fn, class StagingT, class OutputT, class Allocator>
struct DownloadFutureState<Fn, StagingT, OutputT, Allocator, true> {
    std::vector<FutureSubrange<StagingT, Allocator>> subranges;
    Fn                                               producer;
    std::vector<OutputT>                             output;
    std::optional<SemaphoreValue>                    semaphore; // Set by DownloadFuture constructor

    DownloadFutureState(Fn&& fn, size_t outputSize)
        : producer(std::forward<Fn>(fn))
        , output(outputSize) {}

    // Cancelled = output cleared but subranges existed (not just zero-size)
    bool isCancelled() const { return output.empty() && !subranges.empty(); }
    void cancel() {
        // Cancellation can happen for various reasons:
        // - User forgot to call submit() (bug)
        // - Staging pool destroyed before future evaluated (intentional)
        // We can't reliably distinguish these cases
        // assert(!"Most likely a missing staging.submit()");
        output.clear();
    }
};

// Builder for DownloadFuture - used internally by StagingStream during download allocation
template <class Fn, class StagingT, class OutputT, class Allocator, bool HasOutput>
class DownloadFutureBuilder {
public:
    using Subrange = FutureSubrange<StagingT, Allocator>;
    using State    = DownloadFutureState<Fn, StagingT, OutputT, Allocator, HasOutput>;

    // Constructor for transform case (HasOutput = true)
    DownloadFutureBuilder(Fn&& producer, size_t outputSize)
        requires HasOutput
        : m_state(std::make_shared<State>(std::forward<Fn>(producer), outputSize)) {}

    // Constructor for foreach case (HasOutput = false)
    DownloadFutureBuilder(Fn&& producer)
        requires(!HasOutput)
        : m_state(std::make_shared<State>(std::forward<Fn>(producer))) {}

    // Add a subrange and return callback for StagingStream to register with staging allocator
    std::function<void(bool)> addSubrange(VkDeviceSize                         offset,
                                          BufferMapping<StagingT, Allocator>&& mapping) {
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
                    auto outputSpan =
                        std::span(state->output).subspan(sub.offset, sub.mapping->span().size());
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
template <class Fn, class StagingT, class OutputT, class Allocator, bool HasOutput>
class DownloadFuture {
public:
    using Subrange = FutureSubrange<StagingT, Allocator>;
    using State    = DownloadFutureState<Fn, StagingT, OutputT, Allocator, HasOutput>;

    // Constructor for transform case (HasOutput = true)
    template <bool H = HasOutput>
        requires H
    DownloadFuture(DownloadFutureBuilder<Fn, StagingT, OutputT, Allocator, true>&& builder,
                   SemaphoreValue                                                  finalSemaphore)
        : m_state(std::move(builder.m_state)) {
        assert(!m_state->semaphore.has_value() && "Semaphore already set");
        m_state->semaphore = std::move(finalSemaphore);
    }

    // Constructor for foreach case (HasOutput = false)
    template <bool H = HasOutput>
        requires(!H)
    DownloadFuture(DownloadFutureBuilder<Fn, StagingT, OutputT, Allocator, false>&& builder,
                   SemaphoreValue                                                   finalSemaphore)
        : m_state(std::move(builder.m_state)) {
        assert(!m_state->semaphore.has_value() && "Semaphore already set");
        m_state->semaphore = std::move(finalSemaphore);
    }

    // For transform case: get the output vector, evaluating subranges if needed
    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput>
        requires H
    std::vector<OutputT>& get(const DeviceAndCommands& device) {
        // isCancelled() checks: output empty AND subranges exist (not zero-size)
        if (m_state->isCancelled()) {
            throw TimelineSubmitCancel();
        }
        m_state->semaphore->wait(device);
        // Evaluate all subranges that haven't been evaluated yet
        for (auto& subrange : m_state->subranges) {
            if (!subrange.isEvaluated()) {
                auto outputSpan = std::span(m_state->output)
                                      .subspan(subrange.offset, subrange.mapping->span().size());
                m_state->producer(subrange.offset, subrange.mapping->span(), outputSpan);
                subrange.mapping.reset(); // Clear mapping after evaluation (unmaps buffer)
            }
        }
        return m_state->output;
    }

    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput>
        requires H
    const std::vector<OutputT>& get(const DeviceAndCommands& device) const {
        return const_cast<DownloadFuture*>(this)->get(device);
    }

    template <vko::device_and_commands DeviceAndCommands, class Rep, class Period>
    bool waitFor(const DeviceAndCommands&           device,
                 std::chrono::duration<Rep, Period> duration) const {
        return m_state->semaphore->waitFor(device, duration);
    }

    template <vko::device_and_commands DeviceAndCommands, class Clock, class Duration>
    bool waitUntil(const DeviceAndCommands&                 device,
                   std::chrono::time_point<Clock, Duration> deadline) const {
        return m_state->semaphore->waitUntil(device, deadline);
    }

    template <vko::device_and_commands DeviceAndCommands>
    bool ready(const DeviceAndCommands& device) const {
        return m_state->semaphore->isSignaled(device);
    }

    const SemaphoreValue& semaphore() const { return *m_state->semaphore; }

    // For foreach case: wait and call function on all subranges
    template <vko::device_and_commands DeviceAndCommands, bool H = HasOutput>
        requires(!H)
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
template <class Fn, class StagingT, class OutputT, class Allocator>
using DownloadTransformFuture = DownloadFuture<Fn, StagingT, OutputT, Allocator, true>;

template <class Fn, class T, class Allocator>
using DownloadForEachHandle = DownloadFuture<Fn, T, T, Allocator, false>;

// Manages command buffer lifecycle with automatic recycling of completed buffers.
// The destructor waits for all in-flight command buffers to complete, ensuring
// safe cleanup even if GPU work is still executing.
// Note: A CyclingCommandBufferRef version could be created to allow sharing
// a CommandPool across multiple command buffer instances, following the same
// pattern as StagingStreamRef/StagingStream.
template <device_and_commands DeviceAndCommands, class Queue = TimelineQueue>
class CyclingCommandBuffer {
public:
    using DeviceAndCommandsType = DeviceAndCommands;
    using QueueType             = Queue;

    CyclingCommandBuffer(const DeviceAndCommands& device, Queue& queue)
        : m_device(device)
        , m_queue(queue)
        , m_commandPool(device, VkCommandPoolCreateInfo{
                                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .pNext = nullptr,
                                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = queue.familyIndex(),
                                }) {}

    CyclingCommandBuffer(CyclingCommandBuffer&&) = default;
    CyclingCommandBuffer& operator=(CyclingCommandBuffer&& other) noexcept {
        tryWait();
        m_device       = other.m_device;
        m_queue        = other.m_queue;
        m_commandPool  = std::move(other.m_commandPool);
        m_inFlightCmds = std::move(other.m_inFlightCmds);
        m_current      = std::move(other.m_current);
        return *this;
    }

    ~CyclingCommandBuffer() { std::ignore = tryWait(); }

    // Manual submission interface.
    SemaphoreValue submit(VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        return submit({}, {}, timelineSemaphoreStageMask);
    }

    // TODO: to be really generic and match the Vulkan API, I think we want a "Submission" object
    template <class WaitRange   = std::initializer_list<VkSemaphoreSubmitInfo>,
              class SignalRange = std::initializer_list<VkSemaphoreSubmitInfo>>
    SemaphoreValue submit(WaitRange&& waitInfos, SignalRange&& signalInfos,
                          VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        if (!m_current) {
            // No current command buffer. We can't return makeSignalled()
            // because the user may be relying on sequential submissions
            // completing in order. There may also be waits/signals to insert
            // into the queue. Sadly, it's more robust to create an empty command buffer.
            std::ignore = commandBuffer();
        }

        SemaphoreValue result = m_current->promise.futureValue();
        CommandBuffer  cmd(m_current->cmd.end());

        m_queue.get().submit(m_device.get(), std::forward<WaitRange>(waitInfos), cmd,
                             std::move(m_current->promise), timelineSemaphoreStageMask,
                             std::forward<SignalRange>(signalInfos));
        m_current.reset();

        m_inFlightCmds.push_back({std::move(cmd), result});
        return result;
    }

    void submitAndWait(VkPipelineStageFlags2 timelineSemaphoreStageMask) {
        submit(timelineSemaphoreStageMask).wait(m_device.get());
    }

    // TODO: maybe limit access via a callback? LLMs love to store the result
    // and it's really dangerous due to it cycling.
    const TimelineCommandBuffer& commandBuffer() {
        if (!m_current) {
            m_current =
                TimelineCommandBuffer{simple::RecordingCommandBuffer(
                                          m_device.get(), reuseOrMakeCommandBuffer(),
                                          VkCommandBufferBeginInfo{
                                              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .pNext = nullptr,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr,
                                          }),
                                      m_queue.get().submitPromise()};
        }
        return *m_current;
    }

    // Check to see if commandBuffer() was called since the last submit. Handy
    // to skip operations if no commands were recorded.
    bool hasCurrent() const { return m_current.has_value(); }

    // Implicit conversion to VkCommandBuffer for convenience
    // WARNING: Do NOT store the result as VkCommandBuffer - the handle can change after submit!
    // Always use the CyclingCommandBuffer reference directly, which will fetch
    // the current handle on each use via this conversion operator.
    operator VkCommandBuffer() { return commandBuffer(); }

    // Convenience helper to get the semaphore value for the current command buffer's next submit.
    SemaphoreValue nextSubmitSemaphore() { return commandBuffer().promise.futureValue(); }

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
                while (
                    m_inFlightCmds.size() >= 2 &&
                    std::next(m_inFlightCmds.begin())->readySemaphore.isSignaled(m_device.get())) {
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

    // Wait for all in-flight command buffers to complete before destroying the command pool
    bool tryWait() const {
        bool result = true;
        for (auto& inFlight : m_inFlightCmds) {
            // tryWait returns false if cancelled/never submitted, but won't throw
            result = inFlight.readySemaphore.tryWait(m_device.get()) && result;
        }
        return result;
    }

    std::reference_wrapper<const DeviceAndCommands> m_device;
    std::reference_wrapper<Queue>                   m_queue;
    CommandPool                                     m_commandPool;
    std::deque<InFlightCmd>                         m_inFlightCmds;
    std::optional<TimelineCommandBuffer>            m_current;
};

// Concept for types that provide staging stream functionality
// (both StagingStream and StagingStreamRef satisfy this)
template <class T>
concept staging_stream = requires(T& t, VkDeviceSize size) {
    typename T::Allocator;
    typename T::DeviceAndCommands;
    // Core primitives for staging operations
    {
        t.template withStagingBuffer<int>(
            size,
            std::declval<std::function<std::optional<std::function<void(bool)>>(
                VkCommandBuffer, const BoundBuffer<int, typename T::Allocator>&, VkDeviceSize)>>())
    };
    {
        t.template withSingleAndStagingBuffer<int, int>(
            size, std::declval<std::function<std::optional<std::function<void(bool)>>(
                      VkCommandBuffer, const BoundBuffer<int, typename T::Allocator>&,
                      const BoundBuffer<int, typename T::Allocator>&, VkDeviceSize)>>())
    };
    { t.commandBuffer() };
    { t.submit() };
    // Non-blocking: process callbacks for batches that are already complete
    { t.poll() } -> std::same_as<void>;
    // Blocking: wait for all submitted batches to complete and invoke their callbacks
    { t.wait() } -> std::same_as<void>;
};

// Upload to a BufferSpan with a callback
// Callback signature: fn(VkDeviceSize offset, std::span<T> mapped)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands, class Fn>
    requires upload_callback<Fn, T>
void upload(StreamType& stream, const DeviceAndCommands& device, BufferSpan<T> dst, Fn&& fn) {
    stream.template withStagingBuffer<T>(
        dst.size(),
        [&, userOffset =
                VkDeviceSize{0}](VkCommandBuffer cmd, auto& stagingBuf,
                                 VkDeviceSize) mutable -> std::optional<std::function<void(bool)>> {
            fn(userOffset, stagingBuf.map().span());
            copyBuffer(device, cmd, BufferSpan(stagingBuf),
                       dst.subspan(userOffset, stagingBuf.size()));
            userOffset += stagingBuf.size();
            return std::nullopt;
        });
}

// Upload a contiguous range to a BufferSpan
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands,
          std::ranges::contiguous_range SrcRange>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<SrcRange>>, T>
void upload(StreamType& stream, const DeviceAndCommands& device, SrcRange&& srcRange,
            BufferSpan<T> dst) {
    if (std::ranges::size(srcRange) != dst.size())
        throw std::out_of_range("srcRange size must match BufferSpan size");
    auto it = std::ranges::begin(srcRange);
    upload(stream, device, dst, [&](VkDeviceSize offset, std::span<T> mapped) {
        std::copy_n(it + offset, mapped.size(), mapped.begin());
    });
}

// Download from BufferSpan with transform to vector<DstT>
// Transform signature: fn(VkDeviceSize offset, std::span<SrcT> input, std::span<DstT> output)
template <class DstT, class SrcT, staging_stream StreamType, device_and_commands DeviceAndCommands,
          class Fn>
    requires download_transform_callback<Fn, SrcT, DstT>
auto download(StreamType& stream, const DeviceAndCommands& device, BufferSpan<SrcT> src, Fn&& fn) {
    using Allocator = typename StreamType::Allocator;

    DownloadFutureBuilder<Fn, std::remove_const_t<SrcT>, DstT, Allocator, true> builder(
        std::forward<Fn>(fn), src.size());

    stream.template withStagingBuffer<std::remove_const_t<SrcT>>(
        src.size(),
        [&, userOffset =
                VkDeviceSize{0}](VkCommandBuffer cmd, auto& stagingBuf,
                                 VkDeviceSize) mutable -> std::optional<std::function<void(bool)>> {
            copyBuffer(device, cmd, src.subspan(userOffset, stagingBuf.size()),
                       BufferSpan(stagingBuf));
            auto callback = builder.addSubrange(userOffset, stagingBuf.map());
            userOffset += stagingBuf.size();
            return callback;
        });

    SemaphoreValue finalSemaphore = builder.m_state->subranges.empty()
                                        ? SemaphoreValue::makeSignalled()
                                        : stream.commandBuffer().nextSubmitSemaphore();
    return DownloadTransformFuture<Fn, std::remove_const_t<SrcT>, DstT, Allocator>{
        std::move(builder), finalSemaphore};
}

// Download from BufferSpan to vector<T> (identity transform)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands>
auto download(StreamType& stream, const DeviceAndCommands& device, BufferSpan<T> src) {
    return download<std::remove_const_t<T>>(
        stream, device, src,
        [](VkDeviceSize, std::span<T> input, std::span<std::remove_const_t<T>> output) {
            std::ranges::copy(input, output.begin());
        });
}

// Download from BufferSpan and call function on each batch
// Callback signature: fn(VkDeviceSize offset, std::span<T> data)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands, class Fn>
    requires download_foreach_callback<Fn, T>
auto downloadForEach(StreamType& stream, const DeviceAndCommands& device, BufferSpan<T> src,
                     Fn&& fn) {
    using Allocator = typename StreamType::Allocator;

    DownloadFutureBuilder<Fn, std::remove_const_t<T>, std::remove_const_t<T>, Allocator, false>
        builder(std::forward<Fn>(fn));

    stream.template withStagingBuffer<std::remove_const_t<T>>(
        src.size(),
        [&, userOffset =
                VkDeviceSize{0}](VkCommandBuffer cmd, auto& stagingBuf,
                                 VkDeviceSize) mutable -> std::optional<std::function<void(bool)>> {
            copyBuffer(device, cmd, src.subspan(userOffset, stagingBuf.size()),
                       BufferSpan(stagingBuf));
            auto callback = builder.addSubrange(userOffset, stagingBuf.map());
            userOffset += stagingBuf.size();
            return callback;
        });

    SemaphoreValue finalSemaphore = builder.m_state->subranges.empty()
                                        ? SemaphoreValue::makeSignalled()
                                        : stream.commandBuffer().nextSubmitSemaphore();
    return DownloadForEachHandle<Fn, std::remove_const_t<T>, Allocator>{std::move(builder),
                                                                        finalSemaphore};
}

// Convenience overloads accepting buffer types directly (forward to BufferSpan versions)
template <buffer DstBuffer, staging_stream StreamType, device_and_commands DeviceAndCommands,
          class Fn>
    requires upload_callback<Fn, container_view_t<DstBuffer>>
void upload(StreamType& stream, const DeviceAndCommands& device, DstBuffer& dst, Fn&& fn) {
    upload(stream, device, BufferSpan(dst), std::forward<Fn>(fn));
}

template <buffer DstBuffer, staging_stream StreamType, device_and_commands DeviceAndCommands,
          std::ranges::contiguous_range SrcRange>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<SrcRange>>,
                          container_view_t<DstBuffer>>
void upload(StreamType& stream, const DeviceAndCommands& device, SrcRange&& srcRange,
            DstBuffer& dst) {
    upload(stream, device, std::forward<SrcRange>(srcRange),
           BufferSpan(dst).subspan(0, std::ranges::size(srcRange)));
}

template <class DstT, buffer SrcBuffer, staging_stream StreamType,
          device_and_commands DeviceAndCommands, class Fn>
    requires download_transform_callback<Fn, container_view_t<SrcBuffer>, DstT>
auto download(StreamType& stream, const DeviceAndCommands& device, SrcBuffer& src, Fn&& fn) {
    return download<DstT>(stream, device, BufferSpan(src), std::forward<Fn>(fn));
}

template <buffer SrcBuffer, staging_stream StreamType, device_and_commands DeviceAndCommands>
auto download(StreamType& stream, const DeviceAndCommands& device, SrcBuffer& src) {
    return download(stream, device, BufferSpan(src));
}

template <buffer SrcBuffer, staging_stream StreamType, device_and_commands DeviceAndCommands,
          class Fn>
    requires download_foreach_callback<Fn, container_view_t<SrcBuffer>>
auto downloadForEach(StreamType& stream, const DeviceAndCommands& device, SrcBuffer& src, Fn&& fn) {
    return downloadForEach(stream, device, BufferSpan(src), std::forward<Fn>(fn));
}

// Create a device buffer and upload data via callback
// Usage: auto buf = vko::upload<T>(stream, device, allocator, size, usage, fn);
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands,
          allocator AllocatorT, class Fn>
auto upload(StreamType& stream, const DeviceAndCommands& device, AllocatorT& allocator,
            VkDeviceSize size, VkBufferUsageFlags usage, Fn&& fn,
            VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    vko::DeviceBuffer<T, AllocatorT> buffer(
        device, size, usage, vko::vma::allocationCreateInfo(memoryProperties), allocator);
    upload(stream, device, BufferSpan(buffer), std::forward<Fn>(fn));
    return buffer;
}

// Create a device buffer and upload data from a range
// Usage: auto buf = vko::upload(stream, device, allocator, data, usage);
template <std::ranges::sized_range SrcRange, staging_stream StreamType,
          device_and_commands DeviceAndCommands, allocator AllocatorT>
auto upload(StreamType& stream, const DeviceAndCommands& device, AllocatorT& allocator,
            SrcRange&& data, VkBufferUsageFlags usage,
            VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    using T = std::remove_cv_t<std::ranges::range_value_t<SrcRange>>;
    vko::DeviceBuffer<T, AllocatorT> buffer(device, std::ranges::size(data), usage,
                                            vko::vma::allocationCreateInfo(memoryProperties),
                                            allocator);
    upload(stream, device, std::forward<SrcRange>(data), BufferSpan(buffer));
    return buffer;
}

// Upload to VkDeviceAddress using VK_NV_copy_memory_indirect extension
// Callback signature: fn(VkDeviceSize offset, std::span<T> mapped)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands, class Fn>
    requires upload_callback<Fn, T>
void upload(StreamType& stream, const DeviceAndCommands& device, DeviceSpan<T> dst,
            VkDeviceSize size, Fn&& fn) {
    if (size != dst.size())
        throw std::out_of_range("size must match DeviceSpan size");
    stream.template withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, T>(
        size,
        [&, userOffset =
                VkDeviceSize{0}](VkCommandBuffer cmd, auto& indirectCmdBuf, auto& stagingBuf,
                                 VkDeviceSize) mutable -> std::optional<std::function<void(bool)>> {
            // Let user populate the staging buffer
            fn(userOffset, stagingBuf.map().span());

            // Setup indirect copy command: staging → device address
            VkDeviceAddress srcAddr = stagingBuf.address(device);
            VkDeviceAddress dstAddr = dst.data().raw() + userOffset * sizeof(T);

            indirectCmdBuf.map()[0] = VkCopyMemoryIndirectCommandNV{
                .srcAddress = srcAddr,
                .dstAddress = dstAddr,
                .size       = stagingBuf.size() * sizeof(T),
            };

            // Record indirect copy command
            VkDeviceAddress cmdAddr = indirectCmdBuf.address(device);
            device.vkCmdCopyMemoryIndirectNV(
                cmd, cmdAddr, 1, static_cast<uint32_t>(sizeof(VkCopyMemoryIndirectCommandNV)));

            userOffset += stagingBuf.size();
            return std::nullopt;
        });
}

// Upload a range to VkDeviceAddress using VK_NV_copy_memory_indirect extension
template <staging_stream StreamType, device_and_commands DeviceAndCommands,
          std::ranges::contiguous_range SrcRange>
void upload(StreamType& stream, const DeviceAndCommands& device, SrcRange&& srcRange,
            DeviceSpan<std::ranges::range_value_t<SrcRange>> dst) {
    using T = std::ranges::range_value_t<SrcRange>;
    if (std::ranges::size(srcRange) != dst.size())
        throw std::out_of_range("srcRange size must match DeviceSpan size");

    auto it = std::ranges::begin(srcRange);
    upload<T>(stream, device, dst, std::ranges::size(srcRange),
              [&](VkDeviceSize offset, std::span<T> mapped) {
                  std::copy_n(it + offset, mapped.size(), mapped.begin());
              });
}

// Download from VkDeviceAddress with transform using VK_NV_copy_memory_indirect extension
// Transform signature: fn(VkDeviceSize offset, std::span<T> input, std::span<DstT> output)
template <class DstT, class T, staging_stream StreamType, device_and_commands DeviceAndCommands,
          class Fn>
    requires download_transform_callback<Fn, T, DstT>
auto download(StreamType& stream, const DeviceAndCommands& device, DeviceSpan<T> src, Fn&& fn) {
    using Allocator = typename StreamType::Allocator;

    DownloadFutureBuilder<Fn, std::remove_const_t<T>, DstT, Allocator, true> builder(
        std::forward<Fn>(fn), src.size());

    stream
        .template withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::remove_const_t<T>>(
            src.size(),
            [&](VkCommandBuffer cmd, auto& indirectCmdBuf, auto& stagingBuf,
                VkDeviceSize offset) -> std::optional<std::function<void(bool)>> {
                // Setup indirect copy command
                VkDeviceAddress srcAddr = src.data().raw() + offset * sizeof(T);
                VkDeviceAddress dstAddr = stagingBuf.address(device);

                indirectCmdBuf.map()[0] = VkCopyMemoryIndirectCommandNV{
                    .srcAddress = srcAddr,
                    .dstAddress = dstAddr,
                    .size       = stagingBuf.size() * sizeof(std::remove_const_t<T>),
                };

                // Record indirect copy command
                VkDeviceAddress cmdAddr = indirectCmdBuf.address(device);
                device.vkCmdCopyMemoryIndirectNV(
                    cmd, cmdAddr, 1, static_cast<uint32_t>(sizeof(VkCopyMemoryIndirectCommandNV)));

                return builder.addSubrange(offset, std::move(stagingBuf.map()));
            });

    SemaphoreValue finalSemaphore = builder.m_state->subranges.empty()
                                        ? SemaphoreValue::makeSignalled()
                                        : stream.commandBuffer().nextSubmitSemaphore();

    return DownloadTransformFuture<Fn, std::remove_const_t<T>, DstT, Allocator>{std::move(builder),
                                                                                finalSemaphore};
}

// Download from VkDeviceAddress using VK_NV_copy_memory_indirect extension (identity)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands>
auto download(StreamType& stream, const DeviceAndCommands& device, DeviceSpan<T> src) {
    return download<std::remove_const_t<T>>(
        stream, device, src,
        [](VkDeviceSize, std::span<T> input, std::span<std::remove_const_t<T>> output) {
            std::ranges::copy(input, output.begin());
        });
}

// Download from VkDeviceAddress and call a function on each batch
// Callback signature: fn(VkDeviceSize offset, std::span<T> data)
template <class T, staging_stream StreamType, device_and_commands DeviceAndCommands, class Fn>
    requires download_foreach_callback<Fn, T>
auto downloadForEach(StreamType& stream, const DeviceAndCommands& device, DeviceSpan<T> src,
                     Fn&& fn) {
    using Allocator = typename StreamType::Allocator;

    DownloadFutureBuilder<Fn, std::remove_const_t<T>, std::remove_const_t<T>, Allocator, false>
        builder(std::forward<Fn>(fn));

    stream
        .template withSingleAndStagingBuffer<VkCopyMemoryIndirectCommandNV, std::remove_const_t<T>>(
            src.size(),
            [&](VkCommandBuffer cmd, auto& indirectCmdBuf, auto& stagingBuf,
                VkDeviceSize offset) -> std::optional<std::function<void(bool)>> {
                // Setup indirect copy command
                VkDeviceAddress srcAddr = src.data().raw() + offset * sizeof(T);
                VkDeviceAddress dstAddr = stagingBuf.address(device);

                indirectCmdBuf.map()[0] = VkCopyMemoryIndirectCommandNV{
                    .srcAddress = srcAddr,
                    .dstAddress = dstAddr,
                    .size       = stagingBuf.size() * sizeof(std::remove_const_t<T>),
                };

                // Record indirect copy command
                VkDeviceAddress cmdAddr = indirectCmdBuf.address(device);
                device.vkCmdCopyMemoryIndirectNV(
                    cmd, cmdAddr, 1, static_cast<uint32_t>(sizeof(VkCopyMemoryIndirectCommandNV)));

                return builder.addSubrange(offset, std::move(stagingBuf.map()));
            });

    SemaphoreValue finalSemaphore = builder.m_state->subranges.empty()
                                        ? SemaphoreValue::makeSignalled()
                                        : stream.commandBuffer().nextSubmitSemaphore();

    return DownloadForEachHandle<Fn, std::remove_const_t<T>, Allocator>{std::move(builder),
                                                                        finalSemaphore};
}

// Lightweight non-owning view for staging operations. Holds references to a
// CyclingCommandBuffer and StagingAllocator, providing the same API as StagingStream
// but with trivial destructor and no ownership responsibilities. Multiple
// StagingStreamRef instances can reference the same underlying resources.
// Follows the std::atomic_ref pattern.
template <class StagingAllocator,
          class CommandBuffer = CyclingCommandBuffer<typename StagingAllocator::DeviceAndCommands>>
class StagingStreamRef {
public:
    using DeviceAndCommands = typename StagingAllocator::DeviceAndCommands;
    using Allocator         = typename StagingAllocator::Allocator;
    using CommandBufferType = CommandBuffer;

    StagingStreamRef(CommandBuffer& commandBuffer, StagingAllocator& staging)
        : m_commandBuffer(commandBuffer)
        , m_staging(staging) {}

    // Extensibility primitive: chunked single+upTo allocation with callbacks
    // Like upload(), loops calling fn() for each chunk until size exhausted
    // Callback returns optional cleanup for this chunk (auto-registered)
    template <class TSingle, class TUpTo, class Fn>
        requires std::invocable<Fn, VkCommandBuffer, const BoundBuffer<TSingle, Allocator>&,
                                const BoundBuffer<TUpTo, Allocator>&,
                                VkDeviceSize> // offset
    void withSingleAndStagingBuffer(VkDeviceSize totalSize, Fn&& fn) {
        VkDeviceSize userOffset = 0;
        while (totalSize > 0) {
            auto [single, upTo] = allocateSingleAndUpTo<TSingle, TUpTo>(totalSize);

            // Invoke user callback - may return cleanup callback
            auto maybeCallback = fn(commandBuffer(), single, upTo, userOffset);

            // Auto-register cleanup if returned
            if (maybeCallback.has_value()) {
                m_staging.get().registerBatchCallback(std::move(*maybeCallback));
            }

            // upTo may be smaller than requested - loop until done
            userOffset += upTo.size();
            totalSize -= upTo.size();
        }
    }

    // Extensibility primitive: chunked staging buffer allocation with callbacks
    // Like upload(), loops calling fn() for each chunk until size exhausted
    // Callback returns optional cleanup for this chunk (auto-registered)
    template <class T, class Fn>
        requires std::invocable<Fn, VkCommandBuffer, const BoundBuffer<T, Allocator>&,
                                VkDeviceSize> // offset
    void withStagingBuffer(VkDeviceSize totalSize, Fn&& fn) {
        VkDeviceSize userOffset = 0;
        while (totalSize > 0) {
            auto& stagingBuf = allocateUpTo<T>(totalSize);

            // Invoke user callback - may return cleanup callback
            auto maybeCallback = fn(commandBuffer(), stagingBuf, userOffset);

            // Auto-register cleanup if returned
            if (maybeCallback.has_value()) {
                m_staging.get().registerBatchCallback(std::move(*maybeCallback));
            }

            // stagingBuf may be smaller than requested - loop until done
            userOffset += stagingBuf.size();
            totalSize -= stagingBuf.size();
        }
    }

    // Manual submission interface.
    SemaphoreValue submit() {
        std::array<VkSemaphoreSubmitInfo, 0> noWaits{};
        std::array<VkSemaphoreSubmitInfo, 0> noSignals{};
        return submit(noWaits, noSignals);
    }

    // TODO: to be really generic and match the Vulkan API, I think we want a "Submission" object
    template <typename WaitRange, typename SignalRange>
    SemaphoreValue
    submit(WaitRange&& waitInfos, SignalRange&& signalInfos,
           VkPipelineStageFlags2 timelineSemaphoreStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT) {
        SemaphoreValue semaphoreValue = m_commandBuffer.get().submit(
            std::forward<WaitRange>(waitInfos), std::forward<SignalRange>(signalInfos),
            timelineSemaphoreStageMask);
        if (m_hasUnsubmittedAllocations) {
            m_staging.get().endBatch(semaphoreValue);
            m_hasUnsubmittedAllocations = false;
        }
        return semaphoreValue;
    }

    // Get total allocated staging memory size in bytes
    VkDeviceSize size() const { return m_staging.get().size(); }

    // Get total staging memory capacity in bytes
    VkDeviceSize capacity() const { return m_staging.get().capacity(); }

    size_t unsubmittedTransfers() const { return m_staging.get().unsubmittedTransfers(); }
    size_t pendingTransfers() const { return m_staging.get().pendingTransfers(); }

    // Non-blocking: process callbacks for batches that are already complete
    void poll() { m_staging.get().poll(); }

    // Blocking: wait for all submitted batches to complete and invoke their callbacks
    void wait() { m_staging.get().wait(); }

    CommandBuffer& commandBuffer() { return m_commandBuffer.get(); }

private:
    // Helper to allocate from staging or submit and retry once. Guarantees
    // return of valid buffer or throws on persistent failure, indicating a bug
    // with the backing allocator.
    template <class T>
    BoundBuffer<T, Allocator>& allocateUpTo(VkDeviceSize size) {
        while (true) {
            auto* buf = m_staging.get().template allocateUpTo<T>(size);
            if (buf) {
                m_hasUnsubmittedAllocations = true;
                return *buf; // Success
            }

            if constexpr (StagingAllocator::AllocateAlwaysSucceeds) {
                // std::unreachable()
                throw std::runtime_error("AllocateAlwaysSucceeds but allocation failed");
            } else {
                if (!m_hasUnsubmittedAllocations) {
                    throw std::runtime_error("Staging allocation failed even after submit");
                }

                submit();
                // Retry
            }
        }
    }

    // Helper matching allocateUpTo structure for atomic pair allocation
    template <class TSingle, class TUpTo>
    std::pair<BoundBuffer<TSingle, Allocator>&, BoundBuffer<TUpTo, Allocator>&>
    allocateSingleAndUpTo(VkDeviceSize upToSize) {
        while (true) {
            auto result = m_staging.get().template allocateSingleAndUpTo<TSingle, TUpTo>(upToSize);
            if (result) {
                m_hasUnsubmittedAllocations = true;
                return *result; // Success
            }

            if constexpr (StagingAllocator::AllocateAlwaysSucceeds) {
                // std::unreachable()
                throw std::runtime_error("AllocateAlwaysSucceeds but allocation failed");
            } else {
                if (!m_hasUnsubmittedAllocations) {
                    throw std::runtime_error("Staging allocation failed even after submit");
                }

                submit();
                // Retry
            }
        }
    }

    std::reference_wrapper<CommandBuffer>    m_commandBuffer;
    std::reference_wrapper<StagingAllocator> m_staging;
    bool                                     m_hasUnsubmittedAllocations = false;
};

// Staging wrapper that also holds a queue and command buffer for
// staging/streaming memory and automatically submits command buffers as needed
// to reduce staging memory usage.
//
// Design: Separates ownership from usage following std::atomic_ref pattern.
// - StagingStreamRef: Lightweight view (reference_wrappers), contains all logic
// - StagingStream: Owns CyclingCommandBuffer + StagingAllocator, delegates to internal ref
// Benefits: Flexible sharing, multiple views can reference same resources,
// supports dedicated staging (AllocateAlwaysSucceeds) without mid-operation submits.
template <class StagingAllocator>
class StagingStream {
public:
    using DeviceAndCommands = typename StagingAllocator::DeviceAndCommands;
    using Allocator         = typename StagingAllocator::Allocator;

    StagingStream(TimelineQueue& queue, StagingAllocator&& staging)
        : m_commandBuffer(staging.device(), queue)
        , m_staging(std::move(staging))
        , m_ref(m_commandBuffer, m_staging) {}

    template <class TSingle, class TUpTo, class Fn>
        requires std::invocable<Fn, VkCommandBuffer, const BoundBuffer<TSingle, Allocator>&,
                                const BoundBuffer<TUpTo, Allocator>&, VkDeviceSize>
    void withSingleAndStagingBuffer(VkDeviceSize totalSize, Fn&& fn) {
        m_ref.template withSingleAndStagingBuffer<TSingle, TUpTo>(totalSize, std::forward<Fn>(fn));
    }

    template <class T, class Fn>
        requires std::invocable<Fn, VkCommandBuffer, const BoundBuffer<T, Allocator>&, VkDeviceSize>
    void withStagingBuffer(VkDeviceSize totalSize, Fn&& fn) {
        m_ref.template withStagingBuffer<T>(totalSize, std::forward<Fn>(fn));
    }

    SemaphoreValue submit() { return m_ref.submit(); }

    template <typename WaitRange, typename SignalRange>
    SemaphoreValue
    submit(WaitRange&& waitInfos, SignalRange&& signalInfos,
           VkPipelineStageFlags2 timelineSemaphoreStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT) {
        return m_ref.submit(std::forward<WaitRange>(waitInfos),
                            std::forward<SignalRange>(signalInfos), timelineSemaphoreStageMask);
    }

    VkDeviceSize                             size() const { return m_staging.size(); }
    VkDeviceSize                             capacity() const { return m_staging.capacity(); }
    CyclingCommandBuffer<DeviceAndCommands>& commandBuffer() { return m_commandBuffer; }

    // Non-blocking: process callbacks for batches that are already complete
    void poll() { m_ref.poll(); }

    // Blocking: wait for all submitted batches to complete and invoke their callbacks
    void wait() { m_ref.wait(); }

private:
    CyclingCommandBuffer<DeviceAndCommands> m_commandBuffer;
    StagingAllocator                        m_staging;
    StagingStreamRef<StagingAllocator>      m_ref; // Delegates all operations
};

} // namespace vko
