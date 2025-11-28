// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <functional>
#include <stdexcept>
#include <vko/allocator.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>
#include <vko/timeline_queue.hpp>
#include <vko/unique_any.hpp>

namespace vko {

template <class T>
concept staging_allocator =
    allocator<typename T::Allocator> &&
    requires(T t, size_t size,
             std::function<void(const BoundBuffer<int, typename T::Allocator>&)> func) {
        { t.template with<int>(size, func) } -> std::same_as<void>;
        { t.template make<int>(size) } -> std::same_as<BoundBuffer<int, typename T::Allocator>*>;
    };

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
    using DeviceAndCommands = DeviceAndCommandsType;
    using Allocator         = AllocatorType;
    DedicatedStaging(const DeviceAndCommands& device, Allocator& allocator,
                     VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        : device(device)
        , allocator(allocator)
        , m_bufferUsageFlags(usage) {}

    template <class T>
    void with(size_t size, std::function<void(const vko::BoundBuffer<T, Allocator>&)> func) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            allocator.get());
        m_size += ptr->sizeBytes();
        func(*ptr);
        buffers.emplace_back(std::move(any));
    }

    // Returns a non-owning pointer to a temporary buffer to use for staging
    // memory
    template <class T>
    vko::BoundBuffer<T, Allocator>* make(size_t size) {
        auto [any, ptr] = makeUniqueAny<vko::BoundBuffer<T, Allocator>>(
            device.get(), size, m_bufferUsageFlags,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            allocator.get());
        m_size += ptr->sizeBytes();
        buffers.emplace_back(std::move(any));
        return ptr;
    }

    VkDeviceSize size() const { return m_size; }

private:
    std::vector<UniqueAny>                          buffers;
    std::reference_wrapper<const DeviceAndCommands> device;
    std::reference_wrapper<Allocator>               allocator;
    VkDeviceSize                                    m_size             = 0;
    VkBufferUsageFlags                              m_bufferUsageFlags = 0;
};
static_assert(staging_allocator<DedicatedStaging<Device, vko::vma::Allocator>>);

namespace vma {

// Staging buffer allocator that provides temporary buffers from cyclical memory
// pools. The caller must call releaseBuffers() with a SemaphoreValue that will
// be signaled when the buffers are no longer in use so that their pool can be
// reused. Use this when you have many small transfers to make on a regular
// basis to avoid frequent allocations/deallocations.
template <device_and_commands DeviceAndCommandsType>
class RecyclingStagingPool {
public:
    using DeviceAndCommands = DeviceAndCommandsType;
    using Allocator         = vma::Allocator;

    RecyclingStagingPool(const DeviceAndCommands& device, Allocator& allocator,
                         VkDeviceSize       poolSize,
                         VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        : device(device)
        , m_allocator(allocator)
        , m_poolSize(poolSize)
        , m_bufferUsageFlags(usage) {}

    template <class T>
    vko::BoundBuffer<T, Allocator>* make(size_t size) {
        return makeTmpBuffer<T>(size);
    }

    template <class T>
    void with(size_t size, std::function<void(const BoundBuffer<T, Allocator>&)> func) {
        func(*makeTmpBuffer<T>(size));
    }

    // Mark all buffers in the 'current' pools as free to be recycled once the
    // reuseSemaphore is signaled
    void releaseBuffers(SemaphoreValue reuseSemaphore) {
        // Peek at the next few pools in the in the queue. While there are at least
        // two ready, assume we have more than we need and free the front one.
        // TODO: profile - maybe we can reduce this to one? any hiccups in frame
        // time might cause frequent allocations/deallocations though.
        if (m_inUse.size() >= 2 && m_inUse.front().reuseSemaphore.wait(device, 0)) {
            while (m_inUse.size() >= 2 &&
                   std::next(m_inUse.begin())->reuseSemaphore.wait(device, 0)) {
                for(auto& pool : m_inUse.front().pools) {
                    m_totalPoolBytes -= pool.size();
                }
                m_inUse.pop_front();
            }
        }

        // Move current pools and buffers into the in-use queue
        m_inUse.push_back({std::move(m_currentPools), std::move(m_currentBuffers), reuseSemaphore, m_currentBufferBytes});
        m_currentPools.clear();
        m_currentBuffers.clear();
        m_currentBufferBytes = 0;
    }

    VkDeviceSize capacity() const { return m_totalPoolBytes; }
    VkDeviceSize size() const { return m_totalBufferBytes; }

private:
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
            .usage = VMA_MEMORY_USAGE_AUTO,
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
            .minBlockCount          = 0,
            .maxBlockCount          = 0,
            .priority               = 0.0f,
            .minAllocationAlignment = 0,
            .pMemoryAllocateNext    = nullptr,
        };
        return vma::Pool(m_allocator.get(), poolCreateInfo);
    }

    template <class T>
    BoundBuffer<T, Allocator>* makeTmpBuffer(size_t size) {
        // If there is no current pool, see if there is one we can reuse yet
        if (m_currentPools.empty()) {
            if (!m_inUse.empty() && m_inUse.front().reuseSemaphore.wait(device.get())) {
                m_totalBufferBytes -= m_inUse.front().bufferBytes;
                m_inUse.front().buffers.clear();
                m_inUse.front().bufferBytes = 0;
                if (m_inUse.front().pools.size() == 1) {
                    // Take the last pool from the in-use queue and reuse the
                    // std::vector memory
                    m_currentPools = std::move(m_inUse.front().pools);
                    m_currentBuffers = std::move(m_inUse.front().buffers);
                    m_inUse.pop_front();
                } else {
                    // Take just one pool from the in-use queue
                    m_currentPools.push_back(std::move(m_inUse.front().pools.back()));
                    m_inUse.front().pools.pop_back();
                    assert(m_currentBuffers.empty());
                }
            } else if (!m_inUse.empty()) {
                m_inUse
                    .pop_front(); // Skip pools that aren't ready yet (they'll be cleaned up later)
            }
        }

        // If there is no available pool, create a new one
        if (m_currentPools.empty()) {
            m_currentPools.push_back(makePool());
            m_totalPoolBytes += m_poolSize;
        }

        // Try to make a buffer from the current pools
        std::optional<BoundBuffer<T, Allocator>> buffer;
        static_assert(!std::is_default_constructible_v<BoundBuffer<int, vma::Allocator>>,
                      "This wouldn't be very RAII");
        try {
            buffer = BoundBuffer<T, Allocator>(
                device.get(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vma::allocationCreateInfo(m_currentPools.back(),
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                m_allocator.get());
        } catch (const std::exception& e) {
            // Retry once with a fresh pool
            m_currentPools.push_back(makePool());
            m_totalPoolBytes += m_poolSize;
            buffer = BoundBuffer<T, Allocator>(
                device.get(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vma::allocationCreateInfo(m_currentPools.back(),
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                m_allocator.get());
        }
        m_currentBufferBytes += buffer->sizeBytes();
        m_totalBufferBytes += buffer->sizeBytes();

        auto [any, ptr] = toUniqueAnyWithPtr(std::move(*buffer));
        m_currentBuffers.emplace_back(std::move(any));
        return ptr;
    }

private:
    struct InUseMemoryPools {
        std::vector<vma::Pool> pools;
        std::vector<UniqueAny> buffers;
        SemaphoreValue         reuseSemaphore;
        VkDeviceSize           bufferBytes = 0;
    };

    std::reference_wrapper<const DeviceAndCommands> device;
    std::reference_wrapper<Allocator>               m_allocator;
    VkDeviceSize                                    m_poolSize = 0;
    VkDeviceSize                                    m_totalPoolBytes = 0;
    VkDeviceSize                                    m_currentBufferBytes = 0;
    VkDeviceSize                                    m_totalBufferBytes = 0;
    std::vector<vma::Pool>                          m_currentPools;
    std::vector<UniqueAny>                          m_currentBuffers;
    std::deque<InUseMemoryPools>                    m_inUse;
    VkBufferUsageFlags                              m_bufferUsageFlags = 0;
};
static_assert(staging_allocator<RecyclingStagingPool<Device>>);

} // namespace vma

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
    staging.template with<T>(std::ranges::size(data),
                             [&](const vko::BoundBuffer<T, Allocator>& buffer) {
                                 std::ranges::copy(data, buffer.map().begin());
                                 VkBufferCopy bufferCopy{
                                     .srcOffset = 0,
                                     .dstOffset = dstOffsetElements * sizeof(T),
                                     .size      = std::ranges::size(data) * sizeof(T),
                                 };
                                 device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
                             });
}

// Non type-safe upload. Avoid where possible.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
BufferMapping<std::byte, typename StagingAllocator::Allocator>
uploadToBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
              VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size) {
    return staging.template with<std::byte>(
        size, [&](const vko::BoundBuffer<std::byte, typename StagingAllocator::Allocator>& buffer) {
            VkBufferCopy bufferCopy{
                .srcOffset = 0,
                .dstOffset = dstOffset,
                .size      = size,
            };
            device.vkCmdCopyBuffer(cmd, buffer, dstBuffer, 1, &bufferCopy);
            return buffer.map();
        });
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

template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
BufferMapping<std::byte, Allocator>
uploadBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
            VkDeviceAddress dstAddress, VkDeviceSize size) {
    auto* srcBuffer = staging.template make<std::byte>(size);

    // Temporary buffer to hold the VkCopyMemoryIndirectCommandNV
    auto* cmdBuffer     = staging.template make<VkCopyMemoryIndirectCommandNV>(1);
    cmdBuffer->map()[0] = VkCopyMemoryIndirectCommandNV{
        .srcAddress = srcBuffer->address(device),
        .dstAddress = dstAddress,
        .size       = size,
    };
    device.vkCmdCopyMemoryIndirectNV(cmd, cmdBuffer->address(device), 1,
                                     static_cast<uint32_t>(cmdBuffer->sizeBytes()));
    return srcBuffer->map();
}

template <class T, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
BufferMapping<T, Allocator>
download(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
         vko::DeviceBuffer<T, Allocator>& srcBuffer, VkDeviceSize srcOffsetElements = 0u) {
    if (srcOffsetElements >= srcBuffer.size())
        throw std::out_of_range("source offset is beyond buffer size");
    VkDeviceSize copySize = srcBuffer.size() - srcOffsetElements;
    return staging.template with<T>(copySize, [&](const vko::BoundBuffer<T, Allocator>& buffer) {
        VkBufferCopy bufferCopy{
            .srcOffset = srcOffsetElements * sizeof(T),
            .dstOffset = 0,
            .size      = copySize * sizeof(T),
        };
        device.vkCmdCopyBuffer(cmd, srcBuffer, buffer, 1, &bufferCopy);
        return buffer.map();
    });
}

// Non type-safe download. Avoid where possible.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
BufferMapping<std::byte, Allocator>
downloadBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
              VkBuffer srcBuffer, VkDeviceSize offset, size_t size) {
    return staging.template with<std::byte>(
        size, [&](const vko::BoundBuffer<std::byte, Allocator>& buffer) {
            VkBufferCopy bufferCopy{.srcOffset = offset, .dstOffset = 0, .size = size};
            device.vkCmdCopyBuffer(cmd, srcBuffer, buffer, 1, &bufferCopy);
            return buffer.map();
        });
}

template <std::ranges::contiguous_range DstRange, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands, allocator Allocator = vko::vma::Allocator>
    requires(!std::is_const_v<std::ranges::range_value_t<DstRange>>)
void immediateDownloadTo(
    StagingAllocator& staging, const DeviceAndCommands& device, VkCommandPool pool, VkQueue queue,
    vko::DeviceBuffer<std::remove_cv_t<std::ranges::range_value_t<DstRange>>, Allocator>& srcBuffer,
    size_t srcOffsetElements, DstRange&& dstRange) {
    using T = std::remove_cv_t<std::ranges::range_value_t<DstRange>>;
    simple::ImmediateCommandBuffer cmd(device, pool, queue);
    auto mapping = download<T>(staging, device, cmd, srcBuffer, srcOffsetElements);
    if (mapping.size() > std::ranges::size(dstRange))
        throw std::out_of_range("destination range is too small");
    std::ranges::copy(mapping, std::ranges::begin(dstRange));
}

template <class T, staging_allocator StagingAllocator, device_and_commands DeviceAndCommands,
          allocator Allocator = vko::vma::Allocator>
std::vector<T> immediateDownload(StagingAllocator& staging, const DeviceAndCommands& device,
                                 VkCommandPool pool, VkQueue queue,
                                 vko::DeviceBuffer<T, Allocator>& srcBuffer,
                                 size_t                           srcOffsetElements = 0u) {
    std::vector<T> result(srcBuffer.size());
    immediateDownloadTo(staging, device, pool, queue, srcBuffer, srcOffsetElements, result);
    return result;
}

// Non type-safe download. Avoid where possible. Downloads data from a device
// address using VkCmdCopyMemoryIndirectNV. Requires VK_NV_copy_memory_indirect
// to be enabled and the staging allocator must include
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
BufferMapping<std::byte, typename StagingAllocator::Allocator>
downloadBytes(StagingAllocator& staging, const DeviceAndCommands& device, VkCommandBuffer cmd,
              VkDeviceAddress address, VkDeviceSize size) {
    // Temporary host visible staging buffer
    auto* dstBuffer = staging.template make<std::byte>(size);

    // Temporary buffer to hold the VkCopyMemoryIndirectCommandNV
    auto*           cmdBuffer  = staging.template make<VkCopyMemoryIndirectCommandNV>(1);
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

// Non type-safe download. Avoid where possible.
template <std::ranges::contiguous_range Range, staging_allocator StagingAllocator,
          device_and_commands DeviceAndCommands>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<Range>>, std::byte>
void immediateDownloadBytesTo(StagingAllocator& staging, const DeviceAndCommands& device,
                              VkCommandPool pool, VkQueue queue, VkDeviceAddress address,
                              VkDeviceSize size, Range&& dstRange) {
    auto mapping = [&] {
        simple::ImmediateCommandBuffer cmd(device, pool, queue);
        return downloadBytes(staging, device, cmd, address, size);
    }();
    std::ranges::copy(mapping, std::ranges::begin(dstRange));
}

// Non type-safe download. Avoid where possible.
template <staging_allocator StagingAllocator, device_and_commands DeviceAndCommands>
std::vector<std::byte> immediateDownloadBytes(StagingAllocator&        staging,
                                              const DeviceAndCommands& device, VkCommandPool pool,
                                              VkQueue queue, VkDeviceAddress address,
                                              VkDeviceSize size) {
    std::vector<std::byte> result(size);
    immediateDownloadBytesTo(staging, device, pool, queue, address, size, result);
    return result;
}

} // namespace vko
