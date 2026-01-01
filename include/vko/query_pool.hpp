// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <cassert>
#include <chrono>
#include <deque>
#include <functional>
#include <optional>
#include <vector>
#include <vko/adapters.hpp>
#include <vko/handles.hpp>
#include <vko/timeline_queue.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

// Query results can only be 32-bit or 64-bit unsigned integers per Vulkan spec
template <typename T>
concept query_result_type = std::same_as<T, uint32_t> || std::same_as<T, uint64_t>;

// Non-owning query allocation result - just pool handle and index.
// Lifetime managed by the allocator. Templated on result type for type safety.
template <query_result_type T = uint64_t>
struct Query {
    using ResultType = T;
    
    const VkQueryPool pool = VK_NULL_HANDLE;
    const uint32_t    index = 0;
};

// Query result future. Works for timestamps, occlusion, pipeline statistics, etc.
// Type T is determined by the query pool and query allocation.
template <query_result_type T = uint64_t>
class QueryResultFuture {
public:
    using ResultType = T;

    QueryResultFuture(Query<T> query, SemaphoreValue semaphore, VkQueryResultFlags flags = 0)
        : m_query(query)
        , m_semaphore(std::move(semaphore))
        , m_flags(flags | (sizeof(T) == 8 ? VK_QUERY_RESULT_64_BIT : 0)) {}

    // Wait for query completion and return the result
    template <device_and_commands DeviceAndCommands>
    T get(const DeviceAndCommands& device) const {
        m_semaphore.wait(device);
        T result;
        // TODO: Ideally, semaphore wait should be sufficient without WAIT_BIT.
        // Investigate if additional synchronization (barrier/event) can eliminate the need for WAIT_BIT.
        check(device.vkGetQueryPoolResults(device, m_query.pool, m_query.index, 1,
                                          sizeof(T), &result, sizeof(T),
                                          m_flags | VK_QUERY_RESULT_WAIT_BIT));
        return result;
    }

    // Try to get result without waiting. Returns nullopt if not ready.
    template <device_and_commands DeviceAndCommands>
    std::optional<T> tryGet(const DeviceAndCommands& device) const {
        if (!m_semaphore.isSignaled(device)) {
            return std::nullopt;
        }
        T result;
        VkResult r = device.vkGetQueryPoolResults(device, m_query.pool, m_query.index, 1,
                                                  sizeof(T), &result, sizeof(T), m_flags);
        if (r == VK_NOT_READY) {
            return std::nullopt;
        }
        check(r);
        return result;
    }

    // Access to the underlying semaphore for advanced synchronization.
    // Use this for custom wait timeouts, composing with other semaphores, etc.
    const SemaphoreValue& semaphore() const { return m_semaphore; }
    const Query<T>& query() const { return m_query; }

private:
    Query<T>           m_query;
    SemaphoreValue     m_semaphore;
    VkQueryResultFlags m_flags;
};

// Write a timestamp query and return a future for the result.
// The future will wait on the semaphore before fetching results.
template <device_and_commands DeviceAndCommands>
QueryResultFuture<uint64_t> cmdWriteTimestamp(const DeviceAndCommands& device, VkCommandBuffer cmd,
                                               const Query<uint64_t>& query, VkPipelineStageFlags2 stage,
                                               SemaphoreValue semaphore) {
    device.vkCmdWriteTimestamp2(cmd, stage, query.pool, query.index);
    return QueryResultFuture<uint64_t>(query, std::move(semaphore));
}

// RAII wrapper for begin/end query. Automatically ends query on scope exit.
// Templated on result type for type safety through the entire chain.
template <query_result_type T = uint64_t>
class ScopedQuery {
public:
    using ResultType = T;

    template <device_and_commands DeviceAndCommands>
    ScopedQuery(const DeviceAndCommands& device, VkCommandBuffer cmd,
                const Query<T>& query, VkQueryControlFlags flags = 0)
        : m_vkCmdEndQuery(device.vkCmdEndQuery)
        , m_cmd(cmd)
        , m_query(query) {
        device.vkCmdBeginQuery(cmd, query.pool, query.index, flags);
    }

    ~ScopedQuery() {
        m_vkCmdEndQuery(m_cmd, m_query.pool, m_query.index);
    }

    // Non-copyable, non-movable
    ScopedQuery(const ScopedQuery&) = delete;
    ScopedQuery& operator=(const ScopedQuery&) = delete;
    ScopedQuery(ScopedQuery&&) = delete;
    ScopedQuery& operator=(ScopedQuery&&) = delete;

    const Query<T>& query() const { return m_query; }

    // Create a future for the query result. Type is automatically inferred.
    //   ScopedQuery<uint64_t> q(device, cmd, query);
    //   auto future = q.future(submitSemaphore);
    //   uint64_t result = future.get(device);
    QueryResultFuture<T> future(SemaphoreValue semaphore, VkQueryResultFlags flags = 0) const {
        return QueryResultFuture<T>(m_query, std::move(semaphore), flags);
    }

private:
    PFN_vkCmdEndQuery m_vkCmdEndQuery;
    VkCommandBuffer   m_cmd;
    Query<T>          m_query;
};

// Query pool allocator that allocates individual query indices from cyclical
// VkQueryPool objects. The caller must call endBatch() with a SemaphoreValue
// that will be signaled when the queries are no longer in use so that pools
// can be reset and reused. Use this for frame-based profiling where queries
// have predictable, batch-oriented lifetimes.
//
// Templated on result type T for type safety: timestamp queries return uint64_t,
// occlusion queries can return uint32_t or uint64_t depending on flags.
//
// Pattern: Similar to RecyclingStagingPool but for discrete query indices.
// - Multiple VkQueryPool objects, grows from minPools to maxPools
// - Uses vkCmdResetQueryPool to recycle entire pools
// - No per-query tracking or free list needed
// - Returns typed Query<T> struct (non-owning)
// - When full mid-frame: allocates new pool (up to max) or blocks
//
// Example:
//   RecyclingQueryPool<uint64_t> queries(device, VK_QUERY_TYPE_TIMESTAMP, 256, 3, 10);
//   auto q = queries.allocate(cmd);  // Returns Query<uint64_t>
//   auto future = cmdWriteTimestamp(device, cmd, q, stage, semaphore);
//   uint64_t timestamp = future.get(device);  // Type-safe!
template <query_result_type T = uint64_t, device_and_commands DeviceAndCommandsType = Device>
class RecyclingQueryPool {
public:
    using ResultType = T;
    using DeviceAndCommands = DeviceAndCommandsType;

    RecyclingQueryPool(
        const DeviceAndCommands& device,
        VkQueryType              queryType,
        uint32_t                 queriesPerPool = 256,
        size_t                   minPools       = 3,
        std::optional<size_t>    maxPools       = 5, // std::nullopt for unlimited
        VkQueryPipelineStatisticFlags pipelineStatistics = 0)
        : m_device(device)
        , m_queryType(queryType)
        , m_queriesPerPool(queriesPerPool)
        , m_maxPools(maxPools.value_or(0))
        , m_minPools(minPools)
        , m_pipelineStatistics(pipelineStatistics) {

        // Pre-allocate minimum pools as already-signaled batches
        // New pools start with queries in unavailable state, ready to use
        for (size_t i = 0; i < m_minPools; ++i) {
            PoolBatch batch;
            batch.pools.push_back(makePool());
            m_inUse.push_back(std::move(batch), SemaphoreValue::makeSignalled());
            ++m_totalPoolCount;
        }
    }

    RecyclingQueryPool(const RecyclingQueryPool&)            = delete;
    RecyclingQueryPool& operator=(const RecyclingQueryPool&) = delete;
    RecyclingQueryPool(RecyclingQueryPool&&)                 = default;
    RecyclingQueryPool& operator=(RecyclingQueryPool&&)      = default;

    // Allocate a single query index. Returns nullopt if allocator is full and
    // cannot allocate more pools. Caller should endBatch() and try again.
    std::optional<Query<T>> tryAllocate() {
        // Try to allocate from current pool
        if (hasCurrentPool() && m_currentPoolUsed < m_queriesPerPool) {
            return Query<T>{currentPool(), m_currentPoolUsed++};
        }

        // Need a new pool
        if (!addPoolToCurrentBatch()) {
            return std::nullopt; // At capacity
        }

        assert(hasCurrentPool() && m_currentPoolUsed < m_queriesPerPool);
        return Query<T>{currentPool(), m_currentPoolUsed++};
    }

    // Allocate a single query index, blocking if necessary when at max capacity.
    // Resets pools via vkResetQueryPool (host-side) when recycling them.
    Query<T> allocate() {
        while (true) {
            if (auto result = tryAllocate()) {
                return *result;
            }
            // At max capacity, must wait for a pool to become available
            endBatch(SemaphoreValue::makeSignalled()); // Flush current batch
            if (!addPoolToCurrentBatch()) {
                // Still can't allocate - this shouldn't happen if maxPools > 0
                throw std::runtime_error("Failed to allocate query after waiting");
            }
        }
    }

    // Mark all queries in the 'current' pools as recyclable once the
    // reuseSemaphore is signaled. After signaling, pools will be reset via
    // vkCmdResetQueryPool in the next command buffer that needs them.
    void endBatch(SemaphoreValue reuseSemaphore) {
        // Only move batch to m_inUse if it has pools (RAII: no empty batches)
        if (!m_current.pools.empty()) {
            m_inUse.push_back(std::move(m_current), reuseSemaphore);
            m_current = {};
            m_currentPoolUsed = 0;
        }
    }

    // Wait for all batches to finish and free excess pools
    void wait() {
        m_inUse.wait(m_device.get());
        freeExcessPools();
    }

    // Total number of allocated pools
    size_t poolCount() const { return m_totalPoolCount; }

    // Total query capacity across all pools
    uint32_t capacity() const { return static_cast<uint32_t>(m_totalPoolCount) * m_queriesPerPool; }

    const DeviceAndCommands& device() const { return m_device.get(); }

    uint32_t queriesPerPool() const { return m_queriesPerPool; }

private:
    struct PoolBatch {
        std::vector<QueryPool> pools;
    };

    bool hasCurrentPool() const { return !m_current.pools.empty(); }

    VkQueryPool currentPool() const { return m_current.pools.back(); }

    QueryPool makePool() {
        VkQueryPoolCreateInfo createInfo{
            .sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext              = nullptr,
            .flags              = 0,
            .queryType          = m_queryType,
            .queryCount         = m_queriesPerPool,
            .pipelineStatistics = m_pipelineStatistics,
        };
        QueryPool pool = QueryPool(m_device.get(), createInfo);
        
        // Reset new pool from host (queries start in undefined state)
        // vkResetQueryPool is core Vulkan 1.2+, no creation flag needed
        m_device.get().vkResetQueryPool(m_device.get(), pool, 0, m_queriesPerPool);
        
        return pool;
    }

    // Adds a pool to current batch (recycled if available, otherwise new)
    // Returns false if at max capacity and no pools are ready
    bool addPoolToCurrentBatch() {
        // Try to recycle from ready batches (non-blocking)
        while (!m_inUse.empty() && m_inUse.frontSemaphore().isSignaled(m_device.get())) {
            auto& front = m_inUse.front(m_device.get());
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
            ++m_totalPoolCount;
            m_currentPoolUsed = 0;
            return true;
        }

        // At max, block for a pool
        while (!m_inUse.empty()) {
            auto& front = m_inUse.front(m_device.get()); // BLOCKS
            recyclePool(front);

            // Remove batch if it's now empty (no empty batches allowed)
            if (front.pools.empty()) {
                m_inUse.pop_front();
            }

            return true;
        }

        return false;
    }

    // Takes a pool from batch, resets it from the host, adds to current.
    // Precondition: batch.pools is not empty
    // Postcondition: batch.pools may become empty (caller should check and remove)
    void recyclePool(PoolBatch& batch) {
        assert(!batch.pools.empty());

        VkQueryPool pool = batch.pools.back();
        
        // Reset from host (core Vulkan 1.2+, pools created with HOST_RESET_BIT)
        m_device.get().vkResetQueryPool(m_device.get(), pool, 0, m_queriesPerPool);

        m_current.pools.push_back(std::move(batch.pools.back()));
        batch.pools.pop_back();
        m_currentPoolUsed = 0;
    }

    void freeExcessPools() {
        // Free excess pools from ready batches
        m_inUse.consumeReady(m_device.get(), [this](PoolBatch& batch) {
            // Free pools from this batch until we're at minPools
            while (m_totalPoolCount > m_minPools && !batch.pools.empty()) {
                batch.pools.pop_back();
                --m_totalPoolCount;
            }

            // Consume empty batches immediately (RAII: no empty batches)
            return batch.pools.empty();
        });
    }

    std::reference_wrapper<const DeviceAndCommands> m_device;
    VkQueryType                                     m_queryType;
    uint32_t                                        m_queriesPerPool;
    size_t                                          m_maxPools = 0;
    size_t                                          m_minPools = 0;
    size_t                                          m_totalPoolCount = 0;
    VkQueryPipelineStatisticFlags                   m_pipelineStatistics = 0;

    PoolBatch                  m_current;
    uint32_t                   m_currentPoolUsed = 0;
    CompletionQueue<PoolBatch> m_inUse;
};

// ============================================================================
// StreamingQueryPool - PROPOSED (not implemented)
// ============================================================================
//
// For long-lived queries with unpredictable, variable lifetimes that may span
// multiple frames. Uses a free-list approach with async return-to-pool.
//
// Key differences from RecyclingQueryPool:
//   • Free list tracks available indices across ALL pools
//   • RAII wrapper (StreamingQueryHandle) returns index on destruction
//   • No vkCmdResetQueryPool - relies on query lifecycle management
//   • Creates new pools as needed (no max, or very high max)
//   • Handles out-of-order completion via pending return queue
//
// When to use:
//   • Background/streaming queries not tied to frame boundaries
//   • Highly variable query lifetimes (some 1 frame, some 10+)
//   • Occlusion queries with unpredictable GPU execution
//   • 1000s of queries with tight memory control needed
//
// When NOT to use (use RecyclingQueryPool instead):
//   • Frame-based profiling with predictable batch lifecycle
//   • Queries grouped into natural batches
//   • < 1000 queries per frame
//
// Key challenges:
//   • Async return: destructor can't wait (expensive), must post to pending queue
//   • Out-of-order completion: semaphores may signal in any order
//   • Free list management: need efficient lookup for available indices across pools
//   • Memory pressure: when to allocate new pools vs wait for returns
//
// Proposed interface:
//
// class StreamingQueryHandle {
// public:
//     VkQueryPool pool() const;
//     uint32_t index() const;
//     const SemaphoreValue& semaphore() const;
//     ~StreamingQueryHandle(); // Posts async return, doesn't wait
// };
//
// class StreamingQueryPool {
// public:
//     StreamingQueryPool(device, queryType, queriesPerPool, initialPools);
//
//     // Returns RAII handle that auto-returns index on destruction
//     StreamingQueryHandle allocate(SemaphoreValue completionSemaphore);
//
// private:
//     struct PendingReturn {
//         uint32_t globalIndex;  // Index across all pools
//         SemaphoreValue semaphore;
//     };
//
//     void checkPendingReturns();  // Called during allocate()
//     std::vector<QueryPool> m_pools;
//     std::vector<uint32_t> m_freeList;  // Global indices
//     std::vector<PendingReturn> m_pendingReturns;  // Unordered is fine
// };

} // namespace vko
