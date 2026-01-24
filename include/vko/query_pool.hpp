// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

// Query pool utilities for GPU timestamps, occlusion, and statistics. Quick reference:
//
// Automatic QueryStream - safer, but higher level and more overhead
//   TimestampQueryStream<> stream;
//   SharedQuery<uint64_t> q = cmdWriteTimestamp(device, cmd, stream, stage);
//   stream.endBatch(semaphore);               - call after submit to enable recycling
//   uint64_t result = q.get(device);          - blocks until ready, reads result
//   q.semaphore().ready(device);              - non-blocking completion check
//
// QueryStream handles batching and pool recycling. For more control, use the
// underlying builder/batch API directly. Use what fits.
//
// Shared handles (survive batch destruction):
//   SharedQueryBatchBuilder<T, Type> builder;
//   SharedQuery<T> q = builder.with(device, [](Query<T> q) { ... });
//   SharedQueryBatch batch = complete(std::move(builder), semaphore);
//   T result = q.get(device);                 - blocks until semaphore, reads result
//
// Lower level, no shared_ptr:
//   QueryBatchBuilder<T, Type> builder;
//   QueryHandle h = builder.allocate(device);
//   QueryBatch batch = complete(std::move(builder), semaphore);
//   T result = batch.get(device, h);          - blocks until semaphore, reads result
//
// Timestamp helper for builders:
//   SharedQuery<uint64_t> = cmdWriteTimestamp(device, cmd, builder, stage);
//
// ScopedQuery (RAII for begin/end queries, e.g. occlusion):
//   ScopedQuery<T, BuilderType> scope(cmd, builder, flags, index);
//   SharedQuery<T> result = scope.handle();   - if using SharedQueryBatchBuilder
//
// Recycling (used internally by QueryStream, or use directly):
//   QueryPoolRecycler<Type>      - recycles individual query pools
//   QueryBatchRecycler<T, Type>  - recycles pools from completed batches
//
// Type aliases: Timestamp*, Occlusion* variants for common query types

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

template <typename R, typename QueryPoolType>
concept query_recycler = requires(R& r) {
    { r.empty() } -> std::convertible_to<bool>;
    { r.pop() } -> std::same_as<QueryPoolType>;
    { r.emptyVector() } -> std::same_as<std::vector<QueryPoolType>>;
};

constexpr uint32_t                      DefaultPoolCapacity       = 256;
constexpr VkQueryPipelineStatisticFlags DefaultPipelineStatistics = 0;

// Non-owning query allocation result - just pool handle and index.
// Lifetime managed by the allocator. Templated on result type for type safety.
template <query_result_type T>
struct Query {
    using ResultType = T;

    VkQueryPool pool  = VK_NULL_HANDLE;
    uint32_t    index = 0;
};

template <VkQueryType Type, uint32_t Capacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatisticsFlags = DefaultPipelineStatistics>
class TypedQueryPool : public QueryPool {
public:
    static constexpr VkQueryType                   QueryType          = Type;
    static constexpr uint32_t                      PoolCapacity       = Capacity;
    static constexpr VkQueryPipelineStatisticFlags PipelineStatistics = PipelineStatisticsFlags;

    template <device_and_commands DeviceAndCommands>
    TypedQueryPool(const DeviceAndCommands& device, VkQueryPoolCreateFlags flags = 0,
                   const void* pNext = nullptr)
        : QueryPool(device, VkQueryPoolCreateInfo{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                                  .pNext = pNext,
                                                  .flags = flags,
                                                  .queryType          = QueryType,
                                                  .queryCount         = PoolCapacity,
                                                  .pipelineStatistics = PipelineStatistics}) {}

    using QueryPool::operator VkQueryPool;
};

struct QueryHandle {
    size_t   poolIndex;
    uint32_t queryIndex;
};

template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
struct QueryBatchState {
    using QueryPoolType = TypedQueryPool<QueryType, PoolCapacity, PipelineStatistics>;

    template <device_and_commands DeviceAndCommands>
    QueryHandle allocate(const DeviceAndCommands& device) {
        if (m_pools.empty() || m_lastSize == PoolCapacity) {
            m_pools.emplace_back(device); // allocate a new pool
            device.vkResetQueryPool(device, m_pools.back(), 0, PoolCapacity);
            m_lastSize = 0;
        }
        return {m_pools.size() - 1, m_lastSize++};
    }

    template <device_and_commands DeviceAndCommands, class RecyclerType>
    QueryHandle allocate(const DeviceAndCommands& device, RecyclerType& recycler) {
        if (m_pools.empty())
            m_pools = recycler.emptyVector(); // try to avoid a heap allocation
        if (m_pools.empty() || m_lastSize == PoolCapacity) {
            std::optional<QueryPoolType> pool;
            if constexpr (requires { recycler.tryPop(device); })
                pool = recycler.tryPop(device);
            else
                pool = recycler.tryPop();
            if (pool)
                m_pools.push_back(std::move(*pool));
            else
                m_pools.emplace_back(device);
            device.vkResetQueryPool(device, m_pools.back(), 0, PoolCapacity);
            m_lastSize = 0;
        }
        return {m_pools.size() - 1, m_lastSize++};
    }

    Query<T> query(QueryHandle handle) const {
        return {m_pools.at(handle.poolIndex), handle.queryIndex};
    }

    template <device_and_commands DeviceAndCommands>
    T get(const DeviceAndCommands& device, QueryHandle handle) {
        download(device);
        return m_results.at(handle.poolIndex).at(handle.queryIndex);
    }

    template <device_and_commands DeviceAndCommands>
    void download(const DeviceAndCommands& device) {
        // Pools may not exist if already recycled, which implies downloaded
        m_semaphore.value().wait(
            device); // always wait, even if the batch is empty in case core relies on the wait
        if (m_results.empty() && !m_pools.empty()) {
            m_results.reserve(m_pools.size());
            for (const auto& pool : m_pools) {
                m_results.emplace_back();
                bool     last = (m_results.size() == m_pools.size());
                uint32_t size = last ? m_lastSize : PoolCapacity;
                // Ideally this would rely on the semaphore wait rather than
                // blocking with VK_QUERY_RESULT_WAIT_BIT. Without WAIT_BIT,
                // results are garbage though so maybe it triggers some internal
                // cache flush.
                VkQueryResultFlags flags = sizeof(T) == 8 ? VK_QUERY_RESULT_64_BIT : 0;
                flags |= VK_QUERY_RESULT_WAIT_BIT;
                device.vkGetQueryPoolResults(device, pool, 0, size, size * sizeof(T),
                                             m_results.back().data(), sizeof(T), flags);
            }
        }
    }

    template <device_and_commands DeviceAndCommands>
    std::vector<QueryPoolType> recyclePools(const DeviceAndCommands& device) {
        download(device);
        return std::move(m_pools);
    }

    std::vector<std::array<T, PoolCapacity>> m_results;
    std::vector<QueryPoolType>               m_pools;
    uint32_t                                 m_lastSize = 0;

    // Delayed assignment until the builder completes with a submission semaphore
    std::optional<SemaphoreValue> m_semaphore;
};

template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class SharedQuery {
public:
    using QueryBatchStateType = QueryBatchState<T, QueryType, PoolCapacity, PipelineStatistics>;
    SharedQuery(QueryHandle h, std::shared_ptr<QueryBatchStateType> batch)
        : m_batch(std::move(batch))
        , m_handle(h) {}

    template <device_and_commands DeviceAndCommands>
    T get(const DeviceAndCommands& device) const {
        return m_batch->get(device, m_handle);
    }

    Query<T> query() const { return m_batch->query(m_handle); }

    // Use to poll or wait for completion with a timeout.
    const SemaphoreValue& semaphore() const { return m_batch->m_semaphore.value(); }

private:
    std::shared_ptr<QueryBatchStateType> m_batch;
    QueryHandle                          m_handle;
};

// Provides access to the query batch state after it receives a submission
// semaphore via the builder interface.
template <query_result_type T, bool Shared, VkQueryType QueryType,
          uint32_t                      PoolCapacity       = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class QueryBatchImpl {
public:
    using QueryPoolType       = TypedQueryPool<QueryType, PoolCapacity, PipelineStatistics>;
    using QueryBatchStateType = QueryBatchState<T, QueryType, PoolCapacity, PipelineStatistics>;
    using StateType =
        std::conditional_t<Shared, std::shared_ptr<QueryBatchStateType>, QueryBatchStateType>;

    QueryBatchImpl(StateType batchState, SemaphoreValue semaphore)
        : m_state(std::move(batchState)) {
        state().m_semaphore = std::move(semaphore);
    }

    template <device_and_commands DeviceAndCommands>
    T get(const DeviceAndCommands& device, QueryHandle handle) {
        return state().get(device, handle);
    }

    template <device_and_commands DeviceAndCommands>
    void download(const DeviceAndCommands& device) {
        state().download(device);
    }

    // Use to poll or wait for completion with a timeout.
    const SemaphoreValue& semaphore() const { return state().m_semaphore.value(); }

    template <device_and_commands DeviceAndCommands>
    std::vector<QueryPoolType> recyclePools(const DeviceAndCommands& device) {
        return state().recyclePools(device);
    }

private:
    QueryBatchStateType& state() {
        if constexpr (Shared)
            return *m_state;
        else
            return m_state;
    }

    const QueryBatchStateType& state() const {
        if constexpr (Shared)
            return *m_state;
        else
            return m_state;
    }

    StateType m_state;
};

template <VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class QueryPoolRecycler {
public:
    using QueryPoolType = TypedQueryPool<QueryType, PoolCapacity, PipelineStatistics>;

    void push(std::vector<QueryPoolType>&& pools) { m_pools.push_back(std::move(pools)); }

    bool empty() const { return m_pools.empty(); }

    std::optional<QueryPoolType> tryPop() {
        if (m_pools.empty())
            return std::nullopt;
        auto result = std::move(m_pools.back().back());
        m_pools.back().pop_back();
        if (m_pools.back().empty()) {
            m_emptyVector = std::move(m_pools.back());
            m_pools.pop_back();
        }
        return result;
    }

    // Returns an empty vector, possibly reusing internal storage
    std::vector<QueryPoolType> emptyVector() { return std::move(m_emptyVector); }

private:
    std::vector<std::vector<QueryPoolType>> m_pools;
    std::vector<QueryPoolType>              m_emptyVector;
};

template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class QueryBatchRecycler {
public:
    using QueryPoolType         = TypedQueryPool<QueryType, PoolCapacity, PipelineStatistics>;
    using QueryPoolRecyclerType = QueryPoolRecycler<QueryType, PoolCapacity, PipelineStatistics>;
    using QueryBatchType = QueryBatchImpl<T, true, QueryType, PoolCapacity, PipelineStatistics>;

    void push(QueryBatchType batch) { m_pending.push_back(std::move(batch)); }

    // Note: !empty() does not imply there are ready batches to recycle.
    bool empty() const { return m_poolRecycler.empty() && m_pending.empty(); }

    template <device_and_commands DeviceAndCommands>
    std::optional<QueryPoolType> tryPop(const DeviceAndCommands& device) {
        if (auto pool = m_poolRecycler.tryPop())
            return pool;
        reclaimReady(device);
        return m_poolRecycler.tryPop();
    }

    template <device_and_commands DeviceAndCommands>
    void reclaimReady(const DeviceAndCommands& device) {
        while (!m_pending.empty() && m_pending.front().semaphore().isSignaled(device)) {
            m_poolRecycler.push(m_pending.front().recyclePools(device));
            m_pending.pop_front();
        }
    }

    // TODO: premature optimization?
    std::vector<QueryPoolType> emptyVector() { return m_poolRecycler.emptyVector(); }

private:
    QueryPoolRecyclerType      m_poolRecycler;
    std::deque<QueryBatchType> m_pending;
};

// Provides access to the builder interface of the query batch state.
template <query_result_type T, bool Shared, VkQueryType QueryType,
          uint32_t                      PoolCapacity       = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class QueryBatchBuilderImpl {
public:
    using QueryPoolRecyclerType = QueryPoolRecycler<QueryType, PoolCapacity, PipelineStatistics>;
    using QueryPoolType         = TypedQueryPool<QueryType, PoolCapacity, PipelineStatistics>;
    using QueryBatchStateType   = QueryBatchState<T, QueryType, PoolCapacity, PipelineStatistics>;
    using QueryBatchType = QueryBatchImpl<T, Shared, QueryType, PoolCapacity, PipelineStatistics>;
    using StateType =
        std::conditional_t<Shared, std::shared_ptr<QueryBatchStateType>, QueryBatchStateType>;
    using HandleType =
        std::conditional_t<Shared, SharedQuery<T, QueryType, PoolCapacity, PipelineStatistics>,
                           QueryHandle>;

    QueryBatchBuilderImpl()
        requires(Shared)
        : m_state(std::make_shared<QueryBatchStateType>()) {}
    QueryBatchBuilderImpl()
        requires(!Shared)
    {}

    template <device_and_commands DeviceAndCommands, class Fn>
        requires(!Shared)
    QueryHandle with(const DeviceAndCommands& device, Fn&& fn) {
        auto handle = state().allocate(device);
        fn(state().query(handle));
        return handle;
    }

    template <device_and_commands DeviceAndCommands, class RecyclerType, class Fn>
        requires(!Shared)
    QueryHandle with(const DeviceAndCommands& device, RecyclerType& recycler, Fn&& fn) {
        auto handle = state().allocate(device, recycler);
        fn(state().query(handle));
        return handle;
    }

    template <device_and_commands DeviceAndCommands, class Fn>
        requires(Shared)
    HandleType with(const DeviceAndCommands& device, Fn&& fn) {
        auto handle = state().allocate(device);
        fn(state().query(handle));
        return HandleType{handle, m_state};
    }

    template <device_and_commands DeviceAndCommands, class RecyclerType, class Fn>
        requires(Shared)
    HandleType with(const DeviceAndCommands& device, RecyclerType& recycler, Fn&& fn) {
        auto handle = state().allocate(device, recycler);
        fn(state().query(handle));
        return HandleType{handle, m_state};
    }

    // Complete the batch and return a QueryBatch for reading results
    template <class SemaphoreValueType>
    friend QueryBatchType complete(QueryBatchBuilderImpl&& builder,
                                   SemaphoreValueType&&    semaphore) {
        // Builder object is destroyed after moving out its state
        return QueryBatchType(std::move(builder.m_state),
                              std::forward<SemaphoreValueType>(semaphore));
    }

    // Non-shared accessor. Shared handles should access their own state.
    Query<T> query(const QueryHandle& handle) const
        requires(!Shared)
    {
        return state().query(handle);
    }

private:
    QueryBatchStateType& state() {
        if constexpr (Shared)
            return *m_state;
        else
            return m_state;
    }

    const QueryBatchStateType& state() const {
        if constexpr (Shared)
            return *m_state;
        else
            return m_state;
    }

    StateType m_state;
};

// Write a timestamp query and return a handle to query the result from the
// completed QueryBatchBuilder. Return type depends on whether builder is shared.
template <device_and_commands DeviceAndCommands, class BuilderType>
auto cmdWriteTimestamp(const DeviceAndCommands& device, VkCommandBuffer cmd, BuilderType& builder,
                       VkPipelineStageFlags2 stage) {
    return builder.with(device, [&](const Query<uint64_t>& query) {
        device.vkCmdWriteTimestamp2(cmd, stage, query.pool, query.index);
    });
}

// Write a timestamp query with recycler support
template <device_and_commands DeviceAndCommands, class BuilderType, class RecyclerType>
auto cmdWriteTimestamp(const DeviceAndCommands& device, VkCommandBuffer cmd, BuilderType& builder,
                       RecyclerType& recycler, VkPipelineStageFlags2 stage) {
    return builder.with(device, recycler, [&](const Query<uint64_t>& query) {
        device.vkCmdWriteTimestamp2(cmd, stage, query.pool, query.index);
    });
}

// Combines a recycler and a shared builder for simple batch workflows.
template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
class QueryStream {
public:
    using RecyclerType = QueryBatchRecycler<T, QueryType, PoolCapacity, PipelineStatistics>;
    using BuilderType = QueryBatchBuilderImpl<T, true, QueryType, PoolCapacity, PipelineStatistics>;
    using SharedQueryType = SharedQuery<T, QueryType, PoolCapacity, PipelineStatistics>;

    template <device_and_commands DeviceAndCommands, class Fn>
    SharedQueryType with(const DeviceAndCommands& device, Fn&& fn) {
        return m_builder.with(device, m_recycler, std::forward<Fn>(fn));
    }

    void endBatch(SemaphoreValue semaphore) {
        m_recycler.push(complete(std::move(m_builder), std::move(semaphore)));
        m_builder = BuilderType{}; // Re-initialize for next batch
    }

private:
    RecyclerType m_recycler;
    BuilderType  m_builder;
};

// Write a timestamp query via QueryStream
template <device_and_commands DeviceAndCommands, query_result_type T, VkQueryType QueryType,
          uint32_t PoolCapacity, VkQueryPipelineStatisticFlags PipelineStatistics>
auto cmdWriteTimestamp(const DeviceAndCommands& device, VkCommandBuffer cmd,
                       QueryStream<T, QueryType, PoolCapacity, PipelineStatistics>& stream,
                       VkPipelineStageFlags2                                        stage) {
    return stream.with(device, [&](const Query<uint64_t>& query) {
        device.vkCmdWriteTimestamp2(cmd, stage, query.pool, query.index);
    });
}

// RAII wrapper for begin/end query. Automatically ends query on scope exit.
// Use with QueryBatchBuilderImpl to track occlusion or other begin/end queries.
// Note: The builder must outlive this object - destructor queries the builder's state.
template <query_result_type T, class QueryBatchBuilderType>
class ScopedQuery {
public:
    using ResultType = T;
    using HandleType = typename QueryBatchBuilderType::HandleType;

    template <device_and_commands DeviceAndCommands>
    ScopedQuery(const DeviceAndCommands& device, VkCommandBuffer cmd,
                QueryBatchBuilderType& builder, VkQueryControlFlags flags = 0)
        : m_vkCmdEndQuery(device.vkCmdEndQuery)
        , m_cmd(cmd)
        , m_builder(builder)
        , m_handle(builder.with(device, [&](const Query<T>& query) {
            device.vkCmdBeginQuery(cmd, query.pool, query.index, flags);
        })) {}

    template <device_and_commands DeviceAndCommands, class RecyclerType>
    ScopedQuery(const DeviceAndCommands& device, VkCommandBuffer cmd,
                QueryBatchBuilderType& builder, RecyclerType& recycler,
                VkQueryControlFlags flags = 0)
        : m_vkCmdEndQuery(device.vkCmdEndQuery)
        , m_cmd(cmd)
        , m_builder(builder)
        , m_handle(builder.with(device, recycler, [&](const Query<T>& query) {
            device.vkCmdBeginQuery(cmd, query.pool, query.index, flags);
        })) {}

    ScopedQuery(const ScopedQuery&)            = delete;
    ScopedQuery& operator=(const ScopedQuery&) = delete;

    ~ScopedQuery() {
        auto q = query();
        m_vkCmdEndQuery(m_cmd, q.pool, q.index);
    }

    HandleType handle() const { return m_handle; }
    operator HandleType() const { return m_handle; }

private:
    Query<T> query() const {
        if constexpr (requires { m_handle.query(); })
            return m_handle.query(); // SharedQuery keeps state alive
        else
            return m_builder.query(m_handle); // May throw if builder moved
    }

    PFN_vkCmdEndQuery      m_vkCmdEndQuery;
    VkCommandBuffer        m_cmd;
    QueryBatchBuilderType& m_builder;
    HandleType             m_handle;
};

// Generic builder aliases (specify T, QueryType, etc.)
template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
using QueryBatchBuilder =
    QueryBatchBuilderImpl<T, false, QueryType, PoolCapacity, PipelineStatistics>;

template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
using SharedQueryBatchBuilder =
    QueryBatchBuilderImpl<T, true, QueryType, PoolCapacity, PipelineStatistics>;

// Generic batch aliases
template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
using QueryBatch = QueryBatchImpl<T, false, QueryType, PoolCapacity, PipelineStatistics>;

template <query_result_type T, VkQueryType QueryType, uint32_t PoolCapacity = DefaultPoolCapacity,
          VkQueryPipelineStatisticFlags PipelineStatistics = DefaultPipelineStatistics>
using SharedQueryBatch = QueryBatchImpl<T, true, QueryType, PoolCapacity, PipelineStatistics>;

// Timestamp-specific aliases (most common use case)
template <uint32_t PoolCapacity = DefaultPoolCapacity>
using TimestampQueryBatchBuilder =
    QueryBatchBuilder<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using SharedTimestampQueryBatchBuilder =
    SharedQueryBatchBuilder<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using TimestampQueryBatch = QueryBatch<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using SharedTimestampQueryBatch = SharedQueryBatch<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using TimestampQueryPoolRecycler = QueryPoolRecycler<VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using TimestampQueryBatchRecycler =
    QueryBatchRecycler<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using TimestampQueryStream = QueryStream<uint64_t, VK_QUERY_TYPE_TIMESTAMP, PoolCapacity>;

// Occlusion-specific aliases
template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using OcclusionQueryBatchBuilder = QueryBatchBuilder<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using SharedOcclusionQueryBatchBuilder =
    SharedQueryBatchBuilder<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using OcclusionQueryBatch = QueryBatch<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using SharedOcclusionQueryBatch = SharedQueryBatch<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <uint32_t PoolCapacity = DefaultPoolCapacity>
using OcclusionQueryPoolRecycler = QueryPoolRecycler<VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using OcclusionQueryBatchRecycler = QueryBatchRecycler<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

template <query_result_type T = uint64_t, uint32_t PoolCapacity = DefaultPoolCapacity>
using OcclusionQueryStream = QueryStream<T, VK_QUERY_TYPE_OCCLUSION, PoolCapacity>;

// StreamingQueryPool - PROPOSED (not implemented)
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
