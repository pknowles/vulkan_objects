// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <tuple>
#include <utility>
#include <vko/structs.hpp>

namespace vko {

// Helper to create a modifier from a Vulkan struct type
// Usage: chainPNext(nullptr, with<VkPresentIdKHR>(1U, &presentId), [&](auto pNext) { ... })
template <typename StructType, typename... Args>
auto with(Args&&... args) {
    return [... cap = std::forward<Args>(args)](auto&& cont, const void* pNext) mutable {
        auto info = make<StructType>(pNext, cap...);
        return cont(&info);
    };
}

// Simple API: chainPNext(pNext, mod1, mod2, ..., final)
template <typename Func>
auto chainPNext(const void* pNext, Func&& func) {
    return func(pNext);
}

template <typename Mod, typename... Rest>
auto chainPNext(const void* pNext, Mod&& mod, Rest&&... rest) {
    return mod([&](const void* next) { return chainPNext(next, std::forward<Rest>(rest)...); },
               pNext);
}

// Args-aware API: chainPNext(pNext, std::tuple{args...}, std::tuple{mod1, mod2, final})
// Modifiers receive: (args_tuple&, continuation, pNext)
// Final receives: (args..., pNext)
template <typename... Args, typename Final>
auto chainPNext(const void* pNext, std::tuple<Args...>& args, Final&& final) {
    return std::apply([&](auto&... a) { return final(a..., pNext); }, args);
}

template <typename... Args, typename Mod, typename... Rest>
auto chainPNext(const void* pNext, std::tuple<Args...>& args, Mod&& mod, Rest&&... rest) {
    return mod(
        args, [&](const void* next) { return chainPNext(next, args, std::forward<Rest>(rest)...); },
        pNext);
}

template <typename... Args, typename... Callables>
auto chainPNext(const void* pNext, std::tuple<Args...> args, std::tuple<Callables...> callables) {
    return std::apply(
        [&](auto&&... cs) { return chainPNext(pNext, args, std::forward<decltype(cs)>(cs)...); },
        std::move(callables));
}

} // namespace vko
