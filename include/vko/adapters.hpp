// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <ranges>
#include <tuple>
#include <vector>
#include <vko/exceptions.hpp>
#include <vko/gen_functions.hpp>
#include <vko/gen_structures.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

template <typename T>
struct function_traits;

// Dangerous class to allow making a pointer to a temporary. Only works if the
// pointer is not kept longer than the expression.
template <class T>
const T* tmpPtr(const T& t) {
    return &t;
}

// Dangerous equivalent for a span
template <std::ranges::random_access_range Range>
std::span<const std::ranges::range_value_t<Range>> tmpSpan(const Range& t) {
    return t;
}

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
    using args   = std::tuple<Args...>;
    using result = R;

    // Number of arguments
    static constexpr std::size_t arity = sizeof...(Args);

    // Get the type of the Nth argument
    template <std::size_t N>
    struct arg {
        using type = typename std::tuple_element<N, args>::type;
    };
};

template <class T>
using function_args_t = typename function_traits<T>::args;

template <class T>
using function_result_t = typename function_traits<T>::result;

template <class T>
using tuple_last_t = std::tuple_element_t<std::tuple_size_v<T> - 1, T>;

// Helper to return a vector of vulkan objects from calls such as
// vkEnumeratePhysicalDevices that would normally need to be called twice and
// have an output array allocated up front.
template <class EnumerateFunc, typename... Args>
auto toVector(EnumerateFunc enumerateFunc, const Args&... args) {
    using Result = std::remove_pointer_t<tuple_last_t<function_args_t<EnumerateFunc>>>;

    uint32_t count = 0;
    if constexpr (std::is_same_v<function_result_t<EnumerateFunc>, VkResult>) {
        check(enumerateFunc(args..., &count, nullptr));
    } else {
        enumerateFunc(args..., &count, nullptr);
    }
    std::vector<Result> result(count);
    // TODO: do we really need to check for VK_INCOMPLETE? Just sounds like
    // polling to avoid a race condition between these calls.
    if constexpr (std::is_same_v<function_result_t<EnumerateFunc>, VkResult>) {
        check(enumerateFunc(args..., &count, result.data()));
    } else {
        enumerateFunc(args..., &count, result.data());
    }
    return result;
}

// Exploratory helper for vkGet*() calls. Not sure if this is a good idea yet.
// Functions need to be safe to blindly call otherwise users will lose trust,
// which is the whole point of this library.
template <class GetFunc, typename... Args>
auto get(GetFunc getFunc, const Args&... args) {
    using Result = std::remove_pointer_t<tuple_last_t<function_args_t<GetFunc>>>;
    Result result{};
    if constexpr (requires { struct_traits<Result>::sType; })
        result.sType = struct_traits<Result>::sType;
    if constexpr (std::is_same_v<function_result_t<GetFunc>, VkResult>) {
        check(getFunc(args..., &result));
    } else {
        getFunc(args..., &result);
    }
    return result;
}

} // namespace vko
