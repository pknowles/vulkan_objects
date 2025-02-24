// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <tuple>
#include <vector>
#include <vko/exceptions.hpp>
#include <vko/gen_structures.hpp>
#include <vko/gen_functions.hpp>
#include <vulkan/vulkan_core.h>

namespace vko
{

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using args = std::tuple<Args...>;
    using result = R;

    // Number of arguments
    static constexpr std::size_t arity = sizeof...(Args);

    // Get the type of the Nth argument
    template <std::size_t N>
    struct arg {
        using type = typename std::tuple_element<N, args>::type;
    };
};

template<class T>
using function_args_t = typename function_traits<T>::args;

template<class T>
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
    if constexpr(std::is_same_v<function_result_t<EnumerateFunc>, VkResult>) {
        if(VkResult err = enumerateFunc(args..., &count, nullptr); err != VK_SUCCESS)
            throw Exception(toString(err));
    } else {
        enumerateFunc(args..., &count, nullptr);
    }
    std::vector<Result> result(count);
    // TODO: do we really need to check for VK_INCOMPLETE? Just sounds like
    // polling to avoid a race condition between these calls.
    if constexpr (std::is_same_v<function_result_t<EnumerateFunc>, VkResult>) {
        if (VkResult err = enumerateFunc(args..., &count, result.data()); err != VK_SUCCESS)
            throw Exception(toString(err));
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
    if constexpr(std::is_same_v<function_result_t<GetFunc>, VkResult>) {
        if(VkResult err = getFunc(args..., &result); err != VK_SUCCESS)
            throw Exception(toString(err));
    } else {
        getFunc(args..., &result);
    }
    return result;
}

} // namespace vko
