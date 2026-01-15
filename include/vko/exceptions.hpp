// Copyright (c) 2025 Pyarelal Knowles, MIT License
// This file was generated from handles.hpp.txt. Do not edit directly.
#pragma once

#include <concepts>
#include <exception>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

namespace vko {

class Exception : public std::exception {
public:
    template <class Str>
        requires std::constructible_from<std::string, std::decay_t<Str>>
    Exception(Str&& message)
        : m_what(std::forward<Str>(message)) {}

    Exception& addContext(const std::string& context) {
        m_what += "\n" + context;
        return *this;
    }

    virtual const char* what() const noexcept override { return m_what.c_str(); }

private:
    std::string m_what;
};

constexpr inline std::string_view to_string(VkResult result);

template <VkResult result>
class ResultException : public Exception {
public:
    ResultException()
        : Exception("Vulkan error: " + std::string(to_string(result))) {}
};

} // namespace vko

#include <vko/gen_exceptions.hpp>
