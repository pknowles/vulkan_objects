// Copyright (c) 2025 Pyarelal Knowles, MIT License
// This file was generated from handles.hpp.txt. Do not edit directly.
#pragma once

#include <concepts>
#include <exception>
#include <stdexcept>

namespace vko {

class Exception : public std::exception {
public:
    template<class Str>
        requires std::constructible_from<std::string, Str>
    Exception(Str&& message)
        : m_what(std::forward<Str>(message)) {}

    Exception& addContext(const std::string& context) { m_what += "\n" + context; return *this; }

    virtual const char* what() const noexcept override { return m_what.c_str(); }

private:
    std::string m_what;
};

} // namespace vko
