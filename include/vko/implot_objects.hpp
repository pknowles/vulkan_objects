// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <implot.h>

namespace vko {
namespace implot {

// RAII wrapper for ImPlot context
class Context {
public:
    Context() { ImPlot::CreateContext(); }
    ~Context() { ImPlot::DestroyContext(); }
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;
};

} // namespace implot
} // namespace vko
