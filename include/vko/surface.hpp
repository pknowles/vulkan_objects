// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

namespace vko {

#include <vko/gen_handles.hpp>

// SurfaceVariant wrapper to appear as a single VkSurfaceKHR object
class SurfaceKHR {
public:
    SurfaceKHR(SurfaceVariant&& surface)
        : m_surface(std::move(surface)) {}
    operator VkSurfaceKHR() const { return object(); }
    VkSurfaceKHR object() const {
        return std::visit([](auto const& s) -> VkSurfaceKHR { return s.object(); }, m_surface);
    }
    const VkSurfaceKHR* ptr() const {
        return std::visit([](auto const& s) -> const VkSurfaceKHR* { return s.ptr(); }, m_surface);
    }

private:
    SurfaceVariant m_surface;
};

} // namespace vko
