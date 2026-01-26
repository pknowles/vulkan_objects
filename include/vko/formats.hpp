// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <vko/device_address.hpp>
#include <vko/gen_formats.hpp>

namespace vko {

// Total bytes for an image with given extent and format
constexpr VkDeviceSize imageSizeBytes(VkExtent3D extent, uint32_t layerCount,
                                      const FormatInfo& info) {
    VkExtent3D blocks = ceil_div(extent, info.blockExtent);
    return static_cast<VkDeviceSize>(blocks.width) * blocks.height * blocks.depth * layerCount *
           info.blockSize;
}

// Runtime from vulkan format enum
constexpr VkDeviceSize imageSizeBytes(VkExtent3D extent, uint32_t layerCount, VkFormat format) {
    return imageSizeBytes(extent, layerCount, formatInfo(format));
}

// Compile-time from vulkan format enum
template <VkFormat Format>
constexpr VkDeviceSize imageSizeBytes(VkExtent3D extent, uint32_t layerCount) {
    return imageSizeBytes(extent, layerCount, toFormatInfo<Format>());
}

} // namespace vko
