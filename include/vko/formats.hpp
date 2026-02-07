// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <vko/device_address.hpp>
#include <vko/gen_formats.hpp>

namespace vko {

// Calculate extent at a specific mip level
constexpr VkExtent3D mipExtent(VkExtent3D baseExtent, uint32_t mipLevel) {
    return {
        std::max(1u, baseExtent.width >> mipLevel),
        std::max(1u, baseExtent.height >> mipLevel),
        std::max(1u, baseExtent.depth >> mipLevel),
    };
}

// Describes a 3D rectangular region within an image subresource. Mirrors the
// image-side fields of VkBufferImageCopy. A 3D equivalent to a 1D range, except
// with the canonical Vulkan oddity of the third dimension sometimes being
// [offset.z, offset.z + extent.z) and sometimes [subresource.baseArrayLayer,
// subresource.baseArrayLayer + subresource.layerCount).
struct Region {
    VkImageSubresourceLayers subresource = {};
    VkOffset3D               offset      = {0, 0, 0};
    VkExtent3D               extent      = {};

    // Create a sub-region with offset/extent relative to this region, optionally selecting layers
    Region subregion(VkOffset3D subOffset, VkExtent3D subExtent, uint32_t layerOffset = 0,
                     uint32_t layerCount = 1u) const {
        assert(offset.x >= 0 && offset.y >= 0 && offset.z >= 0);
        assert(extent.width > 0 && extent.height > 0 && extent.depth > 0);
        assert(layerCount > 0);
        assert(offset.x + int32_t(extent.width) <= int32_t(extent.width));
        assert(offset.y + int32_t(extent.height) <= int32_t(extent.height));
        assert(offset.z + int32_t(extent.depth) <= int32_t(extent.depth));
        assert(layerOffset + layerCount <= subresource.layerCount);
        return {
            .subresource = {subresource.aspectMask, subresource.mipLevel,
                            subresource.baseArrayLayer + layerOffset, layerCount},
            .offset      = {offset.x + subOffset.x, offset.y + subOffset.y, offset.z + subOffset.z},
            .extent      = subExtent,
        };
    }
};

// Image handle bundled with its subregion descriptor.
class ImageRegion {
public:
    // Construct from image handle with explicit subregion
    ImageRegion(VkImage image, Region subregion)
        : m_image(image)
        , m_region(subregion) {}

    // Construct whole-image region from BoundImage with specified aspect
    template <class BoundImageT>
        requires requires(const BoundImageT& img) {
            { img.extent() } -> std::convertible_to<VkExtent3D>;
            { img.arrayLayers() } -> std::convertible_to<uint32_t>;
        } && std::convertible_to<BoundImageT, VkImage>
    ImageRegion(const BoundImageT& boundImage, VkImageAspectFlags aspectMask,
                uint32_t mipLevel = 0u)
        : m_image(boundImage)
        , m_region{
              .subresource =
                  {
                      .aspectMask     = aspectMask,
                      .mipLevel       = mipLevel,
                      .baseArrayLayer = 0,
                      .layerCount     = boundImage.arrayLayers(),
                  },
              .offset = {0, 0, 0},
              .extent = mipExtent(boundImage.extent(), mipLevel),
          } {
        assert(mipLevel < boundImage.mipLevels());
    }

    // Accessors
    VkImage       image() const { return m_image; }
    const Region& region() const { return m_region; }

    // Create a sub-region with offset/extent relative to this region, optionally selecting layers
    ImageRegion subregion(VkOffset3D offset, VkExtent3D extent, uint32_t layerOffset = 0u,
                          uint32_t layerCount = 1u) const {
        return {m_image, m_region.subregion(offset, extent, layerOffset, layerCount)};
    }

private:
    VkImage m_image  = VK_NULL_HANDLE;
    Region  m_region = {};
};

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

// Size in bytes for an image subregion
constexpr VkDeviceSize imageSizeBytes(const Region& region, VkFormat format) {
    return imageSizeBytes(region.extent, region.subresource.layerCount, format);
}

constexpr VkDeviceSize imageSizeBytes(const Region& region, const FormatInfo& info) {
    return imageSizeBytes(region.extent, region.subresource.layerCount, info);
}

} // namespace vko
