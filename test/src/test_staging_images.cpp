// Copyright (c) 2026 Pyarelal Knowles, MIT License

// TODO: switch to TimelineQueue
#define VULKAN_OBJECTS_ENABLE_FOOTGUNS

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <test_context_fixtures.hpp>
#include <vko/allocator.hpp>
#include <vko/formats.hpp>
#include <vko/staging_memory.hpp>
#include <vko/timeline_queue.hpp>

// =============================================================================
// Test Helpers
// =============================================================================

// Create a test image with standard settings for transfer tests
inline vko::BoundImage<> makeTestImage(Context& ctx, VkExtent3D extent, VkFormat format,
                                       uint32_t arrayLayers = 1, uint32_t mipLevels = 1,
                                       VkImageCreateFlags flags = 0) {
    VkImageType imageType = VK_IMAGE_TYPE_2D;
    if (extent.depth > 1)
        imageType = VK_IMAGE_TYPE_3D;
    else if (extent.height == 1)
        imageType = VK_IMAGE_TYPE_1D;

    return vko::BoundImage(
        ctx.device,
        VkImageCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = flags,
            .imageType   = imageType,
            .format      = format,
            .extent      = extent,
            .mipLevels   = mipLevels,
            .arrayLayers = arrayLayers,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = VK_IMAGE_TILING_OPTIMAL,
            .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        },
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx.allocator);
}

// Transition image layout with a pipeline barrier
template <class StreamType>
void transitionImageLayout(StreamType& stream, const vko::Device& device, VkImage image,
                           VkImageLayout oldLayout, VkImageLayout newLayout,
                           VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           uint32_t baseMipLevel = 0, uint32_t levelCount = 1,
                           uint32_t baseArrayLayer = 0, uint32_t layerCount = 1) {
    // Determine stage/access masks from layouts
    VkPipelineStageFlags2 srcStage  = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage  = VK_PIPELINE_STAGE_2_COPY_BIT;
    VkAccessFlags2        dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        srcStage  = VK_PIPELINE_STAGE_2_COPY_BIT;
        srcAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        srcStage  = VK_PIPELINE_STAGE_2_COPY_BIT;
        srcAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        dstAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        dstAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
    }

    VkImageMemoryBarrier2 barrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = srcStage,
        .srcAccessMask       = srcAccess,
        .dstStageMask        = dstStage,
        .dstAccessMask       = dstAccess,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange =
            {
                .aspectMask     = aspectMask,
                .baseMipLevel   = baseMipLevel,
                .levelCount     = levelCount,
                .baseArrayLayer = baseArrayLayer,
                .layerCount     = layerCount,
            },
    };
    VkDependencyInfo depInfo{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &barrier,
    };
    device.vkCmdPipelineBarrier2(stream.commandBuffer(), &depInfo);
}

// =============================================================================
// SMOKE TEST - Basic API Verification
// =============================================================================

// Minimal round-trip test: upload pattern to image, download it back, verify match.
// This validates the basic upload/download API works end-to-end.
TEST_F(UnitTestFixture, ImageStaging_SmokeTest_RoundTrip) {
    // Setup: queue and staging pool
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16); // 64KB
    vko::StagingStream stream(queue, std::move(staging));

    // Create a small test image: 8x8 RGBA (256 bytes)
    constexpr uint32_t   width  = 8;
    constexpr uint32_t   height = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Prepare test pattern: sequential bytes 0, 1, 2, ...
    const size_t           imageBytes = width * height * 4; // RGBA = 4 bytes per pixel
    std::vector<std::byte> testPattern(imageBytes);
    for (size_t i = 0; i < imageBytes; ++i) {
        testPattern[i] = static_cast<std::byte>(i & 0xFF);
    }

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Transition image to TRANSFER_DST for upload
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Upload the test pattern
    vko::upload(stream, ctx->device, testPattern, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource, extent, format);

    // Transition image to TRANSFER_SRC for download
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download the image
    auto downloadFuture =
        vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource,
                      extent, format);

    // Submit and wait
    stream.submit();
    auto& downloadedData = downloadFuture.get(ctx->device);

    // Verify the data matches
    ASSERT_EQ(downloadedData.size(), testPattern.size());
    EXPECT_EQ(downloadedData, testPattern) << "Downloaded data does not match uploaded pattern";
}

// mipExtent is now in vko::mipExtent (formats.hpp)
using vko::mipExtent;

// Helper: Round-trip test helper (upload pattern, download, verify match)
template <class StagingStreamType, class BoundImageType, class PatternFn>
void roundTripImageTest(StagingStreamType& stream, const vko::Device& device, BoundImageType& image,
                        VkExtent3D extent, VkFormat format,
                        const VkImageSubresourceLayers& subresource, PatternFn&& patternFn) {
    using namespace vko;

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    size_t          imageBytes = imageSizeBytes(extent, subresource.layerCount, fmtInfo);

    // Generate test pattern
    std::vector<std::byte> testPattern(imageBytes);
    patternFn(testPattern);

    // Transition to TRANSFER_DST
    transitionImageLayout(stream, device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource.aspectMask,
                          subresource.mipLevel, 1, subresource.baseArrayLayer,
                          subresource.layerCount);

    // Upload
    upload(stream, device, testPattern, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource,
           extent, format);

    // Transition to TRANSFER_SRC
    transitionImageLayout(stream, device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource.aspectMask,
                          subresource.mipLevel, 1, subresource.baseArrayLayer,
                          subresource.layerCount);

    // Download
    auto downloadFuture = download(stream, device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   subresource, extent, format);

    // Submit and verify
    stream.submit();
    auto& downloadedData = downloadFuture.get(device);

    ASSERT_EQ(downloadedData.size(), testPattern.size());
    EXPECT_EQ(downloadedData, testPattern);
}

// =============================================================================
// UNIT TESTS - Pure Math/Logic (No GPU Required)
// =============================================================================
// These test the helper functions directly without needing Vulkan resources.

// -----------------------------------------------------------------------------
// byteOffsetToImageCoord() Tests
// -----------------------------------------------------------------------------
// Verifies byte offset → (x, y, z) coordinate conversion.
// Formula: address = ((z * height + y) * width + x) * bytesPerTexel
TEST_F(UnitTestFixture, ImageStaging_Math_ByteOffsetAndCoord) {
    struct Coord {
        uint32_t x, y, z;
    };

    auto byteOffsetToCoord = [](VkDeviceSize offsetBytes, Coord size, uint32_t bytesPerTexel) {
        uint32_t element = uint32_t(offsetBytes / bytesPerTexel);
        uint32_t x       = element % size.x;
        uint32_t y       = (element / size.x) % size.y;
        uint32_t z       = element / (size.x * size.y);
        return Coord{x, y, z};
    };
    auto coordToByteOffset = [](Coord c, Coord size, uint32_t bytesPerTexel) {
        uint32_t element = (c.z * size.y + c.y) * size.x + c.x;
        return VkDeviceSize(element) * bytesPerTexel;
    };
    auto expectVec = [](const Coord& v, uint32_t x, uint32_t y, uint32_t z) {
        EXPECT_EQ(v.x, x);
        EXPECT_EQ(v.y, y);
        EXPECT_EQ(v.z, z);
    };

    // Basic cases for 4x3x2 image, 4 bytes per texel
    Coord              size       = {4, 3, 2};
    constexpr uint32_t bpp        = 4;
    const VkDeviceSize rowBytes   = size.x * bpp;
    const VkDeviceSize sliceBytes = rowBytes * size.y;

    expectVec(byteOffsetToCoord(0, size, bpp), 0, 0, 0);
    expectVec(byteOffsetToCoord(8, size, bpp), 2, 0, 0);          // mid-row
    expectVec(byteOffsetToCoord(rowBytes, size, bpp), 0, 1, 0);   // row boundary
    expectVec(byteOffsetToCoord(sliceBytes, size, bpp), 0, 0, 1); // slice boundary

    VkDeviceSize lastByte = (size.x * size.y * size.z - 1) * bpp;
    expectVec(byteOffsetToCoord(lastByte, size, bpp), 3, 2, 1);

    // Verify bytesPerTexel scaling for x
    Coord small = {5, 1, 1};
    expectVec(byteOffsetToCoord(3, small, 1), 3, 0, 0);
    expectVec(byteOffsetToCoord(6, small, 2), 3, 0, 0);
    expectVec(byteOffsetToCoord(12, small, 4), 3, 0, 0);
    expectVec(byteOffsetToCoord(24, small, 8), 3, 0, 0);
    expectVec(byteOffsetToCoord(48, small, 16), 3, 0, 0);

    // Non-power-of-2 dimensions
    Coord        npot       = {100, 75, 3};
    VkDeviceSize npotOffset = (npot.x * 2 + 7) * bpp; // y=2, x=7
    expectVec(byteOffsetToCoord(npotOffset, npot, bpp), 7, 2, 0);

    // Spot-check inverse for a couple of coords
    EXPECT_EQ(coordToByteOffset({0, 0, 0}, size, bpp), 0u);
    EXPECT_EQ(coordToByteOffset({3, 2, 1}, size, bpp), lastByte);
}

// -----------------------------------------------------------------------------
// imageCoordToByteOffset() Tests
// -----------------------------------------------------------------------------
// Verifies (x, y, z) → byte offset conversion (inverse of above).
TEST_F(UnitTestFixture, ImageStaging_Math_RoundTripCoordOffset) {
    struct Coord {
        uint32_t x, y, z;
    };

    auto byteOffsetToCoord = [](VkDeviceSize offsetBytes, Coord size, uint32_t bytesPerTexel) {
        uint32_t element = uint32_t(offsetBytes / bytesPerTexel);
        uint32_t x       = element % size.x;
        uint32_t y       = (element / size.x) % size.y;
        uint32_t z       = element / (size.x * size.y);
        return Coord{x, y, z};
    };
    auto coordToByteOffset = [](Coord c, Coord size, uint32_t bytesPerTexel) {
        uint32_t element = (c.z * size.y + c.y) * size.x + c.x;
        return VkDeviceSize(element) * bytesPerTexel;
    };

    Coord              size       = {6, 5, 4};
    constexpr uint32_t bpp        = 4;
    const VkDeviceSize rowBytes   = size.x * bpp;
    const VkDeviceSize sliceBytes = rowBytes * size.y;

    std::array<VkDeviceSize, 5> offsets = {
        0u, 8u, rowBytes, sliceBytes, (size.x * size.y * size.z - 1) * bpp,
    };
    for (VkDeviceSize offset : offsets) {
        Coord c = byteOffsetToCoord(offset, size, bpp);
        EXPECT_EQ(coordToByteOffset(c, size, bpp), offset);
    }

    std::array<Coord, 6> coords = {
        Coord{0, 0, 0}, Coord{1, 0, 0}, Coord{0, 1, 0},
        Coord{0, 0, 1}, Coord{5, 4, 3}, Coord{2, 3, 1},
    };
    for (const auto& c : coords) {
        VkDeviceSize offset = coordToByteOffset(c, size, bpp);
        Coord        back   = byteOffsetToCoord(offset, size, bpp);
        EXPECT_EQ(back.x, c.x);
        EXPECT_EQ(back.y, c.y);
        EXPECT_EQ(back.z, c.z);
    }
}

// -----------------------------------------------------------------------------
// generateImageCopyRegions() Tests
// -----------------------------------------------------------------------------
// Verifies correct VkBufferImageCopy generation for arbitrary byte ranges.
TEST_F(UnitTestFixture, ImageStaging_GenerateCopyRegions_BasicRows) {
    vko::FormatInfo          fmtInfo = vko::formatInfo(VK_FORMAT_R8G8B8A8_UNORM);
    VkExtent3D               extent  = {4, 3, 1};
    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    VkDeviceSize rowBytes = extent.width * fmtInfo.blockSize;

    // Single complete row
    auto rows = vko::generateImageCopyRegions(0, rowBytes, subresource, extent, fmtInfo);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].imageOffset.x, 0);
    EXPECT_EQ(rows[0].imageOffset.y, 0);
    EXPECT_EQ(rows[0].imageExtent.width, extent.width);
    EXPECT_EQ(rows[0].imageExtent.height, 1u);
    EXPECT_EQ(rows[0].imageExtent.depth, 1u);

    // Partial row: x = 1..2 (2 pixels)
    auto partial = vko::generateImageCopyRegions(4, 12, subresource, extent, fmtInfo);
    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(partial[0].imageOffset.x, 1);
    EXPECT_EQ(partial[0].imageOffset.y, 0);
    EXPECT_EQ(partial[0].imageExtent.width, 2u);
    EXPECT_EQ(partial[0].imageExtent.height, 1u);

    // Exactly two rows should coalesce into a single region
    auto twoRows = vko::generateImageCopyRegions(0, rowBytes * 2, subresource, extent, fmtInfo);
    ASSERT_EQ(twoRows.size(), 1u);
    EXPECT_EQ(twoRows[0].imageExtent.height, 2u);
}

TEST_F(UnitTestFixture, ImageStaging_GenerateCopyRegions_ArrayAnd3D) {
    vko::FormatInfo fmtInfo = vko::formatInfo(VK_FORMAT_R8G8B8A8_UNORM);

    // Array image: baseArrayLayer should pass through, imageOffset.z must be 0
    VkExtent3D               arrayExtent = {2, 2, 1};
    VkImageSubresourceLayers arraySubresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 5,
        .layerCount     = 2,
    };
    VkDeviceSize rowBytes = arrayExtent.width * fmtInfo.blockSize;
    auto         arrayRegions =
        vko::generateImageCopyRegions(0, rowBytes, arraySubresource, arrayExtent, fmtInfo);
    ASSERT_EQ(arrayRegions.size(), 1u);
    EXPECT_EQ(arrayRegions[0].imageSubresource.baseArrayLayer, 5u);
    EXPECT_EQ(arrayRegions[0].imageOffset.z, 0);

    // 3D image: imageOffset.z should move with slice
    VkExtent3D               depthExtent = {2, 2, 2};
    VkImageSubresourceLayers depthSubresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    VkDeviceSize sliceBytes   = depthExtent.width * depthExtent.height * fmtInfo.blockSize;
    auto         depthRegions = vko::generateImageCopyRegions(sliceBytes, sliceBytes + rowBytes,
                                                              depthSubresource, depthExtent, fmtInfo);
    ASSERT_EQ(depthRegions.size(), 1u);
    EXPECT_EQ(depthRegions[0].imageOffset.z, 1);
    EXPECT_EQ(depthRegions[0].imageSubresource.layerCount, 1u);
}

// Note: copyWidth == 0 and empty range paths are unreachable with non-zero staging buffers
// and the beginBytes < endBytes contract of generateImageCopyRegions.

// -----------------------------------------------------------------------------
// makeImageCopyParams() Tests
// -----------------------------------------------------------------------------
// Verifies parameter validation and computation.

// Note: All valid VkFormat enums are supported (including compressed formats like BC7).
// formatInfo() only throws for invalid enum values cast from integers.

TEST_F(UnitTestFixture, ImageStaging_ImageCopyParams_SliceBytes) {
    vko::FormatInfo fmtInfo    = vko::formatInfo(VK_FORMAT_R8G8B8A8_UNORM);
    VkExtent3D      extent     = {7, 9, 1};
    VkDeviceSize    sliceBytes = extent.width * extent.height * fmtInfo.blockSize;
    EXPECT_EQ(sliceBytes, VkDeviceSize(7u * 9u * 4u));

    VkDeviceSize totalBytes = sliceBytes * 3;
    EXPECT_EQ(totalBytes, VkDeviceSize(7u * 9u * 4u * 3u));
}

// -----------------------------------------------------------------------------
// imageSizeBytes() Tests (from formats.hpp)
// -----------------------------------------------------------------------------
TEST_F(UnitTestFixture, ImageStaging_ImageSizeBytes_Basic) {
    VkExtent3D extent = {5, 3, 2};

    vko::FormatInfo r8 = vko::formatInfo(VK_FORMAT_R8_UNORM);
    EXPECT_EQ(vko::imageSizeBytes(extent, 1, r8), VkDeviceSize(5u * 3u * 2u));

    vko::FormatInfo rgba8 = vko::formatInfo(VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_EQ(vko::imageSizeBytes(extent, 1, rgba8), VkDeviceSize(5u * 3u * 2u * 4u));

    // FormatInfo vs VkFormat should match
    EXPECT_EQ(vko::imageSizeBytes(extent, 1, rgba8),
              vko::imageSizeBytes(extent, 1, VK_FORMAT_R8G8B8A8_UNORM));

    // Compile-time traits match runtime info
    EXPECT_EQ(vko::format_traits<VK_FORMAT_R8G8B8A8_UNORM>::blockSize, rgba8.blockSize);
    EXPECT_EQ(vko::format_traits<VK_FORMAT_R8G8B8A8_UNORM>::blockExtent.width,
              rgba8.blockExtent.width);
}

// =============================================================================
// Image Staging Download Tests
// =============================================================================
//
// These tests verify that downloadImage() and downloadForEach() correctly handle
// chunked transfers for various image types. The staging allocator may return
// buffers smaller than the total image size, requiring multiple copy operations
// that may start/end mid-row and span multiple layers.
//
// Key scenarios to test:
// - Arbitrary byte-range chunking (start mid-row, end mid-row, span layers)
// - Correct VkBufferImageCopy region generation
// - Data integrity across chunk boundaries
// =============================================================================

// -----------------------------------------------------------------------------
// Format Coverage - Various Bytes Per Pixel
// -----------------------------------------------------------------------------
// Test representative formats with different pixel sizes to ensure byte offset
// calculations are correct.

// Test VK_FORMAT_R8_UNORM (1 byte per pixel)
// Small image that fits in one chunk
// Verify data integrity
TEST_F(UnitTestFixture, ImageStaging_Format_R8_UNORM_Small) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16;
    constexpr VkFormat   format = VK_FORMAT_R8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test VK_FORMAT_R8G8_UNORM (2 bytes per pixel)
// Image requiring multiple chunks
// Verify chunk boundaries don't corrupt data
TEST_F(UnitTestFixture, ImageStaging_Format_R8G8_UNORM_Chunked) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4,
        /*poolSize=*/256); // Small pool to force chunking
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 64, height = 32; // 64*32*2 = 4096 bytes, needs multiple chunks
    constexpr VkFormat   format = VK_FORMAT_R8G8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test VK_FORMAT_R8G8B8A8_UNORM (4 bytes per pixel)
// Most common format, baseline test
// Multiple chunk sizes
TEST_F(UnitTestFixture, ImageStaging_Format_R8G8B8A8_UNORM_MultipleChunks) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/512); // Force chunking
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 128, height = 64; // 128*64*4 = 32768 bytes, many chunks
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test VK_FORMAT_R16G16B16A16_SFLOAT (8 bytes per pixel)
// Larger pixel size, fewer pixels per row
// Test alignment edge cases
TEST_F(UnitTestFixture, ImageStaging_Format_R16G16B16A16_SFLOAT) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 13, height = 11; // Prime dimensions
    constexpr VkFormat   format = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // 8 bytes per pixel (4 x 16-bit floats)
            constexpr uint32_t width = 13, height = 11;
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    size_t offset = (y * width + x) * 8;
                    // Just fill with recognizable byte pattern
                    for (size_t i = 0; i < 8; ++i) {
                        data[offset + i] = static_cast<std::byte>((x + y + i) & 0xFF);
                    }
                }
            }
        });
}

// Test VK_FORMAT_R32G32B32A32_SFLOAT (16 bytes per pixel)
// Very large pixels
// Verify row calculations with large bytesPerTexel
TEST_F(UnitTestFixture, ImageStaging_Format_R32G32B32A32_SFLOAT) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 7, height = 5; // Small prime dimensions
    constexpr VkFormat   format = VK_FORMAT_R32G32B32A32_SFLOAT;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // 16 bytes per pixel (4 x 32-bit floats)
            constexpr uint32_t width = 7, height = 5;
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    size_t offset = (y * width + x) * 16;
                    // Fill with recognizable pattern
                    for (size_t i = 0; i < 16; ++i) {
                        data[offset + i] = static_cast<std::byte>((x * 16 + y + i) & 0xFF);
                    }
                }
            }
        });
}

// -----------------------------------------------------------------------------
// Compressed Format Coverage (Block-Based)
// -----------------------------------------------------------------------------
// Compressed formats work in blocks (e.g., 4x4 pixels per block). Our code
// operates in "element space" (blocks for compressed, pixels for uncompressed),
// so compressed formats are fully supported.

// Test VK_FORMAT_BC1_RGB_UNORM_BLOCK (8 bytes per 4x4 block)
// Block-compressed format with smaller block size than BC7
// Verify block-aligned chunking works correctly
TEST_F(UnitTestFixture, ImageStaging_Compressed_BC1) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    // 20x16 pixels = 5x4 blocks, each block is 8 bytes = 160 bytes total
    constexpr uint32_t   width = 20, height = 16;
    constexpr VkFormat   format = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // 5x4 blocks * 8 bytes = 160 bytes
            for (size_t block = 0; block < data.size() / 8; ++block) {
                for (size_t i = 0; i < 8; ++i) {
                    data[block * 8 + i] = static_cast<std::byte>((block * 8 + i) & 0xFF);
                }
            }
        });
}

// Test BC1 with chunking (small pool forces multiple chunks)
TEST_F(UnitTestFixture, ImageStaging_Compressed_BC1_Chunked) {
    // Use small pool to force chunking: 32 bytes = 4 blocks
    constexpr size_t poolSize = 32;

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    // 32x32 pixels = 8x8 blocks = 64 blocks * 8 bytes = 512 bytes
    // With 32-byte pool, need 16 chunks
    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Fill each block with unique pattern
                           for (size_t block = 0; block < data.size() / 8; ++block) {
                               for (size_t i = 0; i < 8; ++i) {
                                   data[block * 8 + i] = static_cast<std::byte>((block ^ i) & 0xFF);
                               }
                           }
                       });
}

// Test VK_FORMAT_BC7_UNORM_BLOCK (16 bytes per 4x4 block)
// Higher quality BC format
// Test with dimensions that are multiples of block size
TEST_F(UnitTestFixture, ImageStaging_Compressed_BC7) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    // 16x12 pixels = 4x3 blocks, each block is 16 bytes = 192 bytes total
    constexpr uint32_t   width = 16, height = 12;
    constexpr VkFormat   format = VK_FORMAT_BC7_UNORM_BLOCK;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // 4x3 blocks * 16 bytes = 192 bytes
            // Fill with block-unique patterns
            for (size_t block = 0; block < data.size() / 16; ++block) {
                for (size_t i = 0; i < 16; ++i) {
                    data[block * 16 + i] = static_cast<std::byte>((block * 16 + i) & 0xFF);
                }
            }
        });
}

// Test VK_FORMAT_BC7 with chunking (small pool forces multiple chunks)
TEST_F(UnitTestFixture, ImageStaging_Compressed_BC7_Chunked) {
    // Use small pool to force chunking
    constexpr size_t poolSize = 64; // 4 blocks per chunk

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    // 32x32 pixels = 8x8 blocks = 64 blocks * 16 bytes = 1024 bytes
    // With 64-byte pool, need ~16 chunks
    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_BC7_UNORM_BLOCK;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // Fill each block with unique pattern
            for (size_t block = 0; block < data.size() / 16; ++block) {
                for (size_t i = 0; i < 16; ++i) {
                    data[block * 16 + i] = static_cast<std::byte>((block ^ i) & 0xFF);
                }
            }
        });
}

// Test VK_FORMAT_ASTC_4x4_UNORM_BLOCK (16 bytes per 4x4 block)
// Mobile-friendly compressed format (may not be supported on desktop GPUs)
TEST_F(UnitTestFixture, ImageStaging_Compressed_ASTC_4x4) {
    constexpr VkFormat format = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;

    // Check if format is supported
    VkFormatProperties formatProps;
    ctx->instance.vkGetPhysicalDeviceFormatProperties(ctx->physicalDevice, format, &formatProps);
    if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ||
        !(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
        GTEST_SKIP() << "VK_FORMAT_ASTC_4x4_UNORM_BLOCK not supported for transfer on this GPU";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    // 16x16 pixels = 4x4 blocks, each block is 16 bytes = 256 bytes total
    constexpr uint32_t   width = 16, height = 16;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // 4x4 blocks * 16 bytes = 256 bytes
            for (size_t block = 0; block < data.size() / 16; ++block) {
                for (size_t i = 0; i < 16; ++i) {
                    data[block * 16 + i] = static_cast<std::byte>((block * 16 + i) & 0xFF);
                }
            }
        });
}

// Test VK_FORMAT_ASTC_8x8_UNORM_BLOCK (16 bytes per 8x8 block)
// Larger block size, tests different blockExtent (may not be supported on desktop)
TEST_F(UnitTestFixture, ImageStaging_Compressed_ASTC_8x8) {
    constexpr VkFormat format = VK_FORMAT_ASTC_8x8_UNORM_BLOCK;

    // Check if format is supported
    VkFormatProperties formatProps;
    ctx->instance.vkGetPhysicalDeviceFormatProperties(ctx->physicalDevice, format, &formatProps);
    if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ||
        !(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
        GTEST_SKIP() << "VK_FORMAT_ASTC_8x8_UNORM_BLOCK not supported for transfer on this GPU";
    }

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    // 32x32 pixels = 4x4 blocks (since each block is 8x8), 16 blocks * 16 bytes = 256 bytes
    constexpr uint32_t   width = 32, height = 32;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            // Fill each block with unique pattern
            for (size_t block = 0; block < data.size() / 16; ++block) {
                for (size_t i = 0; i < 16; ++i) {
                    data[block * 16 + i] = static_cast<std::byte>((block ^ i) & 0xFF);
                }
            }
        });
}

// -----------------------------------------------------------------------------
// 2D Array Texture Tests
// -----------------------------------------------------------------------------
// Array textures have multiple layers. Buffer layout is layer-major:
// all rows of layer 0, then all rows of layer 1, etc.

// Test 2D array with 2 layers, small image (fits in one chunk)
// Verify both layers downloaded correctly
// Format: VK_FORMAT_R8G8B8A8_UNORM
TEST_F(UnitTestFixture, ImageStaging_2DArray_TwoLayers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, /*arrayLayers=*/2);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 2,
    };

    // Each layer gets a unique pattern (layer index in each pixel)
    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           size_t layerSize = data.size() / 2;
                           for (size_t layer = 0; layer < 2; ++layer) {
                               for (size_t i = 0; i < layerSize; ++i) {
                                   data[layer * layerSize + i] = static_cast<std::byte>(layer);
                               }
                           }
                       });
}

// Test 2D array with 4 layers, image requires chunking within layers
// Force chunk boundary mid-layer
// Verify layer data doesn't get mixed
TEST_F(UnitTestFixture, ImageStaging_2DArray_ChunkedWithinLayers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Pool size smaller than one layer to force chunking within layers
    // Layer size = 32*32*4 = 4096 bytes, pool = 1024 bytes
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/1024);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, /*arrayLayers=*/4);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 4,
    };

    // Each layer gets unique pattern (layer index repeated)
    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr size_t layerSize = 32 * 32 * 4;
            for (size_t layer = 0; layer < 4; ++layer) {
                for (size_t i = 0; i < layerSize; ++i) {
                    data[layer * layerSize + i] = static_cast<std::byte>(layer * 50 + (i % 50));
                }
            }
        });
}

// Test 2D array with 8 layers, chunk spans multiple layers
// Chunk starts in layer 2, ends in layer 5
// Verify correct VkBufferImageCopy regions generated
TEST_F(UnitTestFixture, ImageStaging_2DArray_ChunkSpansLayers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Layer size = 16*16*4 = 1024 bytes, pool = 3000 bytes (~3 layers per chunk)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/3000);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    vko::BoundImage image(
        ctx->device,
        VkImageCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .imageType   = VK_IMAGE_TYPE_2D,
            .format      = format,
            .extent      = extent,
            .mipLevels   = 1,
            .arrayLayers = 8,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = VK_IMAGE_TILING_OPTIMAL,
            .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        },
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 8,
    };

    // Each layer gets unique pattern
    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr size_t layerSize = 16 * 16 * 4;
            for (size_t layer = 0; layer < 8; ++layer) {
                for (size_t i = 0; i < layerSize; ++i) {
                    data[layer * layerSize + i] = static_cast<std::byte>(layer * 30 + (i % 30));
                }
            }
        });
}

// Test 2D array with many small layers (16+ layers, tiny images)
// Many layer transitions in one download
// Stress test region generation
TEST_F(UnitTestFixture, ImageStaging_2DArray_ManySmallLayers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Pool size = 3 layers to force multiple cross-layer chunks
    constexpr size_t layerBytes = 4 * 4 * 4; // 64 bytes per layer
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/layerBytes * 3);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 4, height = 4, layers = 17; // Prime number of layers
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 4, height = 4, layers = 17;
            constexpr size_t   layerSize = width * height * 4;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t offset    = layer * layerSize + (y * width + x) * 4;
                        data[offset + 0] = static_cast<std::byte>(layer);
                        data[offset + 1] = static_cast<std::byte>(y);
                        data[offset + 2] = static_cast<std::byte>(x);
                        data[offset + 3] = static_cast<std::byte>((layer + y + x) & 0xFF);
                    }
                }
            }
        });
}

// Test 2D array - chunk crosses multiple layer boundaries
// Pool size chosen so each chunk spans ~1.7 layers
// Most complex case: partial row → complete layers → partial row
TEST_F(UnitTestFixture, ImageStaging_2DArray_ChunkCrossesMultipleLayers) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // 8x8 = 256 bytes per layer, pool = 440 bytes (~1.7 layers)
    constexpr size_t poolSize = 440;
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, layers = 5;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 8, height = 8, layers = 5;
            constexpr size_t   layerSize = width * height * 4;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t offset = layer * layerSize + (y * width + x) * 4;
                        // Use linear offset to help debug any ordering issues
                        uint32_t linear  = static_cast<uint32_t>(offset / 4);
                        data[offset + 0] = static_cast<std::byte>(linear & 0xFF);
                        data[offset + 1] = static_cast<std::byte>((linear >> 8) & 0xFF);
                        data[offset + 2] = static_cast<std::byte>(layer);
                        data[offset + 3] = static_cast<std::byte>(0xFF);
                    }
                }
            }
        });
}

// -----------------------------------------------------------------------------
// 3D Texture Tests
// -----------------------------------------------------------------------------
// 3D textures have depth slices. Similar to arrays but imageOffset.z is used
// instead of baseArrayLayer.

// Test 3D texture with depth=2, small (fits in one chunk)
// Verify both depth slices downloaded correctly
// Format: VK_FORMAT_R8G8B8A8_UNORM
TEST_F(UnitTestFixture, ImageStaging_3D_Depth2) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16, depth = 2;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Each depth slice gets a unique pattern (depth index in each pixel)
    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr size_t depth     = 2;
                           size_t           sliceSize = data.size() / depth;
                           for (size_t z = 0; z < depth; ++z) {
                               for (size_t i = 0; i < sliceSize; ++i) {
                                   data[z * sliceSize + i] = static_cast<std::byte>(z);
                               }
                           }
                       });
}

// Test 3D texture with depth=8, requires chunking
// Chunk boundary within depth slices
// Verify Z coordinate handling in copy regions
TEST_F(UnitTestFixture, ImageStaging_3D_ChunkedWithinSlices) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Slice size = 16*16*4 = 1024 bytes, pool = 512 bytes (partial slices)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/512);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16, depth = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Each depth slice gets unique pattern
    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr size_t sliceSize = 16 * 16 * 4;
            constexpr size_t depth     = 8;
            for (size_t z = 0; z < depth; ++z) {
                for (size_t i = 0; i < sliceSize; ++i) {
                    data[z * sliceSize + i] = static_cast<std::byte>(z * 30 + (i % 30));
                }
            }
        });
}

// Test 3D texture with depth=16, chunk spans multiple Z slices
// Similar to array layer spanning test
// Verify imageOffset.z is correct in each region
TEST_F(UnitTestFixture, ImageStaging_3D_ChunkSpansSlices) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Slice size = 8*8*4 = 256 bytes, pool = 1000 bytes (~4 slices per chunk)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/1000);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, depth = 16;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Each depth slice gets unique pattern
    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr size_t sliceSize = 8 * 8 * 4;
            constexpr size_t depth     = 16;
            for (size_t z = 0; z < depth; ++z) {
                for (size_t i = 0; i < sliceSize; ++i) {
                    data[z * sliceSize + i] = static_cast<std::byte>(z * 15 + (i % 15));
                }
            }
        });
}

// Test 3D texture with large depth (64+), tiny XY dimensions
// Many Z transitions per chunk
// Verify no off-by-one errors
TEST_F(UnitTestFixture, ImageStaging_3D_LargeDepth64) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Slice = 4*4*4 = 64 bytes, pool = 200 bytes (~3 slices per chunk)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/32,
                                                               /*poolSize=*/200);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 4, height = 4, depth = 67; // Prime depth > 64
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 4, height = 4, depth = 67;
                           constexpr size_t   sliceSize = width * height * 4;
                           for (uint32_t z = 0; z < depth; ++z) {
                               for (size_t i = 0; i < sliceSize; ++i) {
                                   // Z in R channel for easy debugging
                                   data[z * sliceSize + i] = static_cast<std::byte>((z + i) & 0xFF);
                               }
                           }
                       });
}

// -----------------------------------------------------------------------------
// Mipmap Level Tests
// -----------------------------------------------------------------------------
// Each mip level is downloaded separately (VkImageSubresourceLayers.mipLevel).
// Test that extent matches the mip level's dimensions.

// Test mip level 0 (base level) of a mipmapped image
// Full resolution, standard download
// Verify mipLevel=0 in subresource
TEST_F(UnitTestFixture, ImageStaging_MipLevel_0) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   baseWidth = 64, baseHeight = 64, mipLevels = 4;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D baseExtent = {baseWidth, baseHeight, 1};
    constexpr uint32_t   mipLevel   = 0;
    VkExtent3D           mipExt     = mipExtent(baseExtent, mipLevel);

    auto image =
        makeTestImage(*ctx, baseExtent, format, /*arrayLayers=*/1, /*mipLevels=*/mipLevels);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = mipLevel,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, mipExt, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Fill with pattern based on mip level (0xFF for base level)
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(0xFF);
                           }
                       });
}

// Test mip level 1 (half resolution)
// Extent should be half of base
// Verify correct byte count
TEST_F(UnitTestFixture, ImageStaging_MipLevel_1) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   baseWidth = 64, baseHeight = 64, mipLevels = 4;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D baseExtent = {baseWidth, baseHeight, 1};
    constexpr uint32_t   mipLevel   = 1;
    VkExtent3D           mipExt     = mipExtent(baseExtent, mipLevel);

    auto image =
        makeTestImage(*ctx, baseExtent, format, /*arrayLayers=*/1, /*mipLevels=*/mipLevels);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = mipLevel,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, mipExt, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t mipLevel = 1;
                           // Fill with pattern based on mip level
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(mipLevel);
                           }
                       });
}

// Test mip level 2 (quarter resolution)
// Even smaller extent
// Verify extent calculation is correct
TEST_F(UnitTestFixture, ImageStaging_MipLevel_2) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   baseWidth = 64, baseHeight = 64, mipLevels = 4;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D baseExtent = {baseWidth, baseHeight, 1};
    constexpr uint32_t   mipLevel   = 2;
    VkExtent3D           mipExt     = mipExtent(baseExtent, mipLevel);

    auto image =
        makeTestImage(*ctx, baseExtent, format, /*arrayLayers=*/1, /*mipLevels=*/mipLevels);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = mipLevel,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, mipExt, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t mipLevel = 2;
                           // Fill with pattern based on mip level
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(mipLevel);
                           }
                       });
}

// Test smallest mip level (1x1 or near-1x1)
// Edge case: very small image
// May be smaller than minimum staging buffer
TEST_F(UnitTestFixture, ImageStaging_MipLevel_Smallest) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t baseWidth = 64, baseHeight = 64,
                       mipLevels    = 7; // 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D baseExtent = {baseWidth, baseHeight, 1};
    constexpr uint32_t   mipLevel   = mipLevels - 1; // Last mip = 1x1
    VkExtent3D           mipExt     = mipExtent(baseExtent, mipLevel);

    auto image =
        makeTestImage(*ctx, baseExtent, format, /*arrayLayers=*/1, /*mipLevels=*/mipLevels);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = mipLevel,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, mipExt, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t mipLevel = 6; // mipLevels - 1 = 7 - 1 = 6
                           // Fill with pattern based on mip level
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(mipLevel);
                           }
                       });
}

// Test downloading all mip levels sequentially
// Loop through mip 0 to max
// Verify each level has correct data
TEST_F(UnitTestFixture, ImageStaging_MipLevels_AllSequential) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   baseWidth = 32, baseHeight = 32, mipLevels = 6;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D baseExtent = {baseWidth, baseHeight, 1};

    auto image =
        makeTestImage(*ctx, baseExtent, format, /*arrayLayers=*/1, /*mipLevels=*/mipLevels);

    // Upload all mip levels first
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                          /*baseMipLevel=*/0, /*levelCount=*/mipLevels, /*baseArrayLayer=*/0,
                          /*layerCount=*/1);

    // Upload each mip level
    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        VkExtent3D               mipExt = mipExtent(baseExtent, mip);
        VkImageSubresourceLayers subresource{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel       = mip,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        };
        using namespace vko;
        vko::FormatInfo        fmtInfo    = vko::formatInfo(format);
        size_t                 imageBytes = imageSizeBytes(mipExt, 1, fmtInfo);
        std::vector<std::byte> pattern(imageBytes);
        for (size_t i = 0; i < imageBytes; ++i) {
            pattern[i] = static_cast<std::byte>(mip);
        }
        vko::upload(stream, ctx->device, pattern, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    subresource, mipExt, format);
    }

    // Transition to TRANSFER_SRC for all mips
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                          /*baseMipLevel=*/0, /*levelCount=*/mipLevels, /*baseArrayLayer=*/0,
                          /*layerCount=*/1);

    // Download each mip level and verify
    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        VkExtent3D               mipExt = mipExtent(baseExtent, mip);
        VkImageSubresourceLayers subresource{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel       = mip,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        };
        auto downloadFuture =
            vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          subresource, mipExt, format);
        stream.submit();
        auto& downloadedData = downloadFuture.get(ctx->device);

        // Verify all bytes match the mip level
        for (size_t i = 0; i < downloadedData.size(); ++i) {
            EXPECT_EQ(downloadedData[i], static_cast<std::byte>(mip))
                << "Mip level " << mip << " byte " << i << " mismatch";
        }
    }
}

// -----------------------------------------------------------------------------
// Forced Chunking Tests (Small Staging Buffers)
// -----------------------------------------------------------------------------
// Use RecyclingStagingPool with small pool sizes to force aggressive chunking.

// Test with pool size = 1 row of pixels
// Every row is a separate chunk
// Maximum number of copy regions
TEST_F(UnitTestFixture, ImageStaging_Chunking_OneRowPerChunk) {
    constexpr uint32_t   width = 16, height = 8;
    constexpr VkFormat   format   = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent   = {width, height, 1};
    constexpr size_t     rowBytes = width * 4; // 64 bytes per row

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Pool size = exactly 1 row
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/rowBytes);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 16, height = 8;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset = (y * width + x) * 4;
                                   // Row-unique pattern
                                   data[offset + 0] = static_cast<std::byte>(y);
                                   data[offset + 1] = static_cast<std::byte>(x);
                                   data[offset + 2] = static_cast<std::byte>(y ^ x);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test with pool size = 0.5 rows of pixels
// Partial rows in each chunk
// Tests the "copy width < image width" path
TEST_F(UnitTestFixture, ImageStaging_Chunking_HalfRowPerChunk) {
    constexpr uint32_t   width = 16, height = 8;
    constexpr VkFormat   format   = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent   = {width, height, 1};
    constexpr size_t     rowBytes = width * 4;    // 64 bytes per row
    constexpr size_t     poolSize = rowBytes / 2; // 32 bytes = half row

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 16, height = 8;
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    size_t offset    = (y * width + x) * 4;
                    data[offset + 0] = static_cast<std::byte>(y);
                    data[offset + 1] = static_cast<std::byte>(x);
                    data[offset + 2] = static_cast<std::byte>((y * width + x) & 0xFF);
                    data[offset + 3] = static_cast<std::byte>(0xFF);
                }
            }
        });
}

// Test with pool size = 1.5 rows of pixels
// Mix of complete and partial rows per chunk
// Tests varying region counts
TEST_F(UnitTestFixture, ImageStaging_Chunking_OneAndHalfRowsPerChunk) {
    constexpr uint32_t   width = 16, height = 8;
    constexpr VkFormat   format   = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent   = {width, height, 1};
    constexpr size_t     rowBytes = width * 4;        // 64 bytes per row
    constexpr size_t     poolSize = rowBytes * 3 / 2; // 96 bytes = 1.5 rows

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 16, height = 8;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset    = (y * width + x) * 4;
                                   data[offset + 0] = static_cast<std::byte>(y);
                                   data[offset + 1] = static_cast<std::byte>(x);
                                   data[offset + 2] = static_cast<std::byte>((y + x) & 0xFF);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test with pool size = 1 layer (for 2D array)
// Clean layer boundaries
// Should be simpler region generation
TEST_F(UnitTestFixture, ImageStaging_Chunking_OneLayerPerChunk) {
    constexpr uint32_t   width = 8, height = 8, layers = 4;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent     = {width, height, 1};
    constexpr size_t     layerBytes = width * height * 4; // 256 bytes per layer

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/layerBytes);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 8, height = 8, layers = 4;
                           for (uint32_t layer = 0; layer < layers; ++layer) {
                               for (uint32_t y = 0; y < height; ++y) {
                                   for (uint32_t x = 0; x < width; ++x) {
                                       size_t offset    = ((layer * height + y) * width + x) * 4;
                                       data[offset + 0] = static_cast<std::byte>(layer);
                                       data[offset + 1] = static_cast<std::byte>(y);
                                       data[offset + 2] = static_cast<std::byte>(x);
                                       data[offset + 3] = static_cast<std::byte>(0xFF);
                                   }
                               }
                           }
                       });
}

// Test with pool size = 1.5 layers
// Chunk crosses layer boundary
// Complex region generation
TEST_F(UnitTestFixture, ImageStaging_Chunking_OneAndHalfLayersPerChunk) {
    constexpr uint32_t   width = 8, height = 8, layers = 4;
    constexpr VkFormat   format     = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent     = {width, height, 1};
    constexpr size_t     layerBytes = width * height * 4; // 256 bytes per layer
    constexpr size_t     poolSize   = layerBytes * 3 / 2; // 384 bytes = 1.5 layers

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/poolSize);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 8, height = 8, layers = 4;
                           for (uint32_t layer = 0; layer < layers; ++layer) {
                               for (uint32_t y = 0; y < height; ++y) {
                                   for (uint32_t x = 0; x < width; ++x) {
                                       size_t offset    = ((layer * height + y) * width + x) * 4;
                                       data[offset + 0] = static_cast<std::byte>(layer);
                                       data[offset + 1] = static_cast<std::byte>(y);
                                       data[offset + 2] = static_cast<std::byte>(x);
                                       data[offset + 3] = static_cast<std::byte>(0xFF);
                                   }
                               }
                           }
                       });
}

// -----------------------------------------------------------------------------
// Data Integrity Tests
// -----------------------------------------------------------------------------
// Write known patterns to images, download, verify patterns match.

// Test with sequential byte values
// Fill image with values 0, 1, 2, ... (mod 256)
// Verify downloaded data matches exactly - catches any byte swapping/offset bugs
TEST_F(UnitTestFixture, ImageStaging_DataIntegrity_Sequential) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use small pool to force chunking
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/256);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 16;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Fill with sequential bytes
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test with row-unique patterns
// Each row has distinct pattern (row number in R channel)
// Helps identify off-by-one row errors
TEST_F(UnitTestFixture, ImageStaging_DataIntegrity_RowUnique) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging =
        vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                    /*minPools=*/2, /*maxPools=*/8,
                                                    /*poolSize=*/128); // ~2 rows per chunk
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 16, height = 32;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset = (y * width + x) * 4;
                                   // R = row number, G = column, B = checksum, A = 0xFF
                                   data[offset + 0] = static_cast<std::byte>(y);
                                   data[offset + 1] = static_cast<std::byte>(x);
                                   data[offset + 2] = static_cast<std::byte>(y ^ x);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test with layer-unique patterns (for 2D arrays)
// Each layer has distinct pattern (layer number in R channel)
// Helps identify layer mixing bugs
TEST_F(UnitTestFixture, ImageStaging_DataIntegrity_LayerUnique) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Pool size = 1.5 layers to force cross-layer chunking
    constexpr size_t layerBytes = 8 * 8 * 4; // 256 bytes
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/layerBytes * 3 / 2);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, layers = 6;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 8, height = 8, layers = 6;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t offset = ((layer * height + y) * width + x) * 4;
                        // R = layer, G = row, B = col, A = checksum
                        data[offset + 0] = static_cast<std::byte>(layer);
                        data[offset + 1] = static_cast<std::byte>(y);
                        data[offset + 2] = static_cast<std::byte>(x);
                        data[offset + 3] = static_cast<std::byte>((layer + y + x) & 0xFF);
                    }
                }
            }
        });
}

// Test with slice-unique patterns (for 3D textures)
// Each Z slice has distinct pattern
// Helps identify Z-slice mixing bugs
TEST_F(UnitTestFixture, ImageStaging_DataIntegrity_SliceUnique3D) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Pool size = 1.5 slices to force cross-slice chunking
    constexpr size_t sliceBytes = 8 * 8 * 4; // 256 bytes
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/sliceBytes * 3 / 2);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, depth = 6;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 8, height = 8, depth = 6;
            for (uint32_t z = 0; z < depth; ++z) {
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t offset = ((z * height + y) * width + x) * 4;
                        // R = Z-slice, G = row, B = col, A = checksum
                        data[offset + 0] = static_cast<std::byte>(z);
                        data[offset + 1] = static_cast<std::byte>(y);
                        data[offset + 2] = static_cast<std::byte>(x);
                        data[offset + 3] = static_cast<std::byte>((z * 100 + y * 10 + x) & 0xFF);
                    }
                }
            }
        });
}

// -----------------------------------------------------------------------------
// Edge Cases
// -----------------------------------------------------------------------------

// Test 1x1 image
// Minimum size, single pixel
// Verify no division by zero or similar issues
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_1x1) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {1, 1, 1}; // Single pixel

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Single pixel: RGBA = 0xDEADBEEF
                           data[0] = static_cast<std::byte>(0xDE);
                           data[1] = static_cast<std::byte>(0xAD);
                           data[2] = static_cast<std::byte>(0xBE);
                           data[3] = static_cast<std::byte>(0xEF);
                       });
}

// Test 1xN image (single column)
// Each "row" is just 1 pixel
// Tests row stride edge case
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_1xN) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   height = 31; // Prime height
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {1, height, 1}; // Single column

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t height = 31;
                           for (uint32_t y = 0; y < height; ++y) {
                               size_t offset    = y * 4;
                               data[offset + 0] = static_cast<std::byte>(y);
                               data[offset + 1] = static_cast<std::byte>(y * 2);
                               data[offset + 2] = static_cast<std::byte>(y * 3);
                               data[offset + 3] = static_cast<std::byte>(0xFF);
                           }
                       });
}

// Test Nx1 image (single row)
// Single row, multiple pixels
// Entire image is one "row" for chunking
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_Nx1) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width  = 37; // Prime width
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, 1, 1}; // Single row

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 37;
                           for (uint32_t x = 0; x < width; ++x) {
                               size_t offset    = x * 4;
                               data[offset + 0] = static_cast<std::byte>(x);
                               data[offset + 1] = static_cast<std::byte>(x * 2);
                               data[offset + 2] = static_cast<std::byte>(x * 3);
                               data[offset + 3] = static_cast<std::byte>(0xFF);
                           }
                       });
}

// Test single layer of a larger array
// baseArrayLayer != 0, layerCount = 1
// Verify correct layer offset
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_SingleLayerFromArray) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, totalLayers = 4;
    constexpr uint32_t   targetLayer = 2; // Middle layer
    constexpr VkFormat   format      = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent      = {width, height, 1};

    vko::BoundImage image(
        ctx->device,
        VkImageCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .imageType   = VK_IMAGE_TYPE_2D,
            .format      = format,
            .extent      = extent,
            .mipLevels   = 1,
            .arrayLayers = totalLayers,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = VK_IMAGE_TILING_OPTIMAL,
            .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        },
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    // Only transfer layer 2, not all layers
    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = targetLayer,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 8, height = 8;
                           // Fill with layer-specific pattern
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset    = (y * width + x) * 4;
                                   data[offset + 0] = static_cast<std::byte>(2); // targetLayer
                                   data[offset + 1] = static_cast<std::byte>(x);
                                   data[offset + 2] = static_cast<std::byte>(y);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test non-power-of-2 dimensions
// e.g., 100x75 pixels
// Verify no assumptions about power-of-2
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_NonPowerOf2) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 100, height = 75;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 100, height = 75;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset    = (y * width + x) * 4;
                                   data[offset + 0] = static_cast<std::byte>(x);
                                   data[offset + 1] = static_cast<std::byte>(y);
                                   data[offset + 2] = static_cast<std::byte>((x + y) & 0xFF);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test very wide image (4096x4)
// Row is much larger than typical chunk
// Multiple chunks per row
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_VeryWide) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use small pool to force chunking within rows
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/4096); // 4KB pools
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 4096, height = 4;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};
    // 4096 * 4 * 4 = 65536 bytes, but pool is 4096 bytes
    // Row = 16384 bytes, so each row needs 4 chunks

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 4096, height = 4;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset    = (y * width + x) * 4;
                                   data[offset + 0] = static_cast<std::byte>(x & 0xFF);
                                   data[offset + 1] = static_cast<std::byte>((x >> 8) & 0xFF);
                                   data[offset + 2] = static_cast<std::byte>(y);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// Test very tall image (4x4096)
// Many rows, each row is small
// Many regions but each is simple
TEST_F(UnitTestFixture, ImageStaging_EdgeCase_VeryTall) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use small pool to force many chunks
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/4096); // 4KB pools
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 4, height = 4096;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};
    // 4 * 4096 * 4 = 65536 bytes, pool is 4096 bytes
    // Row = 16 bytes, so many rows per chunk

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 4, height = 4096;
                           for (uint32_t y = 0; y < height; ++y) {
                               for (uint32_t x = 0; x < width; ++x) {
                                   size_t offset    = (y * width + x) * 4;
                                   data[offset + 0] = static_cast<std::byte>(x);
                                   data[offset + 1] = static_cast<std::byte>(y & 0xFF);
                                   data[offset + 2] = static_cast<std::byte>((y >> 8) & 0xFF);
                                   data[offset + 3] = static_cast<std::byte>(0xFF);
                               }
                           }
                       });
}

// -----------------------------------------------------------------------------
// downloadImage() vs downloadForEach() Consistency
// -----------------------------------------------------------------------------

// Test download() vs downloadForEach() consistency
// Download same image with both methods, compare results
// downloadImage() returns vector
// downloadForEach() accumulates in callback
// Results should be identical
TEST_F(UnitTestFixture, ImageStaging_DownloadConsistency) {
    constexpr uint32_t   width = 17, height = 13; // Prime dimensions
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use small pool to force chunking
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/256);
    vko::StagingStream stream(queue, std::move(staging));

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    VkDeviceSize    imageBytes = vko::imageSizeBytes(extent, subresource.layerCount, fmtInfo);

    // Upload test data
    std::vector<std::byte> uploadData(imageBytes);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t offset          = (y * width + x) * 4;
            uploadData[offset + 0] = static_cast<std::byte>(x);
            uploadData[offset + 1] = static_cast<std::byte>(y);
            uploadData[offset + 2] = static_cast<std::byte>((x + y) & 0xFF);
            uploadData[offset + 3] = static_cast<std::byte>(0xFF);
        }
    }

    // Transition to transfer dst
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Upload
    vko::upload(stream, ctx->device, uploadData, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource, extent, format);

    // Transition to transfer src for download
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download using download() (returns vector)
    auto downloadFuture1 =
        vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource,
                      extent, format);

    // Download using downloadForEach() (accumulates in callback)
    std::vector<std::byte>                             downloadedForEach(imageBytes);
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> forEachRanges;
    auto                                               forEachHandle = vko::downloadForEach(
        stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource, extent,
        format, [&](VkDeviceSize offset, std::span<std::byte> chunk) {
            forEachRanges.emplace_back(offset, chunk.size());
            std::copy(chunk.begin(), chunk.end(), downloadedForEach.begin() + offset);
        });

    // Submit and wait
    stream.submit();
    auto& downloadedVec = downloadFuture1.get(ctx->device);
    forEachHandle.wait(ctx->device);

    // Verify forEach ranges are sane
    ASSERT_FALSE(forEachRanges.empty());
    VkDeviceSize totalBytes = 0;
    for (size_t i = 0; i < forEachRanges.size(); ++i) {
        auto [offset, size] = forEachRanges[i];
        EXPECT_GT(size, 0u);
        if (i > 0) {
            EXPECT_GT(offset, forEachRanges[i - 1].first);
        }
        totalBytes += size;
    }
    EXPECT_EQ(totalBytes, imageBytes);

    // Verify both methods produce identical results
    ASSERT_EQ(downloadedVec.size(), imageBytes);
    ASSERT_EQ(downloadedForEach.size(), imageBytes);
    EXPECT_EQ(downloadedVec, downloadedForEach)
        << "download() and downloadForEach() produced different results";

    // Also verify data matches uploaded data
    EXPECT_EQ(downloadedVec, uploadData) << "Downloaded data doesn't match uploaded data";
}

// downloadForEach() partial processing is covered in ImageStaging_DownloadConsistency

// -----------------------------------------------------------------------------
// downloadImageImpl() Internal Path Coverage
// -----------------------------------------------------------------------------
// Note: bytesToCopy == 0 / regions.empty() are unreachable with non-zero staging buffers
// and the beginBytes < endBytes contract of generateImageCopyRegions.

// -----------------------------------------------------------------------------
// Branch Coverage: isArrayImage true vs false
// -----------------------------------------------------------------------------
// Covered in ImageStaging_GenerateCopyRegions_ArrayAnd3D.

// -----------------------------------------------------------------------------
// Concurrent/Stress Tests
// -----------------------------------------------------------------------------

// Download multiple images concurrently (separate streams)
// Verifies no data mixing between streams
TEST_F(UnitTestFixture, ImageStaging_ConcurrentStreams) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);

    auto stagingA = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/2, /*maxPools=*/8,
                                                                /*poolSize=*/256);
    auto stagingB = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                                /*minPools=*/2, /*maxPools=*/8,
                                                                /*poolSize=*/256);
    vko::StagingStream streamA(queue, std::move(stagingA));
    vko::StagingStream streamB(queue, std::move(stagingB));

    auto makeImage = [&](VkExtent3D extent) -> vko::BoundImage<> {
        return makeTestImage(*ctx, extent, VK_FORMAT_R8G8B8A8_UNORM);
    };

    VkExtent3D extentA = {16, 12, 1};
    VkExtent3D extentB = {20, 9, 1};
    auto       imageA  = makeImage(extentA);
    auto       imageB  = makeImage(extentB);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    auto setupAndDownload = [&](auto& stream, const auto& image, VkExtent3D extent,
                                std::vector<std::byte>& uploadData) {
        VkImage         imageHandle = static_cast<VkImage>(image);
        vko::FormatInfo fmtInfo     = vko::formatInfo(VK_FORMAT_R8G8B8A8_UNORM);
        VkDeviceSize    imageBytes  = vko::imageSizeBytes(extent, 1, fmtInfo);

        uploadData.resize(imageBytes);
        for (size_t i = 0; i < uploadData.size(); ++i) {
            uploadData[i] = static_cast<std::byte>((i * 31) & 0xFF);
        }

        // Transition to transfer dst
        transitionImageLayout(stream, ctx->device, imageHandle, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        vko::upload(stream, ctx->device, uploadData, imageHandle,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource, extent,
                    VK_FORMAT_R8G8B8A8_UNORM);

        // Transition to transfer src
        transitionImageLayout(stream, ctx->device, imageHandle,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        return vko::download(stream, ctx->device, imageHandle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             subresource, extent, VK_FORMAT_R8G8B8A8_UNORM);
    };

    std::vector<std::byte> uploadA;
    std::vector<std::byte> uploadB;
    auto                   futureA = setupAndDownload(streamA, imageA, extentA, uploadA);
    auto                   futureB = setupAndDownload(streamB, imageB, extentB, uploadB);

    streamA.submit();
    streamB.submit();

    auto& downloadedA = futureA.get(ctx->device);
    auto& downloadedB = futureB.get(ctx->device);

    EXPECT_EQ(downloadedA, uploadA);
    EXPECT_EQ(downloadedB, uploadB);
}

// Download same image multiple times, verify identical results
// Deterministic behavior
TEST_F(UnitTestFixture, ImageStaging_RepeatedDownload_Deterministic) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/256);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 19, height = 17;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    VkDeviceSize    imageBytes = vko::imageSizeBytes(extent, 1, fmtInfo);

    // Upload test data
    std::vector<std::byte> uploadData(imageBytes);
    for (size_t i = 0; i < uploadData.size(); ++i) {
        uploadData[i] = static_cast<std::byte>((i * 37) & 0xFF);
    }

    // Transition + upload
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::upload(stream, ctx->device, uploadData, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource, extent, format);

    // Transition to src
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download 3 times and verify all identical
    auto future1 = vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 subresource, extent, format);
    auto future2 = vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 subresource, extent, format);
    auto future3 = vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 subresource, extent, format);

    stream.submit();
    auto& downloaded1 = future1.get(ctx->device);
    auto& downloaded2 = future2.get(ctx->device);
    auto& downloaded3 = future3.get(ctx->device);

    EXPECT_EQ(downloaded1, uploadData);
    EXPECT_EQ(downloaded2, uploadData);
    EXPECT_EQ(downloaded3, uploadData);
    EXPECT_EQ(downloaded1, downloaded2);
    EXPECT_EQ(downloaded2, downloaded3);
}

// Large image stress test (2048x2048 RGBA = 16MB)
// Many chunks required
// Verify no memory issues or timeouts
TEST_F(UnitTestFixture, ImageStaging_Stress_2048x2048) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use 128KB pools
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/256,
                                                               /*poolSize=*/128 * 1024);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 2048, height = 2048;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Simple LCG for fast pseudo-random fill
    auto lcg = [](uint32_t& state) {
        state = state * 1103515245u + 12345u;
        return static_cast<std::byte>((state >> 16) & 0xFF);
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [&](std::vector<std::byte>& data) {
                           uint32_t seed = 0xCAFEBABE;
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = lcg(seed);
                           }
                       });
}

// -----------------------------------------------------------------------------
// Transform Callback Tests (downloadImage with DstT)
// -----------------------------------------------------------------------------

// Transform callback test (std::byte -> std::byte)
// Verifies callback is invoked and output is transformed correctly
TEST_F(UnitTestFixture, ImageStaging_DownloadTransform_InvertBytes) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/256);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 13, height = 9;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    VkDeviceSize    imageBytes = vko::imageSizeBytes(extent, subresource.layerCount, fmtInfo);

    std::vector<std::byte> uploadData(imageBytes);
    for (size_t i = 0; i < uploadData.size(); ++i) {
        uploadData[i] = static_cast<std::byte>(i & 0xFF);
    }

    // Transition to transfer dst
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vko::upload(stream, ctx->device, uploadData, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource, extent, format);

    // Transition to transfer src
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    auto downloadFuture = vko::download(
        stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource, extent,
        format, [](VkDeviceSize, std::span<std::byte> input, std::span<std::byte> output) {
            EXPECT_EQ(input.size(), output.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = static_cast<std::byte>(~input[i]);
            }
        });

    stream.submit();
    auto& transformed = downloadFuture.get(ctx->device);

    ASSERT_EQ(transformed.size(), uploadData.size());
    for (size_t i = 0; i < uploadData.size(); ++i) {
        EXPECT_EQ(transformed[i], static_cast<std::byte>(~uploadData[i]));
    }
}

// Note: size-changing transforms are not supported for image downloads because output size
// is fixed to image byte size.

// =============================================================================
// IMAGE UPLOAD TESTS
// =============================================================================
// Upload uses the same region generation as download (generateImageCopyRegions),
// so we focus on upload-specific behavior. Round-trip tests verify both directions.

// -----------------------------------------------------------------------------
// Upload-Specific Tests
// -----------------------------------------------------------------------------
// These test upload-only code paths not covered by shared region logic.
// Test upload callback receives correct offset and span size
TEST_F(UnitTestFixture, ImageStaging_Upload_CallbackOffsets) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/128);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    VkDeviceSize    imageBytes = vko::imageSizeBytes(extent, subresource.layerCount, fmtInfo);
    std::vector<std::byte>                             uploadData(imageBytes);
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> uploadRanges;

    // Transition to transfer dst
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Upload via callback
    vko::upload(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource,
                extent, format, [&](VkDeviceSize offset, std::span<std::byte> chunk) {
                    uploadRanges.emplace_back(offset, chunk.size());
                    for (size_t i = 0; i < chunk.size(); ++i) {
                        std::byte value        = static_cast<std::byte>((offset + i) & 0xFF);
                        chunk[i]               = value;
                        uploadData[offset + i] = value;
                    }
                });

    // Transition to transfer src
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    auto downloadFuture =
        vko::download(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource,
                      extent, format);

    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    // Verify upload callback ranges
    ASSERT_FALSE(uploadRanges.empty());
    VkDeviceSize totalBytes = 0;
    for (size_t i = 0; i < uploadRanges.size(); ++i) {
        auto [offset, size] = uploadRanges[i];
        EXPECT_GT(size, 0u);
        if (i > 0) {
            EXPECT_GT(offset, uploadRanges[i - 1].first);
        }
        totalBytes += size;
    }
    EXPECT_EQ(totalBytes, imageBytes);

    EXPECT_EQ(downloaded, uploadData);
}

// Upload from contiguous range is covered throughout this file (e.g., ImageStaging_DownloadConsistency).

// Test upload srcRange size mismatch → throws out_of_range
TEST_F(UnitTestFixture, ImageStaging_Upload_SizeMismatchThrows) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    vko::FormatInfo fmtInfo    = vko::formatInfo(format);
    VkDeviceSize    imageBytes = vko::imageSizeBytes(extent, subresource.layerCount, fmtInfo);

    std::vector<std::byte> tooSmall(imageBytes - 1);
    std::vector<std::byte> tooLarge(imageBytes + 1);

    EXPECT_THROW(vko::upload(stream, ctx->device, tooSmall, image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource, extent, format),
                 std::out_of_range);
    EXPECT_THROW(vko::upload(stream, ctx->device, tooLarge, image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource, extent, format),
                 std::out_of_range);
}

// -----------------------------------------------------------------------------
// Round-Trip Tests (Upload → Download → Verify)
// -----------------------------------------------------------------------------
// These comprehensively test both upload and download with shared region logic.
// More valuable than separate upload/download tests for each edge case.

// Note: Round-trip coverage for 2D, 2D arrays, 3D, mipmaps, chunking, and non-power-of-2
// is provided by existing tests:
// - ImageStaging_SmokeTest_RoundTrip, ImageStaging_Format_*, ImageStaging_EdgeCase_*
// - ImageStaging_2DArray_*, ImageStaging_3D_*, ImageStaging_MipLevel_*
// - ImageStaging_Chunking_*, ImageStaging_Prime_*

// Round-trip with random data and hash verification
// Fill with PRNG seed, compute hash
// Upload, download, recompute hash
// Hash match proves zero corruption
TEST_F(UnitTestFixture, ImageStaging_RoundTrip_RandomDataHash) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use small pool to force chunking
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/512);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 37, height = 29; // Prime dimensions
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Simple linear congruential generator for reproducible random data
    auto lcg = [](uint32_t& state) {
        state = state * 1103515245u + 12345u;
        return static_cast<std::byte>((state >> 16) & 0xFF);
    };

    // Simple hash function
    auto computeHash = [](const std::vector<std::byte>& data) {
        uint64_t hash = 0x1234567890ABCDEFull;
        for (size_t i = 0; i < data.size(); ++i) {
            hash ^= static_cast<uint64_t>(data[i]) << ((i % 8) * 8);
            hash = (hash << 13) | (hash >> 51);
            hash *= 0x517cc1b727220a95ull;
        }
        return hash;
    };

    uint64_t originalHash = 0;

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [&](std::vector<std::byte>& data) {
                           uint32_t seed = 0xDEADBEEF;
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = lcg(seed);
                           }
                           originalHash = computeHash(data);
                       });

    // The roundTripImageTest already verifies equality, but we also want to
    // confirm our hash matches (proves the data we generated is what we got back)
    EXPECT_NE(originalHash, 0ull) << "Hash should be non-zero for random data";
}

// =============================================================================
// 1D TEXTURE TESTS
// =============================================================================
// 1D textures have height=1, depth=1. Tests degenerate dimension handling.

// Test 1D image (width=31, height=1, depth=1)
// Prime width, single row of pixels
// Verify chunking works with degenerate height/depth
// Format: VK_FORMAT_R8G8B8A8_UNORM
TEST_F(UnitTestFixture, ImageStaging_1D_Width31) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width  = 31;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, 1, 1}; // 1D: height=1, depth=1

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test 1D image with forced chunking (width=127, pool=29 bytes)
// Prime width, prime pool size
// Multiple chunks within the single row
TEST_F(UnitTestFixture, ImageStaging_1D_Chunked_Width127) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto                     staging = vko::vma::RecyclingStagingPool<vko::Device>(
        ctx->device, ctx->allocator,
        /*minPools=*/2, /*maxPools=*/4,
        /*poolSize=*/29); // Prime pool size to force chunking
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width  = 127; // 127 * 4 = 508 bytes, needs many chunks
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, 1, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>(i & 0xFF);
                           }
                       });
}

// Test 1D array (width=17, layers=7)
// Prime dimensions throughout
// Tests layer handling with minimal row complexity
// Verify each layer has correct data
TEST_F(UnitTestFixture, ImageStaging_1DArray_17x7) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 17, layers = 7;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, 1, 1}; // 1D array: height=1, depth=1

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 17, layers = 7;
            // Each layer gets a unique pattern based on layer index
            for (uint32_t layer = 0; layer < layers; ++layer) {
                size_t layerOffset = layer * width * 4;
                for (uint32_t x = 0; x < width; ++x) {
                    size_t pixelOffset    = layerOffset + x * 4;
                    data[pixelOffset + 0] = static_cast<std::byte>(layer);
                    data[pixelOffset + 1] = static_cast<std::byte>(x);
                    data[pixelOffset + 2] = static_cast<std::byte>((layer + x) & 0xFF);
                    data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                }
            }
        });
}

// Test 1D array chunk spanning layers (width=11, layers=13, pool=23 bytes)
// Chunk boundaries cross layer boundaries
// Maximum complexity for 1D case
TEST_F(UnitTestFixture, ImageStaging_1DArray_ChunkSpansLayers_11x13) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/23); // Prime pool size
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 11, layers = 13; // 11 * 13 * 4 = 572 bytes, many chunks
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, 1, 1};

    auto image = makeTestImage(*ctx, extent, format, layers);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = layers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 11, layers = 13;
            // Each layer gets a unique pattern based on layer index
            for (uint32_t layer = 0; layer < layers; ++layer) {
                size_t layerOffset = layer * width * 4;
                for (uint32_t x = 0; x < width; ++x) {
                    size_t pixelOffset    = layerOffset + x * 4;
                    data[pixelOffset + 0] = static_cast<std::byte>(layer);
                    data[pixelOffset + 1] = static_cast<std::byte>(x);
                    data[pixelOffset + 2] = static_cast<std::byte>((layer + x) & 0xFF);
                    data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                }
            }
        });
}

// =============================================================================
// CUBE MAP TESTS
// =============================================================================
// Cube maps are 6-layer 2D arrays with VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT.
// Face order: +X, -X, +Y, -Y, +Z, -Z (layers 0-5).

// Test cube map (width=13, height=13, 6 faces)
// Prime face dimensions
// layerCount=6 with VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
// Verify all 6 faces transfer correctly with unique patterns per face
TEST_F(UnitTestFixture, ImageStaging_CubeMap_13x13) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 13, height = 13, faces = 6;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, faces, /*mipLevels=*/1,
                               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = faces,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 13, height = 13, faces = 6;
                           // Each face gets a unique pattern based on face index (0-5)
                           for (uint32_t face = 0; face < faces; ++face) {
                               size_t faceOffset = face * width * height * 4;
                               for (uint32_t y = 0; y < height; ++y) {
                                   for (uint32_t x = 0; x < width; ++x) {
                                       size_t pixelOffset    = faceOffset + (y * width + x) * 4;
                                       data[pixelOffset + 0] = static_cast<std::byte>(face);
                                       data[pixelOffset + 1] = static_cast<std::byte>(x);
                                       data[pixelOffset + 2] = static_cast<std::byte>(y);
                                       data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                                   }
                               }
                           }
                       });
}

// Test cube map with forced chunking (width=7, height=7, pool=127 bytes)
// Small faces, prime pool size
// Chunks cross face boundaries
// Verify face data doesn't get mixed
TEST_F(UnitTestFixture, ImageStaging_CubeMap_Chunked_7x7) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/127); // Prime pool size
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 7, height = 7, faces = 6; // 7*7*6*4 = 1176 bytes, many chunks
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, faces, /*mipLevels=*/1,
                               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = faces,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           constexpr uint32_t width = 7, height = 7, faces = 6;
                           // Each face gets a unique pattern based on face index
                           for (uint32_t face = 0; face < faces; ++face) {
                               size_t faceOffset = face * width * height * 4;
                               for (uint32_t y = 0; y < height; ++y) {
                                   for (uint32_t x = 0; x < width; ++x) {
                                       size_t pixelOffset    = faceOffset + (y * width + x) * 4;
                                       data[pixelOffset + 0] = static_cast<std::byte>(face);
                                       data[pixelOffset + 1] = static_cast<std::byte>(x);
                                       data[pixelOffset + 2] = static_cast<std::byte>(y);
                                       data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                                   }
                               }
                           }
                       });
}

// Test cube map array (width=11, height=11, 3 cubes = 18 layers)
// Multiple cubes, prime dimensions
// Chunk spanning across cube boundaries (every 6 layers)
// Unique pattern per face per cube
TEST_F(UnitTestFixture, ImageStaging_CubeMapArray_11x11x3) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 11, height = 11, cubes = 3, facesPerCube = 6;
    constexpr uint32_t   totalLayers = cubes * facesPerCube; // 18 layers
    constexpr VkFormat   format      = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent      = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, totalLayers, /*mipLevels=*/1,
                               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = totalLayers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 11, height = 11, cubes = 3, facesPerCube = 6;
            // Each cube and face gets a unique pattern
            for (uint32_t cube = 0; cube < cubes; ++cube) {
                for (uint32_t face = 0; face < facesPerCube; ++face) {
                    uint32_t layer       = cube * facesPerCube + face;
                    size_t   layerOffset = layer * width * height * 4;
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            size_t pixelOffset    = layerOffset + (y * width + x) * 4;
                            data[pixelOffset + 0] = static_cast<std::byte>(cube);
                            data[pixelOffset + 1] = static_cast<std::byte>(face);
                            data[pixelOffset + 2] = static_cast<std::byte>((x + y) & 0xFF);
                            data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                        }
                    }
                }
            }
        });
}

// Test cube map array large (width=19, height=19, 7 cubes = 42 layers)
// Many cubes, stress test layer handling
// Verify no off-by-one in cube/face indexing
TEST_F(UnitTestFixture, ImageStaging_CubeMapArray_Large_19x19x7) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 19, height = 19, cubes = 7, facesPerCube = 6;
    constexpr uint32_t   totalLayers = cubes * facesPerCube; // 42 layers
    constexpr VkFormat   format      = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent      = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format, totalLayers, /*mipLevels=*/1,
                               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = totalLayers,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 19, height = 19, cubes = 7, facesPerCube = 6;
            // Each cube and face gets a unique pattern
            for (uint32_t cube = 0; cube < cubes; ++cube) {
                for (uint32_t face = 0; face < facesPerCube; ++face) {
                    uint32_t layer       = cube * facesPerCube + face;
                    size_t   layerOffset = layer * width * height * 4;
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            size_t pixelOffset    = layerOffset + (y * width + x) * 4;
                            data[pixelOffset + 0] = static_cast<std::byte>(cube);
                            data[pixelOffset + 1] = static_cast<std::byte>(face);
                            data[pixelOffset + 2] = static_cast<std::byte>((x + y) & 0xFF);
                            data[pixelOffset + 3] = static_cast<std::byte>(0xFF);
                        }
                    }
                }
            }
        });
}

// =============================================================================
// PRIME NUMBER STRESS TESTS (Alignment Edge Cases)
// =============================================================================
// Using prime numbers for all dimensions ensures nothing aligns naturally,
// exposing any hidden assumptions about power-of-2 or alignment.

// Test 2D image with prime dimensions (23x19), prime staging pool (251 bytes)
// No natural alignment between any sizes
// Forces maximum region splitting
// Format: VK_FORMAT_R8G8B8A8_UNORM (4 bpp)
TEST_F(UnitTestFixture, ImageStaging_Prime_2D_23x19) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Image = 23*19*4 = 1748 bytes, pool = 251 bytes (prime)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/251);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 23, height = 19;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<std::byte>((i * 17) & 0xFF); // Prime multiplier
            }
        });
}

// Test 2D array (17x13, 7 layers), pool=509 bytes
// All prime: width, height, layers, pool
// Chunk boundaries at unpredictable positions
TEST_F(UnitTestFixture, ImageStaging_Prime_2DArray_17x13x7) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Layer = 17*13*4 = 884 bytes, total = 884*7 = 6188 bytes, pool = 509 bytes (prime)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/16,
                                                               /*poolSize=*/509);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 17, height = 13;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    vko::BoundImage image(
        ctx->device,
        VkImageCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext       = nullptr,
            .flags       = 0,
            .imageType   = VK_IMAGE_TYPE_2D,
            .format      = format,
            .extent      = extent,
            .mipLevels   = 1,
            .arrayLayers = 7,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = VK_IMAGE_TILING_OPTIMAL,
            .usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        },
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ctx->allocator);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 7,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>((i * 23) & 0xFF);
                           }
                       });
}

// Test 3D texture (7x11x13), staging pool = 127 bytes
// All prime dimensions
// Chunks cross all boundary types (row, slice, depth)
// Most complex chunking scenario
TEST_F(UnitTestFixture, ImageStaging_Prime_3D_7x11x13) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Slice = 7*11*4 = 308 bytes, total = 308*13 = 4004 bytes, pool = 127 bytes (prime)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/64,
                                                               /*poolSize=*/127);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 7, height = 11, depth = 13;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>((i * 31) & 0xFF);
                           }
                       });
}

// Test 3D texture (5x7x31), pool=251 bytes
// Small XY, large Z (many depth slices)
// Many depth transitions per chunk
TEST_F(UnitTestFixture, ImageStaging_Prime_3D_5x7x31) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Slice = 5*7*4 = 140 bytes, total = 140*31 = 4340 bytes, pool = 251 bytes (prime)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/32,
                                                               /*poolSize=*/251);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 5, height = 7, depth = 31;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(
        stream, ctx->device, image, extent, format, subresource, [](std::vector<std::byte>& data) {
            constexpr uint32_t width = 5, height = 7, depth = 31;
            for (uint32_t z = 0; z < depth; ++z) {
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t offset    = ((z * height + y) * width + x) * 4;
                        data[offset + 0] = static_cast<std::byte>(z);
                        data[offset + 1] = static_cast<std::byte>(y);
                        data[offset + 2] = static_cast<std::byte>(x);
                        data[offset + 3] = static_cast<std::byte>((z + y + x) & 0xFF);
                    }
                }
            }
        });
}

// Test worst-case alignment: 2D (101x103), 8 bpp, pool=1021 bytes
// Large primes, all coprime to each other
// VK_FORMAT_R16G16B16A16_SFLOAT (8 bytes per pixel)
// Ensures no accidental alignment anywhere
TEST_F(UnitTestFixture, ImageStaging_Prime_WorstCase_101x103) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Row = 101*8 = 808 bytes, total = 808*103 = 83224 bytes, pool = 1021 bytes (prime)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/128,
                                                               /*poolSize=*/1021);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 101, height = 103;
    constexpr VkFormat   format = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Fill with sequential bytes
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = static_cast<std::byte>((i * 37) & 0xFF);
                           }
                       });
}

// Large image stress test: 1024x1024 RGBA
// 4MB image with moderate chunking
// Verifies performance and correctness at scale
TEST_F(UnitTestFixture, ImageStaging_Stress_1024x1024) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use 64KB pools (typical staging buffer size)
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/128,
                                                               /*poolSize=*/64 * 1024);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 1024, height = 1024;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    // Simple LCG for fast pseudo-random fill
    auto lcg = [](uint32_t& state) {
        state = state * 1103515245u + 12345u;
        return static_cast<std::byte>((state >> 16) & 0xFF);
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [&](std::vector<std::byte>& data) {
                           uint32_t seed = 0x12345678;
                           for (size_t i = 0; i < data.size(); ++i) {
                               data[i] = lcg(seed);
                           }
                       });
}

// Large image stress test with aggressive chunking: 512x512, tiny pool
// Forces maximum number of chunks and region calculations
TEST_F(UnitTestFixture, ImageStaging_Stress_512x512_TinyPool) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Use 1KB pools - very aggressive chunking
    // 512*512*4 = 1MB, with 1KB pools = ~1000 chunks
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/2048,
                                                               /*poolSize=*/1024);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 512, height = 512;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    VkImageSubresourceLayers subresource{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    roundTripImageTest(stream, ctx->device, image, extent, format, subresource,
                       [](std::vector<std::byte>& data) {
                           // Use position-based pattern for debugging
                           constexpr uint32_t width = 512;
                           for (size_t i = 0; i < data.size(); i += 4) {
                               uint32_t pixel = static_cast<uint32_t>(i / 4);
                               uint32_t x     = pixel % width;
                               uint32_t y     = pixel / width;
                               data[i + 0]    = static_cast<std::byte>(x & 0xFF);
                               data[i + 1]    = static_cast<std::byte>(y & 0xFF);
                               data[i + 2]    = static_cast<std::byte>((x ^ y) & 0xFF);
                               data[i + 3]    = static_cast<std::byte>(0xFF);
                           }
                       });
}

// ============================================================================
// Sub-region transfer tests
// ============================================================================

// Test downloading a sub-region from a 2D image with X/Y offset
// Upload full image with known pattern, then download just a sub-region
TEST_F(UnitTestFixture, ImageStaging_SubRegion_2D_OffsetXY) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Generate test pattern: each pixel encodes its (x,y) position
    std::vector<std::byte> fullPattern(vko::imageSizeBytes(extent, 1, vko::formatInfo(format)));
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t i           = (y * width + x) * 4;
            fullPattern[i + 0] = static_cast<std::byte>(x);
            fullPattern[i + 1] = static_cast<std::byte>(y);
            fullPattern[i + 2] = static_cast<std::byte>(x ^ y);
            fullPattern[i + 3] = static_cast<std::byte>(0xFF);
        }
    }

    // Upload full image
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    // Transition for download
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download a sub-region with offset (8,4) and extent (12,8)
    constexpr uint32_t subOffsetX = 8, subOffsetY = 4;
    constexpr uint32_t subWidth = 12, subHeight = 8;

    vko::ImageRegion subRegion(image, vko::Region{
                                          .subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                          .offset      = {static_cast<int32_t>(subOffsetX),
                                                          static_cast<int32_t>(subOffsetY), 0},
                                          .extent      = {subWidth, subHeight, 1},
                                      });

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    // Verify downloaded data matches expected sub-region from pattern
    ASSERT_EQ(downloaded.size(), subWidth * subHeight * 4);

    for (uint32_t y = 0; y < subHeight; ++y) {
        for (uint32_t x = 0; x < subWidth; ++x) {
            size_t  dstIdx    = (y * subWidth + x) * 4;
            uint8_t expectedX = uint8_t(subOffsetX + x);
            uint8_t expectedY = uint8_t(subOffsetY + y);

            EXPECT_EQ(static_cast<uint8_t>(downloaded[dstIdx + 0]), expectedX)
                << "Mismatch at sub-region (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[dstIdx + 1]), expectedY)
                << "Mismatch at sub-region (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[dstIdx + 2]),
                      static_cast<uint8_t>(expectedX ^ expectedY))
                << "Mismatch at sub-region (" << x << "," << y << ")";
        }
    }
}

// Test uploading to a sub-region and verifying surrounding pixels are unchanged
TEST_F(UnitTestFixture, ImageStaging_SubRegion_Upload_PreservesSurrounding) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Upload background (all zeros)
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize, std::span<std::byte> dst) {
                    std::fill(dst.begin(), dst.end(), std::byte{0x00});
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Upload distinct pattern to sub-region
    constexpr uint32_t subOffsetX = 10, subOffsetY = 12;
    constexpr uint32_t subWidth = 8, subHeight = 6;

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vko::ImageRegion subRegion(image, vko::Region{
                                          .subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                          .offset      = {static_cast<int32_t>(subOffsetX),
                                                          static_cast<int32_t>(subOffsetY), 0},
                                          .extent      = {subWidth, subHeight, 1},
                                      });

    vko::upload(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize, std::span<std::byte> dst) {
                    std::fill(dst.begin(), dst.end(), std::byte{0xFF});
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download and verify
    auto downloadFuture = vko::download(stream, ctx->device, fullRegion,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx         = (y * width + x) * 4;
            bool   inSubRegion = (x >= subOffsetX && x < subOffsetX + subWidth && y >= subOffsetY &&
                                y < subOffsetY + subHeight);
            uint8_t expected   = inSubRegion ? 0xFF : 0x00;

            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 0]), expected)
                << "Mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 1]), expected)
                << "Mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 2]), expected)
                << "Mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 3]), expected)
                << "Mismatch at (" << x << "," << y << ")";
        }
    }
}

// Test single pixel sub-region (edge case)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_SinglePixel) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {16, 16, 1};

    auto image = makeTestImage(*ctx, extent, format);

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Upload single pixel at (7, 9)
    constexpr uint32_t pixelX = 7, pixelY = 9;
    vko::ImageRegion   singlePixel(
        image, vko::Region{
                     .subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .offset      = {static_cast<int32_t>(pixelX), static_cast<int32_t>(pixelY), 0},
                     .extent      = {1, 1, 1},
               });

    std::byte pixelData[4] = {std::byte{0xDE}, std::byte{0xAD}, std::byte{0xBE}, std::byte{0xEF}};
    vko::upload(stream, ctx->device, singlePixel, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize, std::span<std::byte> dst) {
                    std::copy(std::begin(pixelData), std::end(pixelData), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    auto downloadFuture = vko::download(stream, ctx->device, singlePixel,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), 4u);
    EXPECT_EQ(downloaded[0], std::byte{0xDE});
    EXPECT_EQ(downloaded[1], std::byte{0xAD});
    EXPECT_EQ(downloaded[2], std::byte{0xBE});
    EXPECT_EQ(downloaded[3], std::byte{0xEF});
}

// Test sub-region of a 2D array - download single layer
TEST_F(UnitTestFixture, ImageStaging_SubRegion_2DArray_SingleLayer) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16, layers = 4;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto   image     = makeTestImage(*ctx, extent, format, layers);
    size_t layerSize = width * height * 4;

    // Generate test pattern: each layer filled with its index * 50
    std::vector<std::byte> fullPattern(layerSize * layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        std::fill(fullPattern.begin() + layer * layerSize,
                  fullPattern.begin() + (layer + 1) * layerSize,
                  static_cast<std::byte>(layer * 50));
    }

    // Upload full image
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                          layers);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                          layers);

    // Download just layer 2
    constexpr uint32_t targetLayer = 2;
    vko::ImageRegion   singleLayerRegion(
        image, vko::Region{
                     .subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, targetLayer, 1},
                     .offset      = {0, 0, 0},
                     .extent      = extent,
               });

    auto downloadFuture = vko::download(stream, ctx->device, singleLayerRegion,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), layerSize);

    // Verify all bytes match expected layer pattern
    uint8_t expected = targetLayer * 50;
    for (size_t i = 0; i < downloaded.size(); ++i) {
        EXPECT_EQ(static_cast<uint8_t>(downloaded[i]), expected)
            << "Mismatch at byte " << i << " of layer " << targetLayer;
    }
}

// Test 3D image sub-region - download Z slice range
TEST_F(UnitTestFixture, ImageStaging_SubRegion_3D_SliceSubset) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 8, height = 8, depth = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, depth};

    auto image = makeTestImage(*ctx, extent, format);

    // Generate pattern: each Z slice filled with z value
    size_t                 sliceSize = width * height * 4;
    size_t                 totalSize = sliceSize * depth;
    std::vector<std::byte> fullPattern(totalSize);
    for (uint32_t z = 0; z < depth; ++z) {
        std::fill(fullPattern.begin() + z * sliceSize, fullPattern.begin() + (z + 1) * sliceSize,
                  static_cast<std::byte>(z * 30));
    }

    // Upload full image
    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download Z slices 2-5 (4 slices)
    constexpr uint32_t zOffset = 2, zCount = 4;
    auto               subRegion =
        fullRegion.subregion({0, 0, static_cast<int32_t>(zOffset)}, {width, height, zCount});

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), sliceSize * zCount);

    for (uint32_t z = 0; z < zCount; ++z) {
        uint8_t expected = uint8_t((zOffset + z) * 30);
        for (size_t i = 0; i < sliceSize; ++i) {
            EXPECT_EQ(static_cast<uint8_t>(downloaded[z * sliceSize + i]), expected)
                << "Mismatch at slice " << z << " byte " << i;
        }
    }
}

// Test single row sub-region (edge case)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_SingleRow) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Generate pattern: each row filled with row index
    std::vector<std::byte> fullPattern(width * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        std::fill(fullPattern.begin() + y * width * 4, fullPattern.begin() + (y + 1) * width * 4,
                  static_cast<std::byte>(y));
    }

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download just row 15
    constexpr uint32_t targetRow = 15;
    auto subRegion = fullRegion.subregion({0, static_cast<int32_t>(targetRow), 0}, {width, 1, 1});

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), width * 4);
    for (size_t i = 0; i < downloaded.size(); ++i) {
        EXPECT_EQ(static_cast<uint8_t>(downloaded[i]), targetRow) << "Mismatch at byte " << i;
    }
}

// Test single column sub-region (edge case)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_SingleColumn) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Generate pattern: each column filled with column index
    std::vector<std::byte> fullPattern(width * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 4;
            std::fill(fullPattern.begin() + idx, fullPattern.begin() + idx + 4,
                      static_cast<std::byte>(x));
        }
    }

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download just column 20
    constexpr uint32_t targetCol = 20;
    auto subRegion = fullRegion.subregion({static_cast<int32_t>(targetCol), 0, 0}, {1, height, 1});

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t c = 0; c < 4; ++c) {
            EXPECT_EQ(static_cast<uint8_t>(downloaded[y * 4 + c]), targetCol)
                << "Mismatch at row " << y << " channel " << c;
        }
    }
}

// Test sub-region with chunked transfer (small staging pool)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_Chunked_SmallPool) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    // Tiny pool to force chunking within the sub-region
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/8,
                                                               /*poolSize=*/256);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 32, height = 32;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    // Generate pattern: each pixel encodes position
    std::vector<std::byte> fullPattern(width * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t i           = (y * width + x) * 4;
            fullPattern[i + 0] = static_cast<std::byte>(x);
            fullPattern[i + 1] = static_cast<std::byte>(y);
            fullPattern[i + 2] = static_cast<std::byte>(x ^ y);
            fullPattern[i + 3] = static_cast<std::byte>(0xFF);
        }
    }

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download a sub-region that will require multiple chunks (16x16 = 1024 bytes, pool=256)
    constexpr uint32_t subOffsetX = 8, subOffsetY = 8;
    constexpr uint32_t subWidth = 16, subHeight = 16;
    auto               subRegion = fullRegion.subregion(
        {static_cast<int32_t>(subOffsetX), static_cast<int32_t>(subOffsetY), 0},
        {subWidth, subHeight, 1});

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), subWidth * subHeight * 4);

    for (uint32_t y = 0; y < subHeight; ++y) {
        for (uint32_t x = 0; x < subWidth; ++x) {
            size_t  idx       = (y * subWidth + x) * 4;
            uint8_t expectedX = uint8_t(subOffsetX + x);
            uint8_t expectedY = uint8_t(subOffsetY + y);

            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 0]), expectedX)
                << "X mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 1]), expectedY)
                << "Y mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(static_cast<uint8_t>(downloaded[idx + 2]),
                      static_cast<uint8_t>(expectedX ^ expectedY))
                << "XOR mismatch at (" << x << "," << y << ")";
        }
    }
}

// Test sub-region of array image with layer subset (middle layers)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_2DArray_LayerSubset) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16, layers = 8;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto   image     = makeTestImage(*ctx, extent, format, layers);
    size_t layerSize = width * height * 4;

    // Generate pattern: each layer filled with layer index * 25
    std::vector<std::byte> fullPattern(layerSize * layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        std::fill(fullPattern.begin() + layer * layerSize,
                  fullPattern.begin() + (layer + 1) * layerSize,
                  static_cast<std::byte>(layer * 25));
    }

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                          layers);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                          layers);

    // Download layers 2-5 (4 layers from the middle)
    constexpr uint32_t baseLayer = 2, layerCount = 4;
    auto               subRegion = fullRegion.subregion({0, 0, 0}, extent, baseLayer, layerCount);

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), layerSize * layerCount);

    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        uint8_t expected = uint8_t((baseLayer + layer) * 25);
        for (size_t i = 0; i < layerSize; ++i) {
            EXPECT_EQ(static_cast<uint8_t>(downloaded[layer * layerSize + i]), expected)
                << "Mismatch at layer " << layer << " byte " << i;
        }
    }
}

// Test sub-region that equals full image (offset 0,0,0 with full extent)
TEST_F(UnitTestFixture, ImageStaging_SubRegion_FullImage) {
    vko::SerialTimelineQueue queue(ctx->device, ctx->queueFamilyIndex, 0);
    auto staging = vko::vma::RecyclingStagingPool<vko::Device>(ctx->device, ctx->allocator,
                                                               /*minPools=*/2, /*maxPools=*/4,
                                                               /*poolSize=*/1 << 16);
    vko::StagingStream stream(queue, std::move(staging));

    constexpr uint32_t   width = 16, height = 16;
    constexpr VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkExtent3D extent = {width, height, 1};

    auto image = makeTestImage(*ctx, extent, format);

    std::vector<std::byte> fullPattern(width * height * 4);
    for (size_t i = 0; i < fullPattern.size(); ++i) {
        fullPattern[i] = static_cast<std::byte>(i & 0xFF);
    }

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vko::ImageRegion fullRegion(image, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    vko::upload(stream, ctx->device, fullRegion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format,
                [&](VkDeviceSize offset, std::span<std::byte> dst) {
                    std::copy(fullPattern.begin() + offset,
                              fullPattern.begin() + offset + dst.size(), dst.begin());
                });

    transitionImageLayout(stream, ctx->device, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Download using subregion() with full image bounds
    auto subRegion = fullRegion.subregion({0, 0, 0}, extent);

    auto downloadFuture =
        vko::download(stream, ctx->device, subRegion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format);
    stream.submit();
    auto& downloaded = downloadFuture.get(ctx->device);

    ASSERT_EQ(downloaded.size(), fullPattern.size());
    EXPECT_EQ(downloaded, fullPattern);
}
