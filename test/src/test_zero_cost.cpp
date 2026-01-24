// Copyright (c) 2026 Pyarelal Knowles, MIT License
// Test to verify zero-cost abstractions - compiler optimizations should eliminate overhead

#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vko/pnext_chain.hpp>
#include <vulkan/vulkan_core.h>

// Linker-provided symbols from objcopy (binary data embedded as object file)
// From test_zero_cost_symbols.txt -> test_zero_cost_symbols.o
#ifdef SYMBOL_DATA_SUPPORTED
extern "C" {
extern const char _binary_test_zero_cost_symbols_txt_start[];
extern const char _binary_test_zero_cost_symbols_txt_end[];
}
#endif

// Cross-platform noinline attribute (prevents inlining so we can measure code size)
#if defined(__GNUC__) || defined(__clang__)
    #define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define NOINLINE __declspec(noinline)
#else
    #define NOINLINE
#endif

// Mock present function that we want to inline into
// Mark as noinline so we can see it in the symbol table and measure its size
NOINLINE
int mockPresentDirect(int device, int queue, uint64_t presentId, const void* pNext) {
    // Manually construct the struct (baseline for comparison)
    VkPresentIdKHR presentIdInfo{
        .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
        .pNext          = pNext,
        .swapchainCount = 1U,
        .pPresentIds    = &presentId,
    };

    VkPresentInfoKHR presentInfo{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = &presentIdInfo,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores    = nullptr,
        .swapchainCount     = 1,
        .pSwapchains        = nullptr,
        .pImageIndices      = nullptr,
        .pResults           = nullptr,
    };

    // Simulate vkQueuePresentKHR - just return something that depends on inputs
    return device + queue + static_cast<int>(presentId) + (presentInfo.pNext ? 1 : 0);
}

// Mock present using chainPNext - should inline to same code as above
NOINLINE
int mockPresentChained(int device, int queue, uint64_t presentId, const void* userPNext) {
    return vko::chainPNext(
        userPNext,
        [&presentId](auto&& cont, const void* pNext) {
            VkPresentIdKHR presentIdInfo{
                .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
                .pNext          = pNext,
                .swapchainCount = 1U,
                .pPresentIds    = &presentId,
            };
            return cont(&presentIdInfo);
        },
        [device, queue, presentId, userPNext](const void* chainedPNext) {
            VkPresentInfoKHR presentInfo{
                .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext              = chainedPNext,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores    = nullptr,
                .swapchainCount     = 1,
                .pSwapchains        = nullptr,
                .pImageIndices      = nullptr,
                .pResults           = nullptr,
            };

            return device + queue + static_cast<int>(presentId) + (presentInfo.pNext ? 1 : 0);
        });
}

// Version using the with<> helper
NOINLINE
int mockPresentWithHelper(int device, int queue, uint64_t presentId) {
    // vko::with<> automatically handles pNext, we just pass the struct fields after sType
    return vko::chainPNext(
        nullptr, vko::with<VkPresentIdKHR>(1U, &presentId), // swapchainCount, pPresentIds
        [device, queue, presentId](const void* chainedPNext) {
            VkPresentInfoKHR presentInfo{
                .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext              = chainedPNext,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores    = nullptr,
                .swapchainCount     = 1,
                .pSwapchains        = nullptr,
                .pImageIndices      = nullptr,
                .pResults           = nullptr,
            };

            return device + queue + static_cast<int>(presentId) + (presentInfo.pNext ? 1 : 0);
        });
}

// Version using args-aware tuple API
NOINLINE
int mockPresentArgsTuple(int device, int queue, uint64_t presentId) {
    auto presentIdMod = [](auto&& args, auto&& cont, const void* pNext) {
        auto& [dev, q, pId] = args;
        VkPresentIdKHR presentIdInfo{
            .sType          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .pNext          = pNext,
            .swapchainCount = 1U,
            .pPresentIds    = &pId,
        };
        return cont(&presentIdInfo);
    };

    auto finalFunc = [](int dev, int q, uint64_t pId, const void* chainedPNext) {
        VkPresentInfoKHR presentInfo{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = chainedPNext,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores    = nullptr,
            .swapchainCount     = 1,
            .pSwapchains        = nullptr,
            .pImageIndices      = nullptr,
            .pResults           = nullptr,
        };
        return dev + q + static_cast<int>(pId) + (presentInfo.pNext ? 1 : 0);
    };

    return vko::chainPNext(nullptr, std::tuple{device, queue, presentId},
                           std::tuple{presentIdMod, finalFunc});
}

// GTest: Functional equivalence
TEST(ZeroCost, PNextChaining) {
    int      device    = 1;
    int      queue     = 2;
    uint64_t presentId = 100;

    int result1 = mockPresentDirect(device, queue, presentId, nullptr);
    int result2 = mockPresentChained(device, queue, presentId, nullptr);
    int result3 = mockPresentWithHelper(device, queue, presentId);
    int result4 = mockPresentArgsTuple(device, queue, presentId);

    EXPECT_EQ(result1, result2) << "Chained version produces different result";
    EXPECT_EQ(result2, result3) << "Helper version produces different result";
    EXPECT_EQ(result3, result4) << "Args-tuple version produces different result";
}

// GTest: Symbol size verification (from embedded build-time data)
TEST(ZeroCost, SymbolSizes) {
#ifndef SYMBOL_DATA_SUPPORTED
    GTEST_SKIP() << "Symbol extraction not supported on this platform";
#else
    // Read embedded symbol data (objcopy embeds the text file as binary data)
    std::string nm_data(_binary_test_zero_cost_symbols_txt_start,
                        _binary_test_zero_cost_symbols_txt_end -
                            _binary_test_zero_cost_symbols_txt_start);
    ASSERT_FALSE(nm_data.empty()) << "No symbol data found - build may have failed";

    // Pass 1: Parse all exported mockPresent function symbols
    // Expected nm format: "address size type name"
    // Example: "0000000000000020 0000000000000007 T _Z17mockPresentDirectiimPKv"
    std::map<std::string, size_t> symbols;
    std::istringstream            stream(nm_data);
    std::string                   line;

    while (std::getline(stream, line)) {
        if (line.find("mockPresent") == std::string::npos ||
            line.find(" T ") == std::string::npos) {
            continue;
        }

        std::istringstream line_stream(line);
        std::string        address, type, name;
        size_t             size;

        line_stream >> address >> std::hex >> size >> type >> name;
        if (!line_stream.fail()) {
            symbols[name] = size;
        }
    }

    // Pass 2: Verify expected symbols exist and have identical sizes
    ASSERT_FALSE(symbols.empty()) << "No mockPresent symbols found\nData:\n" << nm_data;

    // Helper to find symbol size by substring (compiler mangles names)
    auto symbolSize = [&](const std::string& substring) -> size_t {
        for (const auto& [name, size] : symbols) {
            if (name.find(substring) != std::string::npos) {
                return size;
            }
        }
        throw std::runtime_error("Symbol containing '" + substring + "' not found in nm output");
    };

    // All three functions must compile to identical sizes (zero-cost abstraction)
    EXPECT_EQ(symbolSize("mockPresentChained"), symbolSize("mockPresentDirect"))
        << "Abstraction looks to have non-zero overhead";

    EXPECT_EQ(symbolSize("mockPresentWithHelper"), symbolSize("mockPresentDirect"))
        << "Abstraction looks to have non-zero overhead";

    EXPECT_EQ(symbolSize("mockPresentArgsTuple"), symbolSize("mockPresentDirect"))
        << "Args-tuple API looks to have non-zero overhead";
#endif
}
