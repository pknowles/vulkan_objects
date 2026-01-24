// Copyright (c) 2026 Pyarelal Knowles, MIT License
// Test to verify pNext chain correctness

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <vko/pnext_chain.hpp>
#include <vulkan/vulkan_core.h>

struct MockContext {
    std::vector<const void*> pnextChain;
    std::vector<std::string> callOrder;
};

// Mock Vulkan-like structs for testing
struct MockBaseInfo {
    VkStructureType sType;
    const void*     pNext;
    uint32_t        value;
};

struct MockExtension1 {
    VkStructureType sType;
    const void*     pNext;
    uint32_t        ext1Value;
};

struct MockExtension2 {
    VkStructureType sType;
    const void*     pNext;
    uint32_t        ext2Value;
};

// Mock API function that verifies the pNext chain is correctly formed
// Returns a checksum of all values found by walking the chain
uint32_t mockVulkanAPI(MockContext& ctx, const MockBaseInfo* pInfo) {
    EXPECT_NE(pInfo, nullptr);
    EXPECT_EQ(pInfo->sType, VK_STRUCTURE_TYPE_APPLICATION_INFO);

    uint32_t checksum = pInfo->value;

    // Walk the pNext chain
    const void* pNext = pInfo->pNext;
    while (pNext != nullptr) {
        ctx.pnextChain.push_back(pNext);
        // Cast to get sType (all Vulkan structs start with sType + pNext)
        const auto* header = static_cast<const VkBaseInStructure*>(pNext);

        if (header->sType == VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO) {
            // This is MockExtension1
            const auto* ext1 = static_cast<const MockExtension1*>(pNext);
            checksum += ext1->ext1Value;
            pNext = ext1->pNext;
        } else if (header->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO) {
            // This is MockExtension2
            const auto* ext2 = static_cast<const MockExtension2*>(pNext);
            checksum += ext2->ext2Value;
            pNext = ext2->pNext;
        } else {
            ADD_FAILURE() << "Unexpected sType in chain: " << header->sType;
            break;
        }
    }

    return checksum;
}

// Test: Single modifier
TEST(PNextChain, SingleModifier) {
    MockContext ctx;
    const void* ext1Ptr = nullptr;

    auto modifier1 = [&](auto&& cont, const void* pNext) {
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = 100,
        };

        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto result = vko::chainPNext(nullptr, modifier1, [&](const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        return mockVulkanAPI(ctx, &info);
    });

    ASSERT_EQ(ctx.pnextChain.size(), 1u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr) << "final should see ext1 as the head of the chain";

    EXPECT_EQ(result, 110); // 10 + 100
}

// Test: Two modifiers - verify they chain together
TEST(PNextChain, TwoModifiers) {
    MockContext ctx;
    const void* ext1Ptr = nullptr;
    const void* ext2Ptr = nullptr;

    auto modifier1 = [&](auto&& cont, const void* pNext) {
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = 100,
        };

        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto modifier2 = [&](auto&& cont, const void* pNext) {
        MockExtension2 ext2{
            .sType     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext     = pNext,
            .ext2Value = 200,
        };

        ext2Ptr = &ext2;
        return cont(&ext2);
    };

    auto result = vko::chainPNext(nullptr, modifier2, modifier1, [&](const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        return mockVulkanAPI(ctx, &info);
    });

    ASSERT_EQ(ctx.pnextChain.size(), 2u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr) << "final should see ext1 as the head of the chain";
    EXPECT_EQ(ctx.pnextChain[1], ext2Ptr) << "final should see ext2 after ext1";

    EXPECT_EQ(result, 310); // 10 + 100 + 200
}

// Test: User provides their own pNext that should be at the end of the chain
TEST(PNextChain, UserProvidedPNext) {
    MockContext ctx;
    const void* ext1Ptr = nullptr;

    MockExtension2 userExtension{
        .sType     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext     = nullptr,
        .ext2Value = 500,
    };

    auto modifier1 = [&](auto&& cont, const void* pNext) {
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = 100,
        };

        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto result = vko::chainPNext(&userExtension, modifier1, [&](const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        return mockVulkanAPI(ctx, &info);
    });

    ASSERT_EQ(ctx.pnextChain.size(), 2u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr) << "final should see ext1 as the head of the chain";
    EXPECT_EQ(ctx.pnextChain[1], &userExtension) << "final should see user pNext after ext1";

    EXPECT_EQ(result, 610); // 10 + 100 + 500
}

// Test: Three modifiers to really stress the recursion
TEST(PNextChain, ThreeModifiers) {
    MockContext ctx;
    const void* ext1Ptr = nullptr;
    const void* ext2Ptr = nullptr;
    const void* ext3Ptr = nullptr;

    auto modifier1 = [&](auto&& cont, const void* pNext) {
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = 100,
        };

        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto modifier2 = [&](auto&& cont, const void* pNext) {
        MockExtension2 ext2{
            .sType     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext     = pNext,
            .ext2Value = 200,
        };

        ext2Ptr = &ext2;
        return cont(&ext2);
    };

    auto modifier3 = [&](auto&& cont, const void* pNext) {
        MockExtension1 ext3{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = 300,
        };

        ext3Ptr = &ext3;
        return cont(&ext3);
    };

    auto result = vko::chainPNext(nullptr, modifier3, modifier2, modifier1, [&](const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        return mockVulkanAPI(ctx, &info);
    });

    ASSERT_EQ(ctx.pnextChain.size(), 3u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr) << "final should see ext1 as the head of the chain";
    EXPECT_EQ(ctx.pnextChain[1], ext2Ptr) << "final should see ext2 after ext1";
    EXPECT_EQ(ctx.pnextChain[2], ext3Ptr) << "final should see ext3 after ext2";

    EXPECT_EQ(result, 610); // 10 + 100 + 200 + 300
}

// ============================================================================
// Args-aware API tests (tuple-based)
// ============================================================================

// Test: Args-aware with one modifier, 4 args (matches original test pattern)
TEST(PNextChain, ArgsTuple_OneModifier_FourArgs) {
    MockContext ctx;
    const void* ext1Ptr   = nullptr;
    int         device    = 1;
    int         queue     = 2;
    uint32_t    ext1Value = 100;

    auto modifier = [&](auto&& args, auto&& cont, const void* pNext) {
        auto& [dev, q, ext1Val, ctxPtr] = args;
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = ext1Val,
        };
        ctxPtr->callOrder.push_back("modifier1");
        ext1Ptr = &ext1;
        // Verify args are accessible
        EXPECT_EQ(dev, 1);
        EXPECT_EQ(q, 2);
        return cont(&ext1);
    };

    auto finalFunc = [](int dev, int q, uint32_t ext1Val, MockContext* ctxPtr, const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        ctxPtr->callOrder.push_back("finalFunc");
        // Verify all args passed through
        EXPECT_EQ(dev, 1);
        EXPECT_EQ(q, 2);
        EXPECT_EQ(ext1Val, 100u);
        return mockVulkanAPI(*ctxPtr, &info);
    };

    auto result = vko::chainPNext(nullptr, std::tuple{device, queue, ext1Value, &ctx},
                                  std::tuple{modifier, finalFunc});

    ASSERT_EQ(ctx.callOrder.size(), 2u);
    EXPECT_EQ(ctx.callOrder[0], "modifier1");
    EXPECT_EQ(ctx.callOrder[1], "finalFunc");

    ASSERT_EQ(ctx.pnextChain.size(), 1u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr);

    EXPECT_EQ(result, 110); // 10 + 100
}

// Test: Args-aware with two modifiers, 3 args
TEST(PNextChain, ArgsTuple_TwoModifiers) {
    MockContext ctx;
    const void* ext1Ptr   = nullptr;
    const void* ext2Ptr   = nullptr;
    uint32_t    ext1Value = 100;
    uint32_t    ext2Value = 200;

    auto modifier1 = [&](auto&& args, auto&& cont, const void* pNext) {
        auto& [e1, e2, ctxPtr] = args;
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = e1,
        };
        ctxPtr->callOrder.push_back("modifier1");
        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto modifier2 = [&](auto&& args, auto&& cont, const void* pNext) {
        auto& [e1, e2, ctxPtr] = args;
        MockExtension2 ext2{
            .sType     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext     = pNext,
            .ext2Value = e2,
        };
        ctxPtr->callOrder.push_back("modifier2");
        ext2Ptr = &ext2;
        return cont(&ext2);
    };

    auto finalFunc = [](uint32_t e1, uint32_t e2, MockContext* ctxPtr, const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        ctxPtr->callOrder.push_back("finalFunc");
        (void)e1;
        (void)e2; // Args available if needed
        return mockVulkanAPI(*ctxPtr, &info);
    };

    auto result = vko::chainPNext(nullptr, std::tuple{ext1Value, ext2Value, &ctx},
                                  std::tuple{modifier2, modifier1, finalFunc});

    ASSERT_EQ(ctx.callOrder.size(), 3u);
    EXPECT_EQ(ctx.callOrder[0], "modifier2");
    EXPECT_EQ(ctx.callOrder[1], "modifier1");
    EXPECT_EQ(ctx.callOrder[2], "finalFunc");

    ASSERT_EQ(ctx.pnextChain.size(), 2u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr);
    EXPECT_EQ(ctx.pnextChain[1], ext2Ptr);

    EXPECT_EQ(result, 310); // 10 + 100 + 200
}

// Test: Args-aware with user-provided pNext
TEST(PNextChain, ArgsTuple_UserProvidedPNext) {
    MockContext    ctx;
    const void*    ext1Ptr = nullptr;
    MockExtension2 userExtension{
        .sType     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext     = nullptr,
        .ext2Value = 500,
    };

    uint32_t ext1Value = 100;

    auto modifier = [&](auto&& args, auto&& cont, const void* pNext) {
        auto& [e1, ctxPtr] = args;
        MockExtension1 ext1{
            .sType     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext     = pNext,
            .ext1Value = e1,
        };
        ctxPtr->callOrder.push_back("modifier1");
        ext1Ptr = &ext1;
        return cont(&ext1);
    };

    auto finalFunc = [](uint32_t e1, MockContext* ctxPtr, const void* pNext) {
        MockBaseInfo info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = pNext,
            .value = 10,
        };
        ctxPtr->callOrder.push_back("finalFunc");
        (void)e1;
        return mockVulkanAPI(*ctxPtr, &info);
    };

    auto result = vko::chainPNext(&userExtension, std::tuple{ext1Value, &ctx},
                                  std::tuple{modifier, finalFunc});

    ASSERT_EQ(ctx.callOrder.size(), 2u);
    EXPECT_EQ(ctx.callOrder[0], "modifier1");
    EXPECT_EQ(ctx.callOrder[1], "finalFunc");

    ASSERT_EQ(ctx.pnextChain.size(), 2u);
    EXPECT_EQ(ctx.pnextChain[0], ext1Ptr);
    EXPECT_EQ(ctx.pnextChain[1], &userExtension);

    EXPECT_EQ(result, 610); // 10 + 100 + 500
}
