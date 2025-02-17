// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include "vulkan/vulkan_core.h"
#include <gtest/gtest.h>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>
#include <vko/handles.hpp>

TEST(Integration, Init)
{
    vko::VulkanLibrary library;
    vko::GlobalCommands globalCommands(library.loader());
    VkInstanceCreateInfo instanceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = nullptr,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };
    vko::Instance instance(vko::InstanceHandle(globalCommands, instanceCreateInfo), library.loader());
    EXPECT_EQ(1, 2);
}
