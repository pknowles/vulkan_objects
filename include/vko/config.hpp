// Copyright (c) 2024 Pyarelal Knowles, MIT License

#pragma once

#if VULKAN_LOADER_VOLK

#include <volk.h>
using InstanceTable = VolkInstanceTable;
using DeviceTable = VolkDeviceTable;

#else

// Define an alternative
#ifndef VULKAN_INSTANCE_TABLE
#error Must set VULKAN_INSTANCE_TABLE or include volk in the cmake configuration
#endif
#ifndef VULKAN_DEVICE_TABLE
#error Must set VULKAN_DEVICE_TABLE or include volk in the cmake configuration
#endif
using InstanceTable = VULKAN_INSTANCE_TABLE;
using DeviceTable = VULKAN_DEVICE_TABLE;

#endif
