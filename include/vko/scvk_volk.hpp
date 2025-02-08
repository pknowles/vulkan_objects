// Copyright (c) 2024 Pyarelal Knowles, MIT License

#pragma once

#if !VULKAN_LOADER_VOLK
#error Must include volk in the cmake configuration to use scvk_volk.hpp
#endif

#include <volk.h>

// Workaround for https://github.com/zeux/volk/issues/137
class ScopedVulkanInstance
{
    ScopedVulkanInstance() { volkInitialize(); }
    ~ScopedVulkanInstance() { volkFinalize(); }
};

class Instance 
{
public:
    Instance()
    {
        volkLoadInstanceTable(&m_instanceTable, m_instance);
    }

private:
    VolkInstanceTable m_instanceTable;
    VkInstance m_instance;
};

class Device
{
public:
    Instance()
    {
        volkLoadInstanceTable(&m_instanceTable, m_instance);
    }

private:
    VolkDeviceTable m_deviceTable;
    VkDevice        m_device;
};

struct Context
{
    Context()
    {
    }

    Instance m_instance;
    Device   m_device;
};
