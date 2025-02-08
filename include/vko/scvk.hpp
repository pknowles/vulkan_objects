// Copyright (c) 2024 Pyarelal Knowles, MIT License

#pragma once

#include <scvk/scvk_base.h>

namespace scvk {

class Instance 
{
public:
    Instance(VkApplicationInfo)
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


}