// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/gen_functions.hpp>
#include <memory>

namespace vko
{

struct VulkanLibrary
{
public:
    VulkanLibrary();
    ~VulkanLibrary();
    PFN_vkGetInstanceProcAddr loader() const;

private:
    std::unique_ptr<struct FindVulkanImpl> m_impl;
};

} // namespace vko
