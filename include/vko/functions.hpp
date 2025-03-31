// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/gen_functions.hpp>
#include <memory>

namespace vko
{

// Concept for a global function pointer table. Only does a single test for performance
// reasons.
template <class T>
concept global_commands = requires(T vk) {
    { vk.vkCreateInstance } -> std::same_as<PFN_vkCreateInstance&>;
};

// Concept for a instance function pointer table. Only does a single test for performance
// reasons.
template <class T>
concept instance_commands = requires(T vk) {
    { vk.vkCreateDevice } -> std::same_as<PFN_vkCreateDevice&>;
};

// Concept for a device function pointer table. Only does a single test for performance
// reasons.
template <class T>
concept device_commands = requires(T vk) {
    { vk.vkGetDeviceQueue } -> std::same_as<PFN_vkGetDeviceQueue&>;
};

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
