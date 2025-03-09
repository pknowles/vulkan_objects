// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <vko/functions.hpp>
// Copyright (c) 2025 Pyarelal Knowles, MIT License
// This file was generated from gen_functions.cpp.txt. Do not edit directly.

#include <array>
#include <memory>
#include <ranges>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>

namespace vko
{

struct FindVulkanImpl
{
    FindVulkanImpl()
    {
        #ifdef _WIN32
        std::array libs = {"vulkan-1.dll"};
        #elif defined(__APPLE__)
        std::array libs = {"libvulkan.1.dylib"};
        #else
        std::array libs = {"libvulkan.so.1", "libvulkan.so"};
        #endif
        std::string loaded;
        std::string errors;
        for(auto lib : libs)
        {
            try {
                m_vulkan = DynamicLibrary(lib);
                loaded = lib;
                break;
            } catch(Exception& e) {
                errors += e.what();
                continue;
            }
        }
        if (!m_vulkan)
        {
            std::string msg = "Could not find Vulkan library. Tried ";
            msg += libs[0];
            for(auto lib : libs | std::views::drop(1))
                msg += std::string(", ") + lib;
            throw Exception(msg + ":\n" + errors);
        }
        m_loader = m_vulkan->get<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if(!m_loader)
            throw Exception("vkGetInstanceProcAddr in " + loaded + " was null:\n" + errors);
    }
    std::optional<DynamicLibrary> m_vulkan;
    PFN_vkGetInstanceProcAddr m_loader;
};

VulkanLibrary::VulkanLibrary() : m_impl(std::make_unique<FindVulkanImpl>()) {}
VulkanLibrary::~VulkanLibrary() {}
PFN_vkGetInstanceProcAddr VulkanLibrary::loader() const { return m_impl->m_loader; }

} // namespace vko
