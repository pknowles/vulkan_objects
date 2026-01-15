// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <vko/functions.hpp>
// Copyright (c) 2025 Pyarelal Knowles, MIT License
// This file was generated from gen_functions.cpp.txt. Do not edit directly.

#include <array>
#include <memory>
#include <ranges>
#include <vko/dynamic_library.hpp>
#include <vko/functions.hpp>

#if _WIN32
    #include <errhandlingapi.h>
    #include <processenv.h>
#else
    #include <stdlib.h>
    #include <string.h>
#endif

namespace vko {

struct FindVulkanImpl {
    FindVulkanImpl() {
#if defined(VVL_DEVELOP_PATH)
    #if _WIN32
        if (GetEnvironmentVariableA("VK_ADD_LAYER_PATH", nullptr, 0) == ERROR_ENVVAR_NOT_FOUND &&
            !SetEnvironmentVariableA("VK_ADD_LAYER_PATH", VVL_DEVELOP_PATH)) {
            fprintf(stderr, "Failed to set VK_ADD_LAYER_PATH: %d\n", GetLastError());
        }
    #else
        // TODO: append rather than replace?
        if (getenv("VK_ADD_LAYER_PATH") == nullptr) {
            if (int result = setenv("VK_ADD_LAYER_PATH", VVL_DEVELOP_PATH, 0); result != 0) {
                fprintf(stderr, "Failed to set VK_ADD_LAYER_PATH: %s\n", strerror(errno));
            }
        }
    #endif
#endif

#ifdef _WIN32
        std::array libs = {"vulkan-1.dll"};
#elif defined(__APPLE__)
        std::array libs = {"libvulkan.1.dylib"};
#else
        std::array libs = {"libvulkan.so.1", "libvulkan.so"};
#endif
        std::string loaded;
        std::string errors;
        for (auto lib : libs) {
            try {
                m_vulkan = DynamicLibrary(lib);
                loaded   = lib;
                break;
            } catch (Exception& e) {
                errors += e.what();
                continue;
            }
        }
        if (!m_vulkan) {
            std::string msg = "Could not find Vulkan library. Tried ";
            msg += libs[0];
            for (auto lib : libs | std::views::drop(1))
                msg += std::string(", ") + lib;
            throw Exception(msg + ":\n" + errors);
        }
        m_loader = m_vulkan->get<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if (!m_loader)
            throw Exception("vkGetInstanceProcAddr in " + loaded + " was null:\n" + errors);
    }
    std::optional<DynamicLibrary> m_vulkan;
    PFN_vkGetInstanceProcAddr     m_loader;
};

VulkanLibrary::VulkanLibrary()
    : m_impl(std::make_unique<FindVulkanImpl>()) {}
VulkanLibrary::~VulkanLibrary() {}
PFN_vkGetInstanceProcAddr VulkanLibrary::loader() const { return m_impl->m_loader; }

} // namespace vko
