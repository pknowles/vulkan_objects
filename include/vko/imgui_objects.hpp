// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_glfw.h>
#include <vko/exceptions.hpp>

namespace vko {
namespace imgui {

// RAII wrapper for ImGui context
class Context {
public:
    Context() {
        if (!IMGUI_CHECKVERSION()) {
            throw Exception("ImGui version check failed");
        }
        ImGui::CreateContext();
    }
    ~Context() {
        ImGui::DestroyContext();
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};

// RAII wrapper for ImGui GLFW initialization
class ScopedGlfwInit {
public:
    ScopedGlfwInit(GLFWwindow* window, bool installCallbacks = true) {
        if (!ImGui::GetCurrentContext()) {
            throw Exception("ImGui context must be created before ScopedGlfwInit");
        }
        ImGui_ImplGlfw_InitForVulkan(window, installCallbacks);
    }
    ~ScopedGlfwInit() {
        ImGui_ImplGlfw_Shutdown();
    }
    ScopedGlfwInit(const ScopedGlfwInit&) = delete;
    ScopedGlfwInit& operator=(const ScopedGlfwInit&) = delete;
};

// RAII wrapper for ImGui_ImplVulkan initialization
// Note: IMGUI_IMPL_VULKAN_NO_PROTOTYPES is expected
class ScopedVulkanInit {
public:
    ScopedVulkanInit() = delete;
    
    ScopedVulkanInit(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr, const ImGui_ImplVulkan_InitInfo& initInfo) {
        if (!ImGui::GetCurrentContext()) {
            throw Exception("ImGui context must be created before ScopedVulkanInit");
        }
        
        // Load Vulkan functions for ImGui (required when VK_NO_PROTOTYPES is defined)
        auto loader = [](const char* functionName, void* userData) -> PFN_vkVoidFunction {
            auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(userData);
            return vkGetInstanceProcAddr(VK_NULL_HANDLE, functionName);
        };
        if (!ImGui_ImplVulkan_LoadFunctions(initInfo.ApiVersion, loader, reinterpret_cast<void*>(vkGetInstanceProcAddr))) {
            throw Exception("ImGui_ImplVulkan_LoadFunctions failed");
        }
        
        // Const cast non-const ImGui_ImplVulkan_Init is probably just a mistake
        if (!ImGui_ImplVulkan_Init(const_cast<ImGui_ImplVulkan_InitInfo*>(&initInfo))) {
            throw Exception("ImGui_ImplVulkan_Init failed");
        }
    }
    ~ScopedVulkanInit() {
        ImGui_ImplVulkan_Shutdown();
    }
    ScopedVulkanInit(const ScopedVulkanInit&) = delete;
    ScopedVulkanInit& operator=(const ScopedVulkanInit&) = delete;
};

} // namespace imgui
} // namespace vko

