// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <vko/exceptions.hpp>
#include <vko/shortcuts.hpp>

namespace vko {
namespace imgui {

// Generic RAII wrapper for paired begin/end function calls
template<auto BeginFunc, auto EndFunc>
class ScopedBlock {
public:
    [[nodiscard]] ScopedBlock() { BeginFunc(); }
    ~ScopedBlock() { EndFunc(); }
    ScopedBlock(const ScopedBlock&) = delete;
    ScopedBlock& operator=(const ScopedBlock&) = delete;
};

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
            auto* params = reinterpret_cast<std::pair<PFN_vkGetInstanceProcAddr, VkInstance>*>(userData);
            // Try instance-specific functions first, then fall back to global functions
            PFN_vkVoidFunction func = params->first(params->second, functionName);
            if (!func)
                func = params->first(VK_NULL_HANDLE, functionName);
            return func;
        };
        auto loaderUserData = std::make_pair(vkGetInstanceProcAddr, initInfo.Instance);
        if (!ImGui_ImplVulkan_LoadFunctions(initInfo.ApiVersion, loader, &loaderUserData)) {
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

// Experimental. Likely a waste of typing and extra work to maintain
#ifndef VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS
#define VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS 1
#endif

#if VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS
// RAII wrapper for ImGui frame (NewFrame/EndFrame)
// Note: Backend NewFrame functions (ImGui_ImplVulkan_NewFrame, ImGui_ImplGlfw_NewFrame, etc.)
// should be called BEFORE creating ScopedFrame. They don't have corresponding End functions.
// Usage:
//   ImGui_ImplVulkan_NewFrame();
//   ImGui_ImplGlfw_NewFrame();
//   ScopedFrame frame;
//   // ... UI code ...
class [[nodiscard]] ScopedFrame {
public:
    ScopedFrame() { ImGui::NewFrame(); }
    ~ScopedFrame() { ImGui::EndFrame(); }
    ScopedFrame(const ScopedFrame&) = delete;
    ScopedFrame& operator=(const ScopedFrame&) = delete;
};

// RAII wrapper for ImGui window (Begin/End)
// Note: End() is ALWAYS called regardless of Begin() return value (ImGui requirement)
class [[nodiscard]] ScopedWindow {
    bool visible;
public:
    ScopedWindow(const char* name, bool* p_open = nullptr, ImGuiWindowFlags flags = 0)
        : visible(ImGui::Begin(name, p_open, flags)) {}
    ~ScopedWindow() { ImGui::End(); }
    explicit operator bool() const { return visible; }
    ScopedWindow(const ScopedWindow&) = delete;
    ScopedWindow& operator=(const ScopedWindow&) = delete;
};

// RAII wrapper for ImGui child window (BeginChild/EndChild)
// Note: EndChild() is ALWAYS called regardless of BeginChild() return value (ImGui requirement)
class [[nodiscard]] ScopedChildWindow {
    bool visible;
public:
    ScopedChildWindow(const char* str_id, const ImVec2& size = ImVec2(0, 0), 
                      ImGuiChildFlags child_flags = 0, ImGuiWindowFlags window_flags = 0)
        : visible(ImGui::BeginChild(str_id, size, child_flags, window_flags)) {}
    ScopedChildWindow(ImGuiID id, const ImVec2& size = ImVec2(0, 0),
                      ImGuiChildFlags child_flags = 0, ImGuiWindowFlags window_flags = 0)
        : visible(ImGui::BeginChild(id, size, child_flags, window_flags)) {}
    ~ScopedChildWindow() { ImGui::EndChild(); }
    explicit operator bool() const { return visible; }
    ScopedChildWindow(const ScopedChildWindow&) = delete;
    ScopedChildWindow& operator=(const ScopedChildWindow&) = delete;
};

// RAII wrapper for ImGui group (BeginGroup/EndGroup)
class [[nodiscard]] ScopedGroup {
public:
    ScopedGroup() { ImGui::BeginGroup(); }
    ~ScopedGroup() { ImGui::EndGroup(); }
    ScopedGroup(const ScopedGroup&) = delete;
    ScopedGroup& operator=(const ScopedGroup&) = delete;
};

// RAII wrapper for ImGui combo (BeginCombo/EndCombo)
class [[nodiscard]] ScopedCombo {
    bool visible;
public:
    ScopedCombo(const char* label, const char* preview_value, ImGuiComboFlags flags = 0)
        : visible(ImGui::BeginCombo(label, preview_value, flags)) {}
    ~ScopedCombo() { if (visible) ImGui::EndCombo(); }
    explicit operator bool() const { return visible; }
    ScopedCombo(const ScopedCombo&) = delete;
    ScopedCombo& operator=(const ScopedCombo&) = delete;
};

// RAII wrapper for ImGui list box (BeginListBox/EndListBox)
class [[nodiscard]] ScopedListBox {
    bool visible;
public:
    ScopedListBox(const char* label, const ImVec2& size = ImVec2(0, 0))
        : visible(ImGui::BeginListBox(label, size)) {}
    ~ScopedListBox() { if (visible) ImGui::EndListBox(); }
    explicit operator bool() const { return visible; }
    ScopedListBox(const ScopedListBox&) = delete;
    ScopedListBox& operator=(const ScopedListBox&) = delete;
};

// RAII wrapper for ImGui menu bar (BeginMenuBar/EndMenuBar)
class [[nodiscard]] ScopedMenuBar {
    bool visible;
public:
    ScopedMenuBar() : visible(ImGui::BeginMenuBar()) {}
    ~ScopedMenuBar() { if (visible) ImGui::EndMenuBar(); }
    explicit operator bool() const { return visible; }
    ScopedMenuBar(const ScopedMenuBar&) = delete;
    ScopedMenuBar& operator=(const ScopedMenuBar&) = delete;
};

// RAII wrapper for ImGui main menu bar (BeginMainMenuBar/EndMainMenuBar)
class [[nodiscard]] ScopedMainMenuBar {
    bool visible;
public:
    ScopedMainMenuBar() : visible(ImGui::BeginMainMenuBar()) {}
    ~ScopedMainMenuBar() { if (visible) ImGui::EndMainMenuBar(); }
    explicit operator bool() const { return visible; }
    ScopedMainMenuBar(const ScopedMainMenuBar&) = delete;
    ScopedMainMenuBar& operator=(const ScopedMainMenuBar&) = delete;
};

// RAII wrapper for ImGui menu (BeginMenu/EndMenu)
class [[nodiscard]] ScopedMenu {
    bool visible;
public:
    ScopedMenu(const char* label, bool enabled = true)
        : visible(ImGui::BeginMenu(label, enabled)) {}
    ~ScopedMenu() { if (visible) ImGui::EndMenu(); }
    explicit operator bool() const { return visible; }
    ScopedMenu(const ScopedMenu&) = delete;
    ScopedMenu& operator=(const ScopedMenu&) = delete;
};

// RAII wrapper for ImGui tooltip (BeginTooltip/EndTooltip)
class [[nodiscard]] ScopedTooltip {
public:
    ScopedTooltip() { ImGui::BeginTooltip(); }
    ~ScopedTooltip() { ImGui::EndTooltip(); }
    ScopedTooltip(const ScopedTooltip&) = delete;
    ScopedTooltip& operator=(const ScopedTooltip&) = delete;
};

// RAII wrapper for ImGui item tooltip (BeginItemTooltip/EndTooltip)
class [[nodiscard]] ScopedItemTooltip {
    bool visible;
public:
    ScopedItemTooltip() : visible(ImGui::BeginItemTooltip()) {}
    ~ScopedItemTooltip() { if (visible) ImGui::EndTooltip(); }
    explicit operator bool() const { return visible; }
    ScopedItemTooltip(const ScopedItemTooltip&) = delete;
    ScopedItemTooltip& operator=(const ScopedItemTooltip&) = delete;
};

// RAII wrapper for ImGui popup (BeginPopup/EndPopup)
class [[nodiscard]] ScopedPopup {
    bool visible;
public:
    ScopedPopup(const char* str_id, ImGuiWindowFlags flags = 0)
        : visible(ImGui::BeginPopup(str_id, flags)) {}
    ~ScopedPopup() { if (visible) ImGui::EndPopup(); }
    explicit operator bool() const { return visible; }
    ScopedPopup(const ScopedPopup&) = delete;
    ScopedPopup& operator=(const ScopedPopup&) = delete;
};

// RAII wrapper for ImGui modal popup (BeginPopupModal/EndPopup)
class [[nodiscard]] ScopedPopupModal {
    bool visible;
public:
    ScopedPopupModal(const char* name, bool* p_open = nullptr, ImGuiWindowFlags flags = 0)
        : visible(ImGui::BeginPopupModal(name, p_open, flags)) {}
    ~ScopedPopupModal() { if (visible) ImGui::EndPopup(); }
    explicit operator bool() const { return visible; }
    ScopedPopupModal(const ScopedPopupModal&) = delete;
    ScopedPopupModal& operator=(const ScopedPopupModal&) = delete;
};

// RAII wrapper for ImGui context item popup (BeginPopupContextItem/EndPopup)
class [[nodiscard]] ScopedPopupContextItem {
    bool visible;
public:
    ScopedPopupContextItem(const char* str_id = nullptr, ImGuiPopupFlags flags = 0)
        : visible(ImGui::BeginPopupContextItem(str_id, flags)) {}
    ~ScopedPopupContextItem() { if (visible) ImGui::EndPopup(); }
    explicit operator bool() const { return visible; }
    ScopedPopupContextItem(const ScopedPopupContextItem&) = delete;
    ScopedPopupContextItem& operator=(const ScopedPopupContextItem&) = delete;
};

// RAII wrapper for ImGui context window popup (BeginPopupContextWindow/EndPopup)
class [[nodiscard]] ScopedPopupContextWindow {
    bool visible;
public:
    ScopedPopupContextWindow(const char* str_id = nullptr, ImGuiPopupFlags flags = 0)
        : visible(ImGui::BeginPopupContextWindow(str_id, flags)) {}
    ~ScopedPopupContextWindow() { if (visible) ImGui::EndPopup(); }
    explicit operator bool() const { return visible; }
    ScopedPopupContextWindow(const ScopedPopupContextWindow&) = delete;
    ScopedPopupContextWindow& operator=(const ScopedPopupContextWindow&) = delete;
};

// RAII wrapper for ImGui context void popup (BeginPopupContextVoid/EndPopup)
class [[nodiscard]] ScopedPopupContextVoid {
    bool visible;
public:
    ScopedPopupContextVoid(const char* str_id = nullptr, ImGuiPopupFlags flags = 0)
        : visible(ImGui::BeginPopupContextVoid(str_id, flags)) {}
    ~ScopedPopupContextVoid() { if (visible) ImGui::EndPopup(); }
    explicit operator bool() const { return visible; }
    ScopedPopupContextVoid(const ScopedPopupContextVoid&) = delete;
    ScopedPopupContextVoid& operator=(const ScopedPopupContextVoid&) = delete;
};

// RAII wrapper for ImGui table (BeginTable/EndTable)
class [[nodiscard]] ScopedTable {
    bool visible;
public:
    ScopedTable(const char* str_id, int columns, ImGuiTableFlags flags = 0,
                const ImVec2& outer_size = ImVec2(0, 0), float inner_width = 0.0f)
        : visible(ImGui::BeginTable(str_id, columns, flags, outer_size, inner_width)) {}
    ~ScopedTable() { if (visible) ImGui::EndTable(); }
    explicit operator bool() const { return visible; }
    ScopedTable(const ScopedTable&) = delete;
    ScopedTable& operator=(const ScopedTable&) = delete;
};

// RAII wrapper for ImGui tab bar (BeginTabBar/EndTabBar)
class [[nodiscard]] ScopedTabBar {
    bool visible;
public:
    ScopedTabBar(const char* str_id, ImGuiTabBarFlags flags = 0)
        : visible(ImGui::BeginTabBar(str_id, flags)) {}
    ~ScopedTabBar() { if (visible) ImGui::EndTabBar(); }
    explicit operator bool() const { return visible; }
    ScopedTabBar(const ScopedTabBar&) = delete;
    ScopedTabBar& operator=(const ScopedTabBar&) = delete;
};

// RAII wrapper for ImGui tab item (BeginTabItem/EndTabItem)
class [[nodiscard]] ScopedTabItem {
    bool visible;
public:
    ScopedTabItem(const char* label, bool* p_open = nullptr, ImGuiTabItemFlags flags = 0)
        : visible(ImGui::BeginTabItem(label, p_open, flags)) {}
    ~ScopedTabItem() { if (visible) ImGui::EndTabItem(); }
    explicit operator bool() const { return visible; }
    ScopedTabItem(const ScopedTabItem&) = delete;
    ScopedTabItem& operator=(const ScopedTabItem&) = delete;
};

// RAII wrapper for ImGui drag drop source (BeginDragDropSource/EndDragDropSource)
class [[nodiscard]] ScopedDragDropSource {
    bool visible;
public:
    ScopedDragDropSource(ImGuiDragDropFlags flags = 0)
        : visible(ImGui::BeginDragDropSource(flags)) {}
    ~ScopedDragDropSource() { if (visible) ImGui::EndDragDropSource(); }
    explicit operator bool() const { return visible; }
    ScopedDragDropSource(const ScopedDragDropSource&) = delete;
    ScopedDragDropSource& operator=(const ScopedDragDropSource&) = delete;
};

// RAII wrapper for ImGui drag drop target (BeginDragDropTarget/EndDragDropTarget)
class [[nodiscard]] ScopedDragDropTarget {
    bool visible;
public:
    ScopedDragDropTarget()
        : visible(ImGui::BeginDragDropTarget()) {}
    ~ScopedDragDropTarget() { if (visible) ImGui::EndDragDropTarget(); }
    explicit operator bool() const { return visible; }
    ScopedDragDropTarget(const ScopedDragDropTarget&) = delete;
    ScopedDragDropTarget& operator=(const ScopedDragDropTarget&) = delete;
};

// RAII wrapper for ImGui disabled state (BeginDisabled/EndDisabled)
class [[nodiscard]] ScopedDisabled {
public:
    ScopedDisabled(bool disabled = true) { ImGui::BeginDisabled(disabled); }
    ~ScopedDisabled() { ImGui::EndDisabled(); }
    ScopedDisabled(const ScopedDisabled&) = delete;
    ScopedDisabled& operator=(const ScopedDisabled&) = delete;
};

#endif // VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS

// RAII wrapper for ImGui texture descriptor set
class Texture {
public:
    Texture(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout) {
        // Explicit global initialization check, but then a null pointer crash
        // happens anyway, so this is just a minor perf hit for the happy path
        if (!ImGui::GetCurrentContext()) {
            throw Exception("ImGui context not initialized - create a vko::imgui::Context first");
        }

        m_descriptor = ImGui_ImplVulkan_AddTexture(sampler, imageView, imageLayout);

        if (m_descriptor == VK_NULL_HANDLE) {
            throw Exception(
                "ImGui_ImplVulkan_AddTexture failed - ensure ImGui_ImplVulkan_Init() was called");
        }
    }

    ~Texture() {
        if (m_descriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(m_descriptor);
        }
    }

    // No copy
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    // Move only
    Texture(Texture&& other) noexcept
        : m_descriptor(other.m_descriptor) {
        other.m_descriptor = VK_NULL_HANDLE;
    }

    Texture& operator=(Texture&& other) noexcept {
        if (m_descriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(m_descriptor);
        }
        m_descriptor       = other.m_descriptor;
        other.m_descriptor = VK_NULL_HANDLE;
        return *this;
    }

    operator VkDescriptorSet() const& { return m_descriptor; }
    operator ImTextureID() const {
        return reinterpret_cast<ImTextureID>(m_descriptor);
    } // for imgui compatibility
    VkDescriptorSet        object() const { return m_descriptor; }
    const VkDescriptorSet* ptr() const { return &m_descriptor; }
    bool engaged() const { return m_descriptor != VK_NULL_HANDLE; } // for moved-from objects

private:
    VkDescriptorSet m_descriptor = VK_NULL_HANDLE;
};

} // namespace imgui
} // namespace vko

