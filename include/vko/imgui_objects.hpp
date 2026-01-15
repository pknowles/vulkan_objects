// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <functional>
#include <imgui.h>
#include <utility>
#include <vko/exceptions.hpp>
#include <vko/shortcuts.hpp>

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
    ~Context() { ImGui::DestroyContext(); }
    Context(const Context&)            = delete;
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
    ~ScopedGlfwInit() { ImGui_ImplGlfw_Shutdown(); }
    ScopedGlfwInit(const ScopedGlfwInit&)            = delete;
    ScopedGlfwInit& operator=(const ScopedGlfwInit&) = delete;
};

// RAII wrapper for ImGui_ImplVulkan initialization
// Note: IMGUI_IMPL_VULKAN_NO_PROTOTYPES is expected
class ScopedVulkanInit {
public:
    ScopedVulkanInit() = delete;

    ScopedVulkanInit(PFN_vkGetInstanceProcAddr        vkGetInstanceProcAddr,
                     const ImGui_ImplVulkan_InitInfo& initInfo) {
        if (!ImGui::GetCurrentContext()) {
            throw Exception("ImGui context must be created before ScopedVulkanInit");
        }

        // Load Vulkan functions for ImGui (required when VK_NO_PROTOTYPES is defined)
        auto loader = [](const char* functionName, void* userData) -> PFN_vkVoidFunction {
            auto* params =
                reinterpret_cast<std::pair<PFN_vkGetInstanceProcAddr, VkInstance>*>(userData);
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
    ~ScopedVulkanInit() { ImGui_ImplVulkan_Shutdown(); }
    ScopedVulkanInit(const ScopedVulkanInit&)            = delete;
    ScopedVulkanInit& operator=(const ScopedVulkanInit&) = delete;
};

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
    ScopedFrame(const ScopedFrame&)            = delete;
    ScopedFrame& operator=(const ScopedFrame&) = delete;
};

// RAII wrappers with factory functions that return auto-end objects.
// Usage:
// // RAII style - traditional
// if (auto w = imgui::window("Demo")) {
//     ImGui::Text("Visible content");
// }
//
// // Callback style - compact
// imgui::window("Demo")([&] {
//     ImGui::Text("Visible content");
// });
//
// // Explicit .active() - same as operator()
// imgui::menu("File").active([&] {
//     ImGui::MenuItem("Open");
// });
//
// // .unconditional() - only for windows/child (runs even when collapsed)
// imgui::window("Stats").unconditional([&] {
//     updateStats();  // Runs even if window is collapsed
// });
#ifndef VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS
    #define VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS 1
#endif

#if VULKAN_OBJECTS_IMGUI_SCOPED_BLOCKS

namespace detail {

// Guard implementation with template policy for controlling available methods
template <bool HasBoolReturn, bool HasUnconditional>
class Guard;

// Specialization 1: Begin returns bool, has .unconditional() (window/child - End always called)
template <>
class [[nodiscard]] Guard<true, true> {
    bool m_active;
    void (*m_end)();

public:
    Guard(bool active, void (*end)())
        : m_active(active)
        , m_end(end) {}
    ~Guard() {
        if (m_end)
            m_end();
    }

    explicit operator bool() const { return m_active; }

    template <class Fn>
    void active(Fn&& fn) {
        if (m_active)
            std::forward<Fn>(fn)();
    }

    template <class Fn>
    void unconditional(Fn&& fn) {
        std::forward<Fn>(fn)();
    }

    template <class Fn>
    void operator()(Fn&& fn) {
        active(std::forward<Fn>(fn));
    }

    Guard(Guard&&)            = delete;
    Guard& operator=(Guard&&) = delete;
};

// Specialization 2: Begin returns bool, no .unconditional() (menu/combo/table - End conditional)
template <>
class [[nodiscard]] Guard<true, false> {
    bool m_active;
    void (*m_end)();

public:
    Guard(bool active, void (*end)())
        : m_active(active)
        , m_end(active ? end : nullptr) {}
    ~Guard() {
        if (m_end)
            m_end();
    }

    explicit operator bool() const { return m_active; }

    template <class Fn>
    void active(Fn&& fn) {
        if (m_active)
            std::forward<Fn>(fn)();
    }

    template <class Fn>
    void operator()(Fn&& fn) {
        active(std::forward<Fn>(fn));
    }

    Guard(Guard&&)            = delete;
    Guard& operator=(Guard&&) = delete;
};

// Specialization 3: Begin returns void (group/disabled/push-pop - no bool to check)
template <>
class [[nodiscard]] Guard<false, false> {
    std::function<void()> m_end;

public:
    template <class Fn>
    explicit Guard(Fn&& end)
        : m_end(std::forward<Fn>(end)) {}
    ~Guard() {
        if (m_end)
            m_end();
    }

    template <class Fn>
    void active(Fn&& fn) {
        std::forward<Fn>(fn)();
    }

    template <class Fn>
    void operator()(Fn&& fn) {
        active(std::forward<Fn>(fn));
    }

    Guard(Guard&&)            = delete;
    Guard& operator=(Guard&&) = delete;
};

} // namespace detail

// Type aliases for clarity
using WindowGuard  = detail::Guard<true, true>;   // bool Begin, has .unconditional()
using ElementGuard = detail::Guard<true, false>;  // bool Begin, no .unconditional()
using ScopeGuard   = detail::Guard<false, false>; // void Begin, no .unconditional()

// Factory functions for ImGui elements
// Windows and child windows (End always called)
[[nodiscard]] inline WindowGuard window(const char* name, bool* p_open = nullptr,
                                        ImGuiWindowFlags flags = 0) {
    if (p_open && !*p_open) {
        return {false, nullptr}; // Skip Begin/End entirely if already closed
    }
    return {ImGui::Begin(name, p_open, flags), &ImGui::End};
}

[[nodiscard]] inline WindowGuard child(const char* str_id, const ImVec2& size = ImVec2(0, 0),
                                       ImGuiChildFlags  child_flags  = 0,
                                       ImGuiWindowFlags window_flags = 0) {
    return {ImGui::BeginChild(str_id, size, child_flags, window_flags), &ImGui::EndChild};
}

[[nodiscard]] inline WindowGuard child(ImGuiID id, const ImVec2& size = ImVec2(0, 0),
                                       ImGuiChildFlags  child_flags  = 0,
                                       ImGuiWindowFlags window_flags = 0) {
    return {ImGui::BeginChild(id, size, child_flags, window_flags), &ImGui::EndChild};
}

// Menus and menu bars (End only if opened)
[[nodiscard]] inline ElementGuard menuBar() {
    bool opened = ImGui::BeginMenuBar();
    return {opened, &ImGui::EndMenuBar};
}

[[nodiscard]] inline ElementGuard mainMenuBar() {
    bool opened = ImGui::BeginMainMenuBar();
    return {opened, &ImGui::EndMainMenuBar};
}

[[nodiscard]] inline ElementGuard menu(const char* label, bool enabled = true) {
    bool opened = ImGui::BeginMenu(label, enabled);
    return {opened, &ImGui::EndMenu};
}

// Combos and list boxes (End only if opened)
[[nodiscard]] inline ElementGuard combo(const char* label, const char* preview_value,
                                        ImGuiComboFlags flags = 0) {
    bool opened = ImGui::BeginCombo(label, preview_value, flags);
    return {opened, &ImGui::EndCombo};
}

[[nodiscard]] inline ElementGuard listBox(const char* label, const ImVec2& size = ImVec2(0, 0)) {
    bool opened = ImGui::BeginListBox(label, size);
    return {opened, &ImGui::EndListBox};
}

// Popups (End only if opened)
[[nodiscard]] inline ElementGuard popup(const char* str_id, ImGuiWindowFlags flags = 0) {
    bool opened = ImGui::BeginPopup(str_id, flags);
    return {opened, &ImGui::EndPopup};
}

[[nodiscard]] inline ElementGuard popupModal(const char* name, bool* p_open = nullptr,
                                             ImGuiWindowFlags flags = 0) {
    if (p_open && !*p_open) {
        return {false, nullptr}; // Skip if already closed
    }
    bool opened = ImGui::BeginPopupModal(name, p_open, flags);
    return {opened, &ImGui::EndPopup};
}

[[nodiscard]] inline ElementGuard popupContextItem(const char*     str_id = nullptr,
                                                   ImGuiPopupFlags flags  = 0) {
    bool opened = ImGui::BeginPopupContextItem(str_id, flags);
    return {opened, &ImGui::EndPopup};
}

[[nodiscard]] inline ElementGuard popupContextWindow(const char*     str_id = nullptr,
                                                     ImGuiPopupFlags flags  = 0) {
    bool opened = ImGui::BeginPopupContextWindow(str_id, flags);
    return {opened, &ImGui::EndPopup};
}

[[nodiscard]] inline ElementGuard popupContextVoid(const char*     str_id = nullptr,
                                                   ImGuiPopupFlags flags  = 0) {
    bool opened = ImGui::BeginPopupContextVoid(str_id, flags);
    return {opened, &ImGui::EndPopup};
}

// Tables, tab bars, and tab items (End only if opened)
[[nodiscard]] inline ElementGuard table(const char* str_id, int columns, ImGuiTableFlags flags = 0,
                                        const ImVec2& outer_size  = ImVec2(0, 0),
                                        float         inner_width = 0.0f) {
    bool opened = ImGui::BeginTable(str_id, columns, flags, outer_size, inner_width);
    return {opened, &ImGui::EndTable};
}

[[nodiscard]] inline ElementGuard tabBar(const char* str_id, ImGuiTabBarFlags flags = 0) {
    bool opened = ImGui::BeginTabBar(str_id, flags);
    return {opened, &ImGui::EndTabBar};
}

[[nodiscard]] inline ElementGuard tabItem(const char* label, bool* p_open = nullptr,
                                          ImGuiTabItemFlags flags = 0) {
    if (p_open && !*p_open) {
        return {false, nullptr}; // Skip if already closed
    }
    bool opened = ImGui::BeginTabItem(label, p_open, flags);
    return {opened, &ImGui::EndTabItem};
}

// Tooltips (no bool return)
[[nodiscard]] inline ScopeGuard tooltip() {
    ImGui::BeginTooltip();
    return ScopeGuard(&ImGui::EndTooltip);
}

[[nodiscard]] inline ElementGuard itemTooltip() {
    bool opened = ImGui::BeginItemTooltip();
    return {opened, &ImGui::EndTooltip};
}

// Drag and drop (End only if active)
[[nodiscard]] inline ElementGuard dragDropSource(ImGuiDragDropFlags flags = 0) {
    bool active = ImGui::BeginDragDropSource(flags);
    return {active, &ImGui::EndDragDropSource};
}

[[nodiscard]] inline ElementGuard dragDropTarget() {
    bool active = ImGui::BeginDragDropTarget();
    return {active, &ImGui::EndDragDropTarget};
}

// Groups and disabled state (no bool return)
[[nodiscard]] inline ScopeGuard group() {
    ImGui::BeginGroup();
    return ScopeGuard(&ImGui::EndGroup);
}

[[nodiscard]] inline ScopeGuard disabled(bool is_disabled = true) {
    ImGui::BeginDisabled(is_disabled);
    return ScopeGuard(&ImGui::EndDisabled);
}

// Helper wrappers for ImGui Pop functions that take optional count parameters
namespace detail {
inline void popStyleColor() { ImGui::PopStyleColor(); }
inline void popStyleVar() { ImGui::PopStyleVar(); }
} // namespace detail

// Push/pop style and state stacks (no bool return)
[[nodiscard]] inline ScopeGuard font(ImFont* font) {
    ImGui::PushFont(font);
    return ScopeGuard(&ImGui::PopFont);
}

[[nodiscard]] inline ScopeGuard styleColor(ImGuiCol idx, ImU32 col) {
    ImGui::PushStyleColor(idx, col);
    return ScopeGuard(&detail::popStyleColor);
}

[[nodiscard]] inline ScopeGuard styleColor(ImGuiCol idx, const ImVec4& col) {
    ImGui::PushStyleColor(idx, col);
    return ScopeGuard(&detail::popStyleColor);
}

[[nodiscard]] inline ScopeGuard styleVar(ImGuiStyleVar idx, float val) {
    ImGui::PushStyleVar(idx, val);
    return ScopeGuard(&detail::popStyleVar);
}

[[nodiscard]] inline ScopeGuard styleVar(ImGuiStyleVar idx, const ImVec2& val) {
    ImGui::PushStyleVar(idx, val);
    return ScopeGuard(&detail::popStyleVar);
}

[[nodiscard]] inline ScopeGuard styleVarX(ImGuiStyleVar idx, float val) {
    ImGui::PushStyleVarX(idx, val);
    return ScopeGuard(&detail::popStyleVar);
}

[[nodiscard]] inline ScopeGuard styleVarY(ImGuiStyleVar idx, float val) {
    ImGui::PushStyleVarY(idx, val);
    return ScopeGuard(&detail::popStyleVar);
}

[[nodiscard]] inline ScopeGuard itemFlag(ImGuiItemFlags option, bool enabled) {
    ImGui::PushItemFlag(option, enabled);
    return ScopeGuard(&ImGui::PopItemFlag);
}

[[nodiscard]] inline ScopeGuard itemWidth(float item_width) {
    ImGui::PushItemWidth(item_width);
    return ScopeGuard(&ImGui::PopItemWidth);
}

[[nodiscard]] inline ScopeGuard textWrapPos(float wrap_pos_x = 0.0f) {
    ImGui::PushTextWrapPos(wrap_pos_x);
    return ScopeGuard(&ImGui::PopTextWrapPos);
}

[[nodiscard]] inline ScopeGuard id(const char* str_id) {
    ImGui::PushID(str_id);
    return ScopeGuard(&ImGui::PopID);
}

[[nodiscard]] inline ScopeGuard id(const void* ptr_id) {
    ImGui::PushID(ptr_id);
    return ScopeGuard(&ImGui::PopID);
}

[[nodiscard]] inline ScopeGuard id(int int_id) {
    ImGui::PushID(int_id);
    return ScopeGuard(&ImGui::PopID);
}

[[nodiscard]] inline ScopeGuard clipRect(const ImVec2& min, const ImVec2& max,
                                         bool intersect_with_current_clip_rect = false) {
    ImGui::PushClipRect(min, max, intersect_with_current_clip_rect);
    return ScopeGuard(&ImGui::PopClipRect);
}

[[nodiscard]] inline ScopeGuard buttonRepeat(bool repeat = true) {
    ImGui::PushButtonRepeat(repeat);
    return ScopeGuard(&ImGui::PopButtonRepeat);
}

[[nodiscard]] inline ScopeGuard tree(const char* str_id) {
    ImGui::TreePush(str_id);
    return ScopeGuard(&ImGui::TreePop);
}

[[nodiscard]] inline ScopeGuard tree(const void* ptr_id) {
    ImGui::TreePush(ptr_id);
    return ScopeGuard(&ImGui::TreePop);
}

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
