// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/adapters.hpp>
#include <vko/exceptions.hpp>
#include <vko/functions.hpp>
#include <vulkan/vulkan_core.h>

namespace vko {

template <class T>
concept instance_and_commands =
    std::constructible_from<VkInstance, const T&> && instance_commands<T>;

template <class T>
concept device_and_commands = std::constructible_from<VkDevice, const T&> && device_commands<T>;

template <class Handle>
struct handle_traits;

template <class T, class CreateFunc>
struct CreateHandle;

// Argument pack for destroying a handle. Similar to an object keeping a
// reference to its allocator. std::function could work too, but I have a
// premature optimization hunch this could be cheaper (naturally, untested).
template <class T>
struct DestroyFunc {
    // In most cases the parent is the first parameter of the destroy function
    using Parent = typename handle_traits<T>::destroy_first_param;

    // template <class ParentAndCommands>
    //     requires std::constructible_from<Parent, ParentAndCommands>
    // DestroyFunc(const ParentAndCommands& vk)
    //     : DestroyFunc(vk, vk) {}

    template <class CommandTable>
    DestroyFunc(const CommandTable& table, Parent parent)
        : destroy(handle_traits<T>::template destroy_command(table))
        , parent(parent) {}
    void operator()(T handle) const { destroy(parent, handle, nullptr); }

    // Copy the function pointer rather than take a reference to the function
    // table itself and risk a dangling pointer
    handle_traits<T>::destroy_t destroy;

    // The owning object. This could be a weak_ptr to be extra safe, but the
    // overhead makes it a step too far.
    Parent parent;
};

// Template class to hold a single vulkan handle
template <class T, class CreateFunc = void>
class Handle;

// Default specialization for constructing handles
template <class T, class CreateFunc>
class Handle {
public:
    using Parent     = typename handle_traits<T>::destroy_first_param;
    using CreateInfo = typename CreateHandle<T, CreateFunc>::CreateInfo;

    template <class ParentAndCommands, class... Args>
        requires std::constructible_from<Parent, const ParentAndCommands&>
    Handle(const ParentAndCommands& vk, const CreateInfo& createInfo, const Args&... args)
        : Handle(vk, static_cast<Parent>(vk), createInfo, args...) {}

    template <class Commands, class... Args>
    Handle(const Commands& vk, Parent parent, const CreateInfo& createInfo, const Args&... args)
        : m_handle(CreateHandle<T, CreateFunc>()(vk, parent, createInfo, args...))
        , m_destroy(DestroyFunc<T>(vk, parent, args...)) {}
    ~Handle() { destroy(); }
    Handle(const Handle& other) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(std::move(other.m_handle))
        , m_destroy(std::move(other.m_destroy)) {
        other.m_handle = VK_NULL_HANDLE;
    }
    Handle& operator=(const Handle& other) = delete;
    Handle& operator=(Handle&& other) {
        destroy();
        m_destroy      = std::move(other.m_destroy);
        m_handle       = std::move(other.m_handle);
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator T() const& { return m_handle; }
    operator T() && = delete;
    bool     engaged() const { return m_handle != VK_NULL_HANDLE; }
    T        object() const& { return m_handle; } // useful to be explicit for type deduction
    const T* ptr() const& { return &m_handle; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle);
    }
    T              m_handle = VK_NULL_HANDLE;
    DestroyFunc<T> m_destroy;
};

// Specialization for non-constructing handles
template <class T>
class Handle<T, void> {
public:
    template <class... Args>
    Handle(T&& handle, const Args&... args)
        : m_handle(handle)
        , m_destroy(DestroyFunc<T>(
              args...,
              m_handle /* Instance and Device need the handle to load their destructors */)) {}
    ~Handle() { destroy(); }
    Handle(const Handle& other) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(std::move(other.m_handle))
        , m_destroy(std::move(other.m_destroy)) {
        other.m_handle = VK_NULL_HANDLE;
    }
    Handle& operator=(const Handle& other) = delete;
    Handle& operator=(Handle&& other) {
        destroy();
        m_destroy      = other.m_destroy;
        m_handle       = other.m_handle;
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator T() const& { return m_handle; } // l-value only for safety
    operator T() && = delete;
    T        object() const { return m_handle; }                    // static_cast<T>() shortcut
    const T* ptr() const { return &m_handle; }                      // direct to vulkan pointer
    bool     engaged() const { return m_handle != VK_NULL_HANDLE; } // for moved-from objects

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle);
    }
    T              m_handle = VK_NULL_HANDLE;
    DestroyFunc<T> m_destroy;
};

template <class T, class CreateFunc>
struct CreateHandleVector;

template <class T>
struct DestroyVectorFunc;

template <class T, class CreateFunc>
class HandleVector {
public:
    using Parent     = typename handle_traits<T>::destroy_first_param;
    using CreateInfo = typename CreateHandleVector<T, CreateFunc>::CreateInfo;

    template <class ParentAndCommands, class... Args>
        requires std::constructible_from<Parent, const ParentAndCommands&>
    HandleVector(const ParentAndCommands& vk, CreateInfo&& createInfo, const Args&... args)
        : HandleVector(vk, static_cast<Parent>(vk), createInfo, args...) {}

    template <class Commands, class... Args>
    HandleVector(const Commands& commands, Parent parent, CreateInfo&& createInfo,
                 const Args&... args)
        : m_handles(CreateHandleVector<T, CreateFunc>()(commands, parent, createInfo, args...))
        , m_destroy(DestroyVectorFunc<T>(commands, parent, createInfo, args...)) {}
    ~HandleVector() { destroy(); }
    HandleVector(const HandleVector& other)            = delete;
    HandleVector(HandleVector&& other) noexcept        = default;
    HandleVector& operator=(const HandleVector& other) = delete;
    HandleVector& operator=(HandleVector&& other) {
        destroy();
        m_destroy = std::move(other.m_destroy);
        m_handles = std::move(other.m_handles);
        return *this;
    }
    auto& operator[](size_t i) const { return m_handles[i]; }
    auto  begin() const { return m_handles.begin(); }
    auto  end() const { return m_handles.end(); }
    auto  data() const { return m_handles.data(); }
    auto  size() const { return m_handles.size(); }
    auto  empty() const { return m_handles.empty(); }

private:
    void destroy() {
        // Moved-from vectors are empty. Not good for perf in general, but
        // something we can exploit here
        if (!m_handles.empty())
            m_destroy(m_handles);
    }
    std::vector<T>       m_handles;
    DestroyVectorFunc<T> m_destroy;
};

} // namespace vko
