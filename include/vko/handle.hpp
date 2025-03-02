// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <string_view>
#include <vko/adaptors.hpp>
#include <vko/exceptions.hpp>
#include <vko/functions.hpp>
#include <vulkan/vulkan_core.h>

namespace vko
{

template<class Handle> struct handle_traits;

template <class T, class CreateFunc, class CreateInfo> struct CreateHandle;

// Argument pack for destroying a handle. Similar to an object keeping a
// reference to its allocator. std::function could work too, but I have a
// premature optimization hunch this could be cheaper (naturally, untested).
template <class T>
struct DestroyFunc{
    // In most cases the parent is the first parameter of the destroy function
    using Parent = handle_traits<T>::destroy_first_param;

    template <class FunctionsAndParent>
        requires std::constructible_from<Parent, FunctionsAndParent>
    DestroyFunc(T handle, const FunctionsAndParent& vk) : DestroyFunc(handle, vk, vk) {
    }
    DestroyFunc(T /* instance/device are special :( */, const handle_traits<T>::table& table, Parent parent)
        : destroy(table.*handle_traits<T>::table_destroy),
          parent(parent) {}
    void operator()(T handle) const { destroy(parent, handle, nullptr); }

    // Copy the function pointer rather than take a reference to the function
    // table itself and risk a dangling pointer
    handle_traits<T>::destroy_t destroy;

    // The owning object. This could be a weak_ptr to be extra safe, but the
    // overhead makes it a step too far.
    Parent parent;
};

template <class T, class CreateFunc = void, class CreateInfo = void>
class Handle;

template <class T, class CreateFunc, class CreateInfo>
class Handle {
public:
    template <class ...Args>
    Handle(const CreateInfo& createInfo,  const Args&... args)
        : m_handle(CreateHandle<T, CreateFunc, CreateInfo>()(createInfo, args...)),
          m_destroy(DestroyFunc<T>(m_handle /* sigh; only needed for instance and device */, args...)) {}
    ~Handle() { destroy(); }
    Handle(const Handle& other) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(other.m_handle)
        , m_destroy(other.m_destroy) {
        other.m_handle = VK_NULL_HANDLE;
    }
    Handle& operator=(const Handle& other) = delete;
    Handle& operator=(Handle&& other) {
        destroy();
        m_destroy = other.m_destroy;
        m_handle = other.m_handle;
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator T() const { return m_handle; }
    explicit operator bool() const { return m_handle != VK_NULL_HANDLE; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle);
    }
    T              m_handle = VK_NULL_HANDLE;
    DestroyFunc<T> m_destroy;
};

template <class T>
class Handle<T, void, void> {
public:
    template <class ...Args>
    Handle(T&& handle,  const Args&... args)
        : m_handle(handle),
          m_destroy(DestroyFunc<T>(m_handle /* sigh; only needed for instance and device */, args...)) {}
    ~Handle() { destroy(); }
    Handle(const Handle& other) = delete;
    Handle(Handle&& other) noexcept
        : m_handle(other.m_handle)
        , m_destroy(other.m_destroy) {
        other.m_handle = VK_NULL_HANDLE;
    }
    Handle& operator=(const Handle& other) = delete;
    Handle& operator=(Handle&& other) {
        destroy();
        m_destroy = other.m_destroy;
        m_handle = other.m_handle;
        other.m_handle = VK_NULL_HANDLE;
        return *this;
    }
    operator T() const { return m_handle; }
    explicit operator bool() const { return m_handle != VK_NULL_HANDLE; }

private:
    void destroy() {
        if (m_handle != VK_NULL_HANDLE)
            m_destroy(m_handle);
    }
    T              m_handle = VK_NULL_HANDLE;
    DestroyFunc<T> m_destroy;
};

} // namespace vko
