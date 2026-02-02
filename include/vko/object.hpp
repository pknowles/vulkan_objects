// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <memory>
#include <utility>

namespace vko {

// It sucks to have to so trivially wrap standard library utilities but these
// are just too damn useful...

// a std::shared_ptr that must be initialized
template <typename T>
class shared_obj {
public:
    shared_obj()
        requires(!std::default_initializable<T>)
    = delete;

    shared_obj()
        requires std::default_initializable<T>
        : m_ptr(std::make_shared<T>()) {}

    template <typename... Args>
        requires std::constructible_from<T, Args...>
    explicit shared_obj(Args&&... args)
        : m_ptr(std::make_shared<T>(std::forward<Args>(args)...)) {}

    template <typename... Args>
        requires std::constructible_from<T, Args...>
    void emplace(Args&&... args) {
        m_ptr.emplace(std::forward<Args>(args)...);
    }

    // Forward access
    T&          operator*() const noexcept { return *m_ptr; }
    T*          operator->() const noexcept { return m_ptr.get(); }
    T*          get() const noexcept { return m_ptr.get(); }
    friend bool operator==(const shared_obj&, const shared_obj&) = default;

private:
    std::shared_ptr<T> m_ptr;
};

// a std::unique_ptr that must be initialized
// TODO: rename to heap_obj? indirect_obj? T itself is often unique. The only
// reason I want this is to heap-allocate.
template <typename T>
class unique_obj {
public:
    unique_obj()
        requires(!std::default_initializable<T>)
    = delete;

    unique_obj()
        requires std::default_initializable<T>
        : m_ptr(std::make_unique<T>()) {}

    template <typename... Args>
        requires std::constructible_from<T, Args...>
    explicit unique_obj(Args&&... args)
        : m_ptr(std::make_unique<T>(std::forward<Args>(args)...)) {}

    template <typename... Args>
        requires std::constructible_from<T, Args...>
    void emplace(Args&&... args) {
        m_ptr = std::make_unique<T>(std::forward<Args>(args)...);
    }

    // Forward access
    T& operator*() const noexcept { return *m_ptr; }
    T* operator->() const noexcept { return m_ptr.get(); }
    T* get() const noexcept { return m_ptr.get(); }

private:
    std::unique_ptr<T> m_ptr;
};

// ... lol jk
// template<class T>
// using optional_obj = T;

} // namespace vko
