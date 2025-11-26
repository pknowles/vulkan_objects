// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace vko {

// For owning an arbitrary object without its type. Unlike std::any, this
// supports move-only types and avoids the overhead of std::any's copy
// semantics.
using UniqueAny = std::unique_ptr<void, void (*)(void*)>;

// Utility to construct a UniqueAny in-place and return a non-owning pointer.
// T must be explicitly specified: makeUniqueAny<MyType>(args...)
// Returns std::pair<UniqueAny, T*> where first is the owning pointer and second is the typed
// pointer.
template <class T, class... Args>
    requires std::is_object_v<T> && std::is_constructible_v<T, Args...>
inline std::pair<UniqueAny, T*> makeUniqueAny(Args&&... args) {
    T* ptr = new T(std::forward<Args>(args)...);
    return {UniqueAny(ptr, [](void* p) { delete reinterpret_cast<T*>(p); }), ptr};
}

// Utility to move an object into a UniqueAny with a deleter lambda.
// Only accepts rvalues to prevent accidental copies.
template <class T>
    requires(std::is_object_v<T> && std::is_move_constructible_v<T> &&
             !std::is_lvalue_reference_v<T>)
inline UniqueAny toUniqueAny(T&& t) {
    return UniqueAny(new T(std::move(t)), [](void* ptr) { delete reinterpret_cast<T*>(ptr); });
}

// Utility to return a pointer to the moved object, which would otherwise lose
// its type. The returned pointer must not outlive the UniqueAny.
// Only accepts rvalues to prevent accidental copies.
// Returns std::pair<UniqueAny, T*> where first is the owning pointer and second is the typed
// pointer.
template <class T>
    requires(std::is_object_v<T> && std::is_move_constructible_v<T> &&
             !std::is_lvalue_reference_v<T>)
inline std::pair<UniqueAny, T*> toUniqueAnyWithPtr(T&& t) {
    T* ptr = new T(std::move(t));
    return {UniqueAny(ptr, [](void* p) { delete reinterpret_cast<T*>(p); }), ptr};
}

} // namespace vko
