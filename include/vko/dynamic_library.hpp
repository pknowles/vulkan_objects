// Copyright (c) 2024-2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/exceptions.hpp>
#include <filesystem>

#ifdef _WIN32

#include <windows.h>

namespace vko
{

class LastError : public Exception {
public:
    LastError()
        : Exception(std::string(strerror(errno))) {}
};

class DynamicLibrary {
public:
    DynamicLibrary() = delete;
    DynamicLibrary(const DynamicLibrary& other) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept
        : m_module(other.m_module) {
        other.m_module = nullptr;
    };
    DynamicLibrary(const fs::path& path)
        : m_module(::LoadLibraryW(path.c_str())) {
        if (!m_module) {
            throw LastError(); //("Failed to load " + path.string());
        }
    }
    DynamicLibrary& operator=(const DynamicLibrary& other) = delete;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        if (m_module)
            FreeLibrary(m_module);
        m_module = other.m_module;
        other.m_module = nullptr;
    }
    ~DynamicLibrary() {
        if (m_module)
            FreeLibrary(m_module);
    }
    operator HMODULE() const { return m_module; }

    template <typename FuncType>
    FuncType* get(const std::string& functionName) const {
        FARPROC functionAddress = ::GetProcAddress(m_module, functionName.c_str());
        if (!functionAddress) {
            throw LastError(); //("Failed to get address for " + functionName);
        }
        return reinterpret_cast<FuncType*>(functionAddress);
    }

private:
    HMODULE m_module = nullptr;
};

} // namespace vko

#else

#include <string.h>
#include <dlfcn.h>

namespace vko
{

class LastError : public Exception {
public:
    LastError()
        : Exception(std::string(strerror(errno))) {}
};

class DynamicLibrary {
public:
    DynamicLibrary() = delete;
    DynamicLibrary(const DynamicLibrary& other) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept
        : m_handle(other.m_handle) {
        other.m_handle = nullptr;
    }
    DynamicLibrary(const std::filesystem::path& path)
        : m_handle(::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
        if (!m_handle) {
            throw Exception(dlerror());
        }
    }
    DynamicLibrary& operator=(const DynamicLibrary& other) = delete;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        destroy();
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        return *this;
    }
    ~DynamicLibrary() {
        destroy();
    }
    operator void*() const { return m_handle; }

    template <typename FuncType>
    FuncType get(const std::string& functionName) const {
        void* functionAddress = ::dlsym(m_handle, functionName.c_str());
        if (!functionAddress) {
            throw Exception(dlerror());
        }
        return reinterpret_cast<FuncType>(functionAddress);
    }

private:
    void destroy()
    {
        if (m_handle)
        {
            ::dlclose(m_handle);
        }
    }

    void* m_handle = nullptr;
};

} // namespace vko

#endif
