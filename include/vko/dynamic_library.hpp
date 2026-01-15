// Copyright (c) 2024-2025 Pyarelal Knowles, MIT License
#pragma once

#include <filesystem>
#include <vko/exceptions.hpp>

#ifdef _WIN32

    #include <windows.h>

namespace vko {

class Message {
public:
    Message()                                = delete;
    Message(const Message& other)            = delete;
    Message& operator=(const Message& other) = delete;
    Message(DWORD error, HMODULE module = NULL)
        : m_size(FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                    FORMAT_MESSAGE_IGNORE_INSERTS |
                                    (module ? FORMAT_MESSAGE_FROM_HMODULE : 0),
                                module, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                (LPSTR)&m_buffer, 0, NULL)) {}
    ~Message() { LocalFree(m_buffer); }
    std::string str() const { return {m_buffer, m_size}; }

private:
    LPSTR m_buffer;
    DWORD m_size;
};

class LastError : public Exception {
public:
    LastError()
        : Exception(Message(::GetLastError()).str()) {}
};

class DynamicLibrary {
public:
    DynamicLibrary()                            = delete;
    DynamicLibrary(const DynamicLibrary& other) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept
        : m_module(other.m_module) {
        other.m_module = nullptr;
    };
    DynamicLibrary(const std::filesystem::path& path)
        : m_module(::LoadLibraryW(path.c_str())) {
        if (!m_module) {
            throw LastError(); //("Failed to load " + path.string());
        }
    }
    DynamicLibrary& operator=(const DynamicLibrary& other) = delete;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        if (m_module)
            FreeLibrary(m_module);
        m_module       = other.m_module;
        other.m_module = nullptr;
        return *this;
    }
    ~DynamicLibrary() {
        if (m_module)
            FreeLibrary(m_module);
    }
    operator HMODULE() const { return m_module; }

    template <typename FuncType>
    FuncType get(const std::string& functionName) const {
        FARPROC functionAddress = ::GetProcAddress(m_module, functionName.c_str());
        if (!functionAddress) {
            throw LastError(); //("Failed to get address for " + functionName);
        }
        return reinterpret_cast<FuncType>(functionAddress);
    }

private:
    HMODULE m_module = nullptr;
};

} // namespace vko

#else

    #include <dlfcn.h>
    #include <string.h>

namespace vko {

class LastError : public Exception {
public:
    LastError()
        : Exception(std::string(strerror(errno))) {}
};

class DynamicLibrary {
public:
    DynamicLibrary()                            = delete;
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
        m_handle       = other.m_handle;
        other.m_handle = nullptr;
        return *this;
    }
    ~DynamicLibrary() { destroy(); }
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
    void destroy() {
        if (m_handle) {
            ::dlclose(m_handle);
        }
    }

    void* m_handle = nullptr;
};

} // namespace vko

#endif
