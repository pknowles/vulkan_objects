// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <string>
#include <vko/exceptions.hpp>

#ifdef VK_USE_PLATFORM_XLIB_KHR
    #pragma push_macro("None")
    #pragma push_macro("Bool")
    #undef None
    #undef Bool
#endif
#include <slang-com-ptr.h>
#include <slang.h>
#ifdef VK_USE_PLATFORM_XLIB_KHR
    #pragma pop_macro("Bool")
    #pragma pop_macro("None")
#endif

namespace vko {
namespace slang {

void check(SlangResult result) {
    if (SLANG_FAILED(result)) {
        // TODO: to-string?
        throw Exception(
            "Slang error, facility: " + std::to_string(SLANG_GET_RESULT_FACILITY(result)) +
            " code: " + std::to_string(SLANG_GET_RESULT_CODE(result)));
    }
}

class GlobalSession {
public:
    using T = ::slang::IGlobalSession;
    GlobalSession(const SlangGlobalSessionDesc& desc = {}) {
        check(::slang::createGlobalSession(&desc, m_session.writeRef()));
    }
    operator T*() const { return m_session.get(); }
    T& operator*() const { return *m_session.get(); }
    T* operator->() const { return m_session.get(); }

private:
    Slang::ComPtr<T> m_session;
};

class Session {
public:
    using T = ::slang::ISession;
    Session(::slang::IGlobalSession* globalSession, const ::slang::SessionDesc& desc = {}) {
        check(globalSession->createSession(desc, m_session.writeRef()));
    }
    operator T*() const { return m_session.get(); }
    T& operator*() const { return *m_session.get(); }
    T* operator->() const { return m_session.get(); }

private:
    Slang::ComPtr<T> m_session;
};

class Module {
public:
    using T = ::slang::IModule;
    Module(::slang::ISession* session, const char* moduleName) {
        auto module = session->loadModule(moduleName, m_diagnostics.writeRef());
        // TODO: proper error handling
        if (m_diagnostics) {
            fprintf(stderr, "%s\n",
                    reinterpret_cast<const char*>(m_diagnostics->getBufferPointer()));
        }
        if (!module) {
            throw Exception("Slang loadModule returned null");
        }
        m_module = Slang::ComPtr<T>{/*Slang::INIT_ATTACH,*/ module};
    }
    Module(::slang::ISession* session, const char* moduleName, const char* path,
           const char* string) {
        auto module =
            session->loadModuleFromSourceString(moduleName, path, string, m_diagnostics.writeRef());
        // TODO: proper error handling
        if (m_diagnostics) {
            fprintf(stderr, "%s\n",
                    reinterpret_cast<const char*>(m_diagnostics->getBufferPointer()));
        }
        if (!module) {
            throw Exception("Slang loadModule returned null");
        }
        m_module = Slang::ComPtr<T>{Slang::INIT_ATTACH, module};
    }
    operator T*() const { return m_module.get(); }
    T& operator*() const { return *m_module.get(); }
    T* operator->() const { return m_module.get(); }

private:
    Slang::ComPtr<T>              m_module;
    Slang::ComPtr<::slang::IBlob> m_diagnostics;
};

class EntryPoint {
public:
    using T = ::slang::IEntryPoint;
    EntryPoint(::slang::IModule* module, const char* entryPointName) {
        check(module->findEntryPointByName(entryPointName, m_entryPoint.writeRef()));
    }
    operator T*() const { return m_entryPoint.get(); }
    T& operator*() const { return *m_entryPoint.get(); }
    T* operator->() const { return m_entryPoint.get(); }

private:
    Slang::ComPtr<T> m_entryPoint;
};

class Composition {
public:
    using T = ::slang::IComponentType;
    Composition(::slang::ISession* session, std::span<::slang::IComponentType* const> components) {
        check(session->createCompositeComponentType(components.data(), components.size(),
                                                    m_componentType.writeRef()));
    }
    operator T*() const { return m_componentType.get(); }
    T& operator*() const { return *m_componentType.get(); }
    T* operator->() const { return m_componentType.get(); }

private:
    Slang::ComPtr<T> m_componentType;
};

class Program {
public:
    using T = ::slang::IComponentType;
    Program(::slang::IComponentType* composition) {
        auto result = composition->link(m_componentType.writeRef(), m_diagnostics.writeRef());
        // TODO: proper error handling
        if (m_diagnostics) {
            fprintf(stderr, "%s\n",
                    reinterpret_cast<const char*>(m_diagnostics->getBufferPointer()));
        }
        check(result);
    }
    operator T*() const { return m_componentType.get(); }
    T& operator*() const { return *m_componentType.get(); }
    T* operator->() const { return m_componentType.get(); }

private:
    Slang::ComPtr<T>              m_componentType;
    Slang::ComPtr<::slang::IBlob> m_diagnostics;
};

class Code {
public:
    using T = ::slang::IBlob;
    Code(::slang::IModule* module, int targetIndex) {
        auto result = module->getTargetCode(targetIndex, m_entryPointCode.writeRef(),
                                            m_diagnostics.writeRef());
        // TODO: proper error handling
        if (m_diagnostics) {
            fprintf(stderr, "%s\n",
                    reinterpret_cast<const char*>(m_diagnostics->getBufferPointer()));
        }
        check(result);
    }
    Code(::slang::IComponentType* linkedProgram, int entryPointIndex, int targetIndex) {
        auto result = linkedProgram->getEntryPointCode(
            entryPointIndex, targetIndex, m_entryPointCode.writeRef(), m_diagnostics.writeRef());
        // TODO: proper error handling
        if (m_diagnostics) {
            fprintf(stderr, "%s\n",
                    reinterpret_cast<const char*>(m_diagnostics->getBufferPointer()));
        }
        check(result);
    }
    operator T*() const { return m_entryPointCode.get(); }
    T&               operator*() const { return *m_entryPointCode.get(); }
    T*               operator->() const { return m_entryPointCode.get(); }
    size_t           size() const { return m_entryPointCode->getBufferSize(); }
    const std::byte* data() const {
        return reinterpret_cast<const std::byte*>(m_entryPointCode->getBufferPointer());
    }
    std::span<const std::byte> bytes() const { return {data(), size()}; }

private:
    Slang::ComPtr<T>              m_entryPointCode;
    Slang::ComPtr<::slang::IBlob> m_diagnostics;
};

} // namespace slang
} // namespace vko
