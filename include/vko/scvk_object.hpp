// Copyright (c) 2024 Pyarelal Knowles, MIT License

#pragma once

#include <stdexcept>
namespace scvk {

template <typename Func>
struct FunctionTraits;

template <typename Ret, typename... Args>
struct FunctionTraits<Ret(Args...)> {
    using return_type = Ret;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t N>
    struct Argument {
        using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
    };
};

template<class VKType, class Context, auto DestroyFunc>
class VSO {
public:
    VSO() = delete;
    VSO(Context& context, VKType&& object) : m_context(context), m_object(object) {
        // Enforce RAII
        if(object == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Null vulkan object handle is not allowed");
        }
    }
    ~VSO()
    {
        if constexpr(FunctionTraits<DestroyFunc>::Argument<i>::arity == 2)
        {
            m_context.vki->DestroyFunc(&m_object, m_context.allocationCallbacks());
        }
        if constexpr(std::is_same_v<FunctionTraits<DestroyFunc>::Argument<i>::type, VkInstance)
        {
            m_context.vki->DestroyFunc(m_context.instance(), &m_object,
                                       m_context.allocationCallbacks());
        }
        if constexpr(std::is_same_v<FunctionTraits<DestroyFunc>::Argument<i>::type, VkDevice)
        {
            m_context.vki->DestroyFunc(m_context.device(), &m_object,
                                       m_context.allocationCallbacks());
        }
    }

    operator const VKType&() { return m_object; }

private:
    Context& m_context;
    VKType m_object;
};

template<class VKType, void (*Destroy)(VkDevice, VKType*, const VkAllocationCallbacks*)>
class DeviceObject {
public:
    DeviceObject(VKType&& object) : m_object(object)
    {
        // Enforce RAII
        if(object == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Null vulkan object handle is not allowed");
        }
    }
    ~DeviceObject()
    {
        Destroy(m_context->device, m_accelerationStructure, m_context->allocationCallbacks);
    }

    operator const VKType&() { return m_object; }

private:
    VKType m_object;
};

};

