// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <list>
#include <unordered_map>
#include <variant>
#include <vector>
#include <vko/handles.hpp>

namespace vko {

struct BindingsAndFlags {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags>     flags;
};

template <device_and_commands DeviceAndCommands>
inline DescriptorSetLayout makeDescriptorSetLayout(const DeviceAndCommands& device,
                                                   const BindingsAndFlags&  bindingsAndFlags,
                                                   VkDescriptorSetLayoutCreateFlags createFlags) {
    return makeDescriptorSetLayout(device, bindingsAndFlags.bindings, bindingsAndFlags.flags,
                                   createFlags);
}

template <device_and_commands DeviceAndCommands>
inline DescriptorSetLayout makeDescriptorSetLayout(
    const DeviceAndCommands& device, std::span<const VkDescriptorSetLayoutBinding> bindings,
    std::span<const VkDescriptorBindingFlags> flags, VkDescriptorSetLayoutCreateFlags createFlags) {
    VkDescriptorSetLayoutBindingFlagsCreateInfo layoutBindingFlagsCreateInfo = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext         = nullptr,
        .bindingCount  = uint32_t(flags.size()),
        .pBindingFlags = flags.data(),
    };
    return DescriptorSetLayout(device,
                               VkDescriptorSetLayoutCreateInfo{
                                   .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                   .pNext = &layoutBindingFlagsCreateInfo,
                                   .flags = createFlags,
                                   .bindingCount = uint32_t(bindings.size()),
                                   .pBindings    = bindings.data(),
                               });
}

struct DescriptorSetPoolSizes {
    DescriptorSetPoolSizes(std::span<const VkDescriptorSetLayoutBinding> bindings) {
        std::unordered_map<VkDescriptorType, uint32_t> typeSizes;
        for (const auto& binding : bindings)
            typeSizes[binding.descriptorType]++;
        sizes.reserve(typeSizes.size());
        for (auto& typeSize : typeSizes)
            sizes.push_back({typeSize.first, typeSize.second});
    }
    std::vector<VkDescriptorPoolSize> sizes;
};

class SingleDescriptorSetPool {
public:
    template <device_and_commands DeviceAndCommands>
    SingleDescriptorSetPool(const DeviceAndCommands&                      device,
                            std::span<const VkDescriptorSetLayoutBinding> bindings,
                            VkDescriptorPoolCreateFlags                   flags)
        : SingleDescriptorSetPool(device, DescriptorSetPoolSizes(bindings).sizes, flags) {}

    template <device_and_commands DeviceAndCommands>
    SingleDescriptorSetPool(const DeviceAndCommands&              device,
                            std::span<const VkDescriptorPoolSize> sizes,
                            VkDescriptorPoolCreateFlags           flags)
        : m_pool(device, VkDescriptorPoolCreateInfo{
                             .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                             .pNext         = nullptr,
                             .flags         = flags,
                             .maxSets       = 1,
                             .poolSizeCount = uint32_t(sizes.size()),
                             .pPoolSizes    = sizes.data(),
                         }) {}
    operator VkDescriptorPool() const { return m_pool; }

private:
    DescriptorPool m_pool;
};

struct SingleDescriptorSet {
    template <device_and_commands DeviceAndCommands>
    SingleDescriptorSet(const DeviceAndCommands& device, const BindingsAndFlags& bindingsAndFlags,
                        VkDescriptorSetLayoutCreateFlags layoutCreateFlags,
                        VkDescriptorPoolCreateFlags      poolCreateFlags)
        : SingleDescriptorSet(device, bindingsAndFlags.bindings, bindingsAndFlags.flags,
                              layoutCreateFlags, poolCreateFlags) {}

    template <device_and_commands DeviceAndCommands>
    SingleDescriptorSet(const DeviceAndCommands&                      device,
                        std::span<const VkDescriptorSetLayoutBinding> bindings,
                        std::span<const VkDescriptorBindingFlags>     flags,
                        VkDescriptorSetLayoutCreateFlags              layoutCreateFlags,
                        VkDescriptorPoolCreateFlags                   poolCreateFlags)
        : layout(makeDescriptorSetLayout(device, bindings, flags, layoutCreateFlags))
        , pool(device, bindings, poolCreateFlags)
        , set(device, nullptr, pool, layout) {}
    DescriptorSetLayout     layout;
    SingleDescriptorSetPool pool;
    DescriptorSet           set;
};

// clang-format off
template <VkDescriptorType Type> struct descriptor_type_traits;
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_SAMPLER> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER> { using Info = VkBufferView; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER> { using Info = VkBufferView; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER> { using Info = VkDescriptorBufferInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_STORAGE_BUFFER> { using Info = VkDescriptorBufferInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC> { using Info = VkDescriptorBufferInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC> { using Info = VkDescriptorBufferInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK> { using Info = const std::byte; using Extend = VkWriteDescriptorSetInlineUniformBlockEXT; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR> { using Info = VkAccelerationStructureKHR; using Extend = VkWriteDescriptorSetAccelerationStructureKHR; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV> { using Info = VkAccelerationStructureNV; using Extend = VkWriteDescriptorSetAccelerationStructureNV; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM> { using Info = VkDescriptorImageInfo; using Extend = void; };
template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM> { using Info = VkDescriptorImageInfo; using Extend = void; };
//template<> struct descriptor_type_traits<VK_DESCRIPTOR_TYPE_MUTABLE_EXT> { using Info = VkMutableDescriptorTypeListEXT; using Extend = ?? VkMutableDescriptorTypeCreateInfoEXT; };
// clang-format on

struct WriteDescriptorSetBuilder {
public:
    template <VkDescriptorType Type>
    void push_back(VkDescriptorSet descriptorSet, const VkDescriptorSetLayoutBinding& binding,
                   uint32_t                                          arrayElement,
                   const typename descriptor_type_traits<Type>::Info descriptorInfo) {
        push_back<Type>(descriptorSet, binding, arrayElement, std::span(&descriptorInfo, 1));
    }
    template <VkDescriptorType Type>
    void push_back(VkDescriptorSet descriptorSet, const VkDescriptorSetLayoutBinding& binding,
                   uint32_t                                                     arrayElement,
                   std::span<const typename descriptor_type_traits<Type>::Info> descriptorInfos) {
        const typename descriptor_type_traits<Type>::Info* descriptorInfoPtr = nullptr;
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info,
                                     VkDescriptorImageInfo>) {
            m_decriptorsImageInfo.emplace_back(descriptorInfos.begin(), descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsImageInfo.back().data();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info, VkBufferView>) {
            m_decriptorsBufferView.emplace_back(descriptorInfos.begin(), descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsBufferView.back().data();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info,
                                     VkDescriptorBufferInfo>) {
            m_decriptorsBufferInfo.emplace_back(descriptorInfos.begin(), descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsBufferInfo.back().data();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info,
                                     const std::byte>) {
            m_decriptorsInlineUniform.emplace_back(descriptorInfos.begin(), descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsInlineUniform.back().data();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info,
                                     VkAccelerationStructureKHR>) {
            m_decriptorsAccelerationStructureKHR.emplace_back(descriptorInfos.begin(),
                                                              descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsAccelerationStructureKHR.back().data();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Info,
                                     VkAccelerationStructureNV>) {
            m_decriptorsAccelerationStructureNV.emplace_back(descriptorInfos.begin(),
                                                             descriptorInfos.end());
            descriptorInfoPtr = m_decriptorsAccelerationStructureNV.back().data();
        }

        void* pNext = nullptr;
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Extend,
                                     VkWriteDescriptorSetInlineUniformBlockEXT>) {
            m_writesInlineUniformBlockEXT.push_back(VkWriteDescriptorSetInlineUniformBlockEXT{
                .sType    = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT,
                .pNext    = nullptr,
                .dataSize = uint32_t(descriptorInfos.size()),
                .pData    = descriptorInfoPtr});
            pNext = &m_writesInlineUniformBlockEXT.back();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Extend,
                                     VkWriteDescriptorSetAccelerationStructureKHR>) {
            m_writesAccelerationStructureKHR.push_back(VkWriteDescriptorSetAccelerationStructureKHR{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .pNext = nullptr,
                .accelerationStructureCount = uint32_t(descriptorInfos.size()),
                .pAccelerationStructures    = descriptorInfoPtr});
            pNext = &m_writesAccelerationStructureKHR.back();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Extend,
                                     VkWriteDescriptorSetAccelerationStructureNV>) {
            m_writesAccelerationStructureNV.push_back(VkWriteDescriptorSetAccelerationStructureNV{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV,
                .pNext = nullptr,
                .accelerationStructureCount = uint32_t(descriptorInfos.size()),
                .pAccelerationStructures    = descriptorInfoPtr});
            pNext = &m_writesAccelerationStructureNV.back();
        }
        if constexpr (std::is_same_v<typename descriptor_type_traits<Type>::Extend,
                                     VkWriteDescriptorSetPartitionedAccelerationStructureNV>) {
            m_writesPartitionedAccelerationStructureNV.push_back(
                VkWriteDescriptorSetPartitionedAccelerationStructureNV{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT,
                    .pNext = nullptr,
                    .accelerationStructureCount = uint32_t(descriptorInfos.size()),
                    .pAccelerationStructures    = descriptorInfoPtr});
            pNext = &m_writesPartitionedAccelerationStructureNV.back();
        }
        m_writes.push_back(VkWriteDescriptorSet{
            .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext            = pNext,
            .dstSet           = descriptorSet,
            .dstBinding       = binding.binding,
            .dstArrayElement  = arrayElement,
            .descriptorCount  = uint32_t(descriptorInfos.size()),
            .descriptorType   = Type,
            .pImageInfo       = ptrIfSameOrNull<const VkDescriptorImageInfo>(descriptorInfoPtr),
            .pBufferInfo      = ptrIfSameOrNull<const VkDescriptorBufferInfo>(descriptorInfoPtr),
            .pTexelBufferView = ptrIfSameOrNull<const VkBufferView>(descriptorInfoPtr),
        });
    }

    template <class DstDescriptorInfo, class SrcDescriptorInfo>
    static constexpr DstDescriptorInfo* ptrIfSameOrNull(const SrcDescriptorInfo* infos) {
        if constexpr (std::is_same_v<std::decay_t<DstDescriptorInfo>,
                                     std::decay_t<SrcDescriptorInfo>>)
            return infos;
        return nullptr;
    }

    std::span<const VkWriteDescriptorSet> writes() const { return m_writes; }

private:
    // *rolls eyes. There's gotta be a better way. Maybe std::tuple<>..Args to
    // keep temporaries alive until making the vulkan call?
    std::list<std::vector<VkDescriptorImageInfo>>           m_decriptorsImageInfo;
    std::list<std::vector<VkBufferView>>                    m_decriptorsBufferView;
    std::list<std::vector<VkDescriptorBufferInfo>>          m_decriptorsBufferInfo;
    std::list<std::vector<std::byte>>                       m_decriptorsInlineUniform;
    std::list<std::vector<VkAccelerationStructureKHR>>      m_decriptorsAccelerationStructureKHR;
    std::list<std::vector<VkAccelerationStructureNV>>       m_decriptorsAccelerationStructureNV;
    std::list<VkWriteDescriptorSetInlineUniformBlockEXT>    m_writesInlineUniformBlockEXT;
    std::list<VkWriteDescriptorSetAccelerationStructureKHR> m_writesAccelerationStructureKHR;
    std::list<VkWriteDescriptorSetAccelerationStructureNV>  m_writesAccelerationStructureNV;
    std::list<VkWriteDescriptorSetPartitionedAccelerationStructureNV>
                                      m_writesPartitionedAccelerationStructureNV;
    std::vector<VkWriteDescriptorSet> m_writes;
};

} // namespace vko