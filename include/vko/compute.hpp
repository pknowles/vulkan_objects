// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <vko/slang_compiler.hpp>

namespace vko {
namespace simple {

// VK_EXT_shader_object provides dynamic shader binding, without creating
// permutations of pipelines. As of writing, VK_NV_per_stage_descriptor_set is
// still an NV extension and a pipeline layout is needed otherwise.
// See: https://github.com/KhronosGroup/Vulkan-Docs/issues/2444#issuecomment-2448561334
template <class PushConstants>
struct ComputeShader {
    template <device_and_commands Device>
    ComputeShader(const Device& device, ::slang::ISession* session, const std::string moduleName,
                  std::span<const VkDescriptorSetLayout> descriptorSetLayouts)
        : ComputeShader(device, session,
                        slang::EntryPoint(slang::Module(session, moduleName.c_str()), "main"),
                        descriptorSetLayouts) {}

    template <device_and_commands Device>
    ComputeShader(const Device& device, ::slang::ISession* session,
                  ::slang::IComponentType*               entryPoint,
                  std::span<const VkDescriptorSetLayout> descriptorSetLayouts)
        : ComputeShader(
              device,
              slang::Code(slang::Program(slang::Composition(session, std::span(&entryPoint, 1))), 0,
                          0)
                  .bytes(),
              descriptorSetLayouts) {}

    template <device_and_commands Device>
    ComputeShader(const Device& device, std::span<const std::byte> spirv,
                  std::span<const VkDescriptorSetLayout> descriptorSetLayouts)
        : shader(device, device,
                 std::to_array({VkShaderCreateInfoEXT{
                     .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                     .pNext                  = nullptr,
                     .flags                  = 0,
                     .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
                     .nextStage              = 0,
                     .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
                     .codeSize               = uint32_t(spirv.size()),
                     .pCode                  = spirv.data(),
                     .pName                  = "main",
                     .setLayoutCount         = 0,
                     .pSetLayouts            = nullptr,
                     .pushConstantRangeCount = 1u,
                     .pPushConstantRanges    = tmpPtr(VkPushConstantRange{
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, // TODO: don't duplicate
                            .offset     = 0u,
                            .size       = uint32_t(sizeof(PushConstants)),
                     }),
                     .pSpecializationInfo    = nullptr,
                 }}))
        , pipelineLayout(device, VkPipelineLayoutCreateInfo{
                                     .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pNext = nullptr,
                                     .flags = 0,
                                     .setLayoutCount = uint32_t(descriptorSetLayouts.size()),
                                     .pSetLayouts    = descriptorSetLayouts.data(),
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges    = tmpPtr(VkPushConstantRange{
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                            .offset     = 0u,
                                            .size       = uint32_t(sizeof(PushConstants))})

                                 }) {}

    ShadersEXT     shader;
    PipelineLayout pipelineLayout; // for push constants and descriptor sets
};

} // namespace simple
} // namespace vko
