// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <vko/handles.hpp>

namespace vko {

template <device_and_commands DeviceAndCommands>
void cmdDynamicRenderingDefaults(const DeviceAndCommands& device, VkCommandBuffer cmd,
                                 uint32_t width, uint32_t height) {
    VkViewport            viewport{0.0F, 0.0F, float(width), float(height), 0.0F, 1.0F};
    VkRect2D              scissor{{0, 0}, {width, height}};
    VkSampleMask          sampleMask      = 0xFU;
    VkBool32              blendEnabled    = VK_FALSE;
    VkColorComponentFlags colorComponents = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    device.vkCmdSetVertexInputEXT(cmd, 0U, nullptr, 0U, nullptr);
    device.vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    device.vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
    device.vkCmdSetViewportWithCount(cmd, 1U, &viewport);
    device.vkCmdSetScissorWithCount(cmd, 1U, &scissor);
    device.vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
    device.vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
    device.vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    device.vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
    device.vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
    device.vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
    device.vkCmdSetFrontFace(cmd, VK_FRONT_FACE_CLOCKWISE);
    device.vkCmdSetDepthTestEnable(cmd, VK_FALSE);
    device.vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
    device.vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
    device.vkCmdSetStencilTestEnable(cmd, VK_FALSE);
    device.vkCmdSetColorBlendEnableEXT(cmd, 0U, 1U, &blendEnabled);
    device.vkCmdSetColorWriteMaskEXT(cmd, 0U, 1U, &colorComponents);
}

struct ImageAccess {
    VkPipelineStageFlags stage  = 0U;
    VkAccessFlags        access = 0U;
    VkImageLayout        layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

template <device_and_commands DeviceAndCommands>
void cmdImageBarrier(const DeviceAndCommands& device, VkCommandBuffer cmd, VkImage image,
                     ImageAccess src, ImageAccess dst,
                     VkImageAspectFlagBits aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier imageBarrier{.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                      .pNext               = nullptr,
                                      .srcAccessMask       = src.access,
                                      .dstAccessMask       = dst.access,
                                      .oldLayout           = src.layout,
                                      .newLayout           = dst.layout,
                                      .srcQueueFamilyIndex = 0U,
                                      .dstQueueFamilyIndex = 0U,
                                      .image               = image,
                                      .subresourceRange{aspect, 0U, 1U, 0U, 1U}};
    device.vkCmdPipelineBarrier(cmd, src.stage, dst.stage, 0U, 0U, nullptr, 0U, nullptr, 1U,
                                &imageBarrier);
}

struct MemoryAccess {
    VkPipelineStageFlags stage  = 0U;
    VkAccessFlags        access = 0U;
};

template <device_and_commands DeviceAndCommands>
void cmdMemoryBarrier(const DeviceAndCommands& device, VkCommandBuffer cmd, MemoryAccess src,
                      MemoryAccess dst) {
    VkMemoryBarrier memoryBarrier{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .pNext         = nullptr,
                                  .srcAccessMask = src.access,
                                  .dstAccessMask = dst.access};
    device.vkCmdPipelineBarrier(cmd, src.stage, dst.stage, 0U, 1U, &memoryBarrier, 0U, nullptr, 0U,
                                nullptr);
}

// Debug messenger with a global callback (not using the user data pointer)
template <instance_and_commands InstanceAndCommands>
struct SimpleDebugMessenger {
    SimpleDebugMessenger(const InstanceAndCommands&           vk,
                         PFN_vkDebugUtilsMessengerCallbackEXT callback)
        : messenger(vk,
                    VkDebugUtilsMessengerCreateInfoEXT{
                        .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                        .pNext           = nullptr,
                        .flags           = 0,
                        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
                        .pfnUserCallback = callback,
                        .pUserData       = nullptr}) {}
    vko::DebugUtilsMessengerEXT messenger;
};

} // namespace vko
