# vko: Vulkan Objects

This library is an RAII wrapper around some core parts of the vulkan SDK.
The aims are:

1. Dependencies are implied by the language

   Delayed initialization allows you to create an object before its dependencies
   are created or even in scope. This makes using it difficult. For example, you
   can't create a VkDevice before a VkInstance.

2. Lifetime and ownership is well defined

# volk vulkan loader
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 21ceafb55b7ca55b15aee758a8541c06e29780cd # 1.3.270
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

