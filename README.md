
# volk vulkan loader
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 21ceafb55b7ca55b15aee758a8541c06e29780cd # 1.3.270
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

