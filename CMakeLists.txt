# Copyright (c) 2024-2025 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

project(vulkan_objects)

add_subdirectory(generate)

set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/src_gen")
set(generated_sources
  src/include/vko/gen_exceptions.hpp.txt
  src/include/vko/gen_functions.hpp.txt
  src/include/vko/gen_handles.hpp.txt
  src/include/vko/gen_structures.hpp.txt
  src/gen_functions.cpp.txt
  )
add_custom_target(vulkan_objects_generated)

add_library(vulkan_objects
  src/functions.cpp
  ${generated_directory}/src/gen_functions.cpp
  ${generated_sources} # Just so these appear in visual studio
  )
target_compile_features(vulkan_objects PUBLIC cxx_std_20)
target_include_directories(vulkan_objects PUBLIC include "${generated_directory}/src/include")
target_compile_definitions(vulkan_objects PUBLIC VK_NO_PROTOTYPES)
add_dependencies(vulkan_objects vulkan_objects_generated)

set(VULKAN_OBJECTS_SPEC_TAG "v1.4.307" CACHE STRING "GIT tag from https://github.com/KhronosGroup/Vulkan-Headers/tags")
set(VULKAN_OBJECTS_VMA_TAG "v3.2.1" CACHE STRING "GIT tag from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/tags")

# Platform-specific window manager enablement
if(WIN32)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_WIN32_KHR)
  target_compile_definitions(vulkan_objects PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(ANDROID)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
elseif(APPLE)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_METAL_EXT)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux|BSD|GNU")
  #find_package(Wayland)
  #if(xcb_FOUND)
  #  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_WAYLAND_KHR)
  #endif()
  #find_package(XCB)
  #if(XCB_FOUND)
  #  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_XCB_KHR)
  #  target_link_libraries(vulkan_objects PUBLIC xcb)
  #endif()
  find_package(X11)
  if(X11_FOUND)
    target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_XLIB_KHR)
    target_link_libraries(vulkan_objects PUBLIC ${X11_LIBRARIES})
  endif()
endif()

# 17MB and counting, for just the text header files!!
include(FetchContent)
FetchContent_Declare(
  vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG ${VULKAN_OBJECTS_SPEC_TAG}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vulkan_headers)
set(VULKAN_SPEC_XML ${vulkan_headers_SOURCE_DIR}/registry/vk.xml)
target_link_libraries(vulkan_objects PUBLIC Vulkan::Headers)

# I don't even want to know. This brings in so much extra stuff, but vulkan dev
# without validation layers is impossible and the philosophy of this project is
# it works out of the box with no manual dependencies. Pre-built binaries don't
# come with debug info, which can be really handy when hitting obscure
# validation layer errors. We gotta have it.
FetchContent_Declare(
  vvl
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-ValidationLayers.git
  GIT_TAG ${VULKAN_OBJECTS_SPEC_TAG}
  GIT_SHALLOW TRUE
)
set(UPDATE_DEPS ON)
set(BUILD_TESTS_backup ${BUILD_TESTS})
set(BUILD_TESTS OFF)
FetchContent_MakeAvailable(vvl)
set(BUILD_TESTS ${BUILD_TESTS_backup})
add_dependencies(vulkan_objects vvl)
target_compile_definitions(vulkan_objects PRIVATE VVL_DEVELOP_PATH="$<TARGET_FILE_DIR:vvl>")

include(FetchContent)
FetchContent_Declare(
  VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG ${VULKAN_OBJECTS_VMA_TAG}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)
add_library(vma_static src/vma_impl.cpp)
target_compile_definitions(vma_static PUBLIC VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1)
target_link_libraries(vma_static PUBLIC Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
target_link_libraries(vulkan_objects PUBLIC vma_static)

function(add_generated_file source)
  get_filename_component(source_directory "${source}" DIRECTORY)
  get_filename_component(output_name "${source}" NAME_WLE)
  file(RELATIVE_PATH relative_output "${CMAKE_CURRENT_SOURCE_DIR}" "${source_directory}")
  set(output "${generated_directory}/${relative_output}/${output_name}")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND vulkan_objects_generator "${VULKAN_SPEC_XML}" "${source}" "${output}"
    DEPENDS vulkan_objects_generator "${VULKAN_SPEC_XML}" "${source}"
    COMMENT "Generating '${output}'"
  )
  target_sources(vulkan_objects_generated PRIVATE "${output}")
endfunction()

foreach(generated_source IN LISTS generated_sources)
  add_generated_file("${CMAKE_CURRENT_SOURCE_DIR}/${generated_source}")
endforeach()

if(BUILD_TESTING)
  option(BUILD_VULKAN_OBJECTS_TESTING "Enable vulkan_objects testing" ON)
  if(BUILD_VULKAN_OBJECTS_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()
