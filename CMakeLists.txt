# Copyright (c) 2024-2025 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

project(vulkan_objects)

add_subdirectory(generate)

set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/include/vko")
file(MAKE_DIRECTORY "${generated_directory}")
set(generated_sources
  src/handles.hpp.txt
  src/functions.hpp.txt
  )
add_custom_target(vulkan_objects_headers)

add_library(vulkan_objects INTERFACE)
target_compile_features(vulkan_objects INTERFACE cxx_std_20)
target_include_directories(vulkan_objects INTERFACE include "${CMAKE_CURRENT_BINARY_DIR}/include")
add_dependencies(vulkan_objects vulkan_objects_headers)

set(VULKAN_OBJECTS_SPEC_TAG "v1.4.307" CACHE STRING "GIT tag from https://github.com/KhronosGroup/Vulkan-Docs/tags")
if(NOT VULKAN_SPEC_XML)
    set(VULKAN_SPEC_XML "${CMAKE_BINARY_DIR}/vk.xml")
    file(DOWNLOAD "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/${VULKAN_OBJECTS_SPEC_TAG}/xml/vk.xml"
        "${VULKAN_SPEC_XML}")
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
target_link_libraries(vulkan_objects INTERFACE Vulkan::Headers)

function(add_generated_file source)
  get_filename_component(output_name "${source}" NAME_WLE)
  set(output "${generated_directory}/${output_name}")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND vulkan_objects_generator "${VULKAN_SPEC_XML}" "${source}" "${output}"
    DEPENDS vulkan_objects_generator "${VULKAN_SPEC_XML}" "${source}"
    COMMENT "Generating '${output}'"
  )
  target_sources(vulkan_objects_headers PRIVATE "${output}")
endfunction()

foreach(generated_source IN LISTS generated_sources)
  add_generated_file("${CMAKE_CURRENT_SOURCE_DIR}/${generated_source}")
endforeach()

if(TARGET volk)
  target_link_libraries(vulkan_objects INTERFACE volk)
  target_compile_definitions(vulkan_objects INTERFACE VULKAN_LOADER_VOLK=1)
else()
  target_compile_definitions(vulkan_objects INTERFACE VULKAN_LOADER_VOLK=0)
endif()

if(BUILD_TESTING)
  option(BUILD_VULKAN_OBJECTS_TESTING "Enable vulkan_objects testing" ON)
  if(BUILD_VULKAN_OBJECTS_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()
