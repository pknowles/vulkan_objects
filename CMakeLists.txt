# Copyright (c) 2024-2025 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

project(vulkan_objects)

add_subdirectory(generate)

set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(generated_sources
  src/include/vko/functions.hpp.txt
  src/include/vko/handles.hpp.txt
  src/include/vko/structures.hpp.txt
  src/functions.cpp.txt
  )
add_custom_target(vulkan_objects_generated)

add_library(vulkan_objects
  ${generated_directory}/src/functions.cpp
  )
target_compile_features(vulkan_objects PUBLIC cxx_std_20)
target_include_directories(vulkan_objects PUBLIC include "${generated_directory}/src/include")
target_compile_definitions(vulkan_objects PUBLIC VK_NO_PROTOTYPES)
add_dependencies(vulkan_objects vulkan_objects_generated)

set(VULKAN_OBJECTS_SPEC_TAG "v1.4.307" CACHE STRING "GIT tag from https://github.com/KhronosGroup/Vulkan-Headers/tags")

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
