# Copyright (c) 2024 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

project(scoped_vulkan)

add_library(scoped_vulkan INTERFACE)
target_include_directories(scoped_vulkan INTERFACE include)

if(TARGET volk)
  target_link_libraries(scoped_vulkan INTERFACE volk)
  target_compile_definitions(scoped_vulkan INTERFACE VULKAN_LOADER_VOLK=1)
else()
  target_compile_definitions(scoped_vulkan INTERFACE VULKAN_LOADER_VOLK=0)
endif()t  

if(BUILD_TESTING)
  option(BUILD_OFFSET_PTR_TESTING "Enable offset_ptr testing" ON)
  if(BUILD_OFFSET_PTR_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()
