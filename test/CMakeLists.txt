# Copyright (c) 2024 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

# Download https://github.com/scottt/debugbreak
# TODO: just inline?
set(DEBUGBREAK_DIR "${CMAKE_BINARY_DIR}/debugbreak")
set(DEBUGBREAK_HEADER "${DEBUGBREAK_DIR}/debugbreak.h")
file(MAKE_DIRECTORY "${DEBUGBREAK_DIR}")
if(NOT EXISTS "${DEBUGBREAK_HEADER}")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/scottt/debugbreak/5dcbe41d2bd4712c8014aa7e843723ad7b40fd74/debugbreak.h"
        "${DEBUGBREAK_HEADER}"
        EXPECTED_HASH SHA256=f691efbd23848c927d9ee98d7b5264a86cf3a36607633b08bd6936f44e2b71c8
        TLS_VERIFY ON
    )
endif()

# Unit tests
add_executable(vulkan_objects_tests
  src/hello_triangle.cpp
  src/test_timeline_queue.cpp
  src/test_staging.cpp
  src/test_query.cpp
  )

# Enable clang-tidy to catch common bugs like use-after-move
# Disabled for now due to false positives in template code
# set_target_properties(vulkan_objects_tests PROPERTIES
#   CXX_CLANG_TIDY "clang-tidy;--checks=bugprone-use-after-move,bugprone-move-forwarding-reference,performance-move-const-arg,performance-unnecessary-value-param"
# )

target_link_libraries(vulkan_objects_tests vulkan_objects glfw gtest_main)
target_include_directories(vulkan_objects_tests PRIVATE "${DEBUGBREAK_DIR}" src)

# TODO: vulkan_objects provides these dependencies. could pick them up
# separately
# TODO: default to fatal errors for disabled tests!
if(TARGET slang)
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_SLANG=1)
else()
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_SLANG=0)
  message(WARNING "Slang not found, vulkan_objects_tests will not be able to compile slang shaders, disabling most tests!")
endif()
if(TARGET vma_static)
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_VMA=1)
else()
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_VMA=0)
  message(FATAL_ERROR "Vulkan Memory Allocator not found, vulkan_objects_tests will not be able to compile vma code")
endif()
if(TARGET vvl)
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_VVL=1)
else()
  target_compile_definitions(vulkan_objects_tests PRIVATE VULKAN_OBJECTS_HAS_VVL=0)
  if(VULKAN_OBJECTS_TESTING_BUILD_ALL)
    message(FATAL_ERROR "Vulkan Validation Layers not found, vulkan_objects_tests will not be able to compile vvl code")
  else()
    message(WARNING "Vulkan Validation Layers not found, tests will run without validation")
  endif()
endif()

if(MSVC)
  target_compile_options(vulkan_objects_tests PRIVATE /W4 /WX)
  # Enable static analysis in Debug builds
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vulkan_objects_tests PRIVATE /analyze)
  endif()
else()
  target_compile_options(vulkan_objects_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vulkan_objects_tests PUBLIC -Wno-nullability-extension -Wno-nullability-completeness)
  endif()
  # Enable AddressSanitizer in debug builds to catch memory issues
  # Disabled temporarily due to NVIDIA driver incompatibility
  #if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  #  target_compile_options(vulkan_objects_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  #  target_link_options(vulkan_objects_tests PRIVATE -fsanitize=address)
  #endif()
endif()

include(GoogleTest)
gtest_discover_tests(vulkan_objects_tests DISCOVERY_MODE PRE_TEST)
