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
add_executable(unit_tests
  src/test.cpp
  )
target_link_libraries(unit_tests vulkan_objects glfw gtest_main)
target_include_directories(unit_tests PRIVATE "${DEBUGBREAK_DIR}")

if(MSVC)
  target_compile_options(unit_tests PRIVATE /W4 /WX)
else()
  target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(unit_tests PUBLIC -Wno-nullability-extension -Wno-nullability-completeness)
  endif()
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)
