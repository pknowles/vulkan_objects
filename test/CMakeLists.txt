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

# Unit tests
add_executable(unit_tests
    )
target_link_libraries(unit_tests gtest_main)

if(MSVC)
  target_compile_options(unit_tests PRIVATE /W4 /WX)
else()
  target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)
