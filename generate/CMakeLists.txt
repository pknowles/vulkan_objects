# Copyright (c) 2025 Pyarelal Knowles, MIT License

find_package(pugixml CONFIG QUIET)
if(NOT TARGET pugixml)
  include(FetchContent)
  FetchContent_Declare(
    pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG v1.15
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pugixml)
endif()

find_package(inja CONFIG QUIET)
if(NOT TARGET inja)
  include(FetchContent)
  FetchContent_Declare(
    inja
    GIT_REPOSITORY https://github.com/pantor/inja.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
  )
  set(INJA_INSTALL OFF CACHE BOOL "" FORCE)
  set(INJA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(inja)
endif()

add_executable(vulkan_objects_generator generate.cpp)
target_compile_features(vulkan_objects_generator PUBLIC cxx_std_20)
target_link_libraries(vulkan_objects_generator PUBLIC pugixml inja)
