# Copyright (c) 2024-2025 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.24)

project(vulkan_objects)

set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/src_gen")
set(generated_sources
  src/include/vko/gen_exceptions.hpp.txt
  src/include/vko/gen_functions.hpp.txt
  src/include/vko/gen_handles.hpp.txt
  src/include/vko/gen_structures.hpp.txt
  src/gen_functions.cpp.txt
  )
add_custom_target(vulkan_objects_generated)

file(GLOB vulkan_objects_headers include/vko/*.hpp)
add_library(vulkan_objects
  src/functions.cpp
  ${generated_directory}/src/gen_functions.cpp
  ${generated_sources} # Just so these appear in visual studio
  ${vulkan_objects_headers} # Same
  )
target_compile_features(vulkan_objects PUBLIC cxx_std_20)
target_include_directories(vulkan_objects PUBLIC include "${generated_directory}/src/include")
target_compile_definitions(vulkan_objects PUBLIC VK_NO_PROTOTYPES)
add_dependencies(vulkan_objects vulkan_objects_generated)

set(VULKAN_OBJECTS_SPEC_OVERRIDE "" CACHE FILEPATH "Override path to own vk.xml. Ignores VULKAN_OBJECTS_SPEC_TAG")
set(VULKAN_OBJECTS_SPEC_TAG "v1.4.335" CACHE STRING "GIT tag from https://github.com/KhronosGroup/Vulkan-Headers/tags")
set(VULKAN_OBJECTS_VMA_TAG "v3.3.0" CACHE STRING "GIT tag from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/tags")
set(VULKAN_OBJECTS_SLANG_TAG "v2026.1" CACHE STRING "GIT tag from https://github.com/shader-slang/slang/tags")
set(VULKAN_OBJECTS_SHADERC_TAG "v2025.5" CACHE STRING "GIT tag from https://github.com/google/shaderc/tags")
set(VULKAN_OBJECTS_SPIRV_HEADERS_TAG "vulkan-sdk-1.4.335.0" CACHE STRING "GIT tag from https://github.com/KhronosGroup/SPIRV-Headers/tags")
set(VULKAN_OBJECTS_SPIRV_TOOLS_TAG "v2025.4" CACHE STRING "GIT tag from https://github.com/KhronosGroup/SPIRV-Tools/tags")
set(VULKAN_OBJECTS_GLSLANG_TAG "16.1.0" CACHE STRING "GIT tag from https://github.com/KhronosGroup/glslang/tags")
set(VULKAN_OBJECTS_GLFW_TAG "3.4" CACHE STRING "GIT tag from https://github.com/glfw/glfw/tags")
option(VULKAN_OBJECTS_FETCH_VVL "Include Vulkan Validation Layers with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_VMA "Include Vulkan Memory Allocator with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_SLANG "Include Slang compiler with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_SHADERC "Include Shaderc compiler with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_GLFW "Include GLFW for XCB workaround library (vulkan_objects_glfw)" OFF)
option(VULKAN_OBJECTS_SHADERC_BUILD_EXECUTABLES "Build glslc command-line compiler (for offline shader compilation)" OFF)

if(VULKAN_OBJECTS_SPEC_OVERRIDE)
  set(VULKAN_SPEC_XML ${VULKAN_OBJECTS_SPEC_OVERRIDE})
else()
  # 17MB and counting, for just the text header files!!
  include(FetchContent)
  FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG ${VULKAN_OBJECTS_SPEC_TAG}
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE  # Makes find_package(VulkanHeaders) use this FetchContent, e.g. from Vulkan-Utility-Libraries
  )
  FetchContent_MakeAvailable(VulkanHeaders)
  set(VULKAN_SPEC_XML ${vulkanheaders_SOURCE_DIR}/registry/vk.xml)
  target_link_libraries(vulkan_objects PUBLIC Vulkan::Headers)
endif()

# Platform-specific window manager enablement
if(WIN32)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_WIN32_KHR)
  target_compile_definitions(vulkan_objects PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(ANDROID)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
elseif(APPLE)
  target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_METAL_EXT)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux|BSD|GNU")
  find_package(X11)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Wayland wayland-client)
  pkg_check_modules(XCB xcb)
  pkg_check_modules(X11_XCB x11-xcb)
  if(Wayland_FOUND)
    target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_WAYLAND_KHR)
    target_include_directories(vulkan_objects PUBLIC ${Wayland_INCLUDE_DIRS})
  endif()
  if(XCB_FOUND)
    target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_XCB_KHR)
    target_include_directories(vulkan_objects PUBLIC ${XCB_INCLUDE_DIRS})
    target_link_libraries(vulkan_objects PUBLIC ${XCB_LIBRARIES} ${X11_LIBRARIES} ${X11_XCB_LIBRARIES}) # required for glfw_xcb_hack.cpp
  endif()
  option(VULKAN_OBJECTS_ENABLE_X11 "Enables X11. Don't. Use XCB." OFF) # https://stackoverflow.com/questions/79583727/how-to-avoid-macros-from-x11-polluting-other-third-party-code
  if(X11_FOUND AND VULKAN_OBJECTS_ENABLE_X11)
    target_compile_definitions(vulkan_objects PUBLIC VK_USE_PLATFORM_XLIB_KHR)
    #target_link_libraries(vulkan_objects PUBLIC ${X11_LIBRARIES})
  endif()
endif()

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

add_subdirectory(generate)
foreach(generated_source IN LISTS generated_sources)
  add_generated_file("${CMAKE_CURRENT_SOURCE_DIR}/${generated_source}")
endforeach()

# Optional dependencies added for user convenience, even though vulkan_objects
# doesn't strictly need them. Slang and Shaderc are both shader compilers -
# Shaderc is simpler for GLSL while Slang supports modules but is more verbose.
# VVL and Shaderc share SPIRV-Tools, SPIRV-Headers and glslang, fetched here to
# avoid duplication.

if (VULKAN_OBJECTS_FETCH_VVL OR VULKAN_OBJECTS_FETCH_SLANG OR VULKAN_OBJECTS_FETCH_SHADERC)
  include(FetchContent)
  
  if (NOT TARGET SPIRV-Tools)
    # SPIRV-Headers: SPIR-V specification headers
    FetchContent_Declare(
      SPIRV-Headers
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
      GIT_TAG ${VULKAN_OBJECTS_SPIRV_HEADERS_TAG}
      GIT_SHALLOW TRUE
      OVERRIDE_FIND_PACKAGE
    )
    
    # SPIRV-Tools: SPIR-V binary manipulation and validation  
    FetchContent_Declare(
      SPIRV-Tools
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
      GIT_TAG ${VULKAN_OBJECTS_SPIRV_TOOLS_TAG}
      GIT_SHALLOW TRUE
      OVERRIDE_FIND_PACKAGE
    )
    
    set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SPIRV-Headers SPIRV-Tools)
  endif()
  
  if (NOT TARGET glslang)
    # glslang: GLSL/HLSL front-end compiler
    FetchContent_Declare(
      glslang
      GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
      GIT_TAG ${VULKAN_OBJECTS_GLSLANG_TAG}
      GIT_SHALLOW TRUE
      OVERRIDE_FIND_PACKAGE
    )
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_JS OFF CACHE BOOL "" FORCE)
    set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
    set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glslang)
  endif()
endif()

if(VULKAN_OBJECTS_FETCH_VVL)
  # Vulkan Validation Layers with manual dependency management
  # We provide SPIRV-Tools, SPIRV-Headers, glslang from the shared section above,
  # and Vulkan-Headers is already fetched at the top of this file.
  # We only need to fetch Vulkan-Utility-Libraries here.
  include(FetchContent)
  
  # Fetch Vulkan-Utility-Libraries
  if (NOT TARGET Vulkan-UtilityLibraries)
    FetchContent_Declare(
      Vulkan-Utility-Libraries
      GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Utility-Libraries.git
      GIT_TAG ${VULKAN_OBJECTS_SPEC_TAG}
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Vulkan-Utility-Libraries)
  endif()
  
  # Fetch VVL itself
  FetchContent_Declare(
    vvl
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-ValidationLayers.git
    GIT_TAG ${VULKAN_OBJECTS_SPEC_TAG}
    GIT_SHALLOW TRUE
  )
  
  # Disable UPDATE_DEPS - we provide dependencies manually via FetchContent
  set(UPDATE_DEPS OFF)
  set(BUILD_TESTS_backup ${BUILD_TESTS})
  set(BUILD_TESTS OFF)
  set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
  
  FetchContent_MakeAvailable(vvl)
  set(BUILD_TESTS ${BUILD_TESTS_backup})
  add_dependencies(vulkan_objects vvl)
  target_compile_definitions(vulkan_objects PRIVATE VVL_DEVELOP_PATH="$<TARGET_FILE_DIR:vvl>")
endif()

if(VULKAN_OBJECTS_FETCH_VMA)
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
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vma_static PRIVATE -Wno-nullability-extension -Wno-nullability-completeness)
  endif()
endif()

if(VULKAN_OBJECTS_FETCH_SLANG)
  # Reuse SPIRV-Tools, SPIRV-Headers from the shared section above. Try to fetch
  # only the submodules slang actually needs (skip spirv-* that we already
  # provide, glslang, imgui, glm, etc.). This should avoid wasting people's
  # quotas and time. Otherwise it's a very slow 1GB+ bunch of clones.
  include(FetchContent)
  FetchContent_Declare(
    slang
    GIT_REPOSITORY https://github.com/shader-slang/slang.git
    GIT_TAG ${VULKAN_OBJECTS_SLANG_TAG}
    GIT_SHALLOW TRUE

    # Deny-list attempt (doesn't work)
    #GIT_CONFIG
    #  "submodule.external/spirv-tools.active=false"
    #  "submodule.external/spirv-headers.active=false"
    #  "submodule.external/glslang.active=false"
    #  "submodule.external/imgui.active=false"
    #  "submodule.external/glm.active=false"
    #  "submodule.external/slang-rhi.active=false"
    #  "submodule.external/vulkan.active=false"
    #  "submodule.external/tinyobjloader.active=false"
    #  "submodule.external/metal-cpp.active=false"
    #  "submodule.external/WindowsToolchain.active=false"
    #  "submodule.external/slang-binaries.active=false"

    # Alternative difficult-to-maintain allow-list
    # TODO: request a SLANG_ENABLE_FIDDLE to avoid external/lua?
    GIT_SUBMODULES external/unordered_dense external/miniz external/lz4 external/lua
  )
  set(SLANG_USE_SYSTEM_VULKAN_HEADERS ON CACHE BOOL "" FORCE)
  set(SLANG_USE_SYSTEM_SPIRV_HEADERS ON CACHE BOOL "" FORCE)
  set(SLANG_USE_SYSTEM_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
  set(SLANG_USE_SYSTEM_GLSLANG ON CACHE BOOL "" FORCE)
  set(SLANG_SPIRV_HEADERS_INCLUDE_DIR "${spirv-headers_SOURCE_DIR}/include" CACHE PATH "" FORCE)
  set(SLANG_ENABLE_DXIL OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANGD OFF CACHE BOOL "" FORCE)

  # slang needs a compiler to build its 'standard modules'
  set(SLANG_ENABLE_SLANGC ON CACHE BOOL "" FORCE)
  if(NOT SLANG_ENABLE_SLANGC)
    set(SLANG_GENERATORS_PATH "${CMAKE_BINARY_DIR}/generators/Debug/bin" CACHE PATH "" FORCE)
  endif()

  set(SLANG_ENABLE_SLANGRT OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANG_GLSLANG ON CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_REPLAYER OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF CACHE BOOL "" FORCE)
  set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE) # only needed for host code gen

  option(SLANG_GLSLANG_ALIAS_WORKAROUND "Workaround for slang's broken alias that conflicts with FetchContent targets" ON)
  if(SLANG_GLSLANG_ALIAS_WORKAROUND)
    message(STATUS "Applying slang glslang alias workaround")
    FetchContent_GetProperties(slang)
    if(NOT slang_POPULATED)
      FetchContent_Populate(slang)
      # Fix slang's broken alias that conflicts with FetchContent targets
      file(READ "${slang_SOURCE_DIR}/CMakeLists.txt" _slang_cmake)
      string(REPLACE 
        "add_library(glslang ALIAS glslang::glslang)"
        "if(NOT TARGET glslang)\n        add_library(glslang ALIAS glslang::glslang)\n    endif()"
        _slang_cmake "${_slang_cmake}")
      file(WRITE "${slang_SOURCE_DIR}/CMakeLists.txt" "${_slang_cmake}")
      add_subdirectory("${slang_SOURCE_DIR}" "${slang_BINARY_DIR}")
    endif()
  else()
    FetchContent_MakeAvailable(slang)
  endif()
  target_compile_options(slang PRIVATE -Wno-overloaded-virtual) # ideally upstreamed
  target_compile_options(core PRIVATE -Wno-overloaded-virtual) # not "slang-core" but just "core"
  target_link_libraries(vulkan_objects PUBLIC slang)
endif()

if(VULKAN_OBJECTS_FETCH_SHADERC)
  # Shaderc (GLSL/HLSL to SPIR-V compiler). Dependencies (SPIRV-Tools,
  # SPIRV-Headers, glslang) are provided by the shared section above.
  include(FetchContent)
  set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
  set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
  set(SHADERC_SKIP_INSTALL ON CACHE BOOL "" FORCE)
  set(SHADERC_SKIP_COPYRIGHT_CHECK ON CACHE BOOL "" FORCE)
  set(SHADERC_ENABLE_WERROR_COMPILE OFF CACHE BOOL "" FORCE)
  
  # Fix shaderc bug: SHADERC_ENABLE_SHARED_CRT defaults to OFF, causing /MT (static CRT)
  # This breaks DLL builds when glslang uses /MD: "error LNK2019: unresolved external symbol __imp_exp2"
  set(SHADERC_ENABLE_SHARED_CRT ON CACHE BOOL "" FORCE)
  
  # Control building of glslc executable for offline shader compilation
  if(NOT VULKAN_OBJECTS_SHADERC_BUILD_EXECUTABLES)
    set(SHADERC_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(
    shaderc
    GIT_REPOSITORY https://github.com/google/shaderc.git
    GIT_TAG ${VULKAN_OBJECTS_SHADERC_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(shaderc)
  
  # Workaround for shaderc failing to export its include directories
  target_include_directories(shaderc_combined INTERFACE "${shaderc_SOURCE_DIR}/libshaderc/include")

  # Statically linked shaderc_combined
  target_link_libraries(vulkan_objects PUBLIC shaderc_combined)
endif()

if(VULKAN_OBJECTS_FETCH_GLFW)
  # GLFW for window/surface creation and XCB workaround
  include(FetchContent)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG ${VULKAN_OBJECTS_GLFW_TAG}
    GIT_SHALLOW TRUE
  )
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

# GLFW XCB workaround library - see https://github.com/glfw/glfw/issues/1061
# GLFW does not expose native XCB handles, only X11/Xlib. This library provides
# glfwGetXCBConnection/Visual/Window functions. Must be compiled separately to
# avoid X11 macro pollution (Success, None, etc.) leaking into other headers.
# Can be removed once GLFW adds native XCB support.
if((TARGET glfw) AND XCB_FOUND)
  add_library(vulkan_objects_glfw STATIC src/glfw_xcb_hack.cpp)
  target_link_libraries(vulkan_objects_glfw PUBLIC vulkan_objects glfw)
  target_link_libraries(vulkan_objects_glfw PUBLIC ${XCB_LIBRARIES} ${X11_LIBRARIES} ${X11_XCB_LIBRARIES})
endif()

option(VULKAN_OBJECTS_TESTING_BUILD_ALL "Require all test dependencies (VVL, VMA, etc) to build tests" ON)

if((NOT VULKAN_OBJECTS_TESTING_BUILD_ALL AND VULKAN_OBJECTS_FETCH_VMA) OR 
   (VULKAN_OBJECTS_FETCH_VVL AND VULKAN_OBJECTS_FETCH_VMA))
  if(BUILD_TESTING)
    option(BUILD_VULKAN_OBJECTS_TESTING "Enable vulkan_objects testing" ON)
    if(BUILD_VULKAN_OBJECTS_TESTING)
      enable_testing()
      add_subdirectory(test)
    endif()
  endif()
endif()
