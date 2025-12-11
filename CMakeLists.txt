# Copyright (c) 2024-2025 Pyarelal Knowles, MIT License

cmake_minimum_required(VERSION 3.20)

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
  src/glfw_xcb_hack.cpp
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
set(VULKAN_OBJECTS_SLANG_TAG "v2025.6.3" CACHE STRING "GIT tag from https://github.com/shader-slang/slang/tags")
set(VULKAN_OBJECTS_SHADERC_TAG "v2025.5" CACHE STRING "GIT tag from https://github.com/google/shaderc/tags")
option(VULKAN_OBJECTS_FETCH_VVL "Include Vulkan Validation Layers with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_VMA "Include Vulkan Memory Allocator with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_SLANG "Include Slang compiler with FetchContent()" OFF)
option(VULKAN_OBJECTS_FETCH_SHADERC "Include Shaderc compiler with FetchContent()" OFF)
option(VULKAN_OBJECTS_SHADERC_BUILD_EXECUTABLES "Build glslc command-line compiler (for offline shader compilation)" OFF)

if(VULKAN_OBJECTS_SPEC_OVERRIDE)
  set(VULKAN_SPEC_XML ${VULKAN_OBJECTS_SPEC_OVERRIDE})
else()
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

# Just for convenience because users would probably want these optional libs
# anyway. vulkan_objects includes wrappers/helpers for them but they
if(VULKAN_OBJECTS_FETCH_VVL)
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
  # TODO: would be nice if UPDATE_DEPS doesn't duplicate the giant spirv-tools
  # binaries to the install directory :(
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
  include(FetchContent)
  FetchContent_Declare(
    slang
    GIT_REPOSITORY https://github.com/shader-slang/slang.git
    GIT_TAG ${VULKAN_OBJECTS_SLANG_TAG}
    GIT_SHALLOW TRUE
  )
  set(SLANG_USE_SYSTEM_VULKAN_HEADERS ON CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_DXIL OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANGD OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANGC OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANGRT OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANG_GLSLANG OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_REPLAYER OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(slang)
  target_link_libraries(vulkan_objects PUBLIC slang)
endif()

if(VULKAN_OBJECTS_FETCH_SHADERC)
  # ============================================================================
  # Shaderc (GLSL/HLSL to SPIR-V compiler)
  # ============================================================================
  # Shaderc is a collection of tools and libraries for compiling GLSL/HLSL
  # shaders to SPIR-V. It wraps glslang (the Khronos reference compiler) and
  # SPIRV-Tools to provide a more stable API and better build integration.
  #
  # DEPENDENCY HANDLING:
  # Shaderc requires three core dependencies from Khronos:
  #   - SPIRV-Headers: SPIR-V specification headers
  #   - SPIRV-Tools:   SPIR-V binary manipulation and validation
  #   - glslang:       GLSL/HLSL front-end compiler
  #
  # Unlike VVL which uses UPDATE_DEPS to auto-fetch dependencies via git-sync,
  # we manually fetch these dependencies with FetchContent to:
  #   1. Use specific named release tags (not commit hashes)
  #   2. Control versions explicitly for reproducible builds
  #   3. Minimize what gets built (tests, examples, etc.)
  #   4. Provide a pure CMake/FetchContent experience
  #
  # NOTE: If VULKAN_OBJECTS_FETCH_VVL is also enabled, there may be duplicate
  # SPIRV-Tools/SPIRV-Headers fetches. The "if (NOT TARGET ...)" guards prevent
  # duplicate builds, with whichever is configured first taking precedence.
  # This can lead to version mismatches. TODO: Consider consolidating these
  # dependencies or adding explicit version conflict detection.
  #
  # VERSIONS (as of Dec 2025):
  #   - Shaderc:       v2025.5 (configurable via VULKAN_OBJECTS_SHADERC_TAG)
  #   - SPIRV-Tools:   v2025.4
  #   - glslang:       16.1.0
  #   - SPIRV-Headers: vulkan-sdk-1.4.335.0 (only has SDK-versioned tags)
  #
  # LINKING:
  # We link to shaderc_combined, which is a static library containing shaderc
  # and all its dependencies (SPIRV-Tools, glslang) in one convenient package.
  # Note: Shaderc also builds separate shaderc and shaderc_shared libraries,
  # but we only use shaderc_combined to minimize linking complexity.
  #
  # OFFLINE SHADER COMPILATION:
  # By default, we skip building the glslc executable (command-line compiler).
  # To enable offline shader compilation, set:
  #   -DVULKAN_OBJECTS_SHADERC_BUILD_EXECUTABLES=ON
  # This builds the glslc tool for compiling shaders at build-time instead of
  # runtime. The executable will be located at:
  #   ${FETCHCONTENT_BASE_DIR}/shaderc-build/glslc/<config>/glslc[.exe]
  # Note: On MSVC, glslc may have linking issues with some math functions
  # (exp2, floor, log2, atof). The library works fine; only the exe is affected.
  # ============================================================================
  include(FetchContent)
  
  # Fetch and build shaderc's dependencies using MakeAvailable
  # This creates the targets, so shaderc will detect them and skip adding them again
  
  if (NOT TARGET SPIRV-Tools)
    FetchContent_Declare(
      SPIRV-Headers
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
      GIT_TAG vulkan-sdk-1.4.335.0
      GIT_SHALLOW TRUE
    )
    FetchContent_Declare(
      SPIRV-Tools
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
      GIT_TAG v2025.4
      GIT_SHALLOW TRUE
    )
    # Configure SPIRV-Tools options before making it available
    set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SPIRV-Headers SPIRV-Tools)
  endif()
  
  if (NOT TARGET glslang)
    FetchContent_Declare(
      glslang
      GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
      GIT_TAG 16.1.0
      GIT_SHALLOW TRUE
    )
    # Configure glslang options before making it available
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(glslang)
  endif()
  
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

  # Use shaderc_combined - it's a static library with all dependencies included
  target_link_libraries(vulkan_objects PUBLIC shaderc_combined)
endif()

if(VULKAN_OBJECTS_FETCH_VVL AND VULKAN_OBJECTS_FETCH_VMA)
  if(BUILD_TESTING)
    option(BUILD_VULKAN_OBJECTS_TESTING "Enable vulkan_objects testing" ON)
    if(BUILD_VULKAN_OBJECTS_TESTING)
      enable_testing()
      add_subdirectory(test)
    endif()
  endif()
endif()
