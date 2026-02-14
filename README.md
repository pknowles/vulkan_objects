
# vko: Vulkan Objects

Self-contained C++ Vulkan 3D Graphics API provider and thin RAII wrapper.

Just clone, build and start rendering with Vulkan.

- Compile-time dependency and lifetime design validation
- All-in-one: just source, generated from the Vulkan spec directly
- Pluggable: consumes C API types for easy integration
- Headers (official) and optional loader included; no Vulkan SDK required
- Optional compiler (shaderc and slang)

Use utilities you want; replace those you don't. These are layered and in
separate headers to make overriding and specialization easy. This is not an
engine or rendering abstraction. It does not suck you into an ecosystem.

`vulkan_objects` core Vulkan handles provide lifetime safety without wrapping
the API.
[vulkan_raii.hpp](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vk_raii_ProgrammingGuide.md)
has similar features but slightly different goals. It is actually provided by
`vulkan_objects` too. Use it instead or as well if it fits.

TL;DR by example:

```cmake
# CMake
set(VULKAN_OBJECTS_FETCH_VMA ON)  # Optional
add_subdirectory(./path/to/vulkan_objects)
target_link_libraries(my_vulkan_app vulkan_objects)
```

```cpp
#include <vko/handles.hpp>
...
vko::VulkanLibrary  library;  // Cross platform, just dlopen()
vko::GlobalCommands globalCommands(library.loader());  // Bootstrap with any vkGetInstanceProcAddr
vko::Instance       instance(globalCommands, VkInstanceCreateInfo{...});  // Standard CreateInfo structs
VkPhysicalDevice    physicalDevice(vko::toVector(instance.vkEnumeratePhysicalDevices, instance)[0]);
vko::Device         device(instance, physicalDevice, VkDeviceCreateInfo{...});

// Standard C API, no vulkan.hpp/vulkan_raii.hpp overhead
device.vkDeviceWaitIdle(device);

// BYO types. vko::Instance and vko::Device are both function tables and VkInstance/VkDevice.
// There are overloads if yours are separate:
vko::Image image(device, VkImageCreateInfo{...})
vko::Image image((vko::DeviceCommands&)device, (VkDevice)device, VkImageCreateInfo{...})
// ... but you may want vko::BoundImage<vko::vma::Allocator> :)

vko::Image                image;  // Error: no default construction
std::optional<vko::Image> image;  // Intentional

// Optional glfw integration
vko::SurfaceKHR surface = vko::glfw::makeSurface(...);
```

For more example code, see
[test/src/hello_triangle.cpp](test/src/hello_triangle.cpp). It includes ray
tracing✨!

## Quick Reference

See [`hello_triangle.cpp`](test/src/hello_triangle.cpp) for usage.

Remember, everything is composable. You can create objects quickly from raw
vulkan types. You aren't forced into using anything. If it doesn't quite fit
your use-case, check the implementation and compose your own. Some templated
utilities may even work with your own types.

```cpp
// Core types
vko::VulkanLibrary
vko::GlobalCommands
vko::Instance
vko::Device : public DeviceHandle, public DeviceCommands {};

// Vulkan Handles, owning, constructors simply take *CreateInfo{}
vko::ImageView view = vko::ImageView(device, VkImageViewCreateInfo{ ... });
vko::CommandBuffer
vko::Buffer         // An unbound VkBuffer; use DeviceBuffer instead
... many more

// Memory Management
vko::vma::Allocator allocator(globalCommands, instance, physicalDevice, device,
                              VK_API_VERSION_1_4,
                              VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
vko::BoundBuffer buffer = vko::BoundBuffer<uint32_t>(device, 1024, VK_BUFFER_USAGE_*, VK_MEMORY_PROPERTY_*, allocator)
vko::BoundImage
vko::DeviceBuffer  // A BoundBuffer with .address()

// Queues and Synchronization
// A timeline vko::Semaphore with a VkQueue
vko::TimelineQueue queue(device, queueFamilyIndex, queueIndex)
// A std::promise of a timeline semaphore value
vko::SubmitPromise submitPromise = queue.submitPromise()
// A shared future semaphore value - .hasValue(), .ready(), .wait(), .waitUntil(), .waitFor()
vko::SemaphoreValue nextSubmitSemaphoreValue = submitPromise.futureValue()
queue.submit(device, waitInfos, commandBuffer, submitPromises, timelineSemaphoreStageMask, extraSignalInfos);
bool signalled = nextSubmitSemaphoreValue.ready();
nextSubmitSemaphoreValue.waitFor(device, std::chrono::seconds(123));

// Command Recording
vko::ImmediateCommandBuffer cmd(device, pool, queue);  // RAII: records in scope, submits+waits on destruction. Convenient but not async/stalls GPU.
vko::CyclingCommandBuffer cmd(device, queue); // vko::CommandBuffer allocator from an internal vko::CommandPool; submit() to a non-owning vko::TimelineQueue reference

// Staging Memory, composable, hard to misuse, see staging_memory.hpp
using StagingStream = vko::StagingStream<vko::vma::RecyclingStagingPool>;
StagingStream stream(...);
vko::DeviceBuffer<int> buf = vko::upload(stream, device, allocator, std::array<int>{1, 2, 3}, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
// vko::cmdMemoryBarrier() and use buf somewhere
stream.submit();

// Queries and Profiling, see query_pool.hpp
vko::QueryStream<uint64_t, VK_QUERY_TYPE_TIMESTAMP, 64> queryStream;
vko::ScopedQuery // RAII begin/end query
auto future = vko::cmdWriteTimestamp(device, cmd, queryStream, stage)
queries.endBatch(submitPromise.futureValue()) // recycling is blocked without
queue.submit(submitPromise, cmd)
uint64_t timestamp = future.get();

// Swapchain, optional GLFW utils
vko::glfw::physicalDevicePresentationSupport()
vko::glfw::platformSurfaceExtension()
vko::glfw::ScopedInit
vko::glfw::Window window  = vko::glfw::makeWindow(800, 600, "Vulkan Window");
vko::SurfaceKHR   surface = vko::glfw::makeSurface(instance, platformSupport, window.get());
vko::Swapchain

// Slang Shaders
vko::slang::GlobalSession
vko::slang::Session
vko::slang::Module
vko::slang::Composition
vko::slang::Program
vko::slang::Code

// GLSLC Shaders
shaderc::CompileOptions 
shaderc::Compiler
options.SetIncluder(std::make_unique<vko::shaderc::FileIncluder>(...));
vko::shaderc::SpirvBinary binary(compiler, source, shaderc_glsl_compute_shader, shaderPath.string(), "main", options);

// Bindings
vko::BindingsAndFlags{{VkDescriptorSetLayoutBinding{}}, {0}};
vko::SingleDescriptorSet
vko::WriteDescriptorSetBuilder writes;
writes.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(...)
device.vkUpdateDescriptorSets(device, writes.writes().size(), writes.writes().data(), 0U, nullptr);

// Ray Tracing
vko::simple::RayTracingPipeline
vko::simple::HitGroupHandles
vko::simple::ShaderBindingTables
vko::as::SimpleGeometryInput
vko::as::Input = vko::as::createBlasInput()
               = vko::as::createTlasInput()
vko::as::Sizes
vko::as::AccelerationStructure
vko::as::cmdBuild()

// NVIDIA DLSS
vko::ngx::requiredInstanceExtensions()
vko::ngx::requiredDeviceExtensions()
vko::ngx::FeatureDiscovery
vko::ngx::ScopedInit
vko::ngx::CapabilityParameter
vko::ngx::OptimalSettings

// Misc utils
vko::check(VkResult)
vko::get(device.vkGetDeviceQueue, device, queueFamilyIndex, 0) -> VkQueue
vko::toVector(instance.vkEnumeratePhysicalDevices, instance) -> std::vector<VkPhysicalDevice>
vko::chainPNext(nullptr, with<VkPresentIdKHR>(1U, &presentId), [&](auto pNext) { ... })
vko::imgui::ScopedGlfwInit
vko::imgui::ScopedVulkanInit
vko::imgui::Context
vko::implot::Context
vko::imgui::window() // ever forget when you call ImGui::End() if ImGui::Begin() returns false?
vko::cmdDynamicRenderingDefaults()
vko::cmdImageBarrier(..., ImageAccess src, ImageAccess dst)
vko::cmdMemoryBarrier(..., MemoryAccess src, MemoryAccess dst)
vko::setName() // VK_EXT_DEBUG_UTILS_EXTENSION_NAME
```

## Building

Cmake and C++20 is required. Currently the following dependencies are
automatically added with FetchContent:

- [Vulkan Headers](https://github.com/KhronosGroup/Vulkan-Headers)

Some other common ones can optionally be added with the below options.
These are currently added as dependencies of the `vulkan_objects` target for convenience.
Admittedly this is not accurate, doesn't quite sit right with me and may change.

| CMake Options                              | Description                                                |
| ------------------------------------------ | ---------------------------------------------------------- |
| `VULKAN_OBJECTS_FETCH_VVL`                 | `ON` fetches [Vulkan Validation Layers][vvl] source (big!) |
| `VULKAN_OBJECTS_FETCH_VMA`                 | `ON` fetches [Vulkan Memory Allocator][vma] source         |
| `VULKAN_OBJECTS_FETCH_SLANG`               | `ON` fetches [Slang Compiler][slang] source                |
| `VULKAN_OBJECTS_FETCH_SHADERC`             | `ON` fetches [Shaderc Compiler][shaderc] source            |
| `VULKAN_OBJECTS_FETCH_GLFW`                | `ON` fetches [GLFW][glfw] and builds XCB workaround library|
| `VULKAN_OBJECTS_SHADERC_BUILD_EXECUTABLES` | `ON` builds `glslc` executable (for offline compilation)   |
| `VULKAN_OBJECTS_SPEC_OVERRIDE`             | `/path/to/vk.xml` (ignores `_SPEC_TAG`)                    |
| `VULKAN_OBJECTS_SPEC_TAG`                  | `<default version>` if not `_OVERRIDE`                     |
| `VULKAN_OBJECTS_VMA_TAG`                   | `<default version>` if `_FETCH_VMA`                        |
| `VULKAN_OBJECTS_SLANG_TAG`                 | `<default version>` if `_FETCH_SLANG`                      |
| `VULKAN_OBJECTS_SHADERC_TAG`               | `<default version>` if `_FETCH_SHADERC`                    |
| `VULKAN_OBJECTS_GLFW_TAG`                  | `<default version>` if `_FETCH_GLFW`                       |

[vvl]: https://github.com/KhronosGroup/Vulkan-ValidationLayers
[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
[slang]: https://github.com/shader-slang/slang
[shaderc]: https://github.com/google/shaderc
[glfw]: https://github.com/glfw/glfw

**Note on GLFW XCB support:** GLFW does not expose native XCB handles (only
X11/Xlib), which causes X11 macro pollution (Success, None, etc.). The
`vulkan_objects_glfw_xcb` library provides `glfwGetXCBConnection/Visual/Window`
workarounds. This is compiled separately to isolate the X11 headers. See
[glfw/glfw#1061](https://github.com/glfw/glfw/issues/1061). External users can
either enable `VULKAN_OBJECTS_FETCH_GLFW` or otherwise provide a `glfw` target
themselves (preferred).

## Issues

- Some `vkCreate*` calls are plural but have singular destruction. E.g.
  `vkCreateGraphicsPipelines` -> `vkDestroyPipeline`. Some `vkCreate*` calls are
  plural and have plural destruction, e.g. `vkAllocateCommandBuffers` ->
  `vkFreeCommandBuffer`. Sticking with matching the API principle, these are
  primarily modelled as a vector of handles. Singular objects are added for
  convenience too. These wrappers for these are currently hand coded until I can
  spend some time to come up with a nicer way.
- Some `vkCreate*` have no destruction. E.g. `vkCreateDisplayModeKHR`. \*shruggie\*

## Design Philosophy

These design decisions reflect the tradeoffs my experience with C++ and Vulkan
has shown to be effective. They may not suit every project.

1. Dependencies are implied by the language

   No default initialization. Delayed initialization would allow you to create
   an object before its dependencies are created or even in scope. This makes
   code ambiguous and error prone. For example, you can't create a VkCommandPool
   before a VkDevice and by forcing initialization a user will immediately be
   reminded to create the VkDevice first. It allows the compiler to help us
   design better.

   If it's truly needed there is always `std::optional` (stack) and
   `std::unique_ptr` (heap). Safety first, RAII by default, that you can
   override in specific places.

2. Objects are general, composable, have minimal dependencies and don't suck you
   into an ecosystem

   For example, it's common to pass around an everything "context" object
   containing the VkDevice, maybe an allocator or queues. This is convenient,
   but then you have to have one of these objects everywhere. In contrast,
   objects here are constructed from native vulkan objects.

   The aim is to expose the full featureset of the API near-verbatim. Objects
   should be reusable and pluggable. A big part of this is sticking to the
   single-responsibility principle.

   Shortcuts are added but special cases should be easy to override and write
   without shortcuts. This is done by layering utilities on top. Higher level
   objects can be replaced without losing much. E.g. users can compose their own
   higher level objects from intermediate ones in this library. No
   all-or-nothing monolith objects.

   A difficulty is that function tables from the included loader need to be
   passed around to make vulkan API calls. To facilitate using your own function
   tables and not lock you into the ecosystem (yes, this is possible! e.g.
   [volk](https://github.com/zeux/volk)), many methods are templates that take a
   function table as the first parameter. For example. see the `device_commands`
   and `device_and_commands` concepts.

3. Simple, singular implementation

   Supporting older versions and multiple ways to do things for different edge
   cases is hard. I'm only one person. I'll pick one way and do it well,
   hopefully without limiting important features.

   This includes vulkan directly from
   https://github.com/KhronosGroup/Vulkan-Headers, just for `vk.xml`,
   `vulkan_core.h` and platform-specific headers. Handles are generated, so this
   library should always support the latest vulkan.

   This library includes its own vulkan function pointer loader, like
   [volk](https://github.com/zeux/volk), but because vulkan_core.h is included,
   there is no need to support different versions. It's all one thing. One
   exception is ifdefs for platform-specific types.

4. Lifetime and ownership is well defined

   Standard RAII: out of scope cleanup, no leaks, help avoid dangling pointers,
   be safe knowing if you have a handle then the object is valid and
   initialized. Most objects are move-only and not copyable. This matches the
   API, e.g. you can't copy a VkDevice.

5. Performance and data oriented

   Avoid forcing heap allocations on the user. Avoid copying memory around to
   restructure data. Instead, take pointers (i.e. `std::span`) already in vulkan
   API compatible ways and let the user decide whether to pay the cost or not.
   Use templates and avoid virtual functions.

6. No effort plumbing

   Use existing structures to hold data. E.g. there are already many
   `*CreateInfo` structs that can be taken as an argument. No need to
   unpack/forward/pack arguments. This is the single definition rule.

   Once objects are allocated, use the Vulkan C API for certain operations. I.e.
   there is no wrapping raw `vk*()` calls as members on objects. It might look
   right to add a `CommandBuffer::drawIndexed()` member to call
   `vkCmdDrawIndexed` or a `Device::createCommandPool()`, but that implies a
   command buffer needs to "know" about drawing and a device needs to "know"
   about command pools.

   Reduces cognitive load for those familiar with the C API and online examples.
   Use templates instead of large generated headers to improve IDE and compiler
   performance.

Vulkan comes with official
[vulkan.hpp](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vulkan/vulkan.hpp)
and
[vulkan_raii.hpp](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vk_raii_ProgrammingGuide.md)
which comprehensively wrap the API in a modern C++ style. Those bindings
reinterpret the Vulkan API around generated wrappers, member functions and
internal dispatch table pointers. Notably, IDE auto-complete is great for member
discoverability.

`vulkan_objects` takes a different approach. It uses the C API directly, adding
only ownership safety and lightweight incremental abstractions. The goal is to
make Vulkan safer to use, without sacrificing performance or redesigning Vulkan
in C++.

| `vulkan_raii.hpp`                          | `vulkan_objects`                     |
| ------------------------------------------ | ------------------------------------ |
| Wraps the entire Vulkan API in C++ classes | C++ ownership and safety only        |
| C++-idiomatic reinterpretation of Vulkan   | Preserves Vulkan’s C API shape       |
| Commands exposed as member functions       | Explicit `.vk*` on dispatch tables   |

## Error handling

**Exceptions**

- To allow the compiler to help us prevent delayed initialization we use
  constructors
- Constructors must be able to fail and the only way for that to happen is
  exceptions

Performance is a common exception concern, but they incur no runtime cost on the
success path. Object construction remains a direct `vkCreate*` call. You get to
code for the happy path. Any runtime cost/overhead is paid only in the failure
case, which should be incredibly rare, and is not as slow as it was 20 years
ago.

There are `std::expected` and `std::error_code`, but their purposes are
different and they don't replace RAII well, a concept the language was designed
around, while offering the same safety benefits, avoiding boilerplate and
plumbing.

See:

- [De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable - Herb Sutter CppCon 2019](https://www.youtube.com/watch?v=ARYP83yNAWk)
- [Exceptionally Bad: The Misuse of Exceptions in C++ & How to Do Better - Peter Muldoon - CppCon 2023](https://www.youtube.com/watch?v=Oy-VTqz1_58)
- [Are exceptions in C++ really slow?](https://stackoverflow.com/questions/13835817/are-exceptions-in-c-really-slow)

C++ and RAII tooling is still rough around the edges. This library accepts that
reality rather than designing around it; it will come. For debugging, it would
be useful to be able to see which variable is being destroyed in the stack and
to easily break for specific exception categories. Big constructor initializer
lists are ugly syntactically, but they're needed for RAII.

## Generated code

Some code is generated directly from vk.xml - see files prefixed with `gen_*`.
The loader in particularly needs this (unless you're using your own). This code
is generated to the build directory and not checked in. Generation is done with
C++ using [pugixml](https://github.com/zeux/pugixml), so there is no dependency
on Python or other tools.
