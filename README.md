
# vko: Vulkan Objects

Self-contained Vulkan 3D Graphics API provider and thin RAII wrapper in C++.

- Dependency and lifetime design validation at compile time
- All-in-one: Generated from the Vulkan spec directly, optional loader included.
- Pluggable: Consumes native API types for easy integration. Use utilities you
  want; replace those you don't.

There are many optional extras to make common operations easy to write.
Naturally, these are layered and in separate headers to make ignoring them or
overriding and specialization easy. This is not an engine or rendering
abstraction. It does not suck you into an ecosystem.

TLDR by example:

```cpp
#include <vko/handles.hpp>
...
vko::VulkanLibrary  library;  // Cross platform, just dlopen()
vko::GlobalCommands globalCommands(library.loader());  // bootstrap with any vkGetInstanceProcAddr
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

// Optional glfw integration
vko::SurfaceKHR surface = vko::glfw::makeSurface(...);
```

For more example code, see [test/src/test.cpp](test/src/test.cpp). It includes ray tracing✨!

The aims are:

1. Dependencies are implied by the language

   No default initialization. Delayed initialization would allow you to create
   an object before its dependencies are created or even in scope. This makes
   code ambiguous and error prone. For example, you can't create a VkCommandPool
   before a VkDevice and by forcing initialization a user will immediately be
   reminded to create the VkDevice first. It allows the compiler to help us
   design better.

   If it's truly needed there is always `std::optional` and `std::unique_ptr`.
   Safety first, RAII by default, that you can override in specific places.

2. Objects are general, have minimal dependencies and don't suck you into an
   ecosystem

   For example, it's common to pass around an everything "context" object
   containing the VkDevice, maybe an allocator or queues. This is convenient,
   but then you have to have one of these objects everywhere. In contrast,
   objects here are constructed from native vulkan objects.

   The aim is to expose the full featureset of the API near-verbatim. Objects
   should be reusable and pluggable. A big part of this is sticking to the
   single-responsibility principle.

   Shortcuts are added but special cases should be easy to override and write
   without shortcuts. This is done by layering utilities on top. Higher level
   objects can be replaced without losing much. No all-or-nothing monolith
   objects.

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

6. No effort plumbing

   Use existing structures to hold data. E.g. there are already many
   `*CreateInfo` structs that can be taken as an argument. No need to
   unpack/forward/pack arguments. This is the single definition rule.

   Once objects are allocated, use the Vulkan C API for certain operations. I.e.
   there is no wrapping raw `vk*()` calls as members on objects. It might look
   right to add a `drawIndexed()` (calling `vkCmdDrawIndexed`) call on a
   `CommandBuffer` object, but maybe that's never used because the raw
   `VkCommandBuffer` is passed to some higher level object and then
   `CommandBuffer` doesn't have to "know" about drawing.

   Vulkan comes with an official C++
   [vulkan.hpp](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vulkan/vulkan.hpp)
   and
   [vulkan_raii.hpp](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vk_raii_ProgrammingGuide.md)
   that do this. They're heavyweight in terms of line count. No really, 15MB+ of
   pure header files. They also mix in helpers, which are great, but there's no
   layering to pick just what you want to use. Admittedly, it's nice to type `.`
   and have your IDE auto-complete methods.

## Building

Cmake and C++20 is required. Currently the following dependencies are
automatically added with FetchContent:

- [Vulkan Headers](https://github.com/KhronosGroup/Vulkan-Headers)

Some other common ones can optionally be added with the below options.

| CMake Options                   | Description                                                |
| ------------------------------- | ---------------------------------------------------------- |
| `VULKAN_OBJECTS_FETCH_VVL`      | `ON` fetches [Vulkan Validation Layers][vvl] source (big!) |
| `VULKAN_OBJECTS_FETCH_VMA`      | `ON` fetches [Vulkan Memory Allocator][vma] source         |
| `VULKAN_OBJECTS_FETCH_SLANG`    | `ON` fetches [Slang Compiler][slang] source                |
| `VULKAN_OBJECTS_SPEC_OVERRIDE`  | `/path/to/vk.xml`                                          |
| `VULKAN_OBJECTS_SPEC_TAG`       | `<default version>` if not `_OVERRIDE`                     |
| `VULKAN_OBJECTS_VMA_TAG`        | `<default version>` if `_FETCH_VMA`                        |

[vvl]: https://github.com/KhronosGroup/Vulkan-ValidationLayers
[slang]: https://github.com/shader-slang/slang
[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator

## Issues

- Some `vkCreate*` calls are plural but have singular destruction. E.g.
  `vkCreateGraphicsPipelines` -> `vkDestroyPipeline`. Some `vkCreate*` calls are
  plural and have plural destruction, e.g. `vkAllocateCommandBuffers` ->
  `vkFreeCommandBuffer`. Sticking with matching the API principle, these are
  primarily modelled as a vector of handles. Singular objects are added for
  convenience too. These wrappers for these are currently hand coded until I can
  spend some time to come up with a nicer way.
- Some `vkCreate*` have no destruction. E.g. `vkCreateDisplayModeKHR`. \*shruggie\*

## Error handling

Exceptions

C++ is really lacking here, re. RTTI. IMO it took so long for us to even get
move semantics (we still have no std::ranges::output_range) and during that time
people understandably got the wrong idea about the language and the workarounds
gave it a bad reputation. There is `std::expected` and `std::error_code`, but
they don't help with constructors.

- We must have constructors for the compiler to help us avoid delayed initialization.
- Constructors must be able to fail and the only way for that to happen is exceptions.

See:

- [De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable - Herb Sutter CppCon 2019](https://www.youtube.com/watch?v=ARYP83yNAWk)
- [Exceptionally Bad: The Misuse of Exceptions in C++ & How to Do Better - Peter Muldoon - CppCon 2023](https://www.youtube.com/watch?v=Oy-VTqz1_58)
- [Are exceptions in C++ really slow?](https://stackoverflow.com/questions/13835817/are-exceptions-in-c-really-slow)

With that decision made, we need improved tooling. Particularly smooth and
intuitive experience debugging IDEs and debuggers to be able to break for
specific exception categories. I also recognise big initializer lists are ugly
AF, but they're needed. Don't throw the baby out with the bathwater.

## Generated code

Some code is generated directly from vk.xml - see files prefixed with `gen_*`.
The loader in particularly needs this (unless you're using your own). This code
is generated to the build directory and not checked in. Generation is done with
C++ using [pugixml](https://github.com/zeux/pugixml), so there is no dependency
on Python or other tools.
