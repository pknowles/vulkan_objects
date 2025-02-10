
> [!CAUTION]
> Incomplete. This doesn't even compile yet.

# vko: Vulkan Objects

This library is an RAII wrapper around some core parts of the vulkan SDK. It is
primarily a thin API wrapper, not an engine or rendering abstraction. That said,
there are some optional utilities to make very specific but common operations
easy to write. Naturally, these are layered on top and in separate headers.

The aims are:

1. Dependencies are implied by the language

   No default initialization. Delayed initialization would allow you to create
   an object before its dependencies are created or even in scope. This would
   make using the library ambiguous and error prone. For example, you can't
   create a VkCommandPool before a VkDevice and by forcing initialization a user
   will immediately be reminded to create the VkDevice first.

   If it's truly needed there is always std::optional and std::unique_ptr, which
   better show the intent of a nullable object.

2. Objects are general and have minimal dependencies

   Expose the full featureset of the API, but make shortcuts and special cases
   easy to write. Objects should be reusable and not impose unnecessary
   limitations. A big part of this is the single-responsibility principle.

3. Simple, singular implementation

   Supporting older versions and multiple ways to do things for different edge
   cases is hard. I'm only one person. I'll pick one way and do it well,
   hopefully without limiting important features.

4. Lifetime and ownership is well defined

   Standard RAII: out of scope cleanup, no leaks, help avoid dangling pointers,
   be safe knowing if you have a handle the object is valid and initialized.
   Most objects are move-only and not copyable. This matches the API, e.g. you
   can't copy a VkDevice.

## Issues

- Some `vkCreate*` calls are plural but have singular destruction. E.g.
  `vkCreateGraphicsPipelines` -> `vkDestroyPipeline`. By default, singular
  constructors will be created. If needed, a `makePipelines()` wrapper can be
  added to allow the driver to create these objects in batches, before splitting
  them to individual C++ objects.
- Some `vkCreate*` calls are plural and have plural destruction, e.g.
  `vkAllocateCommandBuffers` -> `vkFreeCommandBuffer`. Ideally these would be
  modelled with a plural `CommandBuffers` container. On balance being able to
  split up ownership with an individual `CommandBuffer` object is more useful.
  Thus, singular objects will be the norm until someone tells me this impacts
  perf significantly. Nothing prevents plural objects being added in the future.
- Some `vkCreate*` have no destruction. E.g. `vkCreateDisplayModeKHR`. \*shruggie\*


## Error handling

Exceptions

C++ is really lacking here. IMO it took so long for us to even get move
semantics and during that time people understandably got the wrong idea about
the language and the workarounds gave it a bad reputation. There is
`std::expected` and `std::error_code`, but they don't help with constructors.

- We must have constructors for the compiler to help us avoid delayed initialization.
- Constructors must be able to fail and the only way for that to happen is exceptions.

See:

- [De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable - Herb Sutter CppCon 2019](https://www.youtube.com/watch?v=ARYP83yNAWk)
- [Exceptionally Bad: The Misuse of Exceptions in C++ & How to Do Better - Peter Muldoon - CppCon 2023](https://www.youtube.com/watch?v=Oy-VTqz1_58)
- [Are exceptions in C++ really slow?](https://stackoverflow.com/questions/13835817/are-exceptions-in-c-really-slow)

With that decision made, we need improved tooling. Particularly smooth and
intuitive experience debugging IDEs and debuggers to be able to break for
specific exception categories. Don't throw the baby out with the bathwater.

## Generated code



# volk vulkan loader

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 21ceafb55b7ca55b15aee758a8541c06e29780cd # 1.3.270
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

