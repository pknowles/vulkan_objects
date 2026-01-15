# Vulkan Objects - Git History Squash Plan

This plan consolidates 73 commits into ~25 logical milestones following chronological development order. Each milestone represents a coherent, buildable feature or refactoring step.

---

## 1. Project Foundation & vko Namespace
**Commits:** [1-3] b887e5b, 89229f7, 4005284  
**Date Range:** Feb 2024

initialize vko project

Repository structure, MIT license, and vko namespace.
Library philosophy: thin RAII wrappers with low-overhead C++ exceptions.

---

## 2. Vulkan API Code Generation System
**Commits:** [4-11] b915638, 06696d0, 4f9727f, a441960, 52f1396, 745564f, 200d264, efa6559  
**Date Range:** Feb 2024

vulkan api code generation

C++ tool to parse vk.xml using pugixml and inja templates.
RAII handle generation, function pointer tables, and command structures.
Cross-platform dynamic loading (VK_NO_PROTOTYPES) via DynamicLibrary wrapper.
Organized generated API into instance-level and device-level categories.

---

## 3. Core RAII Architecture & Stability
**Commits:** [12-17] 6547534, 6c3b26f, b3d67b9, 2dc84ee, e7fc113, c4b3917  
**Date Range:** Feb-Mar 2024

core raii architecture and integration test suite

Base Handle template for Vulkan objects with specialized deleter logic.
Custom Exception and check() utilities for error reporting.
Integration test verifying Instance, Device, CommandPool creation.
Support for externally managed and destroy-only Vulkan objects.

---

## 4. Windowing & Surface Support
**Commits:** [18-21] 03958ae, bc687b0, f652f3f, 57386f1  
**Date Range:** Mar 2024

windowing, surface support, timeline queue, and swapchain management

GLFW and platform-specific Vulkan surface support (Win32, Xlib, Xcb).
Swapchain abstraction and command recording utilities.
TimelineQueue for synchronization and ImmediateCommandBuffer for one-off work.

---

## 5. Vulkan Validation Layers Integration
**Commits:** [22] 32dfe3b  
**Date Range:** Mar 2024

vulkan validation layers integration

CMake FetchContent support for VVL from source.

---

## 6. Vulkan Memory Allocator Integration
**Commits:** [23] 5a8589f  
**Date Range:** Mar 2024

vulkan memory allocator integration

VMA integration with BoundBuffer and BoundImage resource abstractions.

---

## 7. Slang Shader Compiler Integration
**Commits:** [24] 117c2fc  
**Date Range:** Mar 2024

slang shader compiler integration

Slang compiler dependency and basic rasterization pipeline test.

---

## 8. Ray Tracing Foundations
**Commits:** [25] d86e9df  
**Date Range:** Mar 2024

ray tracing objects

TLAS/BLAS building and acceleration structure management.
Ray tracing pipeline creation via Slang.
Binding utilities for RT pipelines.

---

## 9. API Standardization & C++20 Safety
**Commits:** [26-28] f5da68a, 13fd227, 5f78e60  
**Date Range:** Mar 2024

standardize api signatures and c++20 concepts

Constructor argument order: Device/Commands first, Allocator last.
device_commands and global_commands c++20 concepts for function table api.
Renamed Array to BoundBuffer.

---

## 10. RT & Compute Utilities
**Commits:** [29-30] 32e0308, bf66209  
**Date Range:** Mar 2024

ray tracing and compute utilities, XCB support

TLAS/BLAS creation helpers and compute recording utilities.
XCB support with GLFW compatibility hack.
Alternative to X11 header macro leaking.

---

## 11. NVIDIA NGX Integration
**Commits:** [31] 007844e  
**Date Range:** Mar 2024

nvidia ngx integration for dlss-rr

nv_ngx.hpp with RAII wrappers for NGX handles.
Remove VK_GEOMETRY_OPAQUE_BIT_KHR from default geometry flags.

---

## 12. Surface Architecture Refactor
**Commits:** [32] 1364d89  
**Date Range:** Mar 2024

refactor SurfaceKHR to support generic and platform-specific variants

Separated generic SurfaceKHR wrapper from platform-specific GLFW logic.
Generator produces SurfaceVariant union.

---

## 13. RT Pipeline Improvements & API Polish
**Commits:** [33-36] 4a4573f, 6f11ddc, 82da954, 6155dde  
**Date Range:** Mar 2024

rt pipeline improvements and api polish

C++20 concepts applied to handle constructors.
Fix shader group ordering bugs in RT pipelines.
DeviceBuffer specialization with device address support.
Generalized RT pipeline for arbitrary shader group configurations.
Custom entry point names for compute shaders.

---

## 14. Staging Memory System - Initial Implementation
**Commits:** [37-40] f35122c, a7b7f3a, e719da5, 94dbb49  
**Date Range:** Mar 2024

staging memory system

DedicatedStaging & RecyclingStagingPool allocators w/ TimelineQueue integration.
Type-safe staging uploads returning direct buffer mappings.
L-value ref-qualifiers on handle conversions to prevent use of temporaries.
Staging operation result lifetime is still rather unsafe/ambiguous.

---

## 15. ImGui RAII Wrappers
**Commits:** [41] d75df54  
**Date Range:** Nov 2024

imgui raii wrappers

imgui_objects.hpp and implot_objects.hpp with RAII classes for Vulkan-based
ImGui initialization.

---

## 16. Debugging & Validation Improvements
**Commits:** [42-43] 0499e6b df257fd
**Date Range:** Nov 2024

engaged() fix, debug msg cb update, header version bump

Size validation on BoundBuffer.
Fix engaged() logic in CommandBuffer.
Refactored GlobalDebugMessenger callback management.
Bumped Vulkan-Headers and VMA versions.
Replaced std::println with fprintf.

---

## 18. Staging Memory - Type Safety & ImGui Expansion
**Commits:** [44-46] 8e8d239, eb235e2, 2e391f0  
**Date Range:** Nov-Dec 2024

staging type safety and imgui expansion

uploadMapped utility for type-safe staging.
RAII scope guards for ImGui (ScopedId, ScopedColor, ScopedFont).

---

## 19. Staging Memory - Queue & Timeline Futures
**Commits:** [47-49] 23793be, d15c4c9, e6720cd  
**Date Range:** Dec 2024

StreamingStaging, StagingQueue and timeline synchronization

The birth of StreamingStaging to better handle staging buffer lifetimes.
StagingQueue for managing in-flight staging allocations.
copyBuffer helpers and TimelineFuture for asynchronous resource access.
Blocking waitTimelineSemaphore and SemaphoreQueue.

---

## 20. Staging Memory - Major Refactor & Test Expansion
**Commits:** [50] 4e5b7a2  
**Date Range:** Dec 2024

staging memory refactor and test reorganization

StagingAllocator concept and DedicatedStaging implementation.
Split test.cpp to test_staging.cpp and test_timeline_queue.cpp.
Initial tests for RecyclingStagingPool and TimelineQueue.
Test fixtures and helpers in test_context*.hpp.

---

## 21. Staging Memory - Callback System Refactor
**Commits:** [51-53] d858286, 49efdb6, 9058ee8  
**Date Range:** Dec 2024

staging memory: lazy callback system

Lazy callback invocation during pool recycling.
Stress tests covering multi-threaded allocations.
DownloadFuture evaluation and buffer copy constraints.

---

## 22. Staging Memory - Streaming & Build Improvements
**Commits:** [54-56] ec34159, 773682f, 31821bd  
**Date Range:** Dec 2024

StreamingStaging for submitting and cycling staging memory chunks

StreamingStaging holds a queue and cycles its own command buffers.
Submits as needed to cycle a fixed pool of staging memory.
Add shaderc integration with FetchContent (SPIRV-Tools, glslang dependencies).
Fix Windows CRT bug with SHADERC_ENABLE_SHARED_CRT.
Expose Slang and Shaderc version tags in CMake.

---

## 23. Staging Memory - Bug Fixes & Hardening
**Commits:** [57-61] e37dff6, 34c7531, 6e693a9, 800000a, 201e589  
**Date Range:** Dec 2024

staging memory: fix corruption bugs and harden implementation

Renamed StreamingStaging to StagingStream.
Fix dangling mapping when StagingStream destructs early - wait for completion.
Fix uninitialized future access - wait for last chunk semaphore, not first.
DownloadFutureBuilder facilitates, creating the future with the final semaphore.
Move DownloadFuture logic from timeline_queue.hpp to staging_memory.hpp.
Shared SPIRV and glslang dependencies across build.
Expand stress test suite for RecyclingStagingPool.

---

## 24. Query Pool for GPU Profiling
**Commits:** [62-63] 3c92e42, aa396e6  
**Date Range:** Dec 2024

QueryPool for gpu profiling and occlusion queries

QueryPool wrapper with timestamp and occlusion query support.
Query recycling with host-based resets.
Helpers for NGX instance/device extensions.

---

## 25. Serial Timeline Queue
**Commits:** [64] af23019  
**Date Range:** Jan 2025

serial timeline queue

SerialTimelineQueue for single-queue tracking.
TimelineCommandBuffer to pair recording with completion promises.
Refactored CyclingCommandBuffer to use new queue types.

---

## 26. Staging & ImGui Refinements
**Commits:** [65-67] 90e6b9e, a1435ca, f6dde0d  
**Date Range:** Jan 2025

staging and imgui refinements

Replace cancellation asserts with TimelineSubmitCancel exceptions.
Customizable pipeline stage flags for staging submit.
ImGui Vulkan texture descriptor support.
ImGui RAII scope guards (tree, buttonRepeat, font).

---

## 27. Staging Memory - Reference Pattern
**Commits:** [68] 2e76a82  
**Date Range:** Jan 2025

non-owning StagingStreamRef

StagingStream separates ownership from usage with "Ref" pattern.
Fix resource leaks during staging pool destruction.

---

## 28. Device Address Support
**Commits:** [69] 7590314  
**Date Range:** Jan 2025

staging: device address support

Type-safe DeviceAddress template for Vulkan device pointers.
Staging memory handles transfers with device addresses.
Improved device extension discovery.

---

## 29. Swapchain Present IDs
**Commits:** [70] 4ad2c2d  
**Date Range:** Jan 2025

swapchain present ids with VK_KHR_present_id

---

## 30. Staging Memory - Free Function API Refactor
**Commits:** [71-73] 059e5ce, 71127b0, 20ed698  
**Date Range:** Jan 2025

staging api free function refactor

Refactor staging API from member functions to free functions.
StagingStream operations delegated to StagingStreamRef.
Comprehensive overloads for upload/download (buffers, ranges, device addresses).
Unit tests for creation-on-upload and raw byte transfers.
Minimum CMake version 3.24.

---

## Summary

**Total: 30 milestone commits** (reduced from 73)

This plan maintains buildability at each step while preserving development history.

### Chronological phases:
- Feb-Mar 2024: Foundation, generator, RAII core, surfaces, VMA/VVL, Slang, RT
- Mar 2024: API standardization, NGX, staging initial implementation
- Nov 2024 - Jan 2025: ImGui, staging iteration (queue → refactor → callbacks → streaming → hardening), QueryPool, final API polish

