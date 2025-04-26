// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

// TODO: Move to cmake. It's bad practice to define configuration macros in a
// header
#if defined(VK_USE_PLATFORM_METAL_EXT) || defined(VK_USE_PLATFORM_MACOS_MVK)
#define GLFW_EXPOSE_NATIVE_COCOA // ??
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
#define GLFW_EXPOSE_NATIVE_X11
#endif

// NOTE: including glfw3native.h leaks the various native header includes -
// maybe more than including vulkan.h. You'd need those anyway to create
// surfaces for various platforms, so it's necessary.
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <optional>
#include <span>
#include <vko/exceptions.hpp>
#include <vko/functions.hpp>
#include <vko/handles.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

// *shakes fist at Xlib*
#if defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR)
    #pragma push_macro("None")
    #undef None
namespace vko {
namespace glfw {
static constexpr auto None = 0L;
}
} // namespace vko
    #pragma pop_macro("None")
#endif

namespace vko {
namespace glfw {

// Workaround because GLFW/glfw3native.h does not provide xcb handles
// NOTE: these are defined in vulkan_objects and are NOT currently part of
// GLFW.. but IMO they should be. See: https://github.com/glfw/glfw/issues/1061
#if defined(VK_USE_PLATFORM_XCB_KHR)
xcb_visualid_t glfwGetXCBVisualID();
xcb_connection_t* glfwGetXCBConnection();
xcb_window_t glfwGetXCBWindow(GLFWwindow* window);
#endif

inline const char* errorToString(int errorCode) {
    // clang-format off
    switch (errorCode) {
        case GLFW_NO_ERROR: return "GLFW_NO_ERROR: No error has occurred.";
        case GLFW_NOT_INITIALIZED: return "GLFW_NOT_INITIALIZED: GLFW has not been initialized.";
        case GLFW_NO_CURRENT_CONTEXT: return "GLFW_NO_CURRENT_CONTEXT: No context is current for this thread.";
        case GLFW_INVALID_ENUM: return "GLFW_INVALID_ENUM: One of the arguments to the function was an invalid enum value.";
        case GLFW_INVALID_VALUE: return "GLFW_INVALID_VALUE: One of the arguments to the function was an invalid value.";
        case GLFW_OUT_OF_MEMORY: return "GLFW_OUT_OF_MEMORY: A memory allocation failed.";
        case GLFW_API_UNAVAILABLE: return "GLFW_API_UNAVAILABLE: GLFW could not find support for the requested API on the system.";
        case GLFW_VERSION_UNAVAILABLE: return "GLFW_VERSION_UNAVAILABLE: The requested OpenGL or OpenGL ES version is not available.";
        case GLFW_PLATFORM_ERROR: return "GLFW_PLATFORM_ERROR: A platform-specific error occurred that does not match any of the more specific categories.";
        case GLFW_FORMAT_UNAVAILABLE: return "GLFW_FORMAT_UNAVAILABLE: The requested format is not supported or available.";
        case GLFW_NO_WINDOW_CONTEXT: return "GLFW_NO_WINDOW_CONTEXT: The specified window does not have an OpenGL or OpenGL ES context.";
        default: break;
    }
    return "<invalid error code>";
    // clang-format on
}

inline Exception makeException(std::string msg, int errorCode) {
    return Exception(msg + ": " + errorToString(errorCode));
}

inline Exception makeLastErrorException(std::string msg) {
    const char* descroption;
    int         errorCode = glfwGetError(&descroption);
    return makeException(msg, errorCode).addContext(std::string("(") + descroption + ")");
}

#define CHECK_GLFW_EQ(result, expected) check<expected>::equal(result, #result, #expected)
#define CHECK_GLFW_NE(result, notexpected) check<notexpected>::notequal(result, #result, #notexpected)
template <auto Compare>
struct check {
    using Result = decltype(Compare);
    static Result equal(Result result, const char* resultStr, const char* compareStr) {
        if (result != Compare) {
            throw makeLastErrorException(std::string(resultStr) + " != " + std::string(compareStr));
        }
        return result;
    }
    static Result notequal(Result result, const char* resultStr, const char* compareStr) {
        if (result == Compare) {
            throw makeLastErrorException(std::string(resultStr) + " != " + std::string(compareStr));
        }
        return result;
    }
};

class ScopedInit {
public:
    ScopedInit()
        : ScopedInit(GLFW_ANY_PLATFORM) {}
    ScopedInit(int platform) {
        glfwInitHint(GLFW_PLATFORM, platform);
        if (glfwInit() != GLFW_TRUE) {
            throw makeLastErrorException("glfwInit failed");
        }
    }
    ~ScopedInit() { glfwTerminate(); }
    ScopedInit(const ScopedInit& other) = delete;
    ScopedInit operator=(const ScopedInit& other) = delete;
};

struct PlatformSupport {
    PlatformSupport(std::span<const VkExtensionProperties> availableExtensions) {
        for (auto& ext : availableExtensions) {
            std::string_view name(ext.extensionName);
#if VK_KHR_win32_surface
            if (name == VK_KHR_WIN32_SURFACE_EXTENSION_NAME) {
                win32 = true;
                continue;
            }
#endif
#if VK_KHR_wayland_surface
            if (name == VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME) {
                wayland = true;
                continue;
            }
#endif
#if VK_KHR_xcb_surface
            if (name == VK_KHR_XCB_SURFACE_EXTENSION_NAME) {
                xcb = true;
                continue;
            }
#endif
#if VK_KHR_xlib_surface
            if (name == VK_KHR_XLIB_SURFACE_EXTENSION_NAME) {
                xlib = true;
                continue;
            }
#endif
#if VK_MVK_macos_surface
            if (name == VK_MVK_MACOS_SURFACE_EXTENSION_NAME) {
                macos = true;
                continue;
            }
#endif
#if VK_KHR_android_surface
            if (name == VK_KHR_ANDROID_SURFACE_EXTENSION_NAME) {
                android = true;
                continue;
            }
#endif
        }
    }
    bool win32   = false;
    bool wayland = false;
    bool xcb     = false;
    bool xlib    = false;
    bool macos   = false;
    bool android = false;
};

inline const char* platformSurfaceExtension([[maybe_unused]] PlatformSupport support) {
    int platform = glfwGetPlatform();
    if (platform == 0)
        throw makeLastErrorException("glfwGetPlatform failed");
    switch (platform) {
#if VK_KHR_win32_surface
    case GLFW_PLATFORM_WIN32:
        return VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#endif
#if VK_KHR_wayland_surface
    case GLFW_PLATFORM_WAYLAND:
        return VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
#endif
#if VK_KHR_xcb_surface || VK_KHR_xlib_surface
    case GLFW_PLATFORM_X11:
    #if VK_KHR_xcb_surface
        if (support.xcb)
            return VK_KHR_XCB_SURFACE_EXTENSION_NAME;
    #endif
    #if VK_KHR_xlib_surface
        return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
    #endif
#endif
#if VK_MVK_macos_surface
    case GLFW_PLATFORM_COCOA:
        return VK_MVK_MACOS_SURFACE_EXTENSION_NAME;
#endif
    default:
        break;
    }
    throw Exception("glfwGetPlatform returned unsupported platform " + std::to_string(platform));
}

inline bool physicalDevicePresentationSupport([[maybe_unused]] InstanceCommands& vk,
                                              [[maybe_unused]] PlatformSupport   support,
                                              VkPhysicalDevice                   physicalDevice,
                                              uint32_t                           queueFamilyIndex) {
    int platform = glfwGetPlatform();
    if (platform == 0)
        throw makeLastErrorException("glfwGetPlatform failed");
    switch (platform) {
#if VK_KHR_win32_surface
    case GLFW_PLATFORM_WIN32:
        return vk.vkGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, queueFamilyIndex) ==
               VK_TRUE;
#endif
#if VK_KHR_wayland_surface
    case GLFW_PLATFORM_WAYLAND:
        return vk.vkGetPhysicalDeviceWaylandPresentationSupportKHR(physicalDevice, queueFamilyIndex, glfwGetWaylandDisplay()) == VK_TRUE;
#endif
#if VK_KHR_xcb_surface || VK_KHR_xlib_surface
    case GLFW_PLATFORM_X11:
    #if VK_KHR_xcb_surface
        if (support.xcb) {
            return vk.vkGetPhysicalDeviceXcbPresentationSupportKHR(physicalDevice, queueFamilyIndex,
                                                                   glfwGetXCBConnection(),
                                                                   glfwGetXCBVisualID()) == VK_TRUE;
        }
    #endif
    #if VK_KHR_xlib_surface
        {
            Display* display = glfwGetX11Display();
            if (!display)
                throw makeLastErrorException("glfwGetX11Display failed");
            int      screen = DefaultScreen(display);
            VisualID visualID = XVisualIDFromVisual(DefaultVisual(display, screen));
            return vk.vkGetPhysicalDeviceXlibPresentationSupportKHR(physicalDevice, queueFamilyIndex,
                                                                 display, visualID) == VK_TRUE;
        }
    #endif
#endif
#if VK_MVK_macos_surface
    case GLFW_PLATFORM_COCOA:
        return true;
#endif
    default:
        break;
    }
    throw Exception("glfwGetPlatform returned unsupported platform " + std::to_string(platform));
}

struct WindowDeleter {
    void operator()(GLFWwindow* window) const {
        if (window) {
            glfwDestroyWindow(window);
        }
    }
};

using Window = std::unique_ptr<GLFWwindow, WindowDeleter>;

inline Window createWindow(int width, int height, const char* title, GLFWmonitor* monitor = nullptr,
                           GLFWwindow* share = nullptr) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, title, monitor, share);
    if (!window) {
        throw Exception("glfwCreateWindow failed");
    }
    return Window(window);
}

#if VK_KHR_win32_surface
template <instance_commands InstanceCommands>
Win32SurfaceKHR makeWin32SurfaceKHR(const InstanceCommands& vk, VkInstance instance,
                                    GLFWwindow* window) {
    return Win32SurfaceKHR{
        vk, instance,
        VkWin32SurfaceCreateInfoKHR{
            .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext     = nullptr,
            .flags     = 0,
            .hinstance = GetModuleHandle(
                nullptr), // GLFW loads its own module/.hinstance with GetModuleHandleExW()
            .hwnd = glfwGetWin32Window(window),
        }};
}
#endif

#if VK_KHR_wayland_surface
template <instance_commands InstanceCommands>
WaylandSurfaceKHR makeWaylandSurfaceKHR(const InstanceCommands& vk, VkInstance instance,
                                        GLFWwindow* window) {
    return WaylandSurfaceKHR{
        vk, instance,
        VkWaylandSurfaceCreateInfoKHR{
            .sType   = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext   = nullptr,
            .flags   = 0,
            .display = CHECK_GLFW_NE(glfwGetWaylandDisplay(), (wl_display*)0),
            .surface = CHECK_GLFW_NE(glfwGetWaylandWindow(window), (wl_surface*)0),
        }};
}
#endif

#if VK_KHR_xcb_surface
template <instance_commands InstanceCommands>
XcbSurfaceKHR makeXcbSurfaceKHR(const InstanceCommands& vk, VkInstance instance,
                                GLFWwindow* window) {
    return XcbSurfaceKHR{
        vk, instance,
        VkXcbSurfaceCreateInfoKHR{
            .sType      = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            .pNext      = nullptr,
            .flags      = 0,
            .connection = CHECK_GLFW_NE(glfwGetXCBConnection(), (xcb_connection_t*)0),
            .window     = CHECK_GLFW_NE(glfwGetXCBWindow(window), (xcb_window_t)0),
        }};
}
#endif

#if VK_KHR_xlib_surface
template <instance_commands InstanceCommands>
XlibSurfaceKHR makeXlibSurfaceKHR(const InstanceCommands& vk, VkInstance instance,
                                  GLFWwindow* window) {

    return XlibSurfaceKHR{vk, instance,
                          VkXlibSurfaceCreateInfoKHR{
                              .sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                              .pNext  = nullptr,
                              .flags  = 0,
                              .dpy    = CHECK_GLFW_NE(glfwGetX11Display(), (Display*)0),
                              .window = CHECK_GLFW_NE(glfwGetX11Window(window), (::Window)None),
                          }};
}
#endif

// Hackery to make a std::variant of all supported platform surfaces
// clang-format off
template <class Void, class... Types>
struct SurfaceVariantSelector {
    using type = std::variant<Types...>;
};
using SurfaceVariant = typename SurfaceVariantSelector<
    void // trailing comma workaround
#if VK_KHR_win32_surface
    , Win32SurfaceKHR
#endif
#if VK_KHR_wayland_surface
    , WaylandSurfaceKHR
#endif
#if VK_KHR_xcb_surface
    , XcbSurfaceKHR
#endif
#if VK_KHR_xlib_surface
    , XlibSurfaceKHR
#endif
    >::type;
// clang-format on

template <instance_commands InstanceCommands>
SurfaceVariant makeSurface(const InstanceCommands& vk, VkInstance instance, PlatformSupport support,
                           GLFWwindow* window) {
    std::optional<SurfaceVariant> result;
    std::string                   exceptionStrings;
#if VK_KHR_win32_surface
    if (!result && support.win32) {
        try {
            result = makeWin32SurfaceKHR(vk, instance, window);
        } catch (const Exception& e) {
            exceptionStrings += std::string(e.what()) + "\n";
        }
    }
#endif
#if VK_KHR_wayland_surface
    if (!result && support.wayland) {
        try {
            result = makeWaylandSurfaceKHR(vk, instance, window);
        } catch (const Exception& e) {
            exceptionStrings += std::string(e.what()) + "\n";
        }
    }
#endif
#if VK_KHR_xcb_surface
    if (!result && support.xcb) {
        try {
            result = makeXcbSurfaceKHR(vk, instance, window);
        } catch (const Exception& e) {
            exceptionStrings += std::string(e.what()) + "\n";
        }
    }
#endif
#if VK_KHR_xlib_surface
    if (!result && support.xlib) {
        try {
            result = makeXlibSurfaceKHR(vk, instance, window);
        } catch (const Exception& e) {
            exceptionStrings += std::string(e.what()) + "\n";
        }
    }
#endif
    if (!result) {
        throw Exception("No supported surfaces:\n" + exceptionStrings);
    }
    return std::move(*result);
}

class SurfaceKHR {
public:
    template <instance_and_commands InstanceAndCommands>
    SurfaceKHR(const InstanceAndCommands& vk, const PlatformSupport& support, GLFWwindow* window)
        : SurfaceKHR(vk, vk, support, window) {}

    template <instance_commands InstanceCommands>
    SurfaceKHR(const InstanceCommands& vk, VkInstance instance, PlatformSupport support,
               GLFWwindow* window)
        : m_surface(makeSurface(vk, instance, support, window)) {}
    operator VkSurfaceKHR() const {
        return std::visit(
            [](auto const& s) -> VkSurfaceKHR { return static_cast<VkSurfaceKHR>(s); }, m_surface);
    }
    SurfaceVariant m_surface;
};

#undef CHECK_GLFW_RESULT
} // namespace glfw
} // namespace vko
