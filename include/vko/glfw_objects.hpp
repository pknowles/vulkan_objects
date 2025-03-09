// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#if defined(VK_USE_PLATFORM_METAL_EXT) || defined(VK_USE_PLATFORM_MACOS_MVK)
#define GLFW_EXPOSE_NATIVE_COCOA // ??
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR) || defined(VK_USE_PLATFORM_XCB_KHR)
#define GLFW_EXPOSE_NATIVE_X11
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// ffs, x11
#ifdef GLFW_EXPOSE_NATIVE_X11
    #undef None
namespace vko {
namespace glfw {
static constexpr auto None = 0L;
}
} // namespace vko
#endif

#include <span>
#include <vko/exceptions.hpp>
#include <vko/functions.hpp>
#include <vko/handles.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace vko {
namespace glfw {

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
template <auto Expected>
struct check;
template <class Result, Result Compare>
struct check<Compare> {
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
        return vk.vkGetPhysicalDeviceWaylandPresentationSupportKHR() == VK_TRUE;
#endif
#if VK_KHR_xcb_surface || VK_KHR_xlib_surface
    case GLFW_PLATFORM_X11:
    #if VK_KHR_xcb_surface
        if (support.xcb) {
            xcb_connection_t* connection = blah;
            xcb_visualid_t    visualID = blah;
            return vk.vkGetPhysicalDeviceXcbPresentationSupportKHR(physicalDevice, queueFamilyIndex,
                                                                   connection, visualID) == VK_TRUE;
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

Window createWindow(int width, int height, const char* title, GLFWmonitor* monitor = nullptr,
                    GLFWwindow* share = nullptr) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, title, monitor, share);
    if (!window) {
        throw Exception("glfwCreateWindow failed");
    }
    return Window(window);
}

class SurfaceKHR {
public:
#if VK_KHR_win32_surface
    template <class... Args>
    SurfaceKHR(PlatformSupport, GLFWwindow* window, const Args&... args)
        : m_surface(VkWin32SurfaceCreateInfoKHR{.sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,.pNext=nullptr,
                                          .flags = 0,
                                          .hinstance = GetModuleHandle(nullptr), // GLFW loads its own module/.hinstance with GetModuleHandleExW()
                                          .hwnd      = glfwGetWin32Window(window)},
              args...) {}
    operator VkSurfaceKHR() const { return m_surface; }

private:
    Win32SurfaceKHR m_surface;
#endif
#if VK_KHR_wayland_surface
    template <class... Args>
    SurfaceKHR(PlatformSupport, GLFWwindow* window, const Args& ...args)
        : m_surface(WaylandSurfaceCreateInfoKHR{.sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,.pNext=nullptr,
                                                .flags= 0, .display=CHECK_GLFW_NE(glfwGetWaylandDisplay(), (wl_display*)0, .surface=CHECK_GLFW_NE(glfwGetWaylandWindow(window), (wl_surface*)0}, args...) {}
    operator VkSurfaceKHR() const { return m_surface; }

private:
    WaylandSurfaceKHR m_surface;
#endif
#if VK_KHR_xcb_surface || VK_KHR_xlib_surface
        template <class... Args>
        SurfaceKHR(PlatformSupport support, GLFWwindow* window, const Args&... args) {
    #if VK_KHR_xcb_surface
            if (support.xcb) {
                xcb_connection_t* connection = blah;
                xcb_window_t      window = blah(window);
                m_surface = XcbSurfaceKHR(
                    VkXcbSurfaceCreateInfoKHR {
                        .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
                        .pNext = nullptr;
                        .flags = 0;
                        .connection = connection;
                        .window = window;
                    },
                    args...);
            }
    #endif
    #if VK_KHR_xlib_surface
            if (support.xlib) {
                m_surface = XlibSurfaceKHR(
                    VkXlibSurfaceCreateInfoKHR{
                        .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                        .pNext = nullptr,
                        .flags = 0,
                        .dpy = CHECK_GLFW_NE(glfwGetX11Display(), (Display*)0),
                        .window = CHECK_GLFW_NE(glfwGetX11Window(window), (::Window)None),
                    },
                    args...);
            }
    #endif
            if (!m_surface) {
                throw Exception("No supported surfaces");
            }
        }

    #if VK_KHR_xcb_surface && VK_KHR_xlib_surface
        std::variant<XlibSurfaceKHR, XcbSurfaceKHR> m_surface;
        operator VkSurfaceKHR() const {
            return std::holds_alternative<XlibSurfaceKHR>(m_surface)
                       ? static_cast<VkSurfaceKHR>(std::get<XlibSurfaceKHR>(m_surface))
                       : static_cast<VkSurfaceKHR>(std::get<XcbSurfaceKHR>(m_surface));
        }
    #else
        operator VkSurfaceKHR() const { return *m_surface; }
        private:
        #if VK_KHR_xcb_surface
        std::optional<XcbSurfaceKHR> m_surface;
        #endif
        #if VK_KHR_xlib_surface
        std::optional<XlibSurfaceKHR> m_surface;
        #endif
    #endif
#endif
#if VK_MVK_macos_surface
    case GLFW_PLATFORM_COCOA:
        return true;
#endif
    };

#undef CHECK_GLFW_RESULT
} // namespace glfw
} // namespace vko
