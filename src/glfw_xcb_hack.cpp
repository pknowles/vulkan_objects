// Copyright (c) 2025 Pyarelal Knowles, MIT License

// GLFW XCB Workaround
// ===================
// This file works around the lack of native XCB support in GLFW. GLFW only
// exposes X11/Xlib handles (glfwGetX11Display, glfwGetX11Window), not XCB
// handles. This is problematic because:
//
// 1. XCB is the preferred X11 API (Xlib is legacy)
// 2. X11/Xlib.h pollutes the global namespace with macros (Success, None, etc.)
//    that break other code
//
// This file provides glfwGetXCBConnection/Visual/Window functions by wrapping
// the Xlib functions and converting to XCB types. It must be compiled in a
// separate translation unit to prevent X11 macro pollution from leaking.
//
// To use: Link against the vulkan_objects_glfw library (enable with
// VULKAN_OBJECTS_FETCH_GLFW=ON in CMake).
//
// References:
// - https://github.com/glfw/glfw/issues/1061 (Feature request for native XCB)
// - https://stackoverflow.com/questions/79583727/how-to-avoid-macros-from-x11-polluting-other-third-party-code
//
// This workaround can be removed once GLFW adds native XCB support.
#if defined(VK_USE_PLATFORM_XCB_KHR)
    #define GLFW_EXPOSE_NATIVE_X11

    // NOTE: we cannot include glfw_objects.hpp as that includes GLFW/glfw3native.h
    // without GLFW_EXPOSE_NATIVE_X11
    #include <GLFW/glfw3.h>
    #include <GLFW/glfw3native.h>
    #include <X11/Xlib-xcb.h>
    #include <xcb/xcb.h>

namespace vko {
namespace glfw {

xcb_visualid_t glfwGetXCBVisualID() {
    Display* display = glfwGetX11Display();
    int      screen  = DefaultScreen(display);
    return XVisualIDFromVisual(DefaultVisual(display, screen));
}

xcb_connection_t* glfwGetXCBConnection() {
    Display* display = glfwGetX11Display();
    return XGetXCBConnection(display);
}

xcb_window_t glfwGetXCBWindow(GLFWwindow* window) { return glfwGetX11Window(window); }

} // namespace glfw
} // namespace vko

#endif
