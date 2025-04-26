// Copyright (c) 2025 Pyarelal Knowles, MIT License

// This file works around the lack of native XCB support in GLFW and to hide the
// leaky macros in X11/Xlib.h from everything else
// See: https://github.com/glfw/glfw/issues/1061
// And: https://stackoverflow.com/questions/79583727/how-to-avoid-macros-from-x11-polluting-other-third-party-code
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
    Display*       display  = glfwGetX11Display();
    int            screen   = DefaultScreen(display);
    return XVisualIDFromVisual(DefaultVisual(display, screen));
}

xcb_connection_t* glfwGetXCBConnection()
{
    Display* display = glfwGetX11Display();
    return XGetXCBConnection(display);
}

xcb_window_t glfwGetXCBWindow(GLFWwindow* window)
{
    return glfwGetX11Window(window);
}

} // namespace glfw
} // namespace vko

#endif
