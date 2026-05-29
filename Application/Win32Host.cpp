// Application/Win32Host.cpp — Phase 2 (Phase 3 adds ImGui WndProc handoff)
#include "Win32Host.h"

#include "Core/Logger.h"

// ImGui's win32 backend ships a WndProc shim we forward into so ImGui sees raw
// input. The header keeps the declaration behind `#if 0` so it doesn't drag
// <windows.h> into other TUs; we forward-declare it here ourselves (the symbol
// itself lives in imgui_impl_win32.cpp, archived into RenderingSubsystem.lib).
#include "imgui.h"
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

namespace RS {

namespace {
constexpr wchar_t kClassName[] = L"RSRenderingSubsystemWindow";
}

Win32Host::~Win32Host() {
    Destroy();
}

bool Win32Host::Create(HINSTANCE hInstance, const wchar_t* title, int width, int height) {
    m_Instance = hInstance;

    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc   = &Win32Host::StaticWndProc;
    wc.hInstance     = hInstance;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;          // we paint via Vulkan, not GDI
    wc.lpszClassName = kClassName;

    if (!RegisterClassExW(&wc)) {
        RS_LOG_ERROR("RegisterClassExW failed: %lu", GetLastError());
        return false;
    }

    // Pick a window rect such that the *client* area is exactly width x height.
    DWORD style   = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);  // non-resizable v1
    DWORD exStyle = 0;
    RECT  rect    = { 0, 0, width, height };
    AdjustWindowRectEx(&rect, style, FALSE, exStyle);

    m_Hwnd = CreateWindowExW(
        exStyle, kClassName, title, style,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, hInstance, this);

    if (!m_Hwnd) {
        RS_LOG_ERROR("CreateWindowExW failed: %lu", GetLastError());
        return false;
    }

    ShowWindow(m_Hwnd, SW_SHOW);
    UpdateWindow(m_Hwnd);
    return true;
}

void Win32Host::Destroy() {
    if (m_Hwnd) {
        DestroyWindow(m_Hwnd);
        m_Hwnd = nullptr;
    }
    if (m_Instance) {
        UnregisterClassW(kClassName, m_Instance);
        m_Instance = nullptr;
    }
}

bool Win32Host::PumpMessages() {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            m_QuitPosted = true;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return !m_QuitPosted;
}

void Win32Host::GetClientSize(uint32_t& outW, uint32_t& outH) const {
    RECT r{};
    if (m_Hwnd && GetClientRect(m_Hwnd, &r)) {
        outW = static_cast<uint32_t>(r.right  - r.left);
        outH = static_cast<uint32_t>(r.bottom - r.top);
    } else {
        outW = 0;
        outH = 0;
    }
}

bool Win32Host::ConsumeF1Pressed() {
    const bool was = m_F1Pressed;
    m_F1Pressed = false;
    return was;
}

LRESULT CALLBACK Win32Host::StaticWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    Win32Host* self = nullptr;
    if (msg == WM_NCCREATE) {
        // Stash the `this` pointer passed via CreateWindowEx so subsequent
        // messages can route to InstanceWndProc.
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        self = static_cast<Win32Host*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<Win32Host*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }
    if (self) {
        return self->InstanceWndProc(hwnd, msg, wp, lp);
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

LRESULT Win32Host::InstanceWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    // Let ImGui see input first (only once its backend has been initialised).
    // Returning early on a non-zero result lets ImGui claim focus capture for
    // mouse/keyboard when its widgets are hovered/active.
    if (ImGui::GetCurrentContext() != nullptr) {
        if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp)) {
            return 1;
        }
    }
    switch (msg) {
    case WM_CLOSE:
        PostQuitMessage(0);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (wp == VK_F1) m_F1Pressed = true;
        if (wp == VK_ESCAPE) PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

} // namespace RS
