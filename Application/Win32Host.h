// Application/Win32Host.h — Phase 2
// Minimal Win32 window shell for the standalone exe. Owns the HWND, the window
// class registration, and the message-pump translation into a tiny event queue
// the main loop drains each frame.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>

namespace RS {

class Win32Host {
public:
    Win32Host()  = default;
    ~Win32Host();

    bool Create(HINSTANCE hInstance, const wchar_t* title, int width, int height);
    void Destroy();

    // Drain queued messages. Returns false when the window has been closed
    // (WM_QUIT seen) so the main loop knows to exit.
    bool PumpMessages();

    HWND      WindowHandle() const { return m_Hwnd; }
    HINSTANCE InstanceHandle() const { return m_Instance; }
    void      GetClientSize(uint32_t& outW, uint32_t& outH) const;

    // Tiny event flags consumed by Main once per frame.
    bool ConsumeF1Pressed();   // returns true if F1 fired since last call

private:
    static LRESULT CALLBACK StaticWndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT                 InstanceWndProc(HWND, UINT, WPARAM, LPARAM);

    HINSTANCE m_Instance   = nullptr;
    HWND      m_Hwnd       = nullptr;
    bool      m_QuitPosted = false;
    bool      m_F1Pressed  = false;
};

} // namespace RS
