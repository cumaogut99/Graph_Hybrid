"""
Wiring of the frameless window's native frame, with the Windows API faked
(the tests run on any platform; the Windows behaviour itself is not run).
"""

import ctypes
import types

import pytest
from PyQt5.QtWidgets import QMainWindow

import src.ui.frameless_window as fw


class _FakeUser32:
    def __init__(self):
        self.set_window_pos = []

    def GetWindowLongW(self, hwnd, index):
        return 0

    def SetWindowLongW(self, hwnd, index, value):
        return 0

    def SetWindowPos(self, hwnd, after, x, y, cx, cy, flags):
        self.set_window_pos.append(flags)
        return 1


class _Window(fw.FramelessWindowMixin, QMainWindow):
    pass


@pytest.fixture
def fake_windows(monkeypatch):
    user32 = _FakeUser32()
    monkeypatch.setattr(fw, '_user32', user32, raising=False)
    monkeypatch.setattr(fw, '_dwmapi', types.SimpleNamespace(DwmExtendFrameIntoClientArea=lambda *a: 0), raising=False)
    monkeypatch.setattr(fw, '_MARGINS', lambda *a: ctypes.c_int(0), raising=False)
    for name, value in dict(GWL_STYLE=-16, WS_CAPTION=0, WS_THICKFRAME=0, WS_MINIMIZEBOX=0,
                            WS_MAXIMIZEBOX=0, WS_SYSMENU=0, SWP_FRAMECHANGED_ONLY=0x37).items():
        monkeypatch.setattr(fw, name, value, raising=False)
    return user32


def test_screen_change_resends_wm_nccalcsize(qapp, fake_windows):
    window = _Window()
    window._frameless = True
    window._native_frame_ready = False
    window.show()
    try:
        assert window._native_frame_ready
        # the frame is recalculated once the native styles are set
        assert fake_windows.set_window_pos == [0x37]
        # and again after every screen change, when Qt has recalculated its
        # frame margins from the style
        window.windowHandle().screenChanged.emit(window.screen())
        assert fake_windows.set_window_pos == [0x37, 0x37]
    finally:
        window._native_frame_ready = False
        window.close()


def test_no_native_calls_before_setup(qapp, fake_windows):
    window = _Window()
    window._native_frame_ready = False
    window._refresh_native_frame()
    assert fake_windows.set_window_pos == []
