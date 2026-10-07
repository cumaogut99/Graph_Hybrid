"""
Frameless main window: the application toolbar doubles as the title bar.

On Windows the native caption is removed (WM_NCCALCSIZE) while the window
keeps its WS_CAPTION / WS_THICKFRAME styles, so the system still provides
moving, Aero Snap, double-click maximize, the shadow, the minimize/maximize
animations and resizing at the edges. WM_NCHITTEST tells Windows which parts
of the window act as the caption: every widget marked with the
DRAG_AREA_PROPERTY property, except interactive widgets such as buttons.

On other platforms the native frame is kept and the caption buttons hide
themselves.
"""

import ctypes
import logging
import sys

from PyQt5.QtCore import Qt, QEvent, QPoint, QPointF, QRectF, QSize
from PyQt5.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox, QHBoxLayout,
    QLineEdit, QTabBar, QWidget
)

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
QWIDGETSIZE_MAX = (1 << 24) - 1

# Widgets (and their children) with this property set to True move the window
DRAG_AREA_PROPERTY = "windowDragArea"

_INTERACTIVE = (QAbstractButton, QAbstractSpinBox, QAbstractSlider, QComboBox, QLineEdit, QTabBar)

if IS_WINDOWS:
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _dwmapi = ctypes.windll.dwmapi

    WM_GETMINMAXINFO = 0x0024
    WM_NCCALCSIZE = 0x0083
    WM_NCHITTEST = 0x0084
    GWL_STYLE = -16
    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    WS_MINIMIZEBOX = 0x00020000
    WS_MAXIMIZEBOX = 0x00010000
    WS_SYSMENU = 0x00080000
    MONITOR_DEFAULTTONEAREST = 2
    # SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
    SWP_FRAMECHANGED_ONLY = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020
    HTCAPTION = 2
    HTLEFT, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT = 10, 11, 12, 13, 14
    HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT = 15, 16, 17

    class _NCCALCSIZE_PARAMS(ctypes.Structure):
        _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    class _MINMAXINFO(ctypes.Structure):
        _fields_ = [("ptReserved", wintypes.POINT), ("ptMaxSize", wintypes.POINT),
                    ("ptMaxPosition", wintypes.POINT), ("ptMinTrackSize", wintypes.POINT),
                    ("ptMaxTrackSize", wintypes.POINT)]

    class _MARGINS(ctypes.Structure):
        _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int),
                    ("top", ctypes.c_int), ("bottom", ctypes.c_int)]

    _user32.GetWindowLongW.restype = ctypes.c_long
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.SetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    _user32.IsZoomed.argtypes = [wintypes.HWND]
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.MonitorFromWindow.restype = wintypes.HMONITOR
    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO)]
    _dwmapi.DwmExtendFrameIntoClientArea.argtypes = [wintypes.HWND, ctypes.POINTER(_MARGINS)]


def is_frameless(window: QWidget) -> bool:
    return bool(window is not None and window.windowFlags() & Qt.FramelessWindowHint)


def _is_drag_area(widget: QWidget) -> bool:
    while widget is not None:
        if isinstance(widget, _INTERACTIVE):
            return False
        if widget.property(DRAG_AREA_PROPERTY):
            return True
        widget = widget.parentWidget()
    return False


class FramelessWindowMixin:
    """
    Mixin for a QMainWindow (list it before QMainWindow). Call
    _init_frameless() in __init__; the native part is set up on first show.
    """

    RESIZE_BORDER = 6  # logical pixels inside the window edge

    def _init_frameless(self):
        self._frameless = False
        self._native_frame_ready = False
        if not IS_WINDOWS or QGuiApplication.platformName() != "windows":
            return
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self._frameless = True

    def showEvent(self, event):
        # The native window exists now but is not mapped yet. Creating it
        # earlier (winId() in __init__) makes Qt apply the DPI scale twice.
        if getattr(self, '_frameless', False) and not self._native_frame_ready:
            self._setup_native_frame()
        super().showEvent(event)

    def _setup_native_frame(self):
        try:
            hwnd = int(self.winId())
            # Give the frameless window back the styles of a normal window;
            # its caption is removed again in WM_NCCALCSIZE
            style = _user32.GetWindowLongW(hwnd, GWL_STYLE)
            _user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME
                                   | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
            # Keeps the DWM shadow around the window
            margins = _MARGINS(1, 1, 1, 1)
            _dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))
            self._native_frame_ready = True
            # Recalculate the frame now (WM_NCCALCSIZE) with the new styles
            _user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_FRAMECHANGED_ONLY)
        except Exception as e:
            # Still usable: a frameless window with the caption buttons
            logger.warning(f"Native frame setup failed: {e}")

    def _dpr(self) -> float:
        handle = self.windowHandle()
        return handle.devicePixelRatio() if handle is not None else self.devicePixelRatioF()

    def nativeEvent(self, event_type, message):
        if getattr(self, '_native_frame_ready', False) and bytes(event_type) == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_NCCALCSIZE and msg.wParam:
                if _user32.IsZoomed(msg.hWnd):
                    # A maximized window is larger than the screen by its frame
                    # width; keep the client area on the monitor's work area
                    params = _NCCALCSIZE_PARAMS.from_address(msg.lParam)
                    info = _MONITORINFO()
                    info.cbSize = ctypes.sizeof(_MONITORINFO)
                    monitor = _user32.MonitorFromWindow(msg.hWnd, MONITOR_DEFAULTTONEAREST)
                    if _user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                        params.rgrc[0] = info.rcWork
                # No non-client area: the client covers the whole window
                return True, 0
            if msg.message == WM_GETMINMAXINFO:
                # Qt would add the (removed) caption and borders to the
                # minimum size; the window is exactly its client area
                info = _MINMAXINFO.from_address(msg.lParam)
                dpr = self._dpr()
                minimum, maximum = self.minimumSize(), self.maximumSize()
                info.ptMinTrackSize.x = round(minimum.width() * dpr)
                info.ptMinTrackSize.y = round(minimum.height() * dpr)
                if maximum.width() < QWIDGETSIZE_MAX:
                    info.ptMaxTrackSize.x = round(maximum.width() * dpr)
                if maximum.height() < QWIDGETSIZE_MAX:
                    info.ptMaxTrackSize.y = round(maximum.height() * dpr)
                return True, 0
            if msg.message == WM_NCHITTEST:
                hit = self._hit_test(msg)
                if hit is not None:
                    return True, hit
        return super().nativeEvent(event_type, message)

    def _hit_test(self, msg):
        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
        rect = wintypes.RECT()
        _user32.GetWindowRect(msg.hWnd, ctypes.byref(rect))
        dpr = self._dpr()

        if not (self.isMaximized() or self.isFullScreen()):
            border = max(1, round(self.RESIZE_BORDER * dpr))
            left = x < rect.left + border
            right = x >= rect.right - border
            top = y < rect.top + border
            bottom = y >= rect.bottom - border
            if top and left:
                return HTTOPLEFT
            if top and right:
                return HTTOPRIGHT
            if bottom and left:
                return HTBOTTOMLEFT
            if bottom and right:
                return HTBOTTOMRIGHT
            if left:
                return HTLEFT
            if right:
                return HTRIGHT
            if top:
                return HTTOP
            if bottom:
                return HTBOTTOM

        pos = QPoint(int((x - rect.left) / dpr), int((y - rect.top) / dpr))
        if _is_drag_area(self.childAt(pos)):
            return HTCAPTION
        return None


class _CaptionButton(QAbstractButton):
    """Minimize / maximize / restore / close button drawn like the Windows ones."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(46, 39)
        self.setAttribute(Qt.WA_Hover)

    def sizeHint(self):
        return QSize(46, 39)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        hovered = self.underMouse()
        glyph = QColor("#d4dade")
        if self.kind == "close" and (hovered or self.isDown()):
            p.fillRect(self.rect(), QColor("#c42b1c") if hovered and not self.isDown() else QColor("#b0281a"))
            glyph = QColor("#ffffff")
        elif self.isDown():
            p.fillRect(self.rect(), QColor(255, 255, 255, 14))
        elif hovered:
            p.fillRect(self.rect(), QColor(255, 255, 255, 24))

        pen = QPen(glyph)
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        p.setPen(pen)
        s = 10.0
        r = QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        if self.kind == "min":
            p.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()))
        elif self.kind == "max":
            p.drawRect(r)
        elif self.kind == "restore":
            front = r.adjusted(0, 2, -2, 0)
            p.drawRect(front)
            p.drawLine(QPointF(r.left() + 2, r.top()), QPointF(r.right(), r.top()))
            p.drawLine(QPointF(r.right(), r.top()), QPointF(r.right(), r.bottom() - 2))
        elif self.kind == "close":
            p.setRenderHint(QPainter.Antialiasing, True)
            p.drawLine(r.topLeft(), r.bottomRight())
            p.drawLine(r.topRight(), r.bottomLeft())
        p.end()


class WindowControls(QWidget):
    """Minimize / maximize / close buttons for the frameless main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.min_btn = _CaptionButton("min", self)
        self.max_btn = _CaptionButton("max", self)
        self.close_btn = _CaptionButton("close", self)
        self.min_btn.setToolTip("Minimize")
        self.close_btn.setToolTip("Close")
        for btn in (self.min_btn, self.max_btn, self.close_btn):
            layout.addWidget(btn, 0, Qt.AlignTop)
        self.min_btn.clicked.connect(lambda: self.window().showMinimized())
        self.max_btn.clicked.connect(self._toggle_maximized)
        self.close_btn.clicked.connect(lambda: self.window().close())
        self._watched_window = None
        self._update_max_button()

    def showEvent(self, event):
        window = self.window()
        if window is not self._watched_window:
            if self._watched_window is not None:
                self._watched_window.removeEventFilter(self)
            window.installEventFilter(self)
            self._watched_window = window
        # Only the frameless main window needs its own caption buttons
        self.setVisible(is_frameless(window))
        self._update_max_button()
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if obj is self._watched_window and event.type() == QEvent.WindowStateChange:
            self._update_max_button()
        return False

    def _toggle_maximized(self):
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def _update_max_button(self):
        maximized = self.window().isMaximized()
        self.max_btn.kind = "restore" if maximized else "max"
        self.max_btn.setToolTip("Restore Down" if maximized else "Maximize")
        self.max_btn.update()
