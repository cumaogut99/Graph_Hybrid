# type: ignore
"""
Clickable Color Label

A QLabel that shows a color and opens a QColorDialog when clicked.
Used in statistics panel for signal color selection.
"""

from PyQt5.QtWidgets import QLabel, QColorDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QMouseEvent


class ClickableColorLabel(QLabel):
    """A QLabel that shows a color and opens a QColorDialog when clicked."""
    color_changed = pyqtSignal(str)  # Emits the new color hex string

    def __init__(self, initial_color: str, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.set_color(initial_color)

    def set_color(self, color_hex: str):
        self._color = QColor(color_hex)
        self.setToolTip(f"Click to change color ({color_hex})")
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color_hex};
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 6px;
            }}
            QLabel:hover {{
                border: 2px solid rgba(255, 255, 255, 1);
            }}
        """)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            new_color = QColorDialog.getColor(self._color, self, "Select Signal Color")
            if new_color.isValid():
                new_color_hex = new_color.name()
                self.set_color(new_color_hex)
                self.color_changed.emit(new_color_hex)
        super().mousePressEvent(event)

