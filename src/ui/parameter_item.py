# type: ignore
"""
Draggable Parameter Item Widget

A draggable widget representing a parameter that can be dropped onto graphs.
"""

import logging
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal as Signal
from PyQt5.QtGui import QDrag, QPainter, QColor, QFont, QPalette

logger = logging.getLogger(__name__)


class ParameterItem(QFrame):
    """
    A draggable parameter item widget.
    
    Can be dragged and dropped onto graph plots to visualize the parameter.
    """
    
    # Signals
    drag_started = Signal(str)  # Emits parameter name when drag starts
    
    def __init__(self, parameter_name: str, parameter_formula: str = "", parent=None):
        super().__init__(parent)
        
        self.parameter_name = parameter_name
        self.parameter_formula = parameter_formula
        self.dragging = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the parameter item UI."""
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)
        self.setCursor(Qt.OpenHandCursor)
        
        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        
        # Icon
        icon_label = QLabel("📊")
        icon_label.setFont(QFont("Segoe UI Emoji", 12))
        layout.addWidget(icon_label)
        
        # Parameter name
        name_label = QLabel(self.parameter_name)
        name_label.setFont(QFont("Segoe UI", 10))
        name_label.setStyleSheet("color: #e6f3ff; font-weight: 500;")
        layout.addWidget(name_label, 1)
        
        # Formula hint if available
        if self.parameter_formula:
            formula_label = QLabel(f"({self.parameter_formula})")
            formula_label.setFont(QFont("Segoe UI", 8))
            formula_label.setStyleSheet("color: #a0c0e0; font-style: italic;")
            layout.addWidget(formula_label)
        
        # Styling
        self.setStyleSheet("""
            ParameterItem {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 rgba(74, 144, 226, 0.1));
                border: 1px solid rgba(74, 144, 226, 0.4);
                border-radius: 6px;
                padding: 4px;
            }
            ParameterItem:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.15));
                border-color: rgba(74, 144, 226, 0.6);
            }
        """)
        
        # Set minimum size
        self.setMinimumHeight(36)
        
        # Tooltip
        tooltip_text = f"{self.parameter_name}"
        if self.parameter_formula:
            tooltip_text += f"\n\nFormula: {self.parameter_formula}"
        tooltip_text += "\n\nDrag and drop onto a graph to plot this parameter"
        self.setToolTip(tooltip_text)
    
    def mousePressEvent(self, event):
        """Handle mouse press - initiate drag."""
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.dragging = True
    
    def mouseMoveEvent(self, event):
        """Handle mouse move - start drag operation."""
        if not self.dragging:
            return
        
        if (event.buttons() & Qt.LeftButton) and self.dragging:
            # Create drag object
            drag = QDrag(self)
            mime_data = QMimeData()
            
            # Store parameter information
            mime_data.setText(self.parameter_name)
            mime_data.setData("application/x-parameter", self.parameter_name.encode())
            
            # Optional: Store formula as well
            if self.parameter_formula:
                mime_data.setData("application/x-parameter-formula", self.parameter_formula.encode())
            
            drag.setMimeData(mime_data)
            
            # Create drag pixmap (visual feedback)
            pixmap = self.grab()
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())
            
            # Emit signal
            self.drag_started.emit(self.parameter_name)
            logger.debug(f"Drag started for parameter: {self.parameter_name}")
            
            # Execute drag
            result = drag.exec_(Qt.CopyAction | Qt.MoveAction)
            
            # Reset cursor
            self.setCursor(Qt.OpenHandCursor)
            self.dragging = False
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.setCursor(Qt.OpenHandCursor)
        self.dragging = False

