# type: ignore
"""
Droppable Plot Widget

A PyQtGraph PlotWidget wrapper that accepts drag-and-drop of parameters/columns.
"""

import logging
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class DroppablePlotWidget(pg.PlotWidget):
    """
    A PlotWidget that accepts drag-and-drop of parameters/columns.
    
    When a parameter is dropped, it emits a signal with the parameter name
    so that the parent can plot it on this graph.
    """
    
    # Signals
    parameter_dropped = Signal(str)  # parameter_name
    
    def __init__(self, parent=None, graph_index: int = 0, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.graph_index = graph_index
        self.drag_active = False
        
        # Enable drop events
        self.setAcceptDrops(True)
        
        logger.debug(f"DroppablePlotWidget created for graph {graph_index}")
    
    def dragEnterEvent(self, event):
        """Handle drag enter - check if we can accept this drag."""
        # Check if this is a parameter drag
        if event.mimeData().hasFormat("application/x-parameter") or event.mimeData().hasText():
            event.acceptProposedAction()
            self.drag_active = True
            self.update()  # Trigger repaint to show drop indicator
            logger.debug(f"Drag entered graph {self.graph_index}")
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """Handle drag move - keep accepting while dragging over."""
        if event.mimeData().hasFormat("application/x-parameter") or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave - remove visual feedback."""
        self.drag_active = False
        self.update()  # Trigger repaint to remove drop indicator
        logger.debug(f"Drag left graph {self.graph_index}")
    
    def dropEvent(self, event):
        """Handle drop - extract parameter name and emit signal."""
        self.drag_active = False
        self.update()  # Remove drop indicator
        
        # Extract parameter name from mime data
        parameter_name = None
        
        if event.mimeData().hasFormat("application/x-parameter"):
            # Custom format
            parameter_name = bytes(event.mimeData().data("application/x-parameter")).decode('utf-8')
        elif event.mimeData().hasText():
            # Fallback to text
            parameter_name = event.mimeData().text()
        
        if parameter_name:
            logger.info(f"Parameter '{parameter_name}' dropped on graph {self.graph_index}")
            event.acceptProposedAction()
            
            # Emit signal so parent can plot this parameter
            self.parameter_dropped.emit(parameter_name)
        else:
            logger.warning("Drop event received but no parameter name found")
            event.ignore()
    
    def paintEvent(self, event):
        """Override paint to show drop indicator when dragging."""
        # Call parent paint first
        super().paintEvent(event)
        
        # Draw drop indicator if drag is active
        if self.drag_active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Semi-transparent blue overlay
            overlay_color = QColor(74, 144, 226, 30)  # RGBA
            painter.fillRect(self.rect(), overlay_color)
            
            # Border
            border_pen = QPen(QColor(74, 144, 226, 150), 3, Qt.DashLine)
            painter.setPen(border_pen)
            painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
            
            # Drop hint text
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "📊 Drop here to plot")
            
            painter.end()

