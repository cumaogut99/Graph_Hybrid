# type: ignore
"""
Clickable GroupBox with Drag & Drop

QGroupBox that emits a signal when clicked and supports drag-and-drop for reordering graphs.
"""

from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData
from PyQt5.QtGui import QMouseEvent, QDrag


class ClickableGroupBox(QGroupBox):
    """QGroupBox that emits a signal when clicked and supports drag-and-drop for reordering."""
    
    clicked = pyqtSignal(int)  # graph_index
    drag_started = pyqtSignal(int)  # graph_index
    drop_requested = pyqtSignal(int, int)  # from_index, to_index
    
    def __init__(self, title: str, graph_index: int, parent=None):
        super().__init__(title, parent)
        self.graph_index = graph_index
        self.setCursor(Qt.PointingHandCursor)
        self.drag_start_position = None
        self.has_dragged = False  # Track if drag operation started
        self.setAcceptDrops(True)
    
    def mousePressEvent(self, event: QMouseEvent):
        # Only handle if the click is within the title bar area (approx. 30px height)
        if event.button() == Qt.LeftButton and event.pos().y() < 30:
            # Store drag start position for drag-and-drop
            self.drag_start_position = event.pos()
            self.has_dragged = False  # Reset drag flag
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move to start drag operation."""
        if not (event.buttons() & Qt.LeftButton):
            return
        
        if self.drag_start_position is None:
            return
        
        # Check if mouse has moved enough to start drag (minimum distance)
        if (event.pos() - self.drag_start_position).manhattanLength() < 10:
            return
        
        # Mark that drag has started - this prevents click signal from being emitted
        self.has_dragged = True
        
        # Start drag operation
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(str(self.graph_index))
        drag.setMimeData(mime_data)
        
        # Create a pixmap for the drag icon (optional, can be improved)
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos() - self.drag_start_position)
        
        # Emit signal that drag started
        self.drag_started.emit(self.graph_index)
        
        # Execute drag
        drag.exec_(Qt.MoveAction)
        
        # Reset drag start position
        self.drag_start_position = None
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - emit click signal only if no drag occurred."""
        if event.button() == Qt.LeftButton:
            # Only emit clicked signal if:
            # 1. Click was in title bar area
            # 2. Drag did not start (mouse didn't move enough)
            # 3. Drag start position was set (meaning we were tracking a click)
            if (self.drag_start_position is not None and 
                not self.has_dragged and 
                event.pos().y() < 30):
                self.clicked.emit(self.graph_index)
        
        # Reset drag tracking
        self.drag_start_position = None
        self.has_dragged = False
        
        super().mouseReleaseEvent(event)
    
    def dragEnterEvent(self, event):
        """Accept drag events from other graph sections."""
        if event.mimeData().hasText():
            try:
                dragged_index = int(event.mimeData().text())
                # Only accept if dragging from a different graph
                if dragged_index != self.graph_index:
                    event.acceptProposedAction()
                    # Visual feedback: highlight border
                    self.setStyleSheet(self.styleSheet() + """
                        QGroupBox {
                            border: 3px solid rgba(74, 144, 226, 0.8);
                        }
                    """)
                else:
                    event.ignore()
            except ValueError:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Remove visual feedback when drag leaves."""
        # Reset styling (will be reapplied by parent)
        super().dragLeaveEvent(event)
    
    def dragMoveEvent(self, event):
        """Handle drag move to show drop target."""
        if event.mimeData().hasText():
            try:
                dragged_index = int(event.mimeData().text())
                if dragged_index != self.graph_index:
                    event.acceptProposedAction()
                else:
                    event.ignore()
            except ValueError:
                event.ignore()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop to reorder graphs."""
        if event.mimeData().hasText():
            try:
                dragged_index = int(event.mimeData().text())
                if dragged_index != self.graph_index:
                    # Emit signal to request reordering
                    self.drop_requested.emit(dragged_index, self.graph_index)
                    event.acceptProposedAction()
                else:
                    event.ignore()
            except ValueError:
                event.ignore()
        else:
            event.ignore()

