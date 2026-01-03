# type: ignore
"""
Signal Row Widget

Custom widget for signal rows with context menu support.
"""

import logging
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class SignalRowWidget(QWidget):
    """Custom widget for signal rows with context menu support."""
    
    context_menu_requested = pyqtSignal(object, str, int)  # pos, signal_name, graph_index
    
    def __init__(self, signal_name: str, graph_index: int, parent=None):
        super().__init__(parent)
        self.signal_name = signal_name
        self.graph_index = graph_index
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        logger.info(f"SignalRowWidget created for: {signal_name}, graph: {graph_index}")
    
    def _on_context_menu(self, pos):
        """Handle context menu request."""
        logger.info(f"SignalRowWidget context menu triggered for: {self.signal_name}, graph: {self.graph_index}")
        self.context_menu_requested.emit(pos, self.signal_name, self.graph_index)

