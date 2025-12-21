# type: ignore
"""
Graph Container for Time Graph Widget

A self-contained widget that manages a vertical stack of plots for a single tab.
"""

import logging
from typing import Dict, Any
from PyQt5.QtWidgets import QWidget, QVBoxLayout

# Import necessary components - use absolute imports for standalone app
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.managers.plot_manager import PlotManager
from src.managers.theme_manager import ThemeManager

logger = logging.getLogger(__name__)

class GraphContainer(QWidget):
    """
    Manages a collection of vertically stacked plots within a single tab.
    Each GraphContainer has its own PlotManager.
    Accepts drag-and-drop of parameters/columns to plot them.
    """
    
    # Signals
    from PyQt5.QtCore import pyqtSignal as Signal
    column_dropped = Signal(int, str)  # graph_index, column_name
    
    def __init__(self, theme_manager: ThemeManager, main_widget, tab_index: int = 0, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.main_widget = main_widget  # Store a reliable reference to TimeGraphWidget
        self.tab_index = tab_index  # Store tab index for signal mapping
        
        # Enable drop events
        self.setAcceptDrops(True)
        self.drag_over_graph_index = -1  # Track which graph is being hovered
        self.cursor_manager = None  # Initialize cursor manager attribute
        
        # Get signal_processor from main_widget
        if hasattr(main_widget, 'signal_processor'):
            self.signal_processor = main_widget.signal_processor
        else:
            self.signal_processor = None
            logger.warning("Main widget does not have signal_processor")
        
        # Each container has its own plot manager to handle its own plots
        self.plot_manager = PlotManager(self)
        
        # Main layout for this container
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.plot_manager.get_plot_panel())
        
        logger.info("GraphContainer initialized")

    def set_graph_count(self, count: int):
        """Sets the number of vertically stacked graphs (subplots)."""
        logger.info(f"Setting graph count in container to: {count}")
        self.plot_manager.set_subplot_count(count)

    def add_signal(self, name: str, x_data, y_data, plot_index: int, **kwargs):
        """Adds a signal to a specific subplot within this container."""
        self.plot_manager.add_signal(name, x_data, y_data, plot_index, **kwargs)

    def get_plot_widgets(self):
        """Returns the plot widgets managed by this container."""
        return self.plot_manager.get_plot_widgets()

    def apply_theme(self):
        """Applies the current theme to the plots in this container."""
        plot_colors = self.theme_manager.get_plot_colors()
        self.plot_manager.apply_theme(plot_colors)
    
    def get_global_settings(self) -> dict:
        """Get global settings from the main widget."""
        if hasattr(self.main_widget, 'graph_settings_panel_manager'):
            return self.main_widget.graph_settings_panel_manager.get_global_settings()

        logger.warning("GraphContainer: Main widget reference is invalid or missing graph_settings_panel_manager.")
        # Fallback to default settings
        return {
            'normalize': False,
            'show_grid': True,
            'autoscale': True,
            'show_tooltips': False,
            'snap_to_data': False,
            'line_width': 1,
            'x_axis_mouse': True,
            'y_axis_mouse': True
        }

    def cleanup(self):
        """Temizlik işlemleri."""
        try:
            # Plot manager temizliği
            if hasattr(self.plot_manager, 'cleanup'):
                self.plot_manager.cleanup()
            
            # Cursor manager temizliği
            if self.cursor_manager and hasattr(self.cursor_manager, 'cleanup'):
                self.cursor_manager.cleanup()
            
            # Signal processor temizliği
            if self.signal_processor and hasattr(self.signal_processor, 'cleanup'):
                self.signal_processor.cleanup()
            
            logger.debug("GraphContainer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during GraphContainer cleanup: {e}")
    
    # ========== Drag and Drop Events ==========
    
    def dragEnterEvent(self, event):
        """Handle drag enter - accept parameter/column drops."""
        from PyQt5.QtCore import Qt
        if event.mimeData().hasFormat("application/x-parameter") or event.mimeData().hasText():
            event.acceptProposedAction()
            logger.debug("Drag entered GraphContainer")
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """Handle drag move - determine which graph is being hovered."""
        from PyQt5.QtCore import Qt
        if event.mimeData().hasFormat("application/x-parameter") or event.mimeData().hasText():
            event.acceptProposedAction()
            
            # Determine which graph is under the cursor
            pos = event.pos()
            plot_widgets = self.plot_manager.get_plot_widgets()
            
            self.drag_over_graph_index = -1
            for i, plot_widget in enumerate(plot_widgets):
                if plot_widget.geometry().contains(pos):
                    self.drag_over_graph_index = i
                    break
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave."""
        self.drag_over_graph_index = -1
        logger.debug("Drag left GraphContainer")
    
    def dropEvent(self, event):
        """Handle drop - plot the column on the appropriate graph."""
        from PyQt5.QtCore import Qt
        
        # Extract column name
        column_name = None
        if event.mimeData().hasFormat("application/x-parameter"):
            column_name = bytes(event.mimeData().data("application/x-parameter")).decode('utf-8')
        elif event.mimeData().hasText():
            column_name = event.mimeData().text()
        
        if column_name and self.drag_over_graph_index >= 0:
            logger.info(f"Column '{column_name}' dropped on graph {self.drag_over_graph_index}")
            event.acceptProposedAction()
            
            # Emit signal so main widget can handle plotting
            self.column_dropped.emit(self.drag_over_graph_index, column_name)
        else:
            if not column_name:
                logger.warning("Drop event: no column name found")
            if self.drag_over_graph_index < 0:
                logger.warning("Drop event: no graph index determined")
            event.ignore()
        
        self.drag_over_graph_index = -1

    # ... other methods to delegate calls to plot_manager can be added here ...
    # (e.g., clear_signals, reset_view, etc.)
