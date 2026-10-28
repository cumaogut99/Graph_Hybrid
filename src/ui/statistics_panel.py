# type: ignore
"""
Statistics Panel for Time Analysis Widget - Refactored

✅ REFACTORED: Modular architecture with widget components
- ClickableColorLabel → statistics/color_label.py
- ClickableGroupBox → statistics/clickable_groupbox.py
- SignalRowWidget → statistics/signal_row.py

Displays real-time statistics for each signal:
- Signal Name
- Mean
- Max  
- Min
- RMS (Root Mean Square)
- Standard Deviation

Updates dynamically based on cursor/range selection.
"""

import logging
from typing import Any, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QScrollArea, QTableWidget, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QPoint
from PyQt5.QtGui import QFont, QPalette, QMouseEvent, QDrag, QPainter

# ✅ REFACTORED: Import from modular statistics package
from .statistics import PanelThemeApplier
from src.ui.statistics.columns import StatisticsColumnsMixin
from src.ui.statistics.graph_sections import StatisticsGraphSectionsMixin
from src.ui.statistics.signal_rows import StatisticsSignalRowsMixin
from src.ui.statistics.values import StatisticsValuesMixin

logger = logging.getLogger(__name__)

# ✅ REFACTORED: Widget classes now imported from statistics package
# Old duplicate classes removed - using modular components instead

class StatisticsPanel(StatisticsColumnsMixin, StatisticsGraphSectionsMixin, StatisticsSignalRowsMixin, StatisticsValuesMixin, QWidget):
    """
    Panel for displaying real-time signal statistics organized by graphs.
    
    Features:
    - Separate section for each graph
    - Horizontal layout for efficient space usage
    - Color-coded signal identification
    - Scrollable layout for multiple graphs
    - Coordinated with statistics settings panel
    - Clickable graph titles to open individual graph settings
    """
    
    # Signal emitted when a graph title is clicked
    signal_color_changed = pyqtSignal(str, str) # Emits signal_name and new_color_hex
    signal_remove_requested = pyqtSignal(str, int) # Emits signal_name and graph_index
    graph_reorder_requested = pyqtSignal(int, int)  # from_index, to_index

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.graph_sections = {}  # graph_index -> QGroupBox
        self.signal_data = {}  # full_signal_name -> {graph_index, signal_name, labels_dict}
        self.visible_stats = {'mean', 'max', 'min', 'rms', 'std', 'duty_cycle'}  # Default visible stats
        self.cursor_mode = "dual"  # Cursor mode is permanently 'dual'
        
        # Cursor position tracking
        self.cursor_positions = {}  # Store current cursor positions
        
        # Datetime formatting
        self.is_datetime_axis = False  # Track if we should format cursor values as datetime
        
        # Column width management
        self.column_widths = {
            'signal': 180,      # Signal name column
            'c1': 80,           # Cursor 1 column
            'c2': 80,           # Cursor 2 column
            'min': 80,          # Min column
            'mean': 80,         # Mean column
            'max': 80,          # Max column
            'rms': 80,          # RMS column
            'std': 80,          # Standard deviation column
            'duty_cycle': 90    # Duty cycle column
        }
        
        # ✅ REFACTORED: Initialize theme helper
        self.theme_applier = PanelThemeApplier(self)
        
        # Setup UI
        self._setup_ui()
        self._setup_styling()
        
        # Apply initial theme styling
        self.theme_applier.apply_initial_theme_styling()
        
        logger.debug("StatisticsPanel initialized with graph-based organization")

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def _apply_initial_theme_styling(self):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.apply_initial_theme_styling()

    def _setup_ui(self):
        """Setup the statistics panel UI with separate tables for each graph."""
        # Main layout - optimized for space efficiency
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        self.main_layout.setSpacing(1)  # Minimal spacing
        
        # Common header for all graphs
        self.common_header = self._create_common_header()
        self.main_layout.addWidget(self.common_header)
        
        # Scroll area for graph sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for scroll area - minimal spacing for efficiency
        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(1)  # Reduced spacing between graph sections
        
        scroll_area.setWidget(self.container_widget)
        self.main_layout.addWidget(scroll_area)
        
        # Dictionary to store tables for each graph
        self.graph_tables = {}
        self.graph_sections = {}
        
        # Cursor info panel at the bottom (fixed)
        self.cursor_info_panel = self._create_cursor_info_panel()
        self.cursor_info_panel.hide()  # Initially hidden, shown when dual cursor mode is active
        self.main_layout.addWidget(self.cursor_info_panel)

    def _setup_styling(self):
        """Setup the panel styling with theme support."""
        self._apply_theme_styling()

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def _apply_theme_styling(self, theme_colors=None):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.apply_theme_styling(theme_colors)

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def update_theme(self, theme_colors=None):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.update_theme(theme_colors)

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def _update_header_table_theme(self, theme_colors=None):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.update_header_table_theme(theme_colors)

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def _update_cursor_info_panel_theme(self, theme_colors=None):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.update_cursor_info_panel_theme(theme_colors)

    # ✅ REFACTORED: Backward compatibility - delegates to helper
    def _apply_table_styling_to_single_table(self, table: QTableWidget):
        """✅ REFACTORED: Delegate to theme helper"""
        self.theme_applier.apply_table_styling_to_single_table(table)

    # Button-related methods removed for cleaner interface
