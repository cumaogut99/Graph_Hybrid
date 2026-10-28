# type: ignore
"""
Plot Manager for Time Graph Widget - Refactored

✅ REFACTORED: Modular architecture with helper classes
- DateTimeAxisItem → plot/datetime_axis.py
- Tooltip system → plot/plot_tooltips.py  
- Secondary axis → plot/plot_secondary_axis.py

Manages the plotting interface including:
- Multiple stacked subplots
- Signal visualization
- Plot synchronization
- View management
"""

import logging
from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np
from PyQt5.QtWidgets import QPushButton, QMenu
from PyQt5.QtCore import pyqtSignal as Signal, QObject

# ✅ REFACTORED: Import from modular plot package
from .plot import PlotTooltipsHelper, PlotSecondaryAxisHelper

if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget
from src.managers.plot.lod import PlotLodMixin
from src.managers.plot.subplots import PlotSubplotsMixin
from src.managers.plot.signals import PlotSignalsMixin
from src.managers.plot.view import PlotViewMixin
from src.managers.plot.markers import PlotMarkersMixin

logger = logging.getLogger(__name__)

class PlotManager(PlotLodMixin, PlotSubplotsMixin, PlotSignalsMixin, PlotViewMixin, PlotMarkersMixin, QObject):
    """Manages the plotting interface for the Time Graph Widget."""
    
    # Signals
    plot_clicked = Signal(int, float, float)  # plot_index, x, y
    range_selected = Signal(float, float)  # start, end
    markers_changed = Signal()  # a marker was added or removed
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.plot_widgets = []
        self.current_signals = {}  # Dict of signal_name -> plot_data_item
        self.signal_colors = {}  # Store original colors for each signal
        self.subplot_count = 1
        self.max_subplots = 10
        self.min_subplots = 1
        
        # UI components that will be managed
        self.plot_panel = None
        self.main_layout = None
        self.plot_container = None
        self.settings_container = None
        
        # Grid visibility state
        self.grid_visible = True
        
        # Theme settings that will be updated by apply_theme
        self.theme_colors = {
            'background': '#1e1e1e',
            'axis_pen': '#ffffff',
            'grid_alpha': 0.3
        }
        
        # Y ekseni hizalama için sabit genişlik ayarı
        self.y_axis_width = 80  # Piksel cinsinden sabit genişlik
        
        # Snap to data points feature
        self.snap_to_data_enabled = False
        
        # Downsampling state
        
        # LOD (Level of Detail) debounce timer
        # Prevents too frequent updates during rapid zooming
        self._lod_timer = None
        self._lod_pending_ranges = {}  # plot_index -> ranges
        
        # Store original data ranges for proper view reset (before downsampling)
        # Format: {plot_index: {'x_min': float, 'x_max': float, 'y_min': float, 'y_max': float}}
        self.original_data_ranges = {}
        
        # ✅ REFACTORED: Initialize helper classes
        self.tooltips_helper = PlotTooltipsHelper(self)
        self.secondary_axis_helper = PlotSecondaryAxisHelper(self)
        
        # ✅ NEW: Initialize LOD Engine for view-dependent downsampling
        self._lod_engine = None  # Will be initialized when signal_processor is available
        
        # ✅ NEW: LOD mode cache and hysteresis to prevent flicker
        # Tracks current LOD mode per signal: signal_key -> "RAW" | "AGGREGATED"
        self._current_lod_modes = {}
        # Hysteresis band: 20% tolerance to prevent rapid mode switching at threshold
        self._lod_hysteresis_up = 1.2   # Switch to AGGREGATED when > threshold * 1.2
        self._lod_hysteresis_down = 0.8  # Switch to RAW when < threshold * 0.8
        
        # Markers: numbered vertical lines shown on every plot of this tab.
        # Kept here (not as plot items) so they survive plot clears/rebuilds.
        self.markers: List[Dict[str, Any]] = []  # [{'number': int, 'x': float}]
        self._next_marker_number = 1
        self._marker_items: List[tuple] = []  # (plot_widget, InfiniteLine)
        self._context_menu_x = None  # x where the plot context menu was opened

        self._setup_plot_panel()
        self._rebuild_ui()  # Initial UI creation
    
    # ✅ REFACTORED: Backward compatibility - Properties delegating to helpers
    @property
    def tooltips_enabled(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.tooltips_helper.tooltips_enabled
    
    @tooltips_enabled.setter
    def tooltips_enabled(self, value):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper.tooltips_enabled = value
    
    @property
    def tooltip_items(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.tooltips_helper.tooltip_items
    
    @property
    def secondary_axis_enabled(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.secondary_axis_helper.secondary_axis_enabled
    
    @secondary_axis_enabled.setter
    def secondary_axis_enabled(self, value):
        """✅ REFACTORED: Delegate to helper"""
        self.secondary_axis_helper.secondary_axis_enabled = value
    
    @property
    def secondary_viewboxes(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.secondary_axis_helper.secondary_viewboxes
    
    @property
    def secondary_axes(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.secondary_axis_helper.secondary_axes
    
    @property
    def signal_axis_assignment(self):
        """✅ REFACTORED: Delegate to helper"""
        return self.secondary_axis_helper.signal_axis_assignment
    
    def _get_global_settings(self) -> dict:
        """Get global settings from parent GraphContainer."""
        if hasattr(self.parent, 'get_global_settings'):
            return self.parent.get_global_settings()

        logger.warning(f"PlotManager: Parent {type(self.parent).__name__} does not have get_global_settings method. Critical error.")
        # This fallback should ideally not be reached now.
        return {
            'normalize': False,
            'show_grid': True,
            'autoscale': True,
            'show_legend': True,
            'show_tooltips': False,
            'snap_to_data': True,
            'line_width': 1,
            'x_axis_mouse': True,
            'y_axis_mouse': True,
        }

    def _get_data_manager(self):
        """Get data manager instance."""
        if hasattr(self.parent, 'main_widget') and hasattr(self.parent.main_widget, 'data_manager'):
            return self.parent.main_widget.data_manager
        return None
    
    def _get_signal_processor(self):
        """Get signal processor instance for LOD operations."""
        # PlotManager.parent is GraphContainer
        # GraphContainer.main_widget is TimeGraphWidget
        # TimeGraphWidget.signal_processor is what we need
        if hasattr(self.parent, 'main_widget') and hasattr(self.parent.main_widget, 'signal_processor'):
            return self.parent.main_widget.signal_processor
        return None

    def set_tooltips_enabled(self, enabled: bool):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper.set_tooltips_enabled(enabled)

    def set_tooltips_enabled(self, enabled: bool):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper.set_tooltips_enabled(enabled)

    def _setup_tooltip_for_plot(self, plot_widget, enabled: bool):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper._setup_tooltip_for_plot(plot_widget, enabled)

    def _on_mouse_moved(self, pos, plot_widget):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper._on_mouse_moved(pos, plot_widget)

    def _on_mouse_left(self, plot_widget):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper._on_mouse_left(plot_widget)

    def _find_closest_signal_to_cursor(self, x_pos: float, y_pos: float, plot_widget):
        """✅ REFACTORED: Delegate to helper"""
        return self.tooltips_helper._find_closest_signal_to_cursor(x_pos, y_pos, plot_widget)

    def _generate_enhanced_tooltip_text(self, x_pos: float, y_pos: float, closest_signal_info: dict) -> str:
        """✅ REFACTORED: Delegate to helper"""
        return self.tooltips_helper._generate_enhanced_tooltip_text(x_pos, y_pos, closest_signal_info)

    def _generate_tooltip_text(self, x_pos: float, y_pos: float) -> str:
        """✅ REFACTORED: Delegate to helper"""
        return self.tooltips_helper._generate_tooltip_text(x_pos, y_pos)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------
    MARKER_COLOR = '#ff9f1c'

    def set_secondary_axis_enabled(self, enabled: bool):
        """✅ REFACTORED: Delegate to helper"""
        self.secondary_axis_helper.set_secondary_axis_enabled(enabled)

    def _setup_secondary_axis_for_plot(self, plot_widget, plot_index: int):
        """✅ REFACTORED: Delegate to helper"""
        self.secondary_axis_helper._setup_secondary_axis_for_plot(plot_widget, plot_index)

    def _remove_secondary_axis_for_plot(self, plot_index: int):
        """✅ REFACTORED: Delegate to helper"""
        self.secondary_axis_helper._remove_secondary_axis_for_plot(plot_index)

    def _assign_signal_to_axis(self, signal_name: str, y_data: np.ndarray, plot_index: int) -> str:
        """✅ REFACTORED: Delegate to helper"""
        return self.secondary_axis_helper.assign_signal_to_axis(signal_name, y_data, plot_index)
