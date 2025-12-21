# type: ignore
"""
Advanced Time Graph Widget - Refactored Main Class

Professional architecture with separation of concerns for maintainability and performance.
"""

import logging
import time  # ✅ OPTIMIZED: For throttling timestamp
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea, QLabel, QDialog, QGroupBox, 
    QTabWidget, QGridLayout, QStackedWidget, QToolButton,
    QTabBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QObject, QThread
from PyQt5.QtGui import QIcon

# Import our modular components - Always use absolute imports for standalone app
from src.managers.filter_manager import FilterManager
from src.graphics.graph_renderer import GraphRenderer
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.managers.toolbar_manager import ToolbarManager
from src.managers.plot_manager import PlotManager
from src.managers.legend_manager import LegendManager
from src.data.signal_processor import SignalProcessor
from src.managers.theme_manager import ThemeManager
from src.managers.cursor_manager import CursorManager
from src.ui.statistics_panel import StatisticsPanel
from src.managers.data_manager import TimeSeriesDataManager
from src.managers.settings_panel_manager import SettingsPanelManager
from src.managers.statistics_settings_panel_manager import StatisticsSettingsPanelManager
from src.managers.graph_settings_panel_manager import GraphSettingsPanelManager
from src.managers.parameters_panel_manager import ParametersPanelManager
from src.managers.bitmask_panel_manager import BitmaskPanelManager
from src.ui.graph_settings_dialog import GraphSettingsDialog
from src.ui.graph_advanced_settings_dialog import GraphAdvancedSettingsDialog
from src.graphics.graph_container import GraphContainer
from src.managers.status_bar_manager import StatusBarManager
from src.graphics.loading_overlay import LoadingManager
from src.managers.correlations_panel_manager import CorrelationsPanelManager
from src.utils.feature_stability_tracker import FeatureStabilityTracker


logger = logging.getLogger(__name__)

class SignalProcessingWorker(QObject):
    """Worker thread for processing signal data in the background."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, df, is_normalized, time_column):
        super().__init__()
        self.df = df
        self.is_normalized = is_normalized
        self.time_column = time_column

    def run(self):
        """Processes the data and emits the result."""
        try:
            processor = SignalProcessor()
            all_signals = processor.process_data(self.df, self.is_normalized, self.time_column)
            self.finished.emit(all_signals)
        except Exception as e:
            logger.error(f"Error during signal processing: {e}")
            self.error.emit(str(e))


class TimeGraphWidget(QWidget):
    """
    Advanced Time Graph Widget - Professional Architecture
    
    Modular components:
    - ToolbarManager: Handles all toolbar controls
    - PlotManager: Manages synchronized plots
    - LegendManager: Real-time signal legend
    - SignalProcessor: High-performance signal processing
    - ThemeManager: Professional theming
    """
    
    # Signals for external communication
    data_changed = Signal(object)
    cursor_moved = Signal(str, object)
    range_selected = Signal(tuple)
    statistics_updated = Signal(dict)
    
    def __init__(self, parent=None, loading_manager=None):
        super().__init__(parent)
        self.graph_containers = []  # Initialize the list here
        self.loading_manager = loading_manager
        
        # Duty cycle threshold settings
        self.duty_cycle_threshold_mode = "auto"  # "auto" or "manual"
        self.duty_cycle_threshold_value = 0.0
        
        # Initialize modular components
        self.filter_manager = None
        self.graph_renderer = None
        self.status_bar = None
        
        # Threading for signal processing
        self.processing_thread = None
        self.processing_worker = None
        
        # Initialize managers and components
        self._initialize_managers()
        
        # Setup UI architecture
        self._setup_ui()
        
        # Connect signals between components
        self._setup_connections()
        
        # Apply initial theme
        self._apply_theme()
        
        logger.debug("TimeGraphWidget initialized successfully")
    
    def _initialize_managers(self):
        """Initialize all manager components."""
        # Note: PlotManager is no longer a central manager.
        # Each GraphContainer will have its own.
        self.data_manager = TimeSeriesDataManager()
        self.signal_processor = SignalProcessor()
        
        # Initialize graph renderer after signal processor
        self.graph_renderer = GraphRenderer(self.signal_processor, {}, self)
        
        # Initialize filter manager with parent
        self.filter_manager = FilterManager(self)
        
        self.toolbar_manager = ToolbarManager(self)
        self.legend_manager = LegendManager(self)
        self.settings_panel_manager = SettingsPanelManager(self)
        self.statistics_settings_panel_manager = StatisticsSettingsPanelManager(self)
        self.graph_settings_panel_manager = GraphSettingsPanelManager(self)
        self.parameters_panel_manager = ParametersPanelManager(self)
        self.theme_manager = ThemeManager()
        self.bitmask_panel_manager = BitmaskPanelManager(self.data_manager, self.theme_manager, self)
        
        self.cursor_manager = None
        self.statistics_panel = StatisticsPanel()
        self.channel_stats_widgets = {} # Initialize here to prevent race condition
        
        self.is_normalized = False
        self.current_cursor_position = None
        self.selected_range = None
        
        # PERFORMANCE: Throttling timer for statistics updates
        # ✅ OPTIMIZED: Use throttling instead of debouncing for real-time feel
        self._statistics_update_timer = QTimer()
        self._statistics_update_timer.setSingleShot(True)
        self._statistics_update_timer.setInterval(50)  # 50ms throttle (20 FPS)
        self._statistics_update_timer.timeout.connect(self._perform_statistics_update)
        self._pending_cursor_positions = None
        self._last_statistics_update_time = 0  # Track last update time for throttling
        self.current_cursor_mode = "dual"  # Default cursor mode
        self._last_graph_count = 1
        
        # State for dynamic statistics panel
        self.visible_stats_columns = self.statistics_settings_panel_manager.get_visible_columns()

        # Signal mapping now needs to be aware of tabs
        self.graph_signal_mapping = {} # This will become a dict of dicts: {tab_index: {graph_index: [signals]}}
        
        # Per-graph settings storage
        self.graph_settings = {}  # {tab_index: {graph_index: {setting_name: value}}}

    def _setup_ui(self):
        """Setup the main UI layout with a QTabWidget."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toolbar_manager.get_toolbar())
        
        self.content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.content_splitter)
        
        # --- Left Panel Management using QStackedWidget ---
        self.left_panel_stack = QStackedWidget()
        self.left_panel_stack.setMinimumWidth(280)
        self.left_panel_stack.setMaximumWidth(350)
        
        self.settings_panel = self.settings_panel_manager.get_settings_panel()
        self.statistics_settings_panel = self.statistics_settings_panel_manager.get_settings_panel()
        self.graph_settings_panel = self.graph_settings_panel_manager.get_settings_panel()
        
        # Create new analysis panels
        self.correlations_panel = self._create_correlations_panel()
        self.bitmask_panel = self._create_bitmask_panel()
        self.parameters_panel = self.parameters_panel_manager.get_panel()
        
        self.left_panel_stack.addWidget(self.settings_panel)
        self.left_panel_stack.addWidget(self.statistics_settings_panel)
        self.left_panel_stack.addWidget(self.graph_settings_panel)
        self.left_panel_stack.addWidget(self.parameters_panel)
        self.left_panel_stack.addWidget(self.correlations_panel)
        self.left_panel_stack.addWidget(self.bitmask_panel)
        
        self.left_panel_stack.setVisible(False) # Hide the stack initially
        self.content_splitter.addWidget(self.left_panel_stack)
        # --- End of Left Panel Management ---

        # Create all UI elements first before connecting signals that might use them
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabBarDoubleClicked.connect(self._rename_tab)
        self.tab_widget.tabCloseRequested.connect(self._remove_tab)

        # Add a '+' button to the tab bar for adding new tabs
        self.add_tab_button = QToolButton(self)
        self.add_tab_button.setText("+")
        self.add_tab_button.setCursor(Qt.PointingHandCursor)
        self.add_tab_button.clicked.connect(self._add_tab)
        self.tab_widget.setCornerWidget(self.add_tab_button, Qt.TopRightCorner)
        
        # Apply modern styling
        self._apply_tab_stylesheet()

        self.content_splitter.addWidget(self.tab_widget)
        
        self.channel_stats_panel = self._create_channel_statistics_panel()
        self.content_splitter.addWidget(self.channel_stats_panel)
        
        self.content_splitter.setSizes([280, 660, 300])
        self.content_splitter.setCollapsible(0, True)
        self.content_splitter.setCollapsible(1, False)
        self.content_splitter.setCollapsible(2, False)

        # Now that UI elements exist, connect the tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Add the first tab. This will trigger _on_tab_changed, which requires
        # ati_stats_panel and its layout to exist.
        self._add_tab()
        
        # Initialize cursor manager after UI is fully set up
        QTimer.singleShot(200, self._delayed_initial_setup)
        
    def _create_channel_statistics_panel(self):
        """Creates the main container widget for the statistics panel."""
        # Use our modern StatisticsPanel class
        self.statistics_panel = StatisticsPanel(self)
        self.statistics_panel.setMinimumWidth(300)
        
        # Connect graph settings signal
        self.statistics_panel.graph_settings_requested.connect(self._on_graph_settings_requested)
        self.statistics_panel.signal_color_changed.connect(self._on_signal_color_changed)
        self.statistics_panel.signal_remove_requested.connect(self._on_signal_remove_requested)
        self.statistics_panel.graph_reorder_requested.connect(self._on_graph_reorder_requested)
        
        # Connect theme change signal
        self.theme_manager.theme_changed.connect(lambda: self.statistics_panel.update_theme(self.theme_manager.get_theme_colors()))
        
        return self.statistics_panel

    def _create_correlations_panel(self):
        """Create the correlations analysis panel using the dedicated manager."""
        from src.managers.correlations_panel_manager import CorrelationsPanelManager
        self.correlations_panel_manager = CorrelationsPanelManager(self)
        return self.correlations_panel_manager.get_panel()

    def _create_bitmask_panel(self):
        """Create the bitmask analysis panel."""
        return self.bitmask_panel_manager.get_widget()

    def _delayed_initial_setup(self):
        """Delayed setup after UI is fully initialized."""
        logger.debug("Starting delayed initial setup")
        
        # Initialize cursor manager for the first time with dual mode
        self._initialize_cursor_manager()
        
        # Cursor mode is permanently set to 'dual'
        if self.cursor_manager:
            self.cursor_manager.set_mode("dual")
        
        # Update statistics panel with dual cursor mode
        if hasattr(self, 'statistics_panel') and self.statistics_panel:
            self.statistics_panel.set_cursor_mode("dual")
            
        # Update statistics settings panel with dual cursor mode
        if hasattr(self, 'statistics_settings_panel_manager') and self.statistics_settings_panel_manager:
            self.statistics_settings_panel_manager.set_cursor_mode("dual")
        
        logger.debug("Delayed initial setup completed")

    def _on_graph_count_changed(self, count: int):
        """Delegates graph count change to the active tab's container."""
        active_container = self.get_active_graph_container()
        if active_container:
            # ✅ FIX: Grafik sayısı değiştiğinde graph_signal_mapping'i ÖNCE güncelle
            # Yeni grafikler için boş mapping ekle, mevcut eşlemeleri koru
            # ✅ FIX: Get tab index from GraphContainer instead of tab_widget
            current_tab = active_container.tab_index if hasattr(active_container, 'tab_index') else self.tab_widget.currentIndex()
            if current_tab not in self.graph_signal_mapping:
                self.graph_signal_mapping[current_tab] = {}
            
            # Yeni grafikler için boş liste ekle (mevcut grafikleri koru)
            for i in range(count):
                if i not in self.graph_signal_mapping[current_tab]:
                    self.graph_signal_mapping[current_tab][i] = []
            
            # Mevcut grafik sayısından fazla olan mapping'leri temizle
            # UYARI: Bu sinyalleri mapping'den kaldırıyor ama veriyi kaybetmiyor
            # Çünkü veriler self.signal_processor içinde tutuluyor
            to_remove = [k for k in self.graph_signal_mapping[current_tab].keys() if k >= count]
            for key in to_remove:
                del self.graph_signal_mapping[current_tab][key]
            
            # Graph signal mapping updated for current tab
            
            # SONRA grafik sayısını değiştir (bu işlem graph_signal_mapping'i kullanacak)
            active_container.set_graph_count(count)
            
            # Use delayed initialization to ensure plots are fully ready
            QTimer.singleShot(100, self._delayed_post_graph_change)
            # Rebuild the graph settings panel immediately
            self.graph_settings_panel_manager.rebuild_controls(count)
    
    def _delayed_post_graph_change(self):
        """Delayed initialization after graph count change to ensure plots are ready."""
        active_container = self.get_active_graph_container()
        if not active_container:
            return
        count = active_container.plot_manager.get_subplot_count()
        
        # Store current cursor mode before reinitializing
        current_mode = getattr(self, 'current_cursor_mode', 'dual')
        
        # Re-initialize cursors after plots are fully ready
        self._initialize_cursor_manager()
        
        # Ensure cursor mode is preserved and applied
        if self.cursor_manager and current_mode:
            self.cursor_manager.set_mode(current_mode)
            logger.info(f"Restored cursor mode to '{current_mode}' after graph count change")
        
        # Force cursor mode to match toolbar selection
        self._force_cursor_mode_sync()
        
        # Apply saved graph settings after graph count change
        self._apply_saved_graph_settings()
        
        # Rebuild the bitmask panel
        self.bitmask_panel_manager.update_graph_sections(count)
        
        # Recreate the stats panel to match the new number of graphs
        self._recreate_statistics_panel()
        # DON'T update statistics here - wait for cursor movement for performance

    def _apply_tab_stylesheet(self):
        """Apply a modern stylesheet to the tab widget."""
        colors = self.theme_manager.get_theme_colors()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border-top: 2px solid {colors.get('primary', '#4a90e2')};
                background: {colors.get('surface', '#2d2d2d')};
            }}
            QTabBar::tab {{
                background: {colors.get('surface_variant', '#3c3c3c')};
                color: {colors.get('text_secondary', '#e0e0e0')};
                border: 1px solid {colors.get('surface', '#2d2d2d')};
                border-bottom-color: {colors.get('primary', '#4a90e2')}; 
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 100px;
                padding: 8px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background: {colors.get('surface', '#2d2d2d')};
                color: {colors.get('text_primary', '#ffffff')};
                font-weight: bold;
            }}
            QTabBar::close-button {{
                /* ✅ FIX: Use absolute path for icon */
                image: url({os.path.abspath('icons/x.svg').replace(os.sep, '/')});
                subcontrol-position: right;
                subcontrol-origin: padding;
                border: none;
                background: transparent;
                padding: 4px;
            }}
            QTabBar::close-button:hover {{
                background-color: #e81123;
            }}
            QTabBar::close-button:pressed {{
                background-color: #f1707a;
            }}
            QToolButton {{
                background-color: {colors.get('surface_variant', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {colors.get('surface', '#2d2d2d')};
            }}
            QToolButton:hover {{
                background-color: {colors.get('primary', '#4a90e2')};
            }}
        """)

    def _on_tab_count_changed(self, count: int):
        """Handle tab count changes from toolbar."""
        # This method is now deprecated and should not be used.
        pass

    def _on_tab_changed(self, index: int):
        """Handle tab switching."""
        #logger.debug(f"Switched to tab {index}")
        # Update graph count on toolbar to reflect the new active tab
        active_container = self.get_active_graph_container()
        if active_container:
            graph_count = active_container.plot_manager.get_subplot_count()
            self.toolbar_manager.set_graph_count(graph_count)
            # Rebuild graph settings panel when tab changes
            self.graph_settings_panel_manager.rebuild_controls(graph_count)

        # Use delayed initialization for tab changes too
        QTimer.singleShot(50, self._delayed_post_tab_change)
    
    def _delayed_post_tab_change(self):
        """Delayed initialization after tab change to ensure plots are ready."""
        # Store current cursor mode before reinitializing
        current_mode = getattr(self, 'current_cursor_mode', 'dual')
        
        # Cursors are initialized when tabs change to handle different plot widgets
        self._initialize_cursor_manager()
        
        # Ensure cursor mode is preserved and applied
        if self.cursor_manager and current_mode:
            self.cursor_manager.set_mode(current_mode)
            logger.info(f"Restored cursor mode to '{current_mode}' after tab change")
        
        # Force cursor mode to match toolbar selection
        self._force_cursor_mode_sync()
        
        # Apply saved graph settings for the new tab
        self._apply_saved_graph_settings()
        
        # Update statistics for the new tab
        self._recreate_statistics_panel()

    def _initialize_cursor_manager(self):
        """Initialize or re-initialize cursor manager for the active tab."""
        active_container = self.get_active_graph_container()
        if not active_container:
            if self.cursor_manager:
                self.cursor_manager.deleteLater()
                self.cursor_manager = None
            return

        plot_widgets = active_container.get_plot_widgets()
        
        # Check if plot widgets are properly initialized
        if not plot_widgets:
            logger.warning("No plot widgets available for cursor initialization")
            return
            
        # For initial setup, don't require all widgets to be visible yet
        # Just check if they exist and have valid view boxes
        try:
            for widget in plot_widgets:
                if not hasattr(widget, 'getViewBox') or widget.getViewBox() is None:
                    logger.warning("Plot widget doesn't have valid ViewBox")
                    return
        except Exception as e:
            logger.warning(f"Plot widgets not ready for cursor initialization: {e}")
            return
        
        # This check was causing the initial load bug. Cursors must be re-initialized
        # even if the plot widgets are the same, because their view range might have changed.
        # if self.cursor_manager and self.cursor_manager.plots == plot_widgets:
        #     return

        if self.cursor_manager:
            try:
                self.cursor_manager.cursor_moved.disconnect()
                self.cursor_manager.range_changed.disconnect()
            except TypeError: pass
            self.cursor_manager.deleteLater()
            self.cursor_manager = None
        
        if plot_widgets:
            # Force an update of the view range before creating cursors.
            # This is critical to prevent a race condition where cursors are
            # created before the plot's autorange has been calculated.
            for pw in plot_widgets:
                pw.autoRange()

            self.cursor_manager = CursorManager(plot_widgets)
            
            # Assign the new cursor manager to the active container
            if active_container:
                active_container.cursor_manager = self.cursor_manager
            
            self.cursor_manager.cursor_moved.connect(self._on_cursor_moved)
            self.cursor_manager.range_changed.connect(self._on_range_changed)
            
            # Connect bitmask panel to cursor movement
            self.cursor_manager.cursor_moved.connect(self.bitmask_panel_manager.on_cursor_position_changed)
            
            # Sync initial snap to data setting from graph settings panel
            if hasattr(self, 'graph_settings_panel_manager'):
                initial_snap_setting = self.graph_settings_panel_manager.global_settings.get('snap_to_data', False)
                self.cursor_manager.set_snap_to_data(initial_snap_setting)
                
                # Sync initial tooltip setting from graph settings panel
                initial_tooltip_setting = self.graph_settings_panel_manager.global_settings.get('show_tooltips', True)
                active_container.plot_manager.set_tooltips_enabled(initial_tooltip_setting)
            
            # Viewport lock feature removed - cursors now stay at fixed data coordinates
            
            # Cursor mode is permanently 'dual'
            self.cursor_manager.set_mode("dual")
            self.current_cursor_mode = "dual"
            logger.debug("Applied cursor mode: dual (permanent setting)")

    def _force_cursor_mode_sync(self):
        """Ensure cursor mode is set to dual (permanently)."""
        if not self.cursor_manager:
            return
            
        # Cursor mode is permanently 'dual' - no toolbar selection needed
        current_manager_mode = getattr(self.cursor_manager, 'current_mode', None)
        if current_manager_mode != "dual":
            if hasattr(self.cursor_manager, 'set_mode'):
                self.cursor_manager.set_mode("dual")
                self.current_cursor_mode = "dual"
                logger.debug("Synced cursor mode to dual (permanent setting)")

    def _initialize_signal_mapping(self, signal_names: list[str]):
        """
        Initializes or updates the signal-to-graph mapping for the new tabbed structure.
        By default, graphs start empty - user must manually select which signals to plot.
        """
        self.graph_signal_mapping = {}
        for i in range(self.tab_widget.count()):
            self.graph_signal_mapping[i] = {}

        if not signal_names:
            return

        # NEW BEHAVIOR: Don't auto-distribute signals to graphs
        # Graphs start empty, user must manually select signals to plot
        # This prevents automatic plotting of all parameters on startup
        
        logger.info(f"Signal mapping initialized with {len(signal_names)} available signals - graphs start empty for manual selection")
            
    def _redraw_all_signals(self):
        """Redraws all signals across all tabs based on the current mapping."""
        import time
        start_time = time.time()
        logger.info("[PERF] ========== _redraw_all_signals() START ==========")
        
        # CRITICAL FIX: Disable statistics updates BEFORE any cursor operations
        # This prevents expensive C++ statistics calculations during graph redraw
        self._suppress_statistics_updates = True
        logger.debug("[PERF] Statistics updates suppressed for redraw")
        
        all_signals = self.signal_processor.get_all_signals()
        all_signal_names = sorted(list(all_signals.keys()))
        logger.info(f"[PERF] Got {len(all_signals)} signals in {(time.time()-start_time)*1000:.1f}ms")

        # Store current cursor mode before redrawing
        current_mode = getattr(self, 'current_cursor_mode', 'dual')
        cursor_positions = {}
        
        # Save cursor positions if they exist
        if self.cursor_manager and hasattr(self.cursor_manager, 'get_cursor_positions'):
            cursor_positions = self.cursor_manager.get_cursor_positions()
            logger.debug(f"Saved cursor positions: {cursor_positions}")

        self.legend_manager.clear_all_items()

        for tab_index, container in enumerate(self.graph_containers):
            container.plot_manager.clear_all_signals()
            
            tab_mapping = self.graph_signal_mapping.get(tab_index, {})
            for graph_index, signal_names in tab_mapping.items():
                if graph_index < container.plot_manager.get_subplot_count():
                    for name in signal_names:
                        if name in all_signals:
                            signal_data = all_signals[name]
                            signal_index = all_signal_names.index(name)
                            color = self.theme_manager.get_signal_color(signal_index)
                            
                            container.plot_manager.add_signal(
                                name, 
                                signal_data['x_data'], 
                                signal_data['y_data'], 
                                plot_index=graph_index, 
                                pen=color
                            )
                            
                            # Add to legend only once
                            if not self.legend_manager.has_item(name):
                                last_value = float(signal_data['y_data'][-1]) if signal_data['y_data'].size > 0 else 0.0
                            self.legend_manager.add_legend_item(name, color, last_value)
        
        # Restore cursors after redrawing signals
        if current_mode and current_mode != "none":
            # Use a timer to ensure plots are fully ready before restoring cursors
            QTimer.singleShot(50, lambda: self._restore_cursors_after_redraw(current_mode, cursor_positions))
        
        # Apply saved graph settings after redrawing signals
        self._apply_saved_graph_settings()
        
        # Apply limit lines BEFORE auto-ranging so they are included in the view
        self._apply_limit_lines_to_all_graphs()
        
        # CRITICAL: Auto-range AFTER applying settings AND limit lines to show all data
        # This must come after limit lines are added so they are included in the range calculation
        for tab_index, container in enumerate(self.graph_containers):
            plot_widgets = container.plot_manager.get_plot_widgets()
            for idx, plot_widget in enumerate(plot_widgets):
                
                # First enable auto-range for both axes
                plot_widget.enableAutoRange(axis='x', enable=True)
                plot_widget.enableAutoRange(axis='y', enable=True)
                # Then trigger auto-range to fit all data INCLUDING limit lines
                plot_widget.autoRange()
                # Disable auto-range after initial fit so user can zoom/pan freely
                plot_widget.enableAutoRange(axis='x', enable=False)
                plot_widget.enableAutoRange(axis='y', enable=False)
                

        logger.debug("Auto-ranged all plots (X and Y axes) after signal redraw, settings apply, and limit lines")
        logger.info(f"[PERF] ========== _redraw_all_signals() TOTAL: {(time.time()-start_time)*1000:.1f}ms ==========")
        
        # Update statistics panel after redrawing signals
        self._recreate_statistics_panel()
        # DON'T update statistics here - wait for cursor movement for performance
        
        # Update correlations panel with new parameters
        if hasattr(self, 'correlations_panel_manager') and self.correlations_panel_manager:
            self.correlations_panel_manager.update_available_parameters(all_signal_names)
            self.correlations_panel_manager.on_data_changed()
        
        # Reapply active filters if any exist
        if self.filter_manager.has_active_filters():
            # Use a timer to ensure plots are fully ready before reapplying filters
            # Debounce filter reapplication to prevent multiple rapid calls
            if hasattr(self, '_filter_reapply_timer'):
                self._filter_reapply_timer.stop()
            
            self._filter_reapply_timer = QTimer()
            self._filter_reapply_timer.setSingleShot(True)
            self._filter_reapply_timer.timeout.connect(self._reapply_active_filters)
            self._filter_reapply_timer.start(200)  # 200ms debounce
        
        logger.debug("Redrew all signals across all tabs.")
    
    def _apply_limit_lines_to_all_graphs(self):
        """Apply limit lines to all graphs based on saved settings."""
        try:
            active_tab_index = self.tab_widget.currentIndex()
            if active_tab_index < 0 or active_tab_index >= len(self.graph_containers):
                return
                
            container = self.graph_containers[active_tab_index]
            plot_widgets = container.plot_manager.get_plot_widgets()
            
            for graph_index, plot_widget in enumerate(plot_widgets):
                # Get saved limit settings for this graph
                limits_settings = self._get_graph_setting(graph_index, 'limits', {})
                
                if limits_settings and self.graph_renderer:
                    # Get visible signals for this graph
                    visible_signals = self.graph_signal_mapping.get(active_tab_index, {}).get(graph_index, [])
                    
                    # Apply limit lines
                    self.graph_renderer._apply_limit_lines(plot_widget, graph_index, visible_signals)
                    logger.debug(f"Applied limit lines to graph {graph_index} with {len(limits_settings)} limit configs")
                    
        except Exception as e:
            logger.error(f"Error applying limit lines to all graphs: {e}")

    def _reapply_active_filters(self):
        """Reapply active filters after signal redraw."""
        try:
            active_filters = self.filter_manager.get_active_filters()
            total_filters = sum(len(graphs) for graphs in active_filters.values())
            
            for tab_index, graph_filters in active_filters.items():
                if tab_index < len(self.graph_containers):
                    # Reapply each graph's filter independently
                    for graph_index, filter_data in graph_filters.items():
                        self._apply_range_filter(filter_data)
                else:
                    self.filter_manager.remove_filter(tab_index)
        except Exception as e:
            logger.error(f"Error reapplying filters: {e}")

    def clear_active_filters(self):
        """Clear all active filters and redraw signals using filter manager."""
        self.filter_manager.clear_filters()
        self._redraw_all_signals()

    def _restore_cursors_after_redraw(self, cursor_mode: str, saved_positions: dict):
        """Restore cursors after signal redraw operation."""
        try:
            logger.debug(f"Restoring cursors after redraw: mode={cursor_mode}, positions={saved_positions}")
            
            # NOTE: _suppress_statistics_updates is already True (set at start of _redraw_all_signals)
            # This ensures no expensive calculations happen during cursor restoration
            
            # Reinitialize cursor manager to ensure it's working with current plot widgets
            self._initialize_cursor_manager()
            
            if self.cursor_manager:
                # Set the cursor mode (always 'dual')
                self.cursor_manager.set_mode(cursor_mode)
                
                # If we have saved positions, try to restore them
                if saved_positions:
                    # Give cursors a moment to be created, then restore positions
                    QTimer.singleShot(100, lambda: self._restore_cursor_positions(saved_positions))
                
                logger.debug(f"Successfully restored cursors with mode: {cursor_mode}")
            
            # Re-enable statistics updates after cursor restore completes (300ms to ensure positions are set)
            # This allows first user-initiated cursor movement to trigger statistics calculation
            QTimer.singleShot(300, self._enable_statistics_updates_after_redraw)
            
        except Exception as e:
            logger.error(f"Failed to restore cursors after redraw: {e}")
            self._suppress_statistics_updates = False
            import traceback
            traceback.print_exc()
    
    def _enable_statistics_updates_after_redraw(self):
        """Re-enable statistics updates after redraw operation completes."""
        self._suppress_statistics_updates = False
        logger.debug("[PERF] Statistics updates re-enabled after redraw")

    def _restore_cursor_positions(self, positions: dict):
        """Restore specific cursor positions."""
        try:
            if not self.cursor_manager or not positions:
                return
                
            # Restore cursor positions if available
            if 'cursor1' in positions and hasattr(self.cursor_manager, 'set_cursor_position'):
                self.cursor_manager.set_cursor_position("dual_1", positions['cursor1'])
                
            if 'cursor2' in positions and hasattr(self.cursor_manager, 'set_cursor_position'):
                self.cursor_manager.set_cursor_position("dual_2", positions['cursor2'])
                
            logger.debug(f"Restored cursor positions: {positions}")
            
        except Exception as e:
            logger.warning(f"Could not restore cursor positions: {e}")

    def update_data(self, df, time_column: Optional[str] = None):
        """Main entry point to update the widget with new data."""
        # Check if data source is MpaiReader
        is_mpai = hasattr(df, 'get_header')
        
        if not is_mpai and (df is None or df.height == 0):
            logger.warning("Update_data called with empty or None DataFrame.")
            return

        self.loading_manager.start_operation("processing", "Processing data...")
        self.data_manager.set_data(df, time_column=time_column)

        # --- Threaded Signal Processing ---
        self.processing_thread = QThread()
        self.processing_worker = SignalProcessingWorker(df, self.is_normalized, time_column)
        self.processing_worker.moveToThread(self.processing_thread)

        # Connect signals
        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.finished.connect(self._on_processing_finished)
        self.processing_worker.error.connect(self._on_processing_error)

        # Cleanup
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.finished.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)

        # Start the thread
        self.processing_thread.start()
        
    def _on_processing_finished(self, all_signals: Dict):
        """Handles the signals after they have been processed in a background thread."""
        self.loading_manager.finish_operation("processing")
        
        # CRITICAL FIX: Clear ALL data including original backups before loading new data!
        # This prevents old filtered data from persisting when new data is loaded
        logger.info("[DATA LOAD] Clearing all signal processor data before loading new data")
        self.signal_processor.clear_all_data()
        
        # ✅ CRITICAL FIX: Restore raw_dataframe (MpaiReader) to the main signal_processor
        # The worker used a temporary processor, so the main one doesn't have the MpaiReader reference yet.
        # This is REQUIRED for get_signal_at_time() to load data outside the preview range.
        if hasattr(self, 'processing_worker') and hasattr(self.processing_worker, 'df'):
            df = self.processing_worker.df
            # Check if it is an MpaiReader (has get_header method)
            if hasattr(df, 'get_header'):
                self.signal_processor.raw_dataframe = df
                logger.info("✅ Restored MpaiReader to main SignalProcessor for full data access")
            else:
                # For normal DataFrames, we might also want to store it if lazy loading is used later
                # But currently SignalProcessor.process_data handles caching for DataFrames.
                # For consistency, we can set it here too if needed, but priority is MPAI.
                pass
        
        # Clear all active filters since we're loading new data
        logger.info("[DATA LOAD] Clearing all active filters")
        self.filter_manager.clear_filters()
        
        # CRITICAL FIX: Clear all plots from all graph containers
        # This prevents old signal plots from persisting when switching files
        logger.info("[DATA LOAD] Clearing all plots from all graph containers")
        for container in self.graph_containers:
            if container and hasattr(container, 'plot_manager'):
                try:
                    container.plot_manager.clear_all_signals()
                    logger.debug(f"Cleared plots from container")
                except Exception as e:
                    logger.warning(f"Failed to clear plots from container: {e}")
        
        # CRITICAL FIX: For MPAI files, we must re-run process_data() on the main signal_processor
        # to properly register signals with mpai_reader references. The worker used a separate processor
        # whose signal_data is not accessible from the main thread.
        if hasattr(self, 'processing_worker') and hasattr(self.processing_worker, 'df'):
            df = self.processing_worker.df
            if hasattr(df, 'get_header'):
                # MPAI file - re-run process_data on main processor to register signals properly
                logger.info(f"[DATA LOAD] Re-running process_data on main SignalProcessor for MPAI file")
                self.signal_processor.process_data(df, normalize=False, time_column=self.processing_worker.time_column)
                logger.info(f"[DATA LOAD] MPAI signals registered: {len(self.signal_processor.signal_data)} signals")
            else:
                # Non-MPAI file - use add_signal for each signal
                logger.info(f"[DATA LOAD] Adding {len(all_signals)} CSV signals")
                for signal_name, signal_data in all_signals.items():
                    self.signal_processor.add_signal(
                        signal_name,
                        signal_data['x_data'],
                        signal_data['y_data'],
                        signal_data.get('metadata', {})
                    )
        else:
            # Fallback for non-worker processing
            logger.info(f"[DATA LOAD] Adding {len(all_signals)} signals (no worker)")
            for signal_name, signal_data in all_signals.items():
                self.signal_processor.add_signal(
                    signal_name,
                    signal_data['x_data'],
                    signal_data['y_data'],
                    signal_data.get('metadata', {})
                )
        logger.info("[DATA LOAD] Signal processing complete")
        
        self._initialize_signal_mapping(list(all_signals.keys()))
        
        # DON'T auto-draw signals on startup - let user choose what to plot
        # self._redraw_all_signals()  # Commented out to prevent auto-plotting
        
        # Update parameters panel with loaded columns
        if hasattr(self, 'parameters_panel_manager'):
            time_col = self.data_manager.time_column if hasattr(self.data_manager, 'time_column') else None
            self.parameters_panel_manager.update_columns(list(all_signals.keys()), time_col)
            logger.info(f"Parameters panel updated with {len(all_signals)} columns")
        
        self._initialize_cursor_manager()
        self._recreate_statistics_panel()

        if self.get_active_graph_container():
            count = self.get_active_graph_container().plot_manager.get_subplot_count()
            self.graph_settings_panel_manager.rebuild_controls(count)
            self.bitmask_panel_manager.update_graph_sections(count)
        
        logger.info("Signal processing finished and UI updated - graphs start empty for manual signal selection.")

    def _on_processing_error(self, error_msg: str):
        """Handles errors from the processing thread."""
        self.loading_manager.finish_operation("processing")
        QMessageBox.critical(self, "Processing Error", f"Failed to process data:\n\n{error_msg}")
        logger.error(f"Signal processing thread error: {error_msg}")

    def _on_column_dropped_on_graph(self, graph_index: int, column_name: str):
        """
        Handle when a column is dropped onto a graph.
        Plot that column on the specified graph.
        """
        logger.info(f"Column '{column_name}' dropped on graph {graph_index}")
        
        # Get current tab index
        tab_index = self.tab_widget.currentIndex()
        if tab_index < 0:
            logger.warning("No active tab")
            return
        
        # Check if signal exists in signal processor
        if column_name not in self.signal_processor.signal_data:
            logger.warning(f"Column '{column_name}' not found in signal processor")
            return
        
        # Add signal to graph signal mapping
        if tab_index not in self.graph_signal_mapping:
            self.graph_signal_mapping[tab_index] = {}
        
        if graph_index not in self.graph_signal_mapping[tab_index]:
            self.graph_signal_mapping[tab_index][graph_index] = []
        
        # Add signal if not already mapped to this graph
        if column_name not in self.graph_signal_mapping[tab_index][graph_index]:
            self.graph_signal_mapping[tab_index][graph_index].append(column_name)
            logger.info(f"Added '{column_name}' to graph {graph_index} in tab {tab_index}")
            
            # Redraw the graph with the new signal
            self._redraw_all_signals()
            
            # Update statistics panel (but don't calculate stats yet - wait for cursor movement)
            # This prevents freezing when plotting large MPAI files
            self._recreate_statistics_panel()
        else:
            logger.info(f"Column '{column_name}' already plotted on graph {graph_index}")
    
    def _on_graph_settings_requested(self, graph_index: int):
        """Open the advanced graph settings dialog for comprehensive configuration."""
        # CRITICAL: Capture the tab index when dialog is opened
        # This ensures filter/limits are applied to the CORRECT tab
        target_tab_index = self.tab_widget.currentIndex()
        if target_tab_index < 0:
            return

        logger.debug(f"Advanced settings requested for graph {graph_index} in tab {target_tab_index}")
        
        # Store target_tab_index for later use
        self._dialog_target_tab = target_tab_index
        self._dialog_target_graph = graph_index
        
        # Debug signal processor access
        logger.debug(f"Signal processor: {self.signal_processor}")
        logger.debug(f"Signal processor type: {type(self.signal_processor)}")
        
        all_signals_data = self.signal_processor.get_all_signals()
        logger.debug(f"All signals data: {all_signals_data}")
        logger.debug(f"All signals data type: {type(all_signals_data)}")
        logger.debug(f"All signals count: {len(all_signals_data) if all_signals_data else 0}")
        
        all_signals = list(all_signals_data.keys()) if all_signals_data else []
        logger.debug(f"All signals keys: {all_signals}")
        
        # Get signals currently visible in the specific graph of the target tab
        visible_signals = self.graph_signal_mapping.get(target_tab_index, {}).get(graph_index, [])
        
        # Get saved filter data for this graph if available (from TARGET tab)
        saved_filter_data = None
        if hasattr(self, 'filter_manager') and self.filter_manager:
            active_filters = self.filter_manager.get_active_filters()
            tab_filters = active_filters.get(target_tab_index, {})
            # Get filter for specific graph in this tab
            saved_filter_data = tab_filters.get(graph_index, None)
            logger.debug(f"Retrieved saved filter data for tab {target_tab_index}, graph {graph_index}: {saved_filter_data}")
        
        # Get saved limits data for this graph if available
        saved_limits_data = self._get_graph_setting(graph_index, 'limits', {})
        logger.debug(f"Retrieved saved limits data for graph {graph_index}: {saved_limits_data}")
        
        # Get saved basic deviation data for this graph if available
        saved_basic_deviation_data = self._get_graph_setting(graph_index, 'basic_deviation', {})
        logger.debug(f"Retrieved saved basic deviation data for graph {graph_index}: {saved_basic_deviation_data}")
        
        # Use the new advanced settings dialog - parent=None for taskbar visibility
        dialog = GraphAdvancedSettingsDialog(graph_index, all_signals, visible_signals, 
                                           saved_filter_data, saved_limits_data, 
                                           saved_basic_deviation_data, None)
        
        # Set proper window icon and title for taskbar
        dialog.setWindowIcon(self.windowIcon() if self.windowIcon() else QIcon())
        
        # Center dialog on parent window
        if self.parent():
            parent_geometry = self.parent().geometry()
            dialog.move(
                parent_geometry.center().x() - dialog.width() // 2,
                parent_geometry.center().y() - dialog.height() // 2
            )
        
        # Connect signals to immediately apply when Apply button is clicked
        # ✅ FIX: Add tab_index to filter_data before applying
        def on_range_filter_applied(filter_data):
            # Ensure tab_index is in filter_data
            if 'tab_index' not in filter_data:
                filter_data['tab_index'] = target_tab_index
                logger.info(f"[DIALOG] Added tab_index={target_tab_index} to filter_data")
            self._apply_range_filter(filter_data)
        
        dialog.range_filter_applied.connect(on_range_filter_applied)
        dialog.basic_deviation_applied.connect(self._on_basic_deviation_applied)
        dialog.limits_applied.connect(self._on_limits_applied_from_dialog)
        
        if dialog.exec_() == QDialog.Accepted:
            logger.info(f"[DIALOG] Dialog accepted for graph {graph_index}")
            
            # Update signal selections (parameters panel)
            # CRITICAL: Use target_tab_index (dialog açıldığı tab), not current tab!
            selected_signals = dialog.get_selected_signals()
            
            # Ensure the mapping for the TARGET tab exists
            if target_tab_index not in self.graph_signal_mapping:
                self.graph_signal_mapping[target_tab_index] = {}
                
            self.graph_signal_mapping[target_tab_index][graph_index] = selected_signals
            logger.debug(f"[DIALOG] Updated signals for Tab {target_tab_index}, Graph {graph_index}: {selected_signals}")
            
            # Redraw all signals to show updated parameter selection
            self._redraw_all_signals()
            
            # NOTE: Filter, Limits, ve Deviation ayarları zaten dialog.accept() içinde
            # _apply_settings() tarafından uygulandı. Tekrar apply etmeye gerek yok!
            
    def _on_basic_deviation_applied(self, graph_index: int, deviation_settings: Dict[str, Any]):
        """Handle basic deviation settings application."""
        # CRITICAL: Use target tab from dialog, NOT current active tab!
        target_tab_index = getattr(self, '_dialog_target_tab', self.tab_widget.currentIndex())
        logger.info(f"[DEVIATION] Applying basic deviation settings to graph {graph_index} on tab {target_tab_index}")
        logger.debug(f"[DEVIATION] Settings: {deviation_settings}")

        try:
            # Save settings for persistence
            self._save_graph_setting(graph_index, 'basic_deviation', deviation_settings)
            logger.info(f"[DEVIATION] Saved deviation settings for graph {graph_index}")
            
            # Apply deviation settings to graph renderer
            if hasattr(self, 'graph_renderer') and self.graph_renderer:
                self.graph_renderer.set_basic_deviation_settings(target_tab_index, graph_index, deviation_settings)
                logger.info(f"[DEVIATION] Set basic deviation settings in renderer for graph {graph_index}")
                
                # Force immediate redraw to show deviation lines
                self._redraw_all_signals()
                logger.info(f"[DEVIATION] Triggered redraw to show deviation visualization")
            else:
                logger.warning("[DEVIATION] Graph renderer not available for basic deviation application")

        except Exception as e:
            logger.error(f"[DEVIATION] Error applying basic deviation settings to graph {graph_index}: {e}", exc_info=True)

    def _on_plot_clicked(self, plot_index: int, x: float, y: float):
        """Handle plot clicks."""
        self.current_cursor_position = x
        self._update_legend_values()
        self.cursor_moved.emit("click", (x, y))
    
    def _on_range_selected(self, start: float, end: float):
        """Handle range selection."""
        self.selected_range = (start, end)
        self._update_statistics_for_range(start, end)
        self.range_selected.emit((start, end))
    
    def _on_signal_visibility_changed(self, signal_name: str, visible: bool):
        """Handle signal visibility changes."""
        pass  # Implementation depends on requirements
    
    def _on_signal_selected(self, signal_name: str):
        """Handle signal selection."""
        pass
    
    def _on_processing_started(self):
        """Handle processing start."""
        pass
    
        
    def _on_statistics_updated(self, stats: dict):
        """Handle statistics updates."""
        self.statistics_updated.emit(stats)
    
    def _on_theme_changed(self, theme_name: str):
        """Handle theme changes broadcast from the theme manager."""
        self._apply_theme()
    
    # NOTE: _on_cursor_mode_changed removed - cursor mode is now permanently 'dual'
    # Cursor mode is initialized to 'dual' on startup and never changes
    
    def _on_panel_toggled(self):
        """Handle statistics panel visibility toggle."""
        if hasattr(self, 'channel_stats_panel'):
            self.channel_stats_panel.setVisible(not self.channel_stats_panel.isVisible())
    
    def _on_settings_toggled(self):
        """Handle settings panel visibility toggle."""
        self._toggle_left_panel(self.settings_panel)
        #logger.info(f"Settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.settings_panel}")

    def _on_graph_settings_toggled(self):
        """Handle graph settings panel visibility toggle."""
        self._toggle_left_panel(self.graph_settings_panel)
        #logger.info(f"Graph settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.graph_settings_panel}")

    def _on_parameters_toggled(self):
        """Handle parameters panel visibility toggle."""
        self._toggle_left_panel(self.parameters_panel)

    def _on_statistics_settings_toggled(self):
        """Handle statistics settings panel visibility toggle."""
        self._toggle_left_panel(self.statistics_settings_panel)
            
        #logger.info(f"Statistics settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.statistics_settings_panel}")
    
    def _on_correlations_toggled(self):
        """Handle correlations panel visibility toggle."""
        self._toggle_left_panel(self.correlations_panel)
        
    def _on_bitmask_toggled(self):
        """Handle bitmask panel visibility toggle."""
        self._toggle_left_panel(self.bitmask_panel)

    def _toggle_left_panel(self, panel_to_show):
        """Generic function to toggle visibility of a panel in the left stack."""
        if self.left_panel_stack.currentWidget() == panel_to_show and self.left_panel_stack.isVisible():
            self.left_panel_stack.setVisible(False)
        else:
            self.left_panel_stack.setCurrentWidget(panel_to_show)
            self.left_panel_stack.setVisible(True)

    def _on_per_graph_normalization_toggled(self, graph_index: int, normalize: bool):
        """Handle normalization for a specific graph."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return

        signals_in_graph = self.graph_signal_mapping.get(active_tab_index, {}).get(graph_index, [])
        
        if not signals_in_graph:
            return

        if normalize:
            self.signal_processor.apply_normalization(signal_names=signals_in_graph)
        else:
            self.signal_processor.remove_normalization(signal_names=signals_in_graph)
        
        # Save normalization setting for this graph
        self._save_graph_setting(graph_index, 'normalize', normalize)
        
        # Redraw all signals to reflect the change
        self._redraw_all_signals()
        logger.info(f"Normalization toggled to {normalize} for signals in graph {graph_index}: {signals_in_graph}")

    def _on_per_graph_view_reset(self, graph_index: int):
        """Handle view reset for a specific graph to show all data including limit lines."""
        active_container = self.get_active_graph_container()
        if active_container:
            # Use PlotManager's reset_view() which properly handles downsampled data
            # by using stored original data ranges instead of autoRange()
            active_container.plot_manager.reset_view()
            logger.info(f"View reset for graph {graph_index} - using PlotManager.reset_view() with original data ranges")

    def _on_per_graph_grid_changed(self, graph_index: int, show_grid: bool):
        """Handle grid visibility for a specific graph."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            if 0 <= graph_index < len(plot_widgets):
                plot_widgets[graph_index].showGrid(x=show_grid, y=show_grid)
                
                # Save grid setting for this graph
                self._save_graph_setting(graph_index, 'show_grid', show_grid)
                
                logger.info(f"Grid visibility for graph {graph_index} set to {show_grid}")

    def _on_per_graph_autoscale_changed(self, graph_index: int, autoscale: bool):
        """Handle Y-axis autoscale for a specific graph."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            if 0 <= graph_index < len(plot_widgets):
                plot_widgets[graph_index].enableAutoRange(axis='y', enable=autoscale)
                
                # Save autoscale setting for this graph
                self._save_graph_setting(graph_index, 'autoscale', autoscale)
                
                logger.info(f"Autoscale for graph {graph_index} set to {autoscale}")

    def _on_global_normalization_toggled(self, normalize: bool):
        """Handle global normalization toggle for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            for graph_index in range(len(plot_widgets)):
                self._on_per_graph_normalization_toggled(graph_index, normalize)
        
        # Sync with right-click menu settings
        self.graph_settings_panel_manager.sync_global_settings_from_right_click({'normalize': normalize})
        logger.info(f"Global normalization {'enabled' if normalize else 'disabled'} for all graphs")

    def _on_global_view_reset(self):
        """Handle global view reset for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            for graph_index in range(len(plot_widgets)):
                self._on_per_graph_view_reset(graph_index)
        logger.info("Global view reset applied to all graphs")

    def _on_global_grid_changed(self, show_grid: bool):
        """Handle global grid visibility for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            # Update plot manager's global settings
            if hasattr(active_container, 'plot_manager') and hasattr(active_container.plot_manager, 'update_global_settings'):
                active_container.plot_manager.update_global_settings()
            
            # Also update individual graph settings for consistency
            plot_widgets = active_container.get_plot_widgets()
            for graph_index in range(len(plot_widgets)):
                self._on_per_graph_grid_changed(graph_index, show_grid)
        
        # Sync with right-click menu settings
        self.graph_settings_panel_manager.sync_global_settings_from_right_click({'show_grid': show_grid})
        logger.info(f"Global grid {'shown' if show_grid else 'hidden'} for all graphs")

    def _on_global_autoscale_changed(self, autoscale: bool):
        """Handle global Y-axis autoscale for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            for graph_index in range(len(plot_widgets)):
                self._on_per_graph_autoscale_changed(graph_index, autoscale)
        
        # Sync with right-click menu settings
        self.graph_settings_panel_manager.sync_global_settings_from_right_click({'autoscale': autoscale})
        logger.info(f"Global autoscale {'enabled' if autoscale else 'disabled'} for all graphs")

    def _on_global_legend_visibility_changed(self, visible: bool):
        """Handle global legend visibility for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container and hasattr(active_container, 'plot_manager'):
            active_container.plot_manager.set_legend_visibility(visible)
        
        # Sync with right-click menu settings
        self.graph_settings_panel_manager.sync_global_settings_from_right_click({'show_legend': visible})
        logger.info(f"Global legend visibility set to {visible} for all graphs")

    def _on_global_tooltips_changed(self, enabled: bool):
        """Handle global tooltips toggle for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container and hasattr(active_container, 'plot_manager'):
            active_container.plot_manager.set_tooltips_enabled(enabled)
        
        # Sync with right-click menu settings
        self.graph_settings_panel_manager.sync_global_settings_from_right_click({'show_tooltips': enabled})
        logger.info(f"Global tooltips {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_snap_changed(self, enabled: bool):
        """Handle global snap to data points for all graphs."""
        logger.info(f"Global snap to data {'enabled' if enabled else 'disabled'} for all graphs")
        
        # Update cursor manager with snap setting
        if hasattr(self, 'cursor_manager') and self.cursor_manager:
            self.cursor_manager.set_snap_to_data(enabled)
        
        # Update plot manager with snap setting
        active_container = self.get_active_graph_container()
        if active_container and hasattr(active_container, 'plot_manager'):
            active_container.plot_manager.set_snap_to_data(enabled)

    def _on_global_line_width_changed(self, width: int):
        """Handle global line width change for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            # Use PlotManager's method to set line width
            plot_manager = active_container.plot_manager
            if plot_manager:
                plot_manager.set_line_width(width)
        logger.info(f"Global line width set to {width} for all graphs")

    def _on_global_x_mouse_changed(self, enabled: bool):
        """Handle global X axis mouse interaction for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            for plot_widget in plot_widgets:
                plot_widget.setMouseEnabled(x=enabled, y=plot_widget.getViewBox().state['mouseEnabled'][1])
        logger.info(f"Global X axis mouse {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_y_mouse_changed(self, enabled: bool):
        """Handle global Y axis mouse interaction for all graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            for plot_widget in plot_widgets:
                plot_widget.setMouseEnabled(x=plot_widget.getViewBox().state['mouseEnabled'][0], y=enabled)
        logger.info(f"Global Y axis mouse {'enabled' if enabled else 'disabled'} for all graphs")
    
    def _on_global_secondary_axis_changed(self, enabled: bool):
        """Handle global secondary axis toggle."""
        active_container = self.get_active_graph_container()
        if active_container:
            active_container.plot_manager.set_secondary_axis_enabled(enabled)
            # Redraw all signals to apply axis assignment
            self._redraw_all_signals()
        logger.info(f"Global secondary axis {'enabled' if enabled else 'disabled'}")

    def _add_tab(self, name: Optional[str] = None):
        """Add a new tab with a GraphContainer."""
        if not name:
            name = f"Tab {self.tab_widget.count() + 1}"
        
        # Get the tab index (will be the current count before adding)
        tab_index = self.tab_widget.count()
        
        # Create a new graph container for the tab, passing self as the main_widget and tab_index
        graph_container = GraphContainer(self.theme_manager, main_widget=self, tab_index=tab_index)
        graph_container.signal_processor = self.signal_processor  # Assign signal processor
        
        # Connect the settings button signal from the new container's plot manager
        graph_container.plot_manager.settings_requested.connect(self._on_graph_settings_requested)
        
        # Connect drag-drop signal to plot column on graph
        graph_container.column_dropped.connect(self._on_column_dropped_on_graph)
        
        self.graph_containers.append(graph_container)
        
        # Add the container as a new tab
        self.tab_widget.addTab(graph_container, name)
        self.tab_widget.setCurrentWidget(graph_container)
        
        # Update tab count in toolbar
        self.toolbar_manager.set_graph_count(self.tab_widget.count())
        
        # Apply the current theme to the new container's plots
        graph_container.apply_theme()
        
        return graph_container

    def _rename_tab(self, index):
        """Rename a tab when it is double-clicked."""
        from PyQt5.QtWidgets import QInputDialog, QLineEdit

        old_name = self.tab_widget.tabText(index)
        
        new_name, ok = QInputDialog.getText(
            self, 
            "Sekmeyi Yeniden Adlandır", 
            "Yeni sekme adı:", 
            QLineEdit.Normal, 
            old_name
        )
        
        if ok and new_name and new_name.strip():
            self.tab_widget.setTabText(index, new_name.strip())
            logger.info(f"Tab {index} renamed from '{old_name}' to '{new_name.strip()}'")

    def _remove_tab(self, index: int):
        """Remove a tab at a given index."""
        if self.tab_widget.count() > 1:
            widget_to_remove = self.tab_widget.widget(index)
            self.tab_widget.removeTab(index)
            
            if widget_to_remove in self.graph_containers:
                self.graph_containers.remove(widget_to_remove)
            
            # Clean up the removed widget
            widget_to_remove.deleteLater()

            # Update remaining tab titles
            for i in range(self.tab_widget.count()):
                self.tab_widget.setTabText(i, f"Tab {i + 1}")

    def get_active_graph_container(self) -> Optional['GraphContainer']:
        """Gets the GraphContainer from the currently active tab."""
        if not hasattr(self, 'tab_widget') or not self.tab_widget:
            return None
        current_index = self.tab_widget.currentIndex()
        if 0 <= current_index < len(self.graph_containers):
            return self.graph_containers[current_index]
        return None

    def _on_cursor_moved_old(self, cursor_type: str, position: float):
        """Handle cursor movement events (old signature - deprecated)."""
        self.current_cursor_position = position
        self._update_legend_values()
        self.cursor_moved.emit(cursor_type, position)

    def _on_range_changed(self, start: float, end: float):
        """Handle range changes from cursor manager."""
        self.selected_range = (start, end)
        self._update_statistics_for_range(start, end)
    
    # Data processing methods
    def _apply_normalization(self):
        """Apply normalization to all signals."""
        signal_names = list(self.signal_processor.get_all_signals().keys())
        normalized_data = self.signal_processor.apply_normalization(signal_names)
        
        for signal_name, y_data in normalized_data.items():
            signal_data = self.signal_processor.get_signal_data(signal_name)
            if signal_data and 'x_data' in signal_data:
                self.plot_manager.update_signal_data(signal_name, signal_data['x_data'], y_data, 0)
        
        self._update_legend_values()
    
    def _remove_normalization(self):
        """Remove normalization from all signals."""
        signal_names = list(self.signal_processor.get_all_signals().keys())
        original_data = self.signal_processor.remove_normalization(signal_names)
        
        for signal_name, y_data in original_data.items():
            signal_data = self.signal_processor.get_signal_data(signal_name)
            if signal_data and 'x_data' in signal_data:
                self.plot_manager.update_signal_data(signal_name, signal_data['x_data'], y_data, 0)
        
        self._update_legend_values()

    def _update_statistics(self, cursor_pos=None, selected_range=None):
        """Updates all statistics based on the active tab and cursors."""
        # CRITICAL: Skip statistics updates during cursor restore to prevent slowdown
        if getattr(self, '_suppress_statistics_updates', False):
            return
        
        active_container = self.get_active_graph_container()
        tab_index = self.tab_widget.currentIndex()

        if not active_container or tab_index < 0:
            return

        tab_mapping = self.graph_signal_mapping.get(tab_index, {})
        
        # Get cursor positions if available
        cursor_positions = {}
        if self.cursor_manager and self.current_cursor_mode == 'dual':
            cursor_positions = self.cursor_manager.get_cursor_positions()
        
        # BATCH PRE-FETCH: Get all cursor values at once
        batch_cursor_values = {} # cursor_key -> {signal_name: value}
        if cursor_positions:
            all_signals = []
            for sigs in tab_mapping.values():
                all_signals.extend(sigs)
            
            # Remove duplicates
            all_signals = list(set(all_signals))
            
            if all_signals:
                # Fetch for each cursor
                for c_key, c_pos in cursor_positions.items():
                    try:
                        batch_cursor_values[c_key] = self.signal_processor.get_signals_at_time(all_signals, c_pos)
                    except Exception as e:
                        batch_cursor_values[c_key] = {}

        # ✅ PERFORMANCE FIX: Calculate cursor range ONCE (not per-signal)
        stats_range = selected_range
        if cursor_positions and 'c1' in cursor_positions and 'c2' in cursor_positions:
            c1_pos = cursor_positions['c1']
            c2_pos = cursor_positions['c2']
            stats_range = (min(c1_pos, c2_pos), max(c1_pos, c2_pos))
        
        # ✅ PERFORMANCE FIX: Collect ALL signals and calculate stats in ONE batch call
        all_signals_list = []
        for signal_names in tab_mapping.values():
            all_signals_list.extend(signal_names)
        all_signals_list = list(set(all_signals_list))  # Remove duplicates
        
        # Single batch call for all statistics (O(1) instead of O(n))
        all_stats = {}
        if all_signals_list:
            all_stats = self.signal_processor.calculate_statistics(
                all_signals_list,
                stats_range, 
                self.duty_cycle_threshold_mode, 
                self.duty_cycle_threshold_value
            )
        
        # Update stats for each signal in the modern statistics panel (uses cached results)
        for graph_index, signal_names in tab_mapping.items():
            for signal_name in signal_names:
                stats = all_stats.get(signal_name, {})
                
                if stats:
                    # Add cursor values to stats (from BATCH cache)
                    if cursor_positions:
                        cursor_vals = {}
                        for c_key in cursor_positions:
                            # Use pre-fetched value if available, else 0.0
                            val = batch_cursor_values.get(c_key, {}).get(signal_name, 0.0)
                            cursor_vals[c_key] = val
                            
                        stats.update(cursor_vals)
                        
                        # ✅ PERFORMANCE FIX: Use RMS from stats (already calculated)
                        # Removed expensive per-signal _calculate_rms_to_cursor() call
                    
                    # Update the modern statistics panel
                    full_signal_name = f"{signal_name} (G{graph_index+1})"
                    self.statistics_panel.update_statistics(full_signal_name, stats)

    def _get_cursor_values_for_signal(self, signal_name: str, cursor_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Get signal values at cursor positions with improved interpolation.
        
        ✅ FIX: Now uses signal_processor.get_signal_at_time() which handles MPAI files
        correctly by loading data from file when cursor is outside preview range.
        """
        cursor_values = {}
        
        # Use signal_processor's get_signal_at_time() method which handles MPAI correctly
        for cursor_key, x_pos in cursor_positions.items():
            try:
                # ✅ FIX: Use get_signal_at_time() instead of get_signal_data()
                # This method handles MPAI files by loading data from file when needed
                y_value = self.signal_processor.get_signal_at_time(signal_name, x_pos)
                
                if y_value is not None:
                    cursor_values[cursor_key] = float(y_value)
                    logger.debug(f"[CURSOR VALUE] {cursor_key} at {x_pos:.3f} -> {signal_name} = {y_value:.6f}")
                else:
                    logger.warning(f"[CURSOR VALUE] No value found for {signal_name} at cursor {cursor_key} position {x_pos} - returning 0.0")
                    cursor_values[cursor_key] = 0.0
                
            except Exception as e:
                logger.error(f"[CURSOR VALUE] Failed to get cursor value for {signal_name} at {cursor_key}: {e}", exc_info=True)
                cursor_values[cursor_key] = 0.0
        
        logger.debug(f"[CURSOR VALUE] Final cursor_values for {signal_name}: {cursor_values}")
        return cursor_values

    def _calculate_rms_to_cursor(self, signal_name: str, cursor_x: float) -> Optional[float]:
        """Calculate RMS from signal start to cursor position."""
        signal_data = self.signal_processor.get_signal_data(signal_name)
        if not signal_data:
            return None
            
        x_data = signal_data.get('x_data', [])
        y_data = signal_data.get('y_data', [])
        
        if len(x_data) == 0 or len(y_data) == 0:
            return None
            
        import numpy as np
        
        try:
            # Find indices where x <= cursor_x
            mask = np.array(x_data) <= cursor_x
            if not np.any(mask):
                return None
                
            # Get y values up to cursor position
            y_subset = np.array(y_data)[mask]
            
            if len(y_subset) == 0:
                return None
                
            # Calculate RMS
            rms_value = np.sqrt(np.mean(y_subset**2))
            return float(rms_value)
            
        except Exception as e:
            logger.warning(f"Failed to calculate RMS to cursor for {signal_name}: {e}")
            return None

    def _on_cursor_moved(self, cursor_positions: Dict[str, float]):
        """Handle cursor movement with strict throttling."""
        # STRICT THROTTLING: Limit to ~30 FPS (33ms)
        current_time_ms = time.time() * 1000
        if current_time_ms - getattr(self, '_last_cursor_event_time', 0) < 33:
            return
        self._last_cursor_event_time = current_time_ms
        
        # CRITICAL: Skip statistics updates if suppressed (during graph redraw)
        if getattr(self, '_suppress_statistics_updates', False):
            if hasattr(self, 'statistics_panel') and self.statistics_panel:
                self.statistics_panel.update_cursor_positions(cursor_positions)
            return
        
        # Store cursor positions for other components
        if cursor_positions:
            if 'c1' in cursor_positions:
                self.current_cursor_position = cursor_positions['c1']
            elif 'cursor1' in cursor_positions:
                self.current_cursor_position = cursor_positions['cursor1']
        
        # Update cursor UI elements immediately (lightweight)
        if hasattr(self, 'statistics_panel') and self.statistics_panel:
            self.statistics_panel.update_cursor_positions(cursor_positions)
        
        # Update zoom button state in graph settings panel
        if hasattr(self, 'graph_settings_panel_manager'):
            self.graph_settings_panel_manager.update_zoom_button_state()
        
        # ✅ OPTIMIZED: Throttling for real-time statistics updates
        # Check if enough time has passed since last update (throttle at 20 FPS)
        current_time = time.time() * 1000  # Convert to milliseconds
        time_since_last_update = current_time - self._last_statistics_update_time
        
        if time_since_last_update >= 50:  # 50ms = 20 FPS
            # Enough time has passed, update immediately
            self._pending_cursor_positions = cursor_positions
            self._perform_statistics_update()
            self._last_statistics_update_time = current_time
            # Cancel any pending timer
            self._statistics_update_timer.stop()
        else:
            # Too soon, schedule update for later (throttle)
            self._pending_cursor_positions = cursor_positions
            if not self._statistics_update_timer.isActive():
                # Calculate remaining time until next allowed update
                remaining_time = int(50 - time_since_last_update)
                self._statistics_update_timer.setInterval(max(10, remaining_time))
                self._statistics_update_timer.start()
        
        # Emit signal for external listeners
        if cursor_positions:
            # Convert to old format for backward compatibility
            if 'c1' in cursor_positions:
                self.cursor_moved.emit("cursor1", cursor_positions['c1'])
            elif 'cursor1' in cursor_positions:
                self.cursor_moved.emit("cursor1", cursor_positions['cursor1'])
    
    def _perform_statistics_update(self):
        """Perform the actual statistics update."""
        if self._pending_cursor_positions is None:
            return
            
        t0 = time.perf_counter()
        
        # Update timestamp for throttling
        self._last_statistics_update_time = time.time() * 1000
        
        # 1. Update statistics (includes batch fetch for Stats Panel)
        self._update_statistics()
        t1 = time.perf_counter()
        
        # 2. Update legend values (includes batch fetch for Legend)
        self._update_legend_values()
        t2 = time.perf_counter()
        
        # 3. Update panels
        if hasattr(self, 'correlations_panel_manager') and self.correlations_panel_manager:
            self.correlations_panel_manager.on_cursor_moved(self._pending_cursor_positions)
        
        if hasattr(self, 'bitmask_panel_manager') and self.bitmask_panel_manager:
            self.bitmask_panel_manager.on_cursor_position_changed(self._pending_cursor_positions)
            
        t3 = time.perf_counter()
        
        # Profiling Output
        total_ms = (t3 - t0) * 1000
        if total_ms > 10:
            stats_ms = (t1 - t0) * 1000
            legend_ms = (t2 - t1) * 1000
            panels_ms = (t3 - t2) * 1000
            print(f"SLOW FRAME: Total={total_ms:.2f}ms | Stats(inc Batch)={stats_ms:.2f}ms | Legend(inc Batch)={legend_ms:.2f}ms | Panels={panels_ms:.2f}ms")

    def _update_statistics_for_range(self, start: float, end: float):
        """Update statistics for time range."""
        stats = self.signal_processor.calculate_statistics(time_range=(start, end))
        self.statistics_updated.emit(stats)
        
    def _apply_range_filter(self, filter_data: dict):
        """Apply range filter to the specified graph using modular components."""
        
        logger.info("=" * 80)
        logger.info("[FILTER] _apply_range_filter() CALLED!")
        logger.info(f"[FILTER] Received filter_data: {filter_data}")
        
        try:
            # ✅ FIX: Get target tab index from filter_data (stored during save)
            # This ensures filter is applied to correct tab even during reapplication
            target_tab_index = filter_data.get('tab_index', None)
            
            # Fallback: Use dialog target or current tab
            if target_tab_index is None:
                target_tab_index = getattr(self, '_dialog_target_tab', self.tab_widget.currentIndex())
                logger.warning(f"[FILTER] tab_index not in filter_data (old saved filter or legacy code), using fallback: {target_tab_index}")
                logger.warning(f"[FILTER] This may happen with old MPAI project files. Filter will still work correctly.")
            
            logger.info(f"[FILTER] target_tab_index: {target_tab_index} (from filter_data)")
            logger.info(f"[FILTER] graph_containers count: {len(self.graph_containers)}")
            
            if target_tab_index < 0 or target_tab_index >= len(self.graph_containers):
                logger.error(f"[FILTER] Invalid target_tab_index: {target_tab_index}")
                return
            
            logger.info(f"[FILTER] Applying filter to target_tab_index: {target_tab_index} (dialog opened for this tab)")
            
            # ✅ FIX: Extract conditions and mode FIRST before using them
            graph_index = filter_data.get('graph_index', 0)
            conditions = filter_data.get('conditions', [])
            mode = filter_data.get('mode', 'segmented')
            
            # Check if filter can be applied
            can_apply, reason = self.filter_manager.can_apply_filter(mode, target_tab_index)
            
            if not can_apply:
                logger.warning(f"[FILTER MODE] Filter cannot be applied: {reason}")
                # Show warning to user
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Filter Uygulanamıyor",
                    f"<b>Filter uygulanamıyor!</b><br><br>{reason}"
                )
                return
            
            # ✅ FIX Problem #9: Concatenated mode warning
            if mode == 'concatenated' and conditions:  # Only warn if applying filter, not clearing
                from PyQt5.QtWidgets import QMessageBox
                
                # Check if there are existing limits or deviation settings
                has_limits = False
                has_deviation = False
                for i in range(10):  # Check all possible graphs
                    if self._get_graph_setting(i, 'limits', {}):
                        has_limits = True
                    if self._get_graph_setting(i, 'basic_deviation', {}):
                        has_deviation = True
                
                if has_limits or has_deviation:
                    warning_text = """
<b>⚠️ Concatenated Display Mode Aktif!</b>
<br><br>
Range filter uygulandığında:
<ul>
<li>Filtre <b>TÜM sekmelerdeki TÜM grafiklere</b> uygulanacak (global etki)</li>"""
                    
                    if has_limits:
                        warning_text += "<li>Mevcut <b>Static Limits kaldırılacak</b></li>"
                    if has_deviation:
                        warning_text += "<li>Mevcut <b>Basic Deviation ayarları kaldırılacak</b></li>"
                    
                    warning_text += """
</ul>
<br>
Devam etmek istiyor musunuz?
                    """
                    
                    reply = QMessageBox.question(
                        self,
                        "Concatenated Mode Uyarısı",
                        warning_text,
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    
                    if reply == QMessageBox.No:
                        logger.info("[FILTER MODE] User cancelled concatenated filter application")
                        return
                    
                    # User confirmed, clear limits and deviation
                    logger.info("[FILTER MODE] Clearing limits and deviation settings before concatenated filter")
                    for i in range(10):
                        self._save_graph_setting(i, 'limits', {})
                        self._save_graph_setting(i, 'basic_deviation', {})
                
            container = self.graph_containers[target_tab_index]
            
            # VALIDATION: graph_index container'daki grafik sayısından fazla olmamalı
            max_graphs = container.plot_manager.get_subplot_count()
            if graph_index >= max_graphs:
                logger.error(f"Invalid graph_index {graph_index} for container with {max_graphs} graphs. Using 0.")
                graph_index = 0
            
            
            # Get all signals data
            all_signals = self.signal_processor.get_all_signals()
            
            # Check if this is a reset operation (empty conditions)
            if not conditions:
                logger.info("[FILTER CLEAR] Clearing filter (empty conditions received)")
                
                # Önceki filter mode'unu kontrol et (concatenated ise restore gerekli)
                # CRITICAL: Read BEFORE clearing, because incoming filter_data has current mode!
                # CRITICAL FIX: Use target_tab_index, not active_tab_index!
                previous_filter = self.filter_manager.get_filter_state(target_tab_index, graph_index)
                logger.info(f"[FILTER CLEAR] Previous filter state: {previous_filter}")
                
                # ✅ FIX: Check if concatenated mode is CURRENTLY active (not from incoming data)
                # The FilterManager knows the truth about concatenated mode status
                was_concatenated = self.filter_manager.is_concatenated_mode_active
                logger.info(f"[FILTER CLEAR] Was concatenated mode active? {was_concatenated}")
                
                # Clear filter state (from TARGET tab!)
                logger.info(f"[FILTER CLEAR] Removing filter for tab {target_tab_index}, graph {graph_index}")
                self.filter_manager.remove_filter(target_tab_index, graph_index)
                
                # Verify concatenated mode was cleared
                logger.info(f"[FILTER CLEAR] After removal - is_concatenated_mode_active: {self.filter_manager.is_concatenated_mode_active}")
                
                # Filtre durumunu widget container manager'dan da temizle
                if hasattr(self, 'parent') and hasattr(self.parent, 'widget_container_manager'):
                    self.parent.widget_container_manager.save_current_filter_state()
                
                # SADECE concatenated ise orijinal veriyi restore et
                if was_concatenated and self.signal_processor:
                    self.signal_processor.restore_original_data()
                    logger.info("[FILTER CLEAR] Concatenated filter cleared, restoring original data")
                
                # Manuel grafik güncelleme (sonsuz döngü önlemek için)
                all_signals = self.signal_processor.get_all_signals()
                all_signal_names = sorted(list(all_signals.keys()))
                
                # ✅ FIX: If concatenated mode, restore ALL tabs
                if was_concatenated:
                    logger.info("[FILTER CLEAR] Concatenated mode - restoring ALL tabs")
                    # Restore all tabs with original data
                    for tab_idx, tab_container in enumerate(self.graph_containers):
                        logger.info(f"[FILTER CLEAR] Restoring tab {tab_idx}")
                        
                        # Clear all signals in this tab
                        tab_container.plot_manager.clear_all_signals()
                        
                        # Get signal mapping for this tab
                        tab_mapping = self.graph_signal_mapping.get(tab_idx, {})
                        
                        # Redraw all signals in this tab with original data
                        for g_idx, signal_names in tab_mapping.items():
                            if g_idx < tab_container.plot_manager.get_subplot_count():
                                for name in signal_names:
                                    if name in all_signals:
                                        signal_data = all_signals[name]
                                        signal_index = all_signal_names.index(name)
                                        color = self.theme_manager.get_signal_color(signal_index)
                                        tab_container.plot_manager.add_signal(
                                            name, 
                                            signal_data['x_data'], 
                                            signal_data['y_data'], 
                                            plot_index=g_idx, 
                                            pen=color
                                        )
                        
                        # Apply limit lines to this tab
                        plot_widgets = tab_container.plot_manager.get_plot_widgets()
                        for g_idx, plot_widget in enumerate(plot_widgets):
                            visible_signals = self.graph_signal_mapping.get(tab_idx, {}).get(g_idx, [])
                            if self.graph_renderer:
                                self.graph_renderer._apply_limit_lines(plot_widget, g_idx, visible_signals)
                        
                        # Auto-range all plots in this tab
                        for plot_widget in tab_container.plot_manager.get_plot_widgets():
                            plot_widget.enableAutoRange(axis='x', enable=True)
                            plot_widget.enableAutoRange(axis='y', enable=True)
                            plot_widget.autoRange()
                            plot_widget.enableAutoRange(axis='x', enable=False)
                            plot_widget.enableAutoRange(axis='y', enable=False)
                else:
                    # Segmented mode - only update target tab
                    logger.info(f"[FILTER CLEAR] Segmented mode - restoring only target tab {target_tab_index}")
                    
                    # Tüm plot widget'ları TAMAMEN temizle (InfiniteLines vs. için)
                    plot_widgets = container.get_plot_widgets()
                    for plot_widget in plot_widgets:
                        plot_widget.clear()  # Tüm item'ları temizle
                    
                    # Sadece TARGET container'daki sinyalleri yeniden çiz
                    tab_mapping = self.graph_signal_mapping.get(target_tab_index, {})
                    
                    for g_idx, signal_names in tab_mapping.items():
                        if g_idx < container.plot_manager.get_subplot_count():
                            for name in signal_names:
                                if name in all_signals:
                                    signal_data = all_signals[name]
                                    signal_index = all_signal_names.index(name)
                                    color = self.theme_manager.get_signal_color(signal_index)
                                    container.plot_manager.add_signal(
                                        name, 
                                        signal_data['x_data'], 
                                        signal_data['y_data'], 
                                        plot_index=g_idx, 
                                        pen=color
                                    )
                    
                    # Limit lines uygula
                    self._apply_limit_lines_to_all_graphs()
                    
                    # ✅ FIX Problem #8: Auto-range all plots after filter clear
                    logger.debug(f"[VIEW FIX] Auto-ranging all plots after filter clear")
                    plot_widgets = container.get_plot_widgets()
                    for plot_widget in plot_widgets:
                        plot_widget.enableAutoRange(axis='x', enable=True)
                        plot_widget.enableAutoRange(axis='y', enable=True)
                        plot_widget.autoRange()
                        plot_widget.enableAutoRange(axis='x', enable=False)
                        plot_widget.enableAutoRange(axis='y', enable=False)
                
                return
            
            # Use filter manager to calculate segments in background thread
            
            # Show loading indicator
            if hasattr(self, 'loading_manager'):
                self.loading_manager.start_operation("filtering", "Calculating filter segments...")
            
            # Create callback for when calculation is done
            def on_segments_calculated(time_segments):
                try:
                    
                    # Hide loading indicator - check if widget still exists
                    if hasattr(self, 'loading_manager') and self.loading_manager:
                        self.loading_manager.finish_operation("filtering")
                    
                    # ✅ FIX: Handle debounce case (None = debounced, don't show error)
                    if time_segments is None:
                        logger.info("[FILTER] Calculation was debounced, returning silently")
                        return
                    
                    if not time_segments:
                        from PyQt5.QtWidgets import QMessageBox
                        
                        # Check if self is still valid before creating QMessageBox
                        try:
                            msg = QMessageBox(self)
                            msg.setStyleSheet(self._get_message_box_style())
                            msg.setIcon(QMessageBox.Warning)
                            msg.setText("No time segments match the specified filter conditions.")
                            msg.setWindowTitle("No Matches")
                            msg.exec_()
                        except RuntimeError:
                            logger.warning("Widget was destroyed before showing message box")
                        return
                    
                    # Continue with the rest of the filter application
                    # Check if widget still exists
                    if not hasattr(self, 'graph_renderer'):
                        logger.warning("Widget destroyed before applying filter segments")
                        return
                        
                    # ✅ FIX: Pass target_tab_index to prevent cross-tab filter contamination
                    self._apply_calculated_segments(container, graph_index, time_segments, mode, filter_data, target_tab_index)
                    
                except RuntimeError as e:
                    logger.warning(f"Callback execution failed - widget may be deleted: {e}")
                except Exception as e:
                    logger.error(f"Error in filter callback: {e}")
            
            # Start threaded calculation with tab and graph indices
            # CRITICAL FIX: Use target_tab_index, not active_tab_index!
            self.filter_manager.calculate_filter_segments_threaded(
                all_signals, 
                conditions, 
                on_segments_calculated,
                tab_index=target_tab_index,
                graph_index=graph_index
            )
            return  # Exit here, continuation happens in callback
            
        except RuntimeError as e:
            logger.error(f"Runtime error in _apply_range_filter (widget may be deleted): {e}")
            # Hide loading indicator if it was shown
            if hasattr(self, 'loading_manager') and self.loading_manager:
                try:
                    self.loading_manager.finish_operation("filtering")
                except:
                    pass
        except Exception as e:
            logger.error(f"Error in _apply_range_filter: {e}")
            # Hide loading indicator if it was shown
            if hasattr(self, 'loading_manager') and self.loading_manager:
                try:
                    self.loading_manager.finish_operation("filtering")
                except:
                    pass
    
    def _apply_calculated_segments(self, container, graph_index, time_segments, mode, filter_data, target_tab_index):
        """Apply calculated filter segments to the graph.
        
        Args:
            container: GraphContainer for the target tab
            graph_index: Index of the graph to apply filter to
            time_segments: Calculated time segments
            mode: Filter mode ('segmented' or 'concatenated')
            filter_data: Filter configuration data
            target_tab_index: Target tab index (CRITICAL for multi-tab isolation)
        """
        try:
            # ✅ FIX: Use passed target_tab_index parameter instead of recalculating
            # This ensures filter is applied to correct tab even if user switched tabs
            logger.info(f"[FILTER ISOLATION] Applying filter to tab {target_tab_index}, graph {graph_index}, mode: {mode}")
            
            # Save filter state using filter manager (to TARGET tab!)
            self.filter_manager.save_filter_state(target_tab_index, filter_data)
            
            # Filtre durumunu widget container manager'a da kaydet
            # Bu sayede dosya değiştirme sırasında filtreler korunur
            if hasattr(self, 'parent') and hasattr(self.parent, 'widget_container_manager'):
                self.parent.widget_container_manager.save_current_filter_state()
            
            # Update graph renderer with current signal mapping
            self.graph_renderer.graph_signal_mapping = self.graph_signal_mapping
            
            # Apply filtering based on mode using graph renderer
            if mode == 'segmented':
                # Pass target_tab_index to ensure filter is applied to CORRECT tab!
                self.graph_renderer.apply_segmented_filter(container, graph_index, time_segments, target_tab_index)
            else:  # concatenated
                # ✅ FIX Concatenated Mode: Apply to ALL tabs, not just target tab!
                # Concatenated mode değiştirir signal_processor'daki veriyi
                self.graph_renderer.apply_concatenated_filter(container, time_segments)
                
                # Get filtered signals
                all_signals = self.signal_processor.get_all_signals()
                all_signal_names = sorted(list(all_signals.keys()))
                
                # CRITICAL: Apply concatenated filter to ALL TABS
                logger.info(f"[CONCATENATED] Applying filter to ALL tabs (total: {len(self.graph_containers)})")
                for tab_idx, tab_container in enumerate(self.graph_containers):
                    logger.info(f"[CONCATENATED] Updating tab {tab_idx}")
                    
                    # Clear all signals in this tab
                    tab_container.plot_manager.clear_all_signals()
                    
                    # Get signal mapping for this tab
                    tab_mapping = self.graph_signal_mapping.get(tab_idx, {})
                    
                    # Redraw all signals in this tab with filtered data
                    for g_idx, signal_names in tab_mapping.items():
                        if g_idx < tab_container.plot_manager.get_subplot_count():
                            for name in signal_names:
                                if name in all_signals:
                                    signal_data = all_signals[name]
                                    signal_index = all_signal_names.index(name)
                                    color = self.theme_manager.get_signal_color(signal_index)
                                    tab_container.plot_manager.add_signal(
                                        name, 
                                        signal_data['x_data'], 
                                        signal_data['y_data'], 
                                        plot_index=g_idx, 
                                        pen=color
                                    )
                    
                    # Apply limit lines to this tab
                    plot_widgets = tab_container.plot_manager.get_plot_widgets()
                    for g_idx, plot_widget in enumerate(plot_widgets):
                        visible_signals = self.graph_signal_mapping.get(tab_idx, {}).get(g_idx, [])
                        if self.graph_renderer:
                            self.graph_renderer._apply_limit_lines(plot_widget, g_idx, visible_signals)
                    
                    # Auto-range all plots in this tab
                    for plot_widget in tab_container.plot_manager.get_plot_widgets():
                        plot_widget.enableAutoRange(axis='x', enable=True)
                        plot_widget.enableAutoRange(axis='y', enable=True)
                        plot_widget.autoRange()
                        plot_widget.enableAutoRange(axis='x', enable=False)
                        plot_widget.enableAutoRange(axis='y', enable=False)
                
                logger.info(f"[CONCATENATED] Filter applied to ALL {len(self.graph_containers)} tabs")
                
        except Exception as e:
            logger.error(f"Error applying range filter: {e}")
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setStyleSheet(self._get_message_box_style())
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Error applying filter: {str(e)}")
            msg.setWindowTitle("Filter Error")
            msg.exec_()
    
    def _restore_filter_ui_state(self, saved_filters: dict):
        """
        Kaydedilmiş filtre durumunu UI'da geri yükle.
        Widget container manager tarafından çağrılır.
        
        Args:
            saved_filters: Kaydedilmiş filtre durumu
        """
        try:
            logger.info(f"Restoring filter UI state: {len(saved_filters)} saved filters")
            
            if not saved_filters:
                logger.debug("No saved filters to restore")
                return
            
            # Her tab için filtre durumunu geri yükle
            for tab_index, filter_data in saved_filters.items():
                try:
                    # Filter panel'ı bul ve durumu geri yükle
                    if hasattr(self, 'parameters_panel') and self.parameters_panel:
                        # Parameters panel'daki filter panel'ları kontrol et
                        for widget in self.parameters_panel.findChildren(QWidget):
                            if hasattr(widget, 'graph_index') and hasattr(widget, 'set_range_filter_conditions'):
                                # Bu widget'ın graph_index'i ile tab_index'i eşleştir
                                widget_tab = getattr(widget, 'tab_index', None)
                                if widget_tab == tab_index:
                                    widget.set_range_filter_conditions(filter_data)
                                    logger.debug(f"Restored filter UI for tab {tab_index}")
                                    break
                    
                    logger.debug(f"Filter UI restored for tab {tab_index}")
                except Exception as e:
                    logger.warning(f"Error restoring filter UI for tab {tab_index}: {e}")
            
            logger.info("Filter UI state restoration completed")
            
        except Exception as e:
            logger.error(f"Error in _restore_filter_ui_state: {e}")
    
    def _refresh_graph_display(self, container):
        """Refresh graph display to show all data (remove filters)."""
        try:
            
            # Clear any existing filters on the container
            container.plot_manager.clear_all_signals()
            
            # Get current tab index and redraw signals for this container
            active_tab_index = self.tab_widget.currentIndex()
            if active_tab_index >= 0:
                # Get signal mapping for this tab
                tab_mapping = self.graph_signal_mapping.get(active_tab_index, {})
                all_signals = self.signal_processor.get_all_signals()
                all_signal_names = list(all_signals.keys())
                
                # Redraw all signals for this container
                for graph_index, signal_names in tab_mapping.items():
                    if graph_index < container.plot_manager.get_subplot_count():
                        for name in signal_names:
                            if name in all_signals:
                                signal_data = all_signals[name]
                                signal_index = all_signal_names.index(name)
                                color = self.theme_manager.get_signal_color(signal_index)
                                
                                container.plot_manager.add_signal(
                                    name, 
                                    signal_data['x_data'], 
                                    signal_data['y_data'], 
                                    plot_index=graph_index, 
                                    pen=color
                                )
                
                
        except Exception as e:
            logger.error(f"Error refreshing graph display: {e}")
    
    def _get_message_box_style(self) -> str:
        """Gets a consistent stylesheet for QMessageBox to match the space theme."""
        return """
            QMessageBox {
                background-color: #2d344a;
                font-size: 14px;
                color: #ffffff !important;
            }
            QMessageBox QLabel {
                color: #ffffff !important;
                padding: 10px;
                font-size: 14px;
                font-weight: normal;
                background: transparent;
            }
            QMessageBox * {
                color: #ffffff !important;
                background: transparent;
            }
            QMessageBox QTextEdit {
                color: #ffffff !important;
                background-color: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
            }
            QMessageBox QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a536b, stop:1 #3a4258);
                color: #ffffff !important;
                border: 1px solid #5a647d;
                padding: 8px 16px;
                border-radius: 5px;
                min-width: 90px;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5a647d, stop:1 #4a536b);
                border: 1px solid #7a849d;
                color: #ffffff !important;
            }
            QMessageBox QPushButton:pressed {
                background-color: #3a4258;
                color: #ffffff !important;
            }
        """
                               
    def _calculate_filter_segments(self, all_signals: dict, conditions: list) -> list:
        """Calculate time segments where all conditions are satisfied."""
        
        if not conditions:
            return []
            
        # Get time axis from first signal
        first_signal_name = next(iter(all_signals.keys()))
        first_signal = all_signals[first_signal_name]
        time_data = first_signal.get('x_data', [])
        
        
        if len(time_data) == 0:
            return []
            
        import numpy as np
        time_array = np.array(time_data)
        
        # Initialize mask with all True values
        combined_mask = np.ones(len(time_array), dtype=bool)
        
        # Apply each condition (AND logic between conditions)
        for i, condition in enumerate(conditions):
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            
            if param_name not in all_signals:
                continue
                
            signal_data = all_signals[param_name]
            y_data = np.array(signal_data.get('y_data', []))
            
            
            if len(y_data) != len(time_array):
                continue
                
            # Create mask for this parameter's conditions
            param_mask = np.ones(len(time_array), dtype=bool)
            
            # Apply range conditions (AND logic within parameter)
            for j, range_condition in enumerate(ranges):
                operator = range_condition['operator']
                value = range_condition['value']
                
                
                if operator == '>':
                    range_mask = y_data > value
                elif operator == '>=':
                    range_mask = y_data >= value
                elif operator == '<':
                    range_mask = y_data < value
                elif operator == '<=':
                    range_mask = y_data <= value
                else:
                    continue
                
                matching_points = np.sum(range_mask)
                    
                param_mask = param_mask & range_mask
                
            # Combine with overall mask (AND logic between parameters)
            param_matching = np.sum(param_mask)
            combined_mask = combined_mask & param_mask
            
        # Find continuous segments where mask is True
        total_matching = np.sum(combined_mask)
        
        segments = []
        in_segment = False
        segment_start = None
        
        for i, mask_value in enumerate(combined_mask):
            if mask_value and not in_segment:
                # Start of new segment
                segment_start = time_array[i]
                in_segment = True
            elif not mask_value and in_segment:
                # End of current segment
                segment_end = time_array[i-1]
                segments.append((segment_start, segment_end))
                in_segment = False
                
        # Handle case where segment extends to end of data
        if in_segment:
            segment_end = time_array[-1]
            segments.append((segment_start, segment_end))
            
        return segments
        
    def _apply_segmented_filter(self, container, graph_index: int, time_segments: list):
        """Apply segmented display filter - show matching segments with gaps."""
        
        # Get signals for this graph
        active_tab_index = self.tab_widget.currentIndex()
        visible_signals = self.graph_signal_mapping.get(active_tab_index, {}).get(graph_index, [])
        
        
        if not visible_signals:
            return
            
        # Clear existing plots for this graph
        plot_widgets = container.plot_manager.get_plot_widgets()
        
        if graph_index < len(plot_widgets):
            plot_widget = plot_widgets[graph_index]
            plot_widget.clear()
        else:
            return
            
        # Plot each signal with segmented data
        all_signals = self.signal_processor.get_all_signals()
        
        for signal_name in visible_signals:
            
            if signal_name not in all_signals:
                continue
                
            signal_data = all_signals[signal_name]
            full_x_data = np.array(signal_data.get('x_data', []))
            full_y_data = np.array(signal_data.get('y_data', []))
            
            
            # Create segmented data
            segments_plotted = 0
            for i, (segment_start, segment_end) in enumerate(time_segments):
                # Find indices for this segment
                mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                segment_x = full_x_data[mask]
                segment_y = full_y_data[mask]
                
                if len(segment_x) > 0:
                    # Plot this segment
                    color = self._get_signal_color(signal_name)
                    # Only show legend for the first segment of each signal
                    legend_name = signal_name if segments_plotted == 0 else None
                    plot_widget.plot(segment_x, segment_y, pen=color, name=legend_name)
                    segments_plotted += 1
                    
                    
        logger.info(f"Segmented filter applied successfully to graph {graph_index}")
        
        # Show success message
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setStyleSheet(self._get_message_box_style())
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Segmented filter applied to Graph {graph_index + 1}.\n\n"
            f"Showing {len(time_segments)} time segments with gaps.\n\n"
            "Time synchronization with other graphs is maintained.")
        msg.setWindowTitle("Filter Applied")
        msg.exec_()
        
    def _apply_concatenated_filter(self, container, graph_index: int, time_segments: list):
        """Apply concatenated display filter - join matching segments continuously."""
        logger.info(f"Applying concatenated filter to graph {graph_index} with {len(time_segments)} segments")
        
        # Get signals for this graph
        active_tab_index = self.tab_widget.currentIndex()
        visible_signals = self.graph_signal_mapping.get(active_tab_index, {}).get(graph_index, [])
        
        if not visible_signals:
            logger.warning(f"No visible signals for graph {graph_index}")
            return
            
        # Clear existing plots for this graph
        plot_widgets = container.plot_manager.get_plot_widgets()
        if graph_index < len(plot_widgets):
            plot_widget = plot_widgets[graph_index]
            plot_widget.clear()
        else:
            logger.warning(f"Graph index {graph_index} out of range, available plots: {len(plot_widgets)}")
            return
            
        # Plot each signal with concatenated data
        all_signals = self.signal_processor.get_all_signals()
        
        for signal_name in visible_signals:
            if signal_name not in all_signals:
                continue
                
            signal_data = all_signals[signal_name]
            full_x_data = np.array(signal_data.get('x_data', []))
            full_y_data = np.array(signal_data.get('y_data', []))
            
            # Concatenate all segments
            concatenated_x = []
            concatenated_y = []
            new_time_offset = 0.0
            
            for i, (segment_start, segment_end) in enumerate(time_segments):
                # Find indices for this segment
                mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                segment_x = full_x_data[mask]
                segment_y = full_y_data[mask]
                
                if len(segment_x) > 0:
                    # Adjust time axis for concatenation
                    if i == 0:
                        adjusted_x = segment_x - segment_x[0]  # Start from 0
                    else:
                        adjusted_x = segment_x - segment_x[0] + new_time_offset
                        
                    concatenated_x.extend(adjusted_x)
                    concatenated_y.extend(segment_y)
                    
                    # Update offset for next segment
                    if len(adjusted_x) > 0:
                        new_time_offset = adjusted_x[-1] + (adjusted_x[-1] - adjusted_x[0]) * 0.01  # Small gap
                        
            if concatenated_x:
                # Plot concatenated data
                color = self._get_signal_color(signal_name)
                plot_widget.plot(concatenated_x, concatenated_y, pen=color, name=signal_name)
                
        logger.info(f"Concatenated filter applied successfully to graph {graph_index}")

    def _apply_global_concatenated_filter(self, container, time_segments: list):
        """Apply concatenated display filter globally to all graphs in the tab."""
        logger.info(f"Applying global concatenated filter with {len(time_segments)} segments to all graphs")
        
        # Get all signals from signal processor
        all_signals = self.signal_processor.get_all_signals()
        
        if not all_signals:
            logger.warning("No signals available for global filter")
            return
            
        # Create concatenated time axis and signal data for all signals
        concatenated_signals = self._create_concatenated_signals(all_signals, time_segments)
        
        if not concatenated_signals:
            logger.warning("Failed to create concatenated signals")
            return
            
        # Clear all plots in the container
        plot_widgets = container.plot_manager.get_plot_widgets()
        for plot_widget in plot_widgets:
            plot_widget.clear()
            
        # Get current tab's graph signal mapping
        active_tab_index = self.tab_widget.currentIndex()
        tab_mapping = self.graph_signal_mapping.get(active_tab_index, {})
        
        # Apply concatenated data to all graphs
        for graph_index, signal_names in tab_mapping.items():
            if graph_index < len(plot_widgets):
                plot_widget = plot_widgets[graph_index]
                
                for signal_name in signal_names:
                    if signal_name in concatenated_signals:
                        concat_data = concatenated_signals[signal_name]
                        color = self._get_signal_color(signal_name)
                        plot_widget.plot(
                            concat_data['x_data'], 
                            concat_data['y_data'], 
                            pen=color, 
                            name=signal_name
                        )
        
        # Update signal processor with concatenated data (for statistics etc.)
        self._update_signal_processor_with_concatenated_data(concatenated_signals)
        
        # Show success message
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setStyleSheet(self._get_message_box_style())
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Concatenated filter applied to all graphs in this tab.\n\n"
            f"Time axis shows {len(time_segments)} segments continuously.\n\n"
            f"All {len(concatenated_signals)} signals are now synchronized to the filtered time domain.")
        msg.setWindowTitle("Global Filter Applied")
        msg.exec_()
        
        logger.info(f"Global concatenated filter applied successfully to {len(concatenated_signals)} signals")

    def _create_concatenated_signals(self, all_signals: dict, time_segments: list) -> dict:
        """Create concatenated signal data for all signals."""
        concatenated_signals = {}
        
        for signal_name, signal_data in all_signals.items():
            full_x_data = np.array(signal_data.get('x_data', []))
            full_y_data = np.array(signal_data.get('y_data', []))
            
            # Concatenate all segments for this signal
            concatenated_x = []
            concatenated_y = []
            new_time_offset = 0.0
            
            for i, (segment_start, segment_end) in enumerate(time_segments):
                # Find indices for this segment
                mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                segment_x = full_x_data[mask]
                segment_y = full_y_data[mask]
                
                if len(segment_x) > 0:
                    # Adjust time axis for concatenation
                    if i == 0:
                        adjusted_x = segment_x - segment_x[0]  # Start from 0
                    else:
                        adjusted_x = segment_x - segment_x[0] + new_time_offset
                        
                    concatenated_x.extend(adjusted_x)
                    concatenated_y.extend(segment_y)
                    
                    # Update offset for next segment
                    if len(adjusted_x) > 0:
                        new_time_offset = adjusted_x[-1] + (adjusted_x[-1] - adjusted_x[0]) * 0.01  # Small gap
                        
            if concatenated_x:
                concatenated_signals[signal_name] = {
                    'x_data': np.array(concatenated_x),
                    'y_data': np.array(concatenated_y),
                    'original_x': full_x_data,
                    'original_y': full_y_data,
                    'metadata': signal_data.get('metadata', {})
                }
                
        logger.info(f"Created concatenated data for {len(concatenated_signals)} signals")
        return concatenated_signals

    def _update_signal_processor_with_concatenated_data(self, concatenated_signals: dict):
        """Update signal processor with concatenated data for statistics calculations."""
        try:
            # Clear existing data
            self.signal_processor.clear_all_data()
            
            # Add concatenated signals
            for signal_name, concat_data in concatenated_signals.items():
                self.signal_processor.add_signal(
                    signal_name,
                    concat_data['x_data'],
                    concat_data['y_data'],
                    concat_data.get('metadata', {})
                )
                
            logger.info(f"Updated signal processor with {len(concatenated_signals)} concatenated signals")
            
            # DON'T update statistics here - wait for cursor movement for performance
            
        except Exception as e:
            logger.error(f"Error updating signal processor with concatenated data: {e}")
        
    def _get_signal_color(self, signal_name: str) -> str:
        """Get color for a signal (simplified version)."""
        # Simple color cycling - in real implementation, use proper color management
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
        hash_value = hash(signal_name) % len(colors)
        return colors[hash_value]
    
    def _update_legend_values(self):
        """
        Update legend with current values.
        OPTIMIZED: Uses batch signal retrieval to avoid mutex contention.
        """
        if self.current_cursor_position is not None:
            signal_data = {}
            
            # Get all signal names
            signal_names = list(self.signal_processor.get_all_signals().keys())
            
            # Batch retrieval
            values = self.signal_processor.get_signals_at_time(signal_names, self.current_cursor_position)
            
            # Update data structure provided to legend manager
            for signal_name, value in values.items():
                if value is not None:
                    signal_data[signal_name] = {'y_data': [value]}
                    
            self.legend_manager.update_values_from_data(signal_data)
        else:
            all_signals = self.signal_processor.get_all_signals()
            signal_data = {}
            for signal_name, data in all_signals.items():
                if 'y_data' in data and len(data['y_data']) > 0:
                    signal_data[signal_name] = {'y_data': data['y_data']}
            self.legend_manager.update_values_from_data(signal_data)
        
        # DON'T update statistics here - cursor movement will trigger it
    
    # Public API methods
    def set_theme(self, theme_name: str):
        """Set the visual theme."""
        self.theme_manager.set_theme(theme_name)
    
    def _apply_theme(self):
        """Apply the current theme to all components."""
        theme_name = self.theme_manager.get_current_theme()
        
        # Apply to main widget background and panels
        self.setStyleSheet(f"background-color: {self.theme_manager.get_color('background')};")
        
        # Update toolbar with theme colors
        if hasattr(self.toolbar_manager, 'update_theme'):
            self.toolbar_manager.update_theme()
        else:
            self.toolbar_manager.get_toolbar().setStyleSheet(
                self.theme_manager.get_widget_stylesheet('toolbar', theme_name)
            )
        
        # Update settings panel with theme colors
        if hasattr(self.settings_panel_manager, 'update_theme'):
            self.settings_panel_manager.update_theme()
        else:
            self.settings_panel_manager.get_settings_panel().setStyleSheet(
                self.theme_manager.get_widget_stylesheet('panel', theme_name)
            )
        self.statistics_settings_panel_manager.get_settings_panel().setStyleSheet(
            self.theme_manager.get_widget_stylesheet('panel', theme_name)
        )
        self.graph_settings_panel_manager.get_settings_panel().setStyleSheet(
            self.theme_manager.get_widget_stylesheet('panel', theme_name)
        )
        if hasattr(self, 'channel_stats_panel'):
            self.channel_stats_panel.setStyleSheet(
                self.theme_manager.get_widget_stylesheet('panel', theme_name)
            )
        
        # Update statistics panel with theme colors
        if hasattr(self, 'statistics_panel') and hasattr(self.statistics_panel, 'update_theme'):
            self.statistics_panel.update_theme(self.theme_manager.get_theme_colors())
        
        # Apply to all graph containers
        for container in self.graph_containers:
            container.apply_theme()
        
        # Update correlations panel with theme colors
        if hasattr(self, 'correlations_panel_manager') and hasattr(self.correlations_panel_manager, 'update_theme'):
            self.correlations_panel_manager.update_theme()
        
        # Update graph settings panel with theme colors
        if hasattr(self, 'graph_settings_panel_manager') and hasattr(self.graph_settings_panel_manager, 'update_theme'):
            self.graph_settings_panel_manager.update_theme()
        
        # Redraw signals with new theme colors
        self._redraw_all_signals()
    
    def _setup_connections(self):
        """Setup signal-slot connections for the widget."""
        # Toolbar connections
        # NOTE: cursor_mode_changed connection removed - cursor mode is now permanently 'dual'
        self.toolbar_manager.panel_toggled.connect(self._on_panel_toggled)
        self.toolbar_manager.settings_toggled.connect(self._on_settings_toggled)
        self.toolbar_manager.graph_count_changed.connect(self._on_graph_count_changed)
        
        # Connect the new statistics settings toggle signal
        if hasattr(self.toolbar_manager, 'statistics_settings_toggled'):
            self.toolbar_manager.statistics_settings_toggled.connect(self._on_statistics_settings_toggled)
            
        if hasattr(self.toolbar_manager, 'graph_settings_toggled'):
            self.toolbar_manager.graph_settings_toggled.connect(self._on_graph_settings_toggled)
        
        if hasattr(self.toolbar_manager, 'parameters_toggled'):
            self.toolbar_manager.parameters_toggled.connect(self._on_parameters_toggled)
            
        # Connect new analysis panel signals
        if hasattr(self.toolbar_manager, 'correlations_toggled'):
            self.toolbar_manager.correlations_toggled.connect(self._on_correlations_toggled)
            
        if hasattr(self.toolbar_manager, 'bitmask_toggled'):
            self.toolbar_manager.bitmask_toggled.connect(self._on_bitmask_toggled)

        # Settings panel connections
        self.settings_panel_manager.theme_changed.connect(self.set_theme)

        # Statistics settings panel connections
        self.statistics_settings_panel_manager.visible_columns_changed.connect(self._on_visible_columns_changed)
        self.statistics_settings_panel_manager.duty_cycle_threshold_changed.connect(self._on_duty_cycle_threshold_changed)
        
        # Graph settings panel connections (per-graph - for right-click menu)
        self.graph_settings_panel_manager.normalization_toggled.connect(self._on_per_graph_normalization_toggled)
        self.graph_settings_panel_manager.view_reset_requested.connect(self._on_per_graph_view_reset)
        self.graph_settings_panel_manager.grid_visibility_changed.connect(self._on_per_graph_grid_changed)
        self.graph_settings_panel_manager.autoscale_changed.connect(self._on_per_graph_autoscale_changed)
        
        # Global graph settings panel connections (for panel controls)
        self.graph_settings_panel_manager.global_normalization_toggled.connect(self._on_global_normalization_toggled)
        self.graph_settings_panel_manager.global_view_reset_requested.connect(self._on_global_view_reset)
        self.graph_settings_panel_manager.global_grid_visibility_changed.connect(self._on_global_grid_changed)
        self.graph_settings_panel_manager.global_autoscale_changed.connect(self._on_global_autoscale_changed)
        self.graph_settings_panel_manager.global_legend_visibility_changed.connect(self._on_global_legend_visibility_changed)
        self.graph_settings_panel_manager.global_tooltips_changed.connect(self._on_global_tooltips_changed)
        self.graph_settings_panel_manager.global_snap_to_data_changed.connect(self._on_global_snap_changed)
        self.graph_settings_panel_manager.global_line_width_changed.connect(self._on_global_line_width_changed)
        self.graph_settings_panel_manager.global_x_axis_mouse_changed.connect(self._on_global_x_mouse_changed)
        self.graph_settings_panel_manager.global_y_axis_mouse_changed.connect(self._on_global_y_mouse_changed)
        self.graph_settings_panel_manager.global_secondary_axis_changed.connect(self._on_global_secondary_axis_changed)

        # Theme manager connections
        self.theme_manager.theme_changed.connect(self._on_theme_changed)
        
        # Settings panel connections
        self.settings_panel_manager.theme_changed.connect(self.theme_manager.set_theme)
        
        # Connect bitmask panel to theme changes
        self.theme_manager.theme_changed.connect(self.bitmask_panel_manager.update_theme)
        
        # Legend manager connections
        self.legend_manager.signal_visibility_changed.connect(self._on_signal_visibility_changed)
        self.legend_manager.signal_selected.connect(self._on_signal_selected)
        
        # Signal processor connections
        self.signal_processor.processing_started.connect(self._on_processing_started)
        self.signal_processor.processing_finished.connect(self._on_processing_finished)
        self.signal_processor.statistics_updated.connect(self._on_statistics_updated)
    
    def _on_signal_color_changed(self, signal_name: str, new_color: str):
        """Handle color changes for a specific signal from the statistics panel."""
        logger.info(f"Color change requested for signal '{signal_name}' to {new_color}")
        
        all_signals = self.signal_processor.get_all_signals()
        all_signal_names = sorted(list(all_signals.keys()))
        
        logger.debug(f"Available signals: {all_signal_names}")
        
        if signal_name in all_signal_names:
            signal_index = all_signal_names.index(signal_name)
            logger.info(f"Found signal '{signal_name}' at index {signal_index}")
            
            # Update theme manager with the color override
            self.theme_manager.set_signal_color_override(signal_index, new_color)
            
            # Update legend color if legend manager exists
            if hasattr(self, 'legend_manager') and self.legend_manager:
                self.legend_manager.set_signal_color(signal_name, new_color)
            
            # Redraw all signals to apply the new color
            self._redraw_all_signals()
            logger.info(f"Successfully updated color for signal '{signal_name}'")
        else:
            logger.warning(f"Could not find signal '{signal_name}' to change its color.")
            logger.warning(f"Available signals: {all_signal_names}")
    
    def _on_signal_remove_requested(self, signal_name: str, graph_index: int):
        """
        Handle request to remove a signal from a graph.
        Called when user right-clicks on signal name in statistics panel.
        """
        logger.info(f"Remove requested for signal '{signal_name}' from graph {graph_index}")
        
        # Get current tab index
        tab_index = self.tab_widget.currentIndex()
        if tab_index < 0:
            logger.warning("No active tab")
            return
        
        # Remove signal from graph signal mapping
        if tab_index in self.graph_signal_mapping:
            if graph_index in self.graph_signal_mapping[tab_index]:
                if signal_name in self.graph_signal_mapping[tab_index][graph_index]:
                    self.graph_signal_mapping[tab_index][graph_index].remove(signal_name)
                    logger.info(f"Removed '{signal_name}' from graph {graph_index} in tab {tab_index}")
                    
                    # Redraw graphs to reflect the change
                    self._redraw_all_signals()
                    
                    # Update statistics panel
                    self._recreate_statistics_panel()
                    
                    logger.info(f"Signal '{signal_name}' successfully removed from display")
                else:
                    logger.warning(f"Signal '{signal_name}' not found in graph {graph_index} mapping")
            else:
                logger.warning(f"Graph {graph_index} not found in tab {tab_index} mapping")
        else:
            logger.warning(f"Tab {tab_index} not found in graph signal mapping")
    
    def _on_graph_reorder_requested(self, from_index: int, to_index: int):
        """Handle graph reorder request from statistics panel."""
        logger.info(f"Graph reorder requested: Graph {from_index + 1} -> Graph {to_index + 1}")
        tab_index = self.tab_widget.currentIndex()
        if tab_index < 0:
            return
        
        # Get active container
        active_container = self.get_active_graph_container()
        if not active_container:
            logger.warning("No active container found for graph reordering")
            return
        
        # Reorder graphs in PlotManager
        active_container.plot_manager.reorder_graphs(from_index, to_index)
        
        # Update graph_signal_mapping to reflect new order
        if tab_index in self.graph_signal_mapping:
            # Swap the signal lists
            signals_from = self.graph_signal_mapping[tab_index].get(from_index, []).copy()
            signals_to = self.graph_signal_mapping[tab_index].get(to_index, []).copy()
            
            # Update mapping
            self.graph_signal_mapping[tab_index][from_index] = signals_to
            self.graph_signal_mapping[tab_index][to_index] = signals_from
            
            # Update cursor manager if it exists
            if hasattr(self, 'cursor_manager') and self.cursor_manager:
                # Cursor manager needs to be updated with new plot widget order
                # The plot widgets are already reordered in PlotManager
                plot_widgets = active_container.plot_manager.get_plot_widgets()
                # Reinitialize cursor manager with new order
                from src.managers.cursor_manager import CursorManager
                old_cursor_manager = self.cursor_manager
                self.cursor_manager = CursorManager(plot_widgets)
                # Copy cursor positions if available
                if hasattr(old_cursor_manager, 'dual_cursors_1') and hasattr(old_cursor_manager, 'dual_cursors_2'):
                    # Cursor positions will be maintained by the plot widgets themselves
                    pass
                old_cursor_manager.deleteLater()
            
            # Mark data as modified
            self.is_data_modified = True
            
            # Recreate statistics panel to reflect new order
            self._recreate_statistics_panel()
            
            logger.info(f"Graphs reordered successfully: Graph {from_index + 1} <-> Graph {to_index + 1}")
        else:
            logger.warning(f"Tab {tab_index} not found in graph signal mapping")

    def _on_visible_columns_changed(self, visible_columns: set):
        """Handle changes to visible statistics columns."""
        self.visible_stats_columns = visible_columns
        # Update the statistics panel with new visible columns
        self.statistics_panel.set_visible_stats(visible_columns)
        self._recreate_statistics_panel() # Redraw panel with new columns

    def _on_duty_cycle_threshold_changed(self, threshold_mode: str, threshold_value: float):
        """Handle changes to duty cycle threshold settings."""
        self.duty_cycle_threshold_mode = threshold_mode
        self.duty_cycle_threshold_value = threshold_value
        
        # Update statistics with new threshold settings
        self._update_statistics()
        
        logger.info(f"Duty cycle threshold updated: mode={threshold_mode}, value={threshold_value}")
    
    def get_current_statistics(self) -> Dict[str, Dict]:
        """Get current statistics for all signals."""
        return self.signal_processor.calculate_statistics()
    
    def export_data(self) -> Dict[str, Any]:
        """Export current data and settings."""
        return {
            'signals': self.signal_processor.get_all_signals(),
            'theme': self.theme_manager.get_current_theme(),
            'normalized': self.is_normalized,
            'cursor_position': self.current_cursor_position,
            'selected_range': self.selected_range
        }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return self.signal_processor.get_memory_usage()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up TimeGraphWidget resources")
        # Disconnect any remaining signals, stop timers, etc.
        # This prevents potential issues when the widget is destroyed.
        try:
            # Stop any active processing threads
            if hasattr(self, 'processing_thread') and self.processing_thread:
                try:
                    if self.processing_thread.isRunning():
                        logger.debug("Stopping processing thread...")
                        self.processing_thread.quit()
                        if not self.processing_thread.wait(3000):  # Wait up to 3 seconds
                            logger.warning("Processing thread did not finish, terminating...")
                            self.processing_thread.terminate()
                            self.processing_thread.wait(1000)
                        logger.debug("Processing thread stopped")
                except RuntimeError as e:
                    # Thread already deleted by deleteLater()
                    logger.debug(f"Processing thread already deleted: {e}")
            
            # Clean up graph renderer threads - CRITICAL: Thread temizliği
            if hasattr(self, 'graph_renderer') and self.graph_renderer:
                logger.debug("Cleaning up graph renderer...")
                self.graph_renderer.cleanup()
                logger.debug("Graph renderer cleaned up")
            
            # Clean up filter manager threads - CRITICAL: Thread temizliği
            if hasattr(self, 'filter_manager') and self.filter_manager:
                logger.debug("Cleaning up filter manager...")
                self.filter_manager.cleanup()
                logger.debug("Filter manager cleaned up")
            
            # Clean up plot managers in all containers
            if hasattr(self, 'graph_containers'):
                logger.debug("Cleaning up plot managers...")
                for container in self.graph_containers:
                    if hasattr(container, 'plot_manager') and hasattr(container.plot_manager, 'cleanup'):
                        try:
                            container.plot_manager.cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up plot manager: {e}")
                logger.debug("Plot managers cleaned up")
            
            # Wait a bit for all threads to finish
            import time
            time.sleep(0.1)
            
            logger.info("TimeGraphWidget cleanup complete")
        except Exception as e:
            logger.error(f"Error during TimeGraphWidget cleanup: {e}")

    def closeEvent(self, event):
        """Handle widget close event."""
        self.cleanup()
        super().closeEvent(event)

    def _save_graph_setting(self, graph_index: int, setting_name: str, value):
        """Save a setting for a specific graph in the active tab."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return
            
        # Initialize tab settings if not exists
        if active_tab_index not in self.graph_settings:
            self.graph_settings[active_tab_index] = {}
            
        # Initialize graph settings if not exists
        if graph_index not in self.graph_settings[active_tab_index]:
            self.graph_settings[active_tab_index][graph_index] = {}
            
        # Save the setting
        self.graph_settings[active_tab_index][graph_index][setting_name] = value
        logger.debug(f"Saved setting: Tab {active_tab_index}, Graph {graph_index}, {setting_name} = {value}")

    def _get_graph_setting(self, graph_index: int, setting_name: str, default_value=None):
        """Get a setting for a specific graph in the active tab."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return default_value
            
        return (self.graph_settings
                .get(active_tab_index, {})
                .get(graph_index, {})
                .get(setting_name, default_value))

    def _apply_saved_graph_settings(self):
        """Apply saved settings to all graphs in the active tab."""
        active_container = self.get_active_graph_container()
        if not active_container:
            return
            
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return
            
        plot_widgets = active_container.get_plot_widgets()
        tab_settings = self.graph_settings.get(active_tab_index, {})
        
        for graph_index, graph_settings in tab_settings.items():
            if 0 <= graph_index < len(plot_widgets):
                plot_widget = plot_widgets[graph_index]
                
                # Apply grid setting
                show_grid = graph_settings.get('show_grid', True)  # Default to True
                plot_widget.showGrid(x=show_grid, y=show_grid)
                
                # Apply autoscale setting
                autoscale = graph_settings.get('autoscale', True)  # Default to True
                plot_widget.enableAutoRange(axis='y', enable=autoscale)
                
                logger.debug(f"Applied settings to Graph {graph_index}: grid={show_grid}, autoscale={autoscale}")
        
        # Update graph settings panel to reflect current settings
        self._sync_graph_settings_panel()

    def _sync_graph_settings_panel(self):
        """Synchronize the graph settings panel checkboxes with actual settings."""
        # Global settings don't need per-graph synchronization
        # The global settings are maintained internally by the panel manager
        logger.debug("Graph settings panel sync skipped - using global settings")

    def _recreate_statistics_panel(self):
        """Recreates the channel statistics groups in the ATI panel based on the active tab."""
        # Get active container first
        active_container = self.get_active_graph_container()
        if not active_container:
            return
        
        # Save current column widths before making any changes
        saved_widths = self.statistics_panel._save_current_column_widths()
        logger.debug(f"Saved column widths before recreating statistics panel: {saved_widths}")
            
        # Get current graph count
        num_graphs = active_container.plot_manager.get_subplot_count()
        
        # Update statistics panel graph count (this will remove excess graphs)
        self.statistics_panel.update_graph_count(num_graphs)
        
        # Clear all existing signals from the modern statistics panel
        self.statistics_panel.clear_all()
        
        # Reset the storage for signal tracking
        self.channel_stats_widgets = {}

        # Add signals from all graphs to the statistics panel
        tab_index = self.tab_widget.currentIndex()
        
        # Ensure statistics panel has sections for all graphs
        if num_graphs > 0:
            self.statistics_panel.ensure_graph_sections(num_graphs - 1)
        
        for graph_idx in range(num_graphs):
            # Get signals for this graph
            tab_mapping = self.graph_signal_mapping.get(tab_index, {})
            if graph_idx in tab_mapping:
                signals = tab_mapping[graph_idx]
                
                for signal_name in signals:
                    # Try to get signal color from plot manager, fallback to theme manager
                    color = active_container.plot_manager.get_signal_color(graph_idx, signal_name)
                    if not color:
                        # Fallback: use theme manager to get color by signal index
                        all_signals = list(self.signal_processor.get_all_signals().keys())
                        if signal_name in all_signals:
                            signal_index = all_signals.index(signal_name)
                            color = self.theme_manager.get_signal_color(signal_index)
                        else:
                            color = "#ffffff"  # Default white
                    
                    # Add signal to modern statistics panel
                    full_signal_name = f"{signal_name} (G{graph_idx+1})"
                    self.statistics_panel.add_signal(full_signal_name, graph_idx, signal_name, color)
                    #logger.debug(f"Added signal '{signal_name}' to graph {graph_idx+1} statistics panel with color {color}")
        
        # Restore column widths after recreating panel
        if saved_widths:
            self.statistics_panel._restore_column_widths_to_all_tables(saved_widths)
            logger.debug(f"Restored column widths after recreating statistics panel: {saved_widths}")
        
        # DON'T update statistics here - wait for cursor movement
        # This keeps UI responsive when plotting signals
        # Statistics will update automatically when cursor moves

    def get_layout_config(self):
        """Mevcut sekme, grafik ve sinyal düzenini bir sözlük olarak alır."""
        tabs_config = []
        
        for i in range(self.tab_widget.count()):
            # Her sekme bir GraphContainer widget'ıdır
            container = self.tab_widget.widget(i)
            
            if not isinstance(container, GraphContainer):
                continue
            
            # Sekme ismini al
            tab_name = self.tab_widget.tabText(i)
                
            # Bu container'daki grafik sayısını al
            subplot_count = container.plot_manager.get_subplot_count()
            
            graphs_config = []
            # Her grafik için sinyal listesini al
            for plot_index in range(subplot_count):
                plot_signals = []
                
                # Bu grafikte görünen sinyalleri bul
                current_signals = container.plot_manager.get_current_signals()
                for signal_key, plot_item in current_signals.items():
                    # signal_key formatı: "signal_name_plot_index"
                    if signal_key.endswith(f"_{plot_index}"):
                        signal_name = '_'.join(signal_key.split('_')[:-1])
                        plot_signals.append(signal_name)
                
                graphs_config.append({'signals': plot_signals})
            
            tabs_config.append({
                'name': tab_name,
                'graphs': graphs_config
            })
            
        return {'tabs': tabs_config}

    def set_layout_config(self, config):
        """Verilen konfigürasyona göre sekme, grafik ve sinyal düzenini ayarlar."""
        
        # Check if config is None (e.g., binary MPAI without layout)
        if config is None:
            logger.info("No layout config provided, keeping default layout")
            return
        
        # Mevcut tüm sekmeleri temizle
        self.tab_widget.clear()
        self.graph_containers.clear()

        if 'tabs' not in config:
            return

        for i, tab_config in enumerate(config['tabs']):
            # Sekme ismini al (varsa kaydedilen ismi, yoksa varsayılan ismi kullan)
            tab_name = tab_config.get('name', f"Tab {i+1}")
            
            # Yeni sekme ekle
            container = self._add_tab(tab_name)
            
            if 'graphs' in tab_config:
                graph_count = len(tab_config['graphs'])
                if graph_count > 0:
                    # Grafik sayısını ayarla
                    container.set_graph_count(graph_count)
                    
                    # UI güncellemesini bekle
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()

                    # Her grafik için sinyalleri ekle
                    for plot_index, graph_config in enumerate(tab_config['graphs']):
                        if 'signals' in graph_config:
                            # Mevcut sinyalleri kontrol et
                            all_signals = self.signal_processor.get_all_signals()
                            
                            for signal_name in graph_config['signals']:
                                if signal_name in all_signals:
                                    # Sinyal verilerini signal_processor'dan al
                                    signal_data = self.signal_processor.get_signal_data(signal_name)
                                    if signal_data and 'x_data' in signal_data and 'y_data' in signal_data:
                                        x_data = signal_data['x_data']
                                        y_data = signal_data['y_data']
                                        
                                        # Sinyali belirtilen grafiğe ekle
                                        container.add_signal(signal_name, x_data, y_data, plot_index)
                                        
                                        # Signal mapping'i güncelle
                                        if i not in self.graph_signal_mapping:
                                            self.graph_signal_mapping[i] = {}
                                        if plot_index not in self.graph_signal_mapping[i]:
                                            self.graph_signal_mapping[i][plot_index] = []
                                        if signal_name not in self.graph_signal_mapping[i][plot_index]:
                                            self.graph_signal_mapping[i][plot_index].append(signal_name)
                    
                    # Bu sekme için statistics panel'i güncelle
                    if i == self.tab_widget.currentIndex():  # Sadece aktif sekme için
                        self._recreate_statistics_panel()
                        # DON'T update statistics here - wait for cursor movement

    def _on_add_tab_clicked(self):
        """Handle the click event for adding a new tab."""
        self._add_tab()
        
    def _on_remove_tab_clicked(self):
        """Handle the click event for removing the current tab."""
        self._remove_tab()

    def _on_limits_applied_from_dialog(self, graph_index, limits_config):
        """
        Handle limits applied signal from advanced settings dialog.
        This is called when Apply or OK button is clicked.
        """
        logger.info(f"[LIMITS] Received limits_applied signal for graph {graph_index}")
        logger.debug(f"[LIMITS] Limits config: {limits_config}")
        
        try:
            # Save the settings for persistence
            self._save_graph_setting(graph_index, 'limits', limits_config)
            logger.info(f"[LIMITS] Saved limits to settings for graph {graph_index}")
            
            # Apply to graph renderer (store config)
            if self.graph_renderer:
                self.graph_renderer.set_static_limits(graph_index, limits_config)
                logger.info(f"[LIMITS] Set static limits in renderer for graph {graph_index}: {len(limits_config)} signals")
            
            # CRITICAL: Use target tab from dialog, NOT current active tab!
            target_tab_index = getattr(self, '_dialog_target_tab', self.tab_widget.currentIndex())
            if target_tab_index < 0 or target_tab_index >= len(self.graph_containers):
                logger.warning(f"[LIMITS] Invalid tab index: {target_tab_index}")
                return
                
            container = self.graph_containers[target_tab_index]
            plot_widgets = container.plot_manager.get_plot_widgets()
            
            # Apply limit lines to the specific graph
            if graph_index < len(plot_widgets):
                plot_widget = plot_widgets[graph_index]
                
                # Get visible signals for this graph (from TARGET tab!)
                visible_signals = self.graph_signal_mapping.get(target_tab_index, {}).get(graph_index, [])
                logger.info(f"[LIMITS] Visible signals for graph {graph_index}: {visible_signals}")
                
                # CRITICAL FIX: Clear old limit lines FIRST
                if self.graph_renderer:
                    self.graph_renderer._clear_limit_lines(plot_widget, graph_index)
                    logger.info(f"[LIMITS] Cleared old limit lines for graph {graph_index}")
                
                # Apply NEW limit lines (if any)
                if self.graph_renderer and visible_signals and limits_config:
                    self.graph_renderer._apply_limit_lines(plot_widget, graph_index, visible_signals)
                    logger.info(f"[LIMITS] Applied {len(limits_config)} new limit lines to graph {graph_index}")
                elif not limits_config:
                    logger.info(f"[LIMITS] No limits to apply (limits cleared) for graph {graph_index}")
                else:
                    logger.warning(f"[LIMITS] Cannot apply limit lines - renderer: {self.graph_renderer is not None}, signals: {len(visible_signals)}")
            else:
                logger.warning(f"[LIMITS] Graph index {graph_index} out of range (total plots: {len(plot_widgets)})")
            
            logger.info(f"[LIMITS] Successfully applied and saved limits for graph {graph_index}")
            
        except RuntimeError as e:
            # C++ module error - show user-friendly message
            logger.error(f"[LIMITS] RuntimeError: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Static Limits Error",
                f"<b>Static Limits could not be applied!</b><br><br>{str(e)}"
            )
        except Exception as e:
            logger.error(f"[LIMITS] Error applying static limits: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Static Limits Error",
                f"<b>Unexpected error:</b><br><br>{str(e)}"
            )
    
    def _apply_static_limits(self, graph_index, limits_config):
        """
        Legacy method - redirects to new handler.
        Apply static limits from the advanced settings dialog to the graph renderer.
        """
        self._on_limits_applied_from_dialog(graph_index, limits_config)


