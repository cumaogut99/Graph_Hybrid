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
    QTabBar, QMessageBox
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
        self.bitmask_panel_manager = BitmaskPanelManager(self.signal_processor, self.theme_manager, self)
        
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
        main_layout.addWidget(self.toolbar_manager.get_title_bar())
        
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
                image: url({os.path.abspath('icons/x.svg').replace(os.sep, '/')});
                subcontrol-position: right;
                subcontrol-origin: padding;
                margin: 1px;
                padding: 2px;
                border-radius: 3px;
                width: 14px;
                height: 14px;
            }}
            QTabBar::close-button:hover {{
                background-color: rgba(232, 17, 35, 0.8);
            }}
            QTabBar::close-button:pressed {{
                background-color: rgba(241, 112, 122, 0.9);
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
            # CRITICAL: Call clear_all() first to cancel any pending timers and remove cursors
            self.cursor_manager.clear_all()
            try:
                self.cursor_manager.cursor_moved.disconnect()
                self.cursor_manager.range_changed.disconnect()
            except TypeError: pass
            self.cursor_manager.deleteLater()
            self.cursor_manager = None
        
        if plot_widgets:
            # No autoRange here: the X axes of a tab are linked, so ranging
            # every plot let the last (possibly empty) plot decide the view
            # and discarded the user's zoom on every redraw. The view is set
            # by _redraw_all_signals; new cursors go to 1/3 and 2/3 of it.
            self.cursor_manager = CursorManager(plot_widgets)
            self.cursor_manager.set_snap_provider(self._nearest_sample_time)
            
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

    def _nearest_sample_time(self, x: float):
        """
        Cursor snap target: the sample nearest to x of a signal shown in the
        active tab (the signals of a file share one time base), or of any
        signal if the tab is empty.
        """
        tab_mapping = self.graph_signal_mapping.get(self.tab_widget.currentIndex(), {})
        shown = [name for names in tab_mapping.values() for name in names]
        for name in shown or list(self.signal_processor.signal_data):
            sample_time = self.signal_processor.nearest_sample_time(name, x)
            if sample_time is not None:
                return sample_time
        return None

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

        # Tabs that showed no signal before: their X range is not a user's
        # zoom, so it is fitted to the data drawn now (see below)
        tabs_without_data = {
            tab_index for tab_index, container in enumerate(self.graph_containers)
            if not container.plot_manager.has_data()
        }

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
        
        # Apply saved graph settings after redrawing signals
        self._apply_saved_graph_settings()

        # View ranges. The X axes of a tab are linked to its first plot.
        # - Y: every plot with signals is fitted to its own data.
        # - X: kept as it was (the user's zoom/pan), except in a tab that had
        #   no signal before, where it is fitted to all data of the tab.
        # Empty plots are left alone: ranging them would zoom onto the cursors.
        refitted_tabs = set()
        for tab_index, container in enumerate(self.graph_containers):
            plot_manager = container.plot_manager
            plot_widgets = plot_manager.get_plot_widgets()
            if not plot_widgets:
                continue
            saved_x_range = plot_widgets[0].getViewBox().viewRange()[0]
            for plot_widget in plot_widgets:
                plot_widget.enableAutoRange(axis='x', enable=False)
                plot_widget.enableAutoRange(axis='y', enable=False)
            plot_manager.fit_y_to_data()
            if tab_index in tabs_without_data and plot_manager.has_data():
                plot_manager.fit_x_to_data()
                refitted_tabs.add(tab_index)
            else:
                plot_widgets[0].setXRange(saved_x_range[0], saved_x_range[1], padding=0)

        # Restore cursors after redrawing signals. Where the X range was just
        # fitted to new data the old positions are meaningless (they were
        # placed on an empty plot), so the cursors start at 1/3 and 2/3 again.
        if current_mode and current_mode != "none":
            if self.tab_widget.currentIndex() in refitted_tabs:
                cursor_positions = {}
            # Use a timer to ensure plots are fully ready before restoring cursors
            QTimer.singleShot(50, lambda: self._restore_cursors_after_redraw(current_mode, cursor_positions))

        logger.debug("Auto-ranged all plots (X and Y axes) after signal redraw, settings apply, and limit lines")
        logger.info(f"[PERF] ========== _redraw_all_signals() TOTAL: {(time.time()-start_time)*1000:.1f}ms ==========")
        
        # Update statistics panel after redrawing signals
        self._recreate_statistics_panel()
        # DON'T update statistics here - wait for cursor movement for performance
        
        # Update correlations panel with new parameters
        if hasattr(self, 'correlations_panel_manager') and self.correlations_panel_manager:
            self.correlations_panel_manager.update_available_parameters(all_signal_names)
            self.correlations_panel_manager.on_data_changed()
        
        # A range filter already replaced the signal data drawn above, so it is
        # not re-applied here. Zoom-based (LOD) reloading must stay off while
        # it is active, otherwise zooming would reload unfiltered file data.
        filter_active = self.filter_manager.has_active_filters()
        for container in self.graph_containers:
            container.plot_manager._lod_disabled = filter_active
        if hasattr(self, 'toolbar_manager') and self.toolbar_manager:
            self.toolbar_manager.set_filter_active(filter_active)
        if hasattr(self, 'bitmask_panel_manager') and self.bitmask_panel_manager:
            self.bitmask_panel_manager.set_filter_active(filter_active)
        
        logger.debug("Redrew all signals across all tabs.")
    
    def clear_active_filters(self):
        """Clear the range filter and show the original data again."""
        self._apply_range_filter({'conditions': []})

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
    
    def _refresh_statistics(self, cursor_positions: dict):
        """Recompute statistics for the given cursor positions right away."""
        if getattr(self, '_suppress_statistics_updates', False):
            QTimer.singleShot(100, lambda: self._refresh_statistics(cursor_positions))
            return
        self._pending_cursor_positions = cursor_positions
        self._perform_statistics_update()

    def _enable_statistics_updates_after_redraw(self):
        """Re-enable statistics updates after redraw operation completes."""
        self._suppress_statistics_updates = False
        logger.debug("[PERF] Statistics updates re-enabled after redraw")

    def _restore_cursor_positions(self, positions: dict):
        """Restore specific cursor positions."""
        try:
            if not self.cursor_manager or not positions:
                return
                
            # CursorManager.get_cursor_positions() returns 'c1'/'c2'
            c1 = positions.get('c1', positions.get('cursor1'))
            c2 = positions.get('c2', positions.get('cursor2'))
            if c1 is not None:
                self.cursor_manager.set_cursor_position("dual_1", c1)
            if c2 is not None:
                self.cursor_manager.set_cursor_position("dual_2", c2)
                
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
    
    def _on_filters_requested(self):
        """Open the Advanced Settings (range filter) dialog; applies to all graphs."""
        all_signals_data = self.signal_processor.get_all_signals()
        all_signals = list(all_signals_data.keys()) if all_signals_data else []
        if not all_signals:
            QMessageBox.information(self, "Range Filters", "Load a data file first.")
            return

        dialog = GraphAdvancedSettingsDialog(
            all_signals, self.filter_manager.get_global_filter(), None  # None: own taskbar entry
        )
        dialog.setWindowIcon(self.windowIcon() if self.windowIcon() else QIcon())

        # Center dialog on parent window
        if self.parent():
            parent_geometry = self.parent().geometry()
            dialog.move(
                parent_geometry.center().x() - dialog.width() // 2,
                parent_geometry.center().y() - dialog.height() // 2
            )

        dialog.range_filter_applied.connect(self._apply_range_filter)
        dialog.exec_()

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
        
        # Create styled input dialog
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Sekmeyi Yeniden Adlandır")
        dialog.setLabelText("Yeni sekme adı:")
        dialog.setTextValue(old_name)
        dialog.setInputMode(QInputDialog.TextInput)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QInputDialog {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5ba0f2;
            }
            QPushButton:pressed {
                background-color: #3a80d2;
            }
        """)
        
        ok = dialog.exec_()
        new_name = dialog.textValue()
        
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
        # CRITICAL: Skip statistics updates if suppressed (during graph redraw)
        if getattr(self, '_suppress_statistics_updates', False):
            if hasattr(self, 'statistics_panel') and self.statistics_panel:
                self.statistics_panel.update_cursor_positions(cursor_positions)
            return

        # STRICT THROTTLING: Limit to ~30 FPS (33ms). Events inside the window
        # are deferred, not dropped: otherwise the last position of a drag
        # could be skipped and the statistics would stay at an older position
        current_time_ms = time.time() * 1000
        if current_time_ms - getattr(self, '_last_cursor_event_time', 0) < 33:
            self._pending_cursor_positions = cursor_positions
            if hasattr(self, 'statistics_panel') and self.statistics_panel:
                self.statistics_panel.update_cursor_positions(cursor_positions)
            if not self._statistics_update_timer.isActive():
                self._statistics_update_timer.setInterval(50)
                self._statistics_update_timer.start()
            return
        self._last_cursor_event_time = current_time_ms
        
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
        """
        Apply the range filter to all graphs, or clear it (empty conditions).

        The filter always uses concatenated display: the time ranges where all
        conditions hold are joined into one continuous timeline, the signal
        processor's data is replaced with it and every graph in every tab is
        redrawn. Segments are computed from the original file data, so a new
        filter replaces the previous one instead of filtering its result.
        """
        conditions = filter_data.get('conditions', [])
        logger.info(f"[FILTER] Range filter requested: {len(conditions)} condition(s)")

        if not conditions:
            if self.filter_manager.has_active_filters():
                self.filter_manager.clear_filters()
                self.signal_processor.restore_original_data()
                self._save_filter_state_for_file_switch()
                self._redraw_after_filter_change()
                logger.info("[FILTER] Range filter cleared, original data restored")
            return

        if hasattr(self, 'loading_manager') and self.loading_manager:
            self.loading_manager.start_operation("filtering", "Calculating filter segments...")

        def on_segments_calculated(time_segments):
            try:
                if hasattr(self, 'loading_manager') and self.loading_manager:
                    self.loading_manager.finish_operation("filtering")

                if time_segments is None:
                    return  # debounced

                if not time_segments:
                    # Keep whatever was shown before (previous filter or none)
                    msg = QMessageBox(self)
                    msg.setStyleSheet(self._get_message_box_style())
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("No time segments match the specified filter conditions.\n"
                                "The filter was not applied.")
                    msg.setWindowTitle("No Matches")
                    msg.exec_()
                    return

                # Build the concatenated data from the original data
                if self.filter_manager.has_active_filters():
                    self.signal_processor.restore_original_data()
                self.graph_renderer.graph_signal_mapping = self.graph_signal_mapping
                self.graph_renderer.apply_concatenated_filter(None, time_segments)

                self.filter_manager.set_global_filter(filter_data)
                self._save_filter_state_for_file_switch()
                self._redraw_after_filter_change()
                logger.info(f"[FILTER] Range filter applied to all graphs ({len(time_segments)} segments)")

            except RuntimeError as e:
                logger.warning(f"Filter callback failed - widget may be deleted: {e}")
            except Exception as e:
                logger.error(f"Error applying range filter: {e}", exc_info=True)
                msg = QMessageBox(self)
                msg.setStyleSheet(self._get_message_box_style())
                msg.setIcon(QMessageBox.Critical)
                msg.setText(f"Error applying filter: {str(e)}")
                msg.setWindowTitle("Filter Error")
                msg.exec_()

        try:
            self.filter_manager.calculate_filter_segments_threaded(
                self.signal_processor.get_all_signals(),
                conditions,
                on_segments_calculated,
            )
        except Exception as e:
            logger.error(f"Error starting range filter calculation: {e}", exc_info=True)
            if hasattr(self, 'loading_manager') and self.loading_manager:
                self.loading_manager.finish_operation("filtering")

    def _save_filter_state_for_file_switch(self):
        """Keep the widget container's copy of the filter state up to date."""
        if hasattr(self, 'parent') and hasattr(self.parent, 'widget_container_manager'):
            self.parent.widget_container_manager.save_current_filter_state()

    def _redraw_after_filter_change(self):
        """Redraw every tab after the filter replaced/restored the time axis."""
        self._redraw_all_signals()
        # The timeline changed (concatenated or original), so the previous
        # zoom is meaningless: fit every plot to the new data
        for container in self.graph_containers:
            for plot_widget in container.plot_manager.get_plot_widgets():
                plot_widget.enableAutoRange(axis='x', enable=True)
                plot_widget.enableAutoRange(axis='y', enable=True)
                plot_widget.autoRange()
                plot_widget.enableAutoRange(axis='x', enable=False)
                plot_widget.enableAutoRange(axis='y', enable=False)

    def _restore_filter_ui_state(self, saved_filters: dict):
        """
        Called by the widget container manager after restoring filter state.
        Nothing to rebuild: the filter dialog reads the state when opened.
        """
        if hasattr(self, 'toolbar_manager') and self.toolbar_manager:
            self.toolbar_manager.set_filter_active(self.filter_manager.has_active_filters())
        if hasattr(self, 'bitmask_panel_manager') and self.bitmask_panel_manager:
            self.bitmask_panel_manager.set_filter_active(self.filter_manager.has_active_filters())

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
        # OPTIMIZATION: Legend panel is currently not visible in the UI, 
        # so updating it consumes resources (67ms/frame) for no benefit.
        # Disabling updates until the panel is added to the layout.
        return

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

        self.toolbar_manager.filters_requested.connect(self._on_filters_requested)

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
        # REMOVED: self.signal_processor.processing_finished.connect(self._on_processing_finished) 
        # CAUSE OF INFINITE LOOP: The worker triggers this, we don't need the processor to trigger it again recursively.
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
        
        count = active_container.plot_manager.get_subplot_count()
        if from_index == to_index or not (0 <= from_index < count and 0 <= to_index < count):
            return

        # The plot widgets stay where they are; their CONTENTS are swapped.
        # Everything per graph (callbacks, axis labels, cursors, statistics,
        # legends) is addressed by graph index, so swapping the per-index
        # state and redrawing keeps all of it consistent. Moving the widgets
        # instead left those index-bound parts pointing at the wrong graph.
        def swap(store: dict):
            if store is None:
                return
            a, b = store.pop(from_index, None), store.pop(to_index, None)
            if b is not None:
                store[from_index] = b
            if a is not None:
                store[to_index] = a

        swap(self.graph_signal_mapping.setdefault(tab_index, {}))
        swap(self.graph_settings.get(tab_index))

        swap(active_container.plot_manager.original_data_ranges)

        self.is_data_modified = True

        cursor_positions = (self.cursor_manager.get_cursor_positions()
                            if self.cursor_manager else {})

        # Redraws signals, legends, limit lines and filters, restores the
        # cursors and rebuilds the statistics panel
        self._redraw_all_signals()

        # The rebuilt statistics panel is empty until the next cursor move;
        # the values existed before the swap, so recompute them once the
        # redraw re-enables statistics (~350 ms, see _restore_cursors_after_redraw)
        if cursor_positions:
            QTimer.singleShot(450, lambda: self._refresh_statistics(cursor_positions))

        logger.info(f"Graphs reordered successfully: Graph {from_index + 1} <-> Graph {to_index + 1}")

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
                            logger.info(f"[LAYOUT] Tab {i}, Graph {plot_index}: Restoring signals {graph_config['signals']}")
                            logger.info(f"[LAYOUT] Available signals: {len(all_signals)} signals")
                            
                            for signal_name in graph_config['signals']:
                                if signal_name in all_signals:
                                    # Sinyal verilerini signal_processor'dan al
                                    signal_data = self.signal_processor.get_signal_data(signal_name)
                                    
                                    x_data = None
                                    y_data = None
                                    
                                    # Check if data is directly available (CSV mode)
                                    if signal_data and 'x_data' in signal_data and 'y_data' in signal_data:
                                        x_data = signal_data['x_data']
                                        y_data = signal_data['y_data']
                                    # Otherwise, try to get downsampled view (MPAI mode)
                                    elif signal_data and 'mpai_reader' in signal_data:
                                        logger.debug(f"[LAYOUT] Getting downsampled view for memory-mapped signal '{signal_name}'")
                                        ds_view = self.signal_processor.get_downsampled_view(signal_name)
                                        if ds_view and 'x_data' in ds_view and 'y_data' in ds_view:
                                            x_data = ds_view['x_data']
                                            y_data = ds_view['y_data']
                                    
                                    if x_data is not None and y_data is not None:
                                        # Sinyali belirtilen grafiğe ekle
                                        container.add_signal(signal_name, x_data, y_data, plot_index)
                                        logger.info(f"[LAYOUT] Added signal '{signal_name}' to graph {plot_index}")
                                        
                                        # Signal mapping'i güncelle
                                        if i not in self.graph_signal_mapping:
                                            self.graph_signal_mapping[i] = {}
                                        if plot_index not in self.graph_signal_mapping[i]:
                                            self.graph_signal_mapping[i][plot_index] = []
                                        if signal_name not in self.graph_signal_mapping[i][plot_index]:
                                            self.graph_signal_mapping[i][plot_index].append(signal_name)
                                    else:
                                        logger.warning(f"[LAYOUT] Could not get data for signal '{signal_name}'")
                                else:
                                    logger.warning(f"[LAYOUT] Signal '{signal_name}' not found in signal_processor")
                    
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
