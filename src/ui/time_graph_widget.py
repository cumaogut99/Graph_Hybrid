# type: ignore
"""
Advanced Time Graph Widget - Refactored Main Class

Professional architecture with separation of concerns for maintainability and performance.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea, QLabel, QDialog, QGroupBox, 
    QTabWidget, QGridLayout, QStackedWidget, QToolButton,
    QTabBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QObject, QThread

# Import our modular components - Always use absolute imports for standalone app
from src.managers.filter_manager import FilterManager
from src.graphics.graph_renderer import GraphRenderer

from src.managers.toolbar_manager import ToolbarManager
from src.managers.plot_manager import PlotManager
from src.managers.legend_manager import LegendManager
from src.data.signal_processor import SignalProcessor
from src.managers.theme_manager import ThemeManager
from src.ui.statistics_panel import StatisticsPanel
from src.managers.data_manager import TimeSeriesDataManager
from src.managers.settings_panel_manager import SettingsPanelManager
from src.managers.statistics_settings_panel_manager import StatisticsSettingsPanelManager
from src.managers.graph_settings_panel_manager import GraphSettingsPanelManager
from src.managers.parameters_panel_manager import ParametersPanelManager
from src.managers.bitmask_panel_manager import BitmaskPanelManager
from src.ui.graph_settings_dialog import GraphSettingsDialog
from src.managers.status_bar_manager import StatusBarManager
from src.graphics.loading_overlay import LoadingManager
from src.managers.correlations_panel_manager import CorrelationsPanelManager
from src.ui.time_graph.tabs import TabsMixin
from src.ui.time_graph.cursor_statistics import CursorStatisticsMixin
from src.ui.time_graph.graph_settings import GraphSettingsMixin
from src.ui.time_graph.filters import FiltersMixin
from src.ui.time_graph.panels import PanelsMixin
from src.ui.time_graph.theme import ThemeMixin
from src.ui.time_graph.layout_config import LayoutConfigMixin


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


class TimeGraphWidget(TabsMixin, CursorStatisticsMixin, GraphSettingsMixin, FiltersMixin, PanelsMixin, ThemeMixin, LayoutConfigMixin, QWidget):
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
        main_layout.setSpacing(0)  # no gap between title bar and tab bar
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

    def _initialize_signal_mapping(self, signal_names: List[str]):
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
        
        # Prepare data only for the plotted signals (files can have 1000+ channels)
        all_signal_names = sorted(self.signal_processor.get_signal_names())
        plotted_names = {
            name
            for tab_mapping in self.graph_signal_mapping.values()
            for signal_names in tab_mapping.values()
            for name in signal_names
        }
        all_signals = self.signal_processor.get_all_signals(list(plotted_names))
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
            # "Auto-scale Y-Axis" setting: keep following the data from here
            plot_manager.set_y_autoscale(
                self.graph_settings_panel_manager.global_settings.get('autoscale', True))
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
    
    def _on_report_requested(self):
        """Open the report window for this file's data (one window per file)."""
        if not self.signal_processor.signal_data:
            QMessageBox.information(self, "Report", "Load a data file first.")
            return
        dialog = getattr(self, '_report_dialog', None)
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return

        from src.ui.report_dialog import ReportDialog
        file_name = ""
        file_manager = getattr(self.window(), 'file_manager', None)
        if file_manager is not None:
            active = file_manager.get_active_file_data() or {}
            file_name = active.get('filename', '')
        # Own top-level window (taskbar entry); recreated so new parameters appear
        self._report_dialog = ReportDialog(self, file_name)
        self._report_dialog.setWindowIcon(self.windowIcon())
        self._report_dialog.show()

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
    
    # NOTE: _on_cursor_mode_changed removed - cursor mode is now permanently 'dual'
    # Cursor mode is initialized to 'dual' on startup and never changes
    
    # Data processing methods
    def _apply_normalization(self):
        """Apply normalization to all signals."""
        signal_names = self.signal_processor.get_signal_names()
        normalized_data = self.signal_processor.apply_normalization(signal_names)
        
        for signal_name, y_data in normalized_data.items():
            signal_data = self.signal_processor.get_signal_data(signal_name)
            if signal_data and 'x_data' in signal_data:
                self.plot_manager.update_signal_data(signal_name, signal_data['x_data'], y_data, 0)
        
        self._update_legend_values()
    
    def _remove_normalization(self):
        """Remove normalization from all signals."""
        signal_names = self.signal_processor.get_signal_names()
        original_data = self.signal_processor.remove_normalization(signal_names)
        
        for signal_name, y_data in original_data.items():
            signal_data = self.signal_processor.get_signal_data(signal_name)
            if signal_data and 'x_data' in signal_data:
                self.plot_manager.update_signal_data(signal_name, signal_data['x_data'], y_data, 0)
        
        self._update_legend_values()

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
        self.toolbar_manager.report_requested.connect(self._on_report_requested)

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
