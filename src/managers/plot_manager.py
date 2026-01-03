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
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
import pyqtgraph as pg
<<<<<<< HEAD
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QAction, QMenu, QFrame
=======
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QAction, QMenu
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QColor

# ✅ REFACTORED: Import from modular plot package
from .plot import DateTimeAxisItem, PlotTooltipsHelper, PlotSecondaryAxisHelper
from ..graphics.lod_engine import LODEngine

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class PlotManager(QObject):
    """Manages the plotting interface for the Time Graph Widget."""
    
    # Signals
    plot_clicked = Signal(int, float, float)  # plot_index, x, y
    range_selected = Signal(float, float)  # start, end
    settings_requested = Signal(int)
    
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
            'snap_to_data': False,
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

    def _on_view_changed(self, plot_index: int, ranges):
        """
        Handle view range changes for LOD (Level of Detail) optimization.
        
        Dynamic LOD Switch:
        - RAW mode: sample_count < pixel_width * 2 → fetch exact raw data
        - AGGREGATED mode: sample_count >= pixel_width * 2 → use C++ downsampler
        """
        # Debounce rapid zoom/pan events
        if self._lod_timer is not None:
            self._lod_timer.stop()
        
        self._lod_pending_ranges[plot_index] = ranges
        
        from PyQt5.QtCore import QTimer
        self._lod_timer = QTimer()
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._process_pending_lod_updates)
        self._lod_timer.start(50)  # 50ms debounce for responsive zoom
    
    def _process_pending_lod_updates(self):
        """
        Dynamic LOD Switch - the core of view-dependent visualization.
<<<<<<< HEAD

=======
        
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        For each signal in the visible range:
        - Calculate sample count in visible range
        - IF sample_count < pixel_width * 2: RAW DIRECT mode (exact data from Arrow)
        - IF sample_count >= pixel_width * 2: AGGREGATED mode (C++ downsampler)
        """
<<<<<<< HEAD
        # ✅ CRITICAL FIX: Skip LOD updates when filter is active
        # Segmented/concatenated filters provide pre-filtered PlotDataItems
        # LOD would reload full dataset from MPAI, overriding the filtered view!
        if getattr(self, '_lod_disabled', False):
            logger.debug("[LOD] Skipping LOD update - filter is active (_lod_disabled=True)")
            self._lod_pending_ranges.clear()
            return

        import time
        start_time = time.time()

=======
        import time
        start_time = time.time()
        
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        if not self._lod_pending_ranges:
            return
        
        signal_processor = self._get_signal_processor()
        if signal_processor is None:
            self._lod_pending_ranges.clear()
            return
        
        # Get pixel width from first plot widget
        pixel_width = 1920  # Default
        if self.plot_widgets:
            try:
                pixel_width = self.plot_widgets[0].width()
            except:
                pass
        
        # Threshold: RAW if samples < 10x pixel width (aggressive raw switching)
        # 10x ratio allows ~20,000 points for typical 2000px viewport
        raw_threshold = pixel_width * 10
        
        # Process each plot's pending range
        for plot_index, ranges in self._lod_pending_ranges.items():
            if plot_index >= len(self.plot_widgets):
                continue
            
            x_range = ranges[0]  # (x_min, x_max)
            view_x_min, view_x_max = x_range
            
            # Get signals on this plot
            signals_on_plot = [
                (key, item) for key, item in self.current_signals.items()
                if key.endswith(f"_{plot_index}")
            ]
            
            for signal_key, plot_item in signals_on_plot:
                signal_name = signal_key.rsplit('_', 1)[0]
                
                # Get signal info from processor
                signal_info = signal_processor.signal_data.get(signal_name)
                if signal_info is None:
                    continue
                
                metadata = signal_info.get('metadata', {})
                is_memory_mapped = metadata.get('memory_mapped', False)
                
                if not is_memory_mapped:
                    # CSV data: already in memory, PyQtGraph handles it
                    continue
                
                # ========== MEMORY-MAPPED MPAI: Dynamic LOD Switch ==========
                reader = signal_info.get('mpai_reader')
                if reader is None:
                    continue
                
                col_name = signal_info.get('column_name', signal_name)
                time_col = signal_info.get('time_column', 'time')
                row_count = signal_info.get('row_count', 0)
                time_range = signal_info.get('time_range', (0.0, 0.0))
                
                if row_count == 0 or time_range[1] <= time_range[0]:
                    continue
                
                # Calculate sample count in visible range
                total_time_span = time_range[1] - time_range[0]
                if total_time_span <= 0:
                    continue
                
                visible_time_span = max(0, min(view_x_max, time_range[1]) - max(view_x_min, time_range[0]))
                if visible_time_span <= 0:
                    continue
                
                visible_fraction = visible_time_span / total_time_span
                estimated_sample_count = int(row_count * visible_fraction)
                
                # ========== LOD MODE DECISION WITH HYSTERESIS ==========
                # Get current mode for this signal (default to AGGREGATED for safety)
                current_mode = self._current_lod_modes.get(signal_key, "AGGREGATED")
                
                # Apply hysteresis to prevent flickering at threshold boundary
                if current_mode == "AGGREGATED":
                    # Currently in AGGREGATED mode: need to go well BELOW threshold to switch to RAW
                    if estimated_sample_count < raw_threshold * self._lod_hysteresis_down:
                        lod_mode = "RAW"
                    else:
                        lod_mode = "AGGREGATED"
                else:  # current_mode == "RAW"
                    # Currently in RAW mode: need to go well ABOVE threshold to switch to AGGREGATED
                    if estimated_sample_count > raw_threshold * self._lod_hysteresis_up:
                        lod_mode = "AGGREGATED"
                    else:
                        lod_mode = "RAW"
                
                # Update mode cache
                self._current_lod_modes[signal_key] = lod_mode
                
                # Set target points based on mode
                if lod_mode == "RAW":
                    target_points = estimated_sample_count  # All samples
                else:
                    target_points = pixel_width * 2  # 2 points per pixel (min/max)
                
                logger.info(f"[LOD] {signal_name}: mode={lod_mode}, samples={estimated_sample_count:,}, target={target_points:,}, range=[{view_x_min:.2f}, {view_x_max:.2f}]")
                
                # Fetch data with appropriate resolution
                new_x, new_y = self._fetch_lod_data(
                    reader, time_col, col_name,
                    view_x_min, view_x_max,
                    row_count, time_range,
                    target_points, lod_mode
                )
                
                if new_x is not None and len(new_x) > 0:
                    # Update plot item with new data
                    try:
                        plot_item.setData(new_x, new_y)
                        logger.debug(f"[LOD] Updated {signal_name} with {len(new_x)} points")
                    except Exception as e:
                        logger.error(f"[LOD] Failed to update {signal_name}: {e}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[LOD] Update complete in {elapsed_ms:.1f}ms")
        
        self._lod_pending_ranges.clear()
    
    def _fetch_lod_data(self, reader, time_col: str, signal_col: str,
                        view_x_min: float, view_x_max: float,
                        row_count: int, time_range: tuple,
                        target_points: int, lod_mode: str):
        """
        Fetch data at appropriate LOD resolution.
        
        Args:
            reader: MPAI reader instance
            time_col: Time column name
            signal_col: Signal column name
            view_x_min, view_x_max: Visible time range
            row_count: Total row count
            time_range: Full data time range (min, max)
            target_points: Target output points
            lod_mode: "RAW" or "AGGREGATED"
        
        Returns:
            (x_data, y_data) numpy arrays
        """
        try:
            # Calculate row range from time range
            total_time = time_range[1] - time_range[0]
            if total_time <= 0:
                return None, None
            
            # Clamp to data range
            clamp_x_min = max(view_x_min, time_range[0])
            clamp_x_max = min(view_x_max, time_range[1])
            
            # Estimate row indices
            start_fraction = (clamp_x_min - time_range[0]) / total_time
            end_fraction = (clamp_x_max - time_range[0]) / total_time
            
            start_row = max(0, int(start_fraction * row_count))
            end_row = min(row_count, int(end_fraction * row_count) + 1)
            slice_count = end_row - start_row
            
            if slice_count <= 0:
                return None, None
            
            if lod_mode == "RAW":
                # ✅ RAW MODE: Load exact samples from Arrow slice
                x_data = np.array(reader.load_column_slice(time_col, start_row, slice_count), dtype=np.float64)
                y_data = np.array(reader.load_column_slice(signal_col, start_row, slice_count), dtype=np.float64)
                
                # Filter to exact view range (row estimate may be imprecise)
                mask = (x_data >= view_x_min) & (x_data <= view_x_max)
                return x_data[mask], y_data[mask]
            
            else:
                # ✅ AGGREGATED MODE: Use pre-computed LOD files if available
                try:
                    # Try LOD parquet files first (FAST PATH: ~10ms vs 6000ms)
                    from src.data.lod_reader import (
                        get_lod_container_path, 
                        get_available_lod_levels,
                        select_lod_level,
                        load_lod_data
                    )
                    
                    # Get MPAI path from signal_processor
                    signal_processor = self._get_signal_processor()
                    mpai_path = getattr(signal_processor, 'current_mpai_path', None)
                    
                    if mpai_path:
                        available = get_available_lod_levels(mpai_path)
                        
                        if available:
                            # Calculate visible samples to select LOD level
                            visible_samples = int(row_count * (view_x_max - view_x_min) / (time_range[1] - time_range[0]))
                            lod_level = select_lod_level(visible_samples, available)
                            
                            if lod_level:
                                import time as perf_time
                                t_start = perf_time.perf_counter()
                                
                                x_data, y_data = load_lod_data(
                                    mpai_path, lod_level, signal_col,
                                    view_x_min, view_x_max
                                )
                                
                                if x_data is not None and len(x_data) > 0:
                                    elapsed = (perf_time.perf_counter() - t_start) * 1000
                                    logger.info(f"[LOD-FAST] {signal_col}: {lod_level} → {len(x_data)} points in {elapsed:.1f}ms")
                                    return x_data, y_data
                                else:
                                    logger.warning(f"[LOD] LOD data empty for {signal_col}, falling back to streaming")
                
                except ImportError as e:
                    logger.warning(f"[LOD] LOD reader not available: {e}")
                except Exception as e:
                    logger.warning(f"[LOD] LOD parquet failed: {e}, falling back to streaming")
                
                # FALLBACK: Use C++ streaming downsampler
                try:
<<<<<<< HEAD
                    # NEW: Check for MpaiDirectoryReader (Python-based Zero-Copy Reader)
                    # It has a built-in optimized get_render_data method
                    if hasattr(reader, 'get_render_data') and hasattr(reader, 'name_to_id'):
                        
                        # Resolve Channel ID
                        ch_id = reader.name_to_id.get(signal_col)
                        if ch_id is not None:
                            # Pixel width is roughly target_points / 2 (since target is usually 2x width)
                            pixel_width = max(1, target_points // 2)
                            
                            x_data, y_data, src_type = reader.get_render_data(
                                ch_id, 
                                view_x_min, view_x_max, 
                                pixel_width
                            )
                            
                            if len(x_data) > 0:
                                # logger.debug(f"[LOD-DIR] Used MpaiDirectoryReader ({src_type}) for {signal_col}")
                                return x_data, y_data
                    
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                    import time_graph_cpp
                    
                    config = time_graph_cpp.SmartDownsampleConfig()
                    config.target_points = target_points
                    config.use_lttb = True
                    config.detect_local_extrema = True
                    
                    # Use the C++ streaming downsampler with time range
                    result = time_graph_cpp.downsample_minmax_streaming(
                        reader, time_col, signal_col,
                        target_points,
                        float('nan'), float('nan')  # No warning thresholds
                    )
                    
                    x_data = np.array(result['time'], dtype=np.float64)
                    y_data = np.array(result['value'], dtype=np.float64)
                    
                    # Filter to view range
                    mask = (x_data >= view_x_min) & (x_data <= view_x_max)
                    return x_data[mask], y_data[mask]
                    
                except Exception as e:
                    logger.error(f"[LOD] C++ downsample failed: {e}, falling back to Python")
                    # Fallback: Python min-max
                    return self._python_minmax_downsample(
                        reader, time_col, signal_col,
                        start_row, slice_count, target_points
                    )
                    
        except Exception as e:
            logger.error(f"[LOD] _fetch_lod_data failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _python_minmax_downsample(self, reader, time_col: str, signal_col: str,
                                   start_row: int, slice_count: int, target_points: int):
        """Python fallback for min-max downsampling."""
        try:
            # Load slice
            x_data = np.array(reader.load_column_slice(time_col, start_row, slice_count), dtype=np.float64)
            y_data = np.array(reader.load_column_slice(signal_col, start_row, slice_count), dtype=np.float64)
            
            if len(x_data) <= target_points:
                return x_data, y_data
            
            # Min-max bucketing
            num_buckets = target_points // 2
            bucket_size = len(x_data) // num_buckets
            
            x_out = []
            y_out = []
            
            for i in range(num_buckets):
                start = i * bucket_size
                end = min((i + 1) * bucket_size, len(x_data))
                if start >= end:
                    continue
                
                x_bucket = x_data[start:end]
                y_bucket = y_data[start:end]
                
                min_idx = np.argmin(y_bucket)
                max_idx = np.argmax(y_bucket)
                
                if min_idx < max_idx:
                    x_out.extend([x_bucket[min_idx], x_bucket[max_idx]])
                    y_out.extend([y_bucket[min_idx], y_bucket[max_idx]])
                else:
                    x_out.extend([x_bucket[max_idx], x_bucket[min_idx]])
                    y_out.extend([y_bucket[max_idx], y_bucket[min_idx]])
            
            return np.array(x_out), np.array(y_out)
            
        except Exception as e:
            logger.error(f"[LOD] Python fallback failed: {e}")
            return None, None
    
    def _get_lod_engine(self):
        """Get or create LOD Engine instance."""
        if self._lod_engine is None:
            signal_processor = self._get_signal_processor()
            if signal_processor is not None:
                self._lod_engine = LODEngine(signal_processor)
        return self._lod_engine

    
    def _setup_plot_panel(self):
        """Create the main plot panel and its permanent layout."""
        self.plot_panel = QWidget()
        self.main_layout = QVBoxLayout(self.plot_panel)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(2)

    def set_subplot_count(self, count: int):
        """Set the number of subplots with safe error handling."""
<<<<<<< HEAD
        print(f"[ZOOM] === set_subplot_count START === current={self.subplot_count}, new={count}")
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        if not (self.min_subplots <= count <= self.max_subplots):
            logger.warning(f"Invalid subplot count: {count}. Must be between {self.min_subplots} and {self.max_subplots}")
            return False
        
        if count == self.subplot_count:
<<<<<<< HEAD
            print(f"[ZOOM] Same count, returning early")
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            return True
        
        try:
            # Store signal data before rebuilding UI
            signal_data_to_restore = self._get_signal_data_for_restore()
<<<<<<< HEAD
            print(f"[ZOOM] Signal data to restore: {list(signal_data_to_restore.keys())}")
            
            # CRITICAL FIX: Save current X-range before rebuilding to prevent zoom jumping
            saved_x_range = None
            if self.plot_widgets and len(self.plot_widgets) > 0:
                try:
                    saved_x_range = self.plot_widgets[0].getViewBox().viewRange()[0]
                    print(f"[ZOOM] Step 1: SAVED X-range = {saved_x_range}")
                except Exception as e:
                    print(f"[ZOOM] Could not save X-range: {e}")
            else:
                print(f"[ZOOM] No plot widgets exist, cannot save X-range")
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            
            # Update count and rebuild the entire UI safely
            self.subplot_count = count
            self._rebuild_ui()
            
<<<<<<< HEAD
            # Log after rebuild
            if self.plot_widgets:
                after_rebuild = self.plot_widgets[0].getViewBox().viewRange()[0]
                print(f"[ZOOM] Step 2: AFTER rebuild X-range = {after_rebuild}")
            
            # Restore signals to the new plots
            self._restore_signals(signal_data_to_restore)
            
            # Log after signal restore
            if self.plot_widgets:
                after_restore = self.plot_widgets[0].getViewBox().viewRange()[0]
                print(f"[ZOOM] Step 3: AFTER _restore_signals X-range = {after_restore}")
            
            # CRITICAL FIX: Restore the saved X-range to all plots after rebuild
            if saved_x_range is not None and self.plot_widgets:
                from PyQt5.QtCore import QTimer
                
                # Store for later access
                self._saved_x_range_for_restore = saved_x_range
                
                # IMMEDIATE RESTORE - do it right away first
                try:
                    self.plot_widgets[0].blockSignals(True)
                    self.plot_widgets[0].setXRange(saved_x_range[0], saved_x_range[1], padding=0)
                    self.plot_widgets[0].blockSignals(False)
                    after_immediate = self.plot_widgets[0].getViewBox().viewRange()[0]
                    print(f"[ZOOM] Step 4: IMMEDIATE restore X-range = {after_immediate} (target: {saved_x_range})")
                except Exception as e:
                    print(f"[ZOOM] IMMEDIATE restore failed: {e}")
                
                def restore_x_range_50():
                    try:
                        if self.plot_widgets and hasattr(self, '_saved_x_range_for_restore'):
                            saved = self._saved_x_range_for_restore
                            before = self.plot_widgets[0].getViewBox().viewRange()[0]
                            self.plot_widgets[0].blockSignals(True)
                            self.plot_widgets[0].setXRange(saved[0], saved[1], padding=0)
                            self.plot_widgets[0].blockSignals(False)
                            after = self.plot_widgets[0].getViewBox().viewRange()[0]
                            print(f"[ZOOM] Step 5 (50ms): before={before}, after={after}, target={saved}")
                    except Exception as e:
                        print(f"[ZOOM] 50ms restore failed: {e}")
                
                def restore_x_range_200():
                    try:
                        if self.plot_widgets and hasattr(self, '_saved_x_range_for_restore'):
                            saved = self._saved_x_range_for_restore
                            before = self.plot_widgets[0].getViewBox().viewRange()[0]
                            self.plot_widgets[0].blockSignals(True)
                            self.plot_widgets[0].setXRange(saved[0], saved[1], padding=0)
                            self.plot_widgets[0].blockSignals(False)
                            after = self.plot_widgets[0].getViewBox().viewRange()[0]
                            print(f"[ZOOM] Step 6 (200ms): before={before}, after={after}, target={saved}")
                    except Exception as e:
                        print(f"[ZOOM] 200ms restore failed: {e}")
                
                # Multiple delayed restores to catch any late events
                QTimer.singleShot(50, restore_x_range_50)
                QTimer.singleShot(200, restore_x_range_200)
            else:
                print(f"[ZOOM] Skipping restore: saved_x_range={saved_x_range}, plot_widgets={len(self.plot_widgets) if self.plot_widgets else 0}")
            
            print(f"[ZOOM] === set_subplot_count END ===")
=======
            # Restore signals to the new plots
            self._restore_signals(signal_data_to_restore)
            
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            return True
            
        except Exception as e:
            logger.error(f"Failed to change subplot count: {e}", exc_info=True)
            # Attempt to rollback is complex, better to revert to a stable state
            self.subplot_count = 1
            self._rebuild_ui()
            return False
            
    def _rebuild_ui(self):
        """
        Safely rebuilds the entire plot and settings UI from scratch.
        This is the core of the stable UI update mechanism.
        """
        # 1. Clean up existing UI components safely
        if self.plot_container is not None:
            self.plot_container.deleteLater()
            self.plot_container = None
        
        if self.settings_container is not None:
            self.settings_container.deleteLater()
            self.settings_container = None
            
        self._clear_plots()  # Clear internal lists

        # 2. Re-create the plot widgets and their layout container
        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(1)
        
        # Get global settings from parent
        global_settings = self._get_global_settings()
        
        for i in range(self.subplot_count):
            # Create custom datetime axis for bottom axis
            datetime_axis = DateTimeAxisItem(orientation='bottom')
            plot_widget = pg.PlotWidget(axisItems={'bottom': datetime_axis})
            legend = plot_widget.addLegend() # Add legend to each plot

            # Apply global legend setting
            show_legend = global_settings.get('show_legend', True)
            if legend:
                legend.setVisible(show_legend)

            # Link X-axes and manage visibility
            if i > 0 and self.plot_widgets: # Link current plot to the first one
                plot_widget.setXLink(self.plot_widgets[0])
                # CRITICAL FIX: After linking, sync the X range with the first plot
                # This ensures all linked plots start with the same X range
                try:
                    first_plot_x_range = self.plot_widgets[0].getViewBox().viewRange()[0]
                    plot_widget.setXRange(*first_plot_x_range, padding=0)
                    logger.debug(f"[FIX] Synced plot {i} X range with first plot: {first_plot_x_range}")
                except Exception as e:
                    logger.debug(f"Could not sync X range for plot {i}: {e}")
            
            # Hide X-axis for all but the last plot
            if i < self.subplot_count - 1:
                plot_widget.getAxis('bottom').setStyle(showValues=False)
            
            plot_widget.setLabel('left', f'Channel {i+1}')
            
            # Y ekseni hizalama sorunu için sabit genişlik ayarla
            left_axis = plot_widget.getAxis('left')
            left_axis.setWidth(self.y_axis_width)  # Y ekseni için sabit genişlik (piksel)
            
            # Apply global grid setting instead of hardcoded True
            show_grid = global_settings.get('show_grid', True)
            self.grid_visible = show_grid  # Update internal state
            plot_widget.showGrid(x=show_grid, y=show_grid, alpha=self.theme_colors['grid_alpha'] if show_grid else 0.0)
            plot_widget.setBackground(self.theme_colors['background'])
            
            # PERFORMANCE OPTIMIZATIONS for PyQtGraph
            # Always enable auto downsampling for performance
            plot_widget.setDownsampling(auto=True, mode='peak')  # Auto downsampling
            plot_widget.setClipToView(True)  # Only render visible data
            plot_widget.setAntialiasing(False)  # Disable antialiasing for speed
            
            axis_pen = pg.mkPen(color=self.theme_colors['axis_pen'])
            plot_widget.getAxis('left').setPen(axis_pen)
            plot_widget.getAxis('bottom').setPen(axis_pen)
            
            # Apply global autoscale setting for Y axis
            autoscale = global_settings.get('autoscale', True)
            plot_widget.enableAutoRange(axis='y', enable=autoscale)
            
            # CRITICAL FIX: Disable X-axis auto-range by default
            # This prevents unwanted X-axis zoom changes when data is added
            # We'll manually trigger autoRange() when needed (e.g., on data load)
            plot_widget.enableAutoRange(axis='x', enable=False)
            
            # Connect range changed signal for LOD (MPAI Lazy Loading)
            plot_widget.sigRangeChanged.connect(lambda _, ranges, idx=i: self._on_view_changed(idx, ranges))
            
            # Override ViewBox autoRange to use our custom reset_view() method
            # This ensures "View All" from right-click menu works properly with downsampled data
            view_box = plot_widget.getViewBox()
            view_box._original_autoRange = view_box.autoRange  # Backup original
            view_box.autoRange = lambda *args, **kwargs: self._custom_auto_range_for_plot(i, *args, **kwargs)
            
            # Setup custom context menu with "Zoom to Cursor" option
            self._setup_context_menu_for_plot(plot_widget, i)
            
            # Setup tooltips for this plot widget using global setting
            tooltips_enabled = global_settings.get('show_tooltips', False)
            self.tooltips_enabled = tooltips_enabled  # Update internal state
            self._setup_tooltip_for_plot(plot_widget, tooltips_enabled)
            
            # Setup secondary axis if enabled (use instance variable, not global_settings)
            if self.secondary_axis_enabled:
                self._setup_secondary_axis_for_plot(plot_widget, i)
            
            self.plot_widgets.append(plot_widget)
            # Add widget with stretch factor 1 to ensure equal heights
            plot_layout.addWidget(plot_widget, 1)
<<<<<<< HEAD
            
            # Add a separator line between graphs (except after the last one)
            if i < self.subplot_count - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                separator.setStyleSheet(f"background-color: {self.theme_colors.get('primary', '#4a90e2')}; min-height: 2px; max-height: 2px;")
                plot_layout.addWidget(separator)
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

        # 3. Re-create the settings buttons and their layout container
        self.settings_container = QWidget()
        self.graph_settings_layout = QHBoxLayout(self.settings_container)
        self.graph_settings_layout.setContentsMargins(5, 2, 5, 2)
        self._update_graph_settings_buttons()

        # 4. Add the new containers to the main, permanent layout
        self.main_layout.addWidget(self.plot_container)
        self.main_layout.addWidget(self.settings_container)
        
        logger.info(f"UI rebuilt with {self.subplot_count} subplots.")

    def get_subplot_count(self) -> int:
        """Returns the current number of subplots."""
        return self.subplot_count
    
    def reorder_graphs(self, from_index: int, to_index: int):
        """
        Reorder graphs by swapping positions.
        
        Args:
            from_index: Current index of the graph to move
            to_index: Target index where the graph should be moved
        """
        if not (0 <= from_index < self.subplot_count and 0 <= to_index < self.subplot_count):
            logger.warning(f"Invalid indices for reordering: from={from_index}, to={to_index}, count={self.subplot_count}")
            return
        
        if from_index == to_index:
            return
        
        logger.info(f"Reordering graphs: {from_index} -> {to_index}")
        
        # Get the layout from plot_container
        plot_layout = self.plot_container.layout()
        if not plot_layout:
            logger.error("Plot container has no layout")
            return
        
        # Swap widgets in the list
        self.plot_widgets[from_index], self.plot_widgets[to_index] = \
            self.plot_widgets[to_index], self.plot_widgets[from_index]
        
        # Remove widgets from layout
        widget_from = plot_layout.itemAt(from_index).widget()
        widget_to = plot_layout.itemAt(to_index).widget()
        
        plot_layout.removeWidget(widget_from)
        plot_layout.removeWidget(widget_to)
        
        # Re-insert widgets in swapped positions with equal stretch factor (1)
        # This ensures all graphs maintain equal heights
        if from_index < to_index:
            plot_layout.insertWidget(from_index, widget_to, 1)
            plot_layout.insertWidget(to_index, widget_from, 1)
        else:
            plot_layout.insertWidget(to_index, widget_from, 1)
            plot_layout.insertWidget(from_index, widget_to, 1)
        
        # Update X-axis linking: only the last plot should show X-axis labels
        for i, plot_widget in enumerate(self.plot_widgets):
            if i < len(self.plot_widgets) - 1:
                plot_widget.getAxis('bottom').setStyle(showValues=False)
            else:
                plot_widget.getAxis('bottom').setStyle(showValues=True)
            
            # Re-link X-axes
            if i > 0:
                plot_widget.setXLink(self.plot_widgets[0])
        
        # Update current_signals dictionary keys to reflect new positions
        # Format: "signal_name_plot_index"
        signals_to_update = {}
        for key, plot_item in list(self.current_signals.items()):
            if key.endswith(f"_{from_index}"):
                # This signal belongs to the moved graph
                new_key = key.rsplit('_', 1)[0] + f"_{to_index}"
                signals_to_update[new_key] = plot_item
            elif key.endswith(f"_{to_index}"):
                # This signal belongs to the target graph
                new_key = key.rsplit('_', 1)[0] + f"_{from_index}"
                signals_to_update[new_key] = plot_item
            else:
                # Keep other signals as is
                signals_to_update[key] = plot_item
        
        self.current_signals = signals_to_update
        
        logger.info(f"Graphs reordered successfully: {from_index} <-> {to_index}")
    
    def enable_datetime_axis(self, enable=True):
        """Enable datetime formatting for all plot x-axes."""
        self.datetime_axis_enabled = enable
        
        for plot_widget in self.plot_widgets:
            bottom_axis = plot_widget.getAxis('bottom')
            if isinstance(bottom_axis, DateTimeAxisItem):
                bottom_axis.enable_datetime_mode(enable)
                # Force axis update by triggering a repaint
                bottom_axis.picture = None  # Clear cache
                bottom_axis.update()
                plot_widget.getPlotItem().update()
                logger.debug(f"Updated DateTimeAxisItem in plot widget, datetime mode: {enable}")
        
        logger.debug(f"Datetime axis formatting {'enabled' if enable else 'disabled'}")


    def _get_signal_data_for_restore(self) -> Dict[str, Dict]:
        """Extracts data from current plot items for later restoration."""
        signal_data = {}
        for name, plot_item in self.current_signals.items():
            try:
                original_name = name.rsplit('_', 1)[0]
                if hasattr(plot_item, 'xData') and plot_item.xData is not None and \
                   hasattr(plot_item, 'yData') and plot_item.yData is not None:
                    signal_data[original_name] = {
                        'x': plot_item.xData.copy(),
                        'y': plot_item.yData.copy(),
                        'pen': plot_item.opts.get('pen', 'white')
                    }
            except Exception as e:
                logger.error(f"Failed to store signal data for '{name}': {e}")
        return signal_data

    def _restore_signals(self, signal_data: Dict[str, Dict]):
        """Restores signals to the newly created plots."""
        if not signal_data:
            return

        # Clear any existing signals first
        self.current_signals.clear()
        
        # Get the signal mapping from parent to determine which signals go to which plots
        signal_mapping = {}
        
        # ✅ FIX: PlotManager.parent is GraphContainer, not TimeGraphWidget!
        # We need to access TimeGraphWidget via parent.main_widget
        main_widget = None
        if hasattr(self.parent, 'main_widget'):
            main_widget = self.parent.main_widget
        else:
            logger.error("CRITICAL: GraphContainer does not have main_widget reference!")
            return
        
        if hasattr(main_widget, 'graph_signal_mapping'):
            # ✅ FIX: Get tab index from GraphContainer instead of main_widget
            current_tab = 0
            if hasattr(self.parent, 'tab_index'):
                current_tab = self.parent.tab_index
            else:
                logger.error("CRITICAL: GraphContainer does not have tab_index attribute!")
            
            # Get signal mapping for current tab
            tab_mapping = main_widget.graph_signal_mapping.get(current_tab, {})
            
            # Build reverse mapping: signal_name -> plot_index
            for plot_index, signals in tab_mapping.items():
                for signal_name in signals:
                    signal_mapping[signal_name] = plot_index
        else:
            logger.error("CRITICAL: TimeGraphWidget does not have graph_signal_mapping attribute!")
        
        # Restore signals based on mapping
        signal_names = list(signal_data.keys())
        restored_count = 0
        skipped_count = 0
        
        for name in signal_names:
            data = signal_data[name]
            
            # Only restore signals that have explicit mapping
            if name in signal_mapping:
                # Use saved mapping
                plot_index = signal_mapping[name]
                
                # ✅ FIX: Grafik sayısı artırıldığında sinyalleri GERİ GETİR
                # Eğer plot_index artık mevcut grafik sayısına sığıyorsa, çiz
                if plot_index < self.subplot_count:
                    # Grafik mevcut, sinyali çiz
                    try:
                        self.add_signal(name, data['x'], data['y'], plot_index, pen=data['pen'])
                        restored_count += 1
                    except Exception as e:
                        logger.error(f"Failed to restore signal '{name}' to plot {plot_index}: {e}")
                else:
                    # Grafik henüz yok, sinyali atlayıp mapping'de sakla
                    skipped_count += 1
        
<<<<<<< HEAD
        # ❌ REMOVED: reset_view() was causing X-range expansion when changing graph count
        # The X-range is now explicitly restored in set_subplot_count after this function
        # if restored_count > 0:
        #     self.reset_view()
=======
        # ✅ FIX: Grafik sayısı değiştiğinde otomatik view all yap
        if restored_count > 0:
            self.reset_view()
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        
        # Re-setup tooltips for all plots after signal restoration
        self._ensure_tooltips_after_rebuild()

    def _ensure_tooltips_after_rebuild(self):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper.ensure_tooltips_after_rebuild()

    def _clear_plots(self):
        """Clear all internal references to plot widgets and data items."""
        # ✅ REFACTORED: Delegate tooltip clearing to helper
        self.tooltips_helper.clear_tooltips()
        
        self.plot_widgets.clear()
        self.current_signals.clear()
        
        # ✅ REFACTORED: Delegate secondary axis clearing to helper
        self.secondary_axis_helper.clear_secondary_axes()
    
    def _update_graph_settings_buttons(self):
        """Update graph settings buttons based on current graph count."""
        # Clear existing buttons
        while self.graph_settings_layout.count():
            child = self.graph_settings_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        
        # Buttons removed - graph settings now accessible via statistics panel titles
    
    def _open_graph_settings(self, index: int):
        """Open settings dialog for a specific graph."""
        logger.info(f"Requesting settings for graph {index}")
        self.settings_requested.emit(index)
    
    def get_plot_panel(self) -> QWidget:
        """Get the plot panel widget."""
        return self.plot_panel
    
    def _downsample_data(self, x_data: np.ndarray, y_data: np.ndarray, max_points: int = 5000):
        """
        Downsample data for faster rendering.
        Uses intelligent downsampling to preserve visual appearance and data range.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            max_points: Maximum number of points to display
            
        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        data_len = len(x_data)
        
        # No downsampling needed
        if data_len <= max_points:
            return x_data, y_data
        
        # Calculate step size
        step = max(1, data_len // max_points)
        
        # Simple downsampling - take every Nth point
        # This preserves general shape and is fast
        downsampled_x = x_data[::step]
        downsampled_y = y_data[::step]
        
        # CRITICAL: Ensure first and last points are ALWAYS included
        # This guarantees the full data range is represented in the downsampled data
        result_x = [x_data[0]]
        result_y = [y_data[0]]
        
        # Add the downsampled middle points (skip if they're duplicates of first/last)
        for i in range(len(downsampled_x)):
            if i > 0 and i < len(downsampled_x) - 1:
                result_x.append(downsampled_x[i])
                result_y.append(downsampled_y[i])
        
        # Always add the last point if it's not already there
        if len(x_data) > 1 and (len(result_x) == 0 or x_data[-1] != result_x[-1]):
            result_x.append(x_data[-1])
            result_y.append(y_data[-1])
        
        final_x = np.array(result_x)
        final_y = np.array(result_y)
        
        logger.debug(f"Downsampled {data_len} points to {len(final_x)} points (step={step})")
        return final_x, final_y
    
    def add_signal(self, name: str, x_data: np.ndarray, y_data: np.ndarray, 
                   plot_index: int = 0, pen=None, **kwargs):
        """Add a signal to a specific plot with automatic PyQtGraph optimization."""
        import time
        start_time = time.time()
        
        if not (0 <= plot_index < len(self.plot_widgets)):
            logger.warning(f"Invalid plot index: {plot_index}")
            return None
        
        plot_widget = self.plot_widgets[plot_index]
        
        # Store original data range for proper view reset
        t1 = time.time()
        original_x_min = float(np.min(x_data))
        original_x_max = float(np.max(x_data))
        original_y_min = float(np.min(y_data))
        original_y_max = float(np.max(y_data))
        logger.info(f"[PERF] Min/Max calculation for {name}: {(time.time()-t1)*1000:.1f}ms ({len(x_data)} points)")
        
        # Create plot item WITHOUT manual downsampling
        # PyQtGraph will handle downsampling automatically with setDownsampling()
        if pen is None:
            color = self._get_next_color(len(self.current_signals))
            # PERFORMANCE: pen width=1 is fastest for PyQtGraph rendering
            pen = pg.mkPen(color=color, width=1)
        elif isinstance(pen, str):
            # Convert string color to pen with width=1
            pen = pg.mkPen(color=pen, width=1)
        
        # Store signal reference with a unique key (before axis assignment)
        signal_key = f"{name}_{plot_index}"
        
        # Determine which axis to use (left or right)
        axis_side = self._assign_signal_to_axis(signal_key, y_data, plot_index)
        
        # Plot to appropriate axis
        t2 = time.time()
        if axis_side == 'right' and plot_index in self.secondary_viewboxes:
            # Plot to secondary axis
            secondary_vb = self.secondary_viewboxes[plot_index]
            plot_item = pg.PlotDataItem(x_data, y_data, pen=pen, name=name, **kwargs)
            try:
                secondary_vb.addItem(plot_item)
                # Link X-axis to main plot
                secondary_vb.setXLink(plot_widget.getViewBox())
                
                # Manually add to legend (secondary ViewBox items don't auto-add)
                legend = plot_widget.getPlotItem().legend
                if legend:
                    legend.addItem(plot_item, name)
            except RuntimeError:
                # ViewBox already deleted, fall back to main axis
                logger.warning(f"Secondary ViewBox for plot {plot_index} was deleted, using main axis")
                plot_item = plot_widget.plot(x_data, y_data, pen=pen, name=name, **kwargs)
                axis_side = 'left'  # Update assignment
        else:
            # Plot to main axis (left)
            plot_item = plot_widget.plot(x_data, y_data, pen=pen, name=name, **kwargs)
        logger.info(f"[PERF] PyQtGraph plot() for {name}: {(time.time()-t2)*1000:.1f}ms")
        
        # PERFORMANCE: Enable PyQtGraph's automatic downsampling
        # This is much better than manual downsampling as it adapts to zoom level
        try:
            plot_item.setDownsampling(auto=True, method='peak')
            plot_item.setClipToView(True)  # Only render visible data
            logger.debug(f"Enabled PyQtGraph auto-downsampling for '{name}'")
        except Exception as e:
            logger.warning(f"Could not enable downsampling for '{name}': {e}")
        
        # Store signal reference
        self.current_signals[signal_key] = plot_item
        
        # Store the original color for this signal
        if isinstance(pen, str):
            self.signal_colors[signal_key] = pen
        elif hasattr(pen, 'color'):
            try:
                color = pen.color()
                if hasattr(color, 'name'):
                    self.signal_colors[signal_key] = color.name()
                else:
                    self.signal_colors[signal_key] = str(pen)
            except:
                self.signal_colors[signal_key] = str(pen)
        else:
            self.signal_colors[signal_key] = str(pen)
        
        # Store original data range for this plot (for proper view reset later)
        if plot_index not in self.original_data_ranges:
            self.original_data_ranges[plot_index] = {
                'x_min': original_x_min,
                'x_max': original_x_max,
                'y_min': original_y_min,
                'y_max': original_y_max
            }
        else:
            # Update to encompass all signals in this plot
            self.original_data_ranges[plot_index]['x_min'] = min(
                self.original_data_ranges[plot_index]['x_min'], original_x_min
            )
            self.original_data_ranges[plot_index]['x_max'] = max(
                self.original_data_ranges[plot_index]['x_max'], original_x_max
            )
            self.original_data_ranges[plot_index]['y_min'] = min(
                self.original_data_ranges[plot_index]['y_min'], original_y_min
            )
            self.original_data_ranges[plot_index]['y_max'] = max(
                self.original_data_ranges[plot_index]['y_max'], original_y_max
            )
        
        logger.debug(f"Added signal '{name}' to plot {plot_index} with color: {self.signal_colors[signal_key]}")
        logger.info(f"[PERF] TOTAL add_signal({name}): {(time.time()-start_time)*1000:.1f}ms")
        return plot_item
    
    def remove_signal(self, name: str, plot_index: int = None):
        """Remove a signal from plots."""
        if plot_index is not None:
            # Remove from specific plot
            signal_key = f"{name}_{plot_index}"
            if signal_key in self.current_signals:
                plot_item = self.current_signals[signal_key]
                if hasattr(plot_item, 'getViewBox'):
                    plot_item.getViewBox().removeItem(plot_item)
                del self.current_signals[signal_key]
                # Remove color info as well
                if signal_key in self.signal_colors:
                    del self.signal_colors[signal_key]
        else:
            # Remove from all plots
            keys_to_remove = [key for key in self.current_signals.keys() if key.startswith(f"{name}_")]
            for key in keys_to_remove:
                plot_item = self.current_signals[key]
                if hasattr(plot_item, 'getViewBox'):
                    plot_item.getViewBox().removeItem(plot_item)
                del self.current_signals[key]
                # Remove color info as well
                if key in self.signal_colors:
                    del self.signal_colors[key]
    
    def clear_all_signals(self):
        """Clear all signals from all plots while preserving tooltips and deviation lines."""
        # Store tooltip items before clearing
        tooltip_backup = {}
        deviation_backup = {}
        
        for i, plot_widget in enumerate(self.plot_widgets):
            # Backup tooltips
            if plot_widget in self.tooltip_items:
                tooltip_backup[plot_widget] = self.tooltip_items[plot_widget]
                # Temporarily remove tooltip from plot to prevent it being cleared
                try:
                    plot_widget.removeItem(tooltip_backup[plot_widget])
                except:
                    pass
            
            # Backup deviation lines and other non-signal items
            deviation_backup[plot_widget] = []
            for item in plot_widget.listDataItems():
                # Check if this is a deviation line by looking at its pen color and width
                if hasattr(item, 'opts') and 'pen' in item.opts:
                    pen = item.opts['pen']
                    if hasattr(pen, 'color') and hasattr(pen, 'width'):
                        # Red lines with width >= 3 are likely deviation lines
                        if (pen.color().name() in ['#ff0000', '#FF0000'] and pen.width() >= 3) or \
                           (hasattr(item, 'name') and item.name() and 'deviation' in item.name().lower()):
                            deviation_backup[plot_widget].append(item)
                            try:
                                plot_widget.removeItem(item)
                            except:
                                pass
        
        # Clear all plot content
        for plot_widget in self.plot_widgets:
            plot_widget.clear()
        
        # Restore tooltips
        for plot_widget, tooltip_item in tooltip_backup.items():
            try:
                plot_widget.addItem(tooltip_item)
                tooltip_item.hide()  # Keep hidden until mouse moves
            except Exception as e:
                logger.debug(f"Failed to restore tooltip: {e}")
                # Re-create tooltip if restoration failed
                self._setup_tooltip_for_plot(plot_widget, self.tooltips_enabled)
        
        # Restore deviation lines
        for plot_widget, deviation_items in deviation_backup.items():
            for item in deviation_items:
                try:
                    plot_widget.addItem(item)
                    logger.debug(f"Restored deviation line: {getattr(item, 'name', 'unnamed')}")
                except Exception as e:
                    logger.debug(f"Failed to restore deviation line: {e}")
        
        self.current_signals.clear()
        self.signal_colors.clear()  # Clear color info as well
        self.original_data_ranges.clear()  # Clear stored original ranges
        
        logger.debug("Cleared all signals while preserving tooltips and deviation lines")
    
    def _custom_auto_range_for_plot(self, plot_index, *args, **kwargs):
        """Custom autoRange that uses original data ranges for better view reset."""
        if plot_index in self.original_data_ranges:
            try:
                ranges = self.original_data_ranges[plot_index]
                x_min, x_max = ranges['x_min'], ranges['x_max']
                y_min, y_max = ranges['y_min'], ranges['y_max']
                
                # Add 5% padding for better visibility
                x_padding = (x_max - x_min) * 0.05
                y_padding = (y_max - y_min) * 0.05
                
                # Set the view range to show ALL original data
                plot_widget = self.plot_widgets[plot_index]
                plot_widget.setXRange(x_min - x_padding, x_max + x_padding, padding=0)
                plot_widget.setYRange(y_min - y_padding, y_max + y_padding, padding=0)
                
                logger.debug(f"Custom autoRange for plot {plot_index} using original ranges: X=[{x_min:.2f}, {x_max:.2f}]")
                return
            except Exception as e:
                logger.warning(f"Failed to use custom autoRange for plot {plot_index}: {e}")
        
        # Fallback to original autoRange
        try:
            plot_widget = self.plot_widgets[plot_index]
            view_box = plot_widget.getViewBox()
            if hasattr(view_box, '_original_autoRange'):
                view_box._original_autoRange(*args, **kwargs)
            else:
                # Last resort fallback
                plot_widget.autoRange(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Fallback autoRange also failed for plot {plot_index}: {e}")
    
    def reset_view(self):
        """Reset the plot view to show all data including limit lines using original data ranges."""
        for idx, plot_widget in enumerate(self.plot_widgets):
            # Use stored original data ranges if available (for downsampled data)
            if idx in self.original_data_ranges:
                try:
                    ranges = self.original_data_ranges[idx]
                    x_min, x_max = ranges['x_min'], ranges['x_max']
                    y_min, y_max = ranges['y_min'], ranges['y_max']
                    
                    # Add 5% padding for better visibility
                    x_padding = (x_max - x_min) * 0.05
                    y_padding = (y_max - y_min) * 0.05
                    
                    # Set the view range to show ALL original data
                    plot_widget.setXRange(x_min - x_padding, x_max + x_padding, padding=0)
                    plot_widget.setYRange(y_min - y_padding, y_max + y_padding, padding=0)
                    
                    logger.debug(f"Reset view for plot {idx} using original ranges: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
                except Exception as e:
                    logger.warning(f"Failed to use original range for plot {idx}, falling back to autoRange: {e}")
                    plot_widget.autoRange()
            else:
                # No original range stored, use autoRange (for non-downsampled data)
                plot_widget.autoRange()
                
        logger.info(f"Reset view - all {len(self.plot_widgets)} plots reset to show full data range")
    
    def redraw_all_plots(self):
        """Redraw all plots to reflect updated data."""
        for plot_widget in self.plot_widgets:
            # Force a repaint/redraw of the plot widget
            plot_widget.update()
            plot_widget.repaint()
        
        logger.debug("Redrawn all plots")
    
    def _get_next_color(self, index: int) -> str:
        """Get the next color for a new signal."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        return colors[index % len(colors)]
    
    def get_signal_color(self, plot_index: int, signal_name: str) -> Optional[str]:
        """Get the color of a specific signal in a plot."""
        signal_key = f"{signal_name}_{plot_index}"
        if signal_key in self.current_signals:
            plot_item = self.current_signals[signal_key]
            if hasattr(plot_item, 'opts') and 'pen' in plot_item.opts:
                pen = plot_item.opts['pen']
                if hasattr(pen, 'color'):
                    return pen.color().name()
                elif isinstance(pen, str):
                    return pen
        return None
    
    def get_plot_widgets(self) -> List[pg.PlotWidget]:
        """Get all plot widgets."""
        return self.plot_widgets.copy()
    
    def get_current_signals(self) -> Dict[str, Any]:
        """Get current signals dictionary."""
        return self.current_signals.copy()
    
    def apply_normalization(self, signal_data: Dict[str, Dict]):
        """Apply normalization to signals."""
        for signal_key, plot_item in self.current_signals.items():
            if signal_key in signal_data:
                data = signal_data[signal_key]
                if 'normalized_y' in data:
                    plot_item.setData(data['x'], data['normalized_y'])
    
    def remove_normalization(self, signal_data: Dict[str, Dict]):
        """Remove normalization from signals."""
        for signal_key, plot_item in self.current_signals.items():
            if signal_key in signal_data:
                data = signal_data[signal_key]
                if 'original_y' in data:
                    plot_item.setData(data['x'], data['original_y'])
    
    def clear_signals(self):
        """Tüm sinyalleri temizle."""
        try:
            # Plot widget'larındaki tüm sinyalleri temizle
            for plot_widget in self.plot_widgets:
                plot_widget.clear()
            
            # Sinyal referanslarını temizle
            self.current_signals.clear()
            self.signal_colors.clear()
            
            logger.debug("All signals cleared from plots")
            
        except Exception as e:
            logger.error(f"Error clearing signals: {e}")

    def render_signals(self, all_signals: Dict[str, Any]):
        """Tüm sinyalleri plot'lara render et."""
        try:
            # Mevcut sinyalleri temizle
            self.clear_signals()
            
            # Her sinyali uygun plot'a ekle
            for signal_name, signal_data in all_signals.items():
                if 'x_data' in signal_data and 'y_data' in signal_data:
                    x_data = signal_data['x_data']
                    y_data = signal_data['y_data']
                    
                    # Plot indeksini belirle (şimdilik tüm sinyaller plot 0'a)
                    plot_index = 0
                    
                    # Sinyali ekle
                    self.add_signal(signal_name, x_data, y_data, plot_index)
            
            logger.info(f"Rendered {len(all_signals)} signals to plots")
            
        except Exception as e:
            logger.error(f"Error rendering signals: {e}")
    
    def update_plots(self, all_signals: Dict[str, Any]):
        """Alias for render_signals for backward compatibility."""
        self.render_signals(all_signals)

    def get_subplot_count(self) -> int:
        """Subplot sayısını döndür."""
        return self.subplot_count

    def update_signal_data(self, name: str, x_data: np.ndarray, y_data: np.ndarray, plot_index: int = 0):
        """Update existing signal data."""
        signal_key = f"{name}_{plot_index}"
        if signal_key in self.current_signals:
            plot_item = self.current_signals[signal_key]
            plot_item.setData(x_data, y_data)
        else:
            # Signal doesn't exist, add it
            self.add_signal(name, x_data, y_data, plot_index)
    
    def set_grid_visibility(self, show_grid: bool):
        """Set grid visibility for all plots."""
        self.grid_visible = show_grid
        for plot_widget in self.plot_widgets:
            plot_widget.showGrid(x=show_grid, y=show_grid, alpha=self.theme_colors['grid_alpha'] if show_grid else 0.0)
    
    def update_global_settings(self):
        """Update plot widgets with current global settings."""
        global_settings = self._get_global_settings()
        
        # Update grid visibility
        show_grid = global_settings.get('show_grid', True)
        self.set_grid_visibility(show_grid)
        
        # Update autoscale
        autoscale = global_settings.get('autoscale', True)
        for plot_widget in self.plot_widgets:
            plot_widget.enableAutoRange(axis='y', enable=autoscale)
        
        # Update legend visibility
        show_legend = global_settings.get('show_legend', True)
        self.set_legend_visibility(show_legend)
        
        # Update tooltips
        tooltips_enabled = global_settings.get('show_tooltips', False)
        self.tooltips_enabled = tooltips_enabled
        for plot_widget in self.plot_widgets:
            self._setup_tooltip_for_plot(plot_widget, tooltips_enabled)
        
        logger.debug("Plot widgets updated with current global settings")
    
    def set_tooltips_enabled(self, enabled: bool):
        """✅ REFACTORED: Delegate to helper"""
        self.tooltips_helper.set_tooltips_enabled(enabled)

    def apply_theme(self, colors: Dict[str, str]):
        """Apply a new theme to all plots."""
        self.theme_colors = {
            'background': colors.get('background', '#1e1e1e'),
            'axis_pen': colors.get('axis', '#ffffff'),
            'grid_alpha': 0.3
        }
        
        axis_pen = pg.mkPen(color=self.theme_colors['axis_pen'])
        
        for plot_widget in self.plot_widgets:
            plot_widget.setBackground(self.theme_colors['background'])
            plot_widget.getAxis('left').setPen(axis_pen)
            plot_widget.getAxis('bottom').setPen(axis_pen)
            
            plot_widget.showGrid(x=self.grid_visible, y=self.grid_visible, alpha=self.theme_colors['grid_alpha'] if self.grid_visible else 0.0)
            
    def set_line_width(self, width: int):
        """Set line width for all signals in all plots."""
        updated_count = 0
        
        for signal_key, plot_item in self.current_signals.items():
            try:
                # Önce saklanan orijinal rengi kullanmaya çalış
                pen_color = self.signal_colors.get(signal_key, None)
                pen_style = None
                
                # Eğer saklanan renk yoksa, mevcut pen'den almaya çalış
                if pen_color is None:
                    current_pen = None
                    
                    # opts dictionary'den pen al
                    if hasattr(plot_item, 'opts') and 'pen' in plot_item.opts:
                        current_pen = plot_item.opts['pen']
                    
                    # Mevcut pen'den renk ve stil bilgilerini al
                    if current_pen is not None:
                        try:
                            # Renk bilgisini al
                            if hasattr(current_pen, 'color'):
                                existing_color = current_pen.color()
                                # QColor'dan string'e çevir
                                if hasattr(existing_color, 'name'):
                                    pen_color = existing_color.name()
                                elif hasattr(existing_color, 'getRgb'):
                                    r, g, b, a = existing_color.getRgb()
                                    pen_color = f"#{r:02x}{g:02x}{b:02x}"
                            
                            # Stil bilgisini al
                            if hasattr(current_pen, 'style'):
                                pen_style = current_pen.style()
                                
                        except Exception as e:
                            logger.debug(f"Could not extract pen properties for {signal_key}: {e}")
                
                # Hala renk yoksa, signal index'e göre renk ata
                if pen_color is None:
                    # Signal key'den index çıkar ve renk ata
                    try:
                        # signal_key format: "signal_name_plot_index"
                        parts = signal_key.split('_')
                        if len(parts) >= 2:
                            # Son kısmı plot index olarak al
                            plot_index = int(parts[-1])
                            # Signal adından index hesapla
                            signal_name = '_'.join(parts[:-1])
                            signal_index = len([k for k in self.current_signals.keys() if k.startswith(signal_name)])
                            pen_color = self._get_next_color(signal_index)
                        else:
                            pen_color = self._get_next_color(updated_count)
                    except:
                        pen_color = self._get_next_color(updated_count)
                
                # Yeni pen oluştur - sadece width değişecek
                if pen_style is not None:
                    new_pen = pg.mkPen(color=pen_color, width=width, style=pen_style)
                else:
                    new_pen = pg.mkPen(color=pen_color, width=width)
                
                # Pen'i uygula
                if hasattr(plot_item, 'setPen'):
                    plot_item.setPen(new_pen)
                    updated_count += 1
                    logger.info(f"Updated line width to {width} for signal: {signal_key} (color: {pen_color})")
                        
            except Exception as e:
                logger.error(f"Failed to update line width for signal {signal_key}: {e}")
        
        logger.info(f"Set line width to {width} for {updated_count}/{len(self.current_signals)} signals")

    def set_snap_to_data(self, enabled: bool):
        """Enable or disable snap to data points functionality."""
        self.snap_to_data_enabled = enabled
        logger.debug(f"PlotManager: Snap to data points {'enabled' if enabled else 'disabled'}")
        
        # Notify cursor manager if it exists
        if hasattr(self.parent, 'cursor_manager') and self.parent.cursor_manager:
            self.parent.cursor_manager.set_snap_to_data(enabled)

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

    def get_visible_signals(self) -> List[str]:
        """Get a list of unique signal names currently visible on the plots."""
        signal_names = set()
        for signal_key in self.current_signals.keys():
            # signal_key is in format "signal_name_plot_index"
            # We want to extract just "signal_name"
            base_name = '_'.join(signal_key.split('_')[:-1])
            if base_name:
                signal_names.add(base_name)
        return list(signal_names)

    def _setup_context_menu_for_plot(self, plot_widget, plot_index: int):
        """Setup custom context menu for plot widget with Zoom to Cursor option."""
        view_box = plot_widget.getViewBox()
        menu = view_box.menu
        
        # Add separator before custom actions
        menu.addSeparator()
        
        # Add "Zoom to Cursors" action
        zoom_to_cursors_action = QAction("🎯 Zoom to Cursors", menu)
        zoom_to_cursors_action.setToolTip("Zoom to the range between dual cursors")
        
        # Get cursor_manager - PlotManager.parent is GraphContainer
        def get_cursor_manager():
            cursor_manager = None
            main_widget = None
            
            if hasattr(self.parent, 'main_widget'):
                main_widget = self.parent.main_widget
            
            if main_widget and hasattr(main_widget, 'cursor_manager'):
                cursor_manager = main_widget.cursor_manager
            
            return cursor_manager
        
        # Connect action
        def on_zoom_to_cursors():
            cursor_manager = get_cursor_manager()
            if cursor_manager and hasattr(cursor_manager, 'zoom_to_cursors'):
                success = cursor_manager.zoom_to_cursors()
                if success:
                    logger.info("Zoomed to cursors from context menu")
                else:
                    logger.warning("Failed to zoom to cursors")
        
        zoom_to_cursors_action.triggered.connect(on_zoom_to_cursors)
        
        # Update button state when menu is about to show
        def update_menu():
            cursor_manager = get_cursor_manager()
            if cursor_manager and hasattr(cursor_manager, 'can_zoom_to_cursors'):
                can_zoom = cursor_manager.can_zoom_to_cursors()
                zoom_to_cursors_action.setEnabled(can_zoom)
            else:
                zoom_to_cursors_action.setEnabled(False)
        
        menu.aboutToShow.connect(update_menu)
        menu.addAction(zoom_to_cursors_action)
        
        logger.debug(f"Custom context menu setup for plot {plot_index}")

    def set_legend_visibility(self, visible: bool):
        """Set legend visibility for all plots."""
        for plot_widget in self.plot_widgets:
            # PyQtGraph'ta legend'e erişim için plotItem üzerinden gitmek gerekiyor
            legend = plot_widget.plotItem.legend
            if legend:
                legend.setVisible(visible)
    
    def set_y_axis_width(self, width: int):
        """Y ekseni genişliğini ayarla (grafik hizalama için)."""
        self.y_axis_width = width
        for plot_widget in self.plot_widgets:
            left_axis = plot_widget.getAxis('left')
            left_axis.setWidth(width)
        logger.info(f"Y ekseni genişliği {width} piksel olarak ayarlandı")
    
    def get_y_axis_width(self) -> int:
        """Mevcut Y ekseni genişliğini döndür."""
        return self.y_axis_width
    
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
