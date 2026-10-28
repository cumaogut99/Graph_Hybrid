"""
Cursors and the statistics panel values that follow them.

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging
import time  # ✅ OPTIMIZED: For throttling timestamp
from typing import Dict, Optional
from PyQt5.QtCore import QTimer
from src.managers.cursor_manager import CursorManager

logger = logging.getLogger(__name__)


class CursorStatisticsMixin:
    """Cursors and the statistics panel values that follow them."""

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
                initial_snap_setting = self.graph_settings_panel_manager.global_settings.get('snap_to_data', True)
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

    def _on_cursor_moved_old(self, cursor_type: str, position: float):
        """Handle cursor movement events (old signature - deprecated)."""
        self.current_cursor_position = position
        self._update_legend_values()
        self.cursor_moved.emit(cursor_type, position)

    def _on_range_changed(self, start: float, end: float):
        """Handle range changes from cursor manager."""
        self.selected_range = (start, end)
        self._update_statistics_for_range(start, end)

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
                        all_signals = self.signal_processor.get_signal_names()
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
