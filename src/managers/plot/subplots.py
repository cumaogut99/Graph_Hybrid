"""
Subplot layout: creating and rebuilding the plots of a graph container.

Part of PlotManager; moved from src/managers/plot_manager.py without changes.
"""

import logging
from typing import Dict, TYPE_CHECKING
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame
from src.managers.plot import DateTimeAxisItem
if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)


class PlotSubplotsMixin:
    """Subplot layout: creating and rebuilding the plots of a graph container."""

    def _setup_plot_panel(self):
        """Create the main plot panel and its permanent layout."""
        self.plot_panel = QWidget()
        self.main_layout = QVBoxLayout(self.plot_panel)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(2)

    def set_subplot_count(self, count: int):
        """Set the number of subplots with safe error handling."""
        print(f"[ZOOM] === set_subplot_count START === current={self.subplot_count}, new={count}")
        if not (self.min_subplots <= count <= self.max_subplots):
            logger.warning(f"Invalid subplot count: {count}. Must be between {self.min_subplots} and {self.max_subplots}")
            return False
        
        if count == self.subplot_count:
            print(f"[ZOOM] Same count, returning early")
            return True
        
        try:
            # Store signal data before rebuilding UI
            signal_data_to_restore = self._get_signal_data_for_restore()
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
            
            # Update count and rebuild the entire UI safely
            self.subplot_count = count
            self._rebuild_ui()
            
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
            # Keep the time format when the plots are rebuilt (graph count change)
            datetime_axis.is_datetime_axis = getattr(self, 'datetime_axis_enabled', False)
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

            # Apply global mouse pan/zoom settings
            plot_widget.setMouseEnabled(x=global_settings.get('x_axis_mouse', True),
                                        y=global_settings.get('y_axis_mouse', True))
            
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
            
            # Setup secondary axis if enabled (new tabs follow the panel setting)
            self.secondary_axis_enabled = global_settings.get('secondary_axis', self.secondary_axis_enabled)
            if self.secondary_axis_enabled:
                self._setup_secondary_axis_for_plot(plot_widget, i)
            
            self.plot_widgets.append(plot_widget)
            # Add widget with stretch factor 1 to ensure equal heights
            plot_layout.addWidget(plot_widget, 1)
            
            # Add a separator line between graphs (except after the last one)
            if i < self.subplot_count - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                separator.setStyleSheet(f"background-color: {self.theme_colors.get('primary', '#4a90e2')}; min-height: 2px; max-height: 2px;")
                plot_layout.addWidget(separator)

        # 3. Re-create the settings buttons and their layout container
        self.settings_container = QWidget()
        self.graph_settings_layout = QHBoxLayout(self.settings_container)
        self.graph_settings_layout.setContentsMargins(5, 2, 5, 2)
        self._update_graph_settings_buttons()

        # 4. Add the new containers to the main, permanent layout
        self.main_layout.addWidget(self.plot_container)
        self.main_layout.addWidget(self.settings_container)
        
        self._draw_markers()
        logger.info(f"UI rebuilt with {self.subplot_count} subplots.")

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
        
        # The tab's mapping says which signals go to which plot. A signal can
        # be in several plots, so walk the mapping (not a signal -> plot map,
        # which kept only the last plot of such a signal).
        main_widget = getattr(self.parent, 'main_widget', None)
        if main_widget is None or not hasattr(main_widget, 'graph_signal_mapping'):
            logger.error("CRITICAL: cannot reach the graph signal mapping to restore signals")
            return
        current_tab = getattr(self.parent, 'tab_index', 0)
        tab_mapping = main_widget.graph_signal_mapping.get(current_tab, {})

        for plot_index, names in tab_mapping.items():
            # Plots beyond the new count keep their mapping but are not drawn
            if plot_index >= self.subplot_count:
                continue
            for name in names:
                data = signal_data.get(name)
                if data is None:
                    continue
                try:
                    self.add_signal(name, data['x'], data['y'], plot_index, pen=data['pen'])
                except Exception as e:
                    logger.error(f"Failed to restore signal '{name}' to plot {plot_index}: {e}")

        # New plots start at Y = [0, 1]; X is restored by set_subplot_count
        self.fit_y_to_data()

        # ❌ REMOVED: reset_view() was causing X-range expansion when changing graph count
        # The X-range is now explicitly restored in set_subplot_count after this function
        # if restored_count > 0:
        #     self.reset_view()
        
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

    def get_plot_panel(self) -> QWidget:
        """Get the plot panel widget."""
        return self.plot_panel

    def get_subplot_count(self) -> int:
        """Subplot sayısını döndür."""
        return self.subplot_count
