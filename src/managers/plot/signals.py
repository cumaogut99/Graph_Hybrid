"""
Adding, removing and redrawing the signals of the plots.

Part of PlotManager; moved from src/managers/plot_manager.py without changes.
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
import pyqtgraph as pg
if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)


class PlotSignalsMixin:
    """Adding, removing and redrawing the signals of the plots."""

    def add_signal(self, name: str, x_data: np.ndarray, y_data: np.ndarray, 
                   plot_index: int = 0, pen=None, **kwargs):
        """Add a signal to a specific plot with automatic PyQtGraph optimization."""
        import time
        start_time = time.time()
        
        if not (0 <= plot_index < len(self.plot_widgets)):
            logger.warning(f"Invalid plot index: {plot_index}")
            return None
        
        plot_widget = self.plot_widgets[plot_index]

        if len(x_data) == 0 or len(y_data) == 0:
            logger.warning(f"Signal '{name}' has no data, not plotted")
            return None
        
        # Store original data range for proper view reset
        t1 = time.time()
        original_x_min = float(np.min(x_data))
        original_x_max = float(np.max(x_data))
        original_y_min = float(np.min(y_data))
        original_y_max = float(np.max(y_data))
        logger.info(f"[PERF] Min/Max calculation for {name}: {(time.time()-t1)*1000:.1f}ms ({len(x_data)} points)")
        
        # Create plot item WITHOUT manual downsampling
        # PyQtGraph will handle downsampling automatically with setDownsampling()
        # Line width from the Graph Settings panel (1 is fastest to render)
        line_width = self._get_global_settings().get('line_width', 1)
        if pen is None:
            color = self._get_next_color(len(self.current_signals))
            pen = pg.mkPen(color=color, width=line_width)
        elif isinstance(pen, str):
            pen = pg.mkPen(color=pen, width=line_width)
        
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
        """Clear all signals from all plots while preserving tooltips."""
        # Store tooltip items before clearing
        tooltip_backup = {}
        
        for i, plot_widget in enumerate(self.plot_widgets):
            # Backup tooltips
            if plot_widget in self.tooltip_items:
                tooltip_backup[plot_widget] = self.tooltip_items[plot_widget]
                # Temporarily remove tooltip from plot to prevent it being cleared
                try:
                    plot_widget.removeItem(tooltip_backup[plot_widget])
                except:
                    pass
            
        
        # Clear all plot content (secondary-axis curves are not in the plots)
        self.secondary_axis_helper.clear_secondary_items()
        for plot_widget in self.plot_widgets:
            plot_widget.clear()
        
        # Restore tooltips
        for plot_widget, tooltip_item in tooltip_backup.items():
            try:
                plot_widget.addItem(tooltip_item, ignoreBounds=True)
                tooltip_item.hide()  # Keep hidden until mouse moves
            except Exception as e:
                logger.debug(f"Failed to restore tooltip: {e}")
                # Re-create tooltip if restoration failed
                self._setup_tooltip_for_plot(plot_widget, self.tooltips_enabled)
        
        self.current_signals.clear()
        self.signal_colors.clear()  # Clear color info as well
        self.original_data_ranges.clear()  # Clear stored original ranges

        self._draw_markers()  # plot_widget.clear() removed the marker lines
        
        logger.debug("Cleared all signals while preserving tooltips")

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
            self.secondary_axis_helper.clear_secondary_items()
            # Plot widget'larındaki tüm sinyalleri temizle
            for plot_widget in self.plot_widgets:
                plot_widget.clear()
            
            # Sinyal referanslarını temizle
            self.current_signals.clear()
            self.signal_colors.clear()

            self._draw_markers()  # plot_widget.clear() removed the marker lines
            
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

    def update_signal_data(self, name: str, x_data: np.ndarray, y_data: np.ndarray, plot_index: int = 0):
        """Update existing signal data."""
        signal_key = f"{name}_{plot_index}"
        if signal_key in self.current_signals:
            plot_item = self.current_signals[signal_key]
            plot_item.setData(x_data, y_data)
        else:
            # Signal doesn't exist, add it
            self.add_signal(name, x_data, y_data, plot_index)

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
