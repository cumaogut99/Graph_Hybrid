"""
View range and appearance: autoscale, grid, theme, line width, legend, axis width.

Part of PlotManager; moved from src/managers/plot_manager.py without changes.
"""

import logging
from typing import Dict, TYPE_CHECKING
import pyqtgraph as pg
if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)


class PlotViewMixin:
    """View range and appearance: autoscale, grid, theme, line width, legend, axis width."""

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
        
        # No signal in this plot: leave the view alone. Its X axis is linked
        # to the other plots of the tab, and without data the only items are
        # the cursors, so pyqtgraph's autoRange would zoom X onto them.
        if not self.has_data(plot_index):
            return

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

    def has_data(self, plot_index: int = None) -> bool:
        """True if the plot (or, with None, any plot) shows at least one signal."""
        if plot_index is None:
            return bool(self.original_data_ranges)
        if plot_index in self.original_data_ranges:
            return True
        if 0 <= plot_index < len(self.plot_widgets):
            return bool(self.plot_widgets[plot_index].getPlotItem().listDataItems())
        return False

    def fit_x_to_data(self):
        """Fit the (linked) X axis to the data of all plots, 5% padding."""
        if not self.original_data_ranges or not self.plot_widgets:
            return
        x_min = min(r['x_min'] for r in self.original_data_ranges.values())
        x_max = max(r['x_max'] for r in self.original_data_ranges.values())
        padding = (x_max - x_min) * 0.05 or 0.5
        self.plot_widgets[0].setXRange(x_min - padding, x_max + padding, padding=0)

    def fit_y_to_data(self):
        """Fit each plot's Y axis to its own data; empty plots are left alone."""
        for idx, plot_widget in enumerate(self.plot_widgets):
            ranges = self.original_data_ranges.get(idx)
            if not ranges:
                continue
            y_min, y_max = ranges['y_min'], ranges['y_max']
            padding = (y_max - y_min) * 0.05 or 0.5
            plot_widget.setYRange(y_min - padding, y_max + padding, padding=0)

    def set_y_autoscale(self, enabled: bool):
        """Y auto-range on/off; only plots with data are auto-ranged (an empty
        plot would range onto the cursors)."""
        for idx, plot_widget in enumerate(self.plot_widgets):
            plot_widget.enableAutoRange(axis='y', enable=enabled and self.has_data(idx))

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
            elif self.has_data(idx):
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
