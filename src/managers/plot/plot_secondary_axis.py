# type: ignore
"""
Plot Secondary Axis Helper

Secondary Y-axis functionality for plots:
- Dual Y-axis support
- Automatic signal assignment
- Axis management
"""

import logging
from typing import Dict, Optional
import numpy as np
import pyqtgraph as pg

logger = logging.getLogger(__name__)


class PlotSecondaryAxisHelper:
    """Helper class for managing secondary Y-axis functionality."""
    
    def __init__(self, plot_manager):
        """
        Initialize secondary axis helper.
        
        Args:
            plot_manager: Reference to parent PlotManager instance
        """
        self.plot_manager = plot_manager
        self.secondary_axis_enabled = False
        self.secondary_viewboxes = {}  # {plot_index: ViewBox}
        self.secondary_axes = {}  # {plot_index: AxisItem}
        self.signal_axis_assignment = {}  # {signal_name: 'left' or 'right'}
    
    def set_secondary_axis_enabled(self, enabled: bool):
        """Enable or disable secondary Y-axis for all plots."""
        self.secondary_axis_enabled = enabled
        
        if enabled:
            # Setup secondary axis for all existing plots
            for i, plot_widget in enumerate(self.plot_manager.plot_widgets):
                self._setup_secondary_axis_for_plot(plot_widget, i)
        else:
            # Remove secondary axis from all plots
            for i in list(self.secondary_viewboxes.keys()):
                self._remove_secondary_axis_for_plot(i)
        
        logger.info(f"Secondary axis {'enabled' if enabled else 'disabled'} for all plots")
    
    def _setup_secondary_axis_for_plot(self, plot_widget, plot_index: int):
        """Setup secondary Y-axis for a specific plot."""
        if plot_index in self.secondary_viewboxes:
            logger.debug(f"Secondary axis already exists for plot {plot_index}, skipping")
            return  # Already set up
        
        logger.info(f"Setting up secondary axis for plot {plot_index}")
        
        # Get the plot item
        plot_item = plot_widget.getPlotItem()
        
        # Create secondary ViewBox and overlay it on the main ViewBox
        secondary_vb = pg.ViewBox()
        plot_item.scene().addItem(secondary_vb)
        
        # Create secondary Y-axis on the right
        secondary_axis = pg.AxisItem('right')
        plot_item.layout.addItem(secondary_axis, 2, 3)  # Add to right side (row 2, col 3)
        secondary_axis.linkToView(secondary_vb)
        
        # Link X-axis to main plot
        secondary_vb.setXLink(plot_widget.getViewBox())
        
        # Set axis label
        secondary_axis.setLabel('Secondary Y-Axis')
        
        # Set axis pen color
        axis_pen = pg.mkPen(color=self.plot_manager.theme_colors['axis_pen'])
        secondary_axis.setPen(axis_pen)
        
        # Set width
        secondary_axis.setWidth(self.plot_manager.y_axis_width)
        
        # Make secondary ViewBox resize with main ViewBox
        def update_views():
            try:
                secondary_vb.setGeometry(plot_widget.getViewBox().sceneBoundingRect())
            except RuntimeError:
                pass  # ViewBox deleted
        
        # Connect geometry changes
        plot_widget.getViewBox().sigResized.connect(update_views)
        update_views()  # Initial update
        
        # Store references
        self.secondary_viewboxes[plot_index] = secondary_vb
        self.secondary_axes[plot_index] = secondary_axis
        
        logger.info(f"Secondary axis successfully set up for plot {plot_index}")
    
    def _remove_secondary_axis_for_plot(self, plot_index: int):
        """Remove secondary Y-axis from a specific plot."""
        if plot_index not in self.secondary_viewboxes:
            return
        
        # Remove axis from layout
        if plot_index in self.secondary_axes:
            axis = self.secondary_axes[plot_index]
            try:
                if axis.scene():
                    axis.scene().removeItem(axis)
            except RuntimeError:
                # Object already deleted
                pass
            del self.secondary_axes[plot_index]
        
        # Remove viewbox
        if plot_index in self.secondary_viewboxes:
            vb = self.secondary_viewboxes[plot_index]
            try:
                if vb.scene():
                    vb.scene().removeItem(vb)
            except RuntimeError:
                # Object already deleted
                pass
            del self.secondary_viewboxes[plot_index]
        
        # Clear axis assignments for this plot
        keys_to_remove = [k for k in self.signal_axis_assignment if k.endswith(f"_{plot_index}")]
        for key in keys_to_remove:
            del self.signal_axis_assignment[key]
        
        logger.debug(f"Secondary axis removed from plot {plot_index}")
    
    def assign_signal_to_axis(self, signal_name: str, y_data: np.ndarray, plot_index: int) -> str:
        """
        Automatically assign signal to left or right axis based on value ranges.
        
        Returns 'left' or 'right'
        """
        if not self.secondary_axis_enabled:
            return 'left'
        
        # Get existing signals on this plot
        existing_signals = {name: data for name, data in self.plot_manager.current_signals.items() 
                           if name.endswith(f'_{plot_index}')}
        
        if not existing_signals:
            # First signal goes to left axis
            self.signal_axis_assignment[signal_name] = 'left'
            return 'left'
        
        # Calculate value ranges for existing signals
        existing_ranges = []
        for name, plot_item in existing_signals.items():
            if hasattr(plot_item, 'yData') and plot_item.yData is not None:
                existing_data = plot_item.yData
                if len(existing_data) > 0:
                    existing_ranges.append((np.min(existing_data), np.max(existing_data)))
        
        if not existing_ranges:
            self.signal_axis_assignment[signal_name] = 'left'
            return 'left'
        
        # Calculate range for new signal
        new_range = (np.min(y_data), np.max(y_data))
        new_center = (new_range[0] + new_range[1]) / 2
        new_span = new_range[1] - new_range[0]
        
        # Calculate average range for existing signals
        avg_existing_center = np.mean([(r[0] + r[1]) / 2 for r in existing_ranges])
        avg_existing_span = np.mean([r[1] - r[0] for r in existing_ranges])
        
        # Check if ranges are significantly different
        # If centers are far apart or spans are very different, use different axes
        center_diff = abs(new_center - avg_existing_center)
        span_ratio = new_span / avg_existing_span if avg_existing_span > 0 else 1.0
        
        # Threshold: if center difference is > 50% of average span, or span ratio > 5x, use different axis
        if center_diff > avg_existing_span * 0.5 or span_ratio > 5.0 or span_ratio < 0.2:
            # Check which axis has fewer signals on this plot
            left_count = sum(1 for name, axis in self.signal_axis_assignment.items() 
                           if axis == 'left' and name.endswith(f'_{plot_index}'))
            right_count = sum(1 for name, axis in self.signal_axis_assignment.items() 
                            if axis == 'right' and name.endswith(f'_{plot_index}'))
            
            if right_count < left_count:
                self.signal_axis_assignment[signal_name] = 'right'
                return 'right'
            else:
                self.signal_axis_assignment[signal_name] = 'left'
                return 'left'
        else:
            # Similar ranges, use same axis as most signals
            self.signal_axis_assignment[signal_name] = 'left'
            return 'left'
    
    def clear_secondary_axes(self):
        """Clear all secondary axis references."""
        self.secondary_viewboxes.clear()
        self.secondary_axes.clear()
        self.signal_axis_assignment.clear()
        logger.debug("Cleared secondary axis references")

