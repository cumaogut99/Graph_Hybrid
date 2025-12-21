# type: ignore
"""
Plot Tooltips Helper

Tooltip functionality for plot widgets:
- Mouse tracking
- Signal value display
- Enhanced tooltip generation
"""

import logging
from typing import Dict, Optional, List
import numpy as np
import pyqtgraph as pg

logger = logging.getLogger(__name__)


class PlotTooltipsHelper:
    """Helper class for managing plot tooltips functionality."""
    
    def __init__(self, plot_manager):
        """
        Initialize tooltip helper.
        
        Args:
            plot_manager: Reference to parent PlotManager instance
        """
        self.plot_manager = plot_manager
        self.tooltips_enabled = False
        self.tooltip_items = {}  # Store tooltip items per plot widget
        
    def set_tooltips_enabled(self, enabled: bool):
        """Enable or disable tooltips functionality."""
        self.tooltips_enabled = enabled
        logger.info(f"Tooltips {'enabled' if enabled else 'disabled'}")
        
        # Apply to all existing plot widgets
        for plot_widget in self.plot_manager.plot_widgets:
            self._setup_tooltip_for_plot(plot_widget, enabled)
    
    def ensure_tooltips_after_rebuild(self):
        """Ensure tooltips are properly setup after plot rebuild."""
        try:
            # Clear old tooltip items that may reference deleted plot widgets
            old_tooltip_items = list(self.tooltip_items.keys())
            for old_plot_widget in old_tooltip_items:
                if old_plot_widget not in self.plot_manager.plot_widgets:
                    # Remove tooltip items for deleted plot widgets
                    del self.tooltip_items[old_plot_widget]
            
            # Setup tooltips for all current plot widgets
            for plot_widget in self.plot_manager.plot_widgets:
                if plot_widget not in self.tooltip_items:
                    self._setup_tooltip_for_plot(plot_widget, self.tooltips_enabled)
            
            logger.debug(f"Tooltips ensured for {len(self.plot_manager.plot_widgets)} plot widgets")
            
        except Exception as e:
            logger.warning(f"Failed to ensure tooltips after rebuild: {e}")
    
    def clear_tooltips(self):
        """Clear all tooltip items."""
        for plot_widget in list(self.tooltip_items.keys()):
            if plot_widget in self.tooltip_items:
                tooltip_item = self.tooltip_items[plot_widget]
                try:
                    tooltip_item.hide()
                    if hasattr(plot_widget, 'removeItem'):
                        plot_widget.removeItem(tooltip_item)
                except Exception as e:
                    logger.debug(f"Error removing tooltip item: {e}")
        
        self.tooltip_items.clear()
    
    def _setup_tooltip_for_plot(self, plot_widget, enabled: bool):
        """Setup or remove tooltip functionality for a specific plot widget."""
        try:
            if enabled:
                # Create tooltip text item if it doesn't exist
                if plot_widget not in self.tooltip_items:
                    tooltip_item = pg.TextItem(
                        text="",
                        color=(255, 255, 255),
                        fill=(30, 30, 30, 200),
                        anchor=(0, 1)
                    )
                    tooltip_item.setZValue(1000)
                    plot_widget.addItem(tooltip_item)
                    tooltip_item.hide()
                    self.tooltip_items[plot_widget] = tooltip_item
                    logger.debug(f"Created tooltip item for plot widget")
                    
                    # Store original leaveEvent to restore later
                    if not hasattr(plot_widget, '_original_leaveEvent'):
                        plot_widget._original_leaveEvent = plot_widget.leaveEvent
                
                # Disconnect any existing connections to avoid duplicates
                try:
                    plot_widget.scene().sigMouseMoved.disconnect()
                except:
                    pass
                
                # Connect mouse move events with unique connection
                plot_widget.scene().sigMouseMoved.connect(
                    lambda pos, pw=plot_widget: self._on_mouse_moved(pos, pw)
                )
                
                # Override leaveEvent to hide tooltip
                def custom_leave_event(event, pw=plot_widget):
                    self._on_mouse_left(pw)
                    # Call original leaveEvent if it exists
                    if hasattr(pw, '_original_leaveEvent') and pw._original_leaveEvent:
                        pw._original_leaveEvent(event)
                
                plot_widget.leaveEvent = custom_leave_event
                
            else:
                # Disable tooltips
                if plot_widget in self.tooltip_items:
                    tooltip_item = self.tooltip_items[plot_widget]
                    tooltip_item.hide()
                
                # Disconnect mouse events
                try:
                    plot_widget.scene().sigMouseMoved.disconnect()
                except:
                    pass
                
                # Restore original leaveEvent
                if hasattr(plot_widget, '_original_leaveEvent'):
                    plot_widget.leaveEvent = plot_widget._original_leaveEvent
                    
        except Exception as e:
            logger.warning(f"Failed to setup tooltip for plot widget: {e}")

    def _on_mouse_moved(self, pos, plot_widget):
        """Handle mouse movement over plot widget for tooltip display."""
        if not self.tooltips_enabled:
            return
            
        if plot_widget not in self.tooltip_items:
            logger.warning(f"Tooltip item not found for plot_widget, creating it now")
            self._setup_tooltip_for_plot(plot_widget, True)
            if plot_widget not in self.tooltip_items:
                return
            
        try:
            # Check if the position is valid
            if pos is None:
                return
                
            # Convert scene position to view coordinates
            view_pos = plot_widget.plotItem.vb.mapSceneToView(pos)
            x_pos = view_pos.x()
            y_pos = view_pos.y()
            
            # Check if coordinates are valid (not NaN or infinite)
            if not (isinstance(x_pos, (int, float)) and isinstance(y_pos, (int, float))):
                return
            if abs(x_pos) == float('inf') or abs(y_pos) == float('inf'):
                return
            
            # Get tooltip item
            tooltip_item = self.tooltip_items[plot_widget]
            
            # Find the closest signal to mouse cursor
            closest_signal_info = self._find_closest_signal_to_cursor(x_pos, y_pos, plot_widget)
            
            # Generate tooltip text
            tooltip_text = self._generate_enhanced_tooltip_text(x_pos, y_pos, closest_signal_info)
            
            if tooltip_text and len(tooltip_text.strip()) > 0:
                # Position tooltip with slight offset to avoid cursor overlap
                view_range = plot_widget.plotItem.vb.viewRange()
                x_range = view_range[0]
                y_range = view_range[1]
                
                # Calculate offset (5% of visible range)
                x_offset = (x_range[1] - x_range[0]) * 0.05
                y_offset = (y_range[1] - y_range[0]) * 0.05
                
                tooltip_x = x_pos + x_offset
                tooltip_y = y_pos + y_offset
                
                tooltip_item.setPos(tooltip_x, tooltip_y)
                tooltip_item.setText(tooltip_text)
                tooltip_item.show()
                logger.debug(f"Tooltip shown at ({tooltip_x:.2f}, {tooltip_y:.2f}): {tooltip_text[:30]}")
            else:
                tooltip_item.hide()
                logger.debug(f"Tooltip hidden: empty text")
                
        except Exception as e:
            logger.error(f"Error updating tooltip: {e}", exc_info=True)
            # Hide tooltip on error to prevent stuck tooltips
            if plot_widget in self.tooltip_items:
                self.tooltip_items[plot_widget].hide()

    def _on_mouse_left(self, plot_widget):
        """Handle mouse leaving the plot widget."""
        if plot_widget in self.tooltip_items:
            tooltip_item = self.tooltip_items[plot_widget]
            tooltip_item.hide()

    def _find_closest_signal_to_cursor(self, x_pos: float, y_pos: float, plot_widget):
        """Find the signal closest to the mouse cursor position in Y-axis."""
        try:
            closest_signal = None
            min_y_distance = float('inf')
            
            # Get the plot index for this widget
            try:
                plot_index = self.plot_manager.plot_widgets.index(plot_widget)
            except (ValueError, AttributeError):
                plot_index = 0
            
            # Get signal processor - try parent (GraphContainer) first, then main_widget
            signal_processor = None
            if hasattr(self.plot_manager.parent, 'signal_processor') and self.plot_manager.parent.signal_processor:
                signal_processor = self.plot_manager.parent.signal_processor
            elif hasattr(self.plot_manager.parent, 'main_widget') and hasattr(self.plot_manager.parent.main_widget, 'signal_processor'):
                signal_processor = self.plot_manager.parent.main_widget.signal_processor
            
            if not signal_processor:
                logger.debug("No signal_processor found on parent or main_widget")
                return None
            
            # Get signal names from signal_data keys
            if not hasattr(signal_processor, 'signal_data'):
                logger.debug("signal_processor has no signal_data attribute")
                return None
                
            signal_names = list(signal_processor.signal_data.keys())
            
            if not signal_names:
                logger.debug("No signal names found in signal_data")
                return None
            
            logger.debug(f"Finding closest signal - Available signals: {signal_names[:3]}..., plot_index: {plot_index}")
            
            for signal_name in signal_names:
                try:
                    # Check if this signal is displayed on this plot
                    signal_key = f"{signal_name}_{plot_index}"
                    if signal_key not in self.plot_manager.current_signals:
                        logger.debug(f"Signal key '{signal_key}' not in current_signals")
                        continue
                    
                    signal_data = signal_processor.get_signal_data(signal_name)
                    if not signal_data or 'x_data' not in signal_data or 'y_data' not in signal_data:
                        continue
                        
                    x_data = signal_data['x_data']
                    y_data = signal_data['y_data']
                    
                    if len(x_data) == 0 or len(y_data) == 0:
                        continue
                        
                    x_array = np.array(x_data)
                    y_array = np.array(y_data)
                    
                    # Check for valid arrays
                    if x_array.size == 0 or y_array.size == 0:
                        continue
                    
                    # Check if position is within reasonable data range
                    x_min, x_max = np.min(x_array), np.max(x_array)
                    if x_pos < x_min or x_pos > x_max:
                        continue
                    
                    # Find the Y value at the current mouse X position (interpolated)
                    try:
                        interpolated_y = np.interp(x_pos, x_data, y_data)
                        
                        # Calculate Y-axis distance only
                        y_distance = abs(y_pos - interpolated_y)
                        
                        if y_distance < min_y_distance:
                            min_y_distance = y_distance
                            
                            # Find closest actual X data point for reference
                            idx = np.argmin(np.abs(x_array - x_pos))
                            
                            closest_signal = {
                                'name': signal_name,
                                'x_value': x_pos,  # Use mouse X position
                                'y_value': interpolated_y,  # Use interpolated Y value at mouse X
                                'closest_data_x': x_array[idx],  # Closest actual data point X
                                'closest_data_y': y_array[idx],  # Closest actual data point Y
                                'y_distance': y_distance
                            }
                            logger.debug(f"Found closer signal: {signal_name}, y_distance: {y_distance:.3f}, interpolated_y: {interpolated_y:.3f}")
                    except Exception as e:
                        logger.debug(f"Error interpolating for signal {signal_name}: {e}")
                        continue
                            
                except Exception as e:
                    logger.debug(f"Error processing signal {signal_name}: {e}")
                    continue
            
            if closest_signal:
                logger.debug(f"Returning closest signal: {closest_signal['name']} with y_value: {closest_signal['y_value']:.3f}")
            else:
                logger.debug(f"No closest signal found")
            
            return closest_signal
            
        except Exception as e:
            logger.error(f"Error finding closest signal: {e}", exc_info=True)
            return None

    def _generate_enhanced_tooltip_text(self, x_pos: float, y_pos: float, closest_signal_info: dict) -> str:
        """Generate tooltip text showing parameter name and values at cursor position."""
        try:
            tooltip_lines = []
            
            if closest_signal_info:
                # Parameter name as title - no emoji to avoid Unicode issues
                signal_name = closest_signal_info['name']
                display_name = signal_name
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."
                
                # Show parameter name and its value only
                tooltip_lines.append(display_name)
                tooltip_lines.append(f"Deger: {closest_signal_info['y_value']:.6f}")
                
            else:
                # Fallback if no signal found - show position
                tooltip_lines.append(f"Pozisyon")
                tooltip_lines.append(f"Y: {y_pos:.6f}")
            
            result = "\n".join(tooltip_lines)
            logger.debug(f"Generated tooltip: '{result}' (has_signal: {closest_signal_info is not None})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating enhanced tooltip text: {e}")
            return f"Y: {y_pos:.6f}"

    def _generate_tooltip_text(self, x_pos: float, y_pos: float) -> str:
        """Generate tooltip text showing signal values at the given position."""
        try:
            tooltip_lines = [f"X: {x_pos:.6f}"]
            
            # Get signal values at this X position
            if hasattr(self.plot_manager.parent, 'signal_processor'):
                signal_processor = self.plot_manager.parent.signal_processor
                signal_names = signal_processor.get_signal_names()
                
                values_found = 0
                for signal_name in signal_names[:5]:  # Limit to first 5 signals to avoid clutter
                    signal_data = signal_processor.get_signal_data(signal_name)
                    if signal_data and 'x_data' in signal_data and 'y_data' in signal_data:
                        x_data = signal_data['x_data']
                        y_data = signal_data['y_data']
                        
                        if len(x_data) > 0 and len(y_data) > 0:
                            # Find closest data point
                            x_array = np.array(x_data)
                            
                            # Check if position is within data range
                            if x_pos >= x_array[0] and x_pos <= x_array[-1]:
                                # Interpolate value at x_pos
                                y_value = np.interp(x_pos, x_data, y_data)
                                
                                # Shorten signal name if too long
                                display_name = signal_name
                                if len(display_name) > 20:
                                    display_name = display_name[:17] + "..."
                                
                                tooltip_lines.append(f"{display_name}: {y_value:.6f}")
                                values_found += 1
                
                if values_found == 0:
                    tooltip_lines.append("No signals at this position")
                elif len(signal_names) > 5:
                    tooltip_lines.append(f"... and {len(signal_names) - 5} more signals")
            
            return "\n".join(tooltip_lines)
            
        except Exception as e:
            logger.debug(f"Error generating tooltip text: {e}")
            return f"X: {x_pos:.6f}\nY: {y_pos:.6f}"

