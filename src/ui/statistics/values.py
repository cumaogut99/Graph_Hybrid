"""
Statistics and cursor values shown in the rows.

Part of StatisticsPanel; moved from src/ui/statistics_panel.py without changes.
"""

import logging
from typing import Dict
import datetime
from PyQt5.QtGui import QColor

logger = logging.getLogger(__name__)


class StatisticsValuesMixin:
    """Statistics and cursor values shown in the rows."""

    def set_cursor_mode(self, mode: str):
        """Update cursor mode and refresh table headers."""
        # Force mode to 'dual' - other modes not supported
        if mode != 'dual':
            logger.warning(f"Cursor mode '{mode}' not supported - using 'dual' instead")
            mode = 'dual'
            
        if self.cursor_mode != mode:
            self.cursor_mode = mode
            # Update common header
            self._update_common_header()
            # Recreate table headers with new mode
            self._setup_table_headers()
            # Recreate all signal rows to match new structure
            self._recreate_all_signal_rows()
        
        # Mode is always 'dual', so always show cursor info panel
        if hasattr(self, 'cursor_info_panel'):
            self.cursor_info_panel.show()  # Always show for dual mode

    def set_datetime_axis(self, is_datetime: bool):
        """Enable or disable datetime formatting for cursor values."""
        self.is_datetime_axis = is_datetime
        # Update cursor positions with new formatting
        if self.cursor_positions:
            self.update_cursor_positions(self.cursor_positions)

    def update_statistics(self, signal_name: str, stats: Dict[str, float]):
        """
        Update statistics for a specific signal in its graph table.
        
        Args:
            signal_name: Name of the signal (with graph suffix)
            stats: Dictionary with statistic values
        """
        if signal_name not in self.signal_data:
            logger.warning(f"Signal {signal_name} not found in statistics panel")
            return
        
        signal_info = self.signal_data[signal_name]
        row_index = signal_info['row_index']
        table = signal_info['table']
        
        # Update each statistic with proper formatting
        col_index = 1  # Start after Signal column (no color column anymore)
        stats_info = self._get_stats_info_for_mode()
        
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                if stat_key in stats:
                    value = stats[stat_key]
                    
                    if isinstance(value, (int, float)):
                        # Special formatting for duty cycle
                        if stat_key == 'duty_cycle':
                            formatted_value = f"{value:.1f}%"
                        # Full number formatting - no abbreviations
                        else:
                            # Format numbers without K, M abbreviations or scientific notation
                            if abs(value) >= 1:
                                # For larger numbers, show 5 decimal places
                                formatted_value = f"{value:.5f}"
                            elif abs(value) >= 0.0001:
                                # For small numbers, show more precision
                                formatted_value = f"{value:.6f}"
                            else:
                                # For very small numbers, use scientific notation as last resort
                                formatted_value = f"{value:.2e}"
                    else:
                        formatted_value = str(value)
                    
                    # Update table cell
                    if col_index < table.columnCount():
                        item = table.item(row_index, col_index)
                        if item:
                            item.setText(formatted_value)
                            # Add visual feedback for cursor values
                            if is_cursor_stat:
                                item.setBackground(QColor(74, 144, 226, 50))  # Light blue background
                
                col_index += 1

    def _clear_cursor_values(self):
        """Clear all cursor values from statistics display."""
        for signal_name, signal_info in self.signal_data.items():
            labels = signal_info['labels']
            
            # Clear cursor values (C1, C2)
            for cursor_key in ['c1', 'c2']:
                if cursor_key in labels and labels[cursor_key]:
                    labels[cursor_key].setText("--")
                    # Reset styling for cursor labels
                    # Use theme-appropriate styling
                    labels[cursor_key].setStyleSheet("""
                        QLabel {
                            font-size: 13px;
                            font-weight: normal;
                            background-color: transparent;
                            padding: 2px;
                            border: none;
                            min-width: 80px;
                        }
                    """)

    def update_cursor_positions(self, cursor_positions: Dict[str, float]):
        """Update cursor position information and calculate delta values."""
        self.cursor_positions = cursor_positions.copy()
        
        # Update cursor 1 position
        if 'c1' in cursor_positions:
            c1_time = cursor_positions['c1']
            if self.is_datetime_axis:
                try:
                    dt = datetime.datetime.utcfromtimestamp(c1_time)
                    time_str = dt.strftime('%H:%M:%S.%f')[:-3]  # Show milliseconds
                    self.cursor1_time_label.setText(f"T1: {time_str}")
                except (ValueError, OSError, OverflowError):
                    self.cursor1_time_label.setText(f"T1: {c1_time:.4f}s")
            else:
                self.cursor1_time_label.setText(f"T1: {c1_time:.4f}s")
        else:
            self.cursor1_time_label.setText("T1: --")
            
        # Update cursor 2 position
        if 'c2' in cursor_positions:
            c2_time = cursor_positions['c2']
            if self.is_datetime_axis:
                try:
                    dt = datetime.datetime.utcfromtimestamp(c2_time)
                    time_str = dt.strftime('%H:%M:%S.%f')[:-3]  # Show milliseconds
                    self.cursor2_time_label.setText(f"T2: {time_str}")
                except (ValueError, OSError, OverflowError):
                    self.cursor2_time_label.setText(f"T2: {c2_time:.4f}s")
            else:
                self.cursor2_time_label.setText(f"T2: {c2_time:.4f}s")
        else:
            self.cursor2_time_label.setText("T2: --")
            
        # Calculate and display delta time and frequency
        if 'c1' in cursor_positions and 'c2' in cursor_positions:
            c1_time = cursor_positions['c1']
            c2_time = cursor_positions['c2']
            delta_time = abs(c2_time - c1_time)
            
            # Display delta time
            if delta_time > 0:
                if delta_time >= 1.0:
                    self.delta_time_label.setText(f"ΔT: {delta_time:.5f}s")
                elif delta_time >= 0.001:
                    self.delta_time_label.setText(f"ΔT: {delta_time*1000:.5f}ms")
                else:
                    self.delta_time_label.setText(f"ΔT: {delta_time*1000000:.5f}μs")
                
                # Calculate and display frequency
                frequency = 1.0 / delta_time
                if frequency >= 1000000:
                    self.frequency_label.setText(f"Freq: {frequency/1000000:.5f}MHz")
                elif frequency >= 1000:
                    self.frequency_label.setText(f"Freq: {frequency/1000:.5f}kHz")
                else:
                    self.frequency_label.setText(f"Freq: {frequency:.5f}Hz")
            else:
                self.delta_time_label.setText("ΔT: 0s")
                self.frequency_label.setText("Freq: ∞Hz")
        else:
            self.delta_time_label.setText("ΔT: --")
            self.frequency_label.setText("Freq: --")
            
        logger.debug(f"Updated cursor positions: {cursor_positions}")

    def clear_cursor_info(self):
        """Clear all cursor information."""
        self.cursor_positions = {}
        self.cursor1_time_label.setText("T1: --")
        self.cursor2_time_label.setText("T2: --")
        self.delta_time_label.setText("ΔT: --")
        self.frequency_label.setText("Freq: --")
