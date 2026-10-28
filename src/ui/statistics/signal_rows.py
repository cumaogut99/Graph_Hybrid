"""
Signal rows: adding, removing, colors and the row context menu.

Part of StatisticsPanel; moved from src/ui/statistics_panel.py without changes.
"""

import logging
from typing import Dict
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QColorDialog, QSizePolicy, QSplitter, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from src.ui.statistics import ClickableColorLabel, SignalRowWidget

logger = logging.getLogger(__name__)


class StatisticsSignalRowsMixin:
    """Signal rows: adding, removing, colors and the row context menu."""

    def add_signal(self, full_signal_name: str, graph_index: int, signal_name: str, color: str):
        """
        Add a new signal to the appropriate graph table.
        
        Args:
            full_signal_name: Full signal name with graph suffix (e.g., "RPM (G1)")
            graph_index: Index of the graph this signal belongs to
            signal_name: Base signal name (e.g., "RPM")
            color: Color for visual identification
        """
        # Ensure graph section exists
        if graph_index not in self.graph_sections:
            self._create_graph_section(graph_index)
        
        # Get the table for this graph
        table = self.graph_tables[graph_index]
        
        # Add new row to table
        row_count = table.rowCount()
        table.insertRow(row_count)
        
        # Signal name
        name_item = QTableWidgetItem(signal_name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
        table.setItem(row_count, 0, name_item)
        
        # Statistics columns - initialize with empty values (no color column anymore)
        col_index = 1
        stats_info = self._get_stats_info_for_mode()
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                stat_item = QTableWidgetItem("-")
                stat_item.setFlags(stat_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                stat_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_count, col_index, stat_item)
                col_index += 1
        
        # Store signal data (no color_button anymore)
        self.signal_data[full_signal_name] = {
            'graph_index': graph_index,
            'signal_name': signal_name,
            'color': color,
            'row_index': row_count,
            'table': table
        }
        
        logger.debug(f"Added signal {full_signal_name} to Graph {graph_index + 1} table at row {row_count}")
        
        # Auto-resize columns to content but respect minimum widths
        table.resizeColumnsToContents()
        
        # Ensure minimum column widths are maintained
        for col in range(table.columnCount()):
            current_width = table.columnWidth(col)
            min_width = 150 if col == 0 else 80  # Signal name wider, stats standard
            if current_width < min_width:
                table.setColumnWidth(col, min_width)

    def _change_signal_color(self, full_signal_name: str):
        """Open color dialog to change signal color (called from context menu)."""
        if full_signal_name not in self.signal_data:
            logger.warning(f"Signal '{full_signal_name}' not found in signal_data")
            return
            
        current_color = QColor(self.signal_data[full_signal_name]['color'])
        base_signal_name = self.signal_data[full_signal_name]['signal_name']
        
        # Create color dialog with custom styling for better readability
        color_dialog = QColorDialog(current_color, self)
        color_dialog.setWindowTitle(f"Select color for {base_signal_name}")
        
        # Apply custom stylesheet for better text visibility
        color_dialog.setStyleSheet("""
            QColorDialog {
                background-color: #ffffff;
            }
            QLabel {
                color: #000000;
                background-color: transparent;
                font-size: 12px;
            }
            QPushButton {
                color: #000000;
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QSpinBox, QLineEdit {
                color: #000000;
                background-color: #ffffff;
                border: 1px solid #a0a0a0;
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #d0d0d0;
            }
            QDialogButtonBox QPushButton {
                min-width: 70px;
            }
        """)
        
        # Show dialog and get result
        if color_dialog.exec_() == QColorDialog.Accepted:
            new_color = color_dialog.selectedColor()
            if new_color.isValid():
                new_color_hex = new_color.name()
                # Update stored color
                self.signal_data[full_signal_name]['color'] = new_color_hex
                
                # Emit signal for plot and legend update
                self.signal_color_changed.emit(base_signal_name, new_color_hex)
                logger.info(f"Changed color for signal '{base_signal_name}' to {new_color_hex}")

    def _create_signal_row(self, signal_name: str, color: str, graph_index: int = 0) -> QWidget:
        """Create a horizontal row for a signal with all its statistics using QSplitter."""
        row_widget = SignalRowWidget(signal_name, graph_index)
        row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Connect context menu signal
        row_widget.context_menu_requested.connect(
            lambda pos, sname, gidx: self._show_signal_context_menu(pos, sname, gidx, row_widget)
        )
        
        row_layout = QVBoxLayout(row_widget)
        row_layout.setContentsMargins(5, 3, 5, 3)
        row_layout.setSpacing(0)
        
        # Create splitter for resizable columns that matches header
        row_splitter = QSplitter(Qt.Horizontal)
        row_splitter.setChildrenCollapsible(False)
        row_widget.row_splitter = row_splitter  # Store reference for width updates
        
        # Signal name widget with color indicator
        name_widget = QWidget()
        name_layout = QHBoxLayout(name_widget)
        name_layout.setContentsMargins(5, 0, 5, 0)
        name_layout.setSpacing(5)
        
        # Color indicator
        color_indicator = ClickableColorLabel(color)
        color_indicator.setFixedSize(12, 12)

        # Connect the color change signal to the panel's signal
        color_indicator.color_changed.connect(
            lambda new_color, s_name=signal_name: self.signal_color_changed.emit(s_name, new_color)
        )
        
        name_layout.addWidget(color_indicator)
        
        # Signal name (full name, no truncation)
        name_label = QLabel(signal_name)
        name_label.setWordWrap(False)
        # Use theme-appropriate color for signal name
        name_label.setStyleSheet("")  # Will inherit from parent styling
        name_label.setToolTip(f"{signal_name}\n\nRight-click to remove from graph")  # Show full name and hint in tooltip
        
        name_layout.addWidget(name_label)
        name_layout.addStretch()
        
        name_widget.setMinimumWidth(self.column_widths['signal'])
        row_splitter.addWidget(name_widget)
        
        # Statistics values in separate widgets that match header columns
        stats_info = self._get_stats_info_for_mode()
        
        labels_dict = {}
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                value_widget = QWidget()
                value_layout = QHBoxLayout(value_widget)
                value_layout.setContentsMargins(5, 0, 5, 0)
                
                value_label = self._create_value_label("--")
                value_label.setObjectName(f"value_{stat_key}")
                value_layout.addWidget(value_label)
                
                value_widget.setMinimumWidth(self.column_widths.get(stat_key, 80))
                row_splitter.addWidget(value_widget)
                labels_dict[stat_key] = value_label
        
        # Sync splitter sizes with header if it exists
        if hasattr(self, 'header_splitter'):
            row_splitter.setSizes(self.header_splitter.sizes())
        
        # Connect row splitter to sync with header
        row_splitter.splitterMoved.connect(
            lambda pos, idx, splitter=row_splitter: self._on_row_splitter_moved(pos, idx, splitter)
        )
        
        row_layout.addWidget(row_splitter)
        
        # Use theme-appropriate styling for row widget
        row_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 8px;
                padding: 2px;
            }
            QSplitter::handle {
                background-color: transparent;
                width: 2px;
            }
        """)
        
        # Store labels in the widget for easy access
        row_widget.labels_dict = labels_dict
        
        return row_widget

    def _create_value_label(self, value: str) -> QLabel:
        """Create a responsive value label for statistics."""
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Use theme-appropriate styling - will inherit from parent
        value_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                background-color: rgba(128, 128, 128, 0.1);
                padding: 5px 8px;
                border-radius: 4px;
                min-width: 60px;
                max-width: 150px;
            }
        """)
        return value_label

    def _create_stat_widget(self, icon: str, name: str, value: str) -> QWidget:
        """Create a compact statistic widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 5, 8, 5)
        
        # Header with icon and name
        header_label = QLabel(f"{icon} {name}")
        header_label.setAlignment(Qt.AlignCenter)
        # Use theme-appropriate color for header
        header_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-weight: bold;
            }
        """)
        layout.addWidget(header_label)
        
        # Value label
        value_label = QLabel(value)
        value_label.setObjectName(f"value_{name.lower()}")
        value_label.setAlignment(Qt.AlignCenter)
        # Use theme-appropriate styling
        value_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                background-color: rgba(128, 128, 128, 0.1);
                padding: 3px;
                border-radius: 4px;
                min-width: 60px;
            }
        """)
        layout.addWidget(value_label)
        
        widget.setMinimumWidth(80)
        return widget

    def _extract_labels_from_row(self, row_widget: QWidget) -> Dict[str, QLabel]:
        """Extract value labels from a signal row widget."""
        return getattr(row_widget, 'labels_dict', {})

    def remove_signal(self, signal_name: str):
        """
        Remove a signal from its graph table.
        
        Args:
            signal_name: Name of the signal to remove (with graph suffix)
        """
        if signal_name in self.signal_data:
            signal_info = self.signal_data[signal_name]
            row_index = signal_info['row_index']
            table = signal_info['table']
            graph_index = signal_info['graph_index']
            
            # Remove row from table
            table.removeRow(row_index)
            
            # Update row indices for remaining signals in the same graph
            for other_signal, other_info in self.signal_data.items():
                if (other_info['graph_index'] == graph_index and 
                    other_info['row_index'] > row_index):
                    other_info['row_index'] -= 1
            
            # Clean up references
            del self.signal_data[signal_name]
            
            logger.debug(f"Removed signal: {signal_name}")
            
            # Auto-resize columns to content but respect minimum widths
            table.resizeColumnsToContents()
            
            # Ensure minimum column widths are maintained
            for col in range(table.columnCount()):
                current_width = table.columnWidth(col)
                min_width = 50 if col > 1 else (120 if col == 0 else 40)
                if current_width < min_width:
                    table.setColumnWidth(col, min_width)

    def clear_all(self):
        """Remove all signals from all graph tables."""
        # Clear all graph tables
        for table in self.graph_tables.values():
            table.setRowCount(0)
        
        # Clear signal data
        self.signal_data.clear()
        
        logger.debug("Cleared all statistics")

    def get_signal_count(self) -> int:
        """Get the number of signals currently displayed."""
        return len(self.signal_data)

    def has_signal(self, signal_name: str) -> bool:
        """Check if a signal is currently displayed."""
        return signal_name in self.signal_data

    def _on_table_context_menu(self, pos, graph_index: int, table):
        """
        Handle context menu request on table.
        
        Args:
            pos: Position where menu was requested (relative to table)
            graph_index: Index of the graph
            table: The QTableWidget that was right-clicked
        """
        # Get the row that was clicked
        row = table.rowAt(pos.y())
        if row < 0:
            logger.debug(f"Context menu requested outside of table rows (graph {graph_index})")
            return
        
        # Get signal name from the first column
        name_item = table.item(row, 0)
        if not name_item:
            logger.warning(f"No signal name found at row {row} in graph {graph_index}")
            return
        
        signal_name = name_item.text()
        logger.info(f"Context menu triggered for signal '{signal_name}' at row {row}, graph {graph_index}")
        
        # Show the context menu
        self._show_signal_context_menu(pos, signal_name, graph_index, table)

    def _show_signal_context_menu(self, pos, signal_name: str, graph_index: int, widget):
        """
        Show context menu for signal with various options.
        
        Args:
            pos: Position where menu was requested
            signal_name: Name of the signal
            graph_index: Index of the graph containing this signal
            widget: The widget that was right-clicked
        """
        logger.info(f"Showing context menu for signal: {signal_name}, graph: {graph_index}")
        
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QCursor
        
        # Create context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d4a66;
                color: #e6f3ff;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: rgba(74, 144, 226, 0.3);
            }
            QMenu::separator {
                height: 1px;
                background-color: rgba(74, 144, 226, 0.3);
                margin: 4px 0px;
            }
        """)
        
        # === SIGNAL OPERATIONS ===
        # Add "Change Color" action
        change_color_action = menu.addAction("🎨 Change Color")
        change_color_action.setToolTip(f"Change color for '{signal_name}'")
        
        # Add "Remove from graph" action
        remove_action = menu.addAction("🗑️ Remove from Graph")
        remove_action.setToolTip(f"Remove '{signal_name}' from graph {graph_index}")
        
        # Add separator for future options
        menu.addSeparator()
        
        # === PLACEHOLDER FOR FUTURE FEATURES ===
        # You can add more actions here later, for example:
        # export_action = menu.addAction("💾 Export Data")
        # copy_action = menu.addAction("📋 Copy Statistics")
        # etc.
        
        # Show menu at cursor position (global coordinates)
        global_pos = widget.mapToGlobal(pos)
        logger.debug(f"Showing menu at position: {global_pos}")
        action = menu.exec_(global_pos)
        
        # Handle selected action
        if action == change_color_action:
            logger.info(f"User selected: Change color for signal '{signal_name}'")
            # Find full signal name with graph suffix
            full_signal_name = None
            for fname, sdata in self.signal_data.items():
                if sdata['signal_name'] == signal_name and sdata['graph_index'] == graph_index:
                    full_signal_name = fname
                    break
            
            if full_signal_name:
                self._change_signal_color(full_signal_name)
            else:
                logger.warning(f"Could not find full signal name for '{signal_name}' in graph {graph_index}")
        
        elif action == remove_action:
            logger.info(f"User selected: Remove signal '{signal_name}' from graph {graph_index}")
            self.signal_remove_requested.emit(signal_name, graph_index)
        # elif action == export_action:  # Future feature
        #     self._export_signal_data(signal_name)
        # elif action == copy_action:  # Future feature
        #     self._copy_signal_stats(signal_name)
