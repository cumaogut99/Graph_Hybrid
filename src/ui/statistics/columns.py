"""
Header, cursor info panel and column widths of the statistics tables.

Part of StatisticsPanel; moved from src/ui/statistics_panel.py without changes.
"""

import logging
from typing import Dict
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


class StatisticsColumnsMixin:
    """Header, cursor info panel and column widths of the statistics tables."""

    def _create_cursor_info_panel(self) -> QWidget:
        """Create the compact cursor information panel at the bottom."""
        # Get theme colors
        theme_colors = {}
        if hasattr(self.parent(), 'theme_manager'):
            theme_colors = self.parent().theme_manager.get_theme_colors()
        
        # Determine if this is a light theme
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        text_color = '#ffffff' if not is_light_theme else '#212121'
        
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setFixedHeight(28)  # Slightly smaller
        panel.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(4, 0, 4, 0)  # No top/bottom margins
        layout.setSpacing(6)  # Compact spacing
        
        # T1 (Cursor 1)
        self.cursor1_time_label = QLabel("T1: --")
        self.cursor1_time_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 10px;
                font-weight: 600;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(74, 144, 226, 0.15);
                min-height: 14px;
            }}
        """)
        layout.addWidget(self.cursor1_time_label)
        
        # T2 (Cursor 2)
        self.cursor2_time_label = QLabel("T2: --")
        self.cursor2_time_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 10px;
                font-weight: 600;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(226, 74, 144, 0.15);
                min-height: 14px;
            }}
        """)
        layout.addWidget(self.cursor2_time_label)
        
        # ΔT (Delta time)
        self.delta_time_label = QLabel("ΔT: --")
        self.delta_time_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 10px;
                font-weight: 600;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(144, 226, 74, 0.15);
                min-height: 14px;
            }}
        """)
        layout.addWidget(self.delta_time_label)
        
        # Freq (Frequency)
        self.frequency_label = QLabel("Freq: --")
        self.frequency_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 10px;
                font-weight: 600;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(226, 144, 74, 0.15);
                min-height: 14px;
            }}
        """)
        layout.addWidget(self.frequency_label)
        
        layout.addStretch()
        
        return panel

    def _create_statistics_header(self) -> QWidget:
        """Create the statistics header with resizable columns using QSplitter."""
        header_widget = QWidget()
        header_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(5, 3, 5, 3)
        header_layout.setSpacing(0)
        
        # Create splitter for resizable columns
        self.header_splitter = QSplitter(Qt.Horizontal)
        self.header_splitter.setChildrenCollapsible(False)
        
        # Signal name column
        name_header = QLabel("Signal")
        name_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Use theme-appropriate styling for header
        name_header.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                padding: 5px 8px;
                background-color: rgba(128, 128, 128, 0.1);
                border-radius: 6px;
                border-right: 1px solid rgba(128, 128, 128, 0.3);
            }
        """)
        name_header.setMinimumWidth(self.column_widths['signal'])
        self.header_splitter.addWidget(name_header)
        
        # Statistics headers - dynamic based on cursor mode
        stats_info = self._get_stats_info_for_mode()
        
        column_keys = ['signal']
        
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                stat_header = QLabel(f"{icon} {display_name}")
                stat_header.setAlignment(Qt.AlignCenter)
                # Use theme-appropriate styling for stat headers
                stat_header.setStyleSheet("""
                    QLabel {
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px 8px;
                        background-color: rgba(128, 128, 128, 0.15);
                        border-radius: 6px;
                        border-right: 1px solid rgba(128, 128, 128, 0.3);
                    }
                """)
                stat_header.setMinimumWidth(self.column_widths.get(stat_key, 80))
                self.header_splitter.addWidget(stat_header)
                column_keys.append(stat_key)

        # Set initial sizes for the splitter
        initial_sizes = [self.column_widths.get(key, 80) for key in column_keys]
        self.header_splitter.setSizes(initial_sizes)
        
        # Connect splitter signals to update column widths
        self.header_splitter.splitterMoved.connect(self._on_column_resized)
        
        header_layout.addWidget(self.header_splitter)
        
        # Use theme-appropriate styling for header widget
        header_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(128, 128, 128, 0.05);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 8px;
                margin: 2px 0px;
            }
            QSplitter::handle {
                background-color: rgba(128, 128, 128, 0.3);
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: rgba(74, 144, 226, 0.8);
            }
        """)
        
        return header_widget

    def _on_column_resized(self, pos: int, index: int):
        """Handle column resize events from the header splitter."""
        if not hasattr(self, 'header_splitter'):
            return
            
        # Get current sizes from splitter
        sizes = self.header_splitter.sizes()
        
        # Update column widths dictionary
        stats_info = self._get_stats_info_for_mode()
        column_keys = ['signal'] + [stat_key for stat_key, _, _ in stats_info 
                                   if stat_key in ['c1', 'c2'] or stat_key in self.visible_stats]
        
        for i, size in enumerate(sizes):
            if i < len(column_keys):
                self.column_widths[column_keys[i]] = size
        
        # Update all signal rows to match new column widths
        self._update_all_signal_row_widths()
        
        logger.debug(f"Column resized: {self.column_widths}")

    def _update_all_signal_row_widths(self):
        """Update all signal rows to match current column widths."""
        if not hasattr(self, 'header_splitter'):
            return
            
        header_sizes = self.header_splitter.sizes()
        
        for signal_name, signal_info in self.signal_data.items():
            row_widget = signal_info.get('row_widget')
            if row_widget and hasattr(row_widget, 'row_splitter'):
                # Temporarily disconnect signals to avoid recursion
                try:
                    row_widget.row_splitter.splitterMoved.disconnect()
                except:
                    pass
                
                # Update splitter sizes to match header
                row_widget.row_splitter.setSizes(header_sizes)
                
                # Reconnect signals
                row_widget.row_splitter.splitterMoved.connect(
                    lambda pos, idx, splitter=row_widget.row_splitter: self._on_row_splitter_moved(pos, idx, splitter)
                )

    def _on_row_splitter_moved(self, pos: int, index: int, source_splitter):
        """Handle row splitter movement and sync with header."""
        if not hasattr(self, 'header_splitter'):
            return
            
        # Get sizes from the moved splitter
        new_sizes = source_splitter.sizes()
        
        # Update header splitter to match
        try:
            self.header_splitter.splitterMoved.disconnect()
        except:
            pass
            
        self.header_splitter.setSizes(new_sizes)
        self.header_splitter.splitterMoved.connect(self._on_column_resized)
        
        # Update column widths
        stats_info = self._get_stats_info_for_mode()
        column_keys = ['signal'] + [stat_key for stat_key, _, _ in stats_info 
                                   if stat_key in ['c1', 'c2'] or stat_key in self.visible_stats]
        
        for i, size in enumerate(new_sizes):
            if i < len(column_keys):
                self.column_widths[column_keys[i]] = size
        
        # Sync all other row splitters
        for signal_name, signal_info in self.signal_data.items():
            row_widget = signal_info.get('row_widget')
            if row_widget and hasattr(row_widget, 'row_splitter') and row_widget.row_splitter != source_splitter:
                try:
                    row_widget.row_splitter.splitterMoved.disconnect()
                except:
                    pass
                    
                row_widget.row_splitter.setSizes(new_sizes)
                
                row_widget.row_splitter.splitterMoved.connect(
                    lambda pos, idx, splitter=row_widget.row_splitter: self._on_row_splitter_moved(pos, idx, splitter)
                )

    def _get_stats_info_for_mode(self):
        """Get statistics info for dual cursor mode (permanently)."""
        # Cursor mode is always 'dual'
        return [
            ('c1', '🔴', 'C1'),  # Red circle for cursor 1
            ('c2', '🔵', 'C2'),  # Blue circle for cursor 2
            ('min', '📉', 'Min'),
            ('mean', '📊', 'Mean'),
            ('max', '📈', 'Max'),
            ('rms', '⚡', 'RMS'),
            ('std', '📏', 'Std'),
            ('duty_cycle', '⏱️', 'Duty %')
        ]

    def _recreate_header(self):
        """Recreate the statistics header."""
        # Remove old header
        if hasattr(self, 'header_widget') and self.header_widget:
            self.main_layout.removeWidget(self.header_widget)
            self.header_widget.deleteLater()
        
        # Create new header
        self.header_widget = self._create_statistics_header()
        self.main_layout.insertWidget(1, self.header_widget)  # Insert after title

    def _recreate_all_signal_rows(self):
        """Recreate all signal rows to match new cursor mode structure."""
        # Store current signal data
        current_data = self.signal_data.copy()
        
        # Clear all tables
        for table in self.graph_tables.values():
            table.setRowCount(0)
        self.signal_data.clear()
        
        # Update headers for all tables
        for table in self.graph_tables.values():
            self._setup_table_headers_for_graph(table)
        
        # Recreate all signals with new structure
        for full_signal_name, data in current_data.items():
            graph_index = data['graph_index']
            signal_name = data['signal_name']
            color = data.get('color', '#ffffff')
            
            # Add signal back to appropriate table
            self.add_signal(full_signal_name, graph_index, signal_name, color)

    def _create_common_header(self):
        """Create common header table that syncs with all graph tables."""
        # Get theme colors for styling
        theme_colors = {}
        if hasattr(self.parent(), 'theme_manager'):
            theme_colors = self.parent().theme_manager.get_theme_colors()
        
        if theme_colors is None:
            # Fallback colors for space theme
            theme_colors = {
                'text_primary': '#e6f3ff',
                'text_secondary': '#ffffff',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'border': '#4a90e2'
            }
        
        # Determine if this is a light theme
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        
        # Adjust colors for light theme
        if is_light_theme:
            text_color = '#212121'
            border_color_base = '0, 0, 0'  # Black for light theme
            bg_color_base = '0, 0, 0'      # Black for light theme
            border_opacity = '0.2'
            bg_opacity = '0.05'
        else:
            text_color = theme_colors.get('text_primary', '#ffffff')
            border_color_base = '255, 255, 255' # White for dark themes
            bg_color_base = '255, 255, 255'     # White for dark themes
            border_opacity = '0.2'
            bg_opacity = '0.05'
        
        # Create header table
        self.header_table = QTableWidget()
        self.header_table.setRowCount(1)
        self.header_table.setFixedHeight(40)  # Fixed height for header
        
        # Disable horizontal scrollbar and enable column resizing
        self.header_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        
        # Setup headers based on current mode and visible stats
        headers = ['📊 Signal']  # Removed color column
        stats_info = self._get_stats_info_for_mode()
        
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                headers.append(f"{icon} {display_name}")
        
        self.header_table.setColumnCount(len(headers))
        self.header_table.setHorizontalHeaderLabels(headers)
        
        # Hide vertical header and make it non-interactive
        self.header_table.verticalHeader().setVisible(False)
        self.header_table.setSelectionMode(QTableWidget.NoSelection)
        self.header_table.setFocusPolicy(Qt.NoFocus)
        
        # Fill with empty items to show headers only
        for col in range(len(headers)):
            item = QTableWidgetItem("")
            item.setFlags(Qt.NoItemFlags)  # Make non-interactive
            self.header_table.setItem(0, col, item)
        
        # Set initial column widths with minimum sizes (only if no previous widths exist)
        # Check if we have saved column widths to restore
        has_saved_widths = hasattr(self, '_temp_saved_widths') and self._temp_saved_widths
        
        if not has_saved_widths:
            # Use default widths only for first time initialization
            logger.debug("Using default column widths for header table")
            self.header_table.setColumnWidth(0, 150)  # Signal name - increased width
            for i in range(1, len(headers)):
                self.header_table.setColumnWidth(i, 80)  # Statistics
        else:
            # Restore saved widths
            logger.debug(f"Restoring saved widths in header creation: {self._temp_saved_widths}")
            for col, width in self._temp_saved_widths.items():
                if col < len(headers):
                    self.header_table.setColumnWidth(col, width)
                    logger.debug(f"Set header column {col} to width {width}")
            # Clear temporary saved widths
            delattr(self, '_temp_saved_widths')
        
        # Set minimum column widths to prevent columns from becoming too small
        header = self.header_table.horizontalHeader()
        header.setMinimumSectionSize(50)  # Minimum width for any column
        
        # Style the header table with theme-appropriate colors
        header_table_style = f"""
            QTableWidget {{
                background-color: rgba({bg_color_base}, {bg_opacity});
                border: 1px solid rgba({border_color_base}, {border_opacity});
                border-radius: 8px;
                gridline-color: rgba({border_color_base}, 0.3);
                color: {text_color};
            }}
            QHeaderView::section {{
                background-color: rgba({bg_color_base}, 0.15);
                border: 1px solid rgba({border_color_base}, 0.3);
                padding: 3px 8px;
                font-weight: bold;
                font-size: 12px;
                color: {text_color};
            }}
            QHeaderView::section:first {{
                font-size: 13px;
                color: {text_color};
            }}
            QTableWidget::item {{
                color: {text_color};
                background-color: transparent;
            }}
        """
        self.header_table.setStyleSheet(header_table_style)
        
        # Connect header resize signal to sync all tables
        header = self.header_table.horizontalHeader()
        header.sectionResized.connect(self._sync_column_widths)
        
        return self.header_table

    def _sync_column_widths(self, logical_index: int, old_size: int, new_size: int):
        """Sync column widths across all tables when header table is resized."""
        # Ensure minimum width is respected
        min_width = 50 if logical_index > 1 else (120 if logical_index == 0 else 30)
        new_size = max(new_size, min_width)
        
        # Update all graph tables to match header table column width
        for table in self.graph_tables.values():
            if logical_index < table.columnCount():
                # Temporarily disconnect signals to avoid recursion
                header = table.horizontalHeader()
                try:
                    header.sectionResized.disconnect()
                except:
                    pass  # Ignore if no connections exist
                
                # Set the new width
                table.setColumnWidth(logical_index, new_size)
                
                # Reconnect signals
                header.sectionResized.connect(lambda idx, old, new, t=table: self._on_graph_table_resized(t, idx, old, new))

    def _on_graph_table_resized(self, source_table: QTableWidget, logical_index: int, old_size: int, new_size: int):
        """Handle column resize from any graph table and sync to header and other tables."""
        # Ensure minimum width is respected
        min_width = 50 if logical_index > 1 else (120 if logical_index == 0 else 30)
        new_size = max(new_size, min_width)
        
        # Update header table
        if hasattr(self, 'header_table') and logical_index < self.header_table.columnCount():
            header = self.header_table.horizontalHeader()
            try:
                header.sectionResized.disconnect()
            except:
                pass
            self.header_table.setColumnWidth(logical_index, new_size)
            header.sectionResized.connect(self._sync_column_widths)
        
        # Update all other graph tables
        for table in self.graph_tables.values():
            if table != source_table and logical_index < table.columnCount():
                header = table.horizontalHeader()
                try:
                    header.sectionResized.disconnect()
                except:
                    pass
                table.setColumnWidth(logical_index, new_size)
                header.sectionResized.connect(lambda idx, old, new, t=table: self._on_graph_table_resized(t, idx, old, new))

    def _update_common_header(self):
        """Update the common header when cursor mode or visible stats change."""
        if hasattr(self, 'common_header') and self.common_header:
            # Save current column widths before recreating header
            saved_widths = self._save_current_column_widths()
            
            # Store saved widths temporarily for _create_common_header to use
            if saved_widths:
                self._temp_saved_widths = saved_widths
            
            # Remove old header
            self.main_layout.removeWidget(self.common_header)
            self.common_header.deleteLater()
            
            # Create new header (will use _temp_saved_widths if available)
            self.common_header = self._create_common_header()
            self.main_layout.insertWidget(0, self.common_header)

    def _setup_table_headers(self):
        """Setup table headers for all graph tables based on cursor mode and visible statistics."""
        # Save current column widths before updating headers
        saved_widths = self._save_current_column_widths()
        
        # Update headers for all existing graph tables
        for table in self.graph_tables.values():
            self._setup_table_headers_for_graph(table)
        
        # Restore column widths after header updates
        if saved_widths:
            self._restore_column_widths_to_all_tables(saved_widths)

    def _sync_new_table_widths(self, table: QTableWidget):
        """Sync new table column widths with current header table widths."""
        if hasattr(self, 'header_table') and self.header_table:
            for col in range(min(table.columnCount(), self.header_table.columnCount())):
                current_width = self.header_table.columnWidth(col)
                table.setColumnWidth(col, current_width)

    def _setup_table_headers_for_graph(self, table: QTableWidget):
        """Setup table headers for a specific graph table."""
        headers = ['Signal']  # Removed color column
        
        # Add statistics columns based on visible stats
        stats_info = self._get_stats_info_for_mode()
        for stat_key, icon, display_name in stats_info:
            is_cursor_stat = stat_key in ['c1', 'c2']
            if is_cursor_stat or stat_key in self.visible_stats:
                headers.append(f"{icon} {display_name}")
        
        table.setColumnCount(len(headers))
        
        # Hide table headers since we have common header
        table.horizontalHeader().setVisible(False)
        
        # Set column widths to match header table if it exists
        if hasattr(self, 'header_table') and self.header_table:
            for i in range(min(len(headers), self.header_table.columnCount())):
                width = self.header_table.columnWidth(i)
                table.setColumnWidth(i, width)
        else:
            # Fallback to default widths
            table.setColumnWidth(0, 150)  # Signal name column - increased width since no color column
            for i in range(1, len(headers)):
                table.setColumnWidth(i, 80)  # Statistics columns
        
        # Set minimum column widths to prevent columns from becoming too small
        table_header = table.horizontalHeader()
        table_header.setMinimumSectionSize(50)  # Minimum width for any column
        
        # Connect resize signal for syncing
        header = table.horizontalHeader()
        # Disconnect any existing connections first
        try:
            header.sectionResized.disconnect()
        except:
            pass
        # Connect to sync function
        header.sectionResized.connect(lambda idx, old, new, t=table: self._on_graph_table_resized(t, idx, old, new))

    def _save_current_column_widths(self) -> Dict[int, int]:
        """Save current column widths from header table."""
        saved_widths = {}
        if hasattr(self, 'header_table') and self.header_table:
            for col in range(self.header_table.columnCount()):
                saved_widths[col] = self.header_table.columnWidth(col)
            logger.debug(f"Saved column widths: {saved_widths}")
        return saved_widths

    def _restore_column_widths_to_all_tables(self, saved_widths: Dict[int, int]):
        """Restore column widths to header table and all graph tables."""
        logger.debug(f"Restoring column widths: {saved_widths}")
        
        # Restore to header table
        if hasattr(self, 'header_table') and self.header_table and saved_widths:
            for col, width in saved_widths.items():
                if col < self.header_table.columnCount():
                    self.header_table.setColumnWidth(col, width)
                    logger.debug(f"Restored header column {col} to width {width}")
        
        # Restore to all graph tables
        for table in self.graph_tables.values():
            if table and saved_widths:
                for col, width in saved_widths.items():
                    if col < table.columnCount():
                        table.setColumnWidth(col, width)

    def save_column_widths(self) -> Dict[str, int]:
        """Save current column widths to a dictionary."""
        return self.column_widths.copy()

    def restore_column_widths(self, widths: Dict[str, int]):
        """Restore column widths from a dictionary."""
        self.column_widths.update(widths)
        
        # Update header splitter if it exists
        if hasattr(self, 'header_splitter'):
            stats_info = self._get_stats_info_for_mode()
            column_keys = ['signal'] + [stat_key for stat_key, _, _ in stats_info 
                                       if stat_key in ['c1', 'c2'] or stat_key in self.visible_stats]
            
            sizes = [self.column_widths.get(key, 80) for key in column_keys]
            self.header_splitter.setSizes(sizes)
            
            # Update all signal rows
            self._update_all_signal_row_widths()
        
        logger.debug(f"Restored column widths: {widths}")

    def set_visible_stats(self, visible_stats: set):
        """Update which statistics are visible."""
        self.visible_stats = visible_stats
        logger.debug(f"Updated visible statistics: {visible_stats}")
        
        # Save current column widths before making changes
        saved_widths = self._save_current_column_widths()
        
        # Update common header
        self._update_common_header()
        
        # Update table headers for all graphs
        self._setup_table_headers()
        
        # Recreate all signal rows to match new headers
        self._recreate_all_signal_rows()
        
        # Restore column widths after all changes
        if saved_widths:
            self._restore_column_widths_to_all_tables(saved_widths)
            logger.debug(f"Restored column widths after visible stats change: {saved_widths}")
