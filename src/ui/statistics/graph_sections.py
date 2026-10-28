"""
One section per graph: creating, ordering and sizing them.

Part of StatisticsPanel; moved from src/ui/statistics_panel.py without changes.
"""

import logging
from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QLabel, QScrollArea, QTableWidget, QHeaderView
from PyQt5.QtCore import Qt
from src.ui.statistics import ClickableGroupBox

logger = logging.getLogger(__name__)


class StatisticsGraphSectionsMixin:
    """One section per graph: creating, ordering and sizing them."""

    def _create_graph_section(self, graph_index: int):
        """Create a new graph section with its own table and controls."""
        # Create group box for this graph with new title format
        section_title = f"📊 Graph {graph_index + 1}"
        graph_section = ClickableGroupBox(section_title, graph_index)
        
        # Main layout for the section - minimal padding for space efficiency
        section_layout = QVBoxLayout(graph_section)
        section_layout.setContentsMargins(2, 8, 2, 2)  # Reduced top padding from 20 to 8
        section_layout.setSpacing(2)  # Reduced spacing between elements
        
        # Create table for this graph
        graph_table = QTableWidget()
        graph_table.setAlternatingRowColors(True)
        graph_table.setSelectionBehavior(QTableWidget.SelectRows)
        graph_table.setSelectionMode(QTableWidget.SingleSelection)
        graph_table.verticalHeader().setVisible(False)
        graph_table.verticalHeader().setDefaultSectionSize(24)  # Reduce row height for compact display
        graph_table.horizontalHeader().setStretchLastSection(False)  # Disable auto-stretch for manual resize
        graph_table.setSortingEnabled(False)
        graph_table.setMaximumHeight(200)  # Limit height
        
        # Disable horizontal scrollbar and enable column resizing
        graph_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        graph_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)  # Allow manual column resize
        
        # Enable context menu for right-click operations
        graph_table.setContextMenuPolicy(Qt.CustomContextMenu)
        graph_table.customContextMenuRequested.connect(
            lambda pos, gidx=graph_index, table=graph_table: self._on_table_context_menu(pos, gidx, table)
        )
        logger.info(f"Context menu ENABLED for Graph {graph_index + 1} table")
        
        # Set up table headers
        self._setup_table_headers_for_graph(graph_table)
        
        section_layout.addWidget(graph_table)
        
        # Store references
        self.graph_sections[graph_index] = graph_section
        self.graph_tables[graph_index] = graph_table
        
        # Add to container layout
        self.container_layout.addWidget(graph_section)
        
        # Apply styling
        self._apply_graph_section_styling(graph_section)
        
        # Apply table styling immediately
        self._apply_table_styling_to_single_table(graph_table)
        
        # Apply current column widths from header table if available
        self._sync_new_table_widths(graph_table)
        
        # Connect drag-and-drop signals
        graph_section.drop_requested.connect(self._on_graph_drop_requested)
        
        logger.debug(f"Created graph section for Graph {graph_index + 1}")

    def _apply_graph_section_styling(self, graph_section):
        """Apply styling to a graph section with minimal vertical padding."""
        # Get theme colors
        theme_colors = {}
        if hasattr(self.parent(), 'theme_manager'):
            theme_colors = self.parent().theme_manager.get_theme_colors()
        
        if not theme_colors:
            theme_colors = {
                'text_primary': '#e6f3ff',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'border': '#4a90e2',
                'primary': '#4a90e2'
            }
        
        # Apply styling with minimal padding for the title
        graph_section.setStyleSheet(f"""
            ClickableGroupBox {{
                font-weight: bold;
                font-size: 13px;
                border: 2px solid {theme_colors.get('border', '#4a90e2')};
                border-radius: 8px;
                margin-top: 0px;
                margin-bottom: 1px;
                padding: 1px;
                background-color: {theme_colors.get('surface', '#2d4a66')};
            }}
            ClickableGroupBox:hover {{
                border: 2px solid {theme_colors.get('primary', '#4a90e2')};
                background-color: {theme_colors.get('surface_variant', '#3a5f7a')};
            }}
            ClickableGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 2px 8px;
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 4px;
                color: {theme_colors.get('primary', '#4a90e2')};
            }}
        """)

    def remove_graph_section(self, graph_index: int):
        """Remove a graph section and all its signals."""
        if graph_index in self.graph_sections:
            # Remove all signals from this graph
            signals_to_remove = []
            for signal_name, signal_info in self.signal_data.items():
                if signal_info['graph_index'] == graph_index:
                    signals_to_remove.append(signal_name)
            
            for signal_name in signals_to_remove:
                del self.signal_data[signal_name]
            
            # Remove the graph section widget
            graph_section = self.graph_sections[graph_index]
            if graph_section:
                self.container_layout.removeWidget(graph_section)
                graph_section.deleteLater()
            
            # Remove from dictionaries
            del self.graph_sections[graph_index]
            if graph_index in self.graph_tables:
                del self.graph_tables[graph_index]
            
            logger.debug(f"Removed graph section for Graph {graph_index + 1}")

    def update_graph_count(self, new_graph_count: int):
        """Update the statistics panel when graph count changes."""
        # Save current column widths before making changes
        saved_widths = self._save_current_column_widths()
        
        current_graph_indices = list(self.graph_sections.keys())
        max_current_index = max(current_graph_indices) if current_graph_indices else -1
        
        # Remove graphs that are no longer needed
        for graph_index in current_graph_indices:
            if graph_index >= new_graph_count:
                self.remove_graph_section(graph_index)
        
        # Ensure we have sections for all needed graphs
        self.ensure_graph_sections(new_graph_count - 1)
        
        # Restore column widths after creating new sections
        if saved_widths:
            self._restore_column_widths_to_all_tables(saved_widths)
        
        logger.debug(f"Updated statistics panel for {new_graph_count} graphs with preserved column widths")

    def ensure_graph_sections(self, max_graph_index: int):
        """Ensure graph sections exist for all graphs up to max_graph_index."""
        total_graphs = max_graph_index + 1
        
        # Save current column widths before creating new sections
        saved_widths = self._save_current_column_widths()
        
        # Create missing graph sections in order
        for graph_idx in range(total_graphs):
            if graph_idx not in self.graph_sections:
                self._create_graph_section(graph_idx)
                logger.debug(f"Auto-created graph section for Graph {graph_idx + 1}")
        
        # Apply saved widths to any newly created tables
        if saved_widths:
            for graph_idx in range(total_graphs):
                if graph_idx in self.graph_tables:
                    table = self.graph_tables[graph_idx]
                    for col, width in saved_widths.items():
                        if col < table.columnCount():
                            table.setColumnWidth(col, width)
        
        logger.debug(f"Ensured {total_graphs} graph sections exist with preserved column widths")

    def _reorder_graph_sections(self):
        """Reorder graph sections in the layout to match their indices."""
        # Remove all graph sections from layout
        for graph_section in self.graph_sections.values():
            if graph_section is not None:  # Skip placeholder entries
                self.container_layout.removeWidget(graph_section)
        
        # Add them back in correct order
        for graph_idx in sorted(self.graph_sections.keys()):
            graph_section = self.graph_sections[graph_idx]
            if graph_section is not None:  # Skip placeholder entries
                self.container_layout.addWidget(graph_section)
        
        logger.debug(f"Reordered {len([s for s in self.graph_sections.values() if s is not None])} graph sections")

    def _equalize_graph_heights(self, total_graphs: int):
        """Set equal heights for all graph sections to fill the panel completely."""
        if total_graphs == 0:
            return
            
        # Get the actual scroll area height dynamically
        scroll_area = self.parent().findChild(QScrollArea) if self.parent() else None
        if scroll_area:
            available_height = scroll_area.height()
        else:
            available_height = 700  # Fallback height
            
        header_height = 80  # Header + title height
        spacing = 1 * (total_graphs - 1)  # Spacing between sections
        
        # Calculate height per graph section to fill the entire panel
        section_height = max(200, (available_height - header_height - spacing) // total_graphs)
        
        # Apply equal height to all graph sections
        for graph_section in self.graph_sections.values():
            graph_section.setMinimumHeight(section_height)
            graph_section.setMaximumHeight(section_height)
            
        logger.debug(f"Equalized {total_graphs} graph sections to {section_height}px each (total: {available_height}px)")

    def set_no_data_message(self):
        """Display a message when no data is available."""
        if not self.signal_data:
            # Create temporary message box
            message_box = QGroupBox("ℹ️ No Data")
            message_layout = QVBoxLayout(message_box)
            
            message_label = QLabel("Connect a data source to view statistics")
            message_label.setAlignment(Qt.AlignCenter)
            # Use theme-appropriate color
            message_label.setStyleSheet("font-style: italic; font-size: 14px;")
            message_layout.addWidget(message_label)
            
            # Add to layout temporarily
            self.container_layout.insertWidget(0, message_box)
            
            # Store reference for removal when data is added
            self._no_data_box = message_box

    def _remove_no_data_message(self):
        """Remove the no data message if it exists."""
        if hasattr(self, '_no_data_box'):
            self.container_layout.removeWidget(self._no_data_box)
            self._no_data_box.deleteLater()
            delattr(self, '_no_data_box')

    def _on_graph_drop_requested(self, from_index: int, to_index: int):
        """Handle graph drop request to reorder graphs."""
        logger.info(f"Graph reorder requested: Graph {from_index + 1} -> Graph {to_index + 1}")
        # Emit signal to parent widget to handle the reordering
        self.graph_reorder_requested.emit(from_index, to_index)
