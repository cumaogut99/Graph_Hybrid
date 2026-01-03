# type: ignore
"""
Parameter List Widget

Displays a scrollable list of available parameters that can be dragged onto graphs.
"""

import logging
from typing import Dict, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, 
    QFrame, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtGui import QFont

from src.ui.parameter_item import ParameterItem

logger = logging.getLogger(__name__)


class ParameterList(QWidget):
    """
    A scrollable list of draggable column items from loaded CSV/data.
    
    Shows all available columns that can be dragged onto graphs.
    """
    
    # Signals
    parameter_drag_started = Signal(str)  # column_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parameter_items = {}  # Store column widgets by name
        self.available_columns = []  # Will be populated from data
        
        self._setup_ui()
        self._show_placeholder()
    
    def _setup_ui(self):
        """Setup the parameter list UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        
        # Search box
        search_container = QWidget()
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(8, 8, 8, 8)
        
        search_label = QLabel("🔍")
        search_label.setFont(QFont("Segoe UI Emoji", 12))
        search_layout.addWidget(search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search parameters...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 6px;
                color: #e6f3ff;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: rgba(74, 144, 226, 0.6);
                background: rgba(255, 255, 255, 0.15);
            }
        """)
        self.search_box.textChanged.connect(self._filter_parameters)
        search_layout.addWidget(self.search_box, 1)
        
        main_layout.addWidget(search_container)
        
        # Scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.05);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(74, 144, 226, 0.7);
            }
        """)
        
        # Container for parameter items
        self.parameters_container = QWidget()
        self.parameters_layout = QVBoxLayout(self.parameters_container)
        self.parameters_layout.setContentsMargins(8, 4, 8, 4)
        self.parameters_layout.setSpacing(6)
        
        scroll_area.setWidget(self.parameters_container)
        main_layout.addWidget(scroll_area, 1)
    
    def _show_placeholder(self):
        """Show placeholder when no data is loaded."""
        # Clear existing items
        for item in self.parameter_items.values():
            item.deleteLater()
        self.parameter_items.clear()
        
        # Clear layout
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add placeholder message
        placeholder = QLabel("📂 No data loaded\n\nLoad a CSV file to see available columns")
        placeholder.setFont(QFont("Segoe UI", 10))
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            color: #a0c0e0;
            padding: 40px 20px;
            background: transparent;
        """)
        self.parameters_layout.addWidget(placeholder)
        self.parameters_layout.addStretch()
    
    def update_columns(self, column_names: List[str], time_column: str = None):
        """
        Update the list with columns from loaded data.
        
        Args:
            column_names: List of column names from the loaded data
            time_column: Name of the time column (will be excluded from list)
        """
        self.available_columns = column_names
        
        # Clear existing items
        for item in self.parameter_items.values():
            item.deleteLater()
        self.parameter_items.clear()
        
        # Clear layout
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not column_names:
            self._show_placeholder()
            return
        
        # Info label
        info_label = QLabel(f"📊 Available Columns ({len(column_names)})")
        info_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        info_label.setStyleSheet("""
            color: #4a90e2;
            padding: 8px 4px 8px 4px;
            background: transparent;
        """)
        self.parameters_layout.addWidget(info_label)
        
        # Add each column as a draggable item
        for col_name in sorted(column_names):
            # Skip time column
            if time_column and col_name == time_column:
                continue
            
            # Create draggable item for this column
            col_item = ParameterItem(col_name, "", self)
            col_item.drag_started.connect(self.parameter_drag_started.emit)
            
            self.parameter_items[col_name] = col_item
            self.parameters_layout.addWidget(col_item)
        
        # Add stretch at end
        self.parameters_layout.addStretch()
        
        logger.info(f"Updated column list: {len(column_names)} columns")
    
    def add_parameter(self, param_name: str):
        """
        Add a new parameter to the list without clearing existing ones.
        
        Args:
            param_name: Name of the parameter to add
        """
        if param_name in self.parameter_items:
            logger.warning(f"Parameter '{param_name}' already exists in list")
            return
        
        # Add to available columns
        if param_name not in self.available_columns:
            self.available_columns.append(param_name)
        
        # Create draggable item
        col_item = ParameterItem(param_name, "", self)
        col_item.drag_started.connect(self.parameter_drag_started.emit)
        
        self.parameter_items[param_name] = col_item
        
        # Find the info label and update count, then insert after it
        info_label = None
        info_label_index = -1
        for i in range(self.parameters_layout.count()):
            item = self.parameters_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QLabel) and "Available Columns" in widget.text():
                    info_label = widget
                    info_label_index = i
                    break
        
        if info_label:
            # Update count in info label
            count = len(self.parameter_items)
            info_label.setText(f"📊 Available Columns ({count})")
            
            # Find correct position to maintain alphabetical order
            sorted_params = sorted(self.parameter_items.keys())
            param_position = sorted_params.index(param_name)
            
            # Find where to insert: after info_label + param_position
            # Count existing parameter items before insertion point
            existing_param_count = 0
            for i in range(info_label_index + 1, self.parameters_layout.count()):
                item = self.parameters_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), ParameterItem):
                    existing_param_count += 1
                    # Check if we should insert before this item
                    existing_param_name = item.widget().parameter_name
                    if existing_param_name > param_name:
                        # Insert before this item
                        self.parameters_layout.insertWidget(i, col_item)
                        return
            
            # If we get here, insert at the end of parameters (before stretch)
            # Find stretch position
            stretch_index = self.parameters_layout.count() - 1
            self.parameters_layout.insertWidget(stretch_index, col_item)
        else:
            # No info label, create one and add parameter
            info_label = QLabel(f"📊 Available Columns ({len(self.parameter_items)})")
            info_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            info_label.setStyleSheet("""
                color: #4a90e2;
                padding: 8px 4px 8px 4px;
                background: transparent;
            """)
            self.parameters_layout.insertWidget(0, info_label)
            self.parameters_layout.insertWidget(1, col_item)
        
        logger.info(f"Added parameter '{param_name}' to list")
    
    def _filter_parameters(self, search_text: str):
        """Filter columns based on search text."""
        search_text = search_text.lower()
        
        for col_name, col_widget in self.parameter_items.items():
            # Check if column name matches
            matches = search_text in col_name.lower()
            col_widget.setVisible(matches)

