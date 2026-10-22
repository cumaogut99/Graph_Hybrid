# type: ignore
"""
Parameter List Widget

Displays a scrollable list of available parameters that can be dragged onto graphs.

Uses one QListWidget instead of a widget per parameter: files can have 1000+
channels, and creating, styling and laying out that many widgets froze the UI.
"""

import logging
from typing import Dict, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal as Signal
from PyQt5.QtGui import QDrag, QFont

logger = logging.getLogger(__name__)

_LIST_STYLE = """
    QListWidget {
        border: none;
        background: transparent;
        outline: none;
    }
    QListWidget::item {
        color: #e6f3ff;
        font-size: 11px;
        font-weight: 500;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(74, 144, 226, 0.2), stop:1 rgba(74, 144, 226, 0.1));
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-radius: 6px;
        padding: 4px 6px;
        margin: 2px 8px;
    }
    QListWidget::item:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.15));
        border-color: rgba(74, 144, 226, 0.6);
    }
    QListWidget::item:selected {
        background: rgba(74, 144, 226, 0.45);
        border-color: rgba(74, 144, 226, 0.8);
    }
    QScrollBar:vertical {
        background: rgba(255, 255, 255, 0.05);
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: rgba(74, 144, 226, 0.5);
        border-radius: 6px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background: rgba(74, 144, 226, 0.7);
    }
"""


class _DraggableParameterListWidget(QListWidget):
    """List whose items are dragged onto graphs as "application/x-parameter"."""

    drag_started = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # All rows have the same height: lets the view skip measuring them
        self.setUniformItemSizes(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setStyleSheet(_LIST_STYLE)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)

        mime_data = QMimeData()
        mime_data.setText(name)
        mime_data.setData("application/x-parameter", name.encode())

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        rect = self.visualItemRect(item)
        drag.setPixmap(self.viewport().grab(rect))

        self.drag_started.emit(name)
        logger.debug(f"Drag started for parameter: {name}")
        drag.exec_(Qt.CopyAction | Qt.MoveAction)


class ParameterList(QWidget):
    """
    A scrollable list of draggable column items from loaded CSV/data.

    Shows all available columns that can be dragged onto graphs.
    """

    # Signals
    parameter_drag_started = Signal(str)  # column_name

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parameter_items: Dict[str, QListWidgetItem] = {}  # Items by name
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
        search_layout.setContentsMargins(8, 6, 8, 4)

        search_label = QLabel("🔍")
        search_label.setStyleSheet("font-size: 11px;")
        search_layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search parameters...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 4px 6px;
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

        # Column count header
        self.info_label = QLabel()
        self.info_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.info_label.setStyleSheet("""
            color: #4a90e2;
            padding: 4px 10px;
            background: transparent;
        """)
        main_layout.addWidget(self.info_label)

        # Placeholder when no data is loaded
        self.placeholder = QLabel("📂 No data loaded\n\nLoad a CSV file to see available columns")
        self.placeholder.setFont(QFont("Segoe UI", 8))
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("""
            color: #a0c0e0;
            padding: 40px 20px;
            background: transparent;
        """)
        main_layout.addWidget(self.placeholder)

        self.list_widget = _DraggableParameterListWidget()
        self.list_widget.drag_started.connect(self.parameter_drag_started.emit)
        main_layout.addWidget(self.list_widget, 1)

    def _show_placeholder(self):
        """Show placeholder when no data is loaded."""
        self.list_widget.clear()
        self.parameter_items.clear()
        self.list_widget.hide()
        self.info_label.hide()
        self.placeholder.show()

    def _new_item(self, name: str) -> QListWidgetItem:
        item = QListWidgetItem(f"📊  {name}")
        item.setData(Qt.UserRole, name)
        item.setToolTip(f"{name}\n\nDrag and drop onto a graph to plot this parameter")
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
        return item

    def _update_count(self):
        self.info_label.setText(f"📊 Available Columns ({len(self.parameter_items)})")

    def update_columns(self, column_names: List[str], time_column: str = None):
        """
        Update the list with columns from loaded data.

        Args:
            column_names: List of column names from the loaded data
            time_column: Name of the time column (will be excluded from list)
        """
        self.available_columns = column_names

        if not column_names:
            self._show_placeholder()
            return

        self.list_widget.setUpdatesEnabled(False)
        try:
            self.list_widget.clear()
            self.parameter_items.clear()
            for col_name in sorted(column_names):
                # Skip time column
                if time_column and col_name == time_column:
                    continue
                item = self._new_item(col_name)
                self.parameter_items[col_name] = item
                self.list_widget.addItem(item)
        finally:
            self.list_widget.setUpdatesEnabled(True)

        self._update_count()
        self.placeholder.hide()
        self.info_label.show()
        self.list_widget.show()
        self._filter_parameters(self.search_box.text())

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

        item = self._new_item(param_name)
        self.parameter_items[param_name] = item

        # Keep alphabetical order
        position = sorted(self.parameter_items).index(param_name)
        self.list_widget.insertItem(position, item)

        self._update_count()
        self.placeholder.hide()
        self.info_label.show()
        self.list_widget.show()
        self._filter_parameters(self.search_box.text())

        logger.info(f"Added parameter '{param_name}' to list")

    def _filter_parameters(self, search_text: str):
        """Filter columns based on search text."""
        search_text = search_text.lower()

        self.list_widget.setUpdatesEnabled(False)
        try:
            for col_name, item in self.parameter_items.items():
                # Check if column name matches
                item.setHidden(search_text not in col_name.lower())
        finally:
            self.list_widget.setUpdatesEnabled(True)
