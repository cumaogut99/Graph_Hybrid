# type: ignore
"""
Control Panel Manager for Time Graph Widget

Manages the control panel interface including:
- Data Selection (Time Column, Signal Columns)
- Analysis Settings (Title, Normalization, Statistics, Grid)
- Cursor Controls (No Cursor, Single, Two, Range)
- Export Actions (Reset, Export Statistics, Export Plot)
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGroupBox, QComboBox, QListWidget, QLineEdit, QCheckBox, QRadioButton,
    QButtonGroup, QPushButton, QListWidgetItem, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QFont

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class ControlPanelManager(QObject):
    """Manages the control panel interface for the Time Graph Widget."""
    
    # Signals
    time_column_changed = Signal(str)
    signal_columns_changed = Signal(list)
    title_changed = Signal(str)
    normalization_changed = Signal(bool)
    statistics_changed = Signal(bool)
    grid_changed = Signal(bool)
    cursor_mode_changed = Signal(str)
    reset_analysis = Signal()
    export_statistics = Signal()
    export_plot = Signal()
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.control_panel = None
        self.control_dialog = None
        self.available_columns = []
        self.selected_signals = []
        
        self._setup_control_panel()
    
    def _setup_control_panel(self):
        """Create the main control panel."""
        self.control_panel = QFrame()
        self.control_panel.setMinimumWidth(300)
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setVisible(False)  # Initially hidden
        
        # Apply styling
        self.control_panel.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 10px;
                border: 2px solid #4a90e2;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #4a90e2;
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
                font-size: 9px;
            }
            QComboBox, QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px;
                background-color: #ffffff;
                font-size: 9px;
            }
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
                font-size: 9px;
            }
            QCheckBox, QRadioButton {
                font-size: 9px;
                color: #333333;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 9px;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self.control_panel)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # Create sections
        self._create_data_selection_section(main_layout)
        self._create_analysis_settings_section(main_layout)
        self._create_cursor_controls_section(main_layout)
        self._create_export_actions_section(main_layout)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
    
    def _create_data_selection_section(self, parent_layout):
        """Create Data Selection section."""
        group = QGroupBox("📊 Data Selection")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        
        # Time Column
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Column:"))
        self.time_column_combo = QComboBox()
        self.time_column_combo.currentTextChanged.connect(self.time_column_changed.emit)
        time_layout.addWidget(self.time_column_combo)
        layout.addLayout(time_layout)
        
        # Signal Columns
        layout.addWidget(QLabel("Signal Columns:"))
        self.signal_list = QListWidget()
        self.signal_list.setMaximumHeight(100)
        self.signal_list.setSelectionMode(QListWidget.MultiSelection)
        self.signal_list.itemSelectionChanged.connect(self._on_signal_selection_changed)
        layout.addWidget(self.signal_list)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("<")
        prev_btn.setMaximumWidth(30)
        next_btn = QPushButton(">")
        next_btn.setMaximumWidth(30)
        nav_layout.addWidget(prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(next_btn)
        layout.addLayout(nav_layout)
        
        parent_layout.addWidget(group)
    
    def _create_analysis_settings_section(self, parent_layout):
        """Create Analysis Settings section."""
        group = QGroupBox("⚙️ Analysis Settings")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter analysis title...")
        self.title_edit.textChanged.connect(self.title_changed.emit)
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)
        
        # Checkboxes
        self.peak_norm_cb = QCheckBox("Apply Peak Normalization")
        self.peak_norm_cb.toggled.connect(self.normalization_changed.emit)
        layout.addWidget(self.peak_norm_cb)
        
        self.realtime_stats_cb = QCheckBox("Show Real-time Statistics")
        self.realtime_stats_cb.setChecked(True)
        self.realtime_stats_cb.toggled.connect(self.statistics_changed.emit)
        layout.addWidget(self.realtime_stats_cb)
        
        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(self.grid_changed.emit)
        layout.addWidget(self.show_grid_cb)
        
        parent_layout.addWidget(group)
    
    def _create_cursor_controls_section(self, parent_layout):
        """Create Cursor Controls section."""
        group = QGroupBox("🎯 Cursor Controls")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        
        # Radio buttons for cursor modes
        self.cursor_group = QButtonGroup()
        
        self.no_cursor_rb = QRadioButton("No Cursor")
        self.no_cursor_rb.setChecked(True)
        self.cursor_group.addButton(self.no_cursor_rb, 0)
        layout.addWidget(self.no_cursor_rb)
        
        self.single_cursor_rb = QRadioButton("Single Cursor")
        self.cursor_group.addButton(self.single_cursor_rb, 1)
        layout.addWidget(self.single_cursor_rb)
        
        self.two_cursors_rb = QRadioButton("Two Cursors")
        self.cursor_group.addButton(self.two_cursors_rb, 2)
        layout.addWidget(self.two_cursors_rb)
        
        self.range_selector_rb = QRadioButton("Range Selector")
        self.cursor_group.addButton(self.range_selector_rb, 3)
        layout.addWidget(self.range_selector_rb)
        
        # Connect signal
        self.cursor_group.buttonClicked.connect(self._on_cursor_mode_changed)
        
        parent_layout.addWidget(group)
    
    def _create_export_actions_section(self, parent_layout):
        """Create Export Actions section."""
        group = QGroupBox("📤 Export Actions")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        
        # Reset Analysis button
        reset_btn = QPushButton("🔄 Reset Analysis")
        reset_btn.clicked.connect(self.reset_analysis.emit)
        layout.addWidget(reset_btn)
        
        # Export Statistics button
        export_stats_btn = QPushButton("📊 Export Statistics")
        export_stats_btn.clicked.connect(self.export_statistics.emit)
        layout.addWidget(export_stats_btn)
        
        # Export Plot button
        export_plot_btn = QPushButton("📈 Export Plot")
        export_plot_btn.clicked.connect(self.export_plot.emit)
        layout.addWidget(export_plot_btn)
        
        parent_layout.addWidget(group)
    
    def _on_signal_selection_changed(self):
        """Handle signal selection changes."""
        selected_items = self.signal_list.selectedItems()
        self.selected_signals = [item.text() for item in selected_items]
        self.signal_columns_changed.emit(self.selected_signals)
    
    def _on_cursor_mode_changed(self, button):
        """Handle cursor mode changes."""
        button_id = self.cursor_group.id(button)
        modes = ["none", "single", "dual", "range"]
        if 0 <= button_id < len(modes):
            self.cursor_mode_changed.emit(modes[button_id])
    
    def get_control_panel(self) -> QWidget:
        """Get the control panel widget."""
        return self.control_panel
    
    def get_control_dialog(self) -> QDialog:
        """Get the control panel as a dialog."""
        if self.control_dialog is None:
            self.control_dialog = QDialog()
            self.control_dialog.setWindowTitle("📊 Analysis Controls")
            self.control_dialog.setModal(False)
            self.control_dialog.resize(350, 600)
            
            # Add the control panel to dialog
            dialog_layout = QVBoxLayout(self.control_dialog)
            dialog_layout.setContentsMargins(10, 10, 10, 10)
            dialog_layout.addWidget(self.control_panel)
            
        return self.control_dialog
    
    def update_available_columns(self, columns: List[str]):
        """Update available columns for selection."""
        self.available_columns = columns
        
        # Update time column combo
        self.time_column_combo.clear()
        self.time_column_combo.addItems(columns)
        
        # Auto-select time column
        time_candidates = [col for col in columns if 'time' in col.lower()]
        if time_candidates:
            self.time_column_combo.setCurrentText(time_candidates[0])
        
        # Update signal list
        self.signal_list.clear()
        for col in columns:
            if col not in time_candidates:  # Exclude time columns
                item = QListWidgetItem(col)
                self.signal_list.addItem(item)
        
        # Auto-select first few signals
        for i in range(min(3, self.signal_list.count())):
            self.signal_list.item(i).setSelected(True)
        
        self._on_signal_selection_changed()
    
    def set_title(self, title: str):
        """Set the analysis title."""
        self.title_edit.setText(title)
    
    def get_title(self) -> str:
        """Get the current analysis title."""
        return self.title_edit.text()
    
    def set_cursor_mode(self, mode: str):
        """Set the cursor mode."""
        mode_map = {"none": 0, "single": 1, "dual": 2, "range": 3}
        if mode in mode_map:
            button = self.cursor_group.button(mode_map[mode])
            if button:
                button.setChecked(True)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current panel settings."""
        return {
            'time_column': self.time_column_combo.currentText(),
            'signal_columns': self.selected_signals,
            'title': self.title_edit.text(),
            'peak_normalization': self.peak_norm_cb.isChecked(),
            'realtime_statistics': self.realtime_stats_cb.isChecked(),
            'show_grid': self.show_grid_cb.isChecked(),
            'cursor_mode': self._get_current_cursor_mode()
        }
    
    def _get_current_cursor_mode(self) -> str:
        """Get current cursor mode."""
        checked_button = self.cursor_group.checkedButton()
        if checked_button:
            button_id = self.cursor_group.id(checked_button)
            modes = ["none", "single", "dual", "range"]
            if 0 <= button_id < len(modes):
                return modes[button_id]
        return "none"
