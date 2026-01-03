"""
Parameter Filters Panel - UI components for range filtering
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, 
    QScrollArea, QGroupBox, QFormLayout, QDoubleSpinBox, QCheckBox, 
    QComboBox, QListWidget, QListWidgetItem, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class ParameterFiltersPanel(QWidget):
    """Panel for configuring range filters."""
    
    range_filter_applied = pyqtSignal(dict)

    def __init__(self, graph_index: int, all_signals: List[str], parent=None):
        super().__init__(parent)
        self.graph_index = graph_index
        self.all_signals = all_signals if all_signals else []
        self.conditions = []
        self.condition_widgets = []
        
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """Setup the range filters panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Daha küçük margins
        layout.setSpacing(8)  # Daha az spacing
        
        # Title
        title = QLabel("🔍 Range Filters")
        title.setStyleSheet("""
            QLabel {
                color: #e6f3ff;
                font-size: 18px;
                font-weight: 700;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.1));
                border-radius: 6px;
                border: 1px solid rgba(74, 144, 226, 0.3);
            }
        """)
        layout.addWidget(title)
        
        # Create range filter content directly
        self._create_range_filter_content(layout)
        
        # Apply/Reset buttons
        button_layout = QHBoxLayout()
        
        self.apply_filters_btn = QPushButton("Apply Filters")
        self.reset_filters_btn = QPushButton("Reset All")
        
        button_style = """
            QPushButton {
                padding: 12px 24px;
                border: 2px solid rgba(74, 144, 226, 0.5);
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 transparent);
                color: #e6f3ff;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.5), stop:1 rgba(74, 144, 226, 0.2));
                border-color: #4a90e2;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 rgba(74, 144, 226, 0.4));
            }
        """
        
        for btn in [self.apply_filters_btn, self.reset_filters_btn]:
            btn.setStyleSheet(button_style)
            
        # Connect button signals
        self.apply_filters_btn.clicked.connect(self._apply_range_filters)
        self.reset_filters_btn.clicked.connect(self._reset_range_filters)
            
        button_layout.addWidget(self.apply_filters_btn)
        button_layout.addWidget(self.reset_filters_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
    def _setup_connections(self):
        """Setup signal connections for the filters panel."""
        # No internal connections needed - signals are handled by parent dialog
        pass
        
    def _get_message_box_style(self) -> str:
        """Gets a consistent stylesheet for QMessageBox to match the space theme."""
        return """
            QMessageBox {
                background-color: #2d344a;
                font-size: 14px;
                color: #ffffff !important;
            }
            QMessageBox QLabel {
                color: #ffffff !important;
                padding: 10px;
                font-size: 14px;
                font-weight: normal;
                background: transparent;
            }
            QMessageBox * {
                color: #ffffff !important;
                background: transparent;
            }
            QMessageBox QTextEdit {
                color: #ffffff !important;
                background-color: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
            }
            QMessageBox QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a536b, stop:1 #3a4258);
                color: #ffffff !important;
                border: 1px solid #5a647d;
                padding: 8px 16px;
                border-radius: 5px;
                min-width: 90px;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5a647d, stop:1 #4a536b);
                border: 1px solid #7a849d;
                color: #ffffff !important;
            }
            QMessageBox QPushButton:pressed {
                background-color: #3a4258;
                color: #ffffff !important;
            }
        """

    def _apply_range_filters(self):
        """Gathers the filter conditions and emits a signal."""
        filter_data = self.get_range_filter_conditions()
        
        if not filter_data['conditions']:
            msg = QMessageBox(self)
            msg.setStyleSheet(self._get_message_box_style())
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No filter conditions have been set.")
            msg.setWindowTitle("No Conditions")
            msg.exec_()
            return

        # Add graph index to filter data
        filter_data['graph_index'] = self.graph_index
        
        # Emit signal to parent widget for filter application
        self.range_filter_applied.emit(filter_data)
        
        print(f"[DEBUG] Applying range filters: {filter_data}")
        
        # Show a success message
        msg = QMessageBox(self)
        msg.setStyleSheet(self._get_message_box_style())
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Applied {len(filter_data['conditions'])} range condition(s) in {filter_data['mode']} mode to Graph {self.graph_index + 1}.")
        msg.setWindowTitle("Filters Applied")
        msg.exec_()
        
    def _reset_range_filters(self):
        """Reset all range filter conditions."""
        msg = QMessageBox(self)
        msg.setStyleSheet(self._get_message_box_style())
        msg.setIcon(QMessageBox.Question)
        msg.setText(f'Are you sure you want to reset all range filters for Graph {self.graph_index + 1}?')
        msg.setWindowTitle('Reset Filters')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        reply = msg.exec_()
        
        if reply == QMessageBox.Yes:
            # Clear all conditions
            self.range_conditions.clear()
            
            # Clear the conditions container
            for i in reversed(range(self.conditions_layout.count())):
                child = self.conditions_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Add one default condition
            self._add_range_condition()
            
            # Emit reset signal - CRITICAL: Use current mode, not hardcoded 'segmented'!
            # This ensures concatenated filters are properly cleaned up
            current_mode = 'segmented' if self.segmented_mode_rb.isChecked() else 'concatenated'
            reset_data = {
                'graph_index': self.graph_index,
                'conditions': [],  # Empty conditions = reset/clear filter
                'mode': current_mode  # Preserve current mode for proper cleanup
            }
            self.range_filter_applied.emit(reset_data)
            
            msg = QMessageBox(self)
            msg.setStyleSheet(self._get_message_box_style())
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"Range filters reset for Graph {self.graph_index + 1}.")
            msg.setWindowTitle("Filters Reset")
            msg.exec_()
        else:
            print(f"[DEBUG] Range filters reset cancelled for graph {self.graph_index + 1}")

    def _create_range_filter_content(self, parent_layout):
        """Create the advanced value range filter content with integrated parameter and range selection."""
        
        # Scroll area for conditions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(300)  # Minimum yükseklik eklendi
        scroll.setStyleSheet("""
            QScrollArea { 
                background: transparent; 
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)
        
        # Container for dynamic range conditions
        self.conditions_container = QWidget()
        self.conditions_container.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
        """)
        self.conditions_layout = QVBoxLayout(self.conditions_container)
        self.conditions_layout.setSpacing(5)  # Daha az spacing
        self.conditions_layout.setContentsMargins(5, 5, 5, 5)  # Küçük margins
        
        self.range_conditions = []
        self._add_range_condition()  # Add first condition
        
        scroll.setWidget(self.conditions_container)
        parent_layout.addWidget(scroll)
        
        # Add/Remove buttons
        buttons_layout = QHBoxLayout()
        
        self.add_condition_btn = QPushButton("➕ Add Parameter Condition")
        self.remove_condition_btn = QPushButton("➖ Remove Last Condition")
        
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 transparent);
                color: #e6f3ff;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.1));
                border-color: #4a90e2;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 rgba(74, 144, 226, 0.3));
            }
        """
        
        self.add_condition_btn.setStyleSheet(button_style)
        self.remove_condition_btn.setStyleSheet(button_style)
        
        self.add_condition_btn.clicked.connect(self._add_range_condition)
        self.remove_condition_btn.clicked.connect(self._remove_range_condition)
        
        buttons_layout.addWidget(self.add_condition_btn)
        buttons_layout.addWidget(self.remove_condition_btn)
        buttons_layout.addStretch()
        
        parent_layout.addLayout(buttons_layout)
        
        # Filter mode selection
        mode_group = QGroupBox("🔗 Display Mode")
        mode_group.setStyleSheet(self._get_group_style())
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(3)  # Daha az spacing
        mode_layout.setContentsMargins(8, 15, 8, 8)  # Optimize edilmiş margins
        
        self.segmented_mode_rb = QCheckBox("Segmented Display (Show matching time segments with gaps)")
        self.segmented_mode_rb.setStyleSheet("color: #ffffff; font-size: 12px;")
        
        self.concatenated_mode_rb = QCheckBox("Concatenated Display (Apply global time filter to all graphs)")
        self.concatenated_mode_rb.setStyleSheet("color: #ffffff; font-size: 12px;")
        
        self.segmented_mode_rb.setChecked(True)  # Default to segmented
        
        # Make them mutually exclusive
        self.segmented_mode_rb.toggled.connect(lambda checked: self.concatenated_mode_rb.setChecked(not checked) if checked else None)
        self.concatenated_mode_rb.toggled.connect(lambda checked: self.segmented_mode_rb.setChecked(not checked) if checked else None)
        
        mode_layout.addWidget(self.segmented_mode_rb)
        mode_layout.addWidget(self.concatenated_mode_rb)
        
        parent_layout.addWidget(mode_group)
        
    def _add_range_condition(self):
        """Adds a new parameter condition box to the layout."""
        condition_index = len(self.range_conditions)
        
        # Create group box for this condition
        condition_group = QGroupBox(f"📊 Condition {condition_index + 1}")
        condition_group.setStyleSheet(self._get_group_style())
        condition_layout = QVBoxLayout(condition_group)
        condition_layout.setSpacing(5)  # Daha az spacing
        condition_layout.setContentsMargins(8, 15, 8, 8)  # Optimize edilmiş margins
        
        # Make sure the group box content is transparent
        condition_group.setAutoFillBackground(False)
        
        # Parameter selection
        param_section = QVBoxLayout()
        param_section.setSpacing(3)  # Daha az spacing
        param_label = QLabel("📋 Parameter Selection")
        param_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: 600;
                font-size: 12px;
                margin-bottom: 5px;
            }
        """)
        
        # Parameter search
        param_search = QLineEdit()
        param_search.setPlaceholderText("Search parameters...")
        param_search.setStyleSheet("""
            QLineEdit {
                padding: 6px 10px;
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.05), stop:1 transparent);
                color: #e6f3ff;
                font-size: 11px;
                margin-bottom: 5px;
            }
            QLineEdit:focus {
                border-color: #4a90e2;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.1), stop:1 transparent);
            }
        """)
        
        param_list = QListWidget()
        param_list.setMinimumHeight(150)  # Minimum yükseklik artırıldı
        param_list.setMaximumHeight(200)  # Maximum yükseklik artırıldı
        param_list.setStyleSheet("""
            QListWidget {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                color: #e6f3ff;
                selection-background-color: rgba(74, 144, 226, 0.5);
            }
            QListWidget::item {
                padding: 4px 6px;
                border-bottom: 1px solid rgba(74, 144, 226, 0.2);
            }
            QListWidget::item:hover {
                background: rgba(74, 144, 226, 0.2);
            }
        """)
        
        # PERFORMANCE: Deferred loading - only load first 50 items initially
        param_list.setUpdatesEnabled(False)  # Disable updates during batch operation
        initial_count = min(50, len(self.all_signals))
        
        for signal in self.all_signals[:initial_count]:
            item = QListWidgetItem(signal)
            param_list.addItem(item)
        
        # Show loading indicator if more items exist
        if len(self.all_signals) > initial_count:
            loading_item = QListWidgetItem(f"[{len(self.all_signals) - initial_count} more - use search to find]")
            loading_item.setFlags(Qt.ItemIsEnabled)
            loading_item.setForeground(Qt.yellow)
            param_list.addItem(loading_item)
            
        param_list.setUpdatesEnabled(True)  # Re-enable and refresh once
        
        # Connect search functionality
        param_search.textChanged.connect(lambda text, plist=param_list: self._filter_condition_parameters(plist, text))
        
        param_section.addWidget(param_label)
        param_section.addWidget(param_search)
        param_section.addWidget(param_list)
        condition_layout.addLayout(param_section)
        
        # Range controls
        range_layout = QFormLayout()
        range_layout.setVerticalSpacing(5)  # Daha az vertical spacing
        range_layout.setHorizontalSpacing(8)  # Optimize edilmiş horizontal spacing
        
        # Lower bound
        lower_enabled = QCheckBox("Enable Lower Bound")
        lower_enabled.setStyleSheet("color: #ffffff;")
        lower_operator = QComboBox()
        lower_operator.addItems([">=", ">"])
        lower_operator.setStyleSheet(self._get_combo_style())
        lower_value = QDoubleSpinBox()
        lower_value.setRange(-999999.0, 999999.0)
        lower_value.setDecimals(3)
        lower_value.setStyleSheet(self._get_spinbox_style())
        
        # Upper bound
        upper_enabled = QCheckBox("Enable Upper Bound")
        upper_enabled.setStyleSheet("color: #ffffff;")
        upper_operator = QComboBox()
        upper_operator.addItems(["<=", "<"])
        upper_operator.setStyleSheet(self._get_combo_style())
        upper_value = QDoubleSpinBox()
        upper_value.setRange(-999999.0, 999999.0)
        upper_value.setDecimals(3)
        upper_value.setStyleSheet(self._get_spinbox_style())
        
        # Add to form layout
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(lower_enabled)
        lower_layout.addWidget(lower_operator)
        lower_layout.addWidget(lower_value)
        
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(upper_enabled)
        upper_layout.addWidget(upper_operator)
        upper_layout.addWidget(upper_value)
        
        range_layout.addRow("Lower Bound:", lower_layout)
        range_layout.addRow("Upper Bound:", upper_layout)
        
        condition_layout.addLayout(range_layout)
        
        # Store condition data
        condition_data = {
            'widget': condition_group,
            'param_list': param_list,
            'lower_enabled': lower_enabled,
            'lower_operator': lower_operator,
            'lower_value': lower_value,
            'upper_enabled': upper_enabled,
            'upper_operator': upper_operator,
            'upper_value': upper_value
        }
        
        self.range_conditions.append(condition_data)
        self.conditions_layout.addWidget(condition_group)
        
    def _filter_condition_parameters(self, param_list: QListWidget, text: str):
        """Filter parameter list based on search text with smart loading."""
        # PERFORMANCE: If searching and list is not fully loaded, load all items first
        if text and param_list.count() < len(self.all_signals):
            # Remove loading indicator if exists
            last_item = param_list.item(param_list.count() - 1)
            if last_item and "[" in last_item.text() and "more" in last_item.text():
                param_list.takeItem(param_list.count() - 1)
            
            # Load all remaining items (only when user searches)
            param_list.setUpdatesEnabled(False)
            current_count = param_list.count()
            for signal in self.all_signals[current_count:]:
                item = QListWidgetItem(signal)
                param_list.addItem(item)
            param_list.setUpdatesEnabled(True)
        
        # Now filter
        for i in range(param_list.count()):
            item = param_list.item(i)
            matches = text.lower() in item.text().lower()
            item.setHidden(not matches)
        
    def _remove_range_condition(self):
        """Removes the last parameter condition."""
        if len(self.range_conditions) > 1:  # Keep at least one condition
            condition_data = self.range_conditions.pop()
            condition_data['widget'].setParent(None)
            self._update_condition_titles()
        
    def _update_condition_titles(self):
        """Update condition group box titles after adding/removing conditions."""
        for i, condition_data in enumerate(self.range_conditions):
            condition_data['widget'].setTitle(f"📊 Condition {i + 1}")
            
    def get_range_filter_conditions(self):
        """Get all range filter conditions in the new format."""
        conditions = []
        
        for i, condition_data in enumerate(self.range_conditions):
            # Get selected parameter from list
            selected_items = condition_data['param_list'].selectedItems()
            if not selected_items:
                continue
                
            param_name = selected_items[0].text()
            
            condition = {
                'parameter': param_name,
                'ranges': []
            }
            
            # Add lower bound if enabled
            if condition_data['lower_enabled'].isChecked():
                condition['ranges'].append({
                    'type': 'lower',
                    'operator': condition_data['lower_operator'].currentText(),
                    'value': condition_data['lower_value'].value()
                })
                
            # Add upper bound if enabled
            if condition_data['upper_enabled'].isChecked():
                condition['ranges'].append({
                    'type': 'upper',
                    'operator': condition_data['upper_operator'].currentText(),
                    'value': condition_data['upper_value'].value()
                })
                
            if condition['ranges']:  # Only add if at least one range is enabled
                conditions.append(condition)
                
        return {
            'conditions': conditions,
            'mode': 'segmented' if self.segmented_mode_rb.isChecked() else 'concatenated'
        }
        
    def set_range_filter_conditions(self, filter_data: dict):
        """Set range filter conditions from saved data."""
        try:
            conditions = filter_data.get('conditions', [])
            mode = filter_data.get('mode', 'segmented')
            
            # Clear existing conditions first
            self.range_conditions.clear()
            for i in reversed(range(self.conditions_layout.count())):
                child = self.conditions_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Set mode
            if mode == 'segmented':
                self.segmented_mode_rb.setChecked(True)
                self.concatenated_mode_rb.setChecked(False)
            else:
                self.concatenated_mode_rb.setChecked(True)
                self.segmented_mode_rb.setChecked(False)
            
            # Add conditions or create default if empty
            if conditions:
                for condition in conditions:
                    self._add_range_condition()
                    condition_data = self.range_conditions[-1]
                    
                    # Set parameter selection
                    param_name = condition.get('parameter', '')
                    if param_name:
                        param_list = condition_data['param_list']
                        for i in range(param_list.count()):
                            item = param_list.item(i)
                            if item.text() == param_name:
                                item.setSelected(True)
                                break
                    
                    # Set ranges
                    ranges = condition.get('ranges', [])
                    for range_info in ranges:
                        range_type = range_info.get('type')
                        operator = range_info.get('operator', '>=')
                        value = range_info.get('value', 0.0)
                        
                        if range_type == 'lower':
                            condition_data['lower_enabled'].setChecked(True)
                            condition_data['lower_operator'].setCurrentText(operator)
                            condition_data['lower_value'].setValue(value)
                        elif range_type == 'upper':
                            condition_data['upper_enabled'].setChecked(True)
                            condition_data['upper_operator'].setCurrentText(operator)
                            condition_data['upper_value'].setValue(value)
            else:
                # Add default empty condition
                self._add_range_condition()
                
            logger.info(f"Loaded {len(conditions)} range filter conditions for graph {self.graph_index}")
            
        except Exception as e:
            logger.error(f"Error loading range filter conditions: {e}")
            # Fallback: add default condition
            if not self.range_conditions:
                self._add_range_condition()
        
    def _get_group_style(self) -> str:
        """Get consistent group box styling."""
        return """
            QGroupBox {
                font-weight: 600;
                font-size: 13px;
                color: #e6f3ff;
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 6px;
                margin-top: 5px;
                padding-top: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.05), stop:1 transparent);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #4a90e2;
                font-weight: 700;
            }
        """
    
    def _get_combo_style(self) -> str:
        """Get combo box styling."""
        return """
            QComboBox {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e6f3ff;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.2);
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #e6f3ff;
            }
        """
    
    def _get_spinbox_style(self) -> str:
        """Get spinbox styling."""
        return """
            QDoubleSpinBox {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e6f3ff;
                font-size: 12px;
            }
            QDoubleSpinBox:hover {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.2);
            }
        """
