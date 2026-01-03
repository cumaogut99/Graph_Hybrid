# type: ignore
"""
Parameter Calculator Dialog

Allows users to create new parameters by performing calculations on existing parameters.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QTextEdit, QGroupBox, QMessageBox,
    QFormLayout, QWidget, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class ParameterCalculatorDialog(QDialog):
    """
    Dialog for creating new calculated parameters from existing ones.
    
    Features:
    - Select existing parameters
    - Enter mathematical formula
    - Preview calculation result
    - Save as new parameter
    """
    
    # Signal emitted when a new parameter is created
<<<<<<< HEAD
    # NEW: Emits formula for streaming disk-based calculation instead of data arrays
    parameter_created = pyqtSignal(str, str, list)  # name, formula, used_params
    
    # Legacy signal for small datasets (RAM-based fallback)
    parameter_created_legacy = pyqtSignal(str, np.ndarray, np.ndarray)  # name, x_data, y_data
=======
    parameter_created = pyqtSignal(str, np.ndarray, np.ndarray)  # name, x_data, y_data
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    
    def __init__(self, available_parameters: List[str], signal_data_getter, parent=None):
        """
        Initialize the parameter calculator dialog.
        
        Args:
            available_parameters: List of available parameter names
            signal_data_getter: Function that takes parameter name and returns (x_data, y_data)
            parent: Parent widget
        """
        super().__init__(parent)
        self.available_parameters = available_parameters
        self.signal_data_getter = signal_data_getter
        self.time_data = None  # Will be set from first parameter
        
        self.setWindowTitle("➕ Add New Parameter")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)  # Reduced by ~10% (from 500)
        
        self._setup_ui()
        self._load_available_parameters()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Create New Calculated Parameter")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setStyleSheet("color: #e6f3ff; padding: 5px;")
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Select parameters and enter a mathematical formula to create a new parameter.")
        desc_label.setStyleSheet("color: #a0c0e0; padding: 5px; font-size: 10pt;")
        main_layout.addWidget(desc_label)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Left side: Available parameters
        left_group = QGroupBox("Available Parameters")
        left_layout = QVBoxLayout(left_group)
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("🔍")
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
            }
        """)
        self.search_box.textChanged.connect(self._filter_parameters)
        search_layout.addWidget(self.search_box, 1)
        left_layout.addLayout(search_layout)
        
        # Parameters list
        self.parameters_list = QListWidget()
        self.parameters_list.setSelectionMode(QListWidget.MultiSelection)
        self.parameters_list.setStyleSheet("""
            QListWidget {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                color: #e6f3ff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background: rgba(74, 144, 226, 0.3);
            }
            QListWidget::item:hover {
                background: rgba(74, 144, 226, 0.2);
            }
        """)
        left_layout.addWidget(self.parameters_list)
        
        # Insert button
        self.insert_btn = QPushButton("Insert Selected")
        self.insert_btn.setStyleSheet("""
            QPushButton {
                background: rgba(74, 144, 226, 0.3);
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 4px;
                padding: 6px;
                color: #e6f3ff;
            }
            QPushButton:hover {
                background: rgba(74, 144, 226, 0.5);
            }
        """)
        self.insert_btn.clicked.connect(self._insert_selected_parameters)
        left_layout.addWidget(self.insert_btn)
        
        content_layout.addWidget(left_group, 1)  # Left side takes less space
        
        # Right side: Formula input
        right_group = QGroupBox("Formula & Settings")
        right_layout = QVBoxLayout(right_group)
        
        # Parameter name
        name_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., calculated_signal")
        self.name_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 6px;
                color: #e6f3ff;
            }
        """)
        name_layout.addRow("Parameter Name:", self.name_input)
        right_layout.addLayout(name_layout)
        
        # Formula input
        formula_label = QLabel("Formula (use parameter names as variables):")
        formula_label.setStyleSheet("color: #e6f3ff; padding: 5px 0px;")
        right_layout.addWidget(formula_label)
        
        # Formula input and operators in horizontal layout
        formula_container = QWidget()
        formula_container_layout = QHBoxLayout(formula_container)
        formula_container_layout.setContentsMargins(0, 0, 0, 0)
        formula_container_layout.setSpacing(10)
        
        # Formula input (left side)
        self.formula_input = QTextEdit()
        self.formula_input.setPlaceholderText("Example: param1 + param2 * 2\nor: param1 / param2\nor: np.sin(param1)")
        self.formula_input.setMinimumHeight(120)
        self.formula_input.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 8px;
                color: #e6f3ff;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
            }
        """)
        formula_container_layout.addWidget(self.formula_input, 3)  # Formula takes more space
        
        # Operators panel (right side)
        operators_group = QGroupBox("Operators")
        operators_layout = QVBoxLayout(operators_group)
        operators_layout.setContentsMargins(8, 8, 8, 8)
        operators_layout.setSpacing(5)
        
        # Basic operators
        basic_label = QLabel("Basic:")
        basic_label.setStyleSheet("color: #a0c0e0; font-size: 9pt; font-weight: bold;")
        operators_layout.addWidget(basic_label)
        
        basic_ops_layout = QHBoxLayout()
        basic_ops = [
            ("+", "+"), ("-", "-"), ("*", "*"), ("/", "/"),
            ("**", "**"), ("(", "("), (")", ")")
        ]
        for symbol, text in basic_ops:
            btn = QPushButton(symbol)
            btn.setFixedSize(35, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(74, 144, 226, 0.2);
                    border: 1px solid rgba(74, 144, 226, 0.4);
                    border-radius: 4px;
                    color: #e6f3ff;
                    font-size: 12pt;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(74, 144, 226, 0.4);
                }
                QPushButton:pressed {
                    background: rgba(74, 144, 226, 0.6);
                }
            """)
            btn.clicked.connect(lambda checked, t=text: self._insert_operator(t))
            basic_ops_layout.addWidget(btn)
        operators_layout.addLayout(basic_ops_layout)
        
        # Comparison operators
        comp_label = QLabel("Comparison:")
        comp_label.setStyleSheet("color: #a0c0e0; font-size: 9pt; font-weight: bold;")
        operators_layout.addWidget(comp_label)
        
        comp_ops_layout = QHBoxLayout()
        comp_ops = [
            ("==", "=="), ("!=", "!="), (">", ">"), ("<", "<"),
            (">=", ">="), ("<=", "<=")
        ]
        for symbol, text in comp_ops:
            btn = QPushButton(symbol)
            btn.setFixedSize(35, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(74, 144, 226, 0.2);
                    border: 1px solid rgba(74, 144, 226, 0.4);
                    border-radius: 4px;
                    color: #e6f3ff;
                    font-size: 10pt;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(74, 144, 226, 0.4);
                }
                QPushButton:pressed {
                    background: rgba(74, 144, 226, 0.6);
                }
            """)
            btn.clicked.connect(lambda checked, t=text: self._insert_operator(t))
            comp_ops_layout.addWidget(btn)
        operators_layout.addLayout(comp_ops_layout)
        
        # NumPy functions
        numpy_label = QLabel("NumPy Functions:")
        numpy_label.setStyleSheet("color: #a0c0e0; font-size: 9pt; font-weight: bold;")
        operators_layout.addWidget(numpy_label)
        
        numpy_ops_layout = QVBoxLayout()
        numpy_ops = [
            ("sin", "np.sin("), ("cos", "np.cos("), ("abs", "np.abs("),
            ("sqrt", "np.sqrt("), ("log", "np.log("), ("exp", "np.exp("),
            ("max", "np.max("), ("min", "np.min("), ("mean", "np.mean(")
        ]
        for symbol, text in numpy_ops:
            btn = QPushButton(symbol)
            btn.setFixedHeight(28)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(144, 226, 74, 0.2);
                    border: 1px solid rgba(144, 226, 74, 0.4);
                    border-radius: 4px;
                    color: #e6f3ff;
                    font-size: 9pt;
                    text-align: left;
                    padding-left: 5px;
                }
                QPushButton:hover {
                    background: rgba(144, 226, 74, 0.4);
                }
                QPushButton:pressed {
                    background: rgba(144, 226, 74, 0.6);
                }
            """)
            btn.clicked.connect(lambda checked, t=text: self._insert_operator(t))
            numpy_ops_layout.addWidget(btn)
        operators_layout.addLayout(numpy_ops_layout)
        
        operators_layout.addStretch()
        formula_container_layout.addWidget(operators_group, 1)
        
        right_layout.addWidget(formula_container)
        
        # Help text
        help_text = QLabel("💡 Tip: Use parameter names directly in formulas.\nSupported: +, -, *, /, **, np.sin, np.cos, np.abs, etc.")
        help_text.setStyleSheet("color: #a0c0e0; font-size: 9pt; padding: 5px;")
        help_text.setWordWrap(True)
        right_layout.addWidget(help_text)
        
        # Preview button
        self.preview_btn = QPushButton("🔍 Preview Calculation")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background: rgba(74, 144, 226, 0.3);
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 4px;
                padding: 8px;
                color: #e6f3ff;
            }
            QPushButton:hover {
                background: rgba(74, 144, 226, 0.5);
            }
        """)
        self.preview_btn.clicked.connect(self._preview_calculation)
        right_layout.addWidget(self.preview_btn)
        
        # Preview result
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 8px;
                color: #a0c0e0;
                font-size: 9pt;
            }
        """)
        self.preview_label.setWordWrap(True)
        self.preview_label.hide()
        right_layout.addWidget(self.preview_label)
        
        right_layout.addStretch()
        content_layout.addWidget(right_group, 2)  # Right side takes more space
        
        main_layout.addLayout(content_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: rgba(128, 128, 128, 0.3);
                border: 1px solid rgba(128, 128, 128, 0.5);
                border-radius: 4px;
                padding: 8px 20px;
                color: #e6f3ff;
            }
            QPushButton:hover {
                background: rgba(128, 128, 128, 0.5);
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.create_btn = QPushButton("Create Parameter")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background: rgba(74, 144, 226, 0.6);
                border: 1px solid rgba(74, 144, 226, 0.8);
                border-radius: 4px;
                padding: 8px 20px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(74, 144, 226, 0.8);
            }
        """)
        self.create_btn.clicked.connect(self._create_parameter)
        button_layout.addWidget(self.create_btn)
        
        main_layout.addLayout(button_layout)
        
    def _load_available_parameters(self):
        """Load available parameters into the list."""
        self.parameters_list.clear()
        for param in sorted(self.available_parameters):
            item = QListWidgetItem(param)
            self.parameters_list.addItem(item)
    
    def _filter_parameters(self, search_text: str):
        """Filter parameters based on search text."""
        search_text = search_text.lower()
        for i in range(self.parameters_list.count()):
            item = self.parameters_list.item(i)
            matches = search_text in item.text().lower()
            item.setHidden(not matches)
    
    def _insert_selected_parameters(self):
        """Insert selected parameters into formula."""
        selected_items = self.parameters_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select parameters to insert.")
            return
        
        # Get cursor position
        cursor = self.formula_input.textCursor()
        cursor_pos = cursor.position()
        
        # Insert parameter names
        param_names = [item.text() for item in selected_items]
        text_to_insert = " + ".join(param_names)
        
        # Insert at cursor position
        current_text = self.formula_input.toPlainText()
        new_text = current_text[:cursor_pos] + text_to_insert + current_text[cursor_pos:]
        self.formula_input.setPlainText(new_text)
        
        # Restore cursor position
        cursor.setPosition(cursor_pos + len(text_to_insert))
        self.formula_input.setTextCursor(cursor)
        self.formula_input.setFocus()
        
        # Clear selection after inserting
        self.parameters_list.clearSelection()
    
    def _insert_operator(self, operator: str):
        """Insert operator into formula at cursor position."""
        cursor = self.formula_input.textCursor()
        cursor_pos = cursor.position()
        
        # Insert operator at cursor position
        current_text = self.formula_input.toPlainText()
        new_text = current_text[:cursor_pos] + operator + current_text[cursor_pos:]
        self.formula_input.setPlainText(new_text)
        
        # Restore cursor position
        cursor.setPosition(cursor_pos + len(operator))
        self.formula_input.setTextCursor(cursor)
        self.formula_input.setFocus()
    
    def _preview_calculation(self):
<<<<<<< HEAD
        """Preview the calculation result using SAMPLED data (memory efficient)."""
=======
        """Preview the calculation result."""
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        formula = self.formula_input.toPlainText().strip()
        if not formula:
            self.preview_label.setText("⚠️ Please enter a formula.")
            self.preview_label.setStyleSheet(self.preview_label.styleSheet().replace("#a0c0e0", "#ffaa00"))
            self.preview_label.show()
            return
        
        try:
<<<<<<< HEAD
            # ✅ MEMORY OPT: Only load parameters that are used in the formula
            used_params = [p for p in self.available_parameters if p in formula]
            if not used_params:
                self.preview_label.setText("⚠️ Formula doesn't reference any parameters.")
                self.preview_label.show()
                return
            
            # ✅ STREAMING PREVIEW: Sample only 1000 points for preview
            # This keeps RAM usage minimal even for 50GB files
            PREVIEW_SAMPLE_SIZE = 1000
            
            logger.info(f"[MEMORY OPT] Preview sampling {PREVIEW_SAMPLE_SIZE} points from {len(used_params)} params")
            
            # Get sampled parameter data
            param_data = {}
            sample_size = PREVIEW_SAMPLE_SIZE
            
            for param in used_params:
                try:
                    x_data, y_data = self.signal_data_getter(param)
                    
                    # Sample evenly if data is large
                    if len(y_data) > sample_size:
                        indices = np.linspace(0, len(y_data) - 1, sample_size, dtype=int)
                        y_sampled = y_data[indices]
                        if self.time_data is None and x_data is not None:
                            self.time_data = x_data[indices]
                    else:
                        y_sampled = y_data
                        if self.time_data is None:
                            self.time_data = x_data
                    
                    param_data[param] = y_sampled
=======
            # Get parameter data
            param_data = {}
            for param in self.available_parameters:
                try:
                    x_data, y_data = self.signal_data_getter(param)
                    if self.time_data is None:
                        self.time_data = x_data.copy()
                    param_data[param] = y_data
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                except Exception as e:
                    logger.warning(f"Could not get data for parameter '{param}': {e}")
                    continue
            
            if not param_data:
                self.preview_label.setText("⚠️ No parameter data available.")
                self.preview_label.setStyleSheet(self.preview_label.styleSheet().replace("#a0c0e0", "#ff0000"))
                self.preview_label.show()
                return
            
<<<<<<< HEAD
            # Check for streaming-incompatible formulas
            from src.data.streaming_param_worker import detect_streaming_incompatible
            incompatible = detect_streaming_incompatible(formula)
            
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            # Create safe evaluation environment
            safe_dict = {
                'np': np,
                '__builtins__': {},
            }
            safe_dict.update(param_data)
            
<<<<<<< HEAD
            # Evaluate formula on sampled data
=======
            # Evaluate formula
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            result = eval(formula, safe_dict)
            
            if isinstance(result, np.ndarray):
                result_array = result
            else:
                # Scalar result - create array
<<<<<<< HEAD
                first_data = next(iter(param_data.values()))
                result_array = np.full(len(first_data), result, dtype=np.float64)
            
            # Show preview
            preview_text = f"✅ Formula valid! (Preview from {len(result_array)} samples)\n"
=======
                if self.time_data is not None:
                    result_array = np.full_like(self.time_data, result)
                else:
                    result_array = np.array([result])
            
            # Show preview
            preview_text = f"✅ Calculation successful!\n"
            preview_text += f"Result shape: {result_array.shape}\n"
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            preview_text += f"Min: {np.nanmin(result_array):.4f}\n"
            preview_text += f"Max: {np.nanmax(result_array):.4f}\n"
            preview_text += f"Mean: {np.nanmean(result_array):.4f}"
            
<<<<<<< HEAD
            if incompatible:
                preview_text += f"\n⚠️ Warning: {', '.join(incompatible)} may not work correctly with streaming"
            
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            self.preview_label.setText(preview_text)
            self.preview_label.setStyleSheet(self.preview_label.styleSheet().replace("#ff0000", "#00ff00").replace("#ffaa00", "#00ff00"))
            self.preview_label.show()
            
        except Exception as e:
            error_msg = f"❌ Calculation error: {str(e)}"
            self.preview_label.setText(error_msg)
            self.preview_label.setStyleSheet(self.preview_label.styleSheet().replace("#00ff00", "#ff0000").replace("#a0c0e0", "#ff0000"))
            self.preview_label.show()
            logger.error(f"Preview calculation error: {e}", exc_info=True)
    
    def _create_parameter(self):
<<<<<<< HEAD
        """Create the new parameter using streaming disk-based calculation."""
=======
        """Create the new parameter."""
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        param_name = self.name_input.text().strip()
        formula = self.formula_input.toPlainText().strip()
        
        # Validate inputs
        if not param_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a parameter name.")
            return
        
        if not formula:
            QMessageBox.warning(self, "Invalid Formula", "Please enter a formula.")
            return
        
<<<<<<< HEAD
        # Sanitize parameter name (remove special chars except underscore)
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', param_name)
        if clean_name != param_name:
            reply = QMessageBox.question(
                self, "Sanitize Name",
                f"Parameter name will be sanitized to '{clean_name}'. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            param_name = clean_name
        
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        # Check if parameter name already exists
        if param_name in self.available_parameters:
            reply = QMessageBox.question(
                self, "Parameter Exists",
                f"Parameter '{param_name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        try:
<<<<<<< HEAD
            # ✅ STREAMING: Find used parameters without loading data
            used_params = [p for p in self.available_parameters if p in formula]
            if not used_params:
                QMessageBox.warning(self, "No Parameters Used", 
                    "Formula doesn't reference any existing parameters.")
                return
            
            logger.info(f"[STREAMING] Creating param '{param_name}' with formula: {formula}")
            logger.info(f"[STREAMING] Uses {len(used_params)} parameters: {used_params}")
            
            # Check for streaming-incompatible formulas
            from src.data.streaming_param_worker import detect_streaming_incompatible
            incompatible = detect_streaming_incompatible(formula)
            
            if incompatible:
                reply = QMessageBox.warning(
                    self, "Streaming Warning",
                    f"Formula contains functions that may not work correctly with streaming:\n"
                    f"{', '.join(incompatible)}\n\n"
                    f"The calculation will proceed but results may be incorrect for these operations.\n"
                    f"Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            # ✅ EMIT FORMULA for streaming calculation (not data arrays!)
            # This is handled by ParametersPanelManager._on_parameter_created
            self.parameter_created.emit(param_name, formula, used_params)
            
            # Show success - actual calculation happens in background
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Calculation Started")
            msg.setText(f"Parameter '{param_name}' calculation started.\n\nThis will run in the background.")
=======
            # Get parameter data
            param_data = {}
            for param in self.available_parameters:
                try:
                    x_data, y_data = self.signal_data_getter(param)
                    if self.time_data is None:
                        self.time_data = x_data.copy()
                    param_data[param] = y_data
                except Exception as e:
                    logger.warning(f"Could not get data for parameter '{param}': {e}")
                    continue
            
            if not param_data:
                QMessageBox.critical(self, "Error", "No parameter data available.")
                return
            
            # Create safe evaluation environment
            safe_dict = {
                'np': np,
                '__builtins__': {},
            }
            safe_dict.update(param_data)
            
            # Evaluate formula
            result = eval(formula, safe_dict)
            
            if isinstance(result, np.ndarray):
                result_array = result
            else:
                # Scalar result - create array
                if self.time_data is not None:
                    result_array = np.full_like(self.time_data, result)
                else:
                    result_array = np.array([result])
            
            # Ensure time_data is set
            if self.time_data is None:
                # Try to get time data from first parameter
                try:
                    x_data, _ = self.signal_data_getter(self.available_parameters[0])
                    self.time_data = x_data.copy()
                except:
                    # Create default time array
                    self.time_data = np.arange(len(result_array))
            
            # Emit signal with new parameter
            self.parameter_created.emit(param_name, self.time_data, result_array)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Success")
            msg.setText(f"Parameter '{param_name}' created successfully!")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #1e1e1e;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                    font-size: 11pt;
                }
                QPushButton {
                    background-color: rgba(74, 144, 226, 0.3);
                    border: 1px solid rgba(74, 144, 226, 0.5);
                    border-radius: 4px;
                    padding: 6px 20px;
                    color: #ffffff;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: rgba(74, 144, 226, 0.5);
                }
            """)
            msg.exec_()
            self.accept()
            
        except Exception as e:
<<<<<<< HEAD
            QMessageBox.critical(self, "Error", f"Failed to start calculation:\n{str(e)}")
=======
            QMessageBox.critical(self, "Error", f"Failed to create parameter:\n{str(e)}")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            logger.error(f"Create parameter error: {e}", exc_info=True)

