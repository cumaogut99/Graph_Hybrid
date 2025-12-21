"""
Static Limits Panel - Configure static warning and error limits for signals
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QCheckBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QScrollArea, QFrame, QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import polars as pl

logger = logging.getLogger(__name__)


class StaticLimitsPanel(QWidget):
    """Panel for configuring static warning and error limits for signals."""
    
    limits_changed = pyqtSignal(dict)  # Emits limit configuration

    def __init__(self, all_signals: List[str] = None, parent=None):
        super().__init__(parent)
        self.all_signals = all_signals if all_signals else []
        self.limit_configs = {}  # signal_name -> limit_config
        
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """Setup the static limits panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Static Limits")
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
        
        # Description
        desc = QLabel("Configure warning limits for signal values. Limits will be displayed as horizontal lines on the graph.")
        desc.setStyleSheet("""
            QLabel {
                color: rgba(230, 243, 255, 0.8);
                font-size: 12px;
                margin-bottom: 10px;
                padding: 8px;
                background: rgba(74, 144, 226, 0.1);
                border-radius: 6px;
                border: 1px solid rgba(74, 144, 226, 0.2);
            }
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Search box
        search_group = QGroupBox("Search Parameters")
        search_group.setStyleSheet(self._get_group_style())
        search_layout = QVBoxLayout(search_group)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to search parameters...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 6px;
                padding: 8px 12px;
                color: #e6f3ff;
                font-size: 13px;
            }
            QLineEdit:hover {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.2);
            }
            QLineEdit:focus {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.15);
            }
        """)
        search_layout.addWidget(self.search_box)
        layout.addWidget(search_group)
        
        # Global controls
        controls_group = QGroupBox("Global Controls")
        controls_group.setStyleSheet(self._get_group_style())
        controls_layout = QHBoxLayout(controls_group)
        
        self.enable_all_btn = QPushButton("Enable All Limits")
        self.disable_all_btn = QPushButton("Disable All Limits")
        self.reset_all_btn = QPushButton("Reset All Limits")
        self.import_excel_btn = QPushButton("Import from Excel")
        self.export_sample_btn = QPushButton("Export Sample Excel")
        
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
        """
        
        for btn in [self.enable_all_btn, self.disable_all_btn, self.reset_all_btn, self.import_excel_btn, self.export_sample_btn]:
            btn.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.enable_all_btn)
        controls_layout.addWidget(self.disable_all_btn)
        controls_layout.addWidget(self.reset_all_btn)
        controls_layout.addWidget(self.import_excel_btn)
        controls_layout.addWidget(self.export_sample_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Scroll area for signal limits
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { 
                background: transparent; 
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)
        
        # Container for signal limit configurations
        self.limits_container = QWidget()
        self.limits_container.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
        """)
        self.limits_layout = QVBoxLayout(self.limits_container)
        self.limits_layout.setSpacing(8)
        
        scroll.setWidget(self.limits_container)
        layout.addWidget(scroll)
        
        # Populate with current signals
        self._populate_signal_limits()
        
    def _filter_signals(self, search_text: str):
        """Filter signals based on search text."""
        search_text = search_text.lower().strip()
        
        for i in range(self.limits_layout.count()):
            item = self.limits_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'title'):
                    # Extract signal name from group title (remove emoji and spaces)
                    title = widget.title().replace('', '').strip()
                    if search_text == '' or search_text in title.lower():
                        widget.setVisible(True)
                    else:
                        widget.setVisible(False)
    
    def _import_from_excel(self):
        """Import limit settings from Excel file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Import Limits from Excel", 
                "", 
                "Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Try to import polars for Excel reading
            try:
                pass # Polars is already imported
            except ImportError:
                QMessageBox.critical(
                    self,
                    "Library Not Found",
                    "polars library is required for Excel import.\n\nInstall with: pip install polars openpyxl"
                )
                return
            
            # Read Excel file
            df = pl.read_excel(file_path)
            
            # Validate required columns
            required_columns = ['Parameter', 'Warning_Min', 'Warning_Max']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                QMessageBox.warning(
                    self, 
                    "Invalid Excel Format", 
                    f"Missing required columns: {', '.join(missing_columns)}\n\n" +
                    "Required columns: Parameter, Warning_Min, Warning_Max"
                )
                return
            
            # Apply limits from Excel
            applied_count = 0
            for row in df.iter_rows(named=True):
                param_name = str(row['Parameter']).strip()
                warning_min = float(row['Warning_Min']) if row['Warning_Min'] is not None else 0.0
                warning_max = float(row['Warning_Max']) if row['Warning_Max'] is not None else 0.0
                
                # Check if parameter exists in our signals
                if param_name in self.limit_configs:
                    widgets = self.limit_configs[param_name]
                    widgets['enable'].setChecked(True)
                    widgets['warning_min'].setValue(warning_min)
                    widgets['warning_max'].setValue(warning_max)
                    applied_count += 1
            
            # Emit changes
            self._emit_limits_changed()
            
            # Show success message
            QMessageBox.information(
                self, 
                "Import Successful", 
                f"Successfully imported limits for {applied_count} parameters from Excel file."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Import Error", 
                f"Error importing Excel file:\n\n{str(e)}"
            )
    
    def _export_sample_excel(self):
        """Export a sample Excel file with current parameters."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Sample Excel", 
                "sample_limits.xlsx", 
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Try to import polars for Excel writing
            try:
                pass # Polars is already imported
            except ImportError:
                QMessageBox.critical(
                    self,
                    "Library Not Found",
                    "polars library is required for Excel export.\n\nInstall with: pip install polars openpyxl"
                )
                return
            
            # Create sample data with current parameters
            sample_data = []
            for signal_name in self.all_signals:
                # Get current values if they exist
                if signal_name in self.limit_configs:
                    widgets = self.limit_configs[signal_name]
                    warning_min = widgets['warning_min'].value() if widgets['enable'].isChecked() else 0.0
                    warning_max = widgets['warning_max'].value() if widgets['enable'].isChecked() else 0.0
                else:
                    warning_min = 0.0
                    warning_max = 0.0
                    
                sample_data.append({
                    'Parameter': signal_name,
                    'Warning_Min': warning_min,
                    'Warning_Max': warning_max,
                    'Description': f'Warning limits for {signal_name}'
                })
            
            # Create DataFrame and save to Excel
            df = pl.DataFrame(sample_data)
            df.write_excel(file_path, sheet_name='Limits')
            
            # Show success message
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Sample Excel file exported successfully to:\n\n{file_path}\n\n" +
                "You can modify the Warning_Min and Warning_Max values and import it back."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Error exporting Excel file:\n\n{str(e)}"
            )
        
    def _populate_signal_limits(self):
        """Populate the limits container with signal limit configurations."""
        # Clear existing widgets
        for i in reversed(range(self.limits_layout.count())):
            child = self.limits_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # PERFORMANCE: Add limit configuration for each signal (optimized)
        # Disable updates during batch widget creation
        container = self.limits_layout.parentWidget()
        if container:
            container.setUpdatesEnabled(False)
        
        for signal_name in self.all_signals:
            limit_widget = self._create_signal_limit_widget(signal_name)
            self.limits_layout.addWidget(limit_widget)
        
        # Re-enable updates
        if container:
            container.setUpdatesEnabled(True)
            
    def _create_signal_limit_widget(self, signal_name: str):
        """Create a limit configuration widget for a signal."""
        group = QGroupBox(f"{signal_name}")
        group.setStyleSheet(self._get_group_style())
        layout = QFormLayout(group)
        
        # Enable checkbox
        enable_cb = QCheckBox("Enable Limits")
        enable_cb.setStyleSheet("color: #e6f3ff;")
        layout.addRow(enable_cb)
        
        # Warning limits
        warning_group = QGroupBox("Warning Limits")
        warning_group.setStyleSheet(self._get_subgroup_style())
        warning_layout = QFormLayout(warning_group)
        
        warning_min_sb = QDoubleSpinBox()
        warning_min_sb.setRange(-999999.0, 999999.0)
        warning_min_sb.setDecimals(3)
        warning_min_sb.setStyleSheet(self._get_spinbox_style())
        
        warning_max_sb = QDoubleSpinBox()
        warning_max_sb.setRange(-999999.0, 999999.0)
        warning_max_sb.setDecimals(3)
        warning_max_sb.setStyleSheet(self._get_spinbox_style())
        
        # Create white labels for min/max warning
        min_warning_label = QLabel("Min Warning:")
        min_warning_label.setStyleSheet("color: #ffffff; font-weight: 500;")
        max_warning_label = QLabel("Max Warning:")
        max_warning_label.setStyleSheet("color: #ffffff; font-weight: 500;")
        
        warning_layout.addRow(min_warning_label, warning_min_sb)
        warning_layout.addRow(max_warning_label, warning_max_sb)
        
        layout.addRow(warning_group)
        
        # Store references for later access
        limit_config = {
            'enable': enable_cb,
            'warning_min': warning_min_sb,
            'warning_max': warning_max_sb
        }
        self.limit_configs[signal_name] = limit_config
        
        # Connect signals
        enable_cb.toggled.connect(lambda checked, name=signal_name: self._on_limit_changed(name))
        for widget in [warning_min_sb, warning_max_sb]:
            widget.valueChanged.connect(lambda value, name=signal_name: self._on_limit_changed(name))
        
        return group
        
    def _setup_connections(self):
        """Setup signal connections."""
        self.enable_all_btn.clicked.connect(self._enable_all_limits)
        self.disable_all_btn.clicked.connect(self._disable_all_limits)
        self.reset_all_btn.clicked.connect(self._reset_all_limits)
        self.import_excel_btn.clicked.connect(self._import_from_excel)
        self.export_sample_btn.clicked.connect(self._export_sample_excel)
        self.search_box.textChanged.connect(self._filter_signals)
        
    def _enable_all_limits(self):
        """Enable all signal limits."""
        for config in self.limit_configs.values():
            config['enable'].setChecked(True)
        self._emit_limits_changed()
        
    def _disable_all_limits(self):
        """Disable all signal limits."""
        for config in self.limit_configs.values():
            config['enable'].setChecked(False)
        self._emit_limits_changed()
        
    def _reset_all_limits(self):
        """Reset all limits to default values."""
        for config in self.limit_configs.values():
            config['enable'].setChecked(False)
            config['warning_min'].setValue(0.0)
            config['warning_max'].setValue(0.0)
        self._emit_limits_changed()
        
    def _on_limit_changed(self, signal_name: str):
        """Handle limit configuration change."""
        self._emit_limits_changed()
        
    def _emit_limits_changed(self):
        """Emit signal when limits configuration changes."""
        limits_data = self.get_limits_configuration()
        self.limits_changed.emit(limits_data)
        
    def get_limits_configuration(self) -> Dict[str, Any]:
        """Get the current limits configuration."""
        config = {}
        for signal_name, widgets in self.limit_configs.items():
            if widgets['enable'].isChecked():
                config[signal_name] = {
                    'warning_min': widgets['warning_min'].value(),
                    'warning_max': widgets['warning_max'].value()
                }
        return config
        
    def set_limits_configuration(self, config: Dict[str, Any]):
        """Set the limits configuration."""
        for signal_name, limits in config.items():
            if signal_name in self.limit_configs:
                widgets = self.limit_configs[signal_name]
                widgets['enable'].setChecked(True)
                widgets['warning_min'].setValue(limits.get('warning_min', 0.0))
                widgets['warning_max'].setValue(limits.get('warning_max', 0.0))
                
    def update_available_signals(self, signals: List[str]):
        """Update the list of available signals."""
        self.all_signals = signals
        self._populate_signal_limits()
        
    def _get_group_style(self) -> str:
        """Get consistent group box styling."""
        return """
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #e6f3ff;
                border: 2px solid rgba(74, 144, 226, 0.3);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
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
        
    def _get_subgroup_style(self) -> str:
        """Get subgroup styling."""
        return """
            QGroupBox {
                font-weight: 500;
                font-size: 12px;
                color: #e6f3ff;
                border: 1px solid rgba(74, 144, 226, 0.2);
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background: rgba(74, 144, 226, 0.05);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px 0 6px;
                color: #e6f3ff;
                font-weight: 600;
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
