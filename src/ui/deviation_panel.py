"""
Deviation Panel - Analyze signal deviations and statistical variations
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QCheckBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QScrollArea, QFrame, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class DeviationPanel(QWidget):
    """Panel for analyzing signal deviations and statistical variations."""
    
    deviation_settings_changed = pyqtSignal(dict)  # Emits deviation configuration

    def __init__(self, all_signals: List[str] = None, parent=None):
        super().__init__(parent)
        self.all_signals = all_signals if all_signals else []
        self.deviation_configs = {}  # signal_name -> deviation_config
        
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """Setup the deviation analysis panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📈 Advanced Deviation Analysis")
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
        desc = QLabel("Advanced deviation analysis with comprehensive statistical methods and signal-specific configurations. (Demo UI Only - No Backend Processing)")
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
        
        # Global settings
        global_group = QGroupBox("🌐 Global Deviation Settings")
        global_group.setStyleSheet(self._get_group_style())
        global_layout = QFormLayout(global_group)
        
        # Analysis method
        self.analysis_method = QComboBox()
        self.analysis_method.addItems([
            "Standard Deviation",
            "Mean Absolute Deviation",
            "Median Absolute Deviation",
            "Interquartile Range"
        ])
        self.analysis_method.setStyleSheet(self._get_combo_style())
        global_layout.addRow("Analysis Method:", self.analysis_method)
        
        # Window size for rolling analysis
        self.window_size = QSpinBox()
        self.window_size.setRange(10, 10000)
        self.window_size.setValue(100)
        self.window_size.setStyleSheet(self._get_spinbox_style())
        global_layout.addRow("Rolling Window Size:", self.window_size)
        
        # Sensitivity threshold
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_label = QLabel("Medium (5)")
        self.sensitivity_label.setStyleSheet("color: #e6f3ff; font-size: 11px;")
        
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_label)
        global_layout.addRow("Sensitivity:", sensitivity_layout)
        
        layout.addWidget(global_group)
        
        # Detection settings
        detection_group = QGroupBox("🔍 Outlier Detection")
        detection_group.setStyleSheet(self._get_group_style())
        detection_layout = QFormLayout(detection_group)
        
        self.enable_outlier_detection = QCheckBox("Enable Outlier Detection")
        self.enable_outlier_detection.setStyleSheet("color: #e6f3ff;")
        detection_layout.addRow(self.enable_outlier_detection)
        
        # Z-score threshold
        self.zscore_threshold = QDoubleSpinBox()
        self.zscore_threshold.setRange(1.0, 5.0)
        self.zscore_threshold.setValue(2.0)
        self.zscore_threshold.setDecimals(1)
        self.zscore_threshold.setStyleSheet(self._get_spinbox_style())
        detection_layout.addRow("Z-Score Threshold:", self.zscore_threshold)
        
        # Highlight outliers
        self.highlight_outliers = QCheckBox("Highlight Outliers on Graph")
        self.highlight_outliers.setStyleSheet("color: #e6f3ff;")
        detection_layout.addRow(self.highlight_outliers)
        
        layout.addWidget(detection_group)
        
        # Signal-specific settings
        signals_group = QGroupBox("📊 Signal-Specific Settings")
        signals_group.setStyleSheet(self._get_group_style())
        signals_layout = QVBoxLayout(signals_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.enable_all_btn = QPushButton("Enable All")
        self.disable_all_btn = QPushButton("Disable All")
        self.auto_configure_btn = QPushButton("Auto Configure")
        
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
        
        for btn in [self.enable_all_btn, self.disable_all_btn, self.auto_configure_btn]:
            btn.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.enable_all_btn)
        controls_layout.addWidget(self.disable_all_btn)
        controls_layout.addWidget(self.auto_configure_btn)
        controls_layout.addStretch()
        
        signals_layout.addLayout(controls_layout)
        
        # Scroll area for signal-specific settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        
        self.signals_container = QWidget()
        self.signals_layout = QVBoxLayout(self.signals_container)
        self.signals_layout.setSpacing(8)
        
        scroll.setWidget(self.signals_container)
        signals_layout.addWidget(scroll)
        
        layout.addWidget(signals_group)
        
        # Populate with current signals
        self._populate_signal_settings()
        
    def _populate_signal_settings(self):
        """Populate signal-specific deviation settings."""
        # Clear existing widgets
        for i in reversed(range(self.signals_layout.count())):
            child = self.signals_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add settings for each signal
        for signal_name in self.all_signals:
            signal_widget = self._create_signal_deviation_widget(signal_name)
            self.signals_layout.addWidget(signal_widget)
            
    def _create_signal_deviation_widget(self, signal_name: str):
        """Create deviation settings widget for a signal."""
        group = QGroupBox(f"📊 {signal_name}")
        group.setStyleSheet(self._get_subgroup_style())
        layout = QFormLayout(group)
        
        # Enable checkbox
        enable_cb = QCheckBox("Enable Deviation Analysis")
        enable_cb.setStyleSheet("color: #e6f3ff;")
        layout.addRow(enable_cb)
        
        # Custom threshold
        threshold_sb = QDoubleSpinBox()
        threshold_sb.setRange(0.1, 10.0)
        threshold_sb.setValue(1.0)
        threshold_sb.setDecimals(2)
        threshold_sb.setStyleSheet(self._get_spinbox_style())
        layout.addRow("Custom Threshold:", threshold_sb)
        
        # Baseline method
        baseline_combo = QComboBox()
        baseline_combo.addItems([
            "Rolling Mean",
            "Global Mean",
            "Median",
            "Linear Trend"
        ])
        baseline_combo.setStyleSheet(self._get_combo_style())
        layout.addRow("Baseline Method:", baseline_combo)
        
        # Store references
        config = {
            'enable': enable_cb,
            'threshold': threshold_sb,
            'baseline': baseline_combo
        }
        self.deviation_configs[signal_name] = config
        
        # Connect signals
        enable_cb.toggled.connect(lambda checked, name=signal_name: self._on_setting_changed(name))
        threshold_sb.valueChanged.connect(lambda value, name=signal_name: self._on_setting_changed(name))
        baseline_combo.currentTextChanged.connect(lambda text, name=signal_name: self._on_setting_changed(name))
        
        return group
        
    def _setup_connections(self):
        """Setup signal connections."""
        self.analysis_method.currentTextChanged.connect(self._on_global_setting_changed)
        self.window_size.valueChanged.connect(self._on_global_setting_changed)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.enable_outlier_detection.toggled.connect(self._on_global_setting_changed)
        self.zscore_threshold.valueChanged.connect(self._on_global_setting_changed)
        self.highlight_outliers.toggled.connect(self._on_global_setting_changed)
        
        self.enable_all_btn.clicked.connect(self._enable_all_signals)
        self.disable_all_btn.clicked.connect(self._disable_all_signals)
        self.auto_configure_btn.clicked.connect(self._auto_configure)
        
    def _on_sensitivity_changed(self, value):
        """Handle sensitivity slider change."""
        sensitivity_labels = {
            1: "Very Low (1)", 2: "Low (2)", 3: "Low-Med (3)", 4: "Medium-Low (4)",
            5: "Medium (5)", 6: "Medium-High (6)", 7: "High-Med (7)", 8: "High (8)",
            9: "Very High (9)", 10: "Maximum (10)"
        }
        self.sensitivity_label.setText(sensitivity_labels.get(value, f"({value})"))
        self._on_global_setting_changed()
        
    def _on_global_setting_changed(self):
        """Handle global setting change."""
        self._emit_settings_changed()
        
    def _on_setting_changed(self, signal_name: str):
        """Handle signal-specific setting change."""
        self._emit_settings_changed()
        
    def _enable_all_signals(self):
        """Enable deviation analysis for all signals."""
        for config in self.deviation_configs.values():
            config['enable'].setChecked(True)
        self._emit_settings_changed()
        
    def _disable_all_signals(self):
        """Disable deviation analysis for all signals."""
        for config in self.deviation_configs.values():
            config['enable'].setChecked(False)
        self._emit_settings_changed()
        
    def _auto_configure(self):
        """Auto-configure deviation settings based on signal characteristics."""
        # This would analyze signal data and set appropriate thresholds
        # For now, just set reasonable defaults
        for config in self.deviation_configs.values():
            config['enable'].setChecked(True)
            config['threshold'].setValue(2.0)
            config['baseline'].setCurrentText("Rolling Mean")
        self._emit_settings_changed()
        
    def _emit_settings_changed(self):
        """Emit signal when deviation settings change."""
        settings = self.get_deviation_settings()
        self.deviation_settings_changed.emit(settings)
        
    def get_deviation_settings(self) -> Dict[str, Any]:
        """Get current deviation analysis settings."""
        settings = {
            'global': {
                'analysis_method': self.analysis_method.currentText(),
                'window_size': self.window_size.value(),
                'sensitivity': self.sensitivity_slider.value(),
                'outlier_detection': self.enable_outlier_detection.isChecked(),
                'zscore_threshold': self.zscore_threshold.value(),
                'highlight_outliers': self.highlight_outliers.isChecked()
            },
            'signals': {}
        }
        
        for signal_name, config in self.deviation_configs.items():
            if config['enable'].isChecked():
                settings['signals'][signal_name] = {
                    'threshold': config['threshold'].value(),
                    'baseline_method': config['baseline'].currentText()
                }
                
        return settings
        
    def update_available_signals(self, signals: List[str]):
        """Update the list of available signals."""
        self.all_signals = signals
        self._populate_signal_settings()
        
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
        """
        
    def _get_spinbox_style(self) -> str:
        """Get spinbox styling."""
        return """
            QSpinBox, QDoubleSpinBox {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                color: #e6f3ff;
                font-size: 12px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.2);
            }
        """
