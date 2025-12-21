"""
Basic Deviation Panel - Simple and functional deviation analysis
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QSpinBox, QPushButton,
    QLineEdit, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class BasicDeviationPanel(QWidget):
    """Panel for basic deviation analysis with real backend functionality."""
    
    deviation_settings_changed = pyqtSignal(dict)  # Emits deviation configuration
    
    def __init__(self, all_signals: List[str] = None, parent=None):
        super().__init__(parent)
        self.all_signals = all_signals if all_signals else []
        
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """Setup the basic deviation analysis panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 Basic Deviation Analysis")
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
        desc = QLabel("Simple deviation analysis for trend following and short-term fluctuation detection.")
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
        
        # Trend Analysis Group
        trend_group = QGroupBox("🎯 Trend Analysis")
        trend_group.setStyleSheet(self._get_group_style())
        trend_layout = QFormLayout(trend_group)
        
        # Enable trend following
        self.enable_trend = QCheckBox("Enable Trend Following")
        self.enable_trend.setStyleSheet("color: #e6f3ff;")
        self.enable_trend.setChecked(False)
        trend_layout.addRow(self.enable_trend)
        
        # Trend sensitivity
        self.trend_sensitivity = QSlider(Qt.Horizontal)
        self.trend_sensitivity.setRange(1, 5)
        self.trend_sensitivity.setValue(3)
        self.trend_label = QLabel("Medium (3)")
        self.trend_label.setStyleSheet("color: #e6f3ff; font-size: 11px;")
        
        trend_sens_layout = QHBoxLayout()
        trend_sens_layout.addWidget(self.trend_sensitivity)
        trend_sens_layout.addWidget(self.trend_label)
        trend_layout.addRow("Trend Sensitivity:", trend_sens_layout)
        
        layout.addWidget(trend_group)
        
        # Parameter Selection Group with search and multi-select
        parameter_group = QGroupBox("📊 Parameter Selection")
        parameter_group.setStyleSheet(self._get_group_style())
        parameter_layout = QVBoxLayout(parameter_group)
        parameter_layout.setSpacing(10)
        parameter_layout.setContentsMargins(15, 15, 15, 15)
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #e6f3ff; font-weight: bold; font-size: 13px;")
        
        self.parameter_search = QLineEdit()
        self.parameter_search.setPlaceholderText("Type to search parameters...")
        self.parameter_search.setStyleSheet(self._get_search_style())
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.parameter_search)
        parameter_layout.addLayout(search_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        
        button_style = """
            QPushButton {
                padding: 6px 12px;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 transparent);
                color: #e6f3ff;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.1));
                border-color: #4a90e2;
            }
        """
        
        self.select_all_btn.setStyleSheet(button_style)
        self.deselect_all_btn.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.select_all_btn)
        controls_layout.addWidget(self.deselect_all_btn)
        controls_layout.addStretch()
        
        parameter_layout.addLayout(controls_layout)
        
        # Scroll area for parameter checkboxes
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
        scroll.setMaximumHeight(200)  # Limit height
        
        # Container for parameter checkboxes
        self.parameters_container = QWidget()
        self.parameters_container.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
        """)
        self.parameters_layout = QVBoxLayout(self.parameters_container)
        self.parameters_layout.setSpacing(5)
        
        # Store parameter checkboxes
        self.parameter_checkboxes = {}
        
        # Populate with parameters
        self._populate_parameter_checkboxes()
        
        scroll.setWidget(self.parameters_container)
        parameter_layout.addWidget(scroll)
        
        layout.addWidget(parameter_group)
        
        # Short-term Fluctuation Group
        fluctuation_group = QGroupBox("Short-term Fluctuation Detection")
        fluctuation_group.setStyleSheet(self._get_group_style())
        fluctuation_layout = QFormLayout(fluctuation_group)
        
        # Enable detection
        self.enable_fluctuation = QCheckBox("Enable Fluctuation Detection")
        self.enable_fluctuation.setStyleSheet("color: #e6f3ff;")
        self.enable_fluctuation.setChecked(False)
        fluctuation_layout.addRow(self.enable_fluctuation)
        
        # Sample window (Son X parametre)
        self.sample_window = QSpinBox()
        self.sample_window.setRange(5, 100)
        self.sample_window.setValue(20)
        self.sample_window.setSuffix(" samples")
        self.sample_window.setStyleSheet(self._get_spinbox_style())
        fluctuation_layout.addRow("Sample Window:", self.sample_window)
        
        # Deviation threshold (%Y sapma)
        self.deviation_threshold = QDoubleSpinBox()
        self.deviation_threshold.setRange(1.0, 50.0)
        self.deviation_threshold.setValue(10.0)
        self.deviation_threshold.setSuffix("%")
        self.deviation_threshold.setDecimals(1)
        self.deviation_threshold.setStyleSheet(self._get_spinbox_style())
        fluctuation_layout.addRow("Deviation Threshold:", self.deviation_threshold)
        
        # Highlight on graph
        self.highlight_deviations = QCheckBox("Highlight Deviations on Graph")
        self.highlight_deviations.setStyleSheet("color: #e6f3ff;")
        self.highlight_deviations.setChecked(False)
        fluctuation_layout.addRow(self.highlight_deviations)
        
        # Red highlighting for threshold exceedance
        self.red_highlighting = QCheckBox("Red Highlighting for Threshold Exceedance")
        self.red_highlighting.setStyleSheet("color: #e6f3ff;")
        self.red_highlighting.setChecked(False)
        fluctuation_layout.addRow(self.red_highlighting)
        
        layout.addWidget(fluctuation_group)
        
        # Visual Settings Group
        visual_group = QGroupBox("📈 Visual Settings")
        visual_group.setStyleSheet(self._get_group_style())
        visual_layout = QFormLayout(visual_group)
        
        # Show deviation bands
        self.show_bands = QCheckBox("Show Deviation Bands")
        self.show_bands.setStyleSheet("color: #e6f3ff;")
        self.show_bands.setChecked(False)
        visual_layout.addRow(self.show_bands)
        
        # Band transparency
        self.band_transparency = QSlider(Qt.Horizontal)
        self.band_transparency.setRange(10, 80)
        self.band_transparency.setValue(30)
        self.transparency_label = QLabel("30%")
        self.transparency_label.setStyleSheet("color: #e6f3ff; font-size: 11px;")
        
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(self.band_transparency)
        transparency_layout.addWidget(self.transparency_label)
        visual_layout.addRow("Band Transparency:", transparency_layout)
        
        layout.addWidget(visual_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Settings")
        self.reset_btn = QPushButton("Reset to Defaults")
        
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
        
        self.apply_btn.setStyleSheet(button_style)
        self.reset_btn.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.apply_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addStretch()
        
    def _setup_connections(self):
        """Setup signal connections."""
        # Trend settings
        self.enable_trend.toggled.connect(self._on_settings_changed)
        self.trend_sensitivity.valueChanged.connect(self._on_trend_sensitivity_changed)
        
        # Parameter settings
        self.parameter_search.textChanged.connect(self._filter_parameters)
        self.select_all_btn.clicked.connect(self._select_all_parameters)
        self.deselect_all_btn.clicked.connect(self._deselect_all_parameters)
        
        # Fluctuation settings
        self.enable_fluctuation.toggled.connect(self._on_settings_changed)
        self.sample_window.valueChanged.connect(self._on_settings_changed)
        self.deviation_threshold.valueChanged.connect(self._on_settings_changed)
        self.highlight_deviations.toggled.connect(self._on_settings_changed)
        self.red_highlighting.toggled.connect(self._on_settings_changed)
        
        # Visual settings
        self.show_bands.toggled.connect(self._on_settings_changed)
        self.band_transparency.valueChanged.connect(self._on_transparency_changed)
        
        # Control buttons
        self.apply_btn.clicked.connect(self._apply_settings)
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        
    def _on_trend_sensitivity_changed(self, value):
        """Handle trend sensitivity slider change."""
        sensitivity_labels = {
            1: "Very Low (1)", 2: "Low (2)", 3: "Medium (3)", 
            4: "High (4)", 5: "Very High (5)"
        }
        self.trend_label.setText(sensitivity_labels.get(value, f"({value})"))
        self._on_settings_changed()
        
    def _on_transparency_changed(self, value):
        """Handle transparency slider change."""
        self.transparency_label.setText(f"{value}%")
        self._on_settings_changed()
        
    def _on_settings_changed(self):
        """Handle any setting change."""
        settings = self.get_deviation_settings()
        self.deviation_settings_changed.emit(settings)
        
    def _apply_settings(self):
        """Apply current settings."""
        settings = self.get_deviation_settings()
        logger.info(f"[BASIC DEVIATION] Applied settings: {settings}")
        logger.info(f"[BASIC DEVIATION] Selected parameters: {settings.get('selected_parameters', [])}")
        logger.info(f"[BASIC DEVIATION] Trend enabled: {settings.get('trend_analysis', {}).get('enabled', False)}")
        logger.info(f"[BASIC DEVIATION] Fluctuation enabled: {settings.get('fluctuation_detection', {}).get('enabled', False)}")
        self.deviation_settings_changed.emit(settings)
        
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.enable_trend.setChecked(False)
        self.trend_sensitivity.setValue(3)
        self.enable_fluctuation.setChecked(False)
        self.sample_window.setValue(20)
        self.deviation_threshold.setValue(10.0)
        self.highlight_deviations.setChecked(False)
        self.red_highlighting.setChecked(False)
        self.show_bands.setChecked(False)
        self.band_transparency.setValue(30)
        
    def get_deviation_settings(self) -> Dict[str, Any]:
        """Get current basic deviation settings."""
        # Get selected parameters from checkboxes
        selected_parameters = []
        for param_name, checkbox in self.parameter_checkboxes.items():
            if checkbox.isChecked():
                selected_parameters.append(param_name)
        
        return {
            'type': 'basic',
            'selected_parameters': selected_parameters,
            'trend_analysis': {
                'enabled': self.enable_trend.isChecked(),
                'sensitivity': self.trend_sensitivity.value()
            },
            'fluctuation_detection': {
                'enabled': self.enable_fluctuation.isChecked(),
                'sample_window': self.sample_window.value(),
                'threshold_percent': self.deviation_threshold.value(),
                'highlight_on_graph': self.highlight_deviations.isChecked(),
                'red_highlighting': self.red_highlighting.isChecked()
            },
            'visual_settings': {
                'show_bands': self.show_bands.isChecked(),
                'band_transparency': self.band_transparency.value()
            }
        }
        
    def calculate_deviation_for_signal(self, signal_data: np.ndarray, signal_name: str) -> Dict[str, Any]:
        """
        Calculate deviation for a signal based on current settings.
        This is the real backend functionality.
        """
        if len(signal_data) == 0:
            return {'deviations': [], 'bands': [], 'alerts': []}
            
        settings = self.get_deviation_settings()
        results = {
            'deviations': [],
            'bands': [],
            'alerts': [],
            'trend_line': []
        }
        
        # Trend Analysis
        if settings['trend_analysis']['enabled']:
            trend_line = self._calculate_trend_line(signal_data, settings['trend_analysis']['sensitivity'])
            results['trend_line'] = trend_line
            
        # Fluctuation Detection
        if settings['fluctuation_detection']['enabled']:
            window_size = settings['fluctuation_detection']['sample_window']
            threshold_percent = settings['fluctuation_detection']['threshold_percent']
            
            deviations, alerts = self._detect_fluctuations(
                signal_data, window_size, threshold_percent
            )
            results['deviations'] = deviations
            results['alerts'] = alerts
            
        # Deviation Bands
        if settings['visual_settings']['show_bands']:
            bands = self._calculate_deviation_bands(signal_data, settings)
            results['bands'] = bands
            
        return results
        
    def _calculate_trend_line(self, data: np.ndarray, sensitivity: int) -> List[float]:
        """Calculate trend line based on sensitivity."""
        if len(data) < 2:
            return data.tolist()
            
        # Simple linear regression for trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend_line = np.polyval(coeffs, x)
        
        # Apply sensitivity smoothing
        if sensitivity <= 2:  # Low sensitivity - more smoothing
            window = min(len(data) // 4, 20)
        elif sensitivity >= 4:  # High sensitivity - less smoothing
            window = min(len(data) // 10, 5)
        else:  # Medium sensitivity
            window = min(len(data) // 8, 10)
            
        if window > 1:
            # Apply moving average smoothing
            trend_smoothed = np.convolve(trend_line, np.ones(window)/window, mode='same')
            return trend_smoothed.tolist()
        
        return trend_line.tolist()
        
    def _detect_fluctuations(self, data: np.ndarray, window_size: int, threshold_percent: float) -> tuple:
        """Detect short-term fluctuations."""
        deviations = []
        alerts = []
        
        if len(data) < window_size:
            return deviations, alerts
            
        for i in range(window_size, len(data)):
            # Get last window_size samples
            window_data = data[i-window_size:i]
            window_mean = np.mean(window_data)
            current_value = data[i]
            
            # Calculate percentage deviation
            if window_mean != 0:
                deviation_percent = abs((current_value - window_mean) / window_mean) * 100
                deviations.append(deviation_percent)
                
                # Check if it exceeds threshold
                if deviation_percent > threshold_percent:
                    alerts.append({
                        'index': i,
                        'value': current_value,
                        'expected': window_mean,
                        'deviation_percent': deviation_percent
                    })
            else:
                deviations.append(0.0)
                
        return deviations, alerts
        
    def _calculate_deviation_bands(self, data: np.ndarray, settings: Dict) -> Dict[str, List[float]]:
        """Calculate deviation bands for visualization."""
        if len(data) < 10:
            return {'upper': [], 'lower': []}
            
        window_size = settings['fluctuation_detection']['sample_window']
        threshold_percent = settings['fluctuation_detection']['threshold_percent']
        
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i+1]
            
            if len(window_data) > 0:
                mean_val = np.mean(window_data)
                threshold_val = mean_val * (threshold_percent / 100)
                
                upper_band.append(mean_val + threshold_val)
                lower_band.append(mean_val - threshold_val)
            else:
                upper_band.append(data[i])
                lower_band.append(data[i])
                
        return {'upper': upper_band, 'lower': lower_band}
        
    def set_deviation_settings(self, settings: Dict[str, Any]):
        """Set deviation settings from saved data."""
        try:
            # Set trend analysis settings
            trend_settings = settings.get('trend_analysis', {})
            if 'enabled' in trend_settings:
                self.enable_trend.setChecked(trend_settings['enabled'])
            if 'sensitivity' in trend_settings:
                self.trend_sensitivity.setValue(trend_settings['sensitivity'])
                
            # Set selected parameters
            selected_parameters = settings.get('selected_parameters', [])
            for param_name, checkbox in self.parameter_checkboxes.items():
                checkbox.setChecked(param_name in selected_parameters)
                
            # Set fluctuation detection settings
            fluctuation_settings = settings.get('fluctuation_detection', {})
            if 'enabled' in fluctuation_settings:
                self.enable_fluctuation.setChecked(fluctuation_settings['enabled'])
            if 'sample_window' in fluctuation_settings:
                self.sample_window.setValue(fluctuation_settings['sample_window'])
            if 'threshold_percent' in fluctuation_settings:
                self.deviation_threshold.setValue(fluctuation_settings['threshold_percent'])
            if 'highlight_on_graph' in fluctuation_settings:
                self.highlight_deviations.setChecked(fluctuation_settings['highlight_on_graph'])
            if 'red_highlighting' in fluctuation_settings:
                self.red_highlighting.setChecked(fluctuation_settings['red_highlighting'])
                
            # Set visual settings
            visual_settings = settings.get('visual_settings', {})
            if 'show_bands' in visual_settings:
                self.show_bands.setChecked(visual_settings['show_bands'])
            if 'band_transparency' in visual_settings:
                self.band_transparency.setValue(visual_settings['band_transparency'])
                
            logger.info(f"Basic deviation settings loaded successfully")
            
        except Exception as e:
            logger.error(f"Error setting basic deviation settings: {e}")
        
    def update_available_signals(self, signals: List[str]):
        """Update the list of available signals."""
        self.all_signals = signals
        
        # Store current selections
        current_selections = []
        for param_name, checkbox in self.parameter_checkboxes.items():
            if checkbox.isChecked():
                current_selections.append(param_name)
        
        # Repopulate parameter checkboxes
        self._populate_parameter_checkboxes()
        
        # Restore previous selections if they still exist
        for selection in current_selections:
            if selection in self.parameter_checkboxes:
                self.parameter_checkboxes[selection].setChecked(True)
                
    def _populate_parameter_checkboxes(self):
        """Populate parameter checkboxes."""
        # Clear existing checkboxes
        for i in reversed(range(self.parameters_layout.count())):
            child = self.parameters_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.parameter_checkboxes.clear()
        
        # Add checkboxes for each parameter
        if not self.all_signals:
            # Debug: Add test parameters if no signals available
            test_signals = ["Test Signal 1", "Test Signal 2", "Test Signal 3"]
            logger.warning("No signals provided, using test signals")
            for signal in test_signals:
                self._create_parameter_checkbox(signal)
        else:
            for signal in self.all_signals:
                self._create_parameter_checkbox(signal)
                
    def _create_parameter_checkbox(self, signal_name: str):
        """Create a checkbox for a parameter."""
        checkbox = QCheckBox(signal_name)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #e6f3ff;
                font-size: 12px;
                padding: 4px;
                background: transparent;
                border: none;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid rgba(74, 144, 226, 0.5);
                border-radius: 3px;
                background: rgba(74, 144, 226, 0.1);
            }
            QCheckBox::indicator:hover {
                border-color: #4a90e2;
                background: rgba(74, 144, 226, 0.2);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 #5ba0f2);
                border-color: #4a90e2;
            }
            QCheckBox::indicator:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ba0f2, stop:1 #6bb0ff);
            }
        """)
        
        # Connect to settings change
        checkbox.toggled.connect(self._on_settings_changed)
        
        # Store reference
        self.parameter_checkboxes[signal_name] = checkbox
        
        # Add to layout
        self.parameters_layout.addWidget(checkbox)
        
    def _filter_parameters(self, search_text: str):
        """Filter parameters based on search text."""
        search_text = search_text.lower().strip()
        
        for param_name, checkbox in self.parameter_checkboxes.items():
            if search_text == '' or search_text in param_name.lower():
                checkbox.setVisible(True)
            else:
                checkbox.setVisible(False)
                
    def _select_all_parameters(self):
        """Select all visible parameters."""
        for checkbox in self.parameter_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setChecked(True)
        self._on_settings_changed()
        
    def _deselect_all_parameters(self):
        """Deselect all parameters."""
        for checkbox in self.parameter_checkboxes.values():
            checkbox.setChecked(False)
        self._on_settings_changed()
        
    def _get_search_style(self) -> str:
        """Get search box styling."""
        return """
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
        """
        
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
        
    def _get_combo_style(self) -> str:
        """Get combo box styling."""
        return """
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 rgba(74, 144, 226, 0.1));
                border: 2px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                padding: 8px 12px;
                color: #e6f3ff;
                font-size: 13px;
                font-weight: 500;
                min-width: 200px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #4a90e2;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.2));
            }
            QComboBox:focus {
                border-color: #5ba0f2;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.2));
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 2px;
                border-left-color: rgba(74, 144, 226, 0.5);
                border-left-style: solid;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background: rgba(74, 144, 226, 0.2);
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #e6f3ff;
                width: 0px;
                height: 0px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #ffffff;
            }
            QComboBox QAbstractItemView {
                background: #2d4a66;
                border: 2px solid rgba(74, 144, 226, 0.7);
                border-radius: 4px;
                selection-background-color: rgba(74, 144, 226, 0.5);
                color: #e6f3ff;
                font-size: 13px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                border-bottom: 1px solid rgba(74, 144, 226, 0.2);
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(74, 144, 226, 0.3);
                color: #ffffff;
            }
            QComboBox QAbstractItemView::item:selected {
                background: rgba(74, 144, 226, 0.6);
                color: #ffffff;
                font-weight: bold;
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
