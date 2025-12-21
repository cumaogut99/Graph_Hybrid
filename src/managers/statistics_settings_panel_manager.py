"""
Statistics Settings Panel Manager for Time Graph Widget

Manages the settings panel for statistical calculations and display.
"""

import logging
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QScrollArea,
    QGroupBox, QFormLayout, QLabel, QCheckBox,
    QRadioButton, QDoubleSpinBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class StatisticsSettingsPanelManager(QObject):
    """Manages the statistics settings panel interface."""
    
    # Emits a set of the names of the visible columns
    visible_columns_changed = Signal(set)
    # Emits duty cycle threshold settings (threshold_mode: str, threshold_value: float)
    duty_cycle_threshold_changed = Signal(str, float)
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.settings_panel = None
        self.column_checkboxes = {}
        self.current_cursor_mode = "dual"  # Cursor mode is permanently 'dual'
        
        # Duty cycle threshold settings
        self.duty_cycle_group = None
        self.auto_threshold_radio = None
        self.manual_threshold_radio = None
        self.threshold_spinbox = None
        
        self._setup_settings_panel()
        self._connect_signals()
    
    def _setup_settings_panel(self):
        """Create the main settings panel."""
        self.settings_panel = QFrame()
        self.settings_panel.setMinimumWidth(280)
        self.settings_panel.setMaximumWidth(350)
        # Apply similar styling to the general settings panel
        self.settings_panel.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #444;
                border-radius: 8px;
            }
            QGroupBox {
                color: #e0e0e0;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: #383838;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #5d9cec;
            }
            QLabel {
                color: #d0d0d0;
                font-size: 10px;
            }
        """)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Create the settings sections
        self._create_visibility_settings(main_layout)
        self._create_duty_cycle_settings(main_layout)
        
        main_layout.addStretch()
        
        scroll_area.setWidget(container_widget)
        
        panel_layout = QVBoxLayout(self.settings_panel)
        panel_layout.setContentsMargins(0,0,0,0)
        panel_layout.addWidget(scroll_area)

    def _create_visibility_settings(self, parent_layout):
        """Create a group for toggling statistic column visibility."""
        group = QGroupBox("📊 Visible Columns")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Define the available statistic columns (display_name, key_name)
        stats_options = [
            ("Min", "min"),
            ("Mean", "mean"),
            ("Max", "max"),
            ("RMS (C1)", "rms"),
            ("Std Dev", "std"),
            ("Duty Cycle", "duty_cycle")
        ]
        
        for display_name, key in stats_options:
            checkbox = QCheckBox(display_name)
            checkbox.setChecked(True) # Default to visible
            layout.addWidget(checkbox)
            self.column_checkboxes[key] = checkbox

        parent_layout.addWidget(group)

    def _create_duty_cycle_settings(self, parent_layout):
        """Create duty cycle threshold settings group."""
        self.duty_cycle_group = QGroupBox("⏱️ Duty Cycle Settings")
        layout = QVBoxLayout(self.duty_cycle_group)
        layout.setSpacing(8)
        
        # Threshold mode selection
        threshold_label = QLabel("Threshold Calculation:")
        threshold_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(threshold_label)
        
        # Auto threshold (mean) radio button
        self.auto_threshold_radio = QRadioButton("Automatic (Signal Mean)")
        self.auto_threshold_radio.setChecked(True)  # Default selection
        self.auto_threshold_radio.setToolTip("Use signal mean as threshold for duty cycle calculation")
        layout.addWidget(self.auto_threshold_radio)
        
        # Manual threshold radio button and spinbox
        manual_layout = QHBoxLayout()
        self.manual_threshold_radio = QRadioButton("Manual Threshold:")
        manual_layout.addWidget(self.manual_threshold_radio)
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-1000000.0, 1000000.0)
        self.threshold_spinbox.setDecimals(6)
        self.threshold_spinbox.setValue(0.0)
        self.threshold_spinbox.setSuffix(" V")
        self.threshold_spinbox.setToolTip("Enter custom threshold value for duty cycle calculation")
        self.threshold_spinbox.setEnabled(False)  # Initially disabled
        manual_layout.addWidget(self.threshold_spinbox)
        
        layout.addLayout(manual_layout)
        
        # Info label
        info_label = QLabel("💡 Duty cycle is calculated as the percentage of time\nthe signal stays above the threshold value.")
        info_label.setStyleSheet("color: #a0a0a0; font-size: 9px; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Initially disable the entire group (will be enabled when duty cycle is checked)
        self.duty_cycle_group.setEnabled(False)
        
        parent_layout.addWidget(self.duty_cycle_group)

    def _connect_signals(self):
        """Connect checkbox signals to the handler."""
        for checkbox in self.column_checkboxes.values():
            checkbox.stateChanged.connect(self._on_visibility_changed)
        
        # Connect duty cycle threshold signals
        if self.auto_threshold_radio:
            self.auto_threshold_radio.toggled.connect(self._on_threshold_mode_changed)
        if self.manual_threshold_radio:
            self.manual_threshold_radio.toggled.connect(self._on_threshold_mode_changed)
        if self.threshold_spinbox:
            self.threshold_spinbox.valueChanged.connect(self._on_threshold_value_changed)
        
        # Connect duty cycle checkbox to enable/disable threshold settings
        if 'duty_cycle' in self.column_checkboxes:
            self.column_checkboxes['duty_cycle'].stateChanged.connect(self._on_duty_cycle_toggled)

    def _on_visibility_changed(self):
        """Emit the set of currently visible columns."""
        visible_set = {key for key, cb in self.column_checkboxes.items() if cb.isChecked()}
        self.visible_columns_changed.emit(visible_set)
        logger.info(f"Visible statistics columns changed: {visible_set}")

    def _on_duty_cycle_toggled(self, state):
        """Enable/disable duty cycle threshold settings based on checkbox state."""
        is_enabled = state == Qt.Checked
        if self.duty_cycle_group:
            self.duty_cycle_group.setEnabled(is_enabled)
        
        # Emit threshold settings when duty cycle is enabled
        if is_enabled:
            self._emit_threshold_settings()

    def _on_threshold_mode_changed(self):
        """Handle threshold mode radio button changes."""
        if self.manual_threshold_radio and self.manual_threshold_radio.isChecked():
            self.threshold_spinbox.setEnabled(True)
        else:
            self.threshold_spinbox.setEnabled(False)
        
        self._emit_threshold_settings()

    def _on_threshold_value_changed(self):
        """Handle manual threshold value changes."""
        self._emit_threshold_settings()

    def _emit_threshold_settings(self):
        """Emit current threshold settings."""
        if not self.auto_threshold_radio or not self.manual_threshold_radio:
            return
            
        if self.auto_threshold_radio.isChecked():
            threshold_mode = "auto"
            threshold_value = 0.0  # Will be calculated automatically
        else:
            threshold_mode = "manual"
            threshold_value = self.threshold_spinbox.value()
        
        self.duty_cycle_threshold_changed.emit(threshold_mode, threshold_value)
        logger.info(f"Duty cycle threshold changed: mode={threshold_mode}, value={threshold_value}")

    def get_settings_panel(self) -> QWidget:
        """Get the settings panel widget."""
        return self.settings_panel

    def get_visible_columns(self) -> set:
        """Return the set of currently visible columns."""
        visible_set = {key for key, cb in self.column_checkboxes.items() if cb.isChecked()}
        
        # Remove RMS if cursor mode is not dual
        if self.current_cursor_mode != 'dual':
            visible_set.discard('rms')
            
        return visible_set

    def get_duty_cycle_threshold_settings(self) -> tuple:
        """Get current duty cycle threshold settings."""
        if not self.auto_threshold_radio or not self.manual_threshold_radio:
            return ("auto", 0.0)
            
        if self.auto_threshold_radio.isChecked():
            return ("auto", 0.0)
        else:
            return ("manual", self.threshold_spinbox.value())
    
    def set_cursor_mode(self, mode: str):
        """Update cursor mode and enable/disable RMS checkbox accordingly."""
        # Force mode to 'dual' - other modes not supported
        if mode != 'dual':
            mode = 'dual'
            
        self.current_cursor_mode = mode
        
        # RMS checkbox is always enabled for dual mode (permanent setting)
        if 'rms' in self.column_checkboxes:
            rms_checkbox = self.column_checkboxes['rms']
            rms_checkbox.setEnabled(True)
            rms_checkbox.setText("RMS (C1)")
        
        # Emit updated visible columns
        self._on_visibility_changed()
