# type: ignore
"""
Correlations Panel Manager for Time Graph Widget

Manages correlation analysis between signals with real-time updates.
Features:
- Target parameter selection with search
- Pearson correlation calculation
- Real-time cursor-based updates
- Color-coded results
- Configurable result count
"""

import logging
from typing import Dict, List, Any
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox, QLineEdit
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
from src.managers.correlations.parameter_dialog import ParameterSelectionDialog
from src.managers.correlations.panel_ui import CorrelationsPanelUiMixin
from src.managers.correlations.calculation import CorrelationsCalculationMixin

logger = logging.getLogger(__name__)


class CorrelationsPanelManager(CorrelationsPanelUiMixin, CorrelationsCalculationMixin):
    """Manages the correlations analysis panel."""
    
    # Signals
    correlation_calculated = Signal(dict)  # Emits correlation results
    
    def __init__(self, parent_widget=None):
        self.parent = parent_widget
        self.panel = None
        
        # Analysis settings
        self.is_analysis_active = False
        self.target_parameter = None
        self.max_results = 5
        self.current_correlations = {}
        self.range_text = ""  # Data range of the current results
        
        # UI components
        self.active_checkbox = None
        self.target_button = None
        self.results_spinbox = None
        self.results_list = None
        self.range_label = None
        self.progress_bar = None
        self.available_parameters = []
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._calculate_correlations)
        self.update_timer.setSingleShot(True)  # Only fire once per trigger
        
        self._create_panel()
        
    def update_theme(self):
        """Update the panel styling when theme changes."""
        self._apply_theme_styling()
        
    def get_panel(self) -> QWidget:
        """Get the correlations panel widget."""
        return self.panel
        
    def _show_parameter_dialog(self):
        """Show the parameter selection dialog."""
        selected = ParameterSelectionDialog.get_parameter(self.available_parameters, self.panel)
        if selected:
            self._on_target_changed(selected)

    def _on_analysis_toggled(self, checked: bool):
        """Handle analysis active/inactive toggle."""
        self.is_analysis_active = checked
        logger.info(f"Correlation analysis {'enabled' if checked else 'disabled'}")
        
        if checked:
            self._trigger_calculation()
        else:
            self.update_timer.stop()
            
    def _on_target_changed(self, target_name: str):
        """Handle target parameter selection change."""
        if target_name and target_name != self.target_parameter:
            self.target_parameter = target_name
            self.target_button.setText(target_name)
            # Results of the previous target must not be shown under this one
            self.current_correlations = {}
            self.range_text = ""
            self._update_results_display()
            logger.debug(f"Target parameter changed to: {target_name}")
            if self.is_analysis_active:
                self._trigger_calculation()
                
    def _on_max_results_changed(self, value: int):
        """Handle max results count change."""
        self.max_results = value
        self._update_results_display()
        
    def _filter_target_combo(self, search_text: str):
        """Filter target combo based on search text."""
        # TODO: Implement parameter filtering
        pass
        
    def _trigger_calculation(self):
        """Trigger correlation calculation with a small delay."""
        if self.is_analysis_active and self.target_parameter:
            self.update_timer.stop()
            self.update_timer.start(500)  # 500ms delay to avoid too frequent updates
            
    def update_available_parameters(self, parameters: List[str]):
        """Update the list of available parameters for target selection."""
        self.available_parameters = parameters
        # Set initial or restore previous target
        if self.target_parameter and self.target_parameter in self.available_parameters:
            self.target_button.setText(self.target_parameter)
        elif self.available_parameters:
            # Set a default target if none is selected
            self._on_target_changed(self.available_parameters[0])
        else:
            self.target_button.setText("No parameters available")
            
    def on_cursor_moved(self, cursor_positions: Dict[str, float]):
        """Handle cursor movement for real-time updates."""
        if self.is_analysis_active:
            self._trigger_calculation()
            
    def on_data_changed(self):
        """Handle data changes that might affect correlations."""
        if self.is_analysis_active:
            self._trigger_calculation()
