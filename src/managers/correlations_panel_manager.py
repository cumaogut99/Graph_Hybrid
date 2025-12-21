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
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
    QPushButton, QSpinBox, QListWidget, 
    QListWidgetItem, QCheckBox, QProgressBar, QDialog, QDialogButtonBox, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
from PyQt5.QtGui import QFont, QColor

logger = logging.getLogger(__name__)

class ParameterSelectionDialog(QDialog):
    """A dialog for searching and selecting a parameter."""
    def __init__(self, parameters: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Parameter")
        self.setMinimumWidth(300)
        # Apply theme styling from parent
        if hasattr(parent, 'theme_manager'):
            theme_colors = parent.theme_manager.get_theme_colors()
            is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
            
            if is_light_theme:
                bg_color = theme_colors.get('background', '#ffffff')
                text_color = theme_colors.get('text_primary', '#212121')
                surface_color = theme_colors.get('surface', '#f5f5f5')
                border_color = theme_colors.get('border', '#cccccc')
                primary_color = theme_colors.get('primary', '#2196f3')
            else:
                bg_color = theme_colors.get('background', '#1a2332')
                text_color = theme_colors.get('text_primary', '#e6f3ff')
                surface_color = theme_colors.get('surface', '#2d4a66')
                border_color = theme_colors.get('border', '#4a90e2')
                primary_color = theme_colors.get('primary', '#4a90e2')
            
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {bg_color};
                    color: {text_color};
                }}
                QLineEdit {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                    border-radius: 4px;
                    padding: 6px;
                    color: {text_color};
                }}
                QListWidget {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                    border-radius: 4px;
                    color: {text_color};
                }}
                QListWidget::item:selected {{
                    background-color: {primary_color};
                    color: white;
                }}
                QPushButton {{
                    background-color: {primary_color};
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(parent.styleSheet())

        self.all_parameters = parameters
        self.selected_parameter = None

        layout = QVBoxLayout(self)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.all_parameters)
        self.list_widget.itemDoubleClicked.connect(self._on_item_selected)
        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _filter_list(self, text: str):
        self.list_widget.clear()
        if not text:
            self.list_widget.addItems(self.all_parameters)
        else:
            filtered_items = [p for p in self.all_parameters if text.lower() in p.lower()]
            self.list_widget.addItems(filtered_items)

    def _on_item_selected(self, item: QListWidgetItem):
        self.accept()

    def accept(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            self.selected_parameter = selected_item.text()
        super().accept()

    @staticmethod
    def get_parameter(parameters: List[str], parent=None) -> Optional[str]:
        dialog = ParameterSelectionDialog(parameters, parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_parameter
        return None

class CorrelationsPanelManager:
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
        
        # UI components
        self.active_checkbox = None
        self.target_button = None
        self.results_spinbox = None
        self.results_list = None
        self.progress_bar = None
        self.available_parameters = []
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._calculate_correlations)
        self.update_timer.setSingleShot(True)  # Only fire once per trigger
        
        self._create_panel()
        
    def _apply_theme_styling(self):
        """Apply current theme styling to the correlations panel."""
        # Get theme colors from parent's theme manager
        if hasattr(self.parent, 'theme_manager'):
            theme_colors = self.parent.theme_manager.get_theme_colors()
        else:
            # Fallback to space theme colors
            theme_colors = {
                'background': '#1a2332',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'primary': '#4a90e2',
                'text_primary': '#e6f3ff',
                'text_secondary': '#ffffff',
                'border': '#4a90e2'
            }
        
        # Determine if this is a light theme
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        
        if is_light_theme:
            # Light theme colors
            widget_bg = theme_colors.get('background', '#ffffff')
            surface_bg = theme_colors.get('surface', '#f5f5f5')
            text_color = theme_colors.get('text_primary', '#212121')
            secondary_text = theme_colors.get('text_secondary', '#757575')
            border_color = theme_colors.get('border', '#cccccc')
            primary_color = theme_colors.get('primary', '#2196f3')
        else:
            # Dark theme colors
            widget_bg = theme_colors.get('background', '#1a2332')
            surface_bg = theme_colors.get('surface', '#2d4a66')
            text_color = theme_colors.get('text_primary', '#e6f3ff')
            secondary_text = theme_colors.get('text_secondary', '#ffffff')
            border_color = theme_colors.get('border', '#4a90e2')
            primary_color = theme_colors.get('primary', '#4a90e2')
        
        self.panel.setStyleSheet(f"""
            QWidget {{
                background-color: {widget_bg};
                color: {text_color};
            }}
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {border_color};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {surface_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: {widget_bg};
                border-radius: 4px;
                color: {primary_color};
            }}
            QPushButton {{
                background-color: {surface_bg};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 8px 16px;
                color: {text_color};
                font-size: 13px;
                font-weight: 600;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: {primary_color};
                border-color: {primary_color};
                color: white;
            }}
            QLabel {{
                color: {secondary_text};
                font-size: 13px;
                padding: 2px;
            }}
            QComboBox, QLineEdit, QSpinBox {{
                background-color: {surface_bg};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 6px;
                color: {text_color};
                font-size: 13px;
            }}
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{
                border-color: {primary_color};
            }}
            QCheckBox {{
                color: {text_color};
                font-size: 13px;
                font-weight: 600;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {border_color};
                border-radius: 3px;
                background-color: {surface_bg};
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary_color};
                border-color: {primary_color};
            }}
            QListWidget {{
                background-color: {surface_bg};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 5px;
                font-size: 13px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {border_color};
                border-radius: 4px;
                margin: 2px;
                color: {text_color};
            }}
            QListWidget::item:hover {{
                background-color: {primary_color};
                color: white;
            }}
            QProgressBar {{
                border: 1px solid {border_color};
                border-radius: 4px;
                background-color: {surface_bg};
                text-align: center;
                font-size: 12px;
                color: {text_color};
            }}
            QProgressBar::chunk {{
                background-color: {primary_color};
                border-radius: 3px;
            }}
        """)
    
    def update_theme(self):
        """Update the panel styling when theme changes."""
        self._apply_theme_styling()
        
    def get_panel(self) -> QWidget:
        """Get the correlations panel widget."""
        return self.panel
        
    def _create_panel(self):
        """Create the main correlations panel."""
        self.panel = QWidget()
        # Apply theme-based styling
        self._apply_theme_styling()
        
        layout = QVBoxLayout(self.panel)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("📈 Correlations Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Analysis Control Group
        self._create_analysis_controls(layout)
        
        # Target Selection Group
        self._create_target_selection(layout)
        
        # Results Configuration Group
        self._create_results_config(layout)
        
        # Results Display Group (will expand to fill remaining space)
        self._create_results_display(layout)
        
    def _create_analysis_controls(self, parent_layout):
        """Create analysis control section."""
        control_group = QGroupBox("Analysis Control")
        control_layout = QVBoxLayout(control_group)
        
        # Active/Inactive toggle
        self.active_checkbox = QCheckBox("🔄 Enable Real-time Analysis")
        self.active_checkbox.setToolTip("Enable/disable automatic correlation calculation")
        self.active_checkbox.toggled.connect(self._on_analysis_toggled)
        control_layout.addWidget(self.active_checkbox)
        
        # Progress bar for calculations
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Calculation Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        control_layout.addLayout(progress_layout)
        
        parent_layout.addWidget(control_group)
        
    def _create_target_selection(self, parent_layout):
        """Create target parameter selection section."""
        target_group = QGroupBox("Target Parameter Selection")
        target_layout = QVBoxLayout(target_group)
        
        # Target parameter button
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("🎯 Target:"))
        self.target_button = QPushButton("Select Parameter...")
        self.target_button.setStyleSheet("text-align: left; padding: 4px 8px;")
        self.target_button.clicked.connect(self._show_parameter_dialog)
        combo_layout.addWidget(self.target_button)
        target_layout.addLayout(combo_layout)
        
        parent_layout.addWidget(target_group)
        
    def _show_parameter_dialog(self):
        """Show the parameter selection dialog."""
        selected = ParameterSelectionDialog.get_parameter(self.available_parameters, self.panel)
        if selected:
            self._on_target_changed(selected)

    def _create_results_config(self, parent_layout):
        """Create results configuration section."""
        config_group = QGroupBox("Results Configuration")
        config_layout = QHBoxLayout(config_group)
        
        config_layout.addWidget(QLabel("📊 Show Top:"))
        self.results_spinbox = QSpinBox()
        self.results_spinbox.setRange(1, 50)
        self.results_spinbox.setValue(5)
        self.results_spinbox.setSuffix(" results")
        self.results_spinbox.valueChanged.connect(self._on_max_results_changed)
        config_layout.addWidget(self.results_spinbox)
        
        config_layout.addStretch()
        
        # Manual calculate button
        calc_btn = QPushButton("Calculate")
        calc_btn.setStyleSheet("padding: 4px 12px; font-size: 12px;")
        calc_btn.clicked.connect(self._calculate_correlations)
        config_layout.addWidget(calc_btn)
        
        parent_layout.addWidget(config_group)
        
    def _create_results_display(self, parent_layout):
        """Create results display section."""
        results_group = QGroupBox("Correlation Results")
        results_layout = QVBoxLayout(results_group)
        
        # Info label
        info_label = QLabel("💡 Results show correlation with target parameter (-1 to +1)")
        info_label.setStyleSheet("font-size: 12px; color: #888888; font-style: italic;")
        results_layout.addWidget(info_label)
        
        # Results list - will expand to fill available space
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(200)
        # ✅ FIX: Remove maximum height constraint, allow expansion
        results_layout.addWidget(self.results_list)
        
        # ✅ FIX: Add with stretch factor to make results group expand
        parent_layout.addWidget(results_group, 1)  # Stretch factor = 1
        
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
            
    def _calculate_correlations(self):
        """Calculate correlations between target and other parameters using cursor range."""
        if not self.is_analysis_active or not self.target_parameter:
            return
            
        logger.debug(f"Calculating correlations for target: {self.target_parameter}")
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(10)

            if not hasattr(self.parent, 'signal_processor') or not hasattr(self.parent.signal_processor, 'get_all_signals'):
                logger.warning("Signal processor not found on parent widget.")
                self.current_correlations = {}
                self._update_results_display()
                return

            all_signals = self.parent.signal_processor.get_all_signals()
            if not all_signals or self.target_parameter not in all_signals:
                logger.warning(f"Target parameter '{self.target_parameter}' not found in signals.")
                self.current_correlations = {}
                self._update_results_display()
                return

            # Get cursor positions to determine data range
            cursor_manager = getattr(self.parent, 'cursor_manager', None)
            x_data_full = all_signals[self.target_parameter]['x_data']
            target_data_full = all_signals[self.target_parameter]['y_data']
            
            # Ensure data is numpy array
            if not isinstance(x_data_full, np.ndarray):
                x_data_full = np.array(x_data_full)
            if not isinstance(target_data_full, np.ndarray):
                target_data_full = np.array(target_data_full)
            
            # Get cursor range if available
            start_idx = 0
            end_idx = len(target_data_full)
            
            if cursor_manager and hasattr(cursor_manager, 'can_zoom_to_cursors') and cursor_manager.can_zoom_to_cursors():
                try:
                    # Get cursor positions
                    pos1 = cursor_manager.dual_cursors_1[0].value()
                    pos2 = cursor_manager.dual_cursors_2[0].value()
                    
                    # Ensure proper order
                    start_pos = min(pos1, pos2)
                    end_pos = max(pos1, pos2)
                    
                    # Find indices for cursor range
                    start_idx = np.searchsorted(x_data_full, start_pos, side='left')
                    end_idx = np.searchsorted(x_data_full, end_pos, side='right')
                    
                    # Ensure valid indices
                    start_idx = max(0, start_idx)
                    end_idx = min(len(target_data_full), end_idx)
                    
                    logger.debug(f"Using cursor range: {start_pos:.3f} to {end_pos:.3f} (indices {start_idx}:{end_idx})")
                except Exception as e:
                    logger.warning(f"Failed to get cursor range, using full data: {e}")
            else:
                logger.debug("Cursors not available, using full data range")
            
            # Slice target data to cursor range
            target_data = target_data_full[start_idx:end_idx]
            correlations = {}

            total_signals = len(all_signals) - 1
            processed_signals = 0

            for name, signal_data in all_signals.items():
                if name == self.target_parameter:
                    continue
                
                try:
                    other_data_full = signal_data['y_data']
                    if not isinstance(other_data_full, np.ndarray):
                        other_data_full = np.array(other_data_full)
                    
                    # Slice other data to the same cursor range
                    other_data = other_data_full[start_idx:end_idx]

                    # ROBUST: Ensure data lengths are equal for correlation calculation
                    min_len = min(len(target_data), len(other_data))
                    
                    if min_len < 2:
                        # Correlation requires at least 2 points
                        logger.debug(f"Skipping '{name}': insufficient data points ({min_len})")
                        continue
                    
                    # Veriyi eşitle
                    target_slice = target_data[:min_len]
                    other_slice = other_data[:min_len]
                    
                    # ROBUST: None, NaN, Inf değerleri temizle
                    # Create mask for valid values
                    mask = (
                        ~np.isnan(target_slice) & 
                        ~np.isnan(other_slice) & 
                        ~np.isinf(target_slice) & 
                        ~np.isinf(other_slice) &
                        (target_slice != None) &
                        (other_slice != None)
                    )
                    
                    valid_target = target_slice[mask]
                    valid_other = other_slice[mask]
                    
                    # En az 2 valid değer gerekli
                    if len(valid_target) < 2:
                        logger.debug(f"Skipping '{name}': insufficient valid data after cleaning ({len(valid_target)})")
                        continue
                    
                    # Standart sapma kontrolü - sabit değerler için correlation tanımsız
                    if np.std(valid_target) == 0 or np.std(valid_other) == 0:
                        logger.debug(f"Skipping '{name}': zero variance (constant values)")
                        continue
                    
                    # Korelasyon hesapla
                    corr_matrix = np.corrcoef(valid_target, valid_other)
                    
                    # corrcoef returns a 2x2 matrix, the value is at [0, 1]
                    correlation = corr_matrix[0, 1]
                    
                    # Final validation
                    if not np.isnan(correlation) and not np.isinf(correlation):
                        correlations[name] = correlation
                    else:
                        logger.debug(f"Skipping '{name}': invalid correlation value ({correlation})")
                        
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for '{name}': {e}")
                    continue
                
                processed_signals += 1
                self.progress_bar.setValue(10 + int(90 * (processed_signals / total_signals)))

            self.current_correlations = correlations
            self._update_results_display()
            
            # Hide progress bar after a short delay
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}", exc_info=True)
            self.progress_bar.setVisible(False)
            
    def _update_results_display(self):
        """Update the results list display."""
        self.results_list.clear()
        
        if not self.current_correlations:
            item = QListWidgetItem("No correlations calculated yet")
            item.setForeground(QColor("#888888"))
            self.results_list.addItem(item)
            return
            
        # Sort by absolute correlation value (strongest correlations first)
        sorted_correlations = sorted(
            self.current_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Show only top N results
        for i, (param_name, correlation) in enumerate(sorted_correlations[:self.max_results]):
            self._add_correlation_item(i + 1, param_name, correlation)
            
    def _add_correlation_item(self, rank: int, param_name: str, correlation: float):
        """Add a correlation result item to the list."""
        # Format correlation value
        corr_percent = abs(correlation) * 100
        corr_sign = "+" if correlation >= 0 else "-"
        
        # Create item text
        item_text = f"{rank}. {param_name:<20} {corr_sign}{corr_percent:.1f}% ({correlation:.3f})"
        
        item = QListWidgetItem(item_text)
        
        # Color coding based on correlation strength
        if abs(correlation) >= 0.8:
            # Strong correlation - bright color
            color = QColor("#4CAF50") if correlation > 0 else QColor("#F44336")  # Green/Red
        elif abs(correlation) >= 0.5:
            # Moderate correlation - medium color
            color = QColor("#8BC34A") if correlation > 0 else QColor("#FF7043")  # Light Green/Orange
        else:
            # Weak correlation - muted color
            color = QColor("#9E9E9E")  # Gray
            
        item.setForeground(color)
        
        # Set font weight for strong correlations
        if abs(correlation) >= 0.8:
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            
        self.results_list.addItem(item)
        
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
