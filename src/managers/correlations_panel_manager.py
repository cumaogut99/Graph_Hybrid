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
                font-size: 12px;
                border: 1px solid {border_color};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 6px;
                background-color: {surface_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                background-color: {widget_bg};
                border-radius: 4px;
                color: {primary_color};
            }}
            QPushButton {{
                background-color: {surface_bg};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 3px 10px;
                color: {text_color};
                font-size: 11px;
                font-weight: 600;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {primary_color};
                border-color: {primary_color};
                color: white;
            }}
            QLabel {{
                color: {secondary_text};
                font-size: 11px;
                padding: 1px;
            }}
            QComboBox, QLineEdit, QSpinBox {{
                background-color: {surface_bg};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 3px 6px;
                color: {text_color};
                font-size: 11px;
            }}
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{
                border-color: {primary_color};
            }}
            QCheckBox {{
                color: {text_color};
                font-size: 11px;
                font-weight: 600;
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 13px;
                height: 13px;
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
                padding: 3px;
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 4px 6px;
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
                font-size: 10px;
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
        layout.setSpacing(6)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Title
        title = QLabel("📈 Correlations Analysis")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #4a90e2; margin-bottom: 4px;")
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
        calc_btn.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        calc_btn.clicked.connect(lambda: self._calculate_correlations(force=True))
        config_layout.addWidget(calc_btn)
        
        parent_layout.addWidget(config_group)
        
    def _create_results_display(self, parent_layout):
        """Create results display section."""
        results_group = QGroupBox("Correlation Results")
        results_layout = QVBoxLayout(results_group)
        
        # Info label
        info_label = QLabel("💡 Results show correlation with target parameter (-1 to +1)")
        info_label.setStyleSheet("font-size: 10px; color: #888888; font-style: italic;")
        results_layout.addWidget(info_label)

        # Data range (cursor range) the results were calculated on
        self.range_label = QLabel()
        self.range_label.setStyleSheet("font-size: 11px; color: #b0bec5;")
        self.range_label.setVisible(False)
        results_layout.addWidget(self.range_label)
        
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
            
    def _calculate_correlations(self, force: bool = False):
        """
        Pearson correlation of the target with every other signal, on the raw
        (full-resolution) data between the cursors, or all data without cursors.

        force: calculate even when real-time analysis is off (Calculate button).
        """
        if not self.target_parameter or not (self.is_analysis_active or force):
            return

        logger.debug(f"Calculating correlations for target: {self.target_parameter}")

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(10)

            signal_processor = getattr(self.parent, 'signal_processor', None)
            if signal_processor is None or not hasattr(signal_processor, 'get_raw_range'):
                logger.warning("Signal processor not found on parent widget.")
                self.current_correlations = {}
                self.range_text = ""
                self._update_results_display()
                return

            start_pos, end_pos = self._cursor_range()
            target = signal_processor.get_raw_range(self.target_parameter, start_pos, end_pos)
            if target is None:
                logger.warning(f"Target parameter '{self.target_parameter}' not found in signals.")
                self.current_correlations = {}
                self.range_text = ""
                self._update_results_display()
                return
            target_x, target_y = target
            self.range_text = self._format_range(start_pos, end_pos, len(target_x))

            other_names = [name for name in signal_processor.signal_data if name != self.target_parameter]
            correlations = {}

            for processed, name in enumerate(other_names, 1):
                try:
                    other = signal_processor.get_raw_range(name, start_pos, end_pos)
                    if other is not None:
                        correlation = self._pearson(target_x, target_y, *other)
                        if correlation is not None:
                            correlations[name] = correlation
                        else:
                            logger.debug(f"Skipping '{name}': not enough varying data in range")
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for '{name}': {e}")
                self.progress_bar.setValue(10 + int(90 * processed / len(other_names)))

            self.current_correlations = correlations
            self._update_results_display()

            # Hide progress bar after a short delay
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}", exc_info=True)
            self.progress_bar.setVisible(False)

    def _cursor_range(self) -> Tuple[Optional[float], Optional[float]]:
        """(start, end) between the two cursors, or (None, None): all data."""
        cursor_manager = getattr(self.parent, 'cursor_manager', None)
        if cursor_manager and hasattr(cursor_manager, 'can_zoom_to_cursors') and cursor_manager.can_zoom_to_cursors():
            try:
                pos1 = cursor_manager.dual_cursors_1[0].value()
                pos2 = cursor_manager.dual_cursors_2[0].value()
                return min(pos1, pos2), max(pos1, pos2)
            except Exception as e:
                logger.warning(f"Failed to get cursor range, using full data: {e}")
        return None, None

    def _format_range(self, start: Optional[float], end: Optional[float], sample_count: int) -> str:
        """'Range: <start> – <end> (<n> samples)' in the time axis' format."""
        if start is None:
            return f"Range: all data ({sample_count:,} samples)"
        container = None
        if hasattr(self.parent, 'get_active_graph_container'):
            container = self.parent.get_active_graph_container()
        if container is not None and hasattr(container.plot_manager, 'format_x_value'):
            fmt = container.plot_manager.format_x_value
        else:
            fmt = lambda x: f"{x:.6g}"
        return f"Range: {fmt(start)} – {fmt(end)} ({sample_count:,} samples)"

    @staticmethod
    def _pearson(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Optional[float]:
        """
        Correlation of two signals sampled at the same instants. Signals on
        another time base are interpolated onto the first one's times (only
        where both have data); None if undefined (< 2 points or constant).
        """
        if len(x1) == len(x2) and np.array_equal(x1, x2):
            a, b = y1, y2
        else:
            if len(x2) < 2:
                return None
            overlap = (x1 >= x2[0]) & (x1 <= x2[-1])
            a, b = y1[overlap], np.interp(x1[overlap], x2, y2)

        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return None
        correlation = float(np.corrcoef(a, b)[0, 1])
        return correlation if np.isfinite(correlation) else None

    def _update_results_display(self):
        """Update the results list display."""
        self.range_label.setText(self.range_text)
        self.range_label.setVisible(bool(self.range_text))
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
            font = QFont(self.results_list.font())  # keep the list's size
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
