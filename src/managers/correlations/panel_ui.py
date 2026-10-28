"""
Building the correlations panel, its theme and the results list.

Part of CorrelationsPanelManager; moved from src/managers/correlations_panel_manager.py without changes.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton, QSpinBox, QListWidget, QListWidgetItem, QCheckBox, QProgressBar
from PyQt5.QtGui import QFont, QColor


class CorrelationsPanelUiMixin:
    """Building the correlations panel, its theme and the results list."""

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
