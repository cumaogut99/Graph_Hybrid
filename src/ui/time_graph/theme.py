"""
Theme and signal colors.

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging

logger = logging.getLogger(__name__)


class ThemeMixin:
    """Theme and signal colors."""

    def _on_theme_changed(self, theme_name: str):
        """Handle theme changes broadcast from the theme manager."""
        self._apply_theme()

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

    def _get_signal_color(self, signal_name: str) -> str:
        """Get color for a signal (simplified version)."""
        # Simple color cycling - in real implementation, use proper color management
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
        hash_value = hash(signal_name) % len(colors)
        return colors[hash_value]

    # Public API methods
    def set_theme(self, theme_name: str):
        """Set the visual theme."""
        self.theme_manager.set_theme(theme_name)

    def _apply_theme(self):
        """Apply the current theme to all components."""
        theme_name = self.theme_manager.get_current_theme()
        
        # Apply to main widget background and panels
        self.setStyleSheet(f"background-color: {self.theme_manager.get_color('background')};")
        
        # Update toolbar with theme colors
        if hasattr(self.toolbar_manager, 'update_theme'):
            self.toolbar_manager.update_theme()
        else:
            self.toolbar_manager.get_toolbar().setStyleSheet(
                self.theme_manager.get_widget_stylesheet('toolbar', theme_name)
            )
        
        # Update settings panel with theme colors
        if hasattr(self.settings_panel_manager, 'update_theme'):
            self.settings_panel_manager.update_theme()
        else:
            self.settings_panel_manager.get_settings_panel().setStyleSheet(
                self.theme_manager.get_widget_stylesheet('panel', theme_name)
            )
        self.statistics_settings_panel_manager.get_settings_panel().setStyleSheet(
            self.theme_manager.get_widget_stylesheet('panel', theme_name)
        )
        self.graph_settings_panel_manager.get_settings_panel().setStyleSheet(
            self.theme_manager.get_widget_stylesheet('panel', theme_name)
        )
        if hasattr(self, 'channel_stats_panel'):
            self.channel_stats_panel.setStyleSheet(
                self.theme_manager.get_widget_stylesheet('panel', theme_name)
            )
        
        # Update statistics panel with theme colors
        if hasattr(self, 'statistics_panel') and hasattr(self.statistics_panel, 'update_theme'):
            self.statistics_panel.update_theme(self.theme_manager.get_theme_colors())
        
        # Apply to all graph containers
        for container in self.graph_containers:
            container.apply_theme()
        
        # Update correlations panel with theme colors
        if hasattr(self, 'correlations_panel_manager') and hasattr(self.correlations_panel_manager, 'update_theme'):
            self.correlations_panel_manager.update_theme()
        
        # Update graph settings panel with theme colors
        if hasattr(self, 'graph_settings_panel_manager') and hasattr(self.graph_settings_panel_manager, 'update_theme'):
            self.graph_settings_panel_manager.update_theme()
        
        # Redraw signals with new theme colors
        self._redraw_all_signals()

    def _on_signal_color_changed(self, signal_name: str, new_color: str):
        """Handle color changes for a specific signal from the statistics panel."""
        logger.info(f"Color change requested for signal '{signal_name}' to {new_color}")
        
        all_signal_names = sorted(self.signal_processor.get_signal_names())
        
        if signal_name in all_signal_names:
            signal_index = all_signal_names.index(signal_name)
            logger.info(f"Found signal '{signal_name}' at index {signal_index}")
            
            # Update theme manager with the color override
            self.theme_manager.set_signal_color_override(signal_index, new_color)
            
            # Update legend color if legend manager exists
            if hasattr(self, 'legend_manager') and self.legend_manager:
                self.legend_manager.set_signal_color(signal_name, new_color)
            
            # Redraw all signals to apply the new color
            self._redraw_all_signals()
            logger.info(f"Successfully updated color for signal '{signal_name}'")
        else:
            logger.warning(f"Could not find signal '{signal_name}' to change its color.")
            logger.warning(f"Available signals: {all_signal_names}")
