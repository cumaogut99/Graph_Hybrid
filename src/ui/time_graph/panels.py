"""
Side panels: creating them and switching between them.

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging
from src.ui.statistics_panel import StatisticsPanel

logger = logging.getLogger(__name__)


class PanelsMixin:
    """Side panels: creating them and switching between them."""

    def _create_channel_statistics_panel(self):
        """Creates the main container widget for the statistics panel."""
        # Use our modern StatisticsPanel class
        self.statistics_panel = StatisticsPanel(self)
        self.statistics_panel.setMinimumWidth(300)
        
        # Connect graph settings signal
        self.statistics_panel.signal_color_changed.connect(self._on_signal_color_changed)
        self.statistics_panel.signal_remove_requested.connect(self._on_signal_remove_requested)
        self.statistics_panel.graph_reorder_requested.connect(self._on_graph_reorder_requested)
        
        # Connect theme change signal
        self.theme_manager.theme_changed.connect(lambda: self.statistics_panel.update_theme(self.theme_manager.get_theme_colors()))
        
        return self.statistics_panel

    def _create_correlations_panel(self):
        """Create the correlations analysis panel using the dedicated manager."""
        from src.managers.correlations_panel_manager import CorrelationsPanelManager
        self.correlations_panel_manager = CorrelationsPanelManager(self)
        return self.correlations_panel_manager.get_panel()

    def _create_bitmask_panel(self):
        """Create the bitmask analysis panel."""
        return self.bitmask_panel_manager.get_widget()

    def _on_panel_toggled(self):
        """Handle statistics panel visibility toggle."""
        if hasattr(self, 'channel_stats_panel'):
            self.channel_stats_panel.setVisible(not self.channel_stats_panel.isVisible())

    def _on_settings_toggled(self):
        """Handle settings panel visibility toggle."""
        self._toggle_left_panel(self.settings_panel)
        #logger.info(f"Settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.settings_panel}")

    def _on_graph_settings_toggled(self):
        """Handle graph settings panel visibility toggle."""
        self._toggle_left_panel(self.graph_settings_panel)
        #logger.info(f"Graph settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.graph_settings_panel}")

    def _on_parameters_toggled(self):
        """Handle parameters panel visibility toggle."""
        self._toggle_left_panel(self.parameters_panel)

    def _on_statistics_settings_toggled(self):
        """Handle statistics settings panel visibility toggle."""
        self._toggle_left_panel(self.statistics_settings_panel)
            
        #logger.info(f"Statistics settings panel visibility: {self.left_panel_stack.isVisible() and self.left_panel_stack.currentWidget() == self.statistics_settings_panel}")

    def _on_correlations_toggled(self):
        """Handle correlations panel visibility toggle."""
        self._toggle_left_panel(self.correlations_panel)

    def _on_bitmask_toggled(self):
        """Handle bitmask panel visibility toggle."""
        self._toggle_left_panel(self.bitmask_panel)

    def _toggle_left_panel(self, panel_to_show):
        """Generic function to toggle visibility of a panel in the left stack."""
        if self.left_panel_stack.currentWidget() == panel_to_show and self.left_panel_stack.isVisible():
            self.left_panel_stack.setVisible(False)
        else:
            self.left_panel_stack.setCurrentWidget(panel_to_show)
            self.left_panel_stack.setVisible(True)
