"""
Per-graph and global graph settings (normalization, grid, autoscale, axes, mouse).

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging

logger = logging.getLogger(__name__)


class GraphSettingsMixin:
    """Per-graph and global graph settings (normalization, grid, autoscale, axes, mouse)."""

    def _on_per_graph_normalization_toggled(self, graph_index: int, normalize: bool):
        """Handle normalization for a specific graph."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return

        signals_in_graph = self.graph_signal_mapping.get(active_tab_index, {}).get(graph_index, [])
        
        if not signals_in_graph:
            return

        if normalize:
            self.signal_processor.apply_normalization(signal_names=signals_in_graph)
        else:
            self.signal_processor.remove_normalization(signal_names=signals_in_graph)
        
        # Save normalization setting for this graph
        self._save_graph_setting(graph_index, 'normalize', normalize)
        
        # Redraw all signals to reflect the change
        self._redraw_all_signals()
        logger.info(f"Normalization toggled to {normalize} for signals in graph {graph_index}: {signals_in_graph}")

    def _on_per_graph_view_reset(self, graph_index: int):
        """Handle view reset for a specific graph to show all data including limit lines."""
        active_container = self.get_active_graph_container()
        if active_container:
            # Use PlotManager's reset_view() which properly handles downsampled data
            # by using stored original data ranges instead of autoRange()
            active_container.plot_manager.reset_view()
            logger.info(f"View reset for graph {graph_index} - using PlotManager.reset_view() with original data ranges")

    def _on_per_graph_grid_changed(self, graph_index: int, show_grid: bool):
        """Handle grid visibility for a specific graph."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            if 0 <= graph_index < len(plot_widgets):
                plot_widgets[graph_index].showGrid(x=show_grid, y=show_grid)
                
                # Save grid setting for this graph
                self._save_graph_setting(graph_index, 'show_grid', show_grid)
                
                logger.info(f"Grid visibility for graph {graph_index} set to {show_grid}")

    def _on_per_graph_autoscale_changed(self, graph_index: int, autoscale: bool):
        """Handle Y-axis autoscale for a specific graph."""
        active_container = self.get_active_graph_container()
        if active_container:
            plot_widgets = active_container.get_plot_widgets()
            if 0 <= graph_index < len(plot_widgets):
                plot_widgets[graph_index].enableAutoRange(axis='y', enable=autoscale)
                
                # Save autoscale setting for this graph
                self._save_graph_setting(graph_index, 'autoscale', autoscale)
                
                logger.info(f"Autoscale for graph {graph_index} set to {autoscale}")

    # Graph Settings panel: every setting applies to all graphs of all tabs
    def _on_global_normalization_toggled(self, normalize: bool):
        """Show all signals normalized (display only) or with real values."""
        if normalize:
            self.signal_processor.apply_normalization()
        else:
            self.signal_processor.remove_normalization()
        self._redraw_all_signals()
        logger.info(f"Global normalization {'enabled' if normalize else 'disabled'} for all graphs")

    def _on_global_view_reset(self):
        """Show all data of the active tab's graphs."""
        active_container = self.get_active_graph_container()
        if active_container:
            active_container.plot_manager.reset_view()
        logger.info("Global view reset applied to all graphs")

    def _on_global_grid_changed(self, show_grid: bool):
        """Handle global grid visibility for all graphs."""
        for container in self.graph_containers:
            container.plot_manager.set_grid_visibility(show_grid)
        logger.info(f"Global grid {'shown' if show_grid else 'hidden'} for all graphs")

    def _on_global_autoscale_changed(self, autoscale: bool):
        """Handle global Y-axis autoscale for all graphs."""
        for container in self.graph_containers:
            container.plot_manager.set_y_autoscale(autoscale)
        logger.info(f"Global autoscale {'enabled' if autoscale else 'disabled'} for all graphs")

    def _on_global_legend_visibility_changed(self, visible: bool):
        """Handle global legend visibility for all graphs."""
        for container in self.graph_containers:
            container.plot_manager.set_legend_visibility(visible)
        logger.info(f"Global legend visibility set to {visible} for all graphs")

    def _on_global_tooltips_changed(self, enabled: bool):
        """Handle global tooltips toggle for all graphs."""
        for container in self.graph_containers:
            container.plot_manager.set_tooltips_enabled(enabled)
        logger.info(f"Global tooltips {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_snap_changed(self, enabled: bool):
        """Handle global snap to data points for all graphs."""
        if self.cursor_manager:
            self.cursor_manager.set_snap_to_data(enabled)
        for container in self.graph_containers:
            container.plot_manager.snap_to_data_enabled = enabled
        logger.info(f"Global snap to data {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_line_width_changed(self, width: int):
        """Handle global line width change for all graphs."""
        for container in self.graph_containers:
            container.plot_manager.set_line_width(width)
        logger.info(f"Global line width set to {width} for all graphs")

    def _on_global_x_mouse_changed(self, enabled: bool):
        """Handle global X axis mouse interaction for all graphs."""
        for container in self.graph_containers:
            for plot_widget in container.get_plot_widgets():
                plot_widget.setMouseEnabled(x=enabled, y=plot_widget.getViewBox().state['mouseEnabled'][1])
        logger.info(f"Global X axis mouse {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_y_mouse_changed(self, enabled: bool):
        """Handle global Y axis mouse interaction for all graphs."""
        for container in self.graph_containers:
            for plot_widget in container.get_plot_widgets():
                plot_widget.setMouseEnabled(x=plot_widget.getViewBox().state['mouseEnabled'][0], y=enabled)
        logger.info(f"Global Y axis mouse {'enabled' if enabled else 'disabled'} for all graphs")

    def _on_global_secondary_axis_changed(self, enabled: bool):
        """Handle global secondary axis toggle."""
        for container in self.graph_containers:
            container.plot_manager.set_secondary_axis_enabled(enabled)
        # Redraw all signals to apply axis assignment
        self._redraw_all_signals()
        logger.info(f"Global secondary axis {'enabled' if enabled else 'disabled'}")

    def _save_graph_setting(self, graph_index: int, setting_name: str, value):
        """Save a setting for a specific graph in the active tab."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return
            
        # Initialize tab settings if not exists
        if active_tab_index not in self.graph_settings:
            self.graph_settings[active_tab_index] = {}
            
        # Initialize graph settings if not exists
        if graph_index not in self.graph_settings[active_tab_index]:
            self.graph_settings[active_tab_index][graph_index] = {}
            
        # Save the setting
        self.graph_settings[active_tab_index][graph_index][setting_name] = value
        logger.debug(f"Saved setting: Tab {active_tab_index}, Graph {graph_index}, {setting_name} = {value}")

    def _get_graph_setting(self, graph_index: int, setting_name: str, default_value=None):
        """Get a setting for a specific graph in the active tab."""
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return default_value
            
        return (self.graph_settings
                .get(active_tab_index, {})
                .get(graph_index, {})
                .get(setting_name, default_value))

    def _apply_saved_graph_settings(self):
        """Apply saved settings to all graphs in the active tab."""
        active_container = self.get_active_graph_container()
        if not active_container:
            return
            
        active_tab_index = self.tab_widget.currentIndex()
        if active_tab_index < 0:
            return
            
        plot_widgets = active_container.get_plot_widgets()
        tab_settings = self.graph_settings.get(active_tab_index, {})
        
        for graph_index, graph_settings in tab_settings.items():
            if 0 <= graph_index < len(plot_widgets):
                plot_widget = plot_widgets[graph_index]
                
                # Apply grid setting
                show_grid = graph_settings.get('show_grid', True)  # Default to True
                plot_widget.showGrid(x=show_grid, y=show_grid)
                
                # Apply autoscale setting
                autoscale = graph_settings.get('autoscale', True)  # Default to True
                plot_widget.enableAutoRange(axis='y', enable=autoscale)
                
                logger.debug(f"Applied settings to Graph {graph_index}: grid={show_grid}, autoscale={autoscale}")
        
        # Update graph settings panel to reflect current settings
        self._sync_graph_settings_panel()

    def _sync_graph_settings_panel(self):
        """Synchronize the graph settings panel checkboxes with actual settings."""
        # Global settings don't need per-graph synchronization
        # The global settings are maintained internally by the panel manager
        logger.debug("Graph settings panel sync skipped - using global settings")
