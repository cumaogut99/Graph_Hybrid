"""
Range filters applied from the filter panel.

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from src.ui.graph_advanced_settings_dialog import GraphAdvancedSettingsDialog

logger = logging.getLogger(__name__)


class FiltersMixin:
    """Range filters applied from the filter panel."""

    def clear_active_filters(self):
        """Clear the range filter and show the original data again."""
        self._apply_range_filter({'conditions': []})

    def _on_filters_requested(self):
        """Open the Advanced Settings (range filter) dialog; applies to all graphs."""
        all_signals = self.signal_processor.get_signal_names()
        if not all_signals:
            QMessageBox.information(self, "Range Filters", "Load a data file first.")
            return

        dialog = GraphAdvancedSettingsDialog(
            all_signals, self.filter_manager.get_global_filter(), None  # None: own taskbar entry
        )
        dialog.setWindowIcon(self.windowIcon() if self.windowIcon() else QIcon())

        # Center dialog on parent window
        if self.parent():
            parent_geometry = self.parent().geometry()
            dialog.move(
                parent_geometry.center().x() - dialog.width() // 2,
                parent_geometry.center().y() - dialog.height() // 2
            )

        dialog.range_filter_applied.connect(self._apply_range_filter)
        dialog.exec_()

    def _apply_range_filter(self, filter_data: dict):
        """
        Apply the range filter to all graphs, or clear it (empty conditions).

        The filter always uses concatenated display: the time ranges where all
        conditions hold are joined into one continuous timeline, the signal
        processor's data is replaced with it and every graph in every tab is
        redrawn. Segments are computed from the original file data, so a new
        filter replaces the previous one instead of filtering its result.
        """
        conditions = filter_data.get('conditions', [])
        logger.info(f"[FILTER] Range filter requested: {len(conditions)} condition(s)")

        if not conditions:
            if self.filter_manager.has_active_filters():
                self.filter_manager.clear_filters()
                self.signal_processor.restore_original_data()
                self._save_filter_state_for_file_switch()
                self._redraw_after_filter_change()
                logger.info("[FILTER] Range filter cleared, original data restored")
            return

        if hasattr(self, 'loading_manager') and self.loading_manager:
            self.loading_manager.start_operation("filtering", "Calculating filter segments...")

        def on_segments_calculated(time_segments):
            try:
                if hasattr(self, 'loading_manager') and self.loading_manager:
                    self.loading_manager.finish_operation("filtering")

                if time_segments is None:
                    return  # debounced

                if not time_segments:
                    # Keep whatever was shown before (previous filter or none)
                    msg = QMessageBox(self)
                    msg.setStyleSheet(self._get_message_box_style())
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("No time segments match the specified filter conditions.\n"
                                "The filter was not applied.")
                    msg.setWindowTitle("No Matches")
                    msg.exec_()
                    return

                # Build the concatenated data from the original data
                if self.filter_manager.has_active_filters():
                    self.signal_processor.restore_original_data()
                self.graph_renderer.graph_signal_mapping = self.graph_signal_mapping
                self.graph_renderer.apply_concatenated_filter(None, time_segments)

                self.filter_manager.set_global_filter(filter_data)
                self._save_filter_state_for_file_switch()
                self._redraw_after_filter_change()
                logger.info(f"[FILTER] Range filter applied to all graphs ({len(time_segments)} segments)")

            except RuntimeError as e:
                logger.warning(f"Filter callback failed - widget may be deleted: {e}")
            except Exception as e:
                logger.error(f"Error applying range filter: {e}", exc_info=True)
                msg = QMessageBox(self)
                msg.setStyleSheet(self._get_message_box_style())
                msg.setIcon(QMessageBox.Critical)
                msg.setText(f"Error applying filter: {str(e)}")
                msg.setWindowTitle("Filter Error")
                msg.exec_()

        try:
            # Only the filtered parameters are needed (all share one time base)
            self.filter_manager.calculate_filter_segments_threaded(
                self.signal_processor.get_all_signals([c['parameter'] for c in conditions]),
                conditions,
                on_segments_calculated,
            )
        except Exception as e:
            logger.error(f"Error starting range filter calculation: {e}", exc_info=True)
            if hasattr(self, 'loading_manager') and self.loading_manager:
                self.loading_manager.finish_operation("filtering")

    def _save_filter_state_for_file_switch(self):
        """Keep the widget container's copy of the filter state up to date."""
        if hasattr(self, 'parent') and hasattr(self.parent, 'widget_container_manager'):
            self.parent.widget_container_manager.save_current_filter_state()

    def _redraw_after_filter_change(self):
        """Redraw every tab after the filter replaced/restored the time axis."""
        # Markers are times on the previous timeline: they would point at the
        # wrong data on the new one
        for container in self.graph_containers:
            container.plot_manager.clear_markers()
        self._redraw_all_signals()
        # The timeline changed (concatenated or original), so the previous
        # zoom is meaningless: fit every plot to the new data
        for container in self.graph_containers:
            for plot_widget in container.plot_manager.get_plot_widgets():
                plot_widget.enableAutoRange(axis='x', enable=True)
                plot_widget.enableAutoRange(axis='y', enable=True)
                plot_widget.autoRange()
                plot_widget.enableAutoRange(axis='x', enable=False)
                plot_widget.enableAutoRange(axis='y', enable=False)

    def _restore_filter_ui_state(self, saved_filters: dict):
        """
        Called by the widget container manager after restoring filter state.
        Nothing to rebuild: the filter dialog reads the state when opened.
        """
        if hasattr(self, 'toolbar_manager') and self.toolbar_manager:
            self.toolbar_manager.set_filter_active(self.filter_manager.has_active_filters())
        if hasattr(self, 'bitmask_panel_manager') and self.bitmask_panel_manager:
            self.bitmask_panel_manager.set_filter_active(self.filter_manager.has_active_filters())
