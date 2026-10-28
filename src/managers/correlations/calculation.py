"""
Correlation of the target with every other signal over the cursor range.

Part of CorrelationsPanelManager; moved from src/managers/correlations_panel_manager.py without changes.
"""

import logging
import numpy as np
from typing import Tuple, Optional
from PyQt5.QtCore import QTimer

logger = logging.getLogger(__name__)


class CorrelationsCalculationMixin:
    """Correlation of the target with every other signal over the cursor range."""

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
