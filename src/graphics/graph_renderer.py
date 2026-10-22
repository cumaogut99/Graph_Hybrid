"""
Graph Renderer - Concatenated range filter rendering

Range filters always use concatenated display: the time ranges that match the
filter conditions are joined into one continuous timeline, and the signal
processor's data is replaced with it, so every graph in every tab shows the
filtered data.
"""

import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GraphRenderer:
    """Applies the concatenated range filter to the signal data."""
    
    def __init__(self, signal_processor, graph_signal_mapping, parent_widget=None):
        self.signal_processor = signal_processor
        self.graph_signal_mapping = graph_signal_mapping
        self.parent_widget = parent_widget  # Reference to TimeGraphWidget
        self._is_destroyed = False
    
    def cleanup(self):
        """Release resources (kept for the widget's shutdown sequence)."""
        if self._is_destroyed:
            return
        self._is_destroyed = True
        logger.info("GraphRenderer cleanup completed")
    
    def apply_concatenated_filter(self, container, time_segments: List[Tuple[float, float]], filter_conditions: list = None):
        """Apply concatenated display filter - create continuous timeline from filtered segments.

        Args:
            container: GraphContainer (unused)
            time_segments: List of (start_time, end_time) tuples, already filtered
            filter_conditions: DEPRECATED - no longer used

        The plotted signals are filtered now; the others when first used
        (SignalProcessor.apply_segment_filter), so files with 1000+ channels
        don't filter every channel up front.
        """
        logger.info(f"[CONCATENATED] Time segments: {len(time_segments)} segments")
        plotted = {
            name
            for tab_mapping in (self.graph_signal_mapping or {}).values()
            for signal_names in tab_mapping.values()
            for name in signal_names
        }
        self.signal_processor.apply_segment_filter(time_segments, list(plotted))
        # Redraw is done by TimeGraphWidget._redraw_all_signals()
