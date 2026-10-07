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
            container: GraphContainer
            time_segments: List of (start_time, end_time) tuples - ALREADY filtered by C++ calculate_streaming
            filter_conditions: DEPRECATED - no longer used. C++ already computed correct segments.
        
        Note: The time_segments are already correctly calculated by C++ FilterEngine.calculate_streaming.
              We just need to load data within these time ranges and concatenate them.
        """
        logger.info(f"[CONCATENATED] Starting concatenated filter application")
        logger.info(f"[CONCATENATED] Time segments: {len(time_segments)} segments")

        # Get all signals data
        all_signals = self.signal_processor.get_all_signals()

        # Get MPAI reader and time column
        raw_df = getattr(self.signal_processor, "raw_dataframe", None)
        time_col = getattr(self.signal_processor, "time_column_name", None)
        mpai_reader = raw_df if raw_df and hasattr(raw_df, "load_column_slice") else None

        # Create concatenated time and value arrays with continuous timeline
        concatenated_data = {}

        for signal_name, signal_data in all_signals.items():
            logger.info(f"[CONCATENATED] Processing signal: {signal_name}")

            concat_x = []
            concat_y = []
            current_time_offset = 0.0

            metadata = signal_data.get("metadata", {})
            full_count = metadata.get("full_count", len(signal_data.get('x_data', [])))

            # Sample interval, used to space consecutive segments on the joined timeline
            full_time_range = metadata.get("full_time_range")
            if full_time_range and len(full_time_range) == 2 and full_count > 1:
                sample_dt = (full_time_range[1] - full_time_range[0]) / (full_count - 1)
            else:
                x_all = np.asarray(signal_data.get('x_data', []), dtype=np.float64)
                sample_dt = float(np.median(np.diff(x_all))) if len(x_all) > 1 else 0.0

            for i, (segment_start, segment_end) in enumerate(time_segments):
                segment_x = None
                segment_y = None

                # ✅ SIMPLIFIED: Load segment by time range only
                # time_segments are already correctly filtered by C++ calculate_streaming
                if mpai_reader and time_col and metadata.get("mpai"):
                    try:
                        sample_rate = 1.0
                        full_time_range = metadata.get("full_time_range")
                        if full_time_range and len(full_time_range) == 2:
                            start_time_meta, end_time_meta = full_time_range
                        else:
                            start_time_meta, end_time_meta = 0.0, 1.0

                        duration = max(end_time_meta - start_time_meta, 1e-9)
                        if full_count > 1:
                            sample_rate = (full_count - 1) / duration

                        # Segment bounds are the times of the first and last
                        # matching samples: round (not truncate) and include both
                        start_row = max(0, int(round((segment_start - start_time_meta) * sample_rate)))
                        end_row = min(full_count, int(round((segment_end - start_time_meta) * sample_rate)) + 1)
                        row_count = max(1, end_row - start_row)

                        segment_x = np.array(mpai_reader.load_column_slice(time_col, int(start_row), int(row_count)), dtype=np.float64)
                        segment_y = np.array(mpai_reader.load_column_slice(signal_name, int(start_row), int(row_count)), dtype=np.float64)
                        logger.info(f"[CONCATENATED] MPAI segment {i+1}: {len(segment_x)} points [{segment_start:.2f}, {segment_end:.2f}]")
                    except Exception as e:
                        logger.warning(f"[CONCATENATED] MPAI loading failed: {e}")
                        segment_x = None

                # Fallback: Use preview data
                if segment_x is None:
                    full_x_data = np.array(signal_data['x_data'])
                    full_y_data = np.array(signal_data['y_data'])
                    mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                    segment_x = full_x_data[mask]
                    segment_y = full_y_data[mask]
                    logger.warning(f"[CONCATENATED] Using preview data fallback: {len(segment_x)} points (may be downsampled)")

                if len(segment_x) > 0:
                    # Create continuous timeline by adjusting time values
                    if i == 0:
                        # First segment starts at 0
                        adjusted_x = segment_x - segment_x[0]
                        current_time_offset = adjusted_x[-1] if len(adjusted_x) > 0 else 0
                    else:
                        # Subsequent segments continue one sample interval after
                        # the previous one ended (no duplicate time at the joint)
                        adjusted_x = (segment_x - segment_x[0]) + current_time_offset + sample_dt
                        current_time_offset = adjusted_x[-1]

                    concat_x.extend(adjusted_x)
                    concat_y.extend(segment_y)

            if concat_x:
                concatenated_data[signal_name] = {
                    'time': np.array(concat_x),
                    'values': np.array(concat_y)
                }
                logger.info(f"[CONCATENATED FIX] Signal '{signal_name}': {len(concat_x)} total points, "
                           f"time range: {concat_x[0]:.3f} - {concat_x[-1]:.3f}")

        # Update signal processor with concatenated data
        self.signal_processor.set_filtered_data(concatenated_data)
        logger.info(f"[CONCATENATED FIX] Updated signal processor with concatenated data")

        # NOT: Grafik redraw'ı TimeGraphWidget._redraw_all_signals() tarafından yapılacak
        # container.plot_manager.redraw_all_plots() yeterli değil - sadece repaint yapıyor

        logger.info(f"[CONCATENATED FIX] Concatenated filter applied successfully - continuous timeline created")
