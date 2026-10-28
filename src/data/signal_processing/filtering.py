"""
Range filters: filtered data, time segments and restoring the original data.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalFilteringMixin:
    """Range filters: filtered data, time segments and restoring the original data."""

    def apply_polars_filter(self, conditions: List[Dict]) -> Optional[pl.DataFrame]:
        """
        PERFORMANCE: Polars native filtering - NumPy'dan 5-10x daha hızlı!
        
        Args:
            conditions: List of filter conditions
                [{
                    'parameter': 'Temperature',
                    'ranges': [
                        {'type': 'lower', 'operator': '>=', 'value': 20.0},
                        {'type': 'upper', 'operator': '<=', 'value': 80.0}
                    ]
                }]
        
        Returns:
            Filtered Polars DataFrame (or None if no raw data)
        """
        if self.raw_dataframe is None:
            logger.warning("No raw dataframe available for Polars filtering")
            return None
        
        if not conditions:
            return self.raw_dataframe
        
        logger.info(f"Applying Polars native filter with {len(conditions)} conditions")
        
        # Start with full dataframe
        filtered_df = self.raw_dataframe
        
        # Apply each condition (AND logic between parameters)
        for condition in conditions:
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            if param_name not in filtered_df.columns:
                logger.warning(f"Parameter '{param_name}' not in dataframe")
                continue
            
            # Build OR expression for ranges within same parameter
            range_expr = None
            for range_filter in ranges:
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                # Create Polars expression
                if range_type == 'lower':
                    if operator == '>=':
                        expr = pl.col(param_name) >= value
                    elif operator == '>':
                        expr = pl.col(param_name) > value
                    else:
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        expr = pl.col(param_name) <= value
                    elif operator == '<':
                        expr = pl.col(param_name) < value
                    else:
                        continue
                else:
                    continue
                
                # Combine with OR
                if range_expr is None:
                    range_expr = expr
                else:
                    range_expr = range_expr | expr
            
            # Apply combined range expression (AND with previous conditions)
            if range_expr is not None:
                filtered_df = filtered_df.filter(range_expr)
        
        logger.info(f"Polars filter: {self.raw_dataframe.height} → {filtered_df.height} rows")
        return filtered_df

    def set_filtered_data(self, filtered_data: Dict[str, Dict]):
        """
        Set filtered data for concatenated display mode.
        
        Args:
            filtered_data: Dict of signal_name -> {'time': x_data, 'values': y_data}
        """
        with QMutexLocker(self.mutex):
            for signal_name, data in filtered_data.items():
                if signal_name in self.signal_data:
                    # Ensure data is numpy arrays
                    x_data = np.asarray(data['time'], dtype=np.float64)
                    y_data = np.asarray(data['values'], dtype=np.float64)
                    
                    # Update the signal data with filtered values
                    self.signal_data[signal_name]['x_data'] = x_data
                    self.signal_data[signal_name]['y_data'] = y_data
                    # CRITICAL: Update original_y to match new data size
                    self.signal_data[signal_name]['original_y'] = y_data.copy()
                    # Serve this in-memory data instead of the MPAI file until
                    # restore_original_data() (see _is_file_backed)
                    self.signal_data[signal_name]['filtered'] = True
                    self.signal_data[signal_name].pop('_norm_params', None)

                    # Clear related caches since data changed
                    self._clear_cache(signal_name)
                    self._downsampled_cache.pop(signal_name, None)
                    
                    logger.debug(f"Updated filtered data for signal '{signal_name}' with {len(y_data)} points")
                else:
                    logger.warning(f"Signal '{signal_name}' not found in signal_data, skipping filtered data update")

    def apply_segment_filter(self, time_segments: List[Tuple[float, float]],
                             names: Optional[List[str]] = None):
        """
        Apply a range filter: only the time segments are shown, joined into
        one continuous timeline.

        Only the given signals (the plotted ones) are filtered now; every
        other signal is filtered when it is first used. Filtering all
        channels at once took seconds for files with 1000+ channels.
        """
        with QMutexLocker(self.mutex):
            self._filter_segments = list(time_segments)
            self._filter_pending = set(self.signal_data)
            self._ensure_filtered(names or [])

    def _ensure_filtered(self, names) -> None:
        """Apply the active range filter to these signals if not done yet."""
        if not self._filter_segments or not self._filter_pending:
            return
        with QMutexLocker(self.mutex):
            if names is None:
                names = list(self._filter_pending)
            elif isinstance(names, str):
                names = [names]
            filtered = {}
            for name in names:
                if name not in self._filter_pending:
                    continue
                self._filter_pending.discard(name)
                data = self._concatenate_segments(name, self._filter_segments)
                if data is not None:
                    filtered[name] = data
            if filtered:
                self.set_filtered_data(filtered)

    def _concatenate_segments(self, name: str,
                              time_segments: List[Tuple[float, float]]) -> Optional[Dict]:
        """
        Join the signal's data inside the time segments into one continuous
        timeline: {'time': x, 'values': y}, or None if no data is inside.
        """
        info = self.signal_data.get(name)
        if not info:
            return None
        metadata = info.get('metadata', {})
        # The signal's own reader: a calculated parameter has its own MPAI
        reader = info.get('mpai_reader') if self._is_file_backed(info) else None
        time_col = info.get('time_column')

        if reader is not None and time_col:
            full_count = metadata.get('full_count') or info.get('row_count') or 0
            x_all = y_all = None
        else:
            x_all = np.asarray(info.get('x_data', []), dtype=np.float64)
            y_all = np.asarray(info.get('y_data', []), dtype=np.float64)
            full_count = len(x_all)

        # Sample interval, used to space consecutive segments on the joined timeline
        full_time_range = metadata.get('full_time_range')
        if full_time_range and len(full_time_range) == 2 and full_count > 1:
            start_time_meta = full_time_range[0]
            sample_dt = (full_time_range[1] - full_time_range[0]) / (full_count - 1)
        else:
            start_time_meta = 0.0
            sample_dt = float(np.median(np.diff(x_all))) if x_all is not None and len(x_all) > 1 else 0.0

        parts_x, parts_y = [], []
        current_time_offset = 0.0
        for segment_start, segment_end in time_segments:
            if reader is not None and time_col:
                if sample_dt <= 0:
                    return None
                # Segment bounds are the times of the first and last
                # matching samples: round (not truncate) and include both
                start_row = max(0, int(round((segment_start - start_time_meta) / sample_dt)))
                end_row = min(full_count, int(round((segment_end - start_time_meta) / sample_dt)) + 1)
                row_count = max(1, end_row - start_row)
                segment_x = np.asarray(reader.load_column_slice(time_col, start_row, row_count), dtype=np.float64)
                segment_y = np.asarray(reader.load_column_slice(info.get('column_name', name), start_row, row_count), dtype=np.float64)
            else:
                mask = (x_all >= segment_start) & (x_all <= segment_end)
                segment_x = x_all[mask]
                segment_y = y_all[mask]

            n = min(len(segment_x), len(segment_y))
            if n == 0:
                continue
            segment_x = segment_x[:n]
            if not parts_x:
                # First segment starts at 0
                adjusted_x = segment_x - segment_x[0]
            else:
                # Subsequent segments continue one sample interval after the
                # previous one ended (no duplicate time at the joint)
                adjusted_x = (segment_x - segment_x[0]) + current_time_offset + sample_dt
            current_time_offset = adjusted_x[-1]
            parts_x.append(adjusted_x)
            parts_y.append(segment_y[:n])

        if not parts_x:
            return None
        return {'time': np.concatenate(parts_x), 'values': np.concatenate(parts_y)}

    def restore_original_data(self):
        """
        Restore all signals to their original unfiltered state.
        Used when clearing filters in concatenated display mode.
        """
        with QMutexLocker(self.mutex):
            for signal_name, original_data in self.original_signal_data.items():
                if signal_name in self.signal_data:
                    # Restore original data
                    self.signal_data[signal_name]['x_data'] = original_data['x_data'].copy()
                    self.signal_data[signal_name]['y_data'] = original_data['y_data'].copy()
                    self.signal_data[signal_name]['original_y'] = original_data['y_data'].copy()
                    
                    # Clear related caches since data changed
                    self._clear_cache(signal_name)

                    logger.debug(f"Restored original data for signal '{signal_name}' with {len(original_data['y_data'])} points")

            self._filter_segments = None
            self._filter_pending.clear()

            # File-backed signals read from the MPAI file again
            for info in self.signal_data.values():
                info.pop('filtered', None)
                info.pop('_norm_params', None)
            self._downsampled_cache.clear()
            self._cursor_value_cache.clear()
            
            logger.info("Restored original data for all signals")
