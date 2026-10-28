"""
Values at a time point and raw data of a time range.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalQueriesMixin:
    """Values at a time point and raw data of a time range."""

    def get_signals_at_time(self, signal_names: List[str], timestamp: float) -> Dict[str, float]:
        """
        Get values for multiple signals at a specific time (Batch Optimized).
        
        Optimizations:
        - Batches memory-mapped signals by reader to reduce mutex contention.
        - Uses C++ get_batch_values for O(1) retrieval.
        - Single lock acquisition per batch.
        
        Args:
            signal_names: List of signal names
            timestamp: Time to query
            
        Returns:
            Dictionary mapping signal names to values
        """
        results = {}
        
        # Group signals by source properties
        mpai_groups = {} # reader -> [signal_names]
        other_signals = []
        
        with QMutexLocker(self.mutex):
            self._ensure_filtered(signal_names)
            for name in signal_names:
                if name not in self.signal_data:
                    continue
                    
                info = self.signal_data[name]
                metadata = info.get('metadata', {})
                
                # Check if memory mapped MPAI
                if self._is_file_backed(info) and 'mpai_reader' in info:
                    reader = info['mpai_reader']
                    if reader not in mpai_groups:
                        mpai_groups[reader] = []
                    mpai_groups[reader].append(name)
                else:
                    other_signals.append(name)
            
            # 1. Handle Memory-Mapped Batches (C++)
            for reader, batch_names in mpai_groups.items():
                try:
                    # Get actual column names (might differ from signal names)
                    col_names = []
                    valid_batch_names = []
                    
                    for name in batch_names:
                        info = self.signal_data[name]
                        col_name = info.get('column_name', name)
                        col_names.append(col_name)
                        valid_batch_names.append(name)
                        
                    if not valid_batch_names:
                        continue

                    # Get time column from first signal (assume same for same reader)
                    first_info = self.signal_data[valid_batch_names[0]]
                    time_col = first_info.get('time_column', 'time')
                    
                    # Call C++ batch method
                    # Ensure reader has the method (it was recently added)
                    if hasattr(reader, 'get_batch_values'):
                        batch_results = reader.get_batch_values(timestamp, time_col, col_names)
                        
                        # Map back to signal names
                        for i, name in enumerate(valid_batch_names):
                            col = col_names[i]
                            if col in batch_results:
                                results[name] = batch_results[col]
                    elif hasattr(reader, 'get_cursor_value') and hasattr(reader, 'name_to_id'):
                        # Python MpaiDirectoryReader optimization
                        # It doesn't have batch iterator yet, but get_cursor_value is O(1)
                        # We can loop here safely
                        for i, name in enumerate(valid_batch_names):
                            col = col_names[i]
                            if col in reader.name_to_id:
                                ch_id = reader.name_to_id[col]
                                val = reader.get_cursor_value(ch_id, timestamp)
                                results[name] = val
                    else:
                        # Fallback if C++ method missing (old pyd)
                        # We must release lock to call get_signal_at_time? 
                        # No, get_signal_at_time uses QMutexLocker which is recursive.
                        # But it's better to avoid recursion overhead if possible.
                        # Here we are already inside lock.
                        # We can call get_signal_at_time safely.
                        for name in batch_names:
                            # Note: get_signal_at_time takes lock again (recursive).
                            val = self.get_signal_at_time(name, timestamp)
                            if val is not None:
                                results[name] = val
                            else:
                                results[name] = np.nan # Use NaN for missing values
                    
                except Exception as e:
                    logger.error(f"Batch processing failed for reader {reader}: {e}")
                    # Fallback to individual retrieval
                    for name in batch_names:
                         val = self.get_signal_at_time(name, timestamp)
                         if val is not None:
                             results[name] = val
                                
                except Exception as e:
                    logger.error(f"Error in batch retrieval for {batch_names}: {e}")
                    # Fallback
                    for name in batch_names:
                        val = self.get_signal_at_time(name, timestamp)
                        if val is not None:
                            results[name] = val

            # 2. Handle Other Signals (Legacy/CSV)
            for name in other_signals:
                val = self.get_signal_at_time(name, timestamp)
                if val is not None:
                    results[name] = val
                    
        return results

    def nearest_sample_time(self, signal_name: str, time_point: float) -> Optional[float]:
        """
        Time of the recorded sample of a signal nearest to time_point
        (cursor "snap to data points"), or None if the signal has no data.

        File-backed signals are evenly sampled, so the time is rounded to the
        sample grid of the whole file (the in-memory x_data is only a preview).
        In-memory data (CSV, range-filtered timeline) is searched directly.
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered([signal_name])
            info = self.signal_data.get(signal_name)
            if not info:
                return None
            metadata = info.get('metadata', {})
            full_count = metadata.get('full_count', 0)
            time_range = metadata.get('full_time_range')
            if self._is_file_backed(info) and time_range and full_count > 1:
                t_first, t_last = time_range
                dt = (t_last - t_first) / (full_count - 1)
                if dt > 0:
                    index = min(max(round((time_point - t_first) / dt), 0), full_count - 1)
                    return float(t_first + index * dt)

            x_data = info.get('x_data')
            if x_data is None or len(x_data) == 0:
                return None
            index = int(np.searchsorted(x_data, time_point))
            if index >= len(x_data):
                return float(x_data[-1])
            if index > 0 and time_point - x_data[index - 1] <= x_data[index] - time_point:
                index -= 1
            return float(x_data[index])

    def get_signal_at_time(self, signal_name: str, time_point: float) -> Optional[float]:
        """
        Get signal value at specific time point using interpolation.
        
        For MPAI files, if time_point is outside preview range, loads data from file.
        
        Args:
            signal_name: Signal identifier
            time_point: Time point to query
            
        Returns:
            Interpolated signal value or None if not found
        """
        logger.debug(f"get_signal_at_time called: signal={signal_name}, time={time_point}")
        
        with QMutexLocker(self.mutex):
            self._ensure_filtered([signal_name])
            if signal_name not in self.signal_data:
                logger.warning(f"Signal {signal_name} not found in signal_data")
                return None
            
            signal_info = self.signal_data[signal_name]
            metadata = signal_info.get('metadata', {})
            is_memory_mapped = self._is_file_backed(signal_info)

            # ========== MEMORY-MAPPED MPAI: Use streaming reader ==========
            if is_memory_mapped:
                reader = signal_info.get('mpai_reader')
                if reader is None:
                    logger.error(f"Memory-mapped signal '{signal_name}' has no mpai_reader")
                    return None
                
                col_name = signal_info.get('column_name', signal_name)
                time_col = signal_info.get('time_column', 'time')
                full_count = signal_info.get('row_count', 0)
                time_range = signal_info.get('time_range', (0.0, 0.0))
                
                if full_count == 0:
                    return None
                
                # Check if time_point is within data range
                if time_point < time_range[0] or time_point > time_range[1]:
                    logger.debug(f"Time point {time_point} outside data range {time_range}")
                    return None
                
                try:
                    # Calculate approximate row index from time
                    # Assuming uniform time spacing
                    if time_range[1] > time_range[0]:
                        fraction = (time_point - time_range[0]) / (time_range[1] - time_range[0])
                        center_idx = int(fraction * (full_count - 1))
                    else:
                        center_idx = 0
                    
                    # PERFORMANCE: Use window caching (10k samples per window)
                    WINDOW_SIZE = 10000
                    center_idx = max(0, min(center_idx, full_count - 1))
                    window_start = max(0, (center_idx // WINDOW_SIZE) * WINDOW_SIZE)
                    window_end = min(full_count, window_start + WINDOW_SIZE)
                    
                    # Check cursor value cache first
                    cache_key = (signal_name, window_start)
                    if cache_key in self._cursor_value_cache:
                        x_slice, y_slice = self._cursor_value_cache[cache_key]
                        logger.debug(f"[MPAI CURSOR] Cache HIT for window {window_start}")
                    else:
                        # Cache miss - load window from file
                        row_count = max(1, window_end - window_start)
                        logger.debug(f"[MPAI CURSOR] Cache MISS - loading window {window_start} to {window_end}")
                        
                        y_slice = np.array(reader.load_column_slice(col_name, int(window_start), int(row_count)))
                        x_slice = np.array(reader.load_column_slice(time_col, int(window_start), int(row_count)))
                        
                        # Add to cache (with size limit)
                        if len(self._cursor_value_cache) >= self._cursor_value_cache_max_size:
                            oldest_key = next(iter(self._cursor_value_cache))
                            del self._cursor_value_cache[oldest_key]
                        self._cursor_value_cache[cache_key] = (x_slice, y_slice)
                    
                    # Interpolate
                    if len(x_slice) > 0 and len(y_slice) > 0:
                        if time_point >= x_slice[0] and time_point <= x_slice[-1]:
                            result = float(np.interp(time_point, x_slice, y_slice))
                            logger.debug(f"MPAI cursor value: {signal_name} at {time_point} = {result}")
                            return result
                    
                    return None
                    
                except Exception as e:
                    logger.error(f"Failed to load MPAI data for cursor value at {time_point}: {e}")
                    return None
            
            # ========== CSV SIGNALS: Use in-memory x_data/y_data ==========
            x_data = signal_info.get('x_data')
            y_data = signal_info.get('y_data')
            
            if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
                logger.warning(f"Signal {signal_name} has empty or no data")
                return None
            
            is_mpai = metadata.get('mpai', False) and not signal_info.get('filtered')
            logger.debug(f"Signal info: is_mpai={is_mpai}, preview_range=[{x_data[0]}, {x_data[-1]}], data_len={len(x_data)}")
            
            # Check bounds
            if time_point < x_data[0] or time_point > x_data[-1]:
                logger.debug(f"Time point {time_point} is outside preview range [{x_data[0]}, {x_data[-1]}]")
                # For MPAI, try to load data from file (with caching)
                if is_mpai and hasattr(self, 'raw_dataframe') and hasattr(self.raw_dataframe, 'load_column_slice'):
                    try:
                        # Calculate row index from time
                        full_count = metadata.get('full_count', len(x_data))
                        
                        # Calculate sample rate from preview data (more accurate)
                        if len(x_data) > 1:
                            avg_dt = (x_data[-1] - x_data[0]) / (len(x_data) - 1)
                            sample_rate = 1.0 / avg_dt if avg_dt > 0 else 1.0
                        else:
                            sample_rate = 1.0
                        
                        # PERFORMANCE: Use larger windows (10k samples) for caching
                        WINDOW_SIZE = 10000
                        center_idx = int(time_point * sample_rate)
                        
                        # SAFETY: Ensure indices are within valid range
                        center_idx = max(0, min(center_idx, full_count - 1))
                        window_start = max(0, (center_idx // WINDOW_SIZE) * WINDOW_SIZE)
                        window_end = min(full_count, window_start + WINDOW_SIZE)
                        
                        # Check cursor value cache first
                        cache_key = (signal_name, window_start)
                        if cache_key in self._cursor_value_cache:
                            x_slice, y_slice = self._cursor_value_cache[cache_key]
                            logger.debug(f"[MPAI CURSOR] Cache HIT for window {window_start}")
                        else:
                            # Cache miss - load window from file
                            row_count = max(1, window_end - window_start)
                            logger.debug(f"[MPAI CURSOR] Cache MISS - loading window {window_start} to {window_end}")
                            
                            y_slice = self.raw_dataframe.load_column_slice(signal_name, int(window_start), int(row_count))
                            x_slice = np.arange(window_start, window_start + len(y_slice), dtype=np.float64) / sample_rate
                            
                            # Add to cache (with size limit)
                            if len(self._cursor_value_cache) >= self._cursor_value_cache_max_size:
                                oldest_key = next(iter(self._cursor_value_cache))
                                del self._cursor_value_cache[oldest_key]
                            self._cursor_value_cache[cache_key] = (x_slice, y_slice)
                        
                        # Interpolate
                        if len(x_slice) > 0 and time_point >= x_slice[0] and time_point <= x_slice[-1]:
                            result = float(np.interp(time_point, x_slice, y_slice))
                            logger.debug(f"MPAI cursor value: {signal_name} at {time_point} = {result}")
                            return result
                        
                    except Exception as e:
                        logger.error(f"Failed to load MPAI data for cursor value at {time_point}: {e}", exc_info=True)
                
                # Out of bounds or failed to load
                return None
            
            # Within preview range - use cached data
            result = float(np.interp(time_point, x_data, y_data))
            logger.debug(f"Using cached preview data: {signal_name} at {time_point} = {result}")
            return result

    def get_raw_range(self, signal_name: str, start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Full-resolution (x, y) of a signal in [start_time, end_time] (None: no
        limit), with the real values: not downsampled, not normalized.

        For analysis such as correlation; get_all_signals() returns a
        downsampled min/max view for drawing, which must not be used for that.
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered([signal_name])
            info = self.signal_data.get(signal_name)
            if info is None:
                return None

            if self._is_file_backed(info) or (info.get('_is_calc_param') and '_reader' in info):
                if info.get('_is_calc_param'):
                    reader, col_name, time_col = info['_reader'], signal_name, 'time'
                else:
                    reader = info.get('mpai_reader')
                    col_name = info.get('column_name', signal_name)
                    time_col = info.get('time_column', 'time')
                if reader is None or getattr(reader, 'closed', False):
                    return None
                row_count = reader.get_row_count()
                start_row = 0 if start_time is None else \
                    self._time_to_row(reader, time_col, row_count, start_time, 'left')
                end_row = row_count if end_time is None else \
                    self._time_to_row(reader, time_col, row_count, end_time, 'right')
                if end_row <= start_row:
                    return np.array([]), np.array([])
                count = end_row - start_row
                x = np.asarray(reader.load_column_slice(time_col, start_row, count), dtype=np.float64)
                y = np.asarray(reader.load_column_slice(col_name, start_row, count), dtype=np.float64)
                n = min(len(x), len(y))
                return x[:n], y[:n]

            x = info.get('x_data')
            y = info.get('y_data')
            if x is None or y is None:
                return None
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            mask = np.ones(n, dtype=bool)
            if start_time is not None:
                mask &= x >= start_time
            if end_time is not None:
                mask &= x <= end_time
            return x[mask], y[mask]

    @staticmethod
    def _time_to_row(reader, time_col: str, row_count: int, t: float, side: str) -> int:
        """
        Binary search on the (ascending) time column without loading it:
        first row with time >= t ('left') or > t ('right').
        """
        lo, hi = 0, row_count
        while lo < hi:
            mid = (lo + hi) // 2
            value = float(reader.load_column_slice(time_col, mid, 1)[0])
            if value < t or (side == 'right' and value == t):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get_signal_range(self, signal_name: str, start_time: float, end_time: float) -> Optional[Dict]:
        """
        Get signal data within time range.
        
        Args:
            signal_name: Signal identifier
            start_time: Range start time
            end_time: Range end time
            
        Returns:
            Dict with x_data, y_data, and statistics for the range
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered([signal_name])
            if signal_name not in self.signal_data:
                return None
            
            signal_info = self.signal_data[signal_name]
            metadata = signal_info.get('metadata', {})
            is_memory_mapped = self._is_file_backed(signal_info)

            # ========== MEMORY-MAPPED MPAI: Use streaming reader ==========
            if is_memory_mapped:
                reader = signal_info.get('mpai_reader')
                if reader is None:
                    logger.error(f"Memory-mapped signal '{signal_name}' has no mpai_reader")
                    return None
                
                col_name = signal_info.get('column_name', signal_name)
                time_col = signal_info.get('time_column', 'time')
                
                try:
                    row_count = reader.get_row_count()
                    raw_time = np.asarray(reader.load_column_slice(time_col, 0, row_count), dtype=np.float64)
                    raw_y = np.asarray(reader.load_column_slice(col_name, 0, row_count), dtype=np.float64)
                    mask = (raw_time >= start_time) & (raw_time <= end_time)
                    y_range = raw_y[mask]
                    t_range = raw_time[mask]
                    stats = self._calc_numpy_stats(y_range, t_range)
                    return {'x_data': t_range, 'y_data': y_range, 'statistics': stats}
                except Exception as e:
                    logger.error(f"Failed to get MPAI signal range: {e}")
                    return None
            
            # ========== CSV SIGNALS: Use in-memory x_data/y_data ==========
            x_data = signal_info.get('x_data')
            y_data = signal_info.get('y_data')
            
            if x_data is None or y_data is None:
                logger.warning(f"Signal '{signal_name}' has no x_data/y_data")
                return None
            
            # Find indices within range
            mask = (x_data >= start_time) & (x_data <= end_time)
            
            if not np.any(mask):
                return None
            
            x_range = x_data[mask]
            y_range = y_data[mask]
            
            # Calculate statistics for range (using default auto threshold mode)
            stats = self._calculate_signal_statistics(y_range, x_range)
            
            return {
                'x_data': x_range,
                'y_data': y_range,
                'statistics': stats
            }
