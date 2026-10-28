"""
Statistics of signals and time ranges, with a result cache.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalStatisticsMixin:
    """Statistics of signals and time ranges, with a result cache."""

    def calculate_statistics(self, signal_names: Optional[List[str]] = None, 
                           time_range: Optional[Tuple[float, float]] = None,
                           duty_cycle_threshold_mode: str = "auto",
                           duty_cycle_threshold_value: float = 0.0) -> Dict[str, Dict]:
        """
        Calculate comprehensive statistics for signals.
        
        Args:
            signal_names: Signals to analyze (None for all)
            time_range: Time range for analysis (start, end)
            duty_cycle_threshold_mode: "auto" (use mean) or "manual" (use custom value)
            duty_cycle_threshold_value: Custom threshold value for manual mode
            
        Returns:
            Dict of signal_name -> statistics_dict
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered(signal_names)
            if signal_names is None:
                signal_names = list(self.signal_data.keys())
            
            results = {}
            
            for name in signal_names:
                if name not in self.signal_data:
                    continue
                
                signal_info = self.signal_data[name]
                metadata = signal_info.get('metadata', {})
                # A range-filtered signal holds its (concatenated) data in memory;
                # the file has the unfiltered data and a different timeline
                is_mpai = metadata.get('mpai', False) and not signal_info.get('filtered')
                is_memory_mapped = self._is_file_backed(signal_info)

                # ========== MEMORY-MAPPED MPAI: Use MpaiDirectoryReader ==========
                if is_memory_mapped:
                    reader = signal_info.get('mpai_reader')
                    col_name = signal_info.get('column_name', name)
                    time_col = signal_info.get('time_column', 'time')

                    if reader is None:
                        logger.error(f"[STATS] Memory-mapped signal '{name}' has no mpai_reader")
                        continue

                    # 1. Check if this is the Python MpaiDirectoryReader (has get_statistics_snapshot)
                    if hasattr(reader, 'get_statistics_snapshot') and hasattr(reader, 'name_to_id'):
                            # Use the new Directory Reader API
                            try:
                                # Get Channel ID (required for Reader API)
                                if col_name not in reader.name_to_id:
                                    logger.warning(f"[STATS] Channel '{col_name}' not found in Directory Reader")
                                    continue
                                ch_id = reader.name_to_id[col_name]
                                
                                # Determine time range
                                if time_range is not None:
                                    start_time, end_time = time_range
                                else:
                                    # Full range
                                    start_time, end_time = reader.get_time_range()
                                
                                # Calculate using dual-stream reader
                                reader_stats = reader.get_statistics_snapshot(ch_id, start_time, end_time)
                                
                                # Map to format
                                stats = {
                                    'mean': reader_stats['avg'],
                                    'std': reader_stats['std'],
                                    'min': reader_stats['min'],
                                    'max': reader_stats['max'],
                                    'rms': reader_stats['rms'],
                                    'peak_to_peak': reader_stats['max'] - reader_stats['min'],
                                    # 'count' not returned by snapshot, estimation?
                                    'count': int((end_time - start_time) / reader.dt) if reader.dt > 0 else 0
                                }
                                
                                # Duty Cycle (Use calculated if available, else 50.0)
                                threshold = stats['mean'] if duty_cycle_threshold_mode == "auto" else duty_cycle_threshold_value
                                stats['duty_cycle'] = reader_stats.get('duty_cycle', 50.0)
                                stats['duty_cycle_threshold'] = threshold
                                
                                results[name] = stats
                                continue
                                
                            except Exception as e:
                                logger.error(f"[STATS] MpaiDirectoryReader failed for '{name}': {e}")
                                continue

                    # 2. Pure Python: Load from reader and calculate with NumPy
                    try:
                        row_count = reader.get_row_count()
                        if time_range is not None:
                            start_time, end_time = time_range
                            raw_time = np.asarray(reader.load_column_slice(time_col, 0, row_count), dtype=np.float64)
                            mask = (raw_time >= start_time) & (raw_time <= end_time)
                            raw_data = np.asarray(reader.load_column_slice(col_name, 0, row_count), dtype=np.float64)
                            y_subset = raw_data[mask]
                            x_subset = raw_time[mask]
                        else:
                            y_subset = np.asarray(reader.load_column_slice(col_name, 0, row_count), dtype=np.float64)
                            x_subset = None

                        stats = self._calc_numpy_stats(y_subset, x_subset)
                        results[name] = stats
                    except Exception as e:
                        logger.error(f"[STATS] Python reader stats failed for '{name}': {e}")
                    continue
                
                # ========== CSV SIGNALS: Use in-memory x_data/y_data ==========
                x_data = signal_info.get('x_data')
                y_data = signal_info.get('y_data')
                
                if x_data is None or y_data is None:
                    logger.warning(f"[STATS] Signal '{name}' has no x_data/y_data")
                    continue
                
                # DEBUG: Log signal info
                logger.debug(f"[STATS DEBUG] Signal: {name}, is_mpai: {is_mpai}, data_length: {len(x_data)}")
                
                # ========== MPAI with preview data: Optimized statistics calculation ==========
                if is_mpai and hasattr(self, 'raw_dataframe') and hasattr(self.raw_dataframe, 'get_header'):
                    try:
                        mpai_reader = self.raw_dataframe
                        full_count = metadata.get('full_count', mpai_reader.get_row_count())

                        if time_range is not None:
                            start_time, end_time = time_range
                            # Use preview data if time range is within it
                            if (len(x_data) > 0 and start_time >= x_data[0] and end_time <= x_data[-1]):
                                mask = (x_data >= start_time) & (x_data <= end_time)
                                if np.any(mask):
                                    stats = self._calculate_signal_statistics(
                                        y_data[mask], x_data[mask],
                                        duty_cycle_threshold_mode, duty_cycle_threshold_value
                                    )
                                    results[name] = stats
                                    continue
                            # Load from reader for out-of-preview range
                            time_col_reader = metadata.get('time_column', 'time')
                            raw_time = np.asarray(mpai_reader.load_column_slice(time_col_reader, 0, full_count), dtype=np.float64)
                            raw_y = np.asarray(mpai_reader.load_column_slice(name, 0, full_count), dtype=np.float64)
                            mask = (raw_time >= start_time) & (raw_time <= end_time)
                            stats = self._calc_numpy_stats(raw_y[mask] if np.any(mask) else raw_y[:0])
                        else:
                            raw_y = np.asarray(mpai_reader.load_column_slice(name, 0, full_count), dtype=np.float64)
                            stats = self._calc_numpy_stats(raw_y)

                        threshold = stats['mean'] if duty_cycle_threshold_mode == "auto" else duty_cycle_threshold_value
                        stats['duty_cycle'] = 50.0
                        stats['duty_cycle_threshold'] = threshold
                        results[name] = stats
                        continue

                    except Exception as e:
                        logger.error(f"[STATS] MPAI stats failed for {name}: {e}", exc_info=True)
                        # Fall through to preview data
                
                # ========== CSV/Fallback: Use preview data (Python) ==========
                if time_range is not None:
                    start_time, end_time = time_range
                    mask = (x_data >= start_time) & (x_data <= end_time)
                    if np.any(mask):
                        y_subset = y_data[mask]
                        x_subset = x_data[mask]
                    else:
                        continue
                else:
                    y_subset = y_data
                    x_subset = x_data
                
                # Calculate with Python
                stats = self._calculate_signal_statistics(y_subset, x_subset, duty_cycle_threshold_mode, duty_cycle_threshold_value)
                results[name] = stats
            
            return results

    def get_statistics(self, signal_name: str, time_range: Optional[Tuple[float, float]] = None,
                      duty_cycle_threshold_mode: str = "auto", duty_cycle_threshold_value: float = 0.0) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific signal with caching.
        
        Args:
            signal_name: Signal identifier
            time_range: Optional time range (start, end)
            duty_cycle_threshold_mode: "auto" (use mean) or "manual" (use custom value)
            duty_cycle_threshold_value: Custom threshold value for manual mode
            
        Returns:
            Statistics dictionary or None if signal not found
        """
        # PERFORMANCE: Check cache first
        cache_key = self._make_cache_key(signal_name, time_range, duty_cycle_threshold_mode, duty_cycle_threshold_value)
        
        if cache_key in self._stats_cache:
            logger.debug(f"Statistics cache HIT for {signal_name}")
            return self._stats_cache[cache_key].copy()
        
        # Cache miss - calculate statistics
        logger.debug(f"Statistics cache MISS for {signal_name}")
        stats_dict = self.calculate_statistics([signal_name], time_range, duty_cycle_threshold_mode, duty_cycle_threshold_value)
        result = stats_dict.get(signal_name)
        
        # Store in cache
        if result is not None:
            self._add_to_cache(cache_key, result)
        
        return result

    def _make_cache_key(self, signal_name: str, time_range: Optional[Tuple[float, float]], 
                       threshold_mode: str, threshold_value: float) -> tuple:
        """Create a cache key for statistics."""
        # Round time range to 6 decimal places to avoid floating point comparison issues
        if time_range is not None:
            range_key = (round(time_range[0], 6), round(time_range[1], 6))
        else:
            range_key = None
        
        return (signal_name, range_key, threshold_mode, round(threshold_value, 6))

    def _add_to_cache(self, cache_key: tuple, stats: Dict[str, float]):
        """Add statistics to cache with size management."""
        # If cache is full, remove oldest entry (simple FIFO)
        if len(self._stats_cache) >= self._stats_cache_max_size:
            # Remove first item (oldest)
            first_key = next(iter(self._stats_cache))
            del self._stats_cache[first_key]
            logger.debug(f"Statistics cache full, removed oldest entry")
        
        self._stats_cache[cache_key] = stats.copy()

    def clear_statistics_cache(self):
        """Clear the statistics cache. Call when data changes."""
        self._stats_cache.clear()
        logger.debug("Statistics cache cleared")

    @staticmethod
    def _calc_numpy_stats(y: np.ndarray, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Minimal NumPy statistics dict (used by MPAI reader paths)."""
        valid = y[~np.isnan(y)] if len(y) > 0 else y
        if len(valid) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'rms': 0.0, 'peak_to_peak': 0.0,
                    'count': 0, 'valid_count': 0}
        return {
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid, ddof=1)),
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'median': float(np.median(valid)),
            'rms': float(np.sqrt(np.mean(valid ** 2))),
            'peak_to_peak': float(np.max(valid) - np.min(valid)),
            'count': len(y),
            'valid_count': len(valid),
        }

    def _calculate_signal_statistics(self, y_data: np.ndarray, x_data: np.ndarray, 
                                   duty_cycle_threshold_mode: str = "auto", 
                                   duty_cycle_threshold_value: float = 0.0,
                                   include_percentiles: bool = False) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a single signal.
        
        OPTIMIZED: Lazy percentile calculation - only when requested.
        Percentiles are expensive (require sorting), so we skip them by default.
        
        Args:
            y_data: Signal Y values
            x_data: Signal X values (time)
            duty_cycle_threshold_mode: Threshold mode for duty cycle
            duty_cycle_threshold_value: Threshold value for duty cycle
            include_percentiles: If True, calculate median and percentiles (slower)
        
        Returns:
            Statistics dictionary
        """
        if len(y_data) == 0:
            return {}
        
        # Convert to numpy array if list (for C++ binding compatibility)
        if isinstance(y_data, list):
            y_data = np.array(y_data)
        if isinstance(x_data, list):
            x_data = np.array(x_data)
        
        # PERFORMANCE: Basic statistics (fast, vectorized)
        stats = {
            'count': len(y_data),
            'mean': np.mean(y_data),
            'std': np.std(y_data),
            'min': np.min(y_data),
            'max': np.max(y_data),
            'rms': np.sqrt(np.mean(y_data**2)),
            'peak_to_peak': np.ptp(y_data),
        }
        
        # PERFORMANCE: Lazy percentiles - only calculate if requested
        # Percentiles require sorting which is O(n log n) - expensive!
        if include_percentiles and len(y_data) > 1:
            stats['median'] = np.median(y_data)
            percentiles = np.percentile(y_data, [25, 75])
            stats['q25'] = percentiles[0]
            stats['q75'] = percentiles[1]
            stats['iqr'] = percentiles[1] - percentiles[0]
        
        # Duty Cycle Calculation (based on threshold mode)
        if len(y_data) > 1:
            try:
                # Determine threshold based on mode
                if duty_cycle_threshold_mode == "manual":
                    threshold = duty_cycle_threshold_value
                else:  # auto mode - use mean
                    threshold = stats['mean'] if stats['mean'] is not None else 0.0
                # Find indices where the signal crosses the threshold
                crossings = np.where(np.diff(y_data > threshold))[0]
                
                # Calculate time spent above threshold
                high_time = 0
                is_high = y_data[0] > threshold
                last_cross_time = x_data[0]

                for cross_idx in crossings:
                    cross_time = x_data[cross_idx]
                    if is_high:
                        high_time += cross_time - last_cross_time
                    is_high = not is_high
                    last_cross_time = cross_time
                
                # Add time for the last segment
                if is_high:
                    high_time += x_data[-1] - last_cross_time

                total_duration = x_data[-1] - x_data[0]
                stats['duty_cycle'] = (high_time / total_duration) * 100 if total_duration > 0 else 0
            except Exception as e:
                logger.warning(f"Could not calculate duty cycle: {e}")
                stats['duty_cycle'] = 0
        else:
            stats['duty_cycle'] = 0

        # Time-based statistics
        if len(x_data) > 1:
            dt = np.diff(x_data)
            stats['sample_rate'] = 1.0 / np.mean(dt) if np.mean(dt) > 0 else 0
            stats['duration'] = x_data[-1] - x_data[0]
        
        # Advanced statistics (if enough data points)
        if len(y_data) > 10:
            # Skewness and kurtosis approximation
            centered = y_data - stats['mean']
            if stats['std'] > 0:
                normalized = centered / stats['std']
                stats['skewness'] = np.mean(normalized**3)
                stats['kurtosis'] = np.mean(normalized**4) - 3
        
        return stats
