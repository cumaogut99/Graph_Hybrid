"""
C++ Signal Processor Wrapper
Integrates C++ StatisticsEngine with existing Python infrastructure
"""

import logging
import time
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)

# Try to import C++ module
try:
    import time_graph_cpp as tgcpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("C++ module not available for SignalProcessor")


class CppSignalProcessor(QObject):
    """
    C++ accelerated signal processor.
    
    Features:
    - Fast C++ statistics (5-10x faster)
    - Threshold analysis
    - Automatic fallback to Polars/NumPy
    - Compatible with existing SignalProcessor API
    """
    
    # Signals
    processing_started = Signal()
    processing_finished = Signal()
    statistics_updated = Signal(dict)
    performance_log = Signal(dict)
    
    def __init__(self, use_cpp: bool = True):
        super().__init__()
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.signal_data = {}
        self.raw_dataframe = None
        self.time_column_name = None
        self._stats_cache = {}
        self._performance_metrics = {}
    
    def calculate_statistics(
        self,
        signal_name: str,
        range_start: Optional[float] = None,
        range_end: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a signal.
        
        Args:
            signal_name: Name of the signal
            range_start: Start of range (None = full range)
            range_end: End of range (None = full range)
            threshold: Threshold for analysis (None = no threshold)
        
        Returns:
            Dictionary with statistics
        """
        # Check cache
        cache_key = (signal_name, range_start, range_end, threshold)
        if cache_key in self._stats_cache:
            logger.debug(f"Using cached statistics for {signal_name}")
            return self._stats_cache[cache_key]
        
        t_start = time.perf_counter()
        
        # Try C++ first
        if self.use_cpp:
            try:
                stats = self._calculate_statistics_cpp(
                    signal_name, range_start, range_end, threshold
                )
                method = 'cpp'
            except Exception as e:
                logger.warning(f"C++ statistics failed, falling back to NumPy: {e}")
                stats = self._calculate_statistics_numpy(
                    signal_name, range_start, range_end, threshold
                )
                method = 'numpy_fallback'
        else:
            stats = self._calculate_statistics_numpy(
                signal_name, range_start, range_end, threshold
            )
            method = 'numpy'
        
        t_end = time.perf_counter()
        calc_time = (t_end - t_start) * 1000  # ms
        
        # Add performance info
        stats['_performance'] = {
            'method': method,
            'calc_time_ms': calc_time
        }
        
        # Cache result
        self._stats_cache[cache_key] = stats
        
        # Limit cache size
        if len(self._stats_cache) > 100:
            # Remove oldest entry
            self._stats_cache.pop(next(iter(self._stats_cache)))
        
        logger.debug(f"Statistics ({method}): {calc_time:.2f}ms for {signal_name}")
        
        return stats
    
    def _calculate_statistics_cpp(
        self,
        signal_name: str,
        range_start: Optional[float],
        range_end: Optional[float],
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate statistics using C++ StatisticsEngine.
        
        MPAI: Streaming, no temp CSV.
        Polars: Falls back to CSV bridge (kept for compatibility).
        """
        if self.raw_dataframe is None:
            raise ValueError("No data loaded")
        
        # --- MPAI streaming fast path ---
        if hasattr(self.raw_dataframe, "get_header"):
            reader = self.raw_dataframe
            
            # Threshold mode not supported in streaming; fallback handled by caller
            if threshold is not None and self.time_column_name:
                logger.warning("Threshold stats not supported in streaming mode; falling back to NumPy")
                raise RuntimeError("threshold_not_supported_streaming")
            
            row_count = reader.get_row_count()
            start_row = 0
            count = 0  # 0 => all rows
            
            if range_start is not None and range_end is not None and self.time_column_name:
                try:
                    # Estimate sample rate from preview data if available
                    if signal_name in self.signal_data:
                        x_preview = self.signal_data[signal_name].get("x_data", [])
                        if len(x_preview) > 1:
                            duration = max(x_preview[-1] - x_preview[0], 1e-9)
                            sample_rate = (len(x_preview) - 1) / duration
                        else:
                            sample_rate = 1.0
                    else:
                        sample_rate = 1.0
                    
                    start_row = max(0, int(range_start * sample_rate))
                    end_row = min(row_count, int(range_end * sample_rate))
                    count = max(0, end_row - start_row)
                except Exception:
                    start_row, count = 0, 0
            
            cpp_stats = tgcpp.StatisticsEngine.calculate_streaming(
                reader,
                signal_name,
                int(start_row),
                int(count)
            )
            
            return {
                'mean': cpp_stats.mean,
                'std': cpp_stats.std_dev,
                'min': cpp_stats.min,
                'max': cpp_stats.max,
                'median': cpp_stats.median,
                'sum': cpp_stats.sum,
                'count': cpp_stats.count,
                'valid_count': cpp_stats.valid_count,
                'rms': getattr(cpp_stats, "rms", 0.0),
                'peak_to_peak': getattr(cpp_stats, "peak_to_peak", cpp_stats.max - cpp_stats.min),
            }
        
        # --- Polars DataFrame path (uses CSV bridge) ---
        if signal_name not in self.raw_dataframe.columns:
            raise ValueError(f"Signal '{signal_name}' not found")
        
        cpp_df = self._polars_to_cpp(self.raw_dataframe)
        
        if threshold is not None and self.time_column_name:
            cpp_stats = tgcpp.StatisticsEngine.calculate_with_threshold(
                cpp_df,
                signal_name,
                self.time_column_name,
                threshold
            )
            
            return {
                'mean': cpp_stats.mean,
                'std': cpp_stats.std_dev,
                'min': cpp_stats.min,
                'max': cpp_stats.max,
                'median': cpp_stats.median,
                'sum': cpp_stats.sum,
                'count': cpp_stats.count,
                'valid_count': cpp_stats.valid_count,
                'threshold': cpp_stats.threshold,
                'above_count': cpp_stats.above_count,
                'below_count': cpp_stats.below_count,
                'above_percentage': cpp_stats.above_percentage,
                'below_percentage': cpp_stats.below_percentage,
                'time_above': cpp_stats.time_above,
                'time_below': cpp_stats.time_below
            }
        
        if range_start is not None and range_end is not None and self.time_column_name:
            time_data = self.raw_dataframe[self.time_column_name].to_numpy()
            start_idx = np.searchsorted(time_data, range_start)
            end_idx = np.searchsorted(time_data, range_end)
            
            cpp_stats = tgcpp.StatisticsEngine.calculate_range(
                cpp_df,
                signal_name,
                start_idx,
                end_idx
            )
        else:
            cpp_stats = tgcpp.StatisticsEngine.calculate(
                cpp_df,
                signal_name
            )
        
        return {
            'mean': cpp_stats.mean,
            'std': cpp_stats.std_dev,
            'min': cpp_stats.min,
            'max': cpp_stats.max,
            'median': cpp_stats.median,
            'sum': cpp_stats.sum,
            'count': cpp_stats.count,
            'valid_count': cpp_stats.valid_count
        }
    
    def _polars_to_cpp(self, polars_df):
        """Convert Polars DataFrame to C++ DataFrame."""
        import tempfile
        import os
        
        # Write to temp CSV file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
        try:
            os.close(temp_fd)  # Close file descriptor
            polars_df.write_csv(temp_path)
            
            cpp_opts = tgcpp.CsvOptions()
            cpp_opts.has_header = True
            cpp_df = tgcpp.DataFrame.load_csv(temp_path, cpp_opts)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return cpp_df
    
    def _calculate_statistics_numpy(
        self,
        signal_name: str,
        range_start: Optional[float],
        range_end: Optional[float],
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Fallback: Calculate statistics using NumPy."""
        if self.raw_dataframe is None:
            raise ValueError("No data loaded")
        
        if signal_name not in self.raw_dataframe.columns:
            raise ValueError(f"Signal '{signal_name}' not found")
        
        # Get data
        data = self.raw_dataframe[signal_name].to_numpy()
        
        # Apply range filter
        if range_start is not None and range_end is not None and self.time_column_name:
            time_data = self.raw_dataframe[self.time_column_name].to_numpy()
            mask = (time_data >= range_start) & (time_data <= range_end)
            data = data[mask]
        
        # Remove NaN values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'sum': 0.0,
                'count': len(data),
                'valid_count': 0
            }
        
        # Calculate basic statistics
        stats = {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data, ddof=1)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'median': float(np.median(valid_data)),
            'sum': float(np.sum(valid_data)),
            'count': len(data),
            'valid_count': len(valid_data)
        }
        
        # Threshold analysis
        if threshold is not None:
            above_mask = valid_data > threshold
            stats['threshold'] = threshold
            stats['above_count'] = int(np.sum(above_mask))
            stats['below_count'] = int(len(valid_data) - stats['above_count'])
            stats['above_percentage'] = (stats['above_count'] / len(valid_data)) * 100
            stats['below_percentage'] = (stats['below_count'] / len(valid_data)) * 100
            
            # Time analysis (if time column available)
            if self.time_column_name and range_start is None:
                time_data = self.raw_dataframe[self.time_column_name].to_numpy()
                if len(time_data) > 1:
                    time_diffs = np.diff(time_data)
                    # Ensure masks align
                    if len(above_mask) > len(time_diffs):
                        above_mask_aligned = above_mask[:-1]
                    else:
                        above_mask_aligned = above_mask
                    
                    above_time = np.sum(time_diffs[above_mask_aligned])
                    stats['time_above'] = float(above_time)
                    stats['time_below'] = float(np.sum(time_diffs) - above_time)
                else:
                    stats['time_above'] = 0.0
                    stats['time_below'] = 0.0
            else:
                stats['time_above'] = 0.0
                stats['time_below'] = 0.0
        
        return stats
    
    def set_data(self, df: pl.DataFrame, time_column: str):
        """Set the data for processing."""
        self.raw_dataframe = df
        self.time_column_name = time_column
        self.clear_cache()
    
    def clear_cache(self):
        """Clear statistics cache."""
        self._stats_cache.clear()
    
    @staticmethod
    def is_cpp_available():
        """Check if C++ module is available."""
        return CPP_AVAILABLE

