"""
Graph Renderer - Handles segmented and concatenated graph rendering
"""

import logging
import numpy as np
import pyqtgraph as pg
from typing import List, Dict, Any, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal, QThread

logger = logging.getLogger(__name__)

# Try to import C++ module for fast limit violation detection
try:
    import time_graph_cpp as tgcpp
    CPP_AVAILABLE = True
    logger.info("[LIMIT_VIOLATION] C++ acceleration available")
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("[LIMIT_VIOLATION] C++ module not available, using Python fallback")


class LimitViolationCalculator(QObject):
    """
    DEPRECATED: Python fallback for limit violation calculations.
    This class is no longer used - C++ SIMD acceleration is required.
    Kept for backward compatibility but will raise error if used.
    """
    result_ready = Signal(dict)

    def __init__(self, signal_name: str, x_data: np.ndarray, y_data: np.ndarray, limits: Dict[str, float]):
        super().__init__()
        logger.error("[LIMIT_VIOLATION] DEPRECATED: Python LimitViolationCalculator should not be used!")
        raise RuntimeError(
            "Python limit violation calculator is disabled!\n\n"
            "C++ SIMD acceleration is required for accurate violation detection.\n"
            "Please ensure the C++ module is compiled and available."
        )

    def run(self):
        pass

    def stop(self):
        pass


class DeviationCalculator(QObject):
    """
    Performs deviation calculations in a separate thread.
    Emits results when calculations are complete.
    """
    result_ready = Signal(dict)

    def __init__(self, signal_data: np.ndarray, settings: Dict[str, Any]):
        super().__init__()
        self.signal_data = signal_data
        self.settings = settings
        self.should_stop = False

    def run(self):
        """Starts the calculation."""
        if self.should_stop:
            return
        results = self._calculate_basic_deviation(self.signal_data, self.settings)
        if not self.should_stop:
            self.result_ready.emit(results)
    
    def stop(self):
        """Stop the calculation."""
        self.should_stop = True
        
    def _calculate_basic_deviation(self, signal_data: np.ndarray, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vectorized calculation of basic deviation analysis.
        This optimized version avoids loops for performance.
        """
        if len(signal_data) < 2:
            return {'deviations': [], 'bands': [], 'alerts': [], 'trend_line': [], 'red_segments': []}

        results = {
            'deviations': [], 'bands': [], 'alerts': [], 'trend_line': [], 'red_segments': []
        }

        # Trend Analysis (already reasonably performant)
        if settings.get('trend_analysis', {}).get('enabled', False):
            sensitivity = settings['trend_analysis'].get('sensitivity', 3)
            results['trend_line'] = self._calculate_trend_line(signal_data, sensitivity)
        
        # --- Optimized Fluctuation, Bands, and Red Segments Calculation ---
        fluctuation_settings = settings.get('fluctuation_detection', {})
        if fluctuation_settings.get('enabled', False):
            window_size = fluctuation_settings.get('sample_window', 20)
            threshold_percent = fluctuation_settings.get('threshold_percent', 10.0)
            
            if len(signal_data) < window_size:
                return results

            # Use a single rolling mean calculation for all features
            # Pad the data at the beginning to handle edges correctly
            padded_data = np.pad(signal_data, (window_size - 1, 0), mode='edge')
            
            # Create a rolling window view
            shape = (len(signal_data), window_size)
            strides = (signal_data.strides[0], signal_data.strides[0])
            rolling_view = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
            
            # Calculate rolling mean vectorized
            rolling_mean = np.mean(rolling_view, axis=1)

            # --- 1. Fluctuation Detection ---
            # Calculate deviation percentage, handle division by zero
            epsilon = 1e-9
            deviation_percent = np.abs((signal_data - rolling_mean) / (rolling_mean + epsilon)) * 100
            results['deviations'] = deviation_percent.tolist()

            # Find alerts
            alert_indices = np.where(deviation_percent > threshold_percent)[0]
            results['alerts'] = [{
                'index': int(i),
                'value': signal_data[i],
                'expected': rolling_mean[i],
                'deviation_percent': deviation_percent[i]
            } for i in alert_indices]

            # --- 2. Deviation Bands ---
            if settings.get('visual_settings', {}).get('show_bands', False):
                threshold_val = rolling_mean * (threshold_percent / 100.0)
                upper_band = rolling_mean + threshold_val
                lower_band = rolling_mean - threshold_val
                results['bands'] = {'upper': upper_band.tolist(), 'lower': lower_band.tolist()}

            # --- 3. Red Segments ---
            if fluctuation_settings.get('red_highlighting', False):
                exceeds_threshold = deviation_percent > threshold_percent
                
                # Find start and end of consecutive True blocks
                diff = np.diff(np.concatenate(([False], exceeds_threshold, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0] - 1
                
                red_segments = []
                for start, end in zip(starts, ends):
                    if end >= start:
                        segment_slice = slice(start, end + 1)
                        red_segments.append({
                            'start_index': int(start),
                            'end_index': int(end),
                            'values': signal_data[segment_slice].tolist(),
                            'deviation_percent': np.max(deviation_percent[segment_slice])
                        })
                results['red_segments'] = red_segments

        return results
        
    def _calculate_trend_line(self, data: np.ndarray, sensitivity: int) -> List[float]:
        """Calculate trend line based on sensitivity."""
        if len(data) < 2:
            return data.tolist()
            
        # Simple linear regression for trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend_line = np.polyval(coeffs, x)
        
        # Apply sensitivity smoothing
        if sensitivity <= 2:  # Low sensitivity - more smoothing
            window = min(len(data) // 4, 20)
        elif sensitivity >= 4:  # High sensitivity - less smoothing
            window = min(len(data) // 10, 5)
        else:  # Medium sensitivity
            window = min(len(data) // 8, 10)
            
        if window > 1:
            # Apply moving average smoothing
            trend_smoothed = np.convolve(trend_line, np.ones(window)/window, mode='same')
            return trend_smoothed.tolist()
        
        return trend_line.tolist()
        
    # --- The old, slow methods below are now replaced by the vectorized _calculate_basic_deviation ---
    # We can remove them or keep them for reference, but they are no longer called.

    def _detect_fluctuations(self, data: np.ndarray, window_size: int, threshold_percent: float) -> tuple:
        """
        [DEPRECATED] Detect short-term fluctuations. 
        Replaced by vectorized version in _calculate_basic_deviation.
        """
        deviations = []
        alerts = []
        
        if len(data) < window_size:
            return deviations, alerts

        # Vectorized calculation
        # Create a rolling window view of the data without copying data
        shape = (data.shape[0] - window_size + 1, window_size)
        strides = (data.strides[0], data.strides[0])
        rolling_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
        # Calculate rolling mean
        rolling_mean = np.mean(rolling_data, axis=1)
        
        # Current values are the ones after each window
        current_values = data[window_size:]
        
        # Calculate deviation percentage, handle division by zero
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-9
        deviation_percent = np.abs((current_values - rolling_mean[:-1]) / (rolling_mean[:-1] + epsilon)) * 100
        
        deviations = deviation_percent.tolist()
        
        # Find alerts where deviation exceeds threshold
        alert_indices = np.where(deviation_percent > threshold_percent)[0] + window_size
        
        alerts = [{
            'index': int(i),
            'value': data[i],
            'expected': rolling_mean[i - window_size],
            'deviation_percent': deviation_percent[i - window_size]
        } for i in alert_indices]
        
        return deviations, alerts

    def _calculate_deviation_bands(self, data: np.ndarray, settings: Dict) -> Dict[str, List[float]]:
        """
        [DEPRECATED] Calculate deviation bands for visualization.
        Replaced by vectorized version in _calculate_basic_deviation.
        """
        if len(data) < 2:
            return {'upper': [], 'lower': []}
            
        window_size = settings.get('fluctuation_detection', {}).get('sample_window', 20)
        threshold_percent = settings.get('fluctuation_detection', {}).get('threshold_percent', 10.0)

        # Use pandas for efficient rolling window calculations
        try:
            import pandas as pd
            s = pd.Series(data)
            rolling_mean = s.rolling(window=window_size, min_periods=1).mean().to_numpy()
            
            threshold_val = rolling_mean * (threshold_percent / 100.0)
            
            upper_band = rolling_mean + threshold_val
            lower_band = rolling_mean - threshold_val
            
            return {'upper': upper_band.tolist(), 'lower': lower_band.tolist()}
            
        except ImportError:
            logger.warning("Pandas not found, falling back to slower deviation band calculation.")
            # Fallback to slower method if pandas is not available
            upper_band, lower_band = [], []
            for i in range(len(data)):
                start_idx = max(0, i - window_size + 1)
                window_data = data[start_idx:i+1]
                mean_val = np.mean(window_data)
                threshold_val = mean_val * (threshold_percent / 100)
                upper_band.append(mean_val + threshold_val)
                lower_band.append(mean_val - threshold_val)
            return {'upper': upper_band, 'lower': lower_band}

    def _calculate_red_segments(self, data: np.ndarray, settings: Dict) -> List[Dict[str, Any]]:
        """
        [DEPRECATED] Calculate red segments where signal exceeds threshold.
        Replaced by vectorized version in _calculate_basic_deviation.
        """
        if len(data) < 2:
            return []
            
        window_size = settings.get('fluctuation_detection', {}).get('sample_window', 20)
        threshold_percent = settings.get('fluctuation_detection', {}).get('threshold_percent', 10.0)

        # Use pandas for efficient rolling window calculations
        try:
            import pandas as pd
            s = pd.Series(data)
            rolling_mean = s.rolling(window=window_size, min_periods=1).mean().to_numpy()
            
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-9
            deviation_percent = np.abs((data - rolling_mean) / (rolling_mean + epsilon)) * 100
            
            # Find indices where deviation exceeds threshold
            exceeds_threshold = deviation_percent > threshold_percent
            
            # Find start and end of consecutive True blocks
            diff = np.diff(exceeds_threshold.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]
            
            # Handle cases where the series starts or ends with an exceedance
            if exceeds_threshold[0]:
                starts = np.insert(starts, 0, 0)
            if exceeds_threshold[-1]:
                ends = np.append(ends, len(data) - 1)
                
            red_segments = []
            for start, end in zip(starts, ends):
                if end > start:
                    red_segments.append({
                        'start_index': int(start),
                        'end_index': int(end),
                        'values': data[start:end+1].tolist(),
                        'deviation_percent': np.max(deviation_percent[start:end+1])
                    })
            return red_segments
            
        except ImportError:
            logger.warning("Pandas not found, falling back to slower red segment calculation.")
            # Fallback to the original slower method
            return self._calculate_red_segments_slow(data, settings)

    def _calculate_red_segments_slow(self, data: np.ndarray, settings: Dict) -> List[Dict[str, Any]]:
        """[DEPRECATED] Original, slower implementation for calculating red segments."""
        window_size = settings.get('fluctuation_detection', {}).get('sample_window', 20)
        threshold_percent = settings.get('fluctuation_detection', {}).get('threshold_percent', 10.0)
        
        red_segments = []
        current_segment = None
        
        for i in range(window_size, len(data)):
            window_data = data[i-window_size:i]
            window_mean = np.mean(window_data)
            current_value = data[i]
            
            if window_mean != 0:
                deviation_percent = abs((current_value - window_mean) / window_mean) * 100
                if deviation_percent > threshold_percent:
                    if current_segment is None:
                        current_segment = {'start_index': i, 'end_index': i, 'values': [current_value], 'deviation_percent': deviation_percent}
                    else:
                        current_segment['end_index'] = i
                        current_segment['values'].append(current_value)
                        current_segment['deviation_percent'] = max(current_segment['deviation_percent'], deviation_percent)
                else:
                    if current_segment is not None:
                        red_segments.append(current_segment)
                        current_segment = None
            else:
                if current_segment is not None:
                    red_segments.append(current_segment)
                    current_segment = None
                    
        if current_segment is not None:
            red_segments.append(current_segment)
            
        return red_segments

class GraphRenderer:
    """Handles different graph rendering modes for filtered data."""
    
    def __init__(self, signal_processor, graph_signal_mapping, parent_widget=None):
        self.signal_processor = signal_processor
        self.graph_signal_mapping = graph_signal_mapping
        self.parent_widget = parent_widget  # Reference to TimeGraphWidget
        self.deviation_threads = {} # Store active threads
        self.deviation_workers = {} # Store active workers
        self.limit_lines = {}  # Store limit line references for removal
        self.limits_config = {}  # Store limits configuration
        self.deviation_lines = {}  # Store deviation visualization references
        self.basic_deviation_settings = {}  # Store basic deviation settings per graph
        self._is_destroyed = False
        
        # Limit violation threaded calculation support
        self.limit_violation_threads = {}  # Store active limit violation threads
        self.limit_violation_workers = {}  # Store active limit violation workers
        self._pending_violation_results = {}  # Store pending results for batch rendering
        
        # Downsample thresholds to keep UI responsive
        self.max_points_for_analysis = 200_000
        self.chunk_size_streaming = 1_000_000  # Align with C++ streaming chunk
        
        # TODO: Gelecekte performans optimizasyonu için cache eklenecek
        # self._violation_cache = {}  # Violation hesaplamalarını cache'ler
        # self._limits_cache = {}  # Limit konfigürasyonlarını cache'ler
    
    def set_static_limits(self, graph_index: int, limits: Dict[str, Any]):
        """Store static limits for a specific graph."""
        if not self.limits_config:
            self.limits_config = {}
        self.limits_config[graph_index] = limits
        logger.info(f"Updated static limits for graph {graph_index}: {limits}")

    def __del__(self):
        """Destructor - ensure all threads are cleaned up."""
        if not self._is_destroyed:
            logger.warning("GraphRenderer being destroyed without proper cleanup!")
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources and threads."""
        if self._is_destroyed:
            return
        self._is_destroyed = True
        logger.info("GraphRenderer cleanup started")
        
        # Cleanup limit violation threads
        for worker_key in list(self.limit_violation_workers.keys()):
            self._cancel_limit_violation_calculation(worker_key)
        self.limit_violation_workers.clear()
        self.limit_violation_threads.clear()
        
        self.clear_all_deviation_lines()
        logger.info("GraphRenderer cleanup completed")
    
<<<<<<< HEAD
    def apply_segmented_filter(self, container, graph_index: int, time_segments: List[Tuple[float, float]], tab_index: int = 0, filter_conditions: list = None):
        """Apply segmented display filter - show matching segments with gaps.
        
        Args:
            container: GraphContainer to apply filter to
            graph_index: Index of the graph to apply filter to
            time_segments: List of (start_time, end_time) tuples from C++ calculate_streaming
            tab_index: Tab index for signal mapping
            filter_conditions: Python dict conditions for REQUIRED NumPy Y-value filtering.
                             C++ segments give time ranges where SOME points match, we need to filter further.
        
        IMPORTANT: C++ calculate_streaming returns time segments where SOME points pass the filter.
                   After loading data from those segments, we MUST apply Y-value mask to show only
                   the points that actually match the filter conditions.
        """
        logger.info(f"[SEGMENTED] Starting segmented filter application")
        logger.info(f"[SEGMENTED] Graph index: {graph_index}, Tab index: {tab_index}")
        logger.info(f"[SEGMENTED] Time segments: {len(time_segments)} segments")
        logger.info(f"[SEGMENTED] Filter conditions: {filter_conditions}")
=======
    def apply_segmented_filter(self, container, graph_index: int, time_segments: List[Tuple[float, float]], tab_index: int = 0):
        """Apply segmented display filter - show matching segments with gaps."""
        logger.debug(f"[SEGMENTED DEBUG] Starting segmented filter application")
        logger.debug(f"[SEGMENTED DEBUG] Graph index: {graph_index}, Tab index: {tab_index}")
        logger.debug(f"[SEGMENTED DEBUG] Time segments: {len(time_segments)} segments")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        
        # Get visible signals for the CORRECT tab!
        visible_signals = self._get_visible_signals_for_graph(tab_index, graph_index)
        
        # If no visible signals found, try to get all signals for this graph
        if not visible_signals:
            all_signals = self.signal_processor.get_all_signals()
            visible_signals = list(all_signals.keys())
        
<<<<<<< HEAD
        logger.debug(f"[SEGMENTED] Target tab: {tab_index}, Visible signals: {visible_signals}")
        
        if not visible_signals:
            logger.warning(f"[SEGMENTED] No visible signals for graph {graph_index}")
=======
        logger.debug(f"[SEGMENTED DEBUG] Target tab: {tab_index}")
        logger.debug(f"[SEGMENTED DEBUG] Visible signals: {visible_signals}")
        
        if not visible_signals:
            logger.warning(f"[SEGMENTED DEBUG] No visible signals for graph {graph_index}")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            return
        
        # Get plot widget and clear it
        plot_widgets = container.plot_manager.get_plot_widgets()
<<<<<<< HEAD
        
        if graph_index < len(plot_widgets):
            plot_widget = plot_widgets[graph_index]
            plot_widget.clear()
        else:
            logger.warning(f"[SEGMENTED] Graph index {graph_index} out of range, available plots: {len(plot_widgets)}")
=======
        logger.debug(f"[SEGMENTED DEBUG] Available plot widgets: {len(plot_widgets)}")
        
        if graph_index < len(plot_widgets):
            plot_widget = plot_widgets[graph_index]
            logger.debug(f"[SEGMENTED DEBUG] Clearing plot widget {graph_index}")
            plot_widget.clear()
        else:
            logger.warning(f"[SEGMENTED DEBUG] Graph index {graph_index} out of range, available plots: {len(plot_widgets)}")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            return
        
        # Get all signals data
        all_signals = self.signal_processor.get_all_signals()
        
        # Process each visible signal
        for signal_name in visible_signals:
<<<<<<< HEAD
            logger.info(f"[SEGMENTED] Processing signal: {signal_name}")
            
            if signal_name not in all_signals:
                logger.warning(f"[SEGMENTED] Signal {signal_name} not found in all_signals")
=======
            logger.info(f"[SEGMENTED DEBUG] Processing signal: {signal_name}")
            
            if signal_name not in all_signals:
                logger.warning(f"[SEGMENTED DEBUG] Signal {signal_name} not found in all_signals")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                continue
            
            full_x_data = np.array(all_signals[signal_name]['x_data'])
            full_y_data = np.array(all_signals[signal_name]['y_data'])

            raw_df = getattr(self.signal_processor, "raw_dataframe", None)
            time_col = getattr(self.signal_processor, "time_column_name", None)
            metadata = all_signals[signal_name].get("metadata", {})
            full_count = metadata.get("full_count", len(full_x_data))

<<<<<<< HEAD
            logger.info(f"[SEGMENTED] Signal '{signal_name}': preview={len(full_x_data)} pts, full={full_count} pts")
=======
            limits_config = self._get_limits_configuration(graph_index)
            limits = limits_config.get(signal_name) if limits_config else None

            logger.info(f"[SEGMENTED DEBUG] Signal data length (preview): {len(full_x_data)}")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            
            # Create optimized segmented data with proper NaN handling
            segmented_x = []
            segmented_y = []
            segments_found = 0
            
            # Sort segments by start time to ensure proper ordering
            sorted_segments = sorted(time_segments, key=lambda x: x[0])
            
            for i, (segment_start, segment_end) in enumerate(sorted_segments):
                try:
<<<<<<< HEAD
                    segment_x = None
                    segment_y = None
                    
                    # ✅ FIX: Check raw_df type directly instead of metadata['mpai'] flag
                    # Sometimes metadata doesn't have 'mpai' key even when using MpaiReader
                    is_mpai = hasattr(raw_df, "load_column_slice") and hasattr(raw_df, "get_row_count")
                    
                    # Load segment from MPAI file by time range
                    if is_mpai and time_col:
                        # Estimate sample rate from metadata or time range
                        sample_rate = 1.0
                        start_time_meta = metadata.get("start_time", 0.0)
                        end_time_meta = metadata.get("end_time", 10.0)  # ✅ FIX: Better default
                        
=======
                    if metadata.get("mpai") and hasattr(raw_df, "load_column_slice") and time_col:
                        # Estimate sample rate from metadata or time range (avoid preview data dependence)
                        sample_rate = 1.0
                        start_time_meta = metadata.get("start_time", 0.0)
                        end_time_meta = metadata.get("end_time", 1.0)
                        
                        # Calculate sample rate from full count and duration
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                        duration = max(end_time_meta - start_time_meta, 1e-9)
                        if full_count > 1:
                             sample_rate = (full_count - 1) / duration
                        
<<<<<<< HEAD
=======
                        # Fallback: load first and last time points from disk if metadata unreliable
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                        if sample_rate == 1.0 and hasattr(raw_df, "load_column_slice"):
                             try:
                                 t_start = raw_df.load_column_slice(time_col, 0, 1)[0]
                                 t_end = raw_df.load_column_slice(time_col, full_count - 1, 1)[0]
                                 duration = max(t_end - t_start, 1e-9)
                                 sample_rate = (full_count - 1) / duration
                             except Exception:
<<<<<<< HEAD
                                 pass

                        start_row = max(0, int((segment_start - start_time_meta) * sample_rate))
                        end_row = min(full_count, int((segment_end - start_time_meta) * sample_rate))
                        row_count = max(1, end_row - start_row)

                        segment_x = np.array(raw_df.load_column_slice(time_col, int(start_row), int(row_count)), dtype=np.float64)
                        segment_y = np.array(raw_df.load_column_slice(signal_name, int(start_row), int(row_count)), dtype=np.float64)
                        
                        # ⚠️ WARNING: This fallback loads ALL data in time range without Y filter!
                        # For proper filtering, C++ path is required
                        
                        # Optional: Apply smart downsampling ONLY if segment is too large (>1M points)
=======
                                 pass # Keep default or existing sample_rate

                        start_row = max(0, int((segment_start - start_time_meta) * sample_rate))
                        end_row = min(full_count, int((segment_end - start_time_meta) * sample_rate))
                        
                        # Safety alignment with time column search if needed (more precise but slower):
                        # For now, linear mapping is fast and usually sufficient for segmented view.
                        
                        row_count = max(1, end_row - start_row)

                        # ✅ FIX: NO downsampling for segmented filter!
                        # Segmented filter shows filtered segments - user expects FULL data
                        # Downsampling here causes 10k point limitation bug
                        segment_x = np.array(raw_df.load_column_slice(time_col, int(start_row), int(row_count)), dtype=np.float64)
                        segment_y = np.array(raw_df.load_column_slice(signal_name, int(start_row), int(row_count)), dtype=np.float64)
                        
                        # Optional: Apply smart downsampling ONLY if segment is too large (>1M points)
                        # This prevents UI freeze with massive segments
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                        max_segment_points = 1_000_000
                        if len(segment_x) > max_segment_points:
                            logger.warning(f"[SEGMENTED] Segment too large ({len(segment_x)} points), downsampling to {max_segment_points}")
                            stride = int(np.ceil(len(segment_x) / max_segment_points))
                            segment_x = segment_x[::stride]
                            segment_y = segment_y[::stride]
                    else:
                        # ❌ OLD: Preview-based slicing (causes 10k point limitation)
                        # Preview data is only 10k points - need full data for segmented filter!
                        # mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                        # segment_indices = np.where(mask)[0]
                        # segment_x = full_x_data[segment_indices]
                        # segment_y = full_y_data[segment_indices]
                        
                        # ✅ NEW: Load full segment data from MPAI if available
                        if metadata.get("mpai") and hasattr(raw_df, "load_column_slice") and time_col:
                            # Use same logic as MPAI path above
                            sample_rate = 1.0
                            start_time_meta = metadata.get("start_time", 0.0)
                            end_time_meta = metadata.get("end_time", 1.0)
                            
                            duration = max(end_time_meta - start_time_meta, 1e-9)
                            if full_count > 1:
                                sample_rate = (full_count - 1) / duration
                            
                            if sample_rate == 1.0 and hasattr(raw_df, "load_column_slice"):
                                try:
                                    t_start = raw_df.load_column_slice(time_col, 0, 1)[0]
                                    t_end = raw_df.load_column_slice(time_col, full_count - 1, 1)[0]
                                    duration = max(t_end - t_start, 1e-9)
                                    sample_rate = (full_count - 1) / duration
                                except Exception:
                                    pass
                            
                            start_row = max(0, int((segment_start - start_time_meta) * sample_rate))
                            end_row = min(full_count, int((segment_end - start_time_meta) * sample_rate))
                            row_count = max(1, end_row - start_row)
                            
                            # ✅ FIX: Load WITHOUT downsampling for segmented filter!
                            # User wants to see the filtered segments in FULL detail
                            segment_x = np.array(raw_df.load_column_slice(time_col, int(start_row), int(row_count)), dtype=np.float64)
                            segment_y = np.array(raw_df.load_column_slice(signal_name, int(start_row), int(row_count)), dtype=np.float64)
                        else:
                            # Fallback: Use preview data if MPAI not available
                            mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                            segment_indices = np.where(mask)[0]
                            segment_x = full_x_data[segment_indices]
                            segment_y = full_y_data[segment_indices]
                    
                    if len(segment_x) > 0:
<<<<<<< HEAD
                        # ✅ CRITICAL: Apply Y-value filter using NumPy mask
                        # C++ calculate_streaming returns time ranges where SOME points pass
                        # We must filter to show ONLY points that actually match the condition
                        if filter_conditions:
                            y_mask = np.ones(len(segment_y), dtype=bool)
                            for cond in filter_conditions:
                                param_name = cond.get('parameter', '')
                                # Only apply condition if this signal matches the parameter
                                if param_name == signal_name:
                                    for range_filter in cond.get('ranges', []):
                                        operator = range_filter.get('operator', '')
                                        value = range_filter.get('value', 0.0)
                                        
                                        if operator == '>=':
                                            y_mask &= (segment_y >= value)
                                        elif operator == '>':
                                            y_mask &= (segment_y > value)
                                        elif operator == '<=':
                                            y_mask &= (segment_y <= value)
                                        elif operator == '<':
                                            y_mask &= (segment_y < value)
                                            
                            # Apply the mask to filter points
                            filtered_x = segment_x[y_mask]
                            filtered_y = segment_y[y_mask]
                            logger.info(f"[SEGMENTED] Y-filter applied: {len(segment_x)} -> {len(filtered_x)} points ({signal_name})")
                            segment_x = filtered_x
                            segment_y = filtered_y
                        
                        # Log segment details after filtering
                        logger.info(f"[SEGMENTED] Segment {i+1}/{len(sorted_segments)}: {len(segment_x)} points (time: {segment_start:.2f}-{segment_end:.2f})")
                        
                        if len(segment_x) > 0:
                            if segments_found > 0:
                                segmented_x.append(np.nan)
                                segmented_y.append(np.nan)
                            segmented_x.extend(segment_x)
                            segmented_y.extend(segment_y)
                            segments_found += 1
=======
                        # ✅ DEBUG: Log segment details
                        logger.info(f"[SEGMENTED DEBUG] Segment {i+1}/{len(sorted_segments)}: loaded {len(segment_x)} points (time range: {segment_start:.2f}-{segment_end:.2f})")
                        
                        if segments_found > 0:
                            segmented_x.append(np.nan)
                            segmented_y.append(np.nan)
                        segmented_x.extend(segment_x)
                        segmented_y.extend(segment_y)
                        segments_found += 1
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                except Exception as e:
                    logger.warning(f"[SEGMENTED DEBUG] Segment load failed for {signal_name}: {e}")
                    continue
            
            # Plot all segments as a single line with NaN breaks
            if segmented_x:
                color = self._get_signal_color(signal_name)
                
                # Convert to numpy arrays for better performance
                x_array = np.array(segmented_x)
                y_array = np.array(segmented_y)
                
                # ✅ DEBUG: Log total points
                logger.info(f"[SEGMENTED DEBUG] TOTAL POINTS COLLECTED: {len(x_array)} (including NaN separators)")
                nan_count = np.sum(np.isnan(x_array))
                logger.info(f"[SEGMENTED DEBUG] NaN separators: {nan_count}, Actual data points: {len(x_array) - nan_count}")
                
                # Use PyQtGraph's PlotDataItem for better control
                from pyqtgraph import PlotDataItem
                plot_item = PlotDataItem(
                    x=x_array,
                    y=y_array,
                    pen=color,
                    name=signal_name,
                    connect='finite',  # Only connect finite values, skip NaN
                    skipFiniteCheck=False,  # Ensure NaN handling works properly
                    # ✅ FIX: Disable PyQtGraph auto-downsampling for segmented filter
                    # User filtered data to see FULL detail, not downsampled preview
                    downsample=False,
                    autoDownsample=False
                )
                plot_widget.addItem(plot_item)
<<<<<<< HEAD

                logger.info(f"[SEGMENTED DEBUG] Signal {signal_name}: plotted {segments_found} segments as optimized PlotDataItem")
            else:
                logger.warning(f"[SEGMENTED DEBUG] No valid segments found for signal {signal_name}")

        logger.info(f"Segmented filter applied successfully to graph {graph_index}")

        # Apply limit lines if available
        self._apply_limit_lines(plot_widget, graph_index, visible_signals)

        # ✅ CRITICAL FIX: Disable LOD engine temporarily during auto-range
        # Auto-range triggers sigRangeChanged which causes PlotManager to reload data from MPAI,
        # overriding our filtered segments with downsampled full dataset!
        logger.info(f"[SEGMENTED FIX] Disabling LOD updates via PlotManager")

        # Set a flag on container's plot_manager to skip LOD updates
        plot_manager = getattr(container, 'plot_manager', None)
        if plot_manager:
            # Temporarily disable LOD processing
            plot_manager._lod_disabled = True
            logger.info(f"[SEGMENTED FIX] Set _lod_disabled=True on PlotManager")

        try:
            # NOW perform auto-range WITHOUT triggering LOD
            logger.debug(f"[VIEW FIX] Auto-ranging plot {graph_index} after segmented filter (LOD disabled)")
            plot_widget.enableAutoRange(axis='x', enable=True)
            plot_widget.enableAutoRange(axis='y', enable=True)
            plot_widget.autoRange()
            plot_widget.enableAutoRange(axis='x', enable=False)
            plot_widget.enableAutoRange(axis='y', enable=False)
            logger.info(f"[SEGMENTED FIX] Auto-range completed without LOD interference")

        finally:
            # IMPORTANT: Keep LOD disabled even after auto-range
            # Filtered data should NOT be dynamically reloaded from MPAI!
            # User can clear filter to restore normal LOD behavior
            logger.info(f"[SEGMENTED FIX] Keeping LOD disabled for filtered view (will re-enable on filter clear)")
    
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

            for i, (segment_start, segment_end) in enumerate(time_segments):
                segment_x = None
                segment_y = None

                # ✅ SIMPLIFIED: Load segment by time range only
                # time_segments are already correctly filtered by C++ calculate_streaming
                if mpai_reader and time_col and metadata.get("mpai"):
                    try:
                        sample_rate = 1.0
                        start_time_meta = metadata.get("start_time", 0.0)
                        end_time_meta = metadata.get("end_time", 1.0)

                        duration = max(end_time_meta - start_time_meta, 1e-9)
                        if full_count > 1:
                            sample_rate = (full_count - 1) / duration

                        start_row = max(0, int((segment_start - start_time_meta) * sample_rate))
                        end_row = min(full_count, int((segment_end - start_time_meta) * sample_rate))
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

=======
                
                logger.info(f"[SEGMENTED DEBUG] Signal {signal_name}: plotted {segments_found} segments as optimized PlotDataItem")
            else:
                logger.warning(f"[SEGMENTED DEBUG] No valid segments found for signal {signal_name}")
                    
        logger.info(f"Segmented filter applied successfully to graph {graph_index}")
        
        # Apply limit lines if available
        self._apply_limit_lines(plot_widget, graph_index, visible_signals)
        
        # ✅ FIX Problem #8: Auto-range view after filter to show all data
        logger.debug(f"[VIEW FIX] Auto-ranging plot {graph_index} after segmented filter")
        plot_widget.enableAutoRange(axis='x', enable=True)
        plot_widget.enableAutoRange(axis='y', enable=True)
        plot_widget.autoRange()
        plot_widget.enableAutoRange(axis='x', enable=False)
        plot_widget.enableAutoRange(axis='y', enable=False)
    
    def apply_concatenated_filter(self, container, time_segments: List[Tuple[float, float]]):
        """Apply concatenated display filter - create continuous timeline from filtered segments."""
        logger.info(f"[CONCATENATED DEBUG] Starting concatenated filter application")
        logger.info(f"[CONCATENATED DEBUG] Time segments: {len(time_segments)} segments")
        
        # Get all signals data
        all_signals = self.signal_processor.get_all_signals()
        
        # Create concatenated time and value arrays with continuous timeline
        concatenated_data = {}
        
        for signal_name, signal_data in all_signals.items():
            full_x_data = np.array(signal_data['x_data'])
            full_y_data = np.array(signal_data['y_data'])
            
            concat_x = []
            concat_y = []
            current_time_offset = 0.0
            
            for i, (segment_start, segment_end) in enumerate(time_segments):
                mask = (full_x_data >= segment_start) & (full_x_data <= segment_end)
                segment_x = full_x_data[mask]
                segment_y = full_y_data[mask]
                
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                if len(segment_x) > 0:
                    # Create continuous timeline by adjusting time values
                    if i == 0:
                        # First segment starts at 0
                        adjusted_x = segment_x - segment_x[0]
                        current_time_offset = adjusted_x[-1] if len(adjusted_x) > 0 else 0
                    else:
                        # Subsequent segments continue from where previous ended
                        segment_duration = segment_x[-1] - segment_x[0] if len(segment_x) > 1 else 0
<<<<<<< HEAD
                        adjusted_x = np.linspace(current_time_offset,
                                               current_time_offset + segment_duration,
                                               len(segment_x))
                        current_time_offset = adjusted_x[-1] if len(adjusted_x) > 0 else current_time_offset

                    concat_x.extend(adjusted_x)
                    concat_y.extend(segment_y)

=======
                        adjusted_x = np.linspace(current_time_offset, 
                                               current_time_offset + segment_duration, 
                                               len(segment_x))
                        current_time_offset = adjusted_x[-1] if len(adjusted_x) > 0 else current_time_offset
                    
                    concat_x.extend(adjusted_x)
                    concat_y.extend(segment_y)
            
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            if concat_x:
                concatenated_data[signal_name] = {
                    'time': np.array(concat_x),
                    'values': np.array(concat_y)
                }
<<<<<<< HEAD
                logger.info(f"[CONCATENATED FIX] Signal '{signal_name}': {len(concat_x)} total points, "
                           f"time range: {concat_x[0]:.3f} - {concat_x[-1]:.3f}")

        # Update signal processor with concatenated data
        self.signal_processor.set_filtered_data(concatenated_data)
        logger.info(f"[CONCATENATED FIX] Updated signal processor with concatenated data")

        # NOT: Grafik redraw'ı TimeGraphWidget._redraw_all_signals() tarafından yapılacak
        # container.plot_manager.redraw_all_plots() yeterli değil - sadece repaint yapıyor

        logger.info(f"[CONCATENATED FIX] Concatenated filter applied successfully - continuous timeline created")
=======
                logger.debug(f"[CONCATENATED DEBUG] Signal '{signal_name}': {len(concat_x)} points, "
                           f"time range: {concat_x[0]:.3f} - {concat_x[-1]:.3f}")
        
        # Update signal processor with concatenated data
        self.signal_processor.set_filtered_data(concatenated_data)
        logger.info(f"[CONCATENATED DEBUG] Updated signal processor with concatenated data")
        
        # NOT: Grafik redraw'ı TimeGraphWidget._redraw_all_signals() tarafından yapılacak
        # container.plot_manager.redraw_all_plots() yeterli değil - sadece repaint yapıyor
        
        logger.info(f"Concatenated filter applied successfully - continuous timeline created")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    
    def clear_filters(self, container, graph_index: int):
        """Clear all filters and restore original data display."""
        logger.info(f"[CLEAR DEBUG] Clearing filters for graph {graph_index}")
<<<<<<< HEAD

        try:
            # ✅ CRITICAL FIX: Re-enable LOD engine when clearing filter
            # Filter is being removed, normal LOD behavior should resume
            plot_manager = getattr(container, 'plot_manager', None)
            if plot_manager:
                plot_manager._lod_disabled = False
                logger.info(f"[CLEAR FIX] Re-enabled LOD engine (_lod_disabled=False)")

            # Restore original data in signal processor (important for concatenated display)
            self.signal_processor.restore_original_data()
            logger.info(f"[CLEAR DEBUG] Restored original data in signal processor")

            # Clear all plots and redraw with original data
            container.plot_manager.clear_all_signals()

            # Trigger redraw of all graphs with original data
            container.plot_manager.redraw_all_plots()

            # Apply limit lines to all plots
            self._apply_limit_lines_to_all_plots(container)

=======
        
        try:
            # Restore original data in signal processor (important for concatenated display)
            self.signal_processor.restore_original_data()
            logger.info(f"[CLEAR DEBUG] Restored original data in signal processor")
            
            # Clear all plots and redraw with original data
            container.plot_manager.clear_all_signals()
            
            # Trigger redraw of all graphs with original data
            container.plot_manager.redraw_all_plots()
            
            # Apply limit lines to all plots
            self._apply_limit_lines_to_all_plots(container)
            
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            logger.info(f"[CLEAR DEBUG] Successfully cleared filters and restored original data")
                
        except Exception as e:
            logger.error(f"Error clearing filters: {e}")
    
    def _get_visible_signals_for_graph(self, tab_index: int, graph_index: int) -> List[str]:
        """Get visible signals for a specific graph."""
        if tab_index in self.graph_signal_mapping and graph_index in self.graph_signal_mapping[tab_index]:
            return self.graph_signal_mapping[tab_index][graph_index]
        return []
    
    def _get_signal_color(self, signal_name: str) -> str:
        """Get color for a signal - consistent with theme manager."""
        # ✅ FIX Problem #9: Use theme manager for consistent colors
        # Bu renk statistic panel ve legend'larla uyumlu olmalı
        
        # Try to get signal index from all signals
        all_signals = self.signal_processor.get_all_signals()
        signal_names = sorted(list(all_signals.keys()))
        
        try:
            signal_index = signal_names.index(signal_name)
        except ValueError:
            # Signal not found, use hash-based fallback
            signal_index = hash(signal_name) % 20
        
        # Use theme manager colors if available via parent_widget
        if hasattr(self, 'parent_widget') and self.parent_widget:
            if hasattr(self.parent_widget, 'theme_manager') and self.parent_widget.theme_manager:
                return self.parent_widget.theme_manager.get_signal_color(signal_index)
        
        # Fallback to default colors
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
                  '#ff8800', '#88ff00', '#0088ff', '#ff0088', '#8800ff', '#00ff88']
        return colors[signal_index % len(colors)]
    
    def _apply_limit_lines(self, plot_widget, graph_index: int, visible_signals: List[str]):
        """Apply warning limit lines to the plot."""
        try:
            # Get limit configuration from parent widget if available
            limits_config = self._get_limits_configuration(graph_index)
            if not limits_config:
                return
            
            # Clear existing limit lines for this plot
            self._clear_limit_lines(plot_widget, graph_index)
            
            # Get plot view range for drawing lines across the full width
            view_box = plot_widget.getViewBox()
            if view_box is None:
                return
                
            # Get current view range
            x_range, _ = view_box.viewRange()
            x_min, x_max = x_range
            
            # Draw limit lines for each signal that has limits and is visible
            for signal_name in visible_signals:
                if signal_name in limits_config:
                    limits = limits_config[signal_name]
                    self._draw_signal_limit_lines(plot_widget, graph_index, signal_name, limits, x_min, x_max)
                    
        except Exception as e:
            logger.error(f"Error applying limit lines: {e}")
    
    def _draw_signal_limit_lines(self, plot_widget, graph_index: int, signal_name: str, limits: Dict[str, float], x_min: float, x_max: float):
        """Draw warning limit lines for a specific signal."""
        try:
            warning_min = limits.get('warning_min', 0.0)
            warning_max = limits.get('warning_max', 0.0)
            
            # Create dashed pen for limit lines - more visible with custom dash pattern
            limit_pen = pg.mkPen(color='#FFA500', width=3, style=pg.QtCore.Qt.CustomDashLine)
            limit_pen.setDashPattern([8, 4])  # 8 pixels dash, 4 pixels gap
            
            # Store limit lines for later removal
            limit_key = f"{graph_index}_{signal_name}"
            if limit_key not in self.limit_lines:
                self.limit_lines[limit_key] = []
            
            # Draw warning min line (always draw if limit is configured)
            min_line = pg.InfiniteLine(pos=warning_min, angle=0, pen=limit_pen, 
                                     label=f'{signal_name} Min Warning: {warning_min:.2f}',
                                     labelOpts={'position': 0.1, 'color': '#FFA500'})
            plot_widget.addItem(min_line)
            self.limit_lines[limit_key].append(min_line)
            
            # Draw warning max line (always draw if limit is configured)
            max_line = pg.InfiniteLine(pos=warning_max, angle=0, pen=limit_pen,
                                     label=f'{signal_name} Max Warning: {warning_max:.2f}',
                                     labelOpts={'position': 0.9, 'color': '#FFA500'})
            plot_widget.addItem(max_line)
            self.limit_lines[limit_key].append(max_line)
                
            # Highlight violations if signal data is available
            try:
                self._highlight_limit_violations(plot_widget, graph_index, signal_name, limits)
            except RuntimeError as runtime_err:
                # _highlight_limit_violations already showed error dialog to user
                # Just log and continue - don't crash the app
                logger.warning(f"[LIMIT_VIOLATION] Could not highlight violations for {signal_name}: {runtime_err}")
            except Exception as violation_err:
                # Unexpected error in violation detection
                logger.error(f"[LIMIT_VIOLATION] Unexpected error highlighting violations for {signal_name}: {violation_err}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error drawing limit lines for {signal_name}: {e}", exc_info=True)
    
    def _highlight_limit_violations(self, plot_widget, graph_index: int, signal_name: str, limits: Dict[str, float]):
        """
        Highlight areas where signal violates limits with dashed lines.
        CRITICAL: C++ SIMD acceleration is REQUIRED - checks ALL data points without downsampling.
        Python fallback is DISABLED to prevent missed violations.
        """
        try:
            worker_key = f"{graph_index}_{signal_name}_violation"
            
            # Cancel any existing calculation for this signal
            self._cancel_limit_violation_calculation(worker_key)
            
            warning_min = limits.get('warning_min', 0.0)
            warning_max = limits.get('warning_max', 0.0)
            
            # Check if C++ module is available
            if not CPP_AVAILABLE:
                error_msg = (
                    "C++ module not available!\n\n"
                    "Static Limits requires C++ SIMD acceleration to check ALL data points.\n"
                    "Python fallback would miss violations due to downsampling.\n\n"
                    "Please compile the C++ module:\n"
                    "  cd cpp/build\n"
                    "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
                    "  cmake --build . --config Release\n"
                    "  Copy time_graph_cpp.*.pyd to project root"
                )
                logger.error(f"[LIMIT_VIOLATION] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Check if we have MPAI reader for streaming
            reader = getattr(self.signal_processor, 'raw_dataframe', None)
            time_col_name = getattr(self.signal_processor, 'time_column_name', None)
            is_mpai = hasattr(reader, 'get_header')
            
            # Auto-detect time column if not set
            if is_mpai and not time_col_name and hasattr(reader, 'get_column_names'):
                available_columns = reader.get_column_names()
                # Find column with 'time' in name (case-insensitive)
                for col in available_columns:
                    if 'time' in col.lower():
                        time_col_name = col
                        logger.info(f"[LIMIT_VIOLATION] Auto-detected time column: '{time_col_name}'")
                        break
                
                # If still not found, use first column
                if not time_col_name and available_columns:
                    time_col_name = available_columns[0]
                    logger.warning(f"[LIMIT_VIOLATION] No time column found, using first column: '{time_col_name}'")
            
            if not is_mpai:
                error_msg = (
                    "MPAI format required for Static Limits!\n\n"
                    "Static Limits requires MPAI format to stream ALL data points.\n"
                    "CSV files are automatically converted to MPAI on load.\n\n"
                    "If you see this error, the file was not properly converted."
                )
                logger.error(f"[LIMIT_VIOLATION] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Use C++ streaming - checks ALL data points from disk!
            logger.info(f"[LIMIT_VIOLATION] Using C++ SIMD streaming for {signal_name} (checks ALL data)")
            
            try:
                engine = tgcpp.LimitViolationEngine()
            except AttributeError as e:
                error_msg = (
                    "C++ module outdated or incomplete!\n\n"
                    "LimitViolationEngine not found in C++ module.\n"
                    "Please recompile the C++ module with latest code:\n\n"
                    "  cd cpp\n"
                    "  Remove-Item -Recurse -Force build\n"
                    "  mkdir build\n"
                    "  cd build\n"
                    "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
                    "  cmake --build . --config Release\n"
                    "  cd ..\n"
                    "  Copy-Item build/Release/time_graph_cpp.*.pyd ..\n\n"
                    f"Error details: {e}"
                )
                logger.error(f"[LIMIT_VIOLATION] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Call C++ engine (it does its own internal chunking for memory efficiency)
            logger.info(f"[LIMIT_VIOLATION] Calling C++ SIMD engine with parameters:")
            logger.info(f"  signal='{signal_name}', time='{time_col_name}'")
            logger.info(f"  min={warning_min}, max={warning_max}")
            
            try:
<<<<<<< HEAD
                # =====================================================
                # ARROW BRIDGE: Load data from Python reader, pass to C++
                # This avoids the type mismatch (Python MpaiDirectoryReader vs C++ MpaiReader)
                # =====================================================
                logger.info(f"[LIMIT_VIOLATION] Starting C++ calculation via Arrow bridge...")
                
                # Load data from Python reader
                row_count = reader.get_row_count()
                
                # Load time data
                try:
                    time_data = reader.load_column_slice(time_col_name, 0, row_count)
                    import numpy as np
                    if isinstance(time_data, list):
                        time_data = np.array(time_data, dtype=np.float64)
                except Exception as e:
                    logger.warning(f"[LIMIT_VIOLATION] Could not load time column '{time_col_name}', generating synthetic: {e}")
                    import numpy as np
                    t_start, t_end = reader.get_time_range()
                    time_data = np.linspace(t_start, t_end, row_count, dtype=np.float64)
                
                # Load signal data  
                signal_data = reader.load_column_slice(signal_name, 0, row_count)
                if isinstance(signal_data, list):
                    signal_data = np.array(signal_data, dtype=np.float64)
                
                logger.info(f"[LIMIT_VIOLATION] Loaded {len(signal_data)} points from Python reader")
                
                # Call C++ with arrays (works with any reader type)
                result = engine.calculate_violations_arrays(
                    signal_data,
                    time_data, 
                    warning_min,
                    warning_max
=======
                # Let C++ handle the entire file with its internal 1M-row chunking
                # This avoids double-chunking issues and is optimized for SIMD
                logger.info(f"[LIMIT_VIOLATION] Starting C++ streaming calculation...")
                
                result = engine.calculate_violations_streaming(
                    reader,
                    signal_name,
                    time_col_name,
                    warning_min,
                    warning_max
                    # start_row=0, row_count=0 (defaults) => process entire file
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                )
                
                logger.info(f"[LIMIT_VIOLATION] C++ calculation completed successfully")
                
                # Log results
                if result.violations:
                    logger.info(f"⚠ [LIMIT_VIOLATION] Found {len(result.violations)} violation segments in {signal_name}")
                    logger.info(f"  Total violation points: {result.total_violation_points}/{result.total_data_points}")
                    
<<<<<<< HEAD
                    # Draw segments - need to pass arrays for drawing since we have them
                    self._draw_violation_segments_from_arrays(
                        plot_widget, graph_index, signal_name, result, 
                        time_data, signal_data
                    )
=======
                    # Draw segments in batches to keep UI responsive
                    self._draw_cpp_violation_segments(plot_widget, graph_index, signal_name, result, reader, time_col_name)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                else:
                    logger.info(f"✓ [LIMIT_VIOLATION] No violations found in {signal_name}")
                    
            except AttributeError as attr_error:
                # C++ function not found or incompatible version
                error_msg = (
                    f"C++ module mismatch or outdated!\n\n"
                    f"Function 'calculate_violations_streaming' not found or incompatible.\n"
                    f"Please recompile the C++ module:\n\n"
                    f"  cd cpp\n"
                    f"  Remove-Item -Recurse -Force build\n"
                    f"  mkdir build; cd build\n"
                    f"  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
                    f"  cmake --build . --config Release\n\n"
                    f"Error: {attr_error}"
                )
                logger.error(f"[LIMIT_VIOLATION] {error_msg}", exc_info=True)
                # Don't raise - show warning to user instead
                from PyQt5.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("C++ Module Error")
                msg.setText("Static Limits kapalı")
                msg.setInformativeText("C++ modülü güncel değil. Lütfen yeniden derleyin.")
                msg.setDetailedText(error_msg)
                msg.exec_()
                return
                
            except RuntimeError as runtime_error:
                # C++ internal error (e.g., file I/O, memory)
                error_msg = f"C++ runtime error: {runtime_error}"
                logger.error(f"[LIMIT_VIOLATION] {error_msg}", exc_info=True)
                # Don't crash - show error to user
                from PyQt5.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Static Limits Hatası")
                msg.setText(f"Limit ihlali kontrolü başarısız: {signal_name}")
                msg.setInformativeText("C++ motoru bir hata ile karşılaştı.")
                msg.setDetailedText(str(runtime_error))
                msg.exec_()
                return
                
            except Exception as cpp_error:
                # Unknown error
                error_msg = f"Unexpected error in C++ violation detection: {cpp_error}"
                logger.error(f"[LIMIT_VIOLATION] {error_msg}", exc_info=True)
                # Don't crash - show generic error
                from PyQt5.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Static Limits Hatası")
                msg.setText(f"Beklenmeyen hata: {signal_name}")
                msg.setInformativeText("Limit kontrolü tamamlanamadı.")
                msg.setDetailedText(str(cpp_error))
                msg.exec_()
                return
            
        except RuntimeError:
            # Re-raise RuntimeError (our custom errors)
            raise
        except Exception as e:
            error_msg = f"Unexpected error in limit violation detection: {e}"
            logger.error(f"[LIMIT_VIOLATION] {error_msg}", exc_info=True)
            raise RuntimeError(error_msg)
    
    def _calculate_and_draw_violations_sync(self, plot_widget, graph_index: int, signal_name: str, 
                                            x_data: np.ndarray, y_data: np.ndarray, limits: Dict[str, float]):
        """DEPRECATED: Python fallback - no longer used."""
        raise RuntimeError("Python violation calculation is disabled! C++ SIMD is required.")
    
    def _on_violation_calculation_complete(self, result: dict, plot_widget, graph_index: int):
        """DEPRECATED: Python fallback - no longer used."""
        raise RuntimeError("Python violation calculation is disabled! C++ SIMD is required.")
    
    def _draw_cpp_violation_segments(self, plot_widget, graph_index: int, signal_name: str, cpp_result, reader, time_col_name):
        """Draw violation segments from C++ violation result with UI batching."""
        try:
            if not cpp_result.violations:
                return
            
            from PyQt5.QtWidgets import QApplication
            
            # Create violation highlight pen
            violation_pen = pg.mkPen(color='#FF0000', width=4, style=pg.QtCore.Qt.CustomDashLine)
            violation_pen.setDashPattern([6, 3])
            
            limit_key = f"{graph_index}_{signal_name}"
            if limit_key not in self.limit_lines:
                self.limit_lines[limit_key] = []
            
            total_segments = len(cpp_result.violations)
            segments_drawn = 0
            batch_size = 100  # Draw 100 segments at a time, then process events
            
            for i, segment in enumerate(cpp_result.violations):
                # Load this segment's data from disk
                start_idx = int(segment.start_index)
                end_idx = int(segment.end_index)
                segment_length = end_idx - start_idx + 1
                
                try:
                    time_data = reader.load_column_slice(time_col_name, start_idx, segment_length)
                    signal_data = reader.load_column_slice(signal_name, start_idx, segment_length)
                    
                    if len(time_data) > 0 and len(signal_data) > 0:
                        # Draw this violation segment
                        legend_name = None 
                        if i == 0:  # Only first segment gets legend
                             legend_name = f'{signal_name}_violation'

                        violation_line = plot_widget.plot(time_data, signal_data, pen=violation_pen, name=legend_name)
                        violation_line.setZValue(10)
                        self.limit_lines[limit_key].append(violation_line)
                        segments_drawn += 1
                        
                        # Process events every batch_size segments to keep UI responsive
                        if (i + 1) % batch_size == 0:
                            QApplication.processEvents()
                            
                except Exception as e:
                    logger.warning(f"[LIMIT_VIOLATION] Could not load segment {i} at index {start_idx}: {e}")
            
            logger.info(f"✓ [LIMIT_VIOLATION] Drew {segments_drawn}/{total_segments} violation segments for {signal_name}")
                    
        except Exception as e:
            logger.error(f"[LIMIT_VIOLATION] Error drawing C++ violations for {signal_name}: {e}", exc_info=True)
    
<<<<<<< HEAD
    def _draw_violation_segments_from_arrays(self, plot_widget, graph_index: int, signal_name: str, 
                                              cpp_result, time_data: np.ndarray, signal_data: np.ndarray):
        """Draw violation segments using pre-loaded arrays (Arrow bridge version)."""
        try:
            if not cpp_result.violations:
                return
            
            from PyQt5.QtWidgets import QApplication
            
            # Create violation highlight pen
            violation_pen = pg.mkPen(color='#FF0000', width=4, style=pg.QtCore.Qt.CustomDashLine)
            violation_pen.setDashPattern([6, 3])
            
            limit_key = f"{graph_index}_{signal_name}"
            if limit_key not in self.limit_lines:
                self.limit_lines[limit_key] = []
            
            total_segments = len(cpp_result.violations)
            segments_drawn = 0
            batch_size = 100
            
            for i, segment in enumerate(cpp_result.violations):
                start_idx = int(segment.start_index)
                end_idx = int(segment.end_index)
                
                # Use pre-loaded arrays instead of loading from disk
                if start_idx < len(time_data) and end_idx < len(signal_data):
                    seg_time = time_data[start_idx:end_idx + 1]
                    seg_signal = signal_data[start_idx:end_idx + 1]
                    
                    if len(seg_time) > 0 and len(seg_signal) > 0:
                        legend_name = f'{signal_name}_violation' if i == 0 else None
                        
                        violation_line = plot_widget.plot(seg_time, seg_signal, pen=violation_pen, name=legend_name)
                        violation_line.setZValue(10)
                        self.limit_lines[limit_key].append(violation_line)
                        segments_drawn += 1
                        
                        if (i + 1) % batch_size == 0:
                            QApplication.processEvents()
            
            logger.info(f"✓ [LIMIT_VIOLATION] Drew {segments_drawn}/{total_segments} violation segments for {signal_name}")
                    
        except Exception as e:
            logger.error(f"[LIMIT_VIOLATION] Error drawing violations from arrays for {signal_name}: {e}", exc_info=True)
    
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    def _draw_violation_segments(self, plot_widget, graph_index: int, signal_name: str, 
                                 violations: list, x_data: np.ndarray, y_data: np.ndarray):
        """DEPRECATED: Python fallback - no longer used."""
        raise RuntimeError("Python violation drawing is disabled! Use C++ SIMD results.")
    
    def _cancel_limit_violation_calculation(self, worker_key: str):
        """Cancel an ongoing limit violation calculation."""
        try:
            if worker_key in self.limit_violation_workers:
                worker = self.limit_violation_workers[worker_key]
                worker.stop()
                
            if worker_key in self.limit_violation_threads:
                thread = self.limit_violation_threads[worker_key]
                if thread.isRunning():
                    thread.quit()
                    thread.wait(100)  # Wait max 100ms
        except Exception as e:
            logger.debug(f"[LIMIT_VIOLATION] Error canceling calculation: {e}")
    
    def _cleanup_violation_thread(self, worker_key: str):
        """Cleanup completed violation calculation thread."""
        try:
            if worker_key in self.limit_violation_workers:
                del self.limit_violation_workers[worker_key]
            if worker_key in self.limit_violation_threads:
                del self.limit_violation_threads[worker_key]
        except Exception as e:
            logger.debug(f"[LIMIT_VIOLATION] Error cleaning up thread: {e}")
    
    def _group_consecutive_indices(self, indices: List[int]) -> List[List[int]]:
        """DEPRECATED: Python fallback - no longer used."""
        raise RuntimeError("Python violation grouping is disabled! C++ handles this.")
    
    def _clear_limit_lines(self, plot_widget, graph_index: int):
        """Clear existing limit lines for a specific graph."""
        try:
            keys_to_remove = [key for key in self.limit_lines.keys() if key.startswith(f"{graph_index}_")]
            
            for key in keys_to_remove:
                for line_item in self.limit_lines[key]:
                    try:
                        plot_widget.removeItem(line_item)
                    except:
                        pass  # Item might already be removed
                del self.limit_lines[key]
                
        except Exception as e:
            logger.error(f"Error clearing limit lines: {e}")

    def _get_limits_configuration(self, graph_index: int) -> Dict[str, Any]:
        """Get limits configuration for the specified graph from the stored config."""
        return self.limits_config.get(graph_index, {})

    def set_limits_configuration(self, limits_config: Dict[str, Any]):
        """Set the limits configuration for rendering."""
        self.limits_config = limits_config
        logger.info(f"Limits configuration updated: {len(limits_config)} signals with limits")
    
    def clear_all_limit_lines(self):
        """Clear all limit lines from all plots."""
        self.limit_lines.clear()
    
    def _apply_limit_lines_to_all_plots(self, container):
        """Apply limit lines to all plots in concatenated mode."""
        try:
            plot_widgets = container.plot_manager.get_plot_widgets()
            
            for graph_index, plot_widget in enumerate(plot_widgets):
                # Get visible signals for this graph
                visible_signals = self._get_visible_signals_for_graph(0, graph_index)
                if not visible_signals:
                    # Fallback: use all signals
                    all_signals = self.signal_processor.get_all_signals()
                    visible_signals = list(all_signals.keys())
                    
                self._apply_limit_lines(plot_widget, graph_index, visible_signals)
                
        except Exception as e:
            logger.error(f"Error applying limit lines to all plots: {e}")
            
    def set_basic_deviation_settings(self, tab_index: int, graph_index: int, deviation_settings: Dict[str, Any]):
        """Set basic deviation analysis settings for a specific graph."""
        # Önce mevcut ayarları kontrol et - gereksiz yeniden hesaplama yapma
        current_settings = self.basic_deviation_settings.get(graph_index, {})
        if current_settings == deviation_settings:
            logger.info(f"[DEVIATION_PERFORMANCE] Settings unchanged for graph {graph_index}, skipping update")
            return
            
        self.basic_deviation_settings[graph_index] = deviation_settings
        logger.info(f"[DEVIATION DEBUG] Stored basic deviation settings for graph {graph_index}: {deviation_settings}")
        logger.info(f"[DEVIATION DEBUG] Selected parameters: {deviation_settings.get('selected_parameters', [])}")
        logger.info(f"[DEVIATION DEBUG] Trend enabled: {deviation_settings.get('trend_analysis', {}).get('enabled', False)}")
        logger.info(f"[DEVIATION DEBUG] Fluctuation enabled: {deviation_settings.get('fluctuation_detection', {}).get('enabled', False)}")

        # Apply deviation analysis to the graph - sadece değişiklik varsa
        self._apply_basic_deviation_to_graph(tab_index, graph_index, deviation_settings)

    def _apply_basic_deviation_to_graph(self, tab_index: int, graph_index: int, deviation_settings: Dict[str, Any]):
        """Apply basic deviation analysis visualization to a specific graph."""
        logger.info(f"[DEVIATION DEBUG] Starting deviation application for graph {graph_index} on tab {tab_index}")
        logger.info(f"[DEVIATION DEBUG] Settings: {deviation_settings}")

        try:
            if not self.parent_widget:
                logger.warning("No parent widget available for deviation visualization")
                return
                
            # Get plot widgets from the active container
            active_container = self.parent_widget.get_active_graph_container()
            if not active_container:
                logger.warning("No active graph container available")
                return
                
            plot_widgets = active_container.get_plot_widgets()
            if graph_index >= len(plot_widgets):
                logger.warning(f"Graph index {graph_index} out of range")
                return
                
            plot_widget = plot_widgets[graph_index]

            # Clear existing deviation visualizations for this graph
            self._clear_deviation_lines(graph_index)

            # Get all available signals data
            all_signals_data = self.signal_processor.get_all_signals()
            
            # Determine which signals to process
            selected_parameters = deviation_settings.get('selected_parameters', [])
            
            signals_to_process = []
            if selected_parameters:
                logger.info(f"[DEVIATION DEBUG] Applying to selected parameters: {selected_parameters}")
                signals_to_process = selected_parameters
            else:
                logger.info(f"[DEVIATION DEBUG] No parameters selected, applying to all visible signals on graph.")
                visible_signals = self._get_visible_signals_for_graph(tab_index, graph_index)
                if visible_signals:
                    signals_to_process = visible_signals
                else:
                    # Fallback if no signals are visible for some reason
                    logger.warning(f"No visible signals found for graph {graph_index}, falling back to all signals.")
                    signals_to_process = list(all_signals_data.keys())

            logger.info(f"[DEVIATION DEBUG] Signals to process: {signals_to_process}")

            # Apply deviation analysis to each signal
            for signal_name in signals_to_process:
                if signal_name in all_signals_data:
                    x_data, y_data = self._get_signal_data_for_visualization(signal_name, self.max_points_for_analysis)

                    # --- Threaded Calculation ---
                    thread = QThread(self.parent_widget)  # Set parent for proper cleanup
                    worker = DeviationCalculator(y_data, deviation_settings)
                    worker.moveToThread(thread)

                    # Connect signals and slots
                    thread.started.connect(worker.run)
                    
                    # Create a proper callback without circular reference
                    def create_callback(signal_name, x_data, y_data, graph_index, plot_widget, thread):
                        return lambda results: self.on_deviation_calculation_finished(
                            signal_name, x_data, y_data, results, graph_index, plot_widget, thread
                        )
                    
                    worker.result_ready.connect(create_callback(signal_name, x_data, y_data, graph_index, plot_widget, thread))
                    
                    # Cleanup connections
                    worker.result_ready.connect(thread.quit)
                    worker.result_ready.connect(worker.deleteLater)
                    thread.finished.connect(thread.deleteLater)
                    
                    # Store both thread and worker for cleanup
                    thread_key = f"{graph_index}_{signal_name}"
                    self.deviation_threads[thread_key] = thread
                    self.deviation_workers[thread_key] = worker
                    
                    # Start the thread
                    thread.start()
                    
                else:
                    logger.warning(f"[DEVIATION DEBUG] Signal '{signal_name}' not found in available data.")

        except Exception as e:
            logger.error(f"Error applying basic deviation to graph {graph_index}: {e}", exc_info=True)
            
    def on_deviation_calculation_finished(self, signal_name, x_data, y_data, deviation_results, graph_index, plot_widget, thread):
        """Handle the results from the deviation calculator thread."""
        logger.info(f"[DEVIATION DEBUG] Calculation finished for {signal_name}. Visualizing results.")
        
        # Remove the thread and worker from the tracking dictionaries
        thread_key = f"{graph_index}_{signal_name}"
        if thread_key in self.deviation_threads:
            del self.deviation_threads[thread_key]
        if thread_key in self.deviation_workers:
            del self.deviation_workers[thread_key]

        self._visualize_deviation_results(plot_widget, graph_index, signal_name,
                                          x_data, y_data, deviation_results, 
                                          self.basic_deviation_settings.get(graph_index, {}))

    def _visualize_deviation_results(self, plot_widget, graph_index: int, signal_name: str, 
                                   x_data: np.ndarray, y_data: np.ndarray, 
                                   deviation_results: Dict[str, Any], settings: Dict[str, Any]):
        """Visualize deviation analysis results on the plot."""
        try:
            # Initialize deviation lines storage for this graph if needed
            if graph_index not in self.deviation_lines:
                self.deviation_lines[graph_index] = {}
                
            # Trend Line - Make it more visible
            if deviation_results['trend_line'] and settings.get('trend_analysis', {}).get('enabled', False):
                trend_line = plot_widget.plot(x_data, deviation_results['trend_line'], 
                                            pen=pg.mkPen(color='#FFD700', width=4, style=pg.QtCore.Qt.DashLine),
                                            name=f"{signal_name} Trend Line")
                self.deviation_lines[graph_index][f"{signal_name}_trend"] = trend_line
                logger.debug(f"Added trend line for {signal_name} with {len(deviation_results['trend_line'])} points")
                
            # Deviation Bands
            if deviation_results['bands'] and settings.get('visual_settings', {}).get('show_bands', False):
                upper_band = deviation_results['bands']['upper']
                lower_band = deviation_results['bands']['lower']
                
                if len(upper_band) == len(x_data) and len(lower_band) == len(x_data):
                    transparency = settings.get('visual_settings', {}).get('band_transparency', 30)
                    alpha = int(255 * (transparency / 100))
                    
                    # Upper band - Make more visible
                    upper_line = plot_widget.plot(x_data, upper_band, 
                                                pen=pg.mkPen(color='orange', width=2, style=pg.QtCore.Qt.DotLine),
                                                name=f"{signal_name} Upper Band")
                    self.deviation_lines[graph_index][f"{signal_name}_upper"] = upper_line
                    
                    # Lower band - Make more visible
                    lower_line = plot_widget.plot(x_data, lower_band, 
                                                pen=pg.mkPen(color='orange', width=2, style=pg.QtCore.Qt.DotLine),
                                                name=f"{signal_name} Lower Band")
                    self.deviation_lines[graph_index][f"{signal_name}_lower"] = lower_line
                    
                    logger.debug(f"Added deviation bands for {signal_name}")
                    
            # Alert Points
            if (deviation_results['alerts'] and 
                settings.get('fluctuation_detection', {}).get('highlight_on_graph', False)):
                
                alert_x = []
                alert_y = []
                
                for alert in deviation_results['alerts']:
                    if alert['index'] < len(x_data):
                        alert_x.append(x_data[alert['index']])
                        alert_y.append(alert['value'])
                        
                if alert_x and alert_y:
                    alert_scatter = plot_widget.plot(alert_x, alert_y, 
                                                   pen=None, 
                                                   symbol='o', 
                                                   symbolBrush=pg.mkBrush(color='red'),
                                                   symbolSize=12,
                                                   name=f"{signal_name} Alerts")
                    self.deviation_lines[graph_index][f"{signal_name}_alerts"] = alert_scatter
                    logger.debug(f"Added {len(alert_x)} alert points for {signal_name}")
                    
            # Red Segments for threshold exceedance
            if (deviation_results['red_segments'] and 
                settings.get('fluctuation_detection', {}).get('red_highlighting', False)):
                
                red_segments = deviation_results['red_segments']
                logger.info(f"Adding {len(red_segments)} red segments for {signal_name}")
                
                for i, segment in enumerate(red_segments):
                    start_idx = segment['start_index']
                    end_idx = segment['end_index']
                    
                    if start_idx < len(x_data) and end_idx < len(x_data):
                        # Extract segment data
                        segment_x = x_data[start_idx:end_idx+1]
                        segment_y = y_data[start_idx:end_idx+1]
                        
                        if len(segment_x) > 0 and len(segment_y) > 0:
                            # Create red line for this segment - Make it very visible
                            # Only show legend for the first segment to avoid clutter
                            legend_name = f"{signal_name} Threshold Exceedance" if i == 0 else None
                            red_line = plot_widget.plot(segment_x, segment_y, 
                                                       pen=pg.mkPen(color='#FF0000', width=5),
                                                       name=legend_name)
                            self.deviation_lines[graph_index][f"{signal_name}_red_segment_{i}"] = red_line
                            logger.debug(f"Added red segment {i+1} for {signal_name} from index {start_idx} to {end_idx}")
                    
        except Exception as e:
            logger.error(f"Error visualizing deviation results for {signal_name}: {e}")
            
    def _clear_deviation_lines(self, graph_index: int):
        """Clear deviation visualization lines for a specific graph."""
        if graph_index in self.deviation_lines:
            lines_to_remove = list(self.deviation_lines[graph_index].values())
            logger.info(f"[DEVIATION_PERFORMANCE] Clearing {len(lines_to_remove)} deviation lines for graph {graph_index}")
            
            # Batch remove items to avoid continuous removeItem calls
            for line in lines_to_remove:
                try:
                    if hasattr(line, 'scene') and line.scene():
                        scene = line.scene()
                        if scene:
                            scene.removeItem(line)
                except Exception as e:
                    logger.debug(f"[DEVIATION_PERFORMANCE] Error removing line: {e}")
                    
            # Clear the dictionary once
            self.deviation_lines[graph_index].clear()
            logger.info(f"[DEVIATION_PERFORMANCE] Cleared deviation lines for graph {graph_index}")
            
    def clear_all_deviation_lines(self):
        """Clear all deviation lines from all graphs."""
        # Stop any running threads before clearing lines
        logger.debug("Clearing all deviation lines and stopping threads...")
        
        for thread_key, thread in list(self.deviation_threads.items()):
            try:
                if thread and thread.isRunning():
                    logger.debug(f"Stopping deviation thread: {thread_key}")
                    
                    # Try to stop the worker gracefully first
                    if thread_key in self.deviation_workers:
                        try:
                            worker = self.deviation_workers[thread_key]
                            worker.stop()
                            logger.debug(f"Stopped worker for thread {thread_key}")
                        except Exception as e:
                            logger.debug(f"Could not stop worker gracefully: {e}")
                    
                    # Disconnect signals before quitting thread
                    try:
                        if hasattr(thread, 'finished'):
                            thread.finished.disconnect()
                        if hasattr(thread, 'started'):
                            thread.started.disconnect()
                    except Exception as e:
                        logger.debug(f"Error disconnecting thread signals: {e}")
                    
                    thread.quit()
                    if not thread.wait(3000):  # Wait up to 3 seconds
                        logger.warning(f"Deviation thread {thread_key} did not finish, terminating...")
                        thread.terminate()
                        thread.wait(1000)  # Wait for termination
                    else:
                        logger.debug(f"Deviation thread {thread_key} stopped successfully")
                elif thread:
                    # Thread exists but not running, still need to clean up
                    logger.debug(f"Cleaning up non-running thread: {thread_key}")
            except RuntimeError as e:
                # Thread object already deleted
                logger.debug(f"Thread {thread_key} already deleted: {e}")
            except Exception as e:
                logger.warning(f"Error stopping thread {thread_key}: {e}")
        
        # Clear references
        self.deviation_threads.clear()
        self.deviation_workers.clear()

        # Clear deviation lines
        for graph_index in list(self.deviation_lines.keys()):
            try:
                self._clear_deviation_lines(graph_index)
            except Exception as e:
                logger.warning(f"Error clearing deviation lines for graph {graph_index}: {e}")
        self.deviation_lines.clear()
        
        # Clear limit lines
        try:
            self.clear_all_limit_lines()
        except Exception as e:
            logger.warning(f"Error clearing limit lines: {e}")
        
        logger.debug("All deviation lines and threads cleared")
    
    # ===== Helpers =====
    def _get_signal_data_for_visualization(self, signal_name: str, max_points: int = 200_000, use_min_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (x_data, y_data) for a signal, preferring streaming from MpaiReader
        and downsampling to keep UI responsive.
        """
        try:
            all_signals = self.signal_processor.get_all_signals()
            if signal_name not in all_signals:
                return np.array([]), np.array([])
            
            signal_data = all_signals[signal_name]
            metadata = signal_data.get('metadata', {})
            x_data = np.array(signal_data['x_data'])
            y_data = np.array(signal_data['y_data'])
            
            # If data is already within limits, return as is
            if len(x_data) <= max_points:
                return x_data, y_data
            
            # If MpaiReader is available, stream and downsample
            reader = getattr(self.signal_processor, 'raw_dataframe', None)
            time_col_name = getattr(self.signal_processor, 'time_column_name', None) or 'time'
            is_mpai = hasattr(reader, 'get_header')
            full_count = metadata.get('full_count', len(x_data))
            
            if is_mpai and full_count > 0:
                return self._downsample_from_mpai(reader, signal_name, time_col_name, full_count, max_points, use_min_max=use_min_max)
            
            # Fallback: downsample in-memory data
            return self._downsample_arrays(x_data, y_data, max_points, use_min_max=use_min_max)
        
        except Exception as e:
            logger.error(f"[DOWNSAMPLE] Failed to get visualization data for {signal_name}: {e}", exc_info=True)
            return np.array([]), np.array([])
    
    def _downsample_from_mpai(self, reader, signal_name: str, time_col: str, full_count: int, max_points: int, use_min_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stream data from MpaiReader and downsample to max_points.
        - use_min_max=True: min/max çifti ile görsel sadakati korur (spike kaçırmaz).
        """
        try:
            # Hedef: min/max çift örnekleme ile ~max_points'tan az/denk çıktı
            if use_min_max and max_points > 2:
                target_pairs = max(1, max_points // 2)
                bucket_size = max(1, int(np.ceil(full_count / target_pairs)))
            else:
                target_pairs = None
                bucket_size = None

            chunk_size = min(getattr(self, 'chunk_size_streaming', 1_000_000), full_count)

            xs_out = []
            ys_out = []

            current_min = None
            current_max = None
            current_min_t = None
            current_max_t = None
            bucket_len = 0

            for start in range(0, full_count, chunk_size):
                length = min(chunk_size, full_count - start)
                
                t_chunk = reader.load_column_slice(time_col, int(start), int(length))
                y_chunk = reader.load_column_slice(signal_name, int(start), int(length))
                
                if len(t_chunk) == 0 or len(y_chunk) == 0:
                    continue
                
                if use_min_max and bucket_size:
                    for idx in range(len(t_chunk)):
                        val = y_chunk[idx]
                        ts = t_chunk[idx]
                        bucket_len += 1
                        if current_min is None or val < current_min:
                            current_min = val
                            current_min_t = ts
                        if current_max is None or val > current_max:
                            current_max = val
                            current_max_t = ts
                        if bucket_len >= bucket_size:
                            # Min/max'ı zaman sırasına göre ekle
                            if current_min_t is not None and current_max_t is not None:
                                if current_min_t <= current_max_t:
                                    xs_out.append(current_min_t); ys_out.append(current_min)
                                    if current_max_t != current_min_t:
                                        xs_out.append(current_max_t); ys_out.append(current_max)
                                else:
                                    xs_out.append(current_max_t); ys_out.append(current_max)
                                    if current_min_t != current_max_t:
                                        xs_out.append(current_min_t); ys_out.append(current_min)
                            elif current_min_t is not None:
                                xs_out.append(current_min_t); ys_out.append(current_min)
                            elif current_max_t is not None:
                                xs_out.append(current_max_t); ys_out.append(current_max)
                            # reset bucket
                            current_min = current_max = None
                            current_min_t = current_max_t = None
                            bucket_len = 0
                else:
                    # Eski uniform stride yolu
                    stride = max(1, int(np.ceil(full_count / max_points)))
                    xs_out.append(np.asarray(t_chunk)[::stride])
                    ys_out.append(np.asarray(y_chunk)[::stride])
            
            # Son kalan bucket'i flush et
            if use_min_max and bucket_size and bucket_len > 0:
                if current_min_t is not None and current_max_t is not None:
                    if current_min_t <= current_max_t:
                        xs_out.append(current_min_t); ys_out.append(current_min)
                        if current_max_t != current_min_t:
                            xs_out.append(current_max_t); ys_out.append(current_max)
                    else:
                        xs_out.append(current_max_t); ys_out.append(current_max)
                        if current_min_t != current_max_t:
                            xs_out.append(current_min_t); ys_out.append(current_min)
                elif current_min_t is not None:
                    xs_out.append(current_min_t); ys_out.append(current_min)
                elif current_max_t is not None:
                    xs_out.append(current_max_t); ys_out.append(current_max)

            # Concatenate if stride path used
            if not use_min_max or (not bucket_size):
                if not xs_out or not ys_out:
                    return np.array([]), np.array([])
                x_ds = np.concatenate(xs_out)
                y_ds = np.concatenate(ys_out)
            else:
                x_ds = np.array(xs_out)
                y_ds = np.array(ys_out)
            
            # Safety: enforce max_points
            if len(x_ds) > max_points:
                return self._downsample_arrays(x_ds, y_ds, max_points, use_min_max=use_min_max)
            
            return x_ds, y_ds
        
        except Exception as e:
            logger.error(f"[DOWNSAMPLE] MPAI streaming downsample failed for {signal_name}: {e}", exc_info=True)
            return np.array([]), np.array([])
    
    def _downsample_arrays(self, x: np.ndarray, y: np.ndarray, max_points: int, use_min_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample in-memory arrays.
        - use_min_max=True: her bucket için min/max çifti (spike yakalama).
        - use_min_max=False: uniform stride.
        """
        n = len(x)
        if n <= max_points:
            return x, y

        if use_min_max and max_points > 2:
            target_pairs = max(1, max_points // 2)
            bucket_size = max(1, int(np.ceil(n / target_pairs)))

            xs_out = []
            ys_out = []

            for start in range(0, n, bucket_size):
                end = min(n, start + bucket_size)
                x_slice = x[start:end]
                y_slice = y[start:end]
                if len(y_slice) == 0:
                    continue
                min_idx = int(np.argmin(y_slice))
                max_idx = int(np.argmax(y_slice))
                min_t = x_slice[min_idx]; min_v = y_slice[min_idx]
                max_t = x_slice[max_idx]; max_v = y_slice[max_idx]
                if min_t <= max_t:
                    xs_out.append(min_t); ys_out.append(min_v)
                    if max_t != min_t:
                        xs_out.append(max_t); ys_out.append(max_v)
                else:
                    xs_out.append(max_t); ys_out.append(max_v)
                    if min_t != max_t:
                        xs_out.append(min_t); ys_out.append(min_v)

            x_ds = np.array(xs_out)
            y_ds = np.array(ys_out)
        else:
            stride = int(np.ceil(n / max_points))
            x_ds = x[::stride]
            y_ds = y[::stride]

        # Safety clamp
        if len(x_ds) > max_points:
            stride = int(np.ceil(len(x_ds) / max_points))
            x_ds = x_ds[::stride]
            y_ds = y_ds[::stride]
        return x_ds, y_ds
