"""
LOD (Level of Detail) Engine for Dynamic Visualization

Provides view-dependent data access using C++ SmartDownsampler:
- Zoom-Out: Returns Min-Max downsampled envelope via C++ SIMD algorithms
- Zoom-In: Returns raw data when point count <= screen width

This enables smooth 60fps panning/zooming even with 100M+ data points.

Author: MachinePulseAI Team
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LODEngine:
    """
    View-dependent Level of Detail engine for visualization.
    
    Uses C++ SmartDownsampler for high-performance downsampling:
    - SIMD/AVX2 acceleration for critical point detection
    - LTTB algorithm for visual fidelity
    - Streaming from memory-mapped MPAI files
    
    Statistics are always calculated on actual data via C++ StatisticsEngine,
    NEVER on downsampled visualization data.
    """
    
    def __init__(self, signal_processor=None):
        """
        Initialize LOD Engine.
        
        Args:
            signal_processor: SignalProcessor instance for data access
        """
        self.signal_processor = signal_processor
        self._cache: Dict[str, dict] = {}  # signal_name -> {view_range, data}
        self._cache_max_size = 20
        
        # Import C++ module once
        try:
            import time_graph_cpp
            self._cpp_module = time_graph_cpp
            self._cpp_available = True
            logger.info("[LOD] C++ SmartDownsampler available")
        except ImportError:
            self._cpp_module = None
            self._cpp_available = False
            logger.warning("[LOD] C++ module not available, using Python fallback")
    
    def get_display_data(
        self,
        signal_name: str,
        view_range: Tuple[float, float],
        screen_width: int = 1920,
        target_points: int = 4000
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get optimally downsampled data for current view.
        
        Args:
            signal_name: Name of signal to retrieve
            view_range: Visible time range (x_min, x_max)
            screen_width: Display width in pixels
            target_points: Target output points (default: 2x screen width)
        
        Returns:
            Tuple of (x_data, y_data) arrays, or None if signal not found
            
        Notes:
            - Returns raw data if visible points <= target_points
            - Returns C++ min-max downsampled envelope otherwise
            - Results are cached per view range
        """
        if self.signal_processor is None:
            logger.error("[LOD] No signal processor available")
            return None
        
        # Check cache
        cache_key = f"{signal_name}_{view_range[0]:.6f}_{view_range[1]:.6f}_{screen_width}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached['x_data'], cached['y_data']
        
        # Get signal info
        signal_info = self.signal_processor.signal_data.get(signal_name)
        if signal_info is None:
            logger.warning(f"[LOD] Signal '{signal_name}' not found")
            return None
        
        metadata = signal_info.get('metadata', {})
        is_memory_mapped = metadata.get('memory_mapped', False)
        
        x_min, x_max = view_range
        
        if is_memory_mapped:
            # Memory-mapped MPAI: Use streaming downsampling from C++
            result = self._get_mpai_display_data(
                signal_info, signal_name, x_min, x_max, target_points
            )
        else:
            # In-memory CSV: Use cached numpy arrays
            result = self._get_csv_display_data(
                signal_info, x_min, x_max, target_points
            )
        
        if result is not None:
            # Cache result
            self._cache[cache_key] = {'x_data': result[0], 'y_data': result[1]}
            
            # Limit cache size
            if len(self._cache) > self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        return result
    
    def _get_mpai_display_data(
        self,
        signal_info: dict,
        signal_name: str,
        x_min: float,
        x_max: float,
        target_points: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get display data for memory-mapped MPAI signal using C++ streaming."""
        reader = signal_info.get('mpai_reader')
        if reader is None:
            logger.error(f"[LOD] No mpai_reader for '{signal_name}'")
            return None
        
        col_name = signal_info.get('column_name', signal_name)
        time_col = signal_info.get('time_column', 'time')
        row_count = signal_info.get('row_count', 0)
        
        if row_count == 0:
            return None
        
        if self._cpp_available:
            try:
                # Use C++ SmartDownsampler for streaming
                config = self._cpp_module.SmartDownsampleConfig()
                config.target_points = target_points
                config.use_lttb = True
                config.detect_local_extrema = True
                
                downsampler = self._cpp_module.SmartDownsampler()
                result = downsampler.downsample_streaming(
                    reader, time_col, col_name, config
                )
                
                x_data = np.array(result.x, dtype=np.float64)
                y_data = np.array(result.y, dtype=np.float64)
                
                logger.debug(f"[LOD] C++ streaming: {result.input_size} -> {result.output_size} points")
                return x_data, y_data
                
            except Exception as e:
                logger.error(f"[LOD] C++ streaming failed: {e}")
                # Fall through to Python fallback
        
        # Python fallback: Load visible range only
        return self._python_fallback_mpai(reader, time_col, col_name, x_min, x_max, target_points)
    
    def _get_csv_display_data(
        self,
        signal_info: dict,
        x_min: float,
        x_max: float,
        target_points: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get display data for in-memory CSV signal."""
        x_data = signal_info.get('x_data')
        y_data = signal_info.get('y_data')
        
        if x_data is None or y_data is None:
            return None
        
        # Filter to visible range
        mask = (x_data >= x_min) & (x_data <= x_max)
        x_visible = x_data[mask]
        y_visible = y_data[mask]
        
        if len(x_visible) == 0:
            return None
        
        # If points fit on screen, return raw data
        if len(x_visible) <= target_points:
            return x_visible, y_visible
        
        # Downsample using C++ if available
        if self._cpp_available:
            try:
                result = self._cpp_module.smart_downsample(
                    x_visible.astype(np.float64),
                    y_visible.astype(np.float64),
                    target_points
                )
                return np.array(result.x), np.array(result.y)
            except Exception as e:
                logger.warning(f"[LOD] C++ downsample failed: {e}")
        
        # Python fallback: Simple min-max bucketing
        return self._python_minmax_downsample(x_visible, y_visible, target_points)
    
    def _python_fallback_mpai(
        self,
        reader,
        time_col: str,
        col_name: str,
        x_min: float,
        x_max: float,
        target_points: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Python fallback for MPAI streaming downsample."""
        try:
            row_count = reader.get_row_count()
            
            # Load in chunks for memory efficiency
            chunk_size = 50000
            x_out = []
            y_out = []
            
            for start_row in range(0, row_count, chunk_size):
                chunk_len = min(chunk_size, row_count - start_row)
                
                x_chunk = np.array(reader.load_column_slice(time_col, start_row, chunk_len))
                
                # Filter by time range
                in_range = (x_chunk >= x_min) & (x_chunk <= x_max)
                if not np.any(in_range):
                    continue
                
                y_chunk = np.array(reader.load_column_slice(col_name, start_row, chunk_len))
                
                x_out.append(x_chunk[in_range])
                y_out.append(y_chunk[in_range])
            
            if not x_out:
                return None
            
            x_data = np.concatenate(x_out)
            y_data = np.concatenate(y_out)
            
            if len(x_data) <= target_points:
                return x_data, y_data
            
            return self._python_minmax_downsample(x_data, y_data, target_points)
            
        except Exception as e:
            logger.error(f"[LOD] Python MPAI fallback failed: {e}")
            return None
    
    def _python_minmax_downsample(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        target_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple min-max downsampling in pure Python."""
        n = len(x_data)
        if n <= target_points:
            return x_data, y_data
        
        # Each bucket produces 2 points (min, max)
        num_buckets = target_points // 2
        bucket_size = n // num_buckets
        
        x_out = []
        y_out = []
        
        for i in range(num_buckets):
            start = i * bucket_size
            end = min((i + 1) * bucket_size, n)
            
            if start >= end:
                continue
            
            x_bucket = x_data[start:end]
            y_bucket = y_data[start:end]
            
            min_idx = np.argmin(y_bucket)
            max_idx = np.argmax(y_bucket)
            
            # Add in time order
            if min_idx < max_idx:
                x_out.extend([x_bucket[min_idx], x_bucket[max_idx]])
                y_out.extend([y_bucket[min_idx], y_bucket[max_idx]])
            else:
                x_out.extend([x_bucket[max_idx], x_bucket[min_idx]])
                y_out.extend([y_bucket[max_idx], y_bucket[min_idx]])
        
        return np.array(x_out), np.array(y_out)
    
    def invalidate_cache(self, signal_name: Optional[str] = None):
        """
        Invalidate cached display data.
        
        Args:
            signal_name: Specific signal to invalidate, or None for all
        """
        if signal_name is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(signal_name)]
            for key in keys_to_remove:
                del self._cache[key]
    
    def get_raw_data_for_view(
        self,
        signal_name: str,
        view_range: Tuple[float, float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get raw (non-downsampled) data for a view range.
        
        Used when zoomed in far enough to see individual points.
        """
        if self.signal_processor is None:
            return None
        
        signal_info = self.signal_processor.signal_data.get(signal_name)
        if signal_info is None:
            return None
        
        x_min, x_max = view_range
        is_memory_mapped = signal_info.get('metadata', {}).get('memory_mapped', False)
        
        if is_memory_mapped:
            # Load raw data from MPAI in view range
            reader = signal_info.get('mpai_reader')
            if reader is None:
                return None
            
            col_name = signal_info.get('column_name', signal_name)
            time_col = signal_info.get('time_column', 'time')
            
            try:
                # This is a simplified approach - for production, 
                # use binary search to find row range
                row_count = reader.get_row_count()
                chunk_size = 10000
                
                for start_row in range(0, row_count, chunk_size):
                    chunk_len = min(chunk_size, row_count - start_row)
                    x_chunk = np.array(reader.load_column_slice(time_col, start_row, chunk_len))
                    
                    if x_chunk[-1] < x_min:
                        continue
                    if x_chunk[0] > x_max:
                        break
                    
                    y_chunk = np.array(reader.load_column_slice(col_name, start_row, chunk_len))
                    mask = (x_chunk >= x_min) & (x_chunk <= x_max)
                    
                    return x_chunk[mask], y_chunk[mask]
                    
            except Exception as e:
                logger.error(f"[LOD] Raw data load failed: {e}")
                return None
        else:
            # In-memory: filter existing arrays
            x_data = signal_info.get('x_data')
            y_data = signal_info.get('y_data')
            
            if x_data is None or y_data is None:
                return None
            
            mask = (x_data >= x_min) & (x_data <= x_max)
            return x_data[mask], y_data[mask]
