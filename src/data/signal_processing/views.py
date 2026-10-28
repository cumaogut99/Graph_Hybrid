"""
Downsampled display data and data for a visible time range.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalViewsMixin:
    """Downsampled display data and data for a visible time range."""

    def _load_full_data_downsampled(
        self, 
        reader, 
        columns: List[str], 
        time_column: str, 
        row_count: int, 
        target_points: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load FULL data from MPAI with streaming min/max downsampling.
        
        DeweSoft-like approach:
        - Streams data from disk in chunks (memory efficient)
        - Uses min/max per bucket to preserve ALL spikes
        - Returns downsampled data that represents the ENTIRE dataset
        
        Args:
            reader: MpaiReader instance
            columns: List of column names
            time_column: Name of time column
            row_count: Total number of rows
            target_points: Target number of display points
            
        Returns:
            Tuple of (time_array, {signal_name: signal_array})
        """
        import time as time_module
        start_time = time_module.perf_counter()
        
        # Safety check
        if row_count <= 0:
            logger.warning("[LOD] No data to process (row_count=0)")
            return np.array([]), {}
        
        # Get signal columns (exclude time)
        signal_columns = [col for col in columns if col != time_column]
        if not signal_columns:
            logger.warning("[LOD] No signal columns found")
            return np.array([]), {}
        
        # If data is small enough, load directly without downsampling
        if row_count <= target_points:
            logger.info(f"[LOD] Data small enough ({row_count} rows), loading directly")
            try:
                time_data = np.array(reader.load_column_slice(time_column, 0, row_count))
                if len(time_data) == 0:
                    logger.error("[LOD] Failed to load time column")
                    return np.array([]), {}
                    
                signal_data = {}
                for col in signal_columns:
                    col_data = reader.load_column_slice(col, 0, row_count)
                    signal_data[col] = np.array(col_data) if len(col_data) > 0 else np.zeros(len(time_data))
                return time_data, signal_data
            except Exception as e:
                logger.error(f"[LOD] Direct load failed: {e}")
                return np.array([]), {}
        
        # Calculate bucket size for min/max downsampling
        # Each bucket produces 2 points (min, max) to preserve spikes
        num_buckets = max(1, target_points // 2)
        bucket_size = max(1, row_count // num_buckets)
        
        # Recalculate num_buckets to cover all data
        num_buckets = (row_count + bucket_size - 1) // bucket_size  # Ceiling division
        
        logger.info(f"[LOD] Streaming downsample: {row_count:,} rows, {num_buckets} buckets, bucket_size={bucket_size}")
        
        # Initialize output arrays
        # Use uniform time sampling for simplicity - each bucket gets 2 time points
        time_out = []
        signal_out = {col: [] for col in signal_columns}
        
        # Track progress
        processed_buckets = 0
        failed_buckets = 0
        
        # Debug: log first few buckets
        logger.info(f"[LOD] Starting bucket processing: time_column='{time_column}', signals={signal_columns[:3]}...")
        
        # Process data bucket by bucket
        for bucket_idx in range(num_buckets):
            bucket_start = bucket_idx * bucket_size
            bucket_end = min((bucket_idx + 1) * bucket_size, row_count)
            bucket_len = bucket_end - bucket_start
            
            if bucket_len <= 0:
                continue
            
            try:
                # Strategy: Load full bucket, take first and last values
                # This ensures perfect time/value synchronization
                
                # Load time data for this bucket
                time_bucket = reader.load_column_slice(str(time_column), int(bucket_start), int(bucket_len))
                if len(time_bucket) == 0:
                    failed_buckets += 1
                    if failed_buckets <= 5:
                        logger.warning(f"[LOD] Empty bucket at {bucket_idx}")
                    continue
                
                time_bucket = np.array(time_bucket)
                
                # Take first and last time points
                t_first = float(time_bucket[0])
                t_last = float(time_bucket[-1])
                
                # Add time points (always 2 per bucket for consistent array lengths)
                time_out.append(t_first)
                time_out.append(t_last)
                
                # For each signal, load bucket and take first/last values
                for col in signal_columns:
                    y_bucket = reader.load_column_slice(str(col), int(bucket_start), int(bucket_len))
                    if len(y_bucket) == 0:
                        signal_out[col].append(0.0)
                        signal_out[col].append(0.0)
                        continue
                    
                    y_bucket = np.array(y_bucket)
                    
                    # Take first and last values (matches time points)
                    y_first = float(y_bucket[0])
                    y_last = float(y_bucket[-1])
                    
                    signal_out[col].append(y_first)
                    signal_out[col].append(y_last)
                
                processed_buckets += 1
                    
            except Exception as e:
                failed_buckets += 1
                if failed_buckets <= 5:  # Only log first 5 errors
                    logger.error(f"[LOD] Bucket {bucket_idx} error (start={bucket_start}, len={bucket_len}): {e}")
                # Add placeholder to maintain array sync
                time_out.append(float(bucket_start))
                time_out.append(float(bucket_end - 1))
                for col in signal_columns:
                    signal_out[col].append(0.0)
                    signal_out[col].append(0.0)
                continue
            
            # Progress logging every 10%
            if bucket_idx % max(1, num_buckets // 10) == 0:
                progress = (bucket_idx / num_buckets) * 100
                logger.debug(f"[LOD] Progress: {progress:.0f}%")
        
        logger.info(f"[LOD] Buckets: {processed_buckets} processed, {failed_buckets} failed out of {num_buckets}")
        
        # Convert to numpy arrays
        if len(time_out) == 0:
            logger.error("[LOD] No data was processed")
            return np.array([]), {}
        
        time_array = np.array(time_out, dtype=np.float64)
        
        # Convert signal arrays (should all have same length as time_array)
        signal_arrays = {}
        for col in signal_columns:
            data = signal_out[col]
            if len(data) == 0:
                signal_arrays[col] = np.zeros(len(time_array))
            elif len(data) == len(time_array):
                signal_arrays[col] = np.array(data, dtype=np.float64)
            else:
                # Length mismatch - pad or trim
                arr = np.array(data, dtype=np.float64)
                if len(arr) < len(time_array):
                    arr = np.pad(arr, (0, len(time_array) - len(arr)), mode='edge')
                else:
                    arr = arr[:len(time_array)]
                signal_arrays[col] = arr
        
        elapsed = time_module.perf_counter() - start_time
        logger.info(f"[LOD] Streaming downsample complete: {row_count:,} → {len(time_array)} points in {elapsed:.2f}s")
        
        return time_array, signal_arrays

    def get_downsampled_view(self, signal_name: str, target_points: int = 4000) -> Optional[Dict[str, np.ndarray]]:
        """
        Get downsampled view of FULL dataset for visualization.
        
        ✅ VISUALIZATION BRIDGE: Streams entire dataset from memory-mapped file
        and downsamples using min/max buckets to fit screen width.
        
        Args:
            signal_name: Signal to downsample
            target_points: Target number of points (default: 4000 = 2x 1920px screen)
        
        Returns:
            Dict with 'x_data' and 'y_data' arrays representing FULL dataset
            Returns None if signal not found
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered([signal_name])
            return self._get_downsampled_view_internal(signal_name, target_points)

    def _get_downsampled_view_internal(self, signal_name: str, target_points: int = 4000) -> Optional[Dict[str, np.ndarray]]:
        """
        Internal method for downsampling (assumes mutex is already held).
        """
        if signal_name not in self.signal_data:
            logger.warning(f"[DOWNSAMPLE] Signal '{signal_name}' not found")
            return None
        
        signal_info = self.signal_data[signal_name]
        
        # Check if memory-mapped
        if not self._is_file_backed(signal_info):
            # In-memory data (CSV or range-filtered): return full data
            logger.debug(f"[DOWNSAMPLE] CSV signal, returning full data")
            return {
                'x_data': signal_info.get('x_data', np.array([])),
                'y_data': signal_info.get('y_data', np.array([]))
            }
        
        # ✅ STREAMING DOWNSAMPLING: Process FULL dataset in buckets
        try:
            # Defensive check
            if 'mpai_reader' not in signal_info:
                logger.error(f"[DOWNSAMPLE] signal_info missing 'mpai_reader' key. Keys: {list(signal_info.keys())}")
                logger.error(f"[DOWNSAMPLE] signal_info content: {signal_info}")
                return None
            
            reader = signal_info['mpai_reader']
            if getattr(reader, 'closed', False):
                logger.warning(f"[DOWNSAMPLE] '{signal_name}': MPAI reader is closed")
                return None
            col_name = signal_info['column_name']
            time_col = signal_info['time_column']
            row_count = signal_info['row_count']

            # ✅ OPTIMIZED PATH: Use MpaiDirectoryReader's built-in get_render_data
            # This utilizes the .red (Reduced) files for instant access
            if hasattr(reader, 'get_render_data') and hasattr(reader, 'name_to_id'):
                try:
                    ch_id = reader.name_to_id.get(col_name)
                    if ch_id is not None:
                        # Get full time range for "full" downsample
                        t_min, t_max = reader.get_time_range()
                        # Pixel width approximation (target_points roughly equals 2 * width)
                        pixel_width = max(1, target_points // 2)
                        
                        logger.debug(f"[DOWNSAMPLE-FAST] Using .red file for '{col_name}'")
                        x_fast, y_fast, _ = reader.get_render_data(ch_id, t_min, t_max, pixel_width)
                        
                        if len(x_fast) > 0:
                            return {
                                'x_data': x_fast,
                                'y_data': y_fast
                            }
                except Exception as e:
                    logger.warning(f"[DOWNSAMPLE-FAST] Failed for '{col_name}', falling back: {e}")
            
            # OPTIMIZATION: Handle synthetic time column specially
            # Time column is generated arithmetically, no need to stream from disk
            if col_name.lower() == time_col.lower() or col_name.lower() == 'time':
                logger.debug(f"[DOWNSAMPLE] Generating synthetic time for '{signal_name}'")
                t_min, t_max = reader.get_time_range()
                x_data = np.linspace(t_min, t_max, min(target_points, row_count))
                return {
                    'x_data': x_data,
                    'y_data': x_data  # Time column x==y
                }
            
            # Only log streaming for non-time columns to reduce spam
            logger.debug(f"[DOWNSAMPLE] Streaming {row_count:,} rows → {target_points} points for '{signal_name}'")
            
            # If dataset is small (less than 4x target), just load it all
            # Optimization: Don't bucketize if we have e.g. 5000 rows and want 4000 points
            if row_count <= target_points * 4:
                logger.debug(f"[DOWNSAMPLE] Small dataset ({row_count} rows), loading all")
                x_data = np.array(reader.load_column_slice(time_col, 0, row_count), dtype=np.float64)
                y_data = np.array(reader.load_column_slice(col_name, 0, row_count), dtype=np.float64)
                return {'x_data': x_data, 'y_data': y_data}
            
            # Calculate bucket size (each bucket → 2 points: min, max)
            num_buckets = target_points // 2
            bucket_size = max(1, row_count // num_buckets)
            num_buckets = (row_count + bucket_size - 1) // bucket_size  # Ceiling division
            
            logger.info(f"[DOWNSAMPLE] Buckets: {num_buckets}, bucket_size: {bucket_size}")
            
            # Preallocate output arrays (2 points per bucket)
            x_out = np.zeros(num_buckets * 2, dtype=np.float64)
            y_out = np.zeros(num_buckets * 2, dtype=np.float64)
            out_idx = 0
            
            # Stream and downsample bucket by bucket
            for bucket_idx in range(num_buckets):
                bucket_start = bucket_idx * bucket_size
                bucket_end = min((bucket_idx + 1) * bucket_size, row_count)
                bucket_len = bucket_end - bucket_start
                
                if bucket_len <= 0:
                    continue
                
                try:
                    # Load bucket from memory-mapped file
                    x_bucket = np.array(reader.load_column_slice(time_col, bucket_start, bucket_len), dtype=np.float64)
                    y_bucket = np.array(reader.load_column_slice(col_name, bucket_start, bucket_len), dtype=np.float64)
                    
                    if len(x_bucket) == 0 or len(y_bucket) == 0:
                        continue
                    
                    # Find min/max indices in this bucket
                    min_idx = np.argmin(y_bucket)
                    max_idx = np.argmax(y_bucket)
                    
                    # Store min point first, then max (preserves spikes)
                    if min_idx < max_idx:
                        # Min comes first
                        x_out[out_idx] = x_bucket[min_idx]
                        y_out[out_idx] = y_bucket[min_idx]
                        out_idx += 1
                        x_out[out_idx] = x_bucket[max_idx]
                        y_out[out_idx] = y_bucket[max_idx]
                        out_idx += 1
                    else:
                        # Max comes first
                        x_out[out_idx] = x_bucket[max_idx]
                        y_out[out_idx] = y_bucket[max_idx]
                        out_idx += 1
                        x_out[out_idx] = x_bucket[min_idx]
                        y_out[out_idx] = y_bucket[min_idx]
                        out_idx += 1
                    
                except Exception as e:
                    # Every bucket reads the same files: if one fails (e.g. the
                    # reader was closed) the rest fail too. Give up instead of
                    # logging thousands of warnings per signal.
                    logger.error(f"[DOWNSAMPLE] '{signal_name}' failed at bucket {bucket_idx}/{num_buckets}: {e}")
                    return None
            
            # Trim to actual size
            x_out = x_out[:out_idx]
            y_out = y_out[:out_idx]
            
            logger.info(f"[DOWNSAMPLE] Complete: {row_count:,} rows → {len(x_out):,} points ({len(x_out)/row_count*100:.2f}%)")
            
            return {
                'x_data': x_out,
                'y_data': y_out
            }
            
        except Exception as e:
            logger.error(f"[DOWNSAMPLE] Failed for '{signal_name}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_data_for_view_range(
        self, 
        signal_name: str, 
        view_start: float, 
        view_end: float,
        screen_width: int = 1920
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get signal data optimized for a specific view range (LOD - Level of Detail).
        
        DeweSoft-like behavior:
        - If zoomed out: Return downsampled data (already in memory)
        - If zoomed in beyond threshold: Load full resolution from disk
        
        Args:
            signal_name: Signal identifier
            view_start: Start of visible time range
            view_end: End of visible time range
            screen_width: Screen width for target points calculation
            
        Returns:
            Tuple of (x_data, y_data) optimized for the view range, or None
        """
        self._ensure_filtered([signal_name])
        if signal_name not in self.signal_data:
            return None
        
        signal_info = self.signal_data[signal_name]
        metadata = signal_info.get('metadata', {})
        is_mpai = metadata.get('mpai', False)
        full_count = metadata.get('full_count', 0)
        full_time_range = metadata.get('full_time_range', (0, 1))
        
        # Get current downsampled data
        x_data = signal_info['x_data']
        y_data = signal_info['y_data']
        
        if len(x_data) == 0:
            return None
        
        # Calculate zoom level
        full_duration = full_time_range[1] - full_time_range[0]
        view_duration = view_end - view_start
        
        if full_duration <= 0 or view_duration <= 0:
            return x_data, y_data
        
        zoom_level = full_duration / view_duration
        
        # Target points = 2x screen width
        target_points = screen_width * 2
        
        # Calculate how many original data points are in view range
        points_per_second = full_count / full_duration if full_duration > 0 else 1
        points_in_view = int(view_duration * points_per_second)
        
        logger.debug(f"[LOD] Signal: {signal_name}, zoom: {zoom_level:.1f}x, points_in_view: {points_in_view}")
        
        # Decision logic:
        # 1. If zoomed in a lot (zoom_level > threshold) AND points_in_view < target_points
        #    → Load full resolution from disk
        # 2. Otherwise → Use downsampled data (filter to view range)
        
        if is_mpai and zoom_level > self._lod_zoom_threshold and points_in_view < target_points * 2:
            # Load full resolution data from disk for this range
            logger.info(f"[LOD] Loading full resolution for {signal_name} (zoom={zoom_level:.1f}x)")
            return self._load_range_from_disk(signal_name, view_start, view_end, target_points)
        else:
            # Use downsampled data, filter to view range
            mask = (x_data >= view_start) & (x_data <= view_end)
            if np.any(mask):
                return x_data[mask], y_data[mask]
            else:
                # View range outside data - return edge data
                return x_data, y_data

    def _load_range_from_disk(
        self, 
        signal_name: str, 
        start_time: float, 
        end_time: float,
        max_points: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load a specific time range from disk at full or optimized resolution.
        
        Args:
            signal_name: Signal identifier
            start_time: Start of time range
            end_time: End of time range
            max_points: Maximum points to return
            
        Returns:
            Tuple of (x_data, y_data) or None
        """
        if not hasattr(self, 'raw_dataframe') or self.raw_dataframe is None:
            return None
        
        if not hasattr(self.raw_dataframe, 'load_column_slice'):
            return None
        
        reader = self.raw_dataframe
        time_column = self.time_column_name
        
        # Safety: Ensure time_column is valid
        if time_column is None:
            # Try to get time column from reader
            try:
                columns = reader.get_column_names()
                for col in columns:
                    if 'time' in col.lower():
                        time_column = col
                        break
                if time_column is None and columns:
                    time_column = columns[0]
            except:
                pass
        
        if time_column is None:
            logger.warning("[LOD] No time column available for range loading")
            return None
        
        self._ensure_filtered([signal_name])
        if signal_name not in self.signal_data:
            return None
        
        metadata = self.signal_data[signal_name].get('metadata', {})
        full_count = metadata.get('full_count', 0)
        full_time_range = metadata.get('full_time_range', (0, 1))
        
        if full_count == 0:
            return None
        
        try:
            # Estimate row indices from time
            full_duration = full_time_range[1] - full_time_range[0]
            if full_duration <= 0:
                return None
            
            rows_per_second = full_count / full_duration
            
            start_row = max(0, int((start_time - full_time_range[0]) * rows_per_second))
            end_row = min(full_count, int((end_time - full_time_range[0]) * rows_per_second))
            row_count = end_row - start_row
            
            if row_count <= 0:
                return None
            
            logger.debug(f"[LOD] Loading rows {start_row} to {end_row} ({row_count} rows) for {signal_name}")
            
            # Load data - ensure int types for C++ binding
            x_data = np.array(reader.load_column_slice(str(time_column), int(start_row), int(row_count)))
            y_data = np.array(reader.load_column_slice(str(signal_name), int(start_row), int(row_count)))
            
            # If too many points, apply min/max downsampling
            if len(x_data) > max_points:
                x_data, y_data = self._minmax_downsample(x_data, y_data, max_points)
            
            return x_data, y_data
            
        except Exception as e:
            logger.error(f"[LOD] Failed to load range from disk: {e}")
            return None

    def _minmax_downsample(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        target_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply min/max downsampling to preserve spikes.
        
        Args:
            x_data: Time array
            y_data: Signal array
            target_points: Target number of points
            
        Returns:
            Tuple of downsampled (x_data, y_data)
        """
        n = len(x_data)
        if n <= target_points:
            return x_data, y_data
        
        num_buckets = target_points // 2
        bucket_size = n // num_buckets
        
        x_out = []
        y_out = []
        
        for i in range(num_buckets):
            start = i * bucket_size
            end = min((i + 1) * bucket_size, n)
            
            if start >= end:
                continue
            
            y_slice = y_data[start:end]
            x_slice = x_data[start:end]
            
            min_idx = np.argmin(y_slice)
            max_idx = np.argmax(y_slice)
            
            # Add in time order
            if min_idx <= max_idx:
                x_out.append(x_slice[min_idx])
                y_out.append(y_slice[min_idx])
                if max_idx != min_idx:
                    x_out.append(x_slice[max_idx])
                    y_out.append(y_slice[max_idx])
            else:
                x_out.append(x_slice[max_idx])
                y_out.append(y_slice[max_idx])
                if min_idx != max_idx:
                    x_out.append(x_slice[min_idx])
                    y_out.append(y_slice[min_idx])
        
        return np.array(x_out), np.array(y_out)
