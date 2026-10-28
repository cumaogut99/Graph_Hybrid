"""
Downsampled display data of a signal.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import Dict, Optional
import numpy as np
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalViewsMixin:
    """Downsampled display data of a signal."""

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
