"""
LOD Pyramid Generator for MPAI Files
=====================================

Generates pre-computed Level of Detail (LOD) layers during CSV→MPAI conversion.

DeweSoft-inspired approach:
- LOD1: 100-step min/max aggregation  
- LOD2: 10,000-step min/max aggregation
- LOD3: 100,000-step min/max aggregation

Each LOD layer stores: time_min, time_max, {signal}_min, {signal}_max
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# LOD Configuration: (name, bucket_size)
LOD_CONFIGS = [
    ('lod1_100', 100),        # Every 100 samples → min/max
    ('lod2_10k', 10_000),     # Every 10K samples
    ('lod3_100k', 100_000),   # Every 100K samples
]


class LodGenerator:
    """
    Generates LOD pyramid files from raw MPAI data.
    
    Usage:
        generator = LodGenerator(mpai_container_path)
        generator.generate_all_lods(time_column, signal_columns, row_count)
    """
    
    def __init__(self, container_path: str, progress_callback: Optional[Callable] = None):
        """
        Initialize LOD generator.
        
        Args:
            container_path: Path to MPAI container directory
            progress_callback: Optional callback(message, percentage)
        """
        self.container_path = container_path
        self.progress_callback = progress_callback
        
        # Ensure container directory exists
        os.makedirs(container_path, exist_ok=True)
    
    def generate_all_lods(
        self,
        time_data: np.ndarray,
        signal_data: Dict[str, np.ndarray],
        signal_columns: List[str]
    ) -> Dict[str, str]:
        """
        Generate all LOD levels from source data.
        
        Args:
            time_data: Time column array
            signal_data: Dict of signal_name -> numpy array
            signal_columns: List of signal column names
            
        Returns:
            Dict of lod_name -> file_path
        """
        generated_files = {}
        row_count = len(time_data)
        
        logger.info(f"[LOD] Generating pyramid for {row_count:,} rows, {len(signal_columns)} signals")
        
        for i, (lod_name, bucket_size) in enumerate(LOD_CONFIGS):
            if row_count < bucket_size * 2:
                logger.info(f"[LOD] Skipping {lod_name} (not enough data: {row_count} < {bucket_size * 2})")
                continue
            
            if self.progress_callback:
                pct = 90 + int((i / len(LOD_CONFIGS)) * 8)
                self.progress_callback(f"Generating {lod_name}...", pct)
            
            file_path = self._generate_single_lod(
                lod_name, bucket_size,
                time_data, signal_data, signal_columns
            )
            
            if file_path:
                generated_files[lod_name] = file_path
        
        logger.info(f"[LOD] Generated {len(generated_files)} LOD files")
        return generated_files
    
    def _generate_single_lod(
        self,
        lod_name: str,
        bucket_size: int,
        time_data: np.ndarray,
        signal_data: Dict[str, np.ndarray],
        signal_columns: List[str]
    ) -> Optional[str]:
        """
        Generate a single LOD level using min/max aggregation.
        
        For each bucket of `bucket_size` samples, stores:
        - time_min, time_max: Time range of bucket
        - {signal}_min, {signal}_max: Min/max values in bucket
        """
        try:
            import time
            start_time = time.perf_counter()
            
            row_count = len(time_data)
            num_buckets = (row_count + bucket_size - 1) // bucket_size
            
            logger.info(f"[LOD] Generating {lod_name}: {row_count:,} rows → {num_buckets} buckets (bucket_size={bucket_size})")
            
            # Initialize output arrays
            time_min_out = np.zeros(num_buckets, dtype=np.float64)
            time_max_out = np.zeros(num_buckets, dtype=np.float64)
            
            signal_min_out = {col: np.zeros(num_buckets, dtype=np.float64) for col in signal_columns}
            signal_max_out = {col: np.zeros(num_buckets, dtype=np.float64) for col in signal_columns}
            
            # Process each bucket
            for bucket_idx in range(num_buckets):
                start_row = bucket_idx * bucket_size
                end_row = min((bucket_idx + 1) * bucket_size, row_count)
                
                # Time range
                time_bucket = time_data[start_row:end_row]
                time_min_out[bucket_idx] = np.min(time_bucket)
                time_max_out[bucket_idx] = np.max(time_bucket)
                
                # Signal min/max
                for col in signal_columns:
                    if col in signal_data:
                        signal_bucket = signal_data[col][start_row:end_row]
                        signal_min_out[col][bucket_idx] = np.nanmin(signal_bucket)
                        signal_max_out[col][bucket_idx] = np.nanmax(signal_bucket)
            
            # Build PyArrow table
            columns = {
                'time_min': pa.array(time_min_out),
                'time_max': pa.array(time_max_out),
            }
            
            for col in signal_columns:
                columns[f'{col}_min'] = pa.array(signal_min_out[col])
                columns[f'{col}_max'] = pa.array(signal_max_out[col])
            
            table = pa.table(columns)
            
            # Write Parquet file
            output_path = os.path.join(self.container_path, f'{lod_name}.parquet')
            pq.write_table(table, output_path, compression='zstd')
            
            elapsed = time.perf_counter() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            logger.info(f"[LOD] {lod_name}: {num_buckets} buckets, {file_size_mb:.2f} MB in {elapsed:.2f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"[LOD] Failed to generate {lod_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_from_mpai_reader(
        self,
        reader,  # MpaiReader instance
        time_column: str,
        signal_columns: List[str],
        chunk_size: int = 1_000_000
    ) -> Dict[str, str]:
        """
        Generate LOD layers by streaming from MpaiReader (memory efficient).
        
        For very large files, this method loads data in chunks.
        """
        generated_files = {}
        row_count = reader.get_row_count()
        
        logger.info(f"[LOD] Streaming LOD generation for {row_count:,} rows")
        
        for lod_name, bucket_size in LOD_CONFIGS:
            if row_count < bucket_size * 2:
                logger.info(f"[LOD] Skipping {lod_name} (not enough data)")
                continue
            
            file_path = self._generate_lod_streaming(
                reader, lod_name, bucket_size,
                time_column, signal_columns, chunk_size
            )
            
            if file_path:
                generated_files[lod_name] = file_path
        
        return generated_files
    
    def _generate_lod_streaming(
        self,
        reader,
        lod_name: str,
        bucket_size: int,
        time_column: str,
        signal_columns: List[str],
        chunk_size: int
    ) -> Optional[str]:
        """
        Generate LOD by streaming chunks from MpaiReader.
        
        This approach uses minimal RAM even for 100M+ row files.
        """
        try:
            import time
            start_time = time.perf_counter()
            
            row_count = reader.get_row_count()
            num_buckets = (row_count + bucket_size - 1) // bucket_size
            
            logger.info(f"[LOD] Streaming {lod_name}: {row_count:,} rows → {num_buckets} buckets")
            
            # Output arrays
            time_min_out = np.zeros(num_buckets, dtype=np.float64)
            time_max_out = np.zeros(num_buckets, dtype=np.float64)
            signal_min_out = {col: np.full(num_buckets, np.inf, dtype=np.float64) for col in signal_columns}
            signal_max_out = {col: np.full(num_buckets, -np.inf, dtype=np.float64) for col in signal_columns}
            
            # First pass: get time range per bucket
            time_min_out.fill(np.inf)
            time_max_out.fill(-np.inf)
            
            # Process in streaming chunks
            offset = 0
            while offset < row_count:
                load_size = min(chunk_size, row_count - offset)
                
                # Load time chunk
                time_chunk = np.array(reader.load_column_slice(time_column, offset, load_size))
                
                # Determine which buckets this chunk covers
                bucket_start = offset // bucket_size
                bucket_end = min((offset + load_size - 1) // bucket_size + 1, num_buckets)
                
                # Update each affected bucket
                for bucket_idx in range(bucket_start, bucket_end):
                    bucket_row_start = bucket_idx * bucket_size
                    bucket_row_end = min((bucket_idx + 1) * bucket_size, row_count)
                    
                    # Clip to chunk bounds
                    chunk_start = max(0, bucket_row_start - offset)
                    chunk_end = min(load_size, bucket_row_end - offset)
                    
                    if chunk_start < chunk_end:
                        bucket_data = time_chunk[chunk_start:chunk_end]
                        time_min_out[bucket_idx] = min(time_min_out[bucket_idx], np.min(bucket_data))
                        time_max_out[bucket_idx] = max(time_max_out[bucket_idx], np.max(bucket_data))
                
                # Load and process signal columns
                for col in signal_columns:
                    signal_chunk = np.array(reader.load_column_slice(col, offset, load_size))
                    
                    for bucket_idx in range(bucket_start, bucket_end):
                        bucket_row_start = bucket_idx * bucket_size
                        bucket_row_end = min((bucket_idx + 1) * bucket_size, row_count)
                        
                        chunk_start = max(0, bucket_row_start - offset)
                        chunk_end = min(load_size, bucket_row_end - offset)
                        
                        if chunk_start < chunk_end:
                            bucket_data = signal_chunk[chunk_start:chunk_end]
                            signal_min_out[col][bucket_idx] = min(
                                signal_min_out[col][bucket_idx], 
                                np.nanmin(bucket_data)
                            )
                            signal_max_out[col][bucket_idx] = max(
                                signal_max_out[col][bucket_idx], 
                                np.nanmax(bucket_data)
                            )
                
                offset += load_size
            
            # Build and write Parquet
            columns = {
                'time_min': pa.array(time_min_out),
                'time_max': pa.array(time_max_out),
            }
            
            for col in signal_columns:
                # Replace inf with 0 for empty buckets
                min_arr = signal_min_out[col]
                max_arr = signal_max_out[col]
                min_arr[np.isinf(min_arr)] = 0.0
                max_arr[np.isinf(max_arr)] = 0.0
                
                columns[f'{col}_min'] = pa.array(min_arr)
                columns[f'{col}_max'] = pa.array(max_arr)
            
            table = pa.table(columns)
            output_path = os.path.join(self.container_path, f'{lod_name}.parquet')
            pq.write_table(table, output_path, compression='zstd')
            
            elapsed = time.perf_counter() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            logger.info(f"[LOD] {lod_name}: {num_buckets} buckets, {file_size_mb:.2f} MB in {elapsed:.2f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"[LOD] Streaming LOD failed for {lod_name}: {e}")
            import traceback
            traceback.print_exc()
            return None


def get_lod_for_visible_samples(visible_samples: int) -> str:
    """
    Determine which LOD level to use based on visible sample count.
    
    Args:
        visible_samples: Number of samples in visible view range
        
    Returns:
        'raw', 'lod1_100', 'lod2_10k', or 'lod3_100k'
    """
    if visible_samples < 20_000:
        return 'raw'
    elif visible_samples < 2_000_000:
        return 'lod1_100'
    elif visible_samples < 20_000_000:
        return 'lod2_10k'
    else:
        return 'lod3_100k'
