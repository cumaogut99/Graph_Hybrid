"""
LOD Reader - Fast loading of pre-computed LOD parquet files
============================================================

Provides near-instant data loading for zoomed-out views by reading
from pre-computed min/max aggregation files instead of raw data.
"""

import logging
import os
from typing import Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Cache for loaded LOD data
_lod_cache: Dict[str, Dict] = {}


def get_lod_container_path(mpai_path: str) -> str:
    """Get the LOD container directory path for an MPAI file."""
    return os.path.splitext(mpai_path)[0] + '_lod'


def get_available_lod_levels(mpai_path: str) -> List[str]:
    """Get list of available LOD levels for an MPAI file."""
    container = get_lod_container_path(mpai_path)
    if not os.path.exists(container):
        return []
    
    levels = []
    for name in ['lod1_100', 'lod2_10k', 'lod3_100k']:
        if os.path.exists(os.path.join(container, f'{name}.parquet')):
            levels.append(name)
    
    return levels


def select_lod_level(visible_samples: int, available_levels: List[str]) -> Optional[str]:
    """
    Select best LOD level based on visible sample count.
    
    Returns None if raw data should be used.
    """
    if visible_samples < 20_000:
        return None  # Use raw data
    
    if visible_samples < 2_000_000 and 'lod1_100' in available_levels:
        return 'lod1_100'
    elif visible_samples < 20_000_000 and 'lod2_10k' in available_levels:
        return 'lod2_10k'
    elif 'lod3_100k' in available_levels:
        return 'lod3_100k'
    elif 'lod2_10k' in available_levels:
        return 'lod2_10k'
    elif 'lod1_100' in available_levels:
        return 'lod1_100'
    
    return None


def load_lod_data(
    mpai_path: str,
    lod_level: str,
    signal_name: str,
    view_x_min: float,
    view_x_max: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load pre-computed LOD data for a signal within a time range.
    
    Returns min/max interleaved data for visualization.
    
    Args:
        mpai_path: Path to MPAI file
        lod_level: LOD level name (lod1_100, lod2_10k, lod3_100k)
        signal_name: Name of signal column
        view_x_min, view_x_max: Visible time range
        
    Returns:
        (x_data, y_data) numpy arrays, or (None, None) if not available
    """
    try:
        import pyarrow.parquet as pq
        
        container = get_lod_container_path(mpai_path)
        parquet_path = os.path.join(container, f'{lod_level}.parquet')
        
        if not os.path.exists(parquet_path):
            logger.warning(f"[LOD-READ] {lod_level}.parquet not found")
            return None, None
        
        # Check cache first
        cache_key = f"{parquet_path}:{signal_name}"
        if cache_key in _lod_cache:
            cached = _lod_cache[cache_key]
            return _filter_cached_data(cached, view_x_min, view_x_max)
        
        # Load parquet file
        import time
        start = time.perf_counter()
        
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        # Get time columns
        time_min = df['time_min'].values
        time_max = df['time_max'].values
        
        # Get signal columns
        signal_min_col = f'{signal_name}_min'
        signal_max_col = f'{signal_name}_max'
        
        if signal_min_col not in df.columns:
            logger.warning(f"[LOD-READ] Column {signal_min_col} not in {lod_level}")
            return None, None
        
        signal_min = df[signal_min_col].values
        signal_max = df[signal_max_col].values
        
        # Cache the loaded data
        _lod_cache[cache_key] = {
            'time_min': time_min,
            'time_max': time_max,
            'signal_min': signal_min,
            'signal_max': signal_max
        }
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[LOD-READ] Loaded {lod_level} for {signal_name}: {len(time_min)} buckets in {elapsed:.1f}ms")
        
        return _filter_cached_data(_lod_cache[cache_key], view_x_min, view_x_max)
        
    except Exception as e:
        logger.error(f"[LOD-READ] Failed to load {lod_level}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _filter_cached_data(
    cached: Dict,
    view_x_min: float,
    view_x_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter cached LOD data to view range and create min/max envelope.
    
    Creates interleaved min/max points for proper envelope rendering.
    """
    time_min = cached['time_min']
    time_max = cached['time_max']
    signal_min = cached['signal_min']
    signal_max = cached['signal_max']
    
    # Find buckets overlapping view range
    # A bucket overlaps if: bucket.time_max >= view_x_min AND bucket.time_min <= view_x_max
    mask = (time_max >= view_x_min) & (time_min <= view_x_max)
    
    filtered_time_min = time_min[mask]
    filtered_time_max = time_max[mask]
    filtered_signal_min = signal_min[mask]
    filtered_signal_max = signal_max[mask]
    
    n = len(filtered_time_min)
    if n == 0:
        return np.array([]), np.array([])
    
    # Create min/max envelope: for each bucket, output (time_min, min) and (time_max, max)
    # This creates the characteristic "filled" look of downsampled data
    x_out = np.empty(n * 2, dtype=np.float64)
    y_out = np.empty(n * 2, dtype=np.float64)
    
    # Interleave: min point, then max point
    x_out[0::2] = filtered_time_min  # Min timestamps
    x_out[1::2] = filtered_time_max  # Max timestamps
    y_out[0::2] = filtered_signal_min  # Min values
    y_out[1::2] = filtered_signal_max  # Max values
    
    return x_out, y_out


def clear_lod_cache():
    """Clear the LOD data cache."""
    global _lod_cache
    _lod_cache.clear()
    logger.info("[LOD-READ] Cache cleared")


def get_cache_stats() -> Dict:
    """Get LOD cache statistics."""
    return {
        'entries': len(_lod_cache),
        'signals': list(_lod_cache.keys())
    }
