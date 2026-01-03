# type: ignore
"""
Signal Processor for Time Graph Widget

Handles signal processing operations including:
- Data normalization and scaling
- Statistical calculations
- Signal filtering and conditioning
- Performance-optimized data operations
- LOD (Level of Detail) auto-selection for spike-safe rendering
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl
from PyQt5.QtCore import QObject, pyqtSignal as Signal, QThread, QMutex, QMutexLocker

# Try to import C++ LOD bindings
try:
    import sys
    build_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'build', 'Release')
    if os.path.exists(build_path) and build_path not in sys.path:
        sys.path.insert(0, build_path)
    import pyarrow  # Load pyarrow first for DLL dependencies
    import time_graph_cpp as tgcpp
    HAS_CPP_LOD = hasattr(tgcpp, 'LodReader') and hasattr(tgcpp, 'get_lod_bucket_size')
except ImportError:
    HAS_CPP_LOD = False
    tgcpp = None

logger = logging.getLogger(__name__)

class SignalProcessor(QObject):
    """
    High-performance signal processor for time-series data.
    
    Features:
    - Memory-efficient operations using numpy views
    - Threaded processing for large datasets
    - Caching for repeated calculations
    - Optimized statistical computations
    - LOD auto-selection for spike-safe visualization
    """
    
    # Signals
    processing_started = Signal()
    processing_finished = Signal()
    statistics_updated = Signal(dict)  # Updated statistics
    
    def __init__(self):
        super().__init__()
        self.signal_data = {}  # Dict of signal_name -> data_dict
        self.original_signal_data = {}  # Backup of original data for filter reset
        self.normalized_data = {}  # Cache for normalized data
        self.statistics_cache = {}  # Cache for statistics
        self.mutex = QMutex()  # Thread safety
        
        # PERFORMANCE: Polars DataFrame'i sakla (lazy conversion için)
        self.raw_dataframe = None  # Polars DataFrame
        self.time_column_name = None  # Time column name
        self.current_mpai_path = None  # ✅ Store MPAI file path for LOD lookup
        self.numpy_cache = {}  # Column name -> numpy array cache
        
        # LOD cache for spike-safe rendering
        self.lod_readers = {}  # bucket_size -> LodReader instance
        self.lod_container_path = None  # Path to LOD container directory
        
        # Processing parameters
        self.normalization_method = "peak"  # peak, rms, minmax
        self.statistics_window_size = 1000  # Rolling statistics window
        
        # PERFORMANCE: Enhanced statistics cache
        # Format: (signal_name, range_start, range_end, threshold_mode, threshold_value) -> stats_dict
        self._stats_cache = {}
        self._stats_cache_max_size = 100  # Limit cache size to prevent memory issues
        
        # PERFORMANCE: Cursor value cache for MPAI files
        # Caches small windows of data around recently accessed time points
        # Format: (signal_name, window_start_idx) -> (x_slice, y_slice)
        self._cursor_value_cache = {}
        self._cursor_value_cache_max_size = 20  # Keep 20 windows cached
        
        # LOD (Level of Detail) settings for DeweSoft-like rendering
        # Target points = 2x screen width for smooth rendering
        self._target_display_points = 4000  # ~2x 1920 screen width
        self._lod_zoom_threshold = 5.0  # Zoom level threshold for full resolution
        
        # ✅ DOWNSAMPLED DATA CACHE: Store downsampled views separately
        # Format: signal_name -> {'x_data': array, 'y_data': array}
        # This prevents corruption of the metadata-only signal_data structure
        self._downsampled_cache = {}
        self._downsampled_cache_max_size = 50  # Keep 50 signals cached
        
    def process_data(self, df, normalize: bool = False, time_column: Optional[str] = None) -> Dict[str, Dict]:
        """
        Process Polars DataFrame OR MpaiReader and extract all signals.
        OPTIMIZED: Lazy conversion - DataFrame saklanır, numpy conversion geciktirilir.
        """
        # PERFORMANCE: Clear statistics cache when new data is loaded
        self.clear_statistics_cache()
        
        if df is None:
            logger.warning("process_data called with None")
            return {}
            
        # Check if data source is MpaiReader
        is_mpai = hasattr(df, 'get_header')
        
        if not is_mpai and df.height == 0:
            logger.warning("process_data called with empty DataFrame")
            return {}
            
        self.processing_started.emit()
        
        try:
            # Clear existing data
            self.clear_all_data()
            
            if is_mpai:
                # --- MPAI READER PATH (DeweSoft-like full data with smart downsampling) ---
                logger.info("SignalProcessor: Processing MpaiReader with streaming downsample...")
                
                # ✅ FIX: Store MpaiReader for later use (cursor values, statistics, LOD)
                self.raw_dataframe = df
                
                # Get column names from Reader
                columns = df.get_column_names()
                col_count = df.get_column_count()
                row_count = df.get_row_count()
                
                # Determine time column
                # Debug: Log incoming time_column and available columns
                logger.info(f"[MPAI DEBUG] Requested time_column: '{time_column}'")
                logger.info(f"[MPAI DEBUG] Available columns: {columns}")
                
                if time_column and time_column in columns:
                    # Exact match found
                    logger.info(f"[MPAI DEBUG] Exact match found for time_column: '{time_column}'")
                elif time_column:
                    # time_column provided but not found - try to find a match
                    logger.warning(f"[MPAI DEBUG] time_column '{time_column}' NOT in columns list!")
                    
                    # Try case-insensitive match or partial match
                    matched = False
                    for col in columns:
                        if col.lower() == time_column.lower():
                            logger.info(f"[MPAI DEBUG] Case-insensitive match: '{time_column}' -> '{col}'")
                            time_column = col
                            matched = True
                            break
                        elif 'time' in col.lower() or 'zaman' in col.lower():
                            logger.info(f"[MPAI DEBUG] Fallback to time-like column: '{col}'")
                            time_column = col
                            matched = True
                            break
                    
                    if not matched:
                        # Use first column as fallback
                        time_column = columns[0] if columns else "time"
                        logger.warning(f"[MPAI DEBUG] No time column found, using first column: '{time_column}'")
                else:
                    # No time_column provided - try to find one
                    for col in columns:
                        if 'time' in col.lower() or 'zaman' in col.lower():
                            time_column = col
                            logger.info(f"[MPAI DEBUG] Auto-detected time column: '{time_column}'")
                            break
                    
                    if not time_column:
                        time_column = columns[0] if columns else "time"
                        logger.warning(f"[MPAI DEBUG] No time column found, using first column: '{time_column}'")
                
                self.time_column_name = time_column
                
                # ✅ ZERO-COPY ARCHITECTURE: Store ONLY metadata, NO data arrays
                # MpaiReader is already memory-mapped, data stays on disk
                # Target: < 500 MB RAM even with 50 GB files
                
                logger.info(f"[MMAP] Zero-copy loading: {row_count:,} rows (metadata only)...")
                
                # Get time range (load only 2 values!)
                try:
                    # DEBUG: Log the actual call and result
                    logger.info(f"[MMAP DEBUG] Calling load_column_slice('{time_column}', 0, 1)")
                    first_slice = df.load_column_slice(time_column, 0, 1)
                    logger.info(f"[MMAP DEBUG] first_slice = {first_slice}, type={type(first_slice)}")
                    
                    logger.info(f"[MMAP DEBUG] Calling load_column_slice('{time_column}', {max(0, row_count - 1)}, 1)")
                    last_slice = df.load_column_slice(time_column, max(0, row_count - 1), 1)
                    logger.info(f"[MMAP DEBUG] last_slice = {last_slice}, type={type(last_slice)}")
                    
                    if len(first_slice) > 0 and len(last_slice) > 0:
                        t_first = float(first_slice[0])
                        t_last = float(last_slice[0])
                        full_time_range = (t_first, t_last)
                        logger.info(f"[MMAP] Time range: {t_first:.2f} to {t_last:.2f}")
                    else:
                        logger.warning(f"[MMAP DEBUG] Empty slices returned! first_len={len(first_slice)}, last_len={len(last_slice)}")
                        t_first, t_last = 0.0, float(row_count - 1)
                        full_time_range = (t_first, t_last)
                except Exception as e:
                    logger.error(f"Error getting time range: {e}")
                    import traceback
                    traceback.print_exc()
                    t_first, t_last = 0.0, float(row_count - 1)
                    full_time_range = (t_first, t_last)
                
                # ✅ CRITICAL: Register signals with METADATA ONLY (no data arrays)
                for col in columns:
                    if col != time_column:
                        try:
                            # Metadata - memory-mapped flag
                            metadata = {
                                'mpai': True,
                                'memory_mapped': True,  # ← NEW: Indicates zero-copy
                                'full_count': row_count,
                                'full_time_range': full_time_range,
                            }
                            
                            # Store ONLY metadata (no x_data/y_data arrays!)
                            self.signal_data[col] = {
                                'mpai_reader': df,        # ← FIX: Store reader reference HERE
                                'column_name': col,       # Column name
                                'time_column': time_column,
                                'row_count': row_count,
                                'time_range': full_time_range,
                                'metadata': metadata
                            }
                            
                            logger.info(f"[MMAP] Registered '{col}' (memory-mapped, {row_count:,} rows)")
                            
                        except Exception as e:
                            logger.warning(f"Failed to register signal '{col}': {e}")
                
                logger.info(f"[MMAP] Zero-copy loading complete. RAM: metadata only (~{len(columns) * 100} bytes)")
                
                logger.info(f"[MMAP] Registered {len(self.signal_data)} signals from MPAI ({row_count:,} rows each)")
                
            else:
                # Non-MPAI path is no longer supported
                raise ValueError("Sadece MPAI formatı destekleniyor; lütfen dosyayı MPAI'ye dönüştürün.")
            
            # Apply normalization if requested (works for both paths on loaded data)
            if normalize:
                self.apply_normalization(method=self.normalization_method)
                
            return self.get_all_signals()
            
        finally:
            self.processing_finished.emit()
    
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
    
    def _get_numpy_column(self, col_name: str) -> np.ndarray:
        """
        PERFORMANCE: Cache'lenmiş numpy column getir.
        İlk çağrıda Polars'tan çevir, sonra cache'den döndür.
        ROBUST: NULL, NaN, Inf değerleri güvenli şekilde handle et.
        """
        if col_name not in self.numpy_cache:
            if self.raw_dataframe is None:
                raise ValueError("Raw dataframe not available")
            
            try:
                # Polars → NumPy (ilk kez)
                col_data = self.raw_dataframe.get_column(col_name).to_numpy()
                
                # ROBUST: Veri tipini kontrol et ve dönüştür
                if col_data.dtype == object or col_data.dtype == np.dtype('O'):
                    # Object type - string veya mixed olabilir
                    try:
                        # Sayısala çevirmeyi dene
                        import pandas as pd
                        col_data = pd.to_numeric(pd.Series(col_data), errors='coerce').to_numpy()
                        logger.debug(f"Column '{col_name}' converted from object to numeric")
                    except:
                        # Dönüştürülemez, sıfır array döndür
                        logger.warning(f"Column '{col_name}' cannot be converted to numeric, using zeros")
                        col_data = np.zeros(len(col_data))
                
                # ROBUST: None değerleri NaN yap
                if col_data.dtype == object:
                    col_data = np.where(col_data == None, np.nan, col_data)
                    col_data = col_data.astype(float)
                
                # ROBUST: NaN ve Inf değerleri temizle
                mask_nan = np.isnan(col_data)
                mask_inf = np.isinf(col_data)
                
                if np.any(mask_nan) or np.any(mask_inf):
                    # Önce inf'leri NaN yap
                    col_data[mask_inf] = np.nan
                    
                    # Forward fill için pandas kullan (daha hızlı)
                    import pandas as pd
                    # ✅ FIX: Deprecated fillna(method='ffill') yerine ffill() kullan
                    filled_data = pd.Series(col_data).ffill().fillna(0.0).to_numpy()
                    
                    num_cleaned = np.sum(mask_nan) + np.sum(mask_inf)
                    logger.debug(f"Cleaned {num_cleaned} invalid values in column '{col_name}'")
                    
                    col_data = filled_data
                
                self.numpy_cache[col_name] = col_data
                logger.debug(f"Converted column '{col_name}' to numpy (cached, {len(col_data)} points)")
                
            except Exception as e:
                logger.error(f"Failed to convert column '{col_name}' to numpy: {e}")
                # Fallback: sıfır array döndür
                logger.warning(f"Returning zero array for column '{col_name}'")
                self.numpy_cache[col_name] = np.zeros(len(self.raw_dataframe))
        
        return self.numpy_cache[col_name]

    def add_signal(self, name: str, x_data: np.ndarray, y_data: np.ndarray, 
                   metadata: Optional[Dict] = None):
        """
        Add or update signal data with memory-efficient storage.
        
        Args:
            name: Signal identifier
            x_data: Time/X-axis data (shared reference when possible)
            y_data: Signal values
            metadata: Additional signal information
        """
        with QMutexLocker(self.mutex):
            # Use memory views for efficiency
            if not isinstance(x_data, np.ndarray):
                x_data = np.asarray(x_data, dtype=np.float64)
            if not isinstance(y_data, np.ndarray):
                y_data = np.asarray(y_data, dtype=np.float64)
            
            # Store original data for filter reset (only if not already stored)
            if name not in self.original_signal_data:
                self.original_signal_data[name] = {
                    'x_data': x_data.copy(),
                    'y_data': y_data.copy(),
                    'metadata': metadata or {}
                }
            
            # Store signal data
            self.signal_data[name] = {
                'x_data': x_data,
                'y_data': y_data,
                'original_y': y_data.copy(),  # Keep original for normalization
                'metadata': metadata or {},
                'last_modified': np.datetime64('now')
            }
            
            # Clear related caches
            self._clear_cache(name)
            
            logger.debug(f"Added signal '{name}' with {len(y_data)} points")
    
    def remove_signal(self, name: str):
        """Remove signal and clear associated caches."""
        with QMutexLocker(self.mutex):
            if name in self.signal_data:
                del self.signal_data[name]
            if name in self.normalized_data:
                del self.normalized_data[name]
            if name in self.statistics_cache:
                del self.statistics_cache[name]
            
            logger.debug(f"Removed signal '{name}'")
    
    def _register_calculated_param(self, name: str, reader):
        """
        Register a disk-based calculated parameter for memory-mapped access.
        
        Instead of storing data in RAM, this stores a reference to an MPAI reader
        that provides memory-mapped (zero-RAM) access to the calculated data.
        
        Args:
            name: Parameter name
            reader: MpaiReader instance for the calculated parameter file
        """
        with QMutexLocker(self.mutex):
            # Store reader reference for later access
            if not hasattr(self, '_calc_param_readers'):
                self._calc_param_readers = {}
            self._calc_param_readers[name] = reader
            
            # Register in signal_data with lazy loading placeholder
            row_count = reader.get_row_count()
            t_min, t_max = reader.get_time_range()
            
            self.signal_data[name] = {
                '_is_calc_param': True,  # Flag for special handling
                '_reader': reader,
                'row_count': row_count,
                'time_min': t_min,
                'time_max': t_max,
                # Lazy load - actual data loaded on demand
            }
            
            logger.info(f"[CALC PARAM] Registered disk-based param '{name}' ({row_count} rows, memory-mapped)")
    
    def get_signal_data(self, name: str) -> Optional[Dict]:
        """
        Get signal data safely.
        
        For disk-based calculated parameters: Loads data from MPAI reader (lazy load)
        For regular signals: Returns cached data
        """
        with QMutexLocker(self.mutex):
            data = self.signal_data.get(name, {})
            
            # Check if this is a disk-based calculated parameter
            if data.get('_is_calc_param') and '_reader' in data:
                reader = data['_reader']
                try:
                    # Lazy load from disk (memory-mapped, not RAM)
                    row_count = reader.get_row_count()
                    x_data = reader.load_column_slice('time', 0, row_count)
                    y_data = reader.load_column_slice(name, 0, row_count)
                    
                    return {
                        'x_data': x_data,
                        'y_data': y_data,
                        'metadata': {'calculated': True, 'disk_based': True}
                    }
                except Exception as e:
                    logger.error(f"[CALC PARAM] Failed to load {name}: {e}")
                    return {}
            
            return data.copy() if data else {}
    
    def get_all_signals(self) -> Dict[str, Dict]:
        """
        Get all signal data safely.
        
        For memory-mapped MPAI files: Returns DOWNSAMPLED view of FULL dataset
        For CSV files: Returns full data (legacy)
        """
        with QMutexLocker(self.mutex):
            result = {}
            
            for name, data in self.signal_data.items():
                # Check if this is a memory-mapped signal
                if data.get('metadata', {}).get('memory_mapped'):
                    # ✅ CACHE CHECK: Return cached downsampled data if available
                    if name in self._downsampled_cache:
                        logger.debug(f"[CACHE] Using cached downsampled data for '{name}'")
                        cached = self._downsampled_cache[name]
                        result[name] = {
                            'x_data': cached['x_data'],
                            'y_data': cached['y_data'],
                            'metadata': {
                                **data['metadata'],
                                'downsampled': True,
                                'downsample_points': len(cached['x_data']),
                                'total_rows': data['row_count']
                            }
                        }
                        continue
                    
                    # ✅ VISUALIZATION BRIDGE: Get downsampled view of FULL dataset
                    # Use internal method to avoid mutex deadlock
                    downsampled = self._get_downsampled_view_internal(name, target_points=4000)
                    
                    if downsampled:
                        # ✅ CACHE IT: Store in separate cache (don't modify signal_data!)
                        self._downsampled_cache[name] = {
                            'x_data': downsampled['x_data'],
                            'y_data': downsampled['y_data']
                        }
                        
                        # Limit cache size
                        if len(self._downsampled_cache) > self._downsampled_cache_max_size:
                            # Remove oldest entry (FIFO)
                            oldest_key = next(iter(self._downsampled_cache))
                            del self._downsampled_cache[oldest_key]
                            logger.debug(f"[CACHE] Evicted '{oldest_key}' from downsampled cache")
                        
                        result[name] = {
                            'x_data': downsampled['x_data'],
                            'y_data': downsampled['y_data'],
                            'metadata': {
                                **data['metadata'],
                                'downsampled': True,
                                'downsample_points': len(downsampled['x_data']),
                                'total_rows': data['row_count']
                            }
                        }
                    else:
                        # Fallback: empty data
                        logger.error(f"[MMAP] Failed to get downsampled view for '{name}'")
                        result[name] = {
                            'x_data': np.array([]),
                            'y_data': np.array([]),
                            'metadata': data.get('metadata', {})
                        }
                else:
                    # Legacy path: CSV data (already in memory)
                    result[name] = data.copy()
            
            return result
    
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
        if not signal_info.get('metadata', {}).get('memory_mapped'):
            # Legacy CSV: Return full data (already in memory)
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
                    logger.warning(f"[DOWNSAMPLE] Bucket {bucket_idx} failed: {e}")
                    continue
            
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
                    
                    # Clear related caches since data changed
                    self._clear_cache(signal_name)
                    
                    logger.debug(f"Updated filtered data for signal '{signal_name}' with {len(y_data)} points")
                else:
                    logger.warning(f"Signal '{signal_name}' not found in signal_data, skipping filtered data update")
    
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
            
            logger.info("Restored original data for all signals")
    
    def apply_normalization(self, signal_names: Optional[List[str]] = None, 
                          method: str = "peak") -> Dict[str, np.ndarray]:
        """
        Apply normalization to specified signals or all signals.
        
        Args:
            signal_names: List of signals to normalize (None for all)
            method: Normalization method ('peak', 'rms', 'minmax', 'zscore')
            
        Returns:
            Dict of signal_name -> normalized_y_data
        """
        self.processing_started.emit()
        
        try:
            with QMutexLocker(self.mutex):
                if signal_names is None:
                    signal_names = list(self.signal_data.keys())
                
                normalized_results = {}
                
                for name in signal_names:
                    if name not in self.signal_data:
                        continue
                    
                    y_data = self.signal_data[name]['y_data']
                    
                    # Check cache first
                    cache_key = f"{name}_{method}_{hash(y_data.tobytes())}"
                    if cache_key in self.normalized_data:
                        normalized_y = self.normalized_data[cache_key]
                    else:
                        # Perform normalization
                        normalized_y = self._normalize_array(y_data, method)
                        self.normalized_data[cache_key] = normalized_y
                    
                    # Update signal data
                    self.signal_data[name]['y_data'] = normalized_y
                    self.signal_data[name]['normalized'] = True
                    self.signal_data[name]['normalization_method'] = method
                    
                    normalized_results[name] = normalized_y
                    
                    logger.debug(f"Normalized signal '{name}' using {method} method")
                
                return normalized_results
                
        finally:
            self.processing_finished.emit()
    
    def remove_normalization(self, signal_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Remove normalization and restore original data.
        
        Args:
            signal_names: List of signals to denormalize (None for all)
            
        Returns:
            Dict of signal_name -> original_y_data
        """
        with QMutexLocker(self.mutex):
            if signal_names is None:
                signal_names = list(self.signal_data.keys())
            
            restored_results = {}
            
            for name in signal_names:
                if name not in self.signal_data:
                    continue
                
                # Restore original data
                original_y = self.signal_data[name]['original_y']
                self.signal_data[name]['y_data'] = original_y.copy()
                self.signal_data[name]['normalized'] = False
                
                restored_results[name] = original_y
                
                logger.debug(f"Restored original data for signal '{name}'")
            
            return restored_results
    
    def _normalize_array(self, data: np.ndarray, method: str) -> np.ndarray:
        """
        Normalize array using specified method with optimized algorithms.
        
        Args:
            data: Input array
            method: Normalization method
            
        Returns:
            Normalized array
        """
        if len(data) == 0:
            return data.copy()
        
        if method == "peak":
            # Peak normalization (divide by absolute maximum)
            peak_val = np.max(np.abs(data))
            return data / peak_val if peak_val != 0 else data.copy()
            
        elif method == "rms":
            # RMS normalization
            rms_val = np.sqrt(np.mean(data**2))
            return data / rms_val if rms_val != 0 else data.copy()
            
        elif method == "minmax":
            # Min-Max normalization to [0, 1]
            min_val, max_val = np.min(data), np.max(data)
            range_val = max_val - min_val
            return (data - min_val) / range_val if range_val != 0 else data.copy()
            
        elif method == "zscore":
            # Z-score normalization (mean=0, std=1)
            mean_val, std_val = np.mean(data), np.std(data)
            return (data - mean_val) / std_val if std_val != 0 else data.copy()
            
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data.copy()
    
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
            if signal_names is None:
                signal_names = list(self.signal_data.keys())
            
            results = {}
            
            for name in signal_names:
                if name not in self.signal_data:
                    continue
                
                signal_info = self.signal_data[name]
                metadata = signal_info.get('metadata', {})
                is_mpai = metadata.get('mpai', False)
                is_memory_mapped = metadata.get('memory_mapped', False)
                
                # ========== MEMORY-MAPPED MPAI: Use C++ FastStatsCalculator OR MpaiDirectoryReader ==========
                if is_memory_mapped:
                    try:
                        # Get reader and column info from signal_info
                        reader = signal_info.get('mpai_reader')
                        col_name = signal_info.get('column_name', name)
                        time_col = signal_info.get('time_column', 'time')
                        
                        if reader is None:
                            logger.error(f"[STATS] Memory-mapped signal '{name}' has no mpai_reader")
                            continue
                        
                        # 1. NEW: Check if this is the Python MpaiDirectoryReader (has get_statistics_snapshot)
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

                        # 2. LEGACY C++: Use time_graph_cpp
                        import time_graph_cpp
                        
                        if time_range is not None:
                            start_time, end_time = time_range
                            # ✅ PERFORMANCE FIX: Use FastStatsCalculator instead of streaming
                            # FastStatsCalculator uses:
                            #   - Binary search (O(log N)) to find row indices
                            #   - Pre-aggregated chunk metadata (O(1)) for complete chunks
                            #   - Only loads partial chunks from disk
                            # This is 1000x faster than streaming which scans entire time column!
                            logger.debug(f"[STATS] Using FastStatsCalculator for '{name}' time range [{start_time:.2f}, {end_time:.2f}]")
                            
                            fast_stats = time_graph_cpp.FastStatsCalculator.calculate_time_range_statistics(
                                reader, col_name, start_time, end_time, time_col
                            )
                            
                            stats = {
                                'mean': fast_stats.mean,
                                'std': fast_stats.std_dev,
                                'min': fast_stats.min,
                                'max': fast_stats.max,
                                'median': fast_stats.mean,  # Approximation
                                'rms': fast_stats.rms,
                                'peak_to_peak': fast_stats.max - fast_stats.min,
                                'count': fast_stats.count,
                                'valid_count': fast_stats.count,
                            }
                            results[name] = stats
                        else:
                            # Full statistics without time range - use column metadata
                            col_meta = reader.get_column_metadata(col_name)
                            if col_meta:
                                stats = {
                                    'mean': col_meta.statistics.mean if hasattr(col_meta.statistics, 'mean') else 0.0,
                                    'std': col_meta.statistics.std_dev if hasattr(col_meta.statistics, 'std_dev') else 0.0,
                                    'min': col_meta.statistics.min if hasattr(col_meta.statistics, 'min') else 0.0,
                                    'max': col_meta.statistics.max if hasattr(col_meta.statistics, 'max') else 0.0,
                                    'median': col_meta.statistics.mean if hasattr(col_meta.statistics, 'mean') else 0.0,
                                    'rms': 0.0,
                                    'peak_to_peak': (col_meta.statistics.max - col_meta.statistics.min) if hasattr(col_meta.statistics, 'max') else 0.0,
                                    'count': reader.get_row_count(),
                                    'valid_count': reader.get_row_count(),
                                }
                                results[name] = stats
                            else:
                                logger.warning(f"[STATS] No column metadata for '{col_name}'")
                        continue
                        
                    except Exception as e:
                        logger.error(f"[STATS] FastStatsCalculator failed for '{name}': {e}")
                        import traceback
                        traceback.print_exc()
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
                        import time_graph_cpp
                        
                        if time_range is not None:
                            # PERFORMANCE OPTIMIZATION: For time range statistics on large MPAI files,
                            # use row-based streaming instead of time-based (avoids reading entire time column)
                            start_time, end_time = time_range
                            full_count = metadata.get('full_count', len(x_data))
                            
                            # Check if time range is within preview data (first 10k rows)
                            # Preview time range: x_data[0] to x_data[-1]
                            preview_max_time = x_data[-1] if len(x_data) > 0 else 0
                            preview_min_time = x_data[0] if len(x_data) > 0 else 0
                            
                            if start_time >= preview_min_time and end_time <= preview_max_time:
                                # Time range is within preview - use Python (fast, already in memory)
                                logger.debug(f"[STATS PERF] Using preview data for time range [{start_time:.2f}, {end_time:.2f}]")
                                mask = (x_data >= start_time) & (x_data <= end_time)
                                if np.any(mask):
                                    y_subset = y_data[mask]
                                    x_subset = x_data[mask]
                                    stats = self._calculate_signal_statistics(y_subset, x_subset, duty_cycle_threshold_mode, duty_cycle_threshold_value)
                                    results[name] = stats
                                    continue
                            else:
                                # ✅ NEW: Use FastStatsCalculator for instant statistics (< 16ms)
                                # This uses pre-aggregated chunk metadata for O(1) complete chunks
                                # and only loads edge data from disk
                                try:
                                    logger.debug(f"[FAST STATS] Using FastStatsCalculator for time range [{start_time:.2f}, {end_time:.2f}]")
                                    
                                    # Get MPAI reader from raw_dataframe
                                    mpai_reader = self.raw_dataframe
                                    
                                    # Use FastStatsCalculator with time range
                                    fast_stats = time_graph_cpp.FastStatsCalculator.calculate_time_range_statistics(
                                        mpai_reader, name, start_time, end_time
                                    )
                                    
                                    logger.debug(f"[FAST STATS] Complete chunks: {fast_stats.complete_chunks}, "
                                               f"Partial chunks: {fast_stats.partial_chunks}, "
                                               f"Rows loaded: {fast_stats.rows_loaded}")
                                    
                                    stats = {
                                        'count': fast_stats.count,
                                        'mean': fast_stats.mean,
                                        'std': fast_stats.std_dev,
                                        'min': fast_stats.min,
                                        'max': fast_stats.max,
                                        'rms': fast_stats.rms,
                                        'peak_to_peak': fast_stats.max - fast_stats.min,
                                        'median': fast_stats.mean,  # Approximation
                                    }
                                    
                                    threshold = stats['mean'] if duty_cycle_threshold_mode == "auto" else duty_cycle_threshold_value
                                    stats['duty_cycle'] = 50.0
                                    stats['duty_cycle_threshold'] = threshold
                                    
                                    results[name] = stats
                                    continue
                                    
                                except Exception as e:
                                    logger.warning(f"[FAST STATS] FastStatsCalculator failed for {name}: {e}, falling back to streaming")
                                    # Fall back to old streaming method
                                    # Time range is outside preview - calculate row indices efficiently
                                    # Estimate sample rate from preview data
                                    if len(x_data) > 1:
                                        sample_rate = (len(x_data) - 1) / (x_data[-1] - x_data[0]) if (x_data[-1] - x_data[0]) > 0 else 1.0
                                    else:
                                        sample_rate = 1.0
                                    
                                    # Convert time range to row indices
                                    start_row = max(0, int(start_time * sample_rate))
                                    end_row = min(full_count, int(end_time * sample_rate))
                                    row_count = max(1, end_row - start_row)
                                    
                                    logger.debug(f"[STATS FALLBACK] Streaming calculation for rows {start_row} to {end_row} ({row_count} rows)")
                                    cpp_stats = time_graph_cpp.StatisticsEngine.calculate_streaming(
                                        self.raw_dataframe, name, int(start_row), int(row_count)
                                    )
                                    
                                    stats = {
                                        'count': cpp_stats.count,
                                        'mean': cpp_stats.mean,
                                        'std': cpp_stats.std_dev,
                                        'min': cpp_stats.min,
                                        'max': cpp_stats.max,
                                        'rms': cpp_stats.rms,
                                        'peak_to_peak': cpp_stats.peak_to_peak,
                                        'median': cpp_stats.median if cpp_stats.median != 0.0 else cpp_stats.mean,
                                    }
                                    
                                    threshold = stats['mean'] if duty_cycle_threshold_mode == "auto" else duty_cycle_threshold_value
                                    stats['duty_cycle'] = 50.0
                                    stats['duty_cycle_threshold'] = threshold
                                    
                                    results[name] = stats
                                    continue
                        else:
                            # No time range - calculate statistics for FULL dataset
                            # This ensures 100% accuracy for min/max/mean/std values
                            full_count = metadata.get('full_count', len(x_data))
                            logger.debug(f"[STATS] Full dataset statistics for {name} ({full_count} rows)")
                            
                            cpp_stats = time_graph_cpp.StatisticsEngine.calculate_streaming(
                                self.raw_dataframe, name, 0, int(full_count)
                            )
                            
                            stats = {
                                'count': cpp_stats.count,
                                'mean': cpp_stats.mean,
                                'std': cpp_stats.std_dev,
                                'min': cpp_stats.min,
                                'max': cpp_stats.max,
                                'rms': cpp_stats.rms,
                                'peak_to_peak': cpp_stats.peak_to_peak,
                                'median': cpp_stats.median if cpp_stats.median != 0.0 else cpp_stats.mean,
                            }
                            
                            threshold = stats['mean'] if duty_cycle_threshold_mode == "auto" else duty_cycle_threshold_value
                            stats['duty_cycle'] = 50.0
                            stats['duty_cycle_threshold'] = threshold
                            
                            results[name] = stats
                            continue
                        
                    except Exception as e:
                        logger.error(f"C++ stats EXCEPTION for {name}: {e}", exc_info=True)
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
            for name in signal_names:
                if name not in self.signal_data:
                    continue
                    
                info = self.signal_data[name]
                metadata = info.get('metadata', {})
                
                # Check if memory mapped MPAI
                if metadata.get('memory_mapped', False) and 'mpai_reader' in info:
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
            if signal_name not in self.signal_data:
                logger.warning(f"Signal {signal_name} not found in signal_data")
                return None
            
            signal_info = self.signal_data[signal_name]
            metadata = signal_info.get('metadata', {})
            is_memory_mapped = metadata.get('memory_mapped', False)
            
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
            
            is_mpai = metadata.get('mpai', False)
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
            if signal_name not in self.signal_data:
                return None
            
            signal_info = self.signal_data[signal_name]
            metadata = signal_info.get('metadata', {})
            is_memory_mapped = metadata.get('memory_mapped', False)
            
            # ========== MEMORY-MAPPED MPAI: Use streaming reader ==========
            if is_memory_mapped:
                reader = signal_info.get('mpai_reader')
                if reader is None:
                    logger.error(f"Memory-mapped signal '{signal_name}' has no mpai_reader")
                    return None
                
                col_name = signal_info.get('column_name', signal_name)
                time_col = signal_info.get('time_column', 'time')
                
                try:
                    import time_graph_cpp
                    # Use C++ streaming stats for the range
                    cpp_stats = time_graph_cpp.StatisticsEngine.calculate_time_range_streaming(
                        reader, col_name, time_col, start_time, end_time
                    )
                    
                    return {
                        'x_data': None,  # Not available for streaming
                        'y_data': None,
                        'statistics': {
                            'mean': cpp_stats.mean,
                            'std': cpp_stats.std_dev,
                            'min': cpp_stats.min,
                            'max': cpp_stats.max,
                            'median': cpp_stats.median,
                            'rms': cpp_stats.rms,
                            'peak_to_peak': cpp_stats.peak_to_peak,
                            'count': cpp_stats.count,
                        }
                    }
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
    
    def _clear_cache(self, signal_name: Optional[str] = None):
        """Clear caches for specific signal or all signals."""
        if signal_name:
            # Clear caches for specific signal
            keys_to_remove = [key for key in self.normalized_data.keys() if key.startswith(f"{signal_name}_")]
            for key in keys_to_remove:
                del self.normalized_data[key]
            
            if signal_name in self.statistics_cache:
                del self.statistics_cache[signal_name]
        else:
            # Clear all caches
            self.normalized_data.clear()
            self.statistics_cache.clear()
    
    def clear_all_data(self):
        """Clear all signal data and caches."""
        with QMutexLocker(self.mutex):
            self.signal_data.clear()
            self.original_signal_data.clear()
            self.normalized_data.clear()
            self.statistics_cache.clear()
            
            # PERFORMANCE: Clear Polars cache
            self.numpy_cache.clear()
            self.raw_dataframe = None
            self.time_column_name = None
            
            logger.info("Cleared all signal data and caches")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        with QMutexLocker(self.mutex):
            signal_memory = sum(
                data['x_data'].nbytes + data['y_data'].nbytes + data['original_y'].nbytes
                for data in self.signal_data.values()
            )
            
            cache_memory = sum(
                arr.nbytes for arr in self.normalized_data.values()
            )
            
            return {
                'signal_data_bytes': signal_memory,
                'cache_bytes': cache_memory,
                'total_bytes': signal_memory + cache_memory,
                'signal_count': len(self.signal_data)
            }
    
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
    
    # ==========================================================================
    # LOD (Level of Detail) Support for Spike-Safe Visualization
    # ==========================================================================
    
    def _get_lod_render_data(
        self,
        signal_name: str,
        visible_samples: int,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get spike-safe render data from pre-computed LOD files.
        
        Uses C++ LodReader for zero-copy memory-mapped access to .tlod files.
        Automatically selects appropriate LOD level based on visible sample count.
        
        Args:
            signal_name: Signal column name
            visible_samples: Number of samples in visible range
            time_start: Optional start time for range query
            time_end: Optional end time for range query
            
        Returns:
            Dict with 'x_data' and 'y_data' (2 points per bucket: min/max)
            Returns None if LOD not available
        """
        if not HAS_CPP_LOD:
            return None
        
        # Determine LOD container path
        if not self.lod_container_path and self.current_mpai_path:
            # LOD files are stored in the MPAI container directory
            if os.path.isdir(self.current_mpai_path):
                self.lod_container_path = self.current_mpai_path
            else:
                self.lod_container_path = os.path.dirname(self.current_mpai_path)
        
        if not self.lod_container_path:
            return None
        
        try:
            # Get recommended bucket size from C++
            bucket_size = tgcpp.get_lod_bucket_size(visible_samples)
            
            if bucket_size == 0:
                # Use raw data
                return None
            
            # Get LOD filename
            lod_filename = tgcpp.get_lod_filename(bucket_size)
            lod_path = os.path.join(self.lod_container_path, lod_filename)
            
            if not os.path.exists(lod_path):
                logger.debug(f"[LOD] File not found: {lod_path}")
                return None
            
            # Get or create cached reader
            if bucket_size not in self.lod_readers:
                reader = tgcpp.LodReader()
                if not reader.open(lod_path):
                    logger.warning(f"[LOD] Failed to open: {lod_path}")
                    return None
                self.lod_readers[bucket_size] = reader
            
            reader = self.lod_readers[bucket_size]
            
            # Find column index
            col_idx = reader.get_column_index(signal_name)
            if col_idx < 0:
                logger.debug(f"[LOD] Column '{signal_name}' not found in LOD")
                return None
            
            # Determine bucket range
            if time_start is not None and time_end is not None:
                start_bucket, bucket_count = reader.find_bucket_range(time_start, time_end)
            else:
                start_bucket = 0
                bucket_count = reader.get_bucket_count()
            
            if bucket_count == 0:
                return None
            
            # Get spike-safe render data (returns tuple of time, values)
            x_data, y_data = reader.get_render_data(col_idx, start_bucket, bucket_count)
            
            if len(x_data) == 0:
                return None
            
            logger.debug(f"[LOD] Loaded {len(x_data)} points from LOD bucket_size={bucket_size} for '{signal_name}'")
            
            return {
                'x_data': np.array(x_data, dtype=np.float64),
                'y_data': np.array(y_data, dtype=np.float64)
            }
            
        except Exception as e:
            logger.warning(f"[LOD] Error reading LOD for '{signal_name}': {e}")
            return None
    
    def close_lod_readers(self):
        """Close all cached LOD readers."""
        for reader in self.lod_readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self.lod_readers.clear()

