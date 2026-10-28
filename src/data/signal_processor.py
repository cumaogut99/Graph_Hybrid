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
from typing import Dict, List, Optional
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal, QThread, QMutex, QMutexLocker
from src.data.signal_processing.views import SignalViewsMixin
from src.data.signal_processing.filtering import SignalFilteringMixin
from src.data.signal_processing.normalization import SignalNormalizationMixin
from src.data.signal_processing.statistics import SignalStatisticsMixin
from src.data.signal_processing.queries import SignalQueriesMixin

HAS_CPP_LOD = False  # Pure Python implementation
tgcpp = None

logger = logging.getLogger(__name__)

class SignalProcessor(SignalViewsMixin, SignalFilteringMixin, SignalNormalizationMixin, SignalStatisticsMixin, SignalQueriesMixin, QObject):
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
        # Recursive: public methods call each other while holding the lock
        # (e.g. get_signals_at_time -> get_signal_at_time)
        self.mutex = QMutex(QMutex.Recursive)  # Thread safety
        
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
        self.normalize_all = False  # Normalize every signal (display only)
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

        # Active range filter: time segments joined into one timeline, and
        # the signals not filtered yet (filtered on first use, see
        # apply_segment_filter)
        self._filter_segments = None
        self._filter_pending = set()
        
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

            # Signals are registered without data; their display data is
            # prepared when they are plotted (get_all_signals(names)).
            # Preparing every channel here took seconds and hundreds of MB
            # for files with 1000+ channels.
            return {name: {'metadata': data['metadata']} for name, data in self.signal_data.items()}
            
        finally:
            self.processing_finished.emit()
    
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
            self._ensure_filtered([name])
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
    
    def get_signal_names(self) -> List[str]:
        """Names of all signals, without preparing their data."""
        with QMutexLocker(self.mutex):
            return list(self.signal_data.keys())

    def get_all_signals(self, names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Get all signal data safely.
        
        For memory-mapped MPAI files: Returns DOWNSAMPLED view of FULL dataset
        For CSV files: Returns full data (legacy)

        Args:
            names: Only these signals (default: all). Preparing every signal
                is expensive for files with many channels; callers that need
                only some signals or only the names should not ask for all.
        """
        with QMutexLocker(self.mutex):
            self._ensure_filtered(names)
            result = {}
            if names is None:
                items = self.signal_data.items()
            else:
                items = [(n, self.signal_data[n]) for n in names if n in self.signal_data]
            
            for name, data in items:
                # Check if this is a memory-mapped signal
                if self._is_file_backed(data):
                    # ✅ CACHE CHECK: Return cached downsampled data if available
                    if name in self._downsampled_cache:
                        logger.debug(f"[CACHE] Using cached downsampled data for '{name}'")
                        cached = self._downsampled_cache[name]
                        result[name] = {
                            'x_data': cached['x_data'],
                            'y_data': self.normalize_for_display(name, cached['y_data']),
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
                            'y_data': self.normalize_for_display(name, downsampled['y_data']),
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
                    if 'y_data' in data:
                        result[name]['y_data'] = self.normalize_for_display(name, data['y_data'])
            
            return result
    
    @staticmethod
    def _is_file_backed(signal_info: Dict) -> bool:
        """
        True if the signal's data must be read from the memory-mapped MPAI
        file. Not while a range filter has replaced it with in-memory
        (concatenated) data: the file holds the unfiltered data on a
        different timeline.
        """
        return bool(signal_info.get('metadata', {}).get('memory_mapped')) and not signal_info.get('filtered')

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
            self._filter_segments = None
            self._filter_pending.clear()
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
    
    # ===================================================================