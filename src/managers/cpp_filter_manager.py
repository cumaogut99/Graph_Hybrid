"""
C++ Filter Manager Wrapper
Integrates C++ FilterEngine with existing Python infrastructure
"""

import logging
import os
import sys
import time
import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)

# ============================================================================
# DLL PATH FIX FOR WINDOWS + ARROW
# Arrow DLLs are in pyarrow.libs directory, need to add to search path
# ============================================================================
def _setup_arrow_dll_path():
    """Add pyarrow DLL directories to Windows DLL search path."""
    if sys.platform != 'win32':
        return  # Only needed on Windows
    
    try:
        import pyarrow
        lib_dirs = pyarrow.get_library_dirs()
        
        # Also add pyarrow package directory
        pyarrow_dir = os.path.dirname(pyarrow.__file__)
        
        for lib_dir in lib_dirs + [pyarrow_dir]:
            if os.path.isdir(lib_dir):
                try:
                    os.add_dll_directory(lib_dir)
                    logger.debug(f"Added DLL directory: {lib_dir}")
                except Exception as e:
                    logger.debug(f"Could not add DLL dir {lib_dir}: {e}")
    except ImportError:
        logger.debug("pyarrow not installed, skipping DLL path setup")
    except Exception as e:
        logger.debug(f"DLL path setup failed: {e}")

# Setup DLL paths before importing C++ module
_setup_arrow_dll_path()

# Try to import C++ module
try:
    import time_graph_cpp as tgcpp
    CPP_AVAILABLE = True
    logger.info(f"C++ module loaded successfully, Arrow available: {tgcpp.is_arrow_available()}")
except ImportError as e:
    CPP_AVAILABLE = False
    logger.warning(f"C++ module not available for FilterManager: {e}")


class CppFilterCalculationWorker(QObject):
    """
    C++ accelerated filter calculation worker.
    
    Yeni mimari:
    - MPAI dosyaları için C++ FilterEngine.calculate_streaming kullanır (diskten streaming)
    - Gerekirse NumPy fallback ile uyumlu kalır
    """
    
    finished = Signal(list)  # Emits calculated segments
    error = Signal(str)
    progress = Signal(int)  # Progress percentage
    performance_log = Signal(dict)  # Performance metrics
    
    def __init__(
        self,
        all_signals: dict,
        conditions: list,
        mpai_reader: Optional[object] = None,
        time_column_name: Optional[str] = None,
        use_cpp: bool = True,
    ):
        super().__init__()
        # Deep copy to avoid data race conditions (for fallback/Numpy path)
        self.all_signals = {
            k: {"x_data": v["x_data"], "y_data": v["y_data"]}
            for k, v in all_signals.items()
        }
        self.conditions = [c.copy() for c in conditions]
        self.should_stop = False
        self._is_running = False
        self.use_cpp = use_cpp and CPP_AVAILABLE and mpai_reader is not None
        self._performance_metrics = {}
        
        # MPAI streaming için C++ MpaiReader referansı
        self.mpai_reader = mpai_reader
        self.time_column_name = self._resolve_time_column(mpai_reader, time_column_name)
    
    def run(self):
        """Calculate filter segments."""
        logger.info("[CPP WORKER] run() method started in thread")
        try:
            self._is_running = True
            
            if self.should_stop:
                logger.info("[CPP WORKER] should_stop is True, returning early")
                return
            
            t_start = time.perf_counter()
            logger.info(f"[CPP WORKER] Starting calculation, use_cpp={self.use_cpp}")
            
            # Try C++ (MPAI streaming) first
            if self.use_cpp:
                try:
                    logger.info("[CPP WORKER] Calling _calculate_segments_cpp_streaming()")
                    segments = self._calculate_segments_cpp_streaming()
                    method = "cpp_arrow_bridge"
                    logger.info(f"[CPP WORKER] C++ Arrow bridge returned {len(segments)} segments")
                except Exception as e:
                    # NO FALLBACK TO NUMPY - this would use downsampled data!
                    # Re-raise to indicate filter failure
                    logger.error(f"[CPP WORKER] C++ Arrow bridge filter FAILED: {e}")
                    self._is_running = False
                    self.error.emit(f"C++ filter failed: {e}")
                    raise
            else:
                logger.info("[CPP WORKER] C++ not available, using NumPy")
                segments = self._calculate_segments_numpy()
                method = "numpy"
            
            t_end = time.perf_counter()
            calc_time = (t_end - t_start) * 1000  # ms
            
            # Performance logging
            point_count = 0
            if self.all_signals:
                first_key = next(iter(self.all_signals))
                point_count = len(self.all_signals[first_key].get("x_data", []))
            
            self._performance_metrics = {
                "method": method,
                "calc_time_ms": calc_time,
                "num_conditions": len(self.conditions),
                "num_segments": len(segments),
                "data_points": point_count,
            }
            
            logger.info(
                f"Filter calculation ({method}): {calc_time:.2f}ms, "
                f"{len(segments)} segments, {point_count} preview points"
            )
            
            self.performance_log.emit(self._performance_metrics)
            
            if not self.should_stop:
                logger.info(f"[CPP WORKER] Emitting finished signal with {len(segments)} segments")
                self.finished.emit(segments)
                
        except Exception as e:
            logger.error(f"Filter calculation error: {e}", exc_info=True)
            if not self.should_stop:
                # Hata durumunda UI'nin \"calculating filter segments\" ekranında takılı kalmaması için
                # hem error sinyali hem de boş bir finished sinyali gönder.
                try:
                    self.error.emit(str(e))
                finally:
                    try:
                        self.finished.emit([])
                    except Exception:
                        # Thread veya bağlantılar bu aşamada kopmuş olabilir, sessiz geç
                        pass
        finally:
            logger.info("[CPP WORKER] run() method finished")
            self._is_running = False

    def _resolve_time_column(self, reader: Optional[object], preferred: Optional[str]) -> str:
        """
        Robust time column resolution for C++ streaming path.
        - If preferred is valid in reader columns, use it.
        - Else try common names ('time','Time','timestamp', first column).
        """
        if reader is None or not hasattr(reader, "get_column_names"):
            return preferred or "time"
        try:
            cols = reader.get_column_names()
            if not cols:
                return preferred or "time"
            if preferred and preferred in cols:
                return preferred
            # Auto-detect common names
            common = ["time", "Time", "timestamp", "Timestamp", "datetime", "DateTime", "date", "Date"]
            for name in common:
                if name in cols:
                    logger.info(f"[FILTER][TIME] Auto-detected time column from MPAI: '{name}'")
                    return name
            # Fallback to first column
            fallback = cols[0]
            logger.warning(f"[FILTER][TIME] Preferred time column not found, using first column '{fallback}'")
            return fallback
        except Exception:
            return preferred or "time"
    
    def stop(self):
        """Stop the calculation."""
        self.should_stop = True
    
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._is_running
    
    # ===== C++ MPAI STREAMING PATH =====
    def _calculate_segments_cpp_streaming(self) -> list:
        """
        Use C++ FilterEngine with Arrow Bridge.
        
        Yeni yaklaşım:
        1. Python MpaiDirectoryReader'dan veriyi NumPy array olarak al (mmap zero-copy)
        2. Bu array'leri C++ filter_engine'e gönder
        3. C++ mask hesaplar ve segment'leri döner
        
        Bu sayede C++ MpaiReader tip uyumsuzluğu çözülür.
        """
        logger.info("[CPP STREAMING] Starting C++ Arrow Bridge filter calculation")
        
        if not self.conditions or self.mpai_reader is None:
            logger.warning("[CPP STREAMING] No conditions or mpai_reader is None, returning empty")
            return []
        
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ module 'time_graph_cpp' is not available")
        
        # Convert Python conditions to C++ FilterConditions
        logger.info(f"[CPP STREAMING] Converting {len(self.conditions)} conditions to C++ format")
        cpp_conditions = self._convert_conditions_to_cpp(self.conditions)

        # Store for retrieval by filter_data callback (for graph_renderer Y-filtering)
        self.cpp_conditions = cpp_conditions
        logger.info(f"[CPP STREAMING] ✅ Stored {len(cpp_conditions)} cpp_conditions for graph_renderer access")

        if not cpp_conditions:
            logger.warning("[CPP STREAMING] No valid C++ conditions created, returning empty")
            return []
        
        logger.info(f"[CPP STREAMING] Created {len(cpp_conditions)} C++ conditions")
        
        # =====================================================================
        # ARROW BRIDGE: Extract data from Python reader, send to C++
        # =====================================================================
        try:
            # Check if reader has Arrow bridge methods
            if hasattr(self.mpai_reader, 'get_arrow_array'):
                return self._calculate_via_arrow_bridge(cpp_conditions)
            else:
                # Fallback: try the old streaming method (will likely fail with type error)
                logger.warning("[CPP STREAMING] Reader doesn't have Arrow bridge, trying legacy method")
                return self._calculate_via_legacy_streaming(cpp_conditions)
        except Exception as e:
            logger.error(f"[CPP STREAMING] Arrow bridge failed: {e}")
            raise
    
    def _calculate_via_arrow_bridge(self, cpp_conditions: list) -> list:
        """Use Arrow bridge to pass data from Python reader to C++."""
        import numpy as np
        
        logger.info("[ARROW BRIDGE] Starting Arrow-based filter calculation")
        
        # Get time data
        time_column = self.time_column_name or "time"
        logger.info(f"[ARROW BRIDGE] Loading time column: {time_column}")
        
        try:
            # Load time data as numpy array (zero-copy from mmap)
            row_count = self.mpai_reader.get_row_count()
            time_data = self.mpai_reader.load_column_slice(time_column, 0, row_count)
            
            if isinstance(time_data, list):
                time_data = np.array(time_data, dtype=np.float64)
            
            logger.info(f"[ARROW BRIDGE] Loaded {len(time_data)} time values")
        except Exception as e:
            logger.warning(f"[ARROW BRIDGE] Failed to load time column '{time_column}': {e}")
            # Generate synthetic time
            row_count = self.mpai_reader.get_row_count()
            t_start, t_end = self.mpai_reader.get_time_range()
            time_data = np.linspace(t_start, t_end, row_count, dtype=np.float64)
            logger.info(f"[ARROW BRIDGE] Generated synthetic time: {len(time_data)} values")
        
        # Load column data for each condition
        column_data = {}
        for cond in cpp_conditions:
            col_name = cond.column_name
            if col_name not in column_data and col_name.lower() != "time":
                try:
                    data = self.mpai_reader.load_column_slice(col_name, 0, row_count)
                    if isinstance(data, list):
                        data = np.array(data, dtype=np.float64)
                    column_data[col_name] = data
                    logger.info(f"[ARROW BRIDGE] Loaded column '{col_name}': {len(data)} values")
                except Exception as e:
                    logger.warning(f"[ARROW BRIDGE] Failed to load column '{col_name}': {e}")
        
        # Add time data to column_data dict
        column_data[time_column] = time_data
        column_data["time"] = time_data  # Also add as 'time' for compatibility
        
        # Try C++ Arrow filter if available
        try:
            filter_engine = tgcpp.FilterEngine()
            
            # Check if calculate_segments_from_arrow exists
            if hasattr(filter_engine, 'calculate_segments_from_arrow'):
                logger.info("[ARROW BRIDGE] Using C++ calculate_segments_from_arrow")
                cpp_segments = filter_engine.calculate_segments_from_arrow(
                    time_data,
                    column_data, 
                    cpp_conditions
                )
                segments = [(seg.start_time, seg.end_time) for seg in cpp_segments]
                logger.info(f"[ARROW BRIDGE] C++ returned {len(segments)} segments")
                return segments
            else:
                logger.warning("[ARROW BRIDGE] calculate_segments_from_arrow not available, using NumPy")
                raise NotImplementedError("C++ Arrow function not available")
                
        except Exception as e:
            logger.warning(f"[ARROW BRIDGE] C++ failed: {e}, falling back to NumPy")
            return self._calculate_segments_from_arrays(time_data, column_data, cpp_conditions)
    
    def _calculate_segments_from_arrays(self, time_data: np.ndarray, column_data: dict, cpp_conditions: list) -> list:
        """Pure NumPy fallback when C++ is not available."""
        import numpy as np
        
        logger.info("[NUMPY FALLBACK] Calculating segments from arrays")
        
        length = len(time_data)
        combined_mask = np.ones(length, dtype=bool)
        
        for cond in cpp_conditions:
            col_name = cond.column_name
            
            if col_name not in column_data:
                logger.warning(f"[NUMPY FALLBACK] Column '{col_name}' not found, skipping")
                continue
            
            col_data = column_data[col_name]
            
            # Apply condition
            if cond.type == tgcpp.FilterType.GREATER:
                condition_mask = col_data >= cond.threshold
            elif cond.type == tgcpp.FilterType.LESS:
                condition_mask = col_data <= cond.threshold
            elif cond.type == tgcpp.FilterType.RANGE:
                condition_mask = (col_data >= cond.min_value) & (col_data <= cond.max_value)
            elif cond.type == tgcpp.FilterType.EQUAL:
                condition_mask = np.abs(col_data - cond.threshold) < 1e-10
            elif cond.type == tgcpp.FilterType.NOT_EQUAL:
                condition_mask = np.abs(col_data - cond.threshold) >= 1e-10
            else:
                condition_mask = np.ones(length, dtype=bool)
            
            # Combine with AND (all conditions must pass)
            combined_mask &= condition_mask
        
        # Convert mask to segments
        segments = []
        in_segment = False
        segment_start_time = 0.0
        
        for i in range(length):
            if combined_mask[i] and not in_segment:
                in_segment = True
                segment_start_time = time_data[i]
            elif not combined_mask[i] and in_segment:
                in_segment = False
                segments.append((segment_start_time, time_data[i-1]))
        
        if in_segment:
            segments.append((segment_start_time, time_data[-1]))
        
        logger.info(f"[NUMPY FALLBACK] Found {len(segments)} segments")
        return segments
    
    def _calculate_via_legacy_streaming(self, cpp_conditions: list) -> list:
        """Legacy method - try the old calculate_streaming (will likely fail)."""
        logger.info("[LEGACY] Trying legacy calculate_streaming (may fail with TypeError)")
        
        filter_engine = tgcpp.FilterEngine()
        row_count = 0  # 0 = all rows
        
        cpp_segments = filter_engine.calculate_streaming(
            self.mpai_reader,
            self.time_column_name,
            cpp_conditions,
            0,
            row_count,
        )
        
        segments = [(seg.start_time, seg.end_time) for seg in cpp_segments]
        return segments
    
    def _convert_conditions_to_cpp(self, conditions: list) -> list:
        """Convert Python filter conditions to C++ FilterCondition objects."""
        if not CPP_AVAILABLE:
            return []
        
        cpp_conditions = []
        
        for condition in conditions:
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            logger.info(f"[CPP CONVERT] Processing parameter: '{param_name}' with {len(ranges)} range(s)")
            
            for range_filter in ranges:
                cpp_cond = tgcpp.FilterCondition()
                cpp_cond.column_name = param_name
                
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                logger.info(f"[CPP CONVERT]   Range: type='{range_type}', operator='{operator}', value={value}")
                
                # Map Python filter format to C++ FilterType
                if range_type == 'lower':
                    if operator == '>=':
                        cpp_cond.type = tgcpp.FilterType.GREATER
                        cpp_cond.threshold = value - 1e-10  # Inclusive
                        logger.info(f"[CPP CONVERT]   -> C++ GREATER with threshold={cpp_cond.threshold} (value-1e-10)")
                    elif operator == '>':
                        cpp_cond.type = tgcpp.FilterType.GREATER
                        cpp_cond.threshold = value
                        logger.info(f"[CPP CONVERT]   -> C++ GREATER with threshold={cpp_cond.threshold}")
                    else:
                        logger.warning(f"[CPP CONVERT]   -> SKIPPED (unknown operator)")
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        cpp_cond.type = tgcpp.FilterType.LESS
                        cpp_cond.threshold = value + 1e-10  # Inclusive
                        logger.info(f"[CPP CONVERT]   -> C++ LESS with threshold={cpp_cond.threshold} (value+1e-10)")
                    elif operator == '<':
                        cpp_cond.type = tgcpp.FilterType.LESS
                        cpp_cond.threshold = value
                        logger.info(f"[CPP CONVERT]   -> C++ LESS with threshold={cpp_cond.threshold}")
                    else:
                        logger.warning(f"[CPP CONVERT]   -> SKIPPED (unknown operator)")
                        continue
                else:
                    logger.warning(f"[CPP CONVERT]   -> SKIPPED (unknown range type)")
                    continue
                
                # AND between conditions (uygulamada parametreler arası AND mantığı var)
                cpp_cond.op = tgcpp.FilterOperator.AND
                logger.info(f"[CPP CONVERT]   -> Operator: AND")
                
                cpp_conditions.append(cpp_cond)
        
        logger.info(f"[CPP CONVERT] Total C++ conditions created: {len(cpp_conditions)}")
        return cpp_conditions
    
    def _calculate_segments_numpy(self) -> list:
        """
        Fallback: Calculate segments using NumPy (original implementation).
        Same logic as original FilterCalculationWorker.
        """
        if not self.conditions or not self.all_signals:
            return []
        
        # Get time data
        time_data = None
        for signal_name, signal_data in self.all_signals.items():
            if 'x_data' in signal_data and len(signal_data['x_data']) > 0:
                time_data = np.array(signal_data['x_data'])
                break
        
        if time_data is None:
            return []
        
        # Create boolean mask
        combined_mask = np.ones(len(time_data), dtype=bool)
        
        # Apply each condition
        total_conditions = len(self.conditions)
        for idx, condition in enumerate(self.conditions):
            if self.should_stop:
                return []
            
            # Report progress
            progress = int((idx / total_conditions) * 100)
            self.progress.emit(progress)
            
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            if param_name not in self.all_signals:
                continue
            
            param_data = np.asarray(self.all_signals[param_name]['y_data'])
            
            # FIX: Shape mismatch prevention
            # Downsampled signals may have different lengths than time_data
            if len(param_data) != len(time_data):
                logger.warning(
                    f"[NUMPY FALLBACK] Shape mismatch: time_data={len(time_data)}, "
                    f"{param_name}={len(param_data)}. Skipping condition."
                )
                continue
            
            # Initialize with True so we can use AND logic
            condition_mask = np.ones(len(param_data), dtype=bool)
            
            # Apply all ranges (AND logic within parameter for intersection)
            # Example: Value >= 0 AND Value <= 5
            for range_filter in ranges:
                if self.should_stop:
                    return []
                
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                if range_type == 'lower':
                    if operator == '>=':
                        range_mask = param_data >= value
                    elif operator == '>':
                        range_mask = param_data > value
                    else:
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        range_mask = param_data <= value
                    elif operator == '<':
                        range_mask = param_data < value
                    else:
                        continue
                else:
                    continue
                
                condition_mask &= range_mask
            
            # Combine with overall mask (AND logic between parameters)
            combined_mask &= condition_mask
        
        # Find continuous segments
        segments = self._find_continuous_segments(time_data, combined_mask)
        
        return segments
    
    def _find_continuous_segments(self, time_data: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous time segments where mask is True."""
        if not np.any(mask):
            return []
        
        # Find indices where mask is True
        true_indices = np.where(mask)[0]
        
        if len(true_indices) == 0:
            return []
        
        # Find breaks in continuity
        breaks = np.where(np.diff(true_indices) > 1)[0]
        
        # Split into continuous segments
        segments = []
        start_idx = 0
        
        for break_idx in breaks:
            segment_indices = true_indices[start_idx:break_idx + 1]
            if len(segment_indices) > 0:
                start_time = time_data[segment_indices[0]]
                end_time = time_data[segment_indices[-1]]
                segments.append((start_time, end_time))
            start_idx = break_idx + 1
        
        # Add last segment
        segment_indices = true_indices[start_idx:]
        if len(segment_indices) > 0:
            start_time = time_data[segment_indices[0]]
            end_time = time_data[segment_indices[-1]]
            segments.append((start_time, end_time))
        
        return segments
    
    @staticmethod
    def is_cpp_available():
        """Check if C++ module is available."""
        return CPP_AVAILABLE

