"""
C++ Filter Manager Wrapper
Integrates C++ FilterEngine with existing Python infrastructure
"""

import logging
import time
import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)

# Try to import C++ module
try:
    import time_graph_cpp as tgcpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("C++ module not available for FilterManager")


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
                    method = "cpp_streaming"
                    logger.info(f"[CPP WORKER] C++ streaming returned {len(segments)} segments")
                except Exception as e:
                    logger.warning(f"C++ streaming filter failed, falling back to NumPy: {e}")
                    logger.info("[CPP WORKER] Calling _calculate_segments_numpy() as fallback")
                    segments = self._calculate_segments_numpy()
                    method = "numpy_fallback"
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
        Use C++ FilterEngine.calculate_streaming on MpaiReader.
        
        Bu yol, veriyi diskten chunk'lar halinde okuyup boolean mask'i
        hiç RAM'e tam olarak almadan segment hesaplar.
        """
        logger.info("[CPP STREAMING] Starting C++ streaming filter calculation")
        
        if not self.conditions or self.mpai_reader is None:
            logger.warning("[CPP STREAMING] No conditions or mpai_reader is None, returning empty")
            return []
        
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ module 'time_graph_cpp' is not available")
        
        # Convert Python conditions to C++ FilterConditions
        logger.info(f"[CPP STREAMING] Converting {len(self.conditions)} conditions to C++ format")
        cpp_conditions = self._convert_conditions_to_cpp(self.conditions)
        if not cpp_conditions:
            logger.warning("[CPP STREAMING] No valid C++ conditions created, returning empty")
            return []
        
        logger.info(f"[CPP STREAMING] Created {len(cpp_conditions)} C++ conditions")
        logger.info(f"[CPP STREAMING] Creating FilterEngine instance")
        filter_engine = tgcpp.FilterEngine()
        
        # PERFORMANCE: Limit to first 1M rows for preview (or full if < 1M)
        total_rows = self.mpai_reader.get_row_count() if hasattr(self.mpai_reader, 'get_row_count') else 0
        preview_limit = 1_000_000  # 1M rows for responsive UI
        
        logger.info(f"[CPP STREAMING] Total rows in file: {total_rows:,}")
        
        if total_rows > preview_limit:
            logger.info(f"[FILTER] Large file ({total_rows:,} rows), using preview mode ({preview_limit:,} rows)")
            row_count = preview_limit
        else:
            row_count = 0  # 0 = all rows
        
        # Start from row 0, row_count=0 => tüm satırlar
        logger.info(f"[CPP STREAMING] Calling filter_engine.calculate_streaming()")
        logger.info(f"[CPP STREAMING]   time_column: {self.time_column_name}")
        logger.info(f"[CPP STREAMING]   row_count: {row_count} (0=all)")
        
        cpp_segments = filter_engine.calculate_streaming(
            self.mpai_reader,
            self.time_column_name,
            cpp_conditions,
            0,
            row_count,
        )
        
        logger.info(f"[CPP STREAMING] calculate_streaming returned {len(cpp_segments)} segments")
        
        # Convert C++ segments to Python (start_time, end_time)
        segments = [(seg.start_time, seg.end_time) for seg in cpp_segments]
        logger.info(f"[CPP STREAMING] Converted to Python format: {len(segments)} segments")
        return segments
    
    def _convert_conditions_to_cpp(self, conditions: list) -> list:
        """Convert Python filter conditions to C++ FilterCondition objects."""
        if not CPP_AVAILABLE:
            return []
        
        cpp_conditions = []
        
        for condition in conditions:
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            for range_filter in ranges:
                cpp_cond = tgcpp.FilterCondition()
                cpp_cond.column_name = param_name
                
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                # Map Python filter format to C++ FilterType
                if range_type == 'lower':
                    if operator == '>=':
                        cpp_cond.type = tgcpp.FilterType.GREATER
                        cpp_cond.threshold = value - 1e-10  # Inclusive
                    elif operator == '>':
                        cpp_cond.type = tgcpp.FilterType.GREATER
                        cpp_cond.threshold = value
                    else:
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        cpp_cond.type = tgcpp.FilterType.LESS
                        cpp_cond.threshold = value + 1e-10  # Inclusive
                    elif operator == '<':
                        cpp_cond.type = tgcpp.FilterType.LESS
                        cpp_cond.threshold = value
                    else:
                        continue
                else:
                    continue
                
                # AND between conditions (uygulamada parametreler arası AND mantığı var)
                cpp_cond.op = tgcpp.FilterOperator.AND
                
                cpp_conditions.append(cpp_cond)
        
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
            condition_mask = np.zeros(len(param_data), dtype=bool)
            
            # Apply all ranges (OR logic within parameter)
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
                
                condition_mask |= range_mask
            
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

