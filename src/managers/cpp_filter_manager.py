"""
Filter Calculation Worker
Pure Python/NumPy implementation — no C++ dependency.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)


class CppFilterCalculationWorker(QObject):
    """
    Filter calculation worker using NumPy.

    Reads raw data directly from the Python MPAI reader (MpaiDirectoryReader)
    or from in-memory signal dicts. Produces time-segment lists where all
    filter conditions are satisfied.
    """

    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int)
    performance_log = Signal(dict)

    def __init__(
        self,
        all_signals: dict,
        conditions: list,
        mpai_reader: Optional[object] = None,
        time_column_name: Optional[str] = None,
        use_cpp: bool = True,   # ignored — kept for API compat
    ):
        super().__init__()
        self.all_signals = {
            k: {"x_data": v["x_data"], "y_data": v["y_data"]}
            for k, v in all_signals.items()
        }
        self.conditions = [c.copy() for c in conditions]
        self.should_stop = False
        self._is_running = False
        self._performance_metrics = {}
        self.mpai_reader = mpai_reader
        self.time_column_name = self._resolve_time_column(mpai_reader, time_column_name)

    # ------------------------------------------------------------------
    def run(self):
        logger.info("[FILTER WORKER] run() started")
        try:
            self._is_running = True
            if self.should_stop:
                return

            t_start = time.perf_counter()

            if self.mpai_reader is not None:
                segments = self._calculate_from_mpai_reader()
                method = "mpai_numpy"
            else:
                segments = self._calculate_from_signals()
                method = "numpy"

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            point_count = 0
            if self.all_signals:
                first = next(iter(self.all_signals))
                point_count = len(self.all_signals[first].get("x_data", []))

            self._performance_metrics = {
                "method": method,
                "calc_time_ms": elapsed_ms,
                "num_conditions": len(self.conditions),
                "num_segments": len(segments),
                "data_points": point_count,
            }
            logger.info(
                f"[FILTER] {method}: {elapsed_ms:.1f}ms → {len(segments)} segments"
            )
            self.performance_log.emit(self._performance_metrics)

            if not self.should_stop:
                self.finished.emit(segments)

        except Exception as e:
            logger.error(f"[FILTER WORKER] Error: {e}", exc_info=True)
            if not self.should_stop:
                self.error.emit(str(e))
                self.finished.emit([])
        finally:
            self._is_running = False

    # ------------------------------------------------------------------
    def _resolve_time_column(self, reader, preferred: Optional[str]) -> str:
        if reader is None or not hasattr(reader, "get_column_names"):
            return preferred or "time"
        try:
            cols = reader.get_column_names()
            if preferred and preferred in cols:
                return preferred
            for name in ["time", "Time", "timestamp", "Timestamp", "datetime", "date"]:
                if name in cols:
                    return name
            return cols[0] if cols else (preferred or "time")
        except Exception:
            return preferred or "time"

    # ------------------------------------------------------------------
    def _calculate_from_mpai_reader(self) -> list:
        """Load raw column data from MPAI reader and apply NumPy filter."""
        if not self.conditions:
            return []

        reader = self.mpai_reader
        row_count = reader.get_row_count() if hasattr(reader, "get_row_count") else 0

        # Load time column
        time_col = self.time_column_name or "time"
        try:
            time_data = np.asarray(reader.load_column_slice(time_col, 0, row_count), dtype=np.float64)
        except Exception as e:
            logger.warning(f"[FILTER] Cannot load time column '{time_col}': {e}")
            try:
                t0, t1 = reader.get_time_range()
                time_data = np.linspace(t0, t1, row_count, dtype=np.float64)
            except Exception:
                return []

        combined_mask = np.ones(len(time_data), dtype=bool)

        for condition in self.conditions:
            if self.should_stop:
                return []
            param = condition["parameter"]
            ranges = condition["ranges"]
            try:
                col_data = np.asarray(reader.load_column_slice(param, 0, row_count), dtype=np.float64)
            except Exception as e:
                logger.warning(f"[FILTER] Cannot load column '{param}': {e}")
                continue

            param_mask = self._apply_ranges(col_data, ranges)
            combined_mask &= param_mask

        return self._mask_to_segments(time_data, combined_mask)

    # ------------------------------------------------------------------
    def _calculate_from_signals(self) -> list:
        """Apply filter on in-memory downsampled signal data."""
        if not self.conditions or not self.all_signals:
            return []

        time_data = None
        for sig in self.all_signals.values():
            if sig.get("x_data") is not None and len(sig["x_data"]) > 0:
                time_data = np.asarray(sig["x_data"], dtype=np.float64)
                break
        if time_data is None:
            return []

        combined_mask = np.ones(len(time_data), dtype=bool)
        n = len(self.conditions)

        for idx, condition in enumerate(self.conditions):
            if self.should_stop:
                return []
            self.progress.emit(int(idx / n * 100))

            param = condition["parameter"]
            if param not in self.all_signals:
                continue
            col_data = np.asarray(self.all_signals[param]["y_data"], dtype=np.float64)
            if len(col_data) != len(time_data):
                logger.warning(f"[FILTER] Shape mismatch for '{param}', skipping")
                continue

            param_mask = self._apply_ranges(col_data, condition["ranges"])
            combined_mask &= param_mask

        return self._mask_to_segments(time_data, combined_mask)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_ranges(data: np.ndarray, ranges: list) -> np.ndarray:
        """
        Apply a list of range conditions with AND logic (intersection).
        Each range narrows the valid window further.
        """
        mask = np.ones(len(data), dtype=bool)
        for r in ranges:
            rtype = r.get("type", "")
            op = r.get("operator", "")
            val = r.get("value", 0.0)
            if rtype == "lower":
                if op == ">=":
                    mask &= data >= val
                elif op == ">":
                    mask &= data > val
            elif rtype == "upper":
                if op == "<=":
                    mask &= data <= val
                elif op == "<":
                    mask &= data < val
        return mask

    # ------------------------------------------------------------------
    @staticmethod
    def _mask_to_segments(time_data: np.ndarray, mask: np.ndarray) -> list:
        """Convert a boolean mask to (start_time, end_time) segments."""
        if not np.any(mask):
            return []
        true_idx = np.where(mask)[0]
        breaks = np.where(np.diff(true_idx) > 1)[0]
        segments = []
        start = 0
        for b in breaks:
            seg = true_idx[start : b + 1]
            segments.append((float(time_data[seg[0]]), float(time_data[seg[-1]])))
            start = b + 1
        seg = true_idx[start:]
        if len(seg):
            segments.append((float(time_data[seg[0]]), float(time_data[seg[-1]])))
        return segments

    def stop(self):
        self.should_stop = True

    def is_running(self) -> bool:
        return self._is_running

    @staticmethod
    def is_cpp_available() -> bool:
        return False
