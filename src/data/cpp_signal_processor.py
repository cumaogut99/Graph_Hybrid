"""
Signal Processor
Pure Python/NumPy/Polars implementation — no C++ dependency.
"""

import logging
import time
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)


class CppSignalProcessor(QObject):
    """
    Signal processor using NumPy/Polars.
    Provides the same API as the previous C++-backed version.
    """

    processing_started = Signal()
    processing_finished = Signal()
    statistics_updated = Signal(dict)
    performance_log = Signal(dict)

    def __init__(self, use_cpp: bool = True):  # use_cpp ignored — kept for API compat
        super().__init__()
        self.signal_data: dict = {}
        self.raw_dataframe = None
        self.time_column_name: Optional[str] = None
        self._stats_cache: dict = {}
        self._performance_metrics: dict = {}

    # ------------------------------------------------------------------
    def calculate_statistics(
        self,
        signal_name: str,
        range_start: Optional[float] = None,
        range_end: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        cache_key = (signal_name, range_start, range_end, threshold)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        t0 = time.perf_counter()
        stats = self._calculate_numpy(signal_name, range_start, range_end, threshold)
        elapsed = (time.perf_counter() - t0) * 1000

        stats["_performance"] = {"method": "numpy", "calc_time_ms": elapsed}
        self._stats_cache[cache_key] = stats
        if len(self._stats_cache) > 100:
            self._stats_cache.pop(next(iter(self._stats_cache)))

        logger.debug(f"[STATS] {signal_name}: {elapsed:.1f}ms")
        return stats

    # ------------------------------------------------------------------
    def _calculate_numpy(
        self,
        signal_name: str,
        range_start: Optional[float],
        range_end: Optional[float],
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        if self.raw_dataframe is None:
            raise ValueError("No data loaded")

        # ---- MPAI streaming reader path ----
        if hasattr(self.raw_dataframe, "get_header"):
            return self._calculate_from_mpai(signal_name, range_start, range_end, threshold)

        # ---- Polars DataFrame path ----
        if signal_name not in self.raw_dataframe.columns:
            raise ValueError(f"Signal '{signal_name}' not found")

        data = self.raw_dataframe[signal_name].to_numpy()

        if range_start is not None and range_end is not None and self.time_column_name:
            t = self.raw_dataframe[self.time_column_name].to_numpy()
            data = data[(t >= range_start) & (t <= range_end)]

        return self._numpy_stats(data, threshold, self.raw_dataframe, self.time_column_name)

    # ------------------------------------------------------------------
    def _calculate_from_mpai(
        self,
        signal_name: str,
        range_start: Optional[float],
        range_end: Optional[float],
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        reader = self.raw_dataframe
        row_count = reader.get_row_count()
        start_row, count = 0, row_count

        if range_start is not None and range_end is not None and self.time_column_name:
            try:
                preview = self.signal_data.get(signal_name, {}).get("x_data", [])
                if len(preview) > 1:
                    duration = max(preview[-1] - preview[0], 1e-9)
                    sr = (len(preview) - 1) / duration
                else:
                    sr = 1.0
                start_row = max(0, int(range_start * sr))
                end_row = min(row_count, int(range_end * sr))
                count = max(0, end_row - start_row)
            except Exception:
                start_row, count = 0, row_count

        raw = reader.load_column_slice(signal_name, start_row, start_row + count)
        data = np.asarray(raw, dtype=np.float64)
        return self._numpy_stats(data, threshold, None, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _numpy_stats(
        data: np.ndarray,
        threshold: Optional[float],
        df,
        time_col: Optional[str],
    ) -> Dict[str, Any]:
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "median": 0.0, "sum": 0.0, "rms": 0.0, "peak_to_peak": 0.0,
                "count": len(data), "valid_count": 0,
            }

        stats: Dict[str, Any] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median": float(np.median(valid)),
            "sum": float(np.sum(valid)),
            "rms": float(np.sqrt(np.mean(valid ** 2))),
            "peak_to_peak": float(np.max(valid) - np.min(valid)),
            "count": len(data),
            "valid_count": len(valid),
        }

        if threshold is not None:
            above = valid > threshold
            stats["threshold"] = threshold
            stats["above_count"] = int(np.sum(above))
            stats["below_count"] = int(len(valid) - stats["above_count"])
            stats["above_percentage"] = stats["above_count"] / len(valid) * 100
            stats["below_percentage"] = stats["below_count"] / len(valid) * 100

            if df is not None and time_col is not None:
                t = df[time_col].to_numpy()
                if len(t) > 1:
                    dt = np.diff(t)
                    above_short = above[: len(dt)]
                    stats["time_above"] = float(np.sum(dt[above_short]))
                    stats["time_below"] = float(np.sum(dt) - stats["time_above"])
                else:
                    stats["time_above"] = stats["time_below"] = 0.0
            else:
                stats["time_above"] = stats["time_below"] = 0.0

        return stats

    # ------------------------------------------------------------------
    def set_data(self, df: pl.DataFrame, time_column: str):
        self.raw_dataframe = df
        self.time_column_name = time_column
        self.clear_cache()

    def clear_cache(self):
        self._stats_cache.clear()

    @staticmethod
    def is_cpp_available() -> bool:
        return False
