"""
Report data access: raw (full-resolution, real-valued) data of the active
file for the report range, with statistics and plot-ready downsampling.

Statistics always use the raw data; only the drawn curves are reduced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReportContext:
    """What the report knows about its data; collected in the GUI thread."""
    signal_processor: object
    file_name: str = ""
    parameters: List[str] = field(default_factory=list)
    start: Optional[float] = None       # None: from the beginning
    end: Optional[float] = None         # None: to the end
    datetime_axis: bool = False         # time values are epoch seconds


@dataclass
class Stats:
    count: int
    minimum: float
    mean: float
    maximum: float
    std: float
    rms: float


class ReportData:
    """Raw signal access for one report run (results cached per parameter)."""

    def __init__(self, context: ReportContext):
        self.context = context
        self._cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    def series(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(time, values) of a parameter in the report range, or None."""
        if name not in self._cache:
            result = self.context.signal_processor.get_raw_range(
                name, self.context.start, self.context.end)
            if result is not None and len(result[0]) == 0:
                result = None
            self._cache[name] = result
        return self._cache[name]

    def aligned(self, reference: str, others: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """
        Values of `others` at the sample times of `reference` (same time base:
        as is; other time base: interpolated, only where both have data).
        Returns {name: values} including the reference and '__time__'.
        """
        ref = self.series(reference)
        if ref is None:
            return None
        ref_t, ref_y = ref
        keep = np.ones(len(ref_t), dtype=bool)
        columns = {}
        for name in others:
            other = self.series(name)
            if other is None:
                return None
            t, y = other
            if len(t) == len(ref_t) and np.array_equal(t, ref_t):
                columns[name] = y
            else:
                if len(t) < 2:
                    return None
                keep &= (ref_t >= t[0]) & (ref_t <= t[-1])
                columns[name] = np.interp(ref_t, t, y)
        result = {'__time__': ref_t[keep], reference: ref_y[keep]}
        for name, values in columns.items():
            result[name] = values[keep]
        return result

    # ------------------------------------------------------------------
    # Facts about the data
    # ------------------------------------------------------------------
    def time_span(self, names: List[str]) -> Optional[Tuple[float, float, int]]:
        """(first time, last time, sample count) in range, from the first of
        `names` that has data (the parameters of a file share the time base)."""
        for name in names:
            s = self.series(name)
            if s is not None:
                return float(s[0][0]), float(s[0][-1]), len(s[0])
        return None


def statistics(values: np.ndarray) -> Optional[Stats]:
    """Statistics of the finite values, None if there are none."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return Stats(
        count=len(values),
        minimum=float(np.min(values)),
        mean=float(np.mean(values)),
        maximum=float(np.max(values)),
        std=float(np.std(values)),
        rms=float(np.sqrt(np.mean(values ** 2))),
    )


def envelope(t: np.ndarray, y: np.ndarray, buckets: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Min/max reduction for drawing a time trend: every bucket keeps its lowest
    and highest sample (in time order), so peaks stay visible.
    """
    n = len(t)
    if n <= buckets * 2:
        return t, y
    size = n // buckets
    usable = size * buckets
    yb = y[:usable].reshape(buckets, size)
    tb = t[:usable].reshape(buckets, size)
    rows = np.arange(buckets)
    i_min = np.nanargmin(np.where(np.isfinite(yb), yb, np.inf), axis=1)
    i_max = np.nanargmax(np.where(np.isfinite(yb), yb, -np.inf), axis=1)
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    t_out = np.column_stack((tb[rows, first], tb[rows, second])).ravel()
    y_out = np.column_stack((yb[rows, first], yb[rows, second])).ravel()
    if usable < n:
        t_out = np.append(t_out, t[-1])
        y_out = np.append(y_out, y[-1])
    return t_out, y_out


def thin(*arrays: np.ndarray, limit: int = 20000) -> Tuple[np.ndarray, ...]:
    """Every k-th sample so that at most `limit` remain (for scatter plots)."""
    n = len(arrays[0])
    if n <= limit:
        return arrays
    step = int(np.ceil(n / limit))
    return tuple(a[::step] for a in arrays)
