"""
Display-only normalization of signals.

Part of SignalProcessor; moved from src/data/signal_processor.py without changes.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from PyQt5.QtCore import QMutexLocker

logger = logging.getLogger(__name__)


class SignalNormalizationMixin:
    """Display-only normalization of signals."""

    def apply_normalization(self, signal_names: Optional[List[str]] = None,
                          method: str = "peak") -> List[str]:
        """
        Show signals normalized.

        Display only: the stored data is not changed (memory-mapped MPAI
        signals have no in-memory data to change), so statistics and cursor
        values keep the real values. get_all_signals() and
        normalize_for_display() return the normalized values.

        Args:
            signal_names: Signals to normalize; None normalizes every signal,
                including ones loaded or calculated later
            method: Normalization method ('peak', 'rms', 'minmax', 'zscore')

        Returns:
            Names of the normalized signals
        """
        with QMutexLocker(self.mutex):
            self.normalization_method = method
            if signal_names is None:
                self.normalize_all = True
                signal_names = list(self.signal_data.keys())
            for name in signal_names:
                info = self.signal_data.get(name)
                if info is None:
                    continue
                info['normalized'] = True
                info['normalization_method'] = method
                info.pop('_norm_params', None)
            return [name for name in signal_names if name in self.signal_data]

    def remove_normalization(self, signal_names: Optional[List[str]] = None) -> List[str]:
        """
        Show the real values again.

        Args:
            signal_names: Signals to denormalize (None for all)

        Returns:
            Names of the denormalized signals
        """
        with QMutexLocker(self.mutex):
            if signal_names is None:
                self.normalize_all = False
                signal_names = list(self.signal_data.keys())
            for name in signal_names:
                info = self.signal_data.get(name)
                if info is not None:
                    info['normalized'] = False
            return [name for name in signal_names if name in self.signal_data]

    def is_normalized(self, signal_name: str) -> bool:
        """True if the signal is shown normalized."""
        info = self.signal_data.get(signal_name)
        if info is None:
            return False
        return info.get('normalized', self.normalize_all)

    def normalize_for_display(self, signal_name: str, y_data: np.ndarray) -> np.ndarray:
        """
        y_data (any part of the signal, e.g. a zoomed LOD slice) normalized
        with the signal's whole-data statistics; unchanged if not normalized.
        """
        with QMutexLocker(self.mutex):
            if not self.is_normalized(signal_name):
                return y_data
            offset, scale = self._normalization_params(signal_name)
            return (np.asarray(y_data, dtype=np.float64) - offset) / scale

    def _normalization_params(self, signal_name: str) -> Tuple[float, float]:
        """(offset, scale) of the signal's normalization, cached (mutex held)."""
        info = self.signal_data[signal_name]
        params = info.get('_norm_params')
        if params is None:
            if self._is_file_backed(info):
                # Min/max-bucket view of the full file: keeps the extremes
                view = self._downsampled_cache.get(signal_name) or                     self._get_downsampled_view_internal(signal_name, target_points=4000)
                reference = view['y_data'] if view else np.array([])
            else:
                reference = info.get('y_data', np.array([]))
            method = info.get('normalization_method', self.normalization_method)
            params = self._normalization_offset_scale(np.asarray(reference, dtype=np.float64), method)
            info['_norm_params'] = params
        return params

    def _normalization_offset_scale(self, data: np.ndarray, method: str) -> Tuple[float, float]:
        """(offset, scale) so that (data - offset) / scale is the normalized data."""
        data = data[np.isfinite(data)] if len(data) else data
        if len(data) == 0:
            return 0.0, 1.0
        if method == "peak":
            offset, scale = 0.0, float(np.max(np.abs(data)))
        elif method == "rms":
            offset, scale = 0.0, float(np.sqrt(np.mean(data ** 2)))
        elif method == "minmax":
            offset = float(np.min(data))
            scale = float(np.max(data)) - offset
        elif method == "zscore":
            offset, scale = float(np.mean(data)), float(np.std(data))
        else:
            logger.warning(f"Unknown normalization method: {method}")
            offset, scale = 0.0, 1.0
        return offset, (scale if scale != 0 else 1.0)

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
