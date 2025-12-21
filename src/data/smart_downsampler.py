#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Downsampler Python Wrapper
================================

C++ SmartDownsampler'ın Python-friendly wrapper'ı.
PyQtGraph ile entegrasyon için optimize edilmiştir.

Kullanım:
    from src.data.smart_downsampler import downsample, SmartDownsampler
    
    # Basit kullanım
    x_ds, y_ds = downsample(x_data, y_data, target_points=4000)
    
    # Threshold ile
    x_ds, y_ds = downsample(x_data, y_data, 
                            target_points=4000,
                            threshold_high=100.0,
                            threshold_low=-100.0)
    
    # Detaylı sonuç
    result = SmartDownsampler.downsample_with_stats(x_data, y_data, 4000)
    print(f"Spikes preserved: {result.spike_count}")

Performans:
    - 1M nokta → 4K nokta: <50ms
    - SIMD/AVX2 optimizasyonlu
    - Zero-copy NumPy entegrasyonu
"""

import numpy as np
from typing import Tuple, Optional, Union, NamedTuple
import logging

logger = logging.getLogger(__name__)

# C++ modülünü import et
try:
    import time_graph_cpp as _tg
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    logger.warning("C++ module not available, using Python fallback (slower)")


class DownsampleStats(NamedTuple):
    """Downsampling istatistikleri"""
    input_size: int
    output_size: int
    compression_ratio: float
    spike_count: int
    peak_count: int
    valley_count: int
    critical_count: int
    lttb_count: int


def downsample(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_points: int = 4000,
    threshold_high: Optional[float] = None,
    threshold_low: Optional[float] = None,
    return_indices: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Veriyi akıllı şekilde downsample et.
    
    Spike'ları ve görsel bütünlüğü koruyarak veri noktası sayısını azaltır.
    
    Args:
        x_data: Zaman/X değerleri (NumPy array)
        y_data: Sinyal/Y değerleri (NumPy array)
        target_points: Hedef çıktı boyutu (varsayılan: 4000)
        threshold_high: Üst spike eşiği (None = otomatik)
        threshold_low: Alt spike eşiği (None = otomatik)
        return_indices: True ise orijinal indeksleri de döndür
    
    Returns:
        (x_ds, y_ds) veya (x_ds, y_ds, indices) tuple
    
    Example:
        >>> x = np.arange(1_000_000, dtype=np.float64)
        >>> y = np.sin(x / 10000) + np.random.randn(1_000_000) * 0.1
        >>> x_ds, y_ds = downsample(x, y, 4000)
        >>> print(f"Reduced to {len(x_ds)} points")
    """
    # Input validation
    x_data = np.ascontiguousarray(x_data, dtype=np.float64)
    y_data = np.ascontiguousarray(y_data, dtype=np.float64)
    
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    # Downsampling gerekli değilse direkt döndür
    if len(x_data) <= target_points:
        if return_indices:
            return x_data, y_data, np.arange(len(x_data))
        return x_data, y_data
    
    if _HAS_CPP:
        # C++ implementasyonu kullan
        result = _tg.smart_downsample(
            x_data, y_data, 
            target_points,
            threshold_high,
            threshold_low
        )
        
        x_ds, y_ds = result.to_numpy()
        
        if return_indices:
            return x_ds, y_ds, np.array(result.original_indices)
        return x_ds, y_ds
    else:
        # Python fallback (çok daha yavaş)
        return _python_downsample(x_data, y_data, target_points, 
                                  threshold_high, threshold_low, return_indices)


def downsample_with_stats(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_points: int = 4000,
    threshold_high: Optional[float] = None,
    threshold_low: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, DownsampleStats]:
    """
    Downsample et ve detaylı istatistikler döndür.
    
    Args:
        x_data: Zaman/X değerleri
        y_data: Sinyal/Y değerleri
        target_points: Hedef çıktı boyutu
        threshold_high: Üst spike eşiği
        threshold_low: Alt spike eşiği
    
    Returns:
        (x_ds, y_ds, stats) tuple
    """
    x_data = np.ascontiguousarray(x_data, dtype=np.float64)
    y_data = np.ascontiguousarray(y_data, dtype=np.float64)
    
    if len(x_data) <= target_points:
        stats = DownsampleStats(
            input_size=len(x_data),
            output_size=len(x_data),
            compression_ratio=1.0,
            spike_count=0,
            peak_count=0,
            valley_count=0,
            critical_count=0,
            lttb_count=0
        )
        return x_data, y_data, stats
    
    if _HAS_CPP:
        result = _tg.smart_downsample(
            x_data, y_data,
            target_points,
            threshold_high,
            threshold_low
        )
        
        x_ds, y_ds = result.to_numpy()
        
        stats = DownsampleStats(
            input_size=result.input_size,
            output_size=result.output_size,
            compression_ratio=result.compression_ratio(),
            spike_count=result.spike_count,
            peak_count=result.peak_count,
            valley_count=result.valley_count,
            critical_count=result.critical_points_count,
            lttb_count=result.lttb_points_count
        )
        
        return x_ds, y_ds, stats
    else:
        x_ds, y_ds = _python_downsample(x_data, y_data, target_points,
                                        threshold_high, threshold_low, False)
        stats = DownsampleStats(
            input_size=len(x_data),
            output_size=len(x_ds),
            compression_ratio=len(x_ds) / len(x_data),
            spike_count=0, peak_count=0, valley_count=0,
            critical_count=0, lttb_count=len(x_ds)
        )
        return x_ds, y_ds, stats


class SmartDownsampler:
    """
    SmartDownsampler wrapper class.
    
    PyQtGraph PlotWidget ile kullanım için optimize edilmiştir.
    
    Example:
        >>> ds = SmartDownsampler(target_points=4000)
        >>> ds.set_thresholds(warning_high=100, warning_low=-100)
        >>> 
        >>> # Her plot güncellemesinde
        >>> x_ds, y_ds = ds.process(x_data, y_data)
        >>> plot.setData(x_ds, y_ds)
    """
    
    def __init__(self, 
                 target_points: int = 4000,
                 threshold_high: Optional[float] = None,
                 threshold_low: Optional[float] = None,
                 auto_threshold: bool = True):
        """
        Args:
            target_points: Hedef çıktı boyutu
            threshold_high: Üst spike eşiği (None = otomatik)
            threshold_low: Alt spike eşiği (None = otomatik)
            auto_threshold: Eşik belirtilmezse otomatik hesapla
        """
        self._target = target_points
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low
        self._auto_threshold = auto_threshold
        self._last_stats = None
        
        # C++ downsampler instance (varsa)
        if _HAS_CPP:
            self._cpp_ds = _tg.SmartDownsampler()
            self._config = _tg.SmartDownsampleConfig()
            self._update_config()
        else:
            self._cpp_ds = None
            self._config = None
    
    def _update_config(self):
        """C++ config'i güncelle"""
        if self._config is None:
            return
        
        self._config.target_points = self._target
        
        if self._threshold_high is not None:
            self._config.spike_threshold_high = self._threshold_high
            self._config.use_auto_threshold = False
        else:
            self._config.use_auto_threshold = self._auto_threshold
        
        if self._threshold_low is not None:
            self._config.spike_threshold_low = self._threshold_low
    
    def set_target(self, target_points: int):
        """Hedef nokta sayısını ayarla"""
        self._target = target_points
        self._update_config()
    
    def set_thresholds(self, 
                       warning_high: Optional[float] = None,
                       warning_low: Optional[float] = None):
        """Spike eşiklerini ayarla"""
        self._threshold_high = warning_high
        self._threshold_low = warning_low
        self._update_config()
    
    def process(self, 
                x_data: np.ndarray, 
                y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Veriyi downsample et.
        
        Args:
            x_data: Zaman/X değerleri
            y_data: Sinyal/Y değerleri
        
        Returns:
            (x_downsampled, y_downsampled) tuple
        """
        x_data = np.ascontiguousarray(x_data, dtype=np.float64)
        y_data = np.ascontiguousarray(y_data, dtype=np.float64)
        
        # Downsampling gerekli değilse
        if len(x_data) <= self._target:
            return x_data, y_data
        
        if self._cpp_ds is not None:
            result = self._cpp_ds.downsample(x_data, y_data, self._config)
            self._last_stats = result
            return result.to_numpy()
        else:
            return _python_downsample(x_data, y_data, self._target,
                                      self._threshold_high, self._threshold_low, False)
    
    def get_last_stats(self) -> Optional[DownsampleStats]:
        """Son işlemin istatistiklerini döndür"""
        if self._last_stats is None:
            return None
        
        return DownsampleStats(
            input_size=self._last_stats.input_size,
            output_size=self._last_stats.output_size,
            compression_ratio=self._last_stats.compression_ratio(),
            spike_count=self._last_stats.spike_count,
            peak_count=self._last_stats.peak_count,
            valley_count=self._last_stats.valley_count,
            critical_count=self._last_stats.critical_points_count,
            lttb_count=self._last_stats.lttb_points_count
        )
    
    @property
    def target_points(self) -> int:
        return self._target
    
    @property
    def threshold_high(self) -> Optional[float]:
        return self._threshold_high
    
    @property
    def threshold_low(self) -> Optional[float]:
        return self._threshold_low


def _python_downsample(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_points: int,
    threshold_high: Optional[float],
    threshold_low: Optional[float],
    return_indices: bool
):
    """
    Python fallback LTTB implementasyonu.
    
    C++ modülü yoksa kullanılır. Çok daha yavaştır.
    """
    n = len(x_data)
    
    if n <= target_points:
        if return_indices:
            return x_data, y_data, np.arange(n)
        return x_data, y_data
    
    # Basit LTTB implementasyonu
    indices = [0]  # İlk nokta
    
    bucket_size = (n - 2) / (target_points - 2)
    
    for i in range(target_points - 2):
        # Sonraki bucket ortalaması
        avg_start = int((i + 2) * bucket_size) + 1
        avg_end = min(int((i + 3) * bucket_size) + 1, n)
        
        if avg_end > avg_start:
            avg_x = np.mean(x_data[avg_start:avg_end])
            avg_y = np.mean(y_data[avg_start:avg_end])
        else:
            avg_x = x_data[-1]
            avg_y = y_data[-1]
        
        # Mevcut bucket
        range_start = int((i + 1) * bucket_size) + 1
        range_end = min(int((i + 2) * bucket_size) + 1, n)
        
        # En büyük üçgen alanını bul
        max_area = -1
        max_idx = range_start
        
        a_x = x_data[indices[-1]]
        a_y = y_data[indices[-1]]
        
        for j in range(range_start, range_end):
            area = abs((a_x - avg_x) * (y_data[j] - a_y) - 
                       (a_x - x_data[j]) * (avg_y - a_y)) * 0.5
            
            if area > max_area:
                max_area = area
                max_idx = j
        
        indices.append(max_idx)
    
    indices.append(n - 1)  # Son nokta
    
    # Threshold-based spikes ekle
    if threshold_high is not None or threshold_low is not None:
        spike_indices = []
        for i in range(n):
            if threshold_high is not None and y_data[i] > threshold_high:
                spike_indices.append(i)
            if threshold_low is not None and y_data[i] < threshold_low:
                spike_indices.append(i)
        
        indices = list(set(indices + spike_indices))
        indices.sort()
    
    indices = np.array(indices)
    
    if return_indices:
        return x_data[indices], y_data[indices], indices
    return x_data[indices], y_data[indices]


# Module-level convenience
def is_cpp_available() -> bool:
    """C++ modülünün kullanılabilir olup olmadığını kontrol et"""
    return _HAS_CPP


def get_backend_info() -> dict:
    """Backend bilgilerini döndür"""
    info = {
        "cpp_available": _HAS_CPP,
        "backend": "C++ (SIMD/AVX2)" if _HAS_CPP else "Python (fallback)",
    }
    
    if _HAS_CPP:
        info["simd"] = True
        info["parallel"] = True
    
    return info
