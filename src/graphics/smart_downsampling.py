"""
Smart Downsampling Module

Provides intelligent downsampling for graph rendering with critical points preservation.
Uses C++ LTTB (Largest Triangle Three Buckets) algorithm for speed.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import C++ downsampling functions
try:
    import time_graph_cpp as tgcpp
    CPP_DOWNSAMPLE_AVAILABLE = True
    logger.info("[DOWNSAMPLE] C++ smart downsampling available")
except ImportError:
    CPP_DOWNSAMPLE_AVAILABLE = False
    logger.warning("[DOWNSAMPLE] C++ module not available, using basic fallback")


class SmartDownsampler:
    """
    Smart downsampling with critical points preservation.
    
    Strategies:
    1. Auto mode: Adapts based on data size and context
    2. LTTB mode: Fast visual downsampling
    3. Critical mode: LTTB + critical points (peaks, valleys, limits)
    """
    
    def __init__(self, screen_width: int = 1920):
        """
        Initialize downsampler.
        
        Args:
            screen_width: Screen width in pixels (default: 1920)
        """
        self.screen_width = screen_width
        self.target_points = screen_width * 2  # 2x screen width for smooth rendering
        
    def should_downsample(self, data_length: int) -> bool:
        """
        Determine if downsampling is needed.
        
        Args:
            data_length: Number of data points
            
        Returns:
            True if downsampling is beneficial
        """
        return data_length > self.target_points
    
    def downsample_auto(
        self,
        time_data: np.ndarray,
        signal_data: np.ndarray,
        has_limits: bool = False,
        limits: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Auto-adaptive downsampling.
        
        Chooses strategy based on:
        - Data size
        - Presence of limits
        - Available resources
        
        Args:
            time_data: Time values (NumPy array)
            signal_data: Signal values (NumPy array)
            has_limits: Whether static limits are active
            limits: Dict with 'min' and 'max' limit values (optional)
            
        Returns:
            Tuple of (time_downsampled, signal_downsampled, info_dict)
        """
        data_length = len(time_data)
        
        # No downsampling needed
        if not self.should_downsample(data_length):
            return time_data, signal_data, {
                'downsampled': False,
                'original_points': data_length,
                'final_points': data_length,
                'strategy': 'none'
            }
        
        # Use C++ if available
        if CPP_DOWNSAMPLE_AVAILABLE:
            try:
                result = tgcpp.downsample_auto(
                    time_data,
                    signal_data,
                    self.screen_width,
                    has_limits
                )
                
                time_ds = np.array(result.time)
                signal_ds = np.array(result.value)
                
                strategy = 'lttb+critical' if has_limits else 'lttb'
                
                logger.info(
                    f"[DOWNSAMPLE] Auto: {data_length:,} → {len(time_ds):,} points "
                    f"({strategy}, critical={result.critical_count})"
                )
                
                return time_ds, signal_ds, {
                    'downsampled': True,
                    'original_points': data_length,
                    'final_points': len(time_ds),
                    'critical_points': result.critical_count,
                    'strategy': strategy
                }
                
            except Exception as e:
                logger.warning(f"[DOWNSAMPLE] C++ auto failed: {e}, using fallback")
        
        # Python fallback (simple decimation)
        return self._fallback_downsample(time_data, signal_data)
    
    def downsample_lttb(
        self,
        time_data: np.ndarray,
        signal_data: np.ndarray,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        LTTB (Largest Triangle Three Buckets) downsampling.
        
        Fast visual downsampling that preserves shape.
        
        Args:
            time_data: Time values
            signal_data: Signal values
            max_points: Target number of points (default: 2x screen width)
            
        Returns:
            Tuple of (time_downsampled, signal_downsampled, info_dict)
        """
        if max_points is None:
            max_points = self.target_points
        
        data_length = len(time_data)
        
        if data_length <= max_points:
            return time_data, signal_data, {
                'downsampled': False,
                'original_points': data_length,
                'final_points': data_length,
                'strategy': 'none'
            }
        
        if CPP_DOWNSAMPLE_AVAILABLE:
            try:
                result = tgcpp.downsample_lttb(time_data, signal_data, max_points)
                
                time_ds = np.array(result.time)
                signal_ds = np.array(result.value)
                
                logger.info(
                    f"[DOWNSAMPLE] LTTB: {data_length:,} → {len(time_ds):,} points"
                )
                
                return time_ds, signal_ds, {
                    'downsampled': True,
                    'original_points': data_length,
                    'final_points': len(time_ds),
                    'strategy': 'lttb'
                }
                
            except Exception as e:
                logger.warning(f"[DOWNSAMPLE] C++ LTTB failed: {e}, using fallback")
        
        return self._fallback_downsample(time_data, signal_data, max_points)
    
    def downsample_with_critical(
        self,
        time_data: np.ndarray,
        signal_data: np.ndarray,
        limits: Optional[Dict[str, float]] = None,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Smart downsampling with critical points preservation.
        
        Ensures no loss of important data:
        - Peaks and valleys
        - Sudden changes
        - Limit violations
        
        Args:
            time_data: Time values
            signal_data: Signal values
            limits: Dict with 'min' and 'max' limit values
            max_points: Target number of points (default: 2x screen width)
            
        Returns:
            Tuple of (time_downsampled, signal_downsampled, info_dict)
        """
        if max_points is None:
            max_points = self.target_points
        
        data_length = len(time_data)
        
        if data_length <= max_points:
            return time_data, signal_data, {
                'downsampled': False,
                'original_points': data_length,
                'final_points': data_length,
                'strategy': 'none'
            }
        
        if CPP_DOWNSAMPLE_AVAILABLE:
            try:
                # Configure critical points detection
                config = tgcpp.CriticalPointsConfig()
                config.detect_peaks = True
                config.detect_valleys = True
                config.detect_sudden_changes = True
                config.detect_limit_violations = (limits is not None)
                
                if limits:
                    config.warning_limits = [limits.get('min', float('-inf')), 
                                            limits.get('max', float('inf'))]
                
                config.max_points = 500  # Max 500 critical points
                
                # Run smart downsampling
                result = tgcpp.downsample_lttb_with_critical(
                    time_data, signal_data, max_points, config
                )
                
                time_ds = np.array(result.time)
                signal_ds = np.array(result.value)
                
                logger.info(
                    f"[DOWNSAMPLE] Critical: {data_length:,} → {len(time_ds):,} points "
                    f"(preserved {result.critical_count} critical points)"
                )
                
                return time_ds, signal_ds, {
                    'downsampled': True,
                    'original_points': data_length,
                    'final_points': len(time_ds),
                    'critical_points': result.critical_count,
                    'strategy': 'lttb+critical'
                }
                
            except Exception as e:
                logger.warning(f"[DOWNSAMPLE] C++ critical failed: {e}, using fallback")
        
        return self._fallback_downsample(time_data, signal_data, max_points)
    
    def _fallback_downsample(
        self,
        time_data: np.ndarray,
        signal_data: np.ndarray,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Simple decimation fallback (Python).
        
        Uses uniform sampling when C++ is not available.
        """
        if max_points is None:
            max_points = self.target_points
        
        data_length = len(time_data)
        
        if data_length <= max_points:
            return time_data, signal_data, {
                'downsampled': False,
                'original_points': data_length,
                'final_points': data_length,
                'strategy': 'none'
            }
        
        # Uniform decimation
        step = data_length // max_points
        indices = np.arange(0, data_length, step)
        
        # Always include last point
        if indices[-1] != data_length - 1:
            indices = np.append(indices, data_length - 1)
        
        time_ds = time_data[indices]
        signal_ds = signal_data[indices]
        
        logger.info(
            f"[DOWNSAMPLE] Fallback: {data_length:,} → {len(time_ds):,} points"
        )
        
        return time_ds, signal_ds, {
            'downsampled': True,
            'original_points': data_length,
            'final_points': len(time_ds),
            'strategy': 'decimation (fallback)'
        }


# Global downsampler instance
_downsampler = SmartDownsampler()


def downsample_for_plot(
    time_data: np.ndarray,
    signal_data: np.ndarray,
    has_limits: bool = False,
    limits: Optional[Dict[str, float]] = None,
    screen_width: int = 1920
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience function for plot downsampling.
    
    Uses auto-adaptive strategy by default.
    
    Args:
        time_data: Time values
        signal_data: Signal values
        has_limits: Whether static limits are active
        limits: Dict with 'min' and 'max' limit values
        screen_width: Screen width in pixels
        
    Returns:
        Tuple of (time_downsampled, signal_downsampled, info_dict)
    """
    global _downsampler
    
    if _downsampler.screen_width != screen_width:
        _downsampler = SmartDownsampler(screen_width)
    
    return _downsampler.downsample_auto(time_data, signal_data, has_limits, limits)
