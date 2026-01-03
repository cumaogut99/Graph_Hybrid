"""
Memory Profiler - Track RAM usage during data operations

Provides real-time memory monitoring and profiling tools.
Target: Keep memory usage under 300 MB for 50 GB files.
"""

import logging
import time
import psutil
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size (physical memory)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory usage percentage
    available_mb: float  # Available system memory
    label: str = ""
    
    def __str__(self) -> str:
        return (f"Memory: {self.rss_mb:.2f} MB RSS, {self.vms_mb:.2f} MB VMS, "
                f"{self.percent:.1f}% used, {self.available_mb:.2f} MB available")


@dataclass
class MemoryProfile:
    """Memory profiling session results."""
    name: str
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage."""
        if not self.snapshots:
            return 0.0
        return max(s.rss_mb for s in self.snapshots)
    
    @property
    def memory_delta_mb(self) -> float:
        """Get memory increase from start to end."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        
        return {
            'name': self.name,
            'duration_ms': self.duration_ms,
            'snapshot_count': len(self.snapshots),
            'start_memory_mb': rss_values[0],
            'end_memory_mb': rss_values[-1],
            'peak_memory_mb': max(rss_values),
            'memory_delta_mb': self.memory_delta_mb,
            'avg_memory_mb': sum(rss_values) / len(rss_values),
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        if not summary:
            logger.info(f"Profile '{self.name}': No data")
            return
        
        logger.info(f"Profile '{self.name}' Summary:")
        logger.info(f"  Duration: {summary['duration_ms']:.2f} ms")
        logger.info(f"  Start Memory: {summary['start_memory_mb']:.2f} MB")
        logger.info(f"  End Memory: {summary['end_memory_mb']:.2f} MB")
        logger.info(f"  Peak Memory: {summary['peak_memory_mb']:.2f} MB")
        logger.info(f"  Delta: {summary['memory_delta_mb']:+.2f} MB")
        logger.info(f"  Average: {summary['avg_memory_mb']:.2f} MB")


class MemoryProfiler:
    """
    Memory profiler for tracking RAM usage.
    
    Usage:
        profiler = MemoryProfiler()
        
        # Start profiling
        profiler.start("data_loading")
        
        # ... do work ...
        profiler.snapshot("after_csv_load")
        
        # ... more work ...
        profiler.snapshot("after_processing")
        
        # End profiling
        profiler.stop()
        profiler.print_summary()
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.current_profile: Optional[MemoryProfile] = None
        self.profiles: List[MemoryProfile] = []
    
    def start(self, name: str = "profile") -> MemoryProfile:
        """
        Start a new profiling session.
        
        Args:
            name: Profile name
        
        Returns:
            MemoryProfile object
        """
        # Stop current profile if any
        if self.current_profile:
            self.stop()
        
        # Create new profile
        self.current_profile = MemoryProfile(name=name)
        
        # Take initial snapshot
        self.snapshot("start")
        
        logger.debug(f"Memory profiling started: {name}")
        return self.current_profile
    
    def snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            label: Snapshot label
        
        Returns:
            MemorySnapshot object
        """
        if not self.current_profile:
            raise RuntimeError("No active profile. Call start() first.")
        
        # Get memory info
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=self.process.memory_percent(),
            available_mb=sys_mem.available / (1024 * 1024),
            label=label
        )
        
        self.current_profile.snapshots.append(snapshot)
        
        logger.debug(f"Snapshot '{label}': {snapshot}")
        return snapshot
    
    def stop(self) -> MemoryProfile:
        """
        Stop current profiling session.
        
        Returns:
            Completed MemoryProfile
        """
        if not self.current_profile:
            raise RuntimeError("No active profile.")
        
        # Take final snapshot
        self.snapshot("end")
        
        # Mark end time
        self.current_profile.end_time = time.time()
        
        # Save to history
        self.profiles.append(self.current_profile)
        
        profile = self.current_profile
        self.current_profile = None
        
        logger.debug(f"Memory profiling stopped: {profile.name}")
        return profile
    
    def get_current_memory(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_system_memory(self) -> Dict:
        """
        Get system memory information.
        
        Returns:
            Dict with system memory stats
        """
        sys_mem = psutil.virtual_memory()
        
        return {
            'total_mb': sys_mem.total / (1024 * 1024),
            'available_mb': sys_mem.available / (1024 * 1024),
            'used_mb': sys_mem.used / (1024 * 1024),
            'percent': sys_mem.percent,
            'free_mb': sys_mem.free / (1024 * 1024),
        }
    
    def print_summary(self):
        """Print summary of current profile."""
        if self.current_profile:
            self.current_profile.print_summary()
        else:
            logger.warning("No active profile to summarize")
    
    def print_all_profiles(self):
        """Print summary of all profiles."""
        if not self.profiles:
            logger.info("No completed profiles")
            return
        
        logger.info(f"=== Memory Profiling Summary ({len(self.profiles)} profiles) ===")
        for profile in self.profiles:
            profile.print_summary()
            logger.info("")


def profile_memory(func: Callable) -> Callable:
    """
    Decorator for profiling function memory usage.
    
    Usage:
        @profile_memory
        def my_function():
            # ... code ...
    """
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start(func.__name__)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.stop()
            profiler.print_summary()
    
    return wrapper


class MemoryMonitor:
    """
    Continuous memory monitor with threshold alerts.
    
    Usage:
        monitor = MemoryMonitor(threshold_mb=1000)
        monitor.start()
        
        # ... do work ...
        
        monitor.stop()
    """
    
    def __init__(self, threshold_mb: float = 1000, interval_sec: float = 1.0):
        """
        Initialize memory monitor.
        
        Args:
            threshold_mb: Alert threshold in MB
            interval_sec: Monitoring interval in seconds
        """
        self.threshold_mb = threshold_mb
        self.interval_sec = interval_sec
        self.profiler = MemoryProfiler()
        self.monitoring = False
        self.snapshots: List[MemorySnapshot] = []
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.snapshots = []
        logger.info(f"Memory monitoring started (threshold: {self.threshold_mb} MB)")
    
    def check(self) -> Optional[MemorySnapshot]:
        """
        Check current memory usage.
        
        Returns:
            MemorySnapshot if monitoring, None otherwise
        """
        if not self.monitoring:
            return None
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=self.profiler.get_current_memory(),
            vms_mb=self.profiler.process.memory_info().vms / (1024 * 1024),
            percent=self.profiler.process.memory_percent(),
            available_mb=psutil.virtual_memory().available / (1024 * 1024),
        )
        
        self.snapshots.append(snapshot)
        
        # Check threshold
        if snapshot.rss_mb > self.threshold_mb:
            logger.warning(f"⚠️  Memory threshold exceeded: {snapshot.rss_mb:.2f} MB "
                          f"(threshold: {self.threshold_mb} MB)")
        
        return snapshot
    
    def stop(self) -> Dict:
        """
        Stop monitoring and return summary.
        
        Returns:
            Summary dict
        """
        self.monitoring = False
        
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        
        summary = {
            'duration_sec': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp),
            'snapshot_count': len(self.snapshots),
            'peak_memory_mb': max(rss_values),
            'avg_memory_mb': sum(rss_values) / len(rss_values),
            'threshold_exceeded': any(s.rss_mb > self.threshold_mb for s in self.snapshots),
        }
        
        logger.info(f"Memory monitoring stopped: Peak {summary['peak_memory_mb']:.2f} MB, "
                   f"Avg {summary['avg_memory_mb']:.2f} MB")
        
        return summary


# Global profiler instance
_global_profiler = MemoryProfiler()


def start_profiling(name: str = "profile"):
    """Start global profiling session."""
    return _global_profiler.start(name)


def snapshot(label: str = ""):
    """Take snapshot in global profiling session."""
    return _global_profiler.snapshot(label)


def stop_profiling():
    """Stop global profiling session."""
    return _global_profiler.stop()


def print_memory_summary():
    """Print global profiling summary."""
    _global_profiler.print_summary()


def get_current_memory_mb() -> float:
    """Get current process memory in MB."""
    return _global_profiler.get_current_memory()

