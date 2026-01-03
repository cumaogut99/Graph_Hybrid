
import sys
import os
import time
import numpy as np
from PyQt5.QtCore import QObject

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.signal_processor import SignalProcessor
from src.data.data_reader import MpaiDirectoryReader

def test_integration():
    print("=== Testing SignalProcessor Integration with MpaiDirectoryReader ===")
    
    mpai_path = "test_reader_mpai/PerfTest.mpai"
    if not os.path.exists(mpai_path):
        print(f"ERROR: {mpai_path} does not exist. Run test_data_reader.py first.")
        return
        
    print(f"Loading {mpai_path}...")
    reader = MpaiDirectoryReader(mpai_path)
    
    processor = SignalProcessor()
    
    # 1. Process Data
    print("Processing data...")
    processor.process_data(reader, time_column="time")
    
    # Verify signals are registered
    signals = processor.get_all_signals()
    print(f"Registered signals: {list(signals.keys())}")
    assert "Ch1" in signals
    
    # 2. Calculate Statistics
    print("Calculating statistics (Full Range)...")
    stats_dict = processor.calculate_statistics(signal_names=["Ch1"])
    stats = stats_dict["Ch1"]
    
    print(f"Stats: {stats}")
    
    # Check for new keys
    assert "std" in stats, "Missing 'std' in stats"
    assert "duty_cycle" in stats, "Missing 'duty_cycle' in stats"
    
    # Check values (Sine wave Ch1)
    # Mean ~ 0, Std ~ 3.535 (RMS)
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Duty Cycle: {stats['duty_cycle']:.1f}%")
    
    assert abs(stats['mean']) < 0.1, f"Mean should be near 0, got {stats['mean']}"
    assert abs(stats['std'] - 3.535) < 0.1, f"Std should be near 3.535, got {stats['std']}"
    
    # 3. Calculate Statistics (Time Range)
    print("\nCalculating statistics (Time Range: 1.0 - 2.0s)...")
    stats_range_dict = processor.calculate_statistics(signal_names=["Ch1"], time_range=(1.0, 2.0))
    stats_range = stats_range_dict["Ch1"]
    print(f"Range Stats: {stats_range}")
    
    assert "std" in stats_range
    assert abs(stats_range['std'] - 3.535) < 0.1
    
    print("\n=== SUCCESS: Integration Test Passed ===")

if __name__ == "__main__":
    test_integration()
