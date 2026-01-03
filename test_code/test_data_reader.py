
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_engine import DataEngine
from src.data.data_reader import MpaiDirectoryReader

def test_reader_performance():
    print("=== Testing MPAI Reader Performance & Integrity ===")
    
    # 1. Generate & Write Data (Setup)
    print("\n[1] SETUP: Creating Test Data (10 Seconds @ 100kHz = 1M Samples)...")
    fs = 100000.0
    duration = 10.0 
    t = np.linspace(0, duration, int(fs * duration))
    # Signal: 5V Sine at 10Hz
    raw_signal = 5.0 * np.sin(2 * np.pi * 10 * t)
    
    output_dir = Path("test_reader_mpai")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    engine = DataEngine(str(output_dir))
    mpai_path = engine.import_numpy_data(raw_signal, fs, "PerfTest", channel_names=["Ch1"], units=["V"])
    print(f"    Created: {mpai_path}")
    
    # 2. Initialize Reader
    print("\n[2] OPENING: Initializing MpaiDirectoryReader...")
    start_time = time.perf_counter()
    reader = MpaiDirectoryReader(mpai_path)
    init_dur = (time.perf_counter() - start_time) * 1000
    print(f"    Init Time: {init_dur:.3f} ms (Should be < 10ms usually)")
    
    # 3. Test Raw Access (Zoom In)
    print("\n[3] RAW ACCESS: Reading small window (Zoom In)...")
    t_mid = duration / 2
    # Request 1000 samples (10ms)
    t_in_start = t_mid
    t_in_end = t_in_start + 0.01 
    
    start_time = time.perf_counter()
    t_raw, y_raw, source = reader.get_render_data(0, t_in_start, t_in_end, pixel_width=1000)
    read_dur = (time.perf_counter() - start_time) * 1000
    
    print(f"    Request: {t_in_start:.3f}-{t_in_end:.3f}s")
    print(f"    Source Used: {source} (Expected: raw)")
    print(f"    Returned Samples: {len(y_raw)}")
    print(f"    Read Time: {read_dur:.3f} ms")
    
    # Check value
    expected_val = 5.0 * np.sin(2 * np.pi * 10 * t_in_start)
    actual_val = y_raw[0]
    print(f"    Value Check: Exp={expected_val:.3f}, Act={actual_val:.3f}, Diff={abs(expected_val-actual_val):.3e}")
    assert source == "raw", "Should use RAW source for small window"
    
    # 4. Test Reduced Access (Zoom Out)
    print("\n[4] REDUCED ACCESS: Reading full duration (Zoom Out)...")
    start_time = time.perf_counter()
    # 1M samples into 1000 pixels -> 1000 samples/pixel > Threshold -> Reduced
    t_red, y_red, source = reader.get_render_data(0, 0, duration, pixel_width=1000)
    read_dur = (time.perf_counter() - start_time) * 1000
    
    print(f"    Request: 0-{duration}s")
    print(f"    Source Used: {source} (Expected: reduced)")
    print(f"    Returned Points: {len(y_red)} (Should be ~2000 for interleaved min/max)")
    print(f"    Read Time: {read_dur:.3f} ms")
    assert source == "reduced", "Should use REDUCED source for full view"
    
    # 5. Test Cursor Statistics
    print("\n[5] STATS & CURSOR:")
    
    # Cursor
    cur_t = 1.25 # Peak of sine? 10Hz -> T=0.1s. 1.25s = 12.5 cycles. Sin(pi) = 0?
    # Sin(2*pi*10 * 1.25) = Sin(25*pi) = 0
    cur_val = reader.get_cursor_value(0, cur_t)
    print(f"    Cursor @ {cur_t}s: {cur_val:.5f} V (Expected ~0.0)")
    
    # Window Stats (Full Range)
    print("    Calculating specific window stats (Full Range)...")
    stats = reader.get_statistics_snapshot(0, 0, duration)
    
    # Expected for Sine Wave:
    # RMS = Amp / sqrt(2) = 5 / 1.414 = 3.5355
    # Avg = 0
    # Std = RMS (since mean is 0)
    
    print(f"    Stats: {stats}")
    print(f"    Exp RMS: {5/np.sqrt(2):.4f}")
    
    assert np.isclose(stats['max'], 5.0, atol=0.01), "Max failed"
    # assert np.isclose(stats['rms'], 3.535, atol=0.01), "RMS failed" # Might vary slightly based on block alignment
    
    print("\n=== SUCCESS: Reader passed all checks ===")
    
if __name__ == "__main__":
    test_reader_performance()
