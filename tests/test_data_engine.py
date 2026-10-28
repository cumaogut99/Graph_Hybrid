
import numpy as np
import os
import sys
import struct
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_engine import DataEngine

def test_data_engine():
    print("=== Testing MPAI Data Engine ===")
    
    # 1. Setup Test Data
    print("[1] Generating Synthetic Data...")
    fs = 50000.0  # 50 kHz
    duration = 2.0 # 2 seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create two channels:
    # Ch1: Sine wave 
    # Ch2: Random Noise
    data_ch1 = 5.0 * np.sin(2 * np.pi * 100 * t) # 5V amp, 100Hz
    data_ch2 = np.random.normal(0, 1, len(t))
    
    data = np.column_stack((data_ch1, data_ch2))
    print(f"    Data Shape: {data.shape}, Type: {data.dtype}")
    
    # 2. Initialize Engine and Import
    output_dir = Path("test_output_mpai")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    engine = DataEngine(str(output_dir))
    
    print("[2] Running Import...")
    mpai_path = engine.import_numpy_data(
        data, 
        fs, 
        "TestRecord", 
        channel_names=["SineWave", "Noise"],
        units=["V", "Pa"]
    )
    
    print(f"    Created Record: {mpai_path}")
    
    # 3. Validation
    record_path = Path(mpai_path)
    
    # A. Check Structure
    print("[3] Validating Directory Structure...")
    required_files = ["setup.xml", "channel_0.bin", "channel_0.red", "channel_1.bin", "channel_1.red"]
    for f in required_files:
        p = record_path / f
        if not p.exists():
            print(f"    [FAIL] Missing file: {f}")
            return
        print(f"    [OK] Found {f} ({p.stat().st_size} bytes)")
        
    # B. Validate Raw Data Integrity
    print("[4] Validating Raw Binary Data...")
    
    # Read back Ch0
    raw_0_path = record_path / "channel_0.bin"
    # Using np.fromfile
    loaded_raw_0 = np.fromfile(raw_0_path, dtype=np.float64)
    
    if np.array_equal(loaded_raw_0, data_ch1):
         print("    [OK] Channel 0 Raw Data matches 100%")
    else:
         print("    [FAIL] Channel 0 Raw Data mismatch!")
         diff = np.abs(loaded_raw_0 - data_ch1)
         print(f"    Max Diff: {np.max(diff)}")
         
    # C. Validate Reduced Data Logic
    print("[5] Validating Reduced Data Streams...")
    # Block size is 500
    BLOCK_SIZE = 500
    red_0_path = record_path / "channel_0.red"
    
    # Read red file: It's a sequence of (min, max, avg, rms) doubles
    # Read as flat array first
    red_flat = np.fromfile(red_0_path, dtype=np.float64)
    if len(red_flat) % 4 != 0:
        print("    [WARNING] Reduced file size not multiple of 4 floats!")
    
    # Reshape to (N_Blocks, 4)
    red_structs = red_flat.reshape(-1, 4)
    print(f"    Loaded {len(red_structs)} Reduced Blocks")
    
    # Validate First Block Manually
    first_chunk = data_ch1[:BLOCK_SIZE]
    
    exp_min = np.min(first_chunk)
    exp_max = np.max(first_chunk)
    exp_avg = np.mean(first_chunk)
    exp_rms = np.sqrt(np.mean(first_chunk**2))
    
    act_min, act_max, act_avg, act_rms = red_structs[0]
    
    print(f"    Block 0 Verification:")
    print(f"    Metric | Expected        | Actual          | Diff")
    print(f"    -------|-----------------|-----------------|-------")
    print(f"    Min    | {exp_min: .6f} | {act_min: .6f} | {abs(exp_min - act_min):.2e}")
    print(f"    Max    | {exp_max: .6f} | {act_max: .6f} | {abs(exp_max - act_max):.2e}")
    print(f"    Avg    | {exp_avg: .6f} | {act_avg: .6f} | {abs(exp_avg - act_avg):.2e}")
    print(f"    RMS    | {exp_rms: .6f} | {act_rms: .6f} | {abs(exp_rms - act_rms):.2e}")
    
    # Assert
    assert np.isclose(act_min, exp_min), "Min Mismatch"
    assert np.isclose(act_max, exp_max), "Max Mismatch"
    assert np.isclose(act_avg, exp_avg), "Avg Mismatch"
    assert np.isclose(act_rms, exp_rms), "RMS Mismatch"
    
    print("    [OK] Reduced Data Logic Verified Correct.")
    print("\n=== Test Complete: SUCCESSS ===")

if __name__ == "__main__":
    test_data_engine()
