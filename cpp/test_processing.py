#!/usr/bin/env python3
"""
Test script for Filter and Statistics engines
"""

import sys
import time
import numpy as np

print("=" * 60)
print("Time Graph C++ - Processing Engines Test")
print("=" * 60)
print()

# ==============================================================================
# Test 1: Import
# ==============================================================================
print("[TEST] Import")
print("-" * 40)
try:
    import time_graph_cpp as tgcpp
    print("✅ Module imported successfully")
    print(f"✅ Version: {tgcpp.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()

# ==============================================================================
# Test 2: Load Test Data
# ==============================================================================
print("[TEST] Load Test Data")
print("-" * 40)
try:
    csv_opts = tgcpp.CsvOptions()
    df = tgcpp.DataFrame.load_csv("test_data.csv", csv_opts)
    print(f"✅ CSV loaded: {df.row_count()} rows, {df.column_count()} columns")
    print(f"   Columns: {df.column_names()}")
except Exception as e:
    print(f"❌ Load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==============================================================================
# Test 3: Statistics Engine
# ==============================================================================
print("[TEST] Statistics Engine")
print("-" * 40)
try:
    stats_engine = tgcpp.StatisticsEngine()
    
    # Calculate basic statistics
    stats = tgcpp.StatisticsEngine.calculate(df, "rpm")
    print(f"✅ Basic statistics calculated:")
    print(f"   - Mean: {stats.mean:.2f}")
    print(f"   - Std Dev: {stats.std_dev:.2f}")
    print(f"   - Min: {stats.min:.2f}")
    print(f"   - Max: {stats.max:.2f}")
    print(f"   - Median: {stats.median:.2f}")
    print(f"   - Count: {stats.count} / {stats.valid_count} valid")
    
    # Threshold analysis
    threshold_stats = tgcpp.StatisticsEngine.calculate_with_threshold(
        df, "rpm", "time", 2000.0
    )
    print(f"✅ Threshold analysis (threshold=2000):")
    print(f"   - Above: {threshold_stats.above_count} ({threshold_stats.above_percentage:.1f}%)")
    print(f"   - Below: {threshold_stats.below_count} ({threshold_stats.below_percentage:.1f}%)")
    
    # Percentile
    p50 = tgcpp.StatisticsEngine.percentile(df, "rpm", 50.0)
    p95 = tgcpp.StatisticsEngine.percentile(df, "rpm", 95.0)
    print(f"✅ Percentiles: P50={p50:.2f}, P95={p95:.2f}")
    
except Exception as e:
    print(f"❌ Statistics test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ==============================================================================
# Test 4: Filter Engine
# ==============================================================================
print("[TEST] Filter Engine")
print("-" * 40)
try:
    filter_engine = tgcpp.FilterEngine()
    
    # Create filter condition: rpm > 1500
    cond1 = tgcpp.FilterCondition()
    cond1.column_name = "rpm"
    cond1.type = tgcpp.FilterType.GREATER
    cond1.threshold = 1500.0
    cond1.op = tgcpp.FilterOperator.AND
    
    # Create filter condition: temperature < 90
    cond2 = tgcpp.FilterCondition()
    cond2.column_name = "temperature"
    cond2.type = tgcpp.FilterType.LESS
    cond2.threshold = 90.0
    cond2.op = tgcpp.FilterOperator.AND
    
    conditions = [cond1, cond2]
    
    # Calculate boolean mask
    mask = filter_engine.calculate_mask(df, conditions)
    num_passed = sum(mask)
    print(f"✅ Filter mask calculated: {num_passed}/{len(mask)} points pass filter")
    
    # Calculate time segments
    segments = filter_engine.calculate_segments(df, "time", conditions)
    print(f"✅ Time segments calculated: {len(segments)} segments found")
    for i, seg in enumerate(segments[:3]):  # Show first 3
        print(f"   - Segment {i+1}: [{seg.start_time:.2f}, {seg.end_time:.2f}] "
              f"({seg.end_index - seg.start_index + 1} points)")
    
    # Test single condition check
    test_value = 2000.0
    passes = tgcpp.FilterEngine.check_condition(test_value, cond1)
    print(f"✅ Single check: {test_value} > 1500 = {passes}")
    
except Exception as e:
    print(f"❌ Filter test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ==============================================================================
# Test 5: Performance Benchmark
# ==============================================================================
print("[TEST] Performance Benchmark")
print("-" * 40)

# Create larger test dataset
print("Creating benchmark data (100K points)...")
n_points = 100000
benchmark_data = {
    'time': np.linspace(0, 100, n_points),
    'signal1': np.random.randn(n_points) * 100 + 1000,
    'signal2': np.random.randn(n_points) * 50 + 500,
    'signal3': np.random.randn(n_points) * 20 + 100,
}

# Save to CSV
import csv
with open('benchmark_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'signal1', 'signal2', 'signal3'])
    for i in range(n_points):
        writer.writerow([
            benchmark_data['time'][i],
            benchmark_data['signal1'][i],
            benchmark_data['signal2'][i],
            benchmark_data['signal3'][i]
        ])

# Load with C++
print("Loading benchmark data...")
t0 = time.time()
bench_df = tgcpp.DataFrame.load_csv("benchmark_data.csv", csv_opts)
t_load = (time.time() - t0) * 1000
print(f"✅ Load time: {t_load:.2f} ms")

# Statistics benchmark
print("Running statistics benchmark...")
t0 = time.time()
for _ in range(10):
    stats = tgcpp.StatisticsEngine.calculate(bench_df, "signal1")
t_stats = (time.time() - t0) * 1000 / 10
print(f"✅ Stats time (avg of 10): {t_stats:.2f} ms")

# Filter benchmark
print("Running filter benchmark...")
cond = tgcpp.FilterCondition()
cond.column_name = "signal1"
cond.type = tgcpp.FilterType.RANGE
cond.min_value = 900.0
cond.max_value = 1100.0

t0 = time.time()
for _ in range(10):
    mask = filter_engine.calculate_mask(bench_df, [cond])
t_filter = (time.time() - t0) * 1000 / 10
print(f"✅ Filter time (avg of 10): {t_filter:.2f} ms")

# Segment calculation benchmark
print("Running segment calculation benchmark...")
t0 = time.time()
for _ in range(10):
    segments = filter_engine.calculate_segments(bench_df, "time", [cond])
t_segments = (time.time() - t0) * 1000 / 10
print(f"✅ Segments time (avg of 10): {t_segments:.2f} ms")
print(f"   - Found {len(segments)} segments")

print()

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 60)
print("Performance Summary (100K points)")
print("=" * 60)
print(f"CSV Load:       {t_load:8.2f} ms")
print(f"Statistics:     {t_stats:8.2f} ms")
print(f"Filter Mask:    {t_filter:8.2f} ms")
print(f"Segments:       {t_segments:8.2f} ms")
print()
print("🎉 All tests completed!")

