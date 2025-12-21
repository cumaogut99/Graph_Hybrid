#!/usr/bin/env python3
"""
Test Critical Points Detection & Smart Downsampling
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load PyArrow first
import pyarrow as pa

def test_critical_points():
    """Test critical points detection"""
    print("\n" + "=" * 70)
    print("TEST: Critical Points Detection")
    print("=" * 70)
    
    try:
        import time_graph_cpp as tgcpp
    except ImportError:
        print("❌ C++ module not found - compile first!")
        return False
    
    # Generate test data with clear peaks and valleys
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)
    
    # Add some sudden changes
    signal[3000] += 2.0
    signal[7000] -= 1.5
    
    print(f"\n📊 Test data: {len(t):,} points")
    print(f"   Time range: {t[0]:.2f} to {t[-1]:.2f}")
    print(f"   Signal range: {signal.min():.2f} to {signal.max():.2f}")
    
    # Configure detection
    config = tgcpp.CriticalPointsConfig()
    config.detect_peaks = True
    config.detect_valleys = True
    config.detect_sudden_changes = True
    config.detect_limit_violations = False
    config.window_size = 20
    config.max_points = 100
    
    print("\n⚙️  Config:")
    print(f"   Peaks: {config.detect_peaks}")
    print(f"   Valleys: {config.detect_valleys}")
    print(f"   Sudden changes: {config.detect_sudden_changes}")
    print(f"   Window size: {config.window_size}")
    
    # Detect critical points
    t_start = time.perf_counter()
    critical_points = tgcpp.CriticalPointsDetector.detect(t, signal, config)
    t_end = time.perf_counter()
    
    detection_time = (t_end - t_start) * 1000
    
    print(f"\n✅ Detection complete: {detection_time:.2f}ms")
    print(f"   Found {len(critical_points)} critical points")
    
    # Count by type
    peaks = sum(1 for cp in critical_points if cp.type == tgcpp.CriticalPointType.LOCAL_MAX)
    valleys = sum(1 for cp in critical_points if cp.type == tgcpp.CriticalPointType.LOCAL_MIN)
    changes = sum(1 for cp in critical_points if cp.type == tgcpp.CriticalPointType.SUDDEN_CHANGE)
    
    print(f"\n📈 Breakdown:")
    print(f"   Peaks (LOCAL_MAX): {peaks}")
    print(f"   Valleys (LOCAL_MIN): {valleys}")
    print(f"   Sudden changes: {changes}")
    
    # Show first few
    print(f"\n🔍 First 5 critical points:")
    for i, cp in enumerate(critical_points[:5]):
        type_str = {
            tgcpp.CriticalPointType.LOCAL_MAX: "PEAK",
            tgcpp.CriticalPointType.LOCAL_MIN: "VALLEY",
            tgcpp.CriticalPointType.SUDDEN_CHANGE: "CHANGE",
        }.get(cp.type, "UNKNOWN")
        print(f"   {i+1}. {type_str:8s} @ t={cp.time:6.3f}, val={cp.value:6.3f}, sig={cp.significance:.2f}")
    
    return True


def test_lttb_downsampling():
    """Test LTTB downsampling"""
    print("\n" + "=" * 70)
    print("TEST: LTTB Downsampling")
    print("=" * 70)
    
    try:
        import time_graph_cpp as tgcpp
    except ImportError:
        print("❌ C++ module not found")
        return False
    
    # Generate large dataset
    size = 1_000_000
    t = np.linspace(0, 100, size)
    signal = np.sin(2 * np.pi * t / 10) + 0.3 * np.random.randn(size)
    
    print(f"\n📊 Original data: {size:,} points")
    
    # Target for downsampling
    max_points = 4000
    
    print(f"🎯 Target: {max_points:,} points")
    
    # Run LTTB
    t_start = time.perf_counter()
    result = tgcpp.downsample_lttb(t, signal, max_points)
    t_end = time.perf_counter()
    
    downsample_time = (t_end - t_start) * 1000
    
    print(f"\n✅ Downsampling complete: {downsample_time:.2f}ms")
    print(f"   Result: {len(result.time):,} points")
    print(f"   Reduction: {size / len(result.time):.1f}x")
    print(f"   Speed: {size / downsample_time / 1000:.1f}K points/ms")
    
    # Verify time integrity
    time_diff_downsampled = np.diff(result.time)
    
    print(f"\n⏱️  Time Integrity Check:")
    print(f"   Original time range: {t[0]:.2f} to {t[-1]:.2f}")
    print(f"   Downsampled time range: {result.time[0]:.2f} to {result.time[-1]:.2f}")
    print(f"   Time boundaries match: {'✅' if abs(t[0] - result.time[0]) < 1e-10 and abs(t[-1] - result.time[-1]) < 1e-10 else '❌'}")
    print(f"   Monotonic: {'✅' if np.all(time_diff_downsampled > 0) else '❌'}")
    
    # Verify indices
    print(f"\n🔢 Index Verification:")
    print(f"   Indices available: {len(result.indices) > 0}")
    if len(result.indices) > 0:
        print(f"   First index: {result.indices[0]}")
        print(f"   Last index: {result.indices[-1]}")
        print(f"   Indices match time: {'✅' if result.time[0] == t[result.indices[0]] else '❌'}")
    
    return True


def test_smart_downsampling():
    """Test smart downsampling with critical points"""
    print("\n" + "=" * 70)
    print("TEST: Smart Downsampling (LTTB + Critical)")
    print("=" * 70)
    
    try:
        import time_graph_cpp as tgcpp
    except ImportError:
        print("❌ C++ module not found")
        return False
    
    # Generate data with peaks
    size = 500_000
    t = np.linspace(0, 100, size)
    signal = 3.0 * np.sin(2 * np.pi * t / 20)
    
    # Add some peaks
    peak_indices = [50000, 150000, 250000, 350000, 450000]
    for idx in peak_indices:
        if idx + 100 < size:
            signal[idx:idx+100] += 5.0  # Sharp peak
    
    print(f"\n📊 Test data: {size:,} points")
    print(f"   Artificial peaks: {len(peak_indices)}")
    print(f"   Signal range: {signal.min():.2f} to {signal.max():.2f}")
    
    # Configure
    config = tgcpp.CriticalPointsConfig()
    config.detect_peaks = True
    config.detect_valleys = True
    config.detect_sudden_changes = True
    config.detect_limit_violations = True
    config.warning_limits = [-6.0, 6.0]
    config.max_points = 500
    
    max_points = 4000
    
    print(f"\n⚙️  Config:")
    print(f"   Target points: {max_points:,}")
    print(f"   Max critical points: {config.max_points}")
    print(f"   Warning limits: {config.warning_limits}")
    
    # Run smart downsampling
    t_start = time.perf_counter()
    result = tgcpp.downsample_lttb_with_critical(t, signal, max_points, config)
    t_end = time.perf_counter()
    
    downsample_time = (t_end - t_start) * 1000
    
    print(f"\n✅ Smart downsampling complete: {downsample_time:.2f}ms")
    print(f"   Original: {size:,} points")
    print(f"   Final: {len(result.time):,} points")
    print(f"   Critical points preserved: {result.critical_count}")
    print(f"   Reduction: {size / len(result.time):.1f}x")
    
    # Check if peaks are preserved
    result_time_array = np.array(result.time)
    
    peaks_found = 0
    for peak_idx in peak_indices:
        peak_time = t[peak_idx]
        # Find closest point in result
        closest_idx = np.argmin(np.abs(result_time_array - peak_time))
        if np.abs(result_time_array[closest_idx] - peak_time) < 1.0:  # Within 1 second
            peaks_found += 1
    
    print(f"\n🔍 Peak Preservation:")
    print(f"   Artificial peaks: {len(peak_indices)}")
    print(f"   Peaks found in result: {peaks_found}")
    print(f"   Preservation rate: {peaks_found / len(peak_indices) * 100:.0f}%")
    
    return peaks_found >= len(peak_indices) * 0.8  # At least 80% preserved


def test_python_module():
    """Test Python downsampling module"""
    print("\n" + "=" * 70)
    print("TEST: Python Downsampling Module")
    print("=" * 70)
    
    try:
        from src.graphics.smart_downsampling import downsample_for_plot
    except ImportError as e:
        print(f"❌ Python module import failed: {e}")
        return False
    
    # Test data
    size = 600_000
    t = np.linspace(0, 50, size)
    signal = 3.0 * np.sin(2 * np.pi * t / 10)
    
    print(f"\n📊 Test data: {size:,} points")
    
    # Test auto mode
    print("\n🔹 Testing downsample_for_plot()...")
    t_start = time.perf_counter()
    time_ds, signal_ds, info = downsample_for_plot(
        t, signal,
        has_limits=True,
        limits={'min': -4.0, 'max': 4.0},
        screen_width=1920
    )
    t_end = time.perf_counter()
    
    print(f"\n✅ Success!")
    print(f"   Time: {(t_end - t_start) * 1000:.2f}ms")
    print(f"   Original: {info['original_points']:,}")
    print(f"   Final: {info['final_points']:,}")
    print(f"   Strategy: {info['strategy']}")
    print(f"   Downsampled: {info['downsampled']}")
    if 'critical_points' in info:
        print(f"   Critical points: {info['critical_points']}")
    
    # Verify time integrity
    print(f"\n⏱️  Time Integrity:")
    print(f"   Original range: {t[0]:.2f} to {t[-1]:.2f}")
    print(f"   Downsampled range: {time_ds[0]:.2f} to {time_ds[-1]:.2f}")
    print(f"   Match: {'✅' if abs(t[0] - time_ds[0]) < 1e-10 and abs(t[-1] - time_ds[-1]) < 1e-10 else '❌'}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print(" Critical Points & Downsampling Tests")
    print("🧪" * 35)
    
    tests = [
        ("Critical Points Detection", test_critical_points),
        ("LTTB Downsampling", test_lttb_downsampling),
        ("Smart Downsampling", test_smart_downsampling),
        ("Python Module", test_python_module),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
