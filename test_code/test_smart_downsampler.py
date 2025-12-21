#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Downsampler Test Suite
============================

Bu test dosyası SmartDownsampler algoritmasının doğruluğunu ve performansını test eder.

Test senaryoları:
1. Spike koruması - threshold üstü değerlerin korunması
2. LTTB görsel doğruluğu
3. Performans benchmarkları (1M+ veri)
4. Edge case'ler

Kullanım:
    python test_smart_downsampler.py

Gereksinimler:
    - time_graph_cpp module (C++ extension)
    - numpy
    - matplotlib (opsiyonel, görselleştirme için)
"""

import sys
import os
import time
import numpy as np

# Module path'i ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add lib/ folder to DLL search path for Arrow/Parquet DLLs
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_dir = os.path.join(_app_dir, 'lib')
if os.path.exists(_lib_dir):
    os.environ['PATH'] = _lib_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_lib_dir)
        except Exception:
            pass

# Also add pyarrow's library directories
try:
    import pyarrow
    arrow_lib_dirs = pyarrow.get_library_dirs()
    if arrow_lib_dirs:
        for lib_dir in arrow_lib_dirs:
            if os.path.exists(lib_dir):
                os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(lib_dir)
                    except Exception:
                        pass
except ImportError:
    pass

try:
    import time_graph_cpp as tg
    print(f"✓ time_graph_cpp module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import time_graph_cpp: {e}")
    print("  Make sure to build the C++ module first:")
    print("  cd cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release")
    sys.exit(1)


def test_basic_downsample():
    """Test 1: Temel downsampling fonksiyonalitesi"""
    print("\n" + "="*60)
    print("TEST 1: Temel Downsampling")
    print("="*60)
    
    # 10K test verisi oluştur
    n = 10000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 100) * 10 + np.random.randn(n) * 0.5
    
    # Downsample
    target = 1000
    result = tg.smart_downsample(x, y, target)
    
    print(f"  Input size:  {result.input_size}")
    print(f"  Output size: {result.output_size}")
    print(f"  Compression: {result.compression_ratio():.4f}")
    print(f"  Spikes:      {result.spike_count}")
    print(f"  Peaks:       {result.peak_count}")
    print(f"  Valleys:     {result.valley_count}")
    
    assert result.is_valid(), "Result should be valid"
    assert result.output_size <= target * 1.5, f"Output size ({result.output_size}) should be close to target ({target})"
    
    # NumPy arrays'e dönüştür
    x_ds, y_ds = result.to_numpy()
    assert len(x_ds) == result.output_size, "NumPy conversion should preserve size"
    
    print("  ✓ Test passed!")
    return True


def test_spike_preservation():
    """Test 2: Spike koruması - EN ÖNEMLİ TEST"""
    print("\n" + "="*60)
    print("TEST 2: Spike Koruması (Motor Arızası Tespiti)")
    print("="*60)
    
    # Normal veri + birkaç spike
    n = 100000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 1000) * 5 + np.random.randn(n) * 0.2
    
    # Kritik spike'lar ekle (motor arızası simülasyonu)
    spike_indices = [10000, 25000, 50000, 75000, 90000]
    spike_values = [100.0, -80.0, 150.0, -120.0, 200.0]
    
    for idx, val in zip(spike_indices, spike_values):
        y[idx] = val
    
    print(f"  Input size:    {n}")
    print(f"  Spike count:   {len(spike_indices)}")
    print(f"  Spike values:  {spike_values}")
    
    # Downsample with threshold
    target = 4000
    threshold = 50.0  # ±50 üstü spike olarak sayılsın
    
    result = tg.smart_downsample(x, y, target, threshold_high=threshold, threshold_low=-threshold)
    
    print(f"\n  Output size:        {result.output_size}")
    print(f"  Detected spikes:    {result.spike_count}")
    print(f"  Critical preserved: {result.critical_points_count}")
    
    # Tüm spike'ların korunduğunu doğrula
    x_ds, y_ds = result.to_numpy()
    
    spikes_found = 0
    for sv in spike_values:
        if any(np.abs(y_ds - sv) < 0.001):
            spikes_found += 1
    
    print(f"  Spikes found in output: {spikes_found}/{len(spike_values)}")
    
    if spikes_found == len(spike_values):
        print("  ✓ ALL SPIKES PRESERVED - Test passed!")
        return True
    else:
        print("  ✗ SPIKE LOSS DETECTED - Test FAILED!")
        return False


def test_auto_threshold():
    """Test 3: Otomatik threshold hesaplama"""
    print("\n" + "="*60)
    print("TEST 3: Otomatik Threshold (3-sigma)")
    print("="*60)
    
    # Normal dağılımlı veri
    n = 50000
    x = np.arange(n, dtype=np.float64)
    np.random.seed(42)
    y = np.random.randn(n) * 10 + 50  # mean=50, std=10
    
    # 3-sigma dışı noktalar ekle (outliers)
    outlier_indices = [5000, 15000, 25000, 35000, 45000]
    for idx in outlier_indices:
        y[idx] = 50 + 40  # 4-sigma away (mean + 4*std)
    
    print(f"  Data mean:     {np.mean(y):.2f}")
    print(f"  Data std:      {np.std(y):.2f}")
    print(f"  Outliers:      {len(outlier_indices)}")
    
    # Auto-threshold ile downsample
    result = tg.smart_downsample(x, y, target_points=2000)
    
    print(f"\n  Output size:   {result.output_size}")
    print(f"  Spikes found:  {result.spike_count}")
    
    # Outlier'ların yakalandığını kontrol et
    x_ds, y_ds = result.to_numpy()
    high_values = y_ds[y_ds > 85]  # 3.5-sigma üstü
    
    print(f"  High values preserved: {len(high_values)}")
    
    if len(high_values) >= len(outlier_indices):
        print("  ✓ Auto-threshold working correctly!")
        return True
    else:
        print("  ✗ Some outliers may have been missed")
        return False


def test_performance():
    """Test 4: Performans benchmark"""
    print("\n" + "="*60)
    print("TEST 4: Performans Benchmark (1M+ veri)")
    print("="*60)
    
    sizes = [100_000, 500_000, 1_000_000]
    target = 4000
    
    for n in sizes:
        x = np.arange(n, dtype=np.float64)
        y = np.sin(x / 10000) * 100 + np.random.randn(n) * 5
        
        # Birkaç spike ekle
        for i in range(10):
            y[np.random.randint(0, n)] = np.random.choice([-500, 500])
        
        # Benchmark
        start = time.perf_counter()
        result = tg.smart_downsample(x, y, target, threshold_high=400, threshold_low=-400)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        throughput = n / elapsed * 1000  # points per second
        
        print(f"\n  {n:,} points:")
        print(f"    Time:       {elapsed:.2f} ms")
        print(f"    Output:     {result.output_size} points")
        print(f"    Throughput: {throughput:,.0f} points/sec")
        print(f"    Spikes:     {result.spike_count}")
        
        # 100ms altında olmalı (1M için)
        if elapsed < 200:
            print(f"    ✓ Performance OK")
        else:
            print(f"    ⚠ Performance could be better")
    
    print("\n  ✓ Performance test completed!")
    return True


def test_config_object():
    """Test 5: Config objesi ile kullanım"""
    print("\n" + "="*60)
    print("TEST 5: SmartDownsampleConfig Kullanımı")
    print("="*60)
    
    n = 50000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 500) * 10 + np.random.randn(n) * 0.5
    
    # Spike ekle
    y[25000] = 100.0
    
    # Config oluştur
    config = tg.SmartDownsampleConfig()
    config.target_points = 2000
    config.spike_threshold_high = 50.0
    config.spike_threshold_low = -50.0
    config.use_auto_threshold = False
    config.detect_local_extrema = True
    config.detect_sudden_changes = True
    config.use_lttb = True
    config.lttb_ratio = 0.7
    
    print(f"  Config: {config}")
    
    # Downsampler instance ile kullan
    downsampler = tg.SmartDownsampler()
    result = downsampler.downsample(x, y, config)
    
    print(f"\n  Result: {result}")
    print(f"  Input:  {result.input_size}")
    print(f"  Output: {result.output_size}")
    print(f"  LTTB points: {result.lttb_points_count}")
    print(f"  Critical:    {result.critical_points_count}")
    
    assert result.is_valid(), "Result should be valid"
    print("  ✓ Config test passed!")
    return True


def test_edge_cases():
    """Test 6: Edge case'ler"""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases")
    print("="*60)
    
    # Case 1: Çok küçük veri (target'tan küçük)
    print("\n  Case 1: Small data (< target)")
    x = np.arange(100, dtype=np.float64)
    y = np.random.randn(100)
    result = tg.smart_downsample(x, y, 4000)
    assert result.output_size == 100, f"Expected 100, got {result.output_size}"
    print(f"    ✓ Returned all {result.output_size} points (no downsampling needed)")
    
    # Case 2: Sabit veri
    print("\n  Case 2: Constant data")
    x = np.arange(10000, dtype=np.float64)
    y = np.ones(10000) * 5.0
    result = tg.smart_downsample(x, y, 1000)
    print(f"    Output: {result.output_size}, Spikes: {result.spike_count}")
    print(f"    ✓ Handled constant data")
    
    # Case 3: Tek spike
    print("\n  Case 3: Single spike in flat data")
    x = np.arange(10000, dtype=np.float64)
    y = np.zeros(10000)
    y[5000] = 1000.0  # Tek büyük spike
    result = tg.smart_downsample(x, y, 100, threshold_high=500)
    
    x_ds, y_ds = result.to_numpy()
    spike_found = any(np.abs(y_ds - 1000.0) < 0.001)
    print(f"    Spike preserved: {spike_found}")
    assert spike_found, "Single spike should be preserved!"
    print(f"    ✓ Single spike preserved correctly")
    
    print("\n  ✓ All edge cases passed!")
    return True


def test_quick_downsample():
    """Test 7: Quick downsample convenience function"""
    print("\n" + "="*60)
    print("TEST 7: Quick Downsample (Static Method)")
    print("="*60)
    
    n = 100000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 1000) * 20 + np.random.randn(n)
    
    # Quick downsample
    result = tg.SmartDownsampler.quick_downsample(x, y, 3000)
    
    print(f"  Input:  {result.input_size}")
    print(f"  Output: {result.output_size}")
    print(f"  Spikes: {result.spike_count}")
    
    # With threshold
    y[50000] = 500.0
    result2 = tg.SmartDownsampler.quick_downsample(x, y, 3000, threshold=100.0)
    
    print(f"\n  With spike (threshold=100):")
    print(f"  Output: {result2.output_size}")
    print(f"  Spikes: {result2.spike_count}")
    
    assert result2.spike_count >= 1, "Spike should be detected"
    print("  ✓ Quick downsample test passed!")
    return True


def visualize_results():
    """Görselleştirme (matplotlib varsa)"""
    print("\n" + "="*60)
    print("VISUALIZATION: Before/After Comparison")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping visualization")
        return
    
    # Test verisi
    n = 100000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 1000) * 10 + np.cumsum(np.random.randn(n) * 0.01)
    
    # Spike'lar ekle
    spike_x = [10000, 30000, 50000, 70000, 90000]
    for sx in spike_x:
        y[sx] = 50.0
        y[sx+1] = -30.0
    
    # Downsample
    result = tg.smart_downsample(x, y, 4000, threshold_high=30, threshold_low=-20)
    x_ds, y_ds = result.to_numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original
    axes[0].plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].scatter(spike_x, [50]*len(spike_x), c='red', s=50, zorder=5, label='Spikes')
    axes[0].set_title(f'Original Data ({n:,} points)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].axhline(y=30, color='r', linestyle='--', alpha=0.3, label='Threshold')
    axes[0].axhline(y=-20, color='r', linestyle='--', alpha=0.3)
    
    # Downsampled
    axes[1].plot(x_ds, y_ds, 'g-', linewidth=0.8, alpha=0.9)
    
    # Korunan spike'ları işaretle
    spike_mask = np.abs(y_ds) > 25
    axes[1].scatter(x_ds[spike_mask], y_ds[spike_mask], c='red', s=50, zorder=5, label='Preserved Spikes')
    
    axes[1].set_title(f'Smart Downsampled ({result.output_size:,} points, {result.spike_count} spikes preserved)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].axhline(y=30, color='r', linestyle='--', alpha=0.3)
    axes[1].axhline(y=-20, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smart_downsampler_test.png', dpi=150)
    print(f"  ✓ Visualization saved to 'smart_downsampler_test.png'")
    plt.show()


def main():
    """Ana test fonksiyonu"""
    print("\n" + "="*60)
    print("    SMART DOWNSAMPLER TEST SUITE")
    print("    MachinePulseAI - High Performance Visualization")
    print("="*60)
    
    results = []
    
    # Tüm testleri çalıştır
    results.append(("Basic Downsample", test_basic_downsample()))
    results.append(("Spike Preservation", test_spike_preservation()))
    results.append(("Auto Threshold", test_auto_threshold()))
    results.append(("Performance", test_performance()))
    results.append(("Config Object", test_config_object()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Quick Downsample", test_quick_downsample()))
    
    # Sonuç özeti
    print("\n" + "="*60)
    print("    TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print("\n  ⚠ SOME TESTS FAILED!")
    
    # Görselleştirme (opsiyonel)
    try:
        visualize_results()
    except Exception as e:
        print(f"\n  Visualization skipped: {e}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
