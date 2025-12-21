#!/usr/bin/env python3
"""
Test script for C++ module build
Tests basic import and version
"""

import sys

def test_import():
    """Test if module can be imported"""
    try:
        import time_graph_cpp as tgcpp
        print("✅ Module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_version():
    """Test version function"""
    try:
        import time_graph_cpp as tgcpp
        version = tgcpp.get_version()
        print(f"✅ Version: {version}")
        print(f"✅ Module version attr: {tgcpp.__version__}")
        return True
    except Exception as e:
        print(f"❌ Version test failed: {e}")
        return False

def test_dataframe_class():
    """Test DataFrame class exists"""
    try:
        import time_graph_cpp as tgcpp
        df = tgcpp.DataFrame()
        print(f"✅ DataFrame created: {df}")
        print(f"✅ Row count: {df.row_count()}")
        print(f"✅ Column count: {df.column_count()}")
        return True
    except Exception as e:
        print(f"❌ DataFrame test failed: {e}")
        return False

def test_options_class():
    """Test CsvOptions class"""
    try:
        import time_graph_cpp as tgcpp
        opts = tgcpp.CsvOptions()
        print(f"✅ CsvOptions created: {opts}")
        print(f"   - delimiter: '{opts.delimiter}'")
        print(f"   - has_header: {opts.has_header}")
        print(f"   - encoding: {opts.encoding}")
        
        # Test modification
        opts.delimiter = ';'
        opts.has_header = False
        print(f"✅ CsvOptions modified: {opts}")
        return True
    except Exception as e:
        print(f"❌ CsvOptions test failed: {e}")
        return False

def test_column_type_enum():
    """Test ColumnType enum"""
    try:
        import time_graph_cpp as tgcpp
        print(f"✅ ColumnType.FLOAT64: {tgcpp.ColumnType.FLOAT64}")
        print(f"✅ ColumnType.INT64: {tgcpp.ColumnType.INT64}")
        print(f"✅ ColumnType.STRING: {tgcpp.ColumnType.STRING}")
        return True
    except Exception as e:
        print(f"❌ ColumnType test failed: {e}")
        return False

def test_csv_loading():
    """Test CSV loading functionality"""
    try:
        import time_graph_cpp as tgcpp
        import os
        
        # Check if test file exists
        test_file = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        if not os.path.exists(test_file):
            print(f"⚠️  Test data file not found: {test_file}")
            return True  # Skip test, not a failure
        
        # Load CSV
        opts = tgcpp.CsvOptions()
        opts.delimiter = ','
        opts.has_header = True
        
        df = tgcpp.DataFrame.load_csv(test_file, opts)
        print(f"✅ CSV loaded successfully")
        print(f"   - Rows: {df.row_count()}")
        print(f"   - Columns: {df.column_count()}")
        print(f"   - Column names: {df.column_names()}")
        
        # Test column access
        if 'time' in df.column_names():
            time_col = df.get_column_f64('time')
            print(f"✅ Column access works: time column has {len(time_col)} values")
            print(f"   - First 3 values: {time_col[:3]}")
        
        return True
    except Exception as e:
        print(f"❌ CSV loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Time Graph C++ Module - Build Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Import", test_import),
        ("Version", test_version),
        ("DataFrame", test_dataframe_class),
        ("CsvOptions", test_options_class),
        ("ColumnType", test_column_type_enum),
        ("CSV Loading", test_csv_loading),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 40)
        result = test_func()
        results.append((name, result))
        print()
    
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

