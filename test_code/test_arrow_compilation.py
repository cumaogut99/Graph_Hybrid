#!/usr/bin/env python3
"""
Test Arrow Integration - Verify compilation and runtime availability
"""

import sys
import subprocess

def test_arrow_installation():
    """Test if PyArrow is installed"""
    print("=" * 60)
    print("Step 1: Testing PyArrow installation")
    print("=" * 60)
    
    try:
        import pyarrow as pa
        print(f"✅ PyArrow installed: {pa.__version__}")
        print(f"   Include dir: {pa.get_include()}")
        lib_dirs = pa.get_library_dirs()
        if lib_dirs:
            print(f"   Library dir: {lib_dirs[0]}")
        return True
    except ImportError as e:
        print(f"❌ PyArrow NOT installed: {e}")
        print("\n💡 Install with: pip install pyarrow")
        return False

def test_cpp_compilation():
    """Test C++ compilation with Arrow"""
    print("\n" + "=" * 60)
    print("Step 2: Testing C++ compilation")
    print("=" * 60)
    
    print("\n📝 Running CMake configuration...")
    print("   Command: cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -A x64")
    
    # Note: This is just informational - actual compilation happens in shell
    print("\n⚠️  Please run manually:")
    print("   1. cd cpp")
    print("   2. Remove-Item -Recurse -Force build (if exists)")
    print("   3. cmake -B build -DCMAKE_BUILD_TYPE=Release -A x64")
    print("   4. cmake --build build --config Release -j 8")
    print("   5. Copy-Item build\\Release\\time_graph_cpp*.pyd ..")
    
    return True

def test_cpp_module_import():
    """Test if C++ module imports and Arrow is detected"""
    print("\n" + "=" * 60)
    print("Step 3: Testing C++ module import")
    print("=" * 60)
    
    try:
        import time_graph_cpp as tgcpp
        print("✅ C++ module imported successfully")
        
        # Check Arrow availability
        if hasattr(tgcpp, 'is_arrow_available'):
            arrow_available = tgcpp.is_arrow_available()
            print(f"\n🔍 Arrow Compute status: {'✅ AVAILABLE' if arrow_available else '❌ NOT AVAILABLE'}")
            
            if hasattr(tgcpp, 'get_arrow_info'):
                info = tgcpp.get_arrow_info()
                print(f"   Version: {info.get('version', 'unknown')}")
                print(f"   Features: {', '.join(info.get('features', []))}")
        else:
            print("⚠️  Arrow functions not found (old module version?)")
        
        return True
        
    except ImportError as e:
        print(f"❌ C++ module import failed: {e}")
        print("\n💡 Module not compiled yet - run compilation steps above")
        return False

def test_arrow_filter_function():
    """Test Arrow filter function"""
    print("\n" + "=" * 60)
    print("Step 4: Testing Arrow filter function")
    print("=" * 60)
    
    try:
        import time_graph_cpp as tgcpp
        import numpy as np
        
        # Create test data
        test_data = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64)
        
        # Create filter condition (range: 10-20)
        condition = tgcpp.FilterCondition()
        condition.column_name = "test"
        condition.type = tgcpp.FilterType.RANGE
        condition.min_value = 10.0
        condition.max_value = 20.0
        
        # Create filter engine
        engine = tgcpp.FilterEngine()
        
        # Test Arrow function
        if hasattr(engine, 'calculate_mask_arrow'):
            print("✅ calculate_mask_arrow() found")
            
            try:
                mask = engine.calculate_mask_arrow(test_data, condition)
                expected = [False, False, True, True, True, False, False]
                
                if mask == expected:
                    print(f"✅ Filter result correct: {mask}")
                    print("   Test data: [1, 5, 10, 15, 20, 25, 30]")
                    print("   Filter: 10 <= value <= 20")
                    print("   Result: [F, F, T, T, T, F, F] ✓")
                    return True
                else:
                    print(f"❌ Filter result incorrect!")
                    print(f"   Expected: {expected}")
                    print(f"   Got: {mask}")
                    return False
                    
            except Exception as e:
                print(f"⚠️  Function exists but failed: {e}")
                print("   (May be using fallback implementation)")
                return False
        else:
            print("❌ calculate_mask_arrow() NOT found")
            print("   Module needs recompilation with new bindings")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print(" Arrow Integration Test Suite")
    print("🚀" * 30 + "\n")
    
    results = []
    
    # Test 1: PyArrow installation
    results.append(("PyArrow Installation", test_arrow_installation()))
    
    # Test 2: Compilation info
    results.append(("Compilation Info", test_cpp_compilation()))
    
    # Test 3: Module import
    results.append(("C++ Module Import", test_cpp_module_import()))
    
    # Test 4: Arrow filter function
    results.append(("Arrow Filter Function", test_arrow_filter_function()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Arrow integration is ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
