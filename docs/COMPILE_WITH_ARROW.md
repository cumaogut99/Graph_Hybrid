# Arrow Integration - Compilation Guide

## Prerequisites

### 1. Install PyArrow
```powershell
pip install pyarrow
```

Verify installation:
```powershell
python -c "import pyarrow; print(pyarrow.__version__); print(pyarrow.get_include())"
```

Expected output:
```
15.0.0 (or similar)
C:\Users\...\site-packages\pyarrow\include
```

## Compilation Steps

### Step 1: Clean Build
```powershell
cd cpp
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path build -Force
```

### Step 2: CMake Configuration
```powershell
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -A x64
```

**Expected output:**
```
-- Arrow found: PyArrow runtime
-- Arrow include: C:\Users\...\site-packages\pyarrow\include
...
-- Arrow:             1
```

✅ If you see "Arrow: 1", Arrow integration is enabled!
❌ If you see "Arrow: 0", check PyArrow installation.

### Step 3: Build
```powershell
cmake --build . --config Release -j 8
```

**Build time:** ~2-3 minutes (first build)

### Step 4: Copy Module
```powershell
cd ..
Copy-Item build\Release\time_graph_cpp*.pyd .. -Force
```

### Step 5: Test
```powershell
cd ..
python test_arrow_compilation.py
```

**Expected:**
```
✅ PASS PyArrow Installation
✅ PASS C++ Module Import
✅ PASS Arrow Filter Function
```

## Troubleshooting

### Problem: "Arrow not found"

**Solution 1:** Check PyArrow installation
```powershell
pip show pyarrow
python -c "import pyarrow; print(pyarrow.get_include())"
```

**Solution 2:** Manual CMake configuration
```powershell
$ARROW_INCLUDE = python -c "import pyarrow; print(pyarrow.get_include())"
cmake .. -DCMAKE_BUILD_TYPE=Release -A x64 -DArrow_INCLUDE_DIRS="$ARROW_INCLUDE"
```

### Problem: "Unresolved external symbol" (linking error)

**Solution:** Arrow library not found. Check library path:
```powershell
python -c "import pyarrow; print(pyarrow.get_library_dirs())"
```

### Problem: Module imports but Arrow not available

Check with:
```python
import time_graph_cpp as tgcpp
print(tgcpp.is_arrow_available())  # Should be True
```

If False, recompile with `-DHAVE_ARROW=1`:
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -A x64 -DHAVE_ARROW=1
```

## Verification

### Quick Test
```python
import time_graph_cpp as tgcpp
import numpy as np

# Check Arrow
print("Arrow available:", tgcpp.is_arrow_available())
print("Arrow info:", tgcpp.get_arrow_info())

# Test filter
data = np.array([1.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
cond = tgcpp.FilterCondition()
cond.type = tgcpp.FilterType.RANGE
cond.min_value = 10.0
cond.max_value = 20.0

engine = tgcpp.FilterEngine()
mask = engine.calculate_mask_arrow(data, cond)
print("Mask:", mask)  # [False, False, True, True, True]
```

## Performance Check

Compare Arrow vs Native:
```python
import time_graph_cpp as tgcpp
import numpy as np
import time

# Large dataset
data = np.random.randn(1_000_000)

# Native SIMD
cond = tgcpp.FilterCondition()
cond.type = tgcpp.FilterType.RANGE
cond.min_value = -1.0
cond.max_value = 1.0

engine = tgcpp.FilterEngine()

# Benchmark
t1 = time.perf_counter()
mask_arrow = engine.calculate_mask_arrow(data, cond)
t2 = time.perf_counter()

print(f"Arrow Compute: {(t2-t1)*1000:.1f}ms")
print(f"Points filtered: {sum(mask_arrow)}")
```

Expected: **10-15ms for 1M points** (with Arrow Compute)

## Next Steps

After successful compilation:
1. Run full test suite: `python test_arrow_compilation.py`
2. Benchmark performance: See "Performance Check" above
3. Proceed to Day 2: Statistics Engine integration
