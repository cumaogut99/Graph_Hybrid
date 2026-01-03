# Time Graph C++ Core Library

High-performance C++ core for Time Graph application.

## Requirements

- **CMake** 3.18+
- **C++17** compiler (MSVC 2019+, GCC 9+, Clang 10+)
- **Qt5** 5.15+ (Core, Widgets, PrintSupport)
- **Python** 3.8+ with development headers
- **Pybind11** 2.10+ (bundled if not found)
- **OpenMP** (optional, for parallel processing)

## Building

### Windows (MSVC)

```cmd
# Configure
cmake -B build -G "Visual Studio 16 2019" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DQt5_DIR=C:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5

# Build
cmake --build build --config Release

# Install (copies .pyd to parent directory)
cmake --install build --config Release
```

### Linux (GCC)

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -- -j$(nproc)

# Install
cmake --install build
```

### macOS (Clang)

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -- -j$(sysctl -n hw.ncpu)

# Install
cmake --install build
```

## Build Options

- `CMAKE_BUILD_TYPE`: `Release` (default) or `Debug`
- `BUILD_TESTS`: Build C++ unit tests (default: OFF)
- `Qt5_DIR`: Path to Qt5 CMake config (auto-detected if in PATH)

## Testing

```bash
# Enable tests
cmake -B build -DBUILD_TESTS=ON

# Run tests
cd build
ctest --output-on-failure
```

## Directory Structure

```
cpp/
├── include/timegraph/     # Public headers
│   ├── data/              # DataFrame, CSV loader
│   ├── processing/        # Filters, statistics
│   ├── plot/              # QCustomPlot wrapper
│   └── core/              # Core utilities
├── src/                   # Implementation files
│   ├── data/
│   ├── processing/
│   ├── plot/
│   └── core/
├── bindings/              # Pybind11 Python bindings
├── tests/                 # Unit tests (Google Test)
└── thirdparty/            # Third-party libraries
    ├── pybind11/          # Python bindings
    ├── qcustomplot/       # Plotting library (future)
    └── mio/               # Memory-mapped I/O (future)
```

## Usage (Python)

After building and installing:

```python
import time_graph_cpp as tgcpp

# Load CSV
opts = tgcpp.CsvOptions()
opts.delimiter = ','
opts.has_header = True

df = tgcpp.DataFrame.load_csv('data.csv', opts)
print(f"Loaded {df.row_count()} rows, {df.column_count()} columns")

# Get column (zero-copy)
time_col = df.get_column_f64('time')
print(f"Time column shape: {time_col.shape}")
```

## Performance

Expected performance improvements over pure Python:

- CSV Loading: **10-30x faster** (100MB CSV: ~150ms vs ~3s)
- Filtering: **20-50x faster** (1M points: ~20ms vs ~500ms)
- Statistics: **15-40x faster** (1M points: ~10ms vs ~150ms)

## License

See parent project LICENSE file.

## Development

See `../docs/CPP_DEVELOPMENT_LOG.md` for development notes.

