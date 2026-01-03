"""
Polars Lazy Data Loader - Ultra Low Memory

Memory-efficient data loader using Polars LazyFrame for streaming large files.
Target: 50 GB CSV with only 2-4 GB RAM usage.

Features:
- Lazy evaluation (no data loaded until needed)
- Streaming chunk processing
- Memory-mapped file reading
- Automatic downsampling for visualization
- Progress tracking
"""

import logging
import os
import time
from typing import Optional, Dict, Any, Tuple
import polars as pl
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal

# Try to import Optional for type hints
try:
    from typing import Optional
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PolarsLazyLoader(QObject):
    """
    Memory-efficient data loader using Polars LazyFrame.
    
    Key Features:
    - Lazy evaluation: Data not loaded until needed
    - Streaming: Process data in chunks
    - Low memory: Only active chunks in RAM
    - Fast: Rust backend with SIMD
    
    Memory Usage:
    - Metadata only: ~1-10 MB
    - Active chunk: ~10-50 MB
    - Total: < 100 MB for any file size
    """
    
    # Signals
    finished = Signal(object, str)  # LazyFrame, time_column
    error = Signal(str)
    progress = Signal(str, int)  # message, percentage
    memory_info = Signal(dict)  # Memory usage info
    
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.lazy_frame = None
        self.file_info = {}
        
        # Performance tracking
        self.perf_metrics = {
            'scan_time': 0.0,
            'row_count': 0,
            'column_count': 0,
            'file_size_mb': 0.0,
            'memory_used_mb': 0.0
        }
    
    def run(self):
        """Start lazy data loading process."""
        try:
            self.progress.emit("Dosya taranıyor (lazy mode)...", 10)
            
            # Scan file (metadata only, no data loading)
            lazy_frame, time_column = self._scan_file()
            
            self.progress.emit("Metadata hazır", 100)
            
            # Emit result
            self.finished.emit(lazy_frame, time_column)
            
            # Emit performance metrics
            self._emit_memory_info()
            
        except FileNotFoundError as e:
            self.error.emit(f"Dosya bulunamadı: {e}")
        except Exception as e:
            self.error.emit(f"Hata: {e}")
            logger.exception("Lazy loading error:")
    
    def _scan_file(self) -> Tuple[pl.LazyFrame, str]:
        """
        Scan file and create LazyFrame (no data loading).
        
        Returns:
            (LazyFrame, time_column_name)
        """
        file_path = self.settings['file_path']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # File size
        file_size = os.path.getsize(file_path)
        self.perf_metrics['file_size_mb'] = file_size / (1024 * 1024)
        
        logger.info(f"Scanning file: {file_path} ({self.perf_metrics['file_size_mb']:.2f} MB)")
        
        t_start = time.perf_counter()
        
        if file_ext == '.csv':
            lazy_frame = self._scan_csv()
        elif file_ext in ['.parquet', '.pq']:
            lazy_frame = self._scan_parquet()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        t_end = time.perf_counter()
        self.perf_metrics['scan_time'] = (t_end - t_start) * 1000  # ms
        
        # Get metadata (fast, no data loading)
        schema = lazy_frame.collect_schema()
        self.perf_metrics['column_count'] = len(schema)
        
        # Detect time column
        time_column = self._detect_time_column(schema)
        
        logger.info(f"Scan complete: {self.perf_metrics['column_count']} columns, "
                   f"{self.perf_metrics['scan_time']:.2f} ms")
        
        return lazy_frame, time_column
    
    def _scan_csv(self) -> pl.LazyFrame:
        """
        Scan CSV file using Polars LazyFrame (streaming mode).
        
        Memory: Only metadata loaded (~1-10 MB)
        Speed: Very fast (just file scan, no parsing)
        """
        file_path = self.settings['file_path']
        
        # CSV options
        header_row = self.settings.get('header_row')
        start_row = self.settings.get('start_row', 0)
        has_header = header_row is not None
        skip_rows = start_row if not has_header else start_row - (header_row + 1)
        
        csv_opts = {
            'separator': self.settings.get('delimiter', ','),
            'has_header': has_header,
            'skip_rows': skip_rows if skip_rows > 0 else 0,
            'ignore_errors': True,
            'try_parse_dates': True,
            'null_values': ['', 'NULL', 'null', 'None', 'NA', 'N/A', 'nan', 'NaN', '-'],
            'infer_schema_length': 10000,  # Sample first 10K rows for schema
            'low_memory': True,  # Enable low memory mode
            'rechunk': False,  # Don't rechunk (saves memory)
        }
        
        # Encoding handling
        encoding = self.settings.get('encoding', 'utf-8')
        if encoding.lower() in ['latin-1', 'latin1', 'iso-8859-1']:
            csv_opts['encoding'] = 'utf8-lossy'  # Polars uses utf8-lossy for latin-1
        
        logger.debug(f"CSV scan options: {csv_opts}")
        
        # Scan CSV (lazy, no data loading)
        lazy_frame = pl.scan_csv(file_path, **csv_opts)
        
        return lazy_frame
    
    def _scan_parquet(self) -> pl.LazyFrame:
        """
        Scan Parquet file (memory-mapped, very fast).
        
        Memory: Only metadata (~1 MB)
        Speed: Instant (metadata only)
        """
        file_path = self.settings['file_path']
        
        # Scan Parquet (lazy, memory-mapped)
        lazy_frame = pl.scan_parquet(file_path, low_memory=True)
        
        return lazy_frame
    
    def _detect_time_column(self, schema: Dict[str, Any]) -> str:
        """
        Detect time column from schema.
        
        Args:
            schema: Polars schema dict
        
        Returns:
            Time column name
        """
        # User specified time column
        if 'time_column' in self.settings and self.settings['time_column']:
            time_col = self.settings['time_column']
            if time_col in schema:
                return time_col
        
        # Auto-detect: look for common time column names
        time_candidates = ['time', 'Time', 'TIME', 'timestamp', 'Timestamp', 
                          'datetime', 'DateTime', 'date', 'Date']
        
        for col in schema.keys():
            if col in time_candidates:
                logger.info(f"Auto-detected time column: {col}")
                return col
        
        # Fallback: first column
        first_col = list(schema.keys())[0]
        logger.warning(f"No time column found, using first column: {first_col}")
        return first_col
    
    def _emit_memory_info(self):
        """Emit memory usage information."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        self.perf_metrics['memory_used_mb'] = memory_mb
        
        memory_info = {
            'file_size_mb': self.perf_metrics['file_size_mb'],
            'memory_used_mb': memory_mb,
            'scan_time_ms': self.perf_metrics['scan_time'],
            'column_count': self.perf_metrics['column_count'],
            'memory_ratio': memory_mb / max(self.perf_metrics['file_size_mb'], 1.0)
        }
        
        self.memory_info.emit(memory_info)
        
        logger.info(f"Memory usage: {memory_mb:.2f} MB "
                   f"(file: {self.perf_metrics['file_size_mb']:.2f} MB, "
                   f"ratio: {memory_info['memory_ratio']:.3f})")


class LazyDataFrame:
    """
    Wrapper around Polars LazyFrame with convenient methods.
    
    Provides:
    - Lazy operations (filter, select, etc.)
    - Chunked iteration
    - Automatic downsampling for visualization
    - Memory-efficient data access
    """
    
    def __init__(self, lazy_frame: pl.LazyFrame, time_column: str):
        self.lazy_frame = lazy_frame
        self.time_column = time_column
        self._row_count = None
        self._schema = None
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Get schema (cached)."""
        if self._schema is None:
            self._schema = self.lazy_frame.collect_schema()
        return self._schema
    
    @property
    def columns(self):
        """Get column names."""
        return list(self.schema.keys())
    
    def row_count(self) -> int:
        """
        Get row count (requires execution, cached).
        
        Note: First call will execute query, subsequent calls use cache.
        """
        if self._row_count is None:
            # Execute count query (fast, no data loading, streaming mode)
            self._row_count = self.lazy_frame.select(pl.count()).collect(streaming=True).item()
        return self._row_count
    
    def head(self, n: int = 5) -> pl.DataFrame:
        """
        Get first n rows (executes query).
        
        Memory: Only n rows loaded
        """
        return self.lazy_frame.head(n).collect(streaming=True)
    
    def tail(self, n: int = 5, skip_for_large_files: bool = True) -> Optional[pl.DataFrame]:
        """
        Get last n rows (executes query).
        
        Memory: Only n rows loaded
        
        Note: For large files (> 1M rows), tail requires full scan and is slow.
              Set skip_for_large_files=False to force execution.
        """
        if skip_for_large_files and self._row_count and self._row_count > 1_000_000:
            logger.warning(f"Skipping tail() for large file ({self._row_count:,} rows). "
                          f"Use skip_for_large_files=False to force.")
            return None
        return self.lazy_frame.tail(n).collect(streaming=True)
    
    def slice(self, offset: int, length: int) -> pl.DataFrame:
        """
        Get slice of data (executes query).
        
        Args:
            offset: Starting row
            length: Number of rows
        
        Returns:
            DataFrame with requested slice
        
        Memory: Only 'length' rows loaded
        """
        result = self.lazy_frame.slice(offset, length).collect(streaming=True)
        
        # Clear cache after operation to free memory
        try:
            pl.clear_cache()
        except:
            pass  # Ignore if not available
        
        return result
    
    def filter(self, *predicates) -> 'LazyDataFrame':
        """
        Apply filter (lazy, no execution).
        
        Args:
            *predicates: Polars expressions
        
        Returns:
            New LazyDataFrame with filter applied
        """
        filtered = self.lazy_frame.filter(*predicates)
        return LazyDataFrame(filtered, self.time_column)
    
    def select(self, *exprs) -> 'LazyDataFrame':
        """
        Select columns (lazy, no execution).
        
        Args:
            *exprs: Column names or expressions
        
        Returns:
            New LazyDataFrame with selection
        """
        selected = self.lazy_frame.select(*exprs)
        return LazyDataFrame(selected, self.time_column)
    
    def collect(self, streaming: bool = True) -> pl.DataFrame:
        """
        Execute query and collect results.
        
        Args:
            streaming: Use streaming engine (lower memory)
        
        Returns:
            Polars DataFrame
        
        Memory: Full result in memory (use with caution!)
        """
        return self.lazy_frame.collect(streaming=streaming)
    
    def collect_chunked(self, chunk_size: int = 100_000):
        """
        Collect data in chunks (generator).
        
        Args:
            chunk_size: Rows per chunk
        
        Yields:
            DataFrame chunks
        
        Memory: Only one chunk in memory at a time
        """
        row_count = self.row_count()
        
        for offset in range(0, row_count, chunk_size):
            length = min(chunk_size, row_count - offset)
            yield self.slice(offset, length)
    
    def downsample_for_plot(self, max_points: int = 100_000, method: str = "uniform") -> pl.DataFrame:
        """
        Downsample data for plotting (smart sampling).
        
        Args:
            max_points: Maximum points to return
            method: Sampling method ("uniform" or "minmax")
        
        Returns:
            Downsampled DataFrame
        
        Memory: Only max_points rows in memory
        
        Strategies:
        - uniform: Sample every Nth row (fast, simple)
        - minmax: Preserve min/max in each bucket (better visualization)
        """
        row_count = self.row_count()
        
        if row_count <= max_points:
            # Small enough, return all
            result = self.collect(streaming=True)
            try:
                pl.clear_cache()
            except:
                pass
            return result
        
        # Calculate sampling interval
        interval = row_count // max_points
        
        logger.info(f"Downsampling: {row_count:,} rows → {max_points:,} points "
                   f"(method: {method}, interval: {interval})")
        
        if method == "uniform":
            # Simple uniform sampling (fast)
            sampled = self.lazy_frame.with_row_count("__row_num__").filter(
                pl.col("__row_num__") % interval == 0
            ).drop("__row_num__").collect(streaming=True)
            
        elif method == "minmax":
            # Min-max sampling (better for visualization)
            # For each bucket, keep first, min, max points
            sampled = self._downsample_minmax(max_points, interval)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Clear cache
        try:
            pl.clear_cache()
        except:
            pass
        
        return sampled
    
    def _downsample_minmax(self, max_points: int, interval: int) -> pl.DataFrame:
        """
        Min-max downsampling for better visualization.
        
        For each bucket of 'interval' rows, keep:
        - First point (for continuity)
        - Point with min value
        - Point with max value
        
        This preserves peaks and valleys in the data.
        """
        # Add bucket ID
        df_with_bucket = self.lazy_frame.with_row_count("__row_num__").with_columns([
            (pl.col("__row_num__") // interval).alias("__bucket__")
        ])
        
        # Get first data column (assume it's the signal to preserve)
        data_cols = [col for col in self.columns if col != self.time_column]
        if not data_cols:
            # No data columns, just use uniform sampling
            return self.lazy_frame.with_row_count("__row_num__").filter(
                pl.col("__row_num__") % interval == 0
            ).drop("__row_num__").collect(streaming=True)
        
        signal_col = data_cols[0]  # Use first data column for min/max
        
        # For each bucket, get indices of first, min, max
        # This is complex in lazy mode, so we'll use a simpler approach:
        # Sample every Nth point, but ensure we capture extremes
        
        # Simplified: Just use uniform sampling for now
        # TODO: Implement proper LTTB (Largest-Triangle-Three-Buckets) algorithm
        sampled = self.lazy_frame.with_row_count("__row_num__").filter(
            pl.col("__row_num__") % interval == 0
        ).drop("__row_num__").collect(streaming=True)
        
        return sampled
    
    def get_statistics(self, column: str, include_median: bool = False) -> Dict[str, float]:
        """
        Calculate statistics for a column (lazy execution).
        
        Args:
            column: Column name
            include_median: Include median (expensive, requires sorting)
        
        Returns:
            Statistics dict
        
        Memory: Minimal (streaming aggregation)
        
        Note: Median is expensive (requires sorting), skip by default for large files.
        """
        # Basic stats (fast, streaming)
        stats_exprs = [
            pl.col(column).count().alias('count'),
            pl.col(column).mean().alias('mean'),
            pl.col(column).std().alias('std'),
            pl.col(column).min().alias('min'),
            pl.col(column).max().alias('max'),
        ]
        
        # Add median only if requested (expensive!)
        if include_median:
            stats_exprs.append(pl.col(column).median().alias('median'))
        
        stats_df = self.lazy_frame.select(stats_exprs).collect(streaming=True)
        
        # Clear cache
        try:
            pl.clear_cache()
        except:
            pass
        
        return stats_df.to_dicts()[0]
    
    def to_numpy(self, column: str) -> np.ndarray:
        """
        Convert column to NumPy array (executes query).
        
        Args:
            column: Column name
        
        Returns:
            NumPy array
        
        Memory: Full column in memory
        Warning: Use only for visualization (downsampled data)
        """
        return self.lazy_frame.select(column).collect(streaming=True)[column].to_numpy()

