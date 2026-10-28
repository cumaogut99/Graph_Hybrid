import logging
import os
import time
import psutil
import shutil
import io
import zipfile
from typing import Optional, Dict, Any, Callable, List
import polars as pl
from PyQt5.QtCore import QObject, pyqtSignal as Signal


# Import MpaiProjectManager for ZIP64 container support
try:
    from src.data.mpai_project_manager import MpaiProjectManager, ProjectMetadata
    HAS_PROJECT_MANAGER = True
except ImportError:
    HAS_PROJECT_MANAGER = False
from src.data.csv_conversion.reading import CsvReadingMixin
from src.data.csv_conversion.time_column import TimeColumnMixin
from src.data.csv_conversion.numeric import NumericColumnsMixin
from src.data.csv_conversion.writing import MpaiWritingMixin

logger = logging.getLogger(__name__)


class CsvToMpaiConverter(CsvReadingMixin, TimeColumnMixin, NumericColumnsMixin, MpaiWritingMixin, QObject):
    """
    Convert CSV to MPAI format using streaming.
    
    Features:
    - Polars streaming input (low memory)
    - C++ MPAI writer (fast, compressed)
    - Pre-compute statistics
    - Progress tracking
    - Dynamic chunk sizing based on available RAM
    - Robust Data Cleaning & Time Column Management (Architecture Compliant)
    - Auto-fix for quote-wrapped files (Excel export bug)
    
    Memory Usage: Configurable (default < 20% of system RAM)
    """
    
    # Bump when the converted output changes for the same input and settings;
    # DataLoader includes it in the cache key so stale MPAIs are regenerated
    CONVERTER_VERSION = 8

    # Signals
    progress = Signal(str, int)  # message, percentage
    finished = Signal(str)  # output_file
    error = Signal(str)  # error_message
    statistics_computed = Signal(dict)  # column_name -> stats
    
    def __init__(self, csv_path: str, mpai_path: str, 
                 chunk_size: int = 1_000_000,
                 compression_level: int = 0,  # TEMPORARY: Set to 0 to bypass ZSTD issues
                 memory_limit_percent: float = 20.0,
                 settings: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.csv_path = csv_path
        self.mpai_path = mpai_path
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.memory_limit_percent = memory_limit_percent
        self.settings = settings or {}

        # Format settings from the import dialog
        self.delimiter = self.settings.get('delimiter') or ','
        self.decimal_comma = bool(self.settings.get('decimal_comma')) and self.delimiter != ','
        self.encoding = self.settings.get('encoding') or 'utf-8'

        # Effective header/data rows in working_csv_path. These change after
        # preprocessing, but self.settings['header_row'/'start_row'] are never
        # modified: DataLoader retries with the same settings dict and stores
        # it with the loaded file, so it must keep the user's choice.
        self._read_header_row = self.settings.get('header_row')  # None = no header
        self._read_start_row = self.settings.get('start_row', 0) or 0

        self.cancelled = False
        # True if the time column held dates/times (stored as epoch seconds);
        # the loader enables the datetime axis from this
        self.time_is_datetime = False
        # Column -> [values that could not be read as numbers (stored as 0),
        #            non-empty values in the column]
        self.non_numeric_report: Dict[str, List[int]] = {}
        self._datetime_columns = set()  # text columns stored as epoch seconds
        # Column -> how its text times were read (kept for the next batches)
        self._time_parsers: Dict[str, tuple] = {}
        # Column -> (days added, last clock value): clock times run past
        # midnight (23:59:59 -> 00:00:00) across batches
        self._clock_state: Dict[str, tuple] = {}
        self.start_time = 0.0
        self.current_time_offset = 0.0  # For streaming time generation
        
        # Temp file management
        self.temp_dir = None
        self.working_csv_path = self.csv_path # May change if fixing quotes
        
        # ZIP64 container support
        self.use_container = settings.get('use_container', True) if settings else True
        self.lod_data = {}  # Will store LOD parquet bytes for packaging
        
        # Performance metrics
        self.metrics = {
            'csv_size_mb': 0.0,
            'mpai_size_mb': 0.0,
            'compression_ratio': 0.0,
            'conversion_time_sec': 0.0,
            'throughput_mb_per_sec': 0.0,
            'row_count': 0,
            'column_count': 0,
        }
    
    def cancel(self):
        """Cancel conversion."""
        self.cancelled = True
        logger.info("Conversion cancelled by user")
    
    def convert(self):
        """
        Convert CSV to MPAI format.
        """
        try:
            self.start_time = time.time()
            perf_log = self._log_performance
            
            # Check if CSV exists
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
            csv_size = os.path.getsize(self.csv_path)
            self.metrics['csv_size_mb'] = csv_size / (1024 * 1024)
            
            self.progress.emit(f"Converting {self.metrics['csv_size_mb']:.2f} MB CSV...", 0)
            
            # Step 0: Check & Fix Quote Wrapping (Excel Bug)
            self.progress.emit("Veri formatı kontrol ediliyor...", 2)
            self._check_and_fix_quote_wrapping()
            
            # Step 1: Scan CSV (MetaData)
            t_scan = time.perf_counter()
            self.progress.emit("Step 1/4: Scanning CSV...", 5)
            # Just to get initial schema and row count estimate
            # Cleaning will be applied per-batch during writing
            lazy_frame, initial_schema = self._scan_csv()
            
            # Clean column names in schema for metadata
            cleaned_columns = self._get_cleaned_column_names(initial_schema.keys())
            
            # Update schema keys
            schema = {cleaned_columns[k]: v for k, v in initial_schema.items()}
            
            # Add time column to schema if it will be generated
            if self._will_generate_time_column(schema):
                 time_col = self.settings.get('new_time_column_name', 'time_generated')
                 # If time column is generated, it's not in the input schema, but will be in output
                 schema[time_col] = pl.Float64
            
            perf_log("scan_csv", t_scan, extra={"columns": len(schema)})
            
            # Calculate optimal chunk size based on RAM
            self._calculate_optimal_chunk_size(schema)
            
            # Step 2: Write MPAI (Read -> Clean -> Write Stream)
            t_write = time.perf_counter()
            self.progress.emit("Step 2/4: Processing & Writing...", 10)
            
            # Default stats (placeholders)
            column_stats = self._get_default_statistics(schema)
            
            # Main processing loop
            row_count = self._write_mpai_streaming(schema, column_stats)
            
            self.metrics['row_count'] = row_count
            self.metrics['column_count'] = len(schema)
            perf_log("write_mpai", t_write, extra={"rows": row_count})
            
            # Step 3: Generate LOD Pyramid (Pre-computed aggregations)
            t_lod = time.perf_counter()
            self.progress.emit("Step 3/5: Generating LOD pyramid...", 88)
            self._generate_lod_pyramid(schema, row_count)
            perf_log("lod_pyramid", t_lod)
            
            # Step 4: Finalize
            t_finalize = time.perf_counter()
            self.progress.emit("Step 4/5: Finalizing...", 95)
            self._finalize()
            perf_log("finalize", t_finalize)
            
            # Success!
            self.progress.emit("Conversion complete!", 100)
            self.finished.emit(self.mpai_path)
            return True
            
        except Exception as e:
            logger.exception("Conversion failed:")
            self.error.emit(str(e))
            return False
        finally:
            # Cleanup temp files
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Cleaned up temp dir: {self.temp_dir}")
                except:
                    pass

    def _process_batch(self, df: pl.DataFrame, old_to_new_cols: Dict[str, str]) -> pl.DataFrame:
        """Apply cleaning and time generation to a single batch."""
        
        # 1. Rename Columns
        df = df.rename(old_to_new_cols)

        # Blank lines (e.g. trailing newline at EOF) come through as all-null
        # rows; filling them with 0 below would add fake (0, 0) samples
        if df.width > 0:
            df = df.filter(~pl.all_horizontal(pl.all().is_null()))

        # 2. Null & Inf Handling
        # Eager execution on batch
        fill_exprs = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                 fill_exprs.append(pl.col(col).fill_null(0).alias(col))
        
        if fill_exprs:
            df = df.with_columns(fill_exprs)
            
        # Inf handling
        inf_exprs = []
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64]:
                 inf_exprs.append(
                     pl.when(pl.col(col).is_infinite())
                     .then(None)
                     .otherwise(pl.col(col))
                     .fill_null(0.0)
                     .alias(col)
                 )
        if inf_exprs:
            df = df.with_columns(inf_exprs)

        # 3. Time Column Generation
        df = self._handle_time_column_batch(df)
        
        return df

    def _log_performance(self, stage: str, t_start: float, extra: Optional[Dict[str, Any]] = None):
        """Lightweight perf log helper."""
        try:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            log_payload = {"stage": stage, "elapsed_ms": round(elapsed_ms, 2), "rss_mb": round(rss_mb, 2)}
            if extra:
                log_payload.update(extra)
            logger.info(f"[PERF][CSV->MPAI] {log_payload}")
        except Exception:
            pass
    
    def _finalize(self):
        """Finalize conversion and calculate metrics."""
        if os.path.exists(self.mpai_path):
            mpai_size = os.path.getsize(self.mpai_path)
            self.metrics['mpai_size_mb'] = mpai_size / (1024 * 1024)
        
        if self.metrics['mpai_size_mb'] > 0:
            self.metrics['compression_ratio'] = (
                self.metrics['csv_size_mb'] / self.metrics['mpai_size_mb']
            )
        
        self.metrics['conversion_time_sec'] = time.time() - self.start_time
        
        if self.metrics['conversion_time_sec'] > 0:
            self.metrics['throughput_mb_per_sec'] = (
                self.metrics['csv_size_mb'] / self.metrics['conversion_time_sec']
            )
        
        # Package into ZIP64 container if enabled
        if self.use_container and HAS_PROJECT_MANAGER:
            try:
                self._package_to_container()
            except Exception as e:
                logger.error(f"Failed to package into container: {e}")
                # Continue without packaging - file is still usable in binary format
        
        logger.info("Conversion Summary")
        logger.info(f"Throughput: {self.metrics['throughput_mb_per_sec']:.2f} MB/s")
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()


def convert_csv_to_mpai(csv_path: str, mpai_path: Optional[str] = None,
                       progress_callback: Optional[Callable] = None,
                       settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert CSV to MPAI format (convenience function).
    """
    if mpai_path is None:
        mpai_path = os.path.splitext(csv_path)[0] + '.mpai'
    
    converter = CsvToMpaiConverter(csv_path, mpai_path, settings=settings)
    
    if progress_callback:
        converter.progress.connect(progress_callback)
    
    # Run conversion
    converter.convert()
    
    return converter.get_metrics()
