import logging
import os
import time
import psutil
import tempfile
import shutil
import io
import zipfile
from typing import Optional, Dict, Any, Callable, List
import polars as pl
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal

# Date + time formats tried when Polars cannot infer the format itself.
# %.f makes fractional seconds optional.
DATETIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S%.f %p", "%d/%m/%Y %I:%M:%S%.f %p", "%Y-%m-%d %I:%M:%S%.f %p",
    "%d.%m.%Y %H:%M:%S%.f", "%d/%m/%Y %H:%M:%S%.f", "%m/%d/%Y %H:%M:%S%.f",
    "%Y/%m/%d %H:%M:%S%.f", "%d-%m-%Y %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f",
    "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M",
)

# Times without a date. Seconds may have a fraction (. or ,).
#   h:m:s, h/m/s, h:m:s:ms (optionally AM/PM)  -> clock time (or elapsed hours)
#   m:s                                         -> minutes and seconds
#   1h2m3.5s, 2m 30s, 45s                       -> duration
_SECONDS = r"(\d{1,2}(?:[.,]\d+)?)"
CLOCK_PATTERN = (r"^(\d{1,3})[:/](\d{1,2})[:/]" + _SECONDS
                 + r"(?::(\d{1,3}))?\s*([AaPp]\.?[Mm]\.?)?$")
MIN_SEC_PATTERN = r"^(\d{1,4}):" + _SECONDS + r"$"
DURATION_PATTERN = (r"(?i)^(?:(\d+(?:[.,]\d+)?)\s*h(?:ours?|rs?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*m(?:in(?:utes?)?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*s(?:ec(?:onds?)?)?)?$")

# "Zaman Birimi" of the import dialog -> factor to seconds
TIME_UNIT_FACTORS = {'saniye': 1.0, 'milisaniye': 1e-3, 'mikrosaniye': 1e-6, 'nanosaniye': 1e-9}

# Import MpaiProjectManager for ZIP64 container support
try:
    from src.data.mpai_project_manager import MpaiProjectManager, ProjectMetadata
    HAS_PROJECT_MANAGER = True
except ImportError:
    HAS_PROJECT_MANAGER = False

logger = logging.getLogger(__name__)


class CsvToMpaiConverter(QObject):
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

    def _check_and_fix_quote_wrapping(self):
        """
        Check if CSV has entire lines wrapped in quotes (Excel export bug).
        Also apply header_row and start_row settings by creating a preprocessed temp file.
        This ensures consistent behavior with the import dialog preview.
        """
        try:
            encoding = self.encoding
            header_row = self.settings.get('header_row')  # None means no header
            start_row = self.settings.get('start_row', 0) or 0  # 0-indexed data start

            if header_row is not None and start_row <= header_row:
                raise ValueError(
                    f"Veri başlangıç satırı ({start_row}) header satırından ({header_row}) "
                    f"sonra olmalı. Import ayarlarını kontrol edin."
                )

            # Check for quote wrapping first
            needs_quote_fix = False
            with open(self.csv_path, 'r', encoding=encoding, errors='replace') as f:
                lines_checked = 0
                quote_wrapped_count = 0

                for line in f:
                    line = line.strip()
                    if not line: continue

                    if (line.startswith('"') and line.endswith('"') and
                        self.delimiter in line):
                        quote_wrapped_count += 1

                    lines_checked += 1
                    if lines_checked >= 5: break

                if lines_checked > 0 and (quote_wrapped_count / lines_checked) > 0.5:
                    needs_quote_fix = True
                    logger.warning("Detected quote-wrapped CSV lines. Applying auto-fix.")

            # Check if we need row preprocessing (non-standard header/start positions)
            needs_row_preprocessing = (header_row is not None and header_row > 0) or \
                                      (header_row is None and start_row > 0) or \
                                      (header_row is not None and start_row != header_row + 1)

            # Polars only reads UTF-8: re-encode other encodings if the file
            # has any non-ASCII bytes (pure ASCII is identical in UTF-8)
            needs_reencode = self._needs_reencode(encoding)

            if needs_quote_fix or needs_row_preprocessing or needs_reencode:
                logger.info(f"[CSV PREPROCESS] header_row={header_row}, start_row={start_row}, "
                           f"quote_fix={needs_quote_fix}, row_preprocess={needs_row_preprocessing}, "
                           f"reencode={needs_reencode} ({encoding})")

                # Create temp directory if not exists
                if not self.temp_dir:
                    self.temp_dir = tempfile.mkdtemp()
                temp_csv = os.path.join(self.temp_dir, "preprocessed_data.csv")

                # Stream line by line (files can be several GB)
                lines_written = 0
                header_found = header_row is None
                with open(self.csv_path, 'r', encoding=encoding, errors='replace') as fin, \
                     open(temp_csv, 'w', encoding='utf-8', newline='') as fout:

                    for i, line in enumerate(fin):
                        if i != header_row and i < start_row:
                            continue

                        line = line.rstrip('\r\n')

                        # Remove wrapping quotes if needed
                        if needs_quote_fix and line.startswith('"') and line.endswith('"') and len(line) > 1:
                            line = line[1:-1]

                        if i == header_row:
                            header_found = True
                            logger.info(f"[CSV PREPROCESS] Header from line {header_row}: {line[:50]}...")

                        fout.write(line + '\n')
                        lines_written += 1

                if not header_found:
                    raise ValueError(f"Header satırı ({header_row}) dosya boyutunu aşıyor")

                logger.info(f"[CSV PREPROCESS] Wrote {lines_written} lines to temp file")

                # Switch to using preprocessed file: header (if any) is now
                # line 0 and data follows directly
                self.working_csv_path = temp_csv
                self._read_header_row = 0 if header_row is not None else None
                self._read_start_row = 1 if header_row is not None else 0

                # IMPORTANT: Update time_column to match cleaned column names
                # This ensures user's time column selection works after column name cleaning
                if 'time_column' in self.settings and self.settings['time_column']:
                    original_time = self.settings['time_column']
                    # Apply same cleaning logic that will be used for all columns
                    cleaned_time = str(original_time).strip()
                    if cleaned_time != original_time:
                        self.settings['time_column'] = cleaned_time
                        logger.info(f"[CSV PREPROCESS] Updated time_column: '{original_time}' -> '{cleaned_time}'")
                
                logger.info(f"[CSV PREPROCESS] Using preprocessed temp CSV: {temp_csv}")

        except ValueError:
            raise
        except Exception as e:
            # Reading the original file with skip settings is the closest
            # fallback; never silently drop the user's header/start rows.
            logger.exception(f"CSV preprocessing failed: {e}")
            self.working_csv_path = self.csv_path
            self._read_header_row = self.settings.get('header_row')
            self._read_start_row = self.settings.get('start_row', 0) or 0

    def _needs_reencode(self, encoding: str, chunk_size: int = 16 * 1024 * 1024) -> bool:
        """True if the file is not UTF-8 and contains non-ASCII bytes."""
        normalized = encoding.lower().replace('_', '-')
        if normalized in ('utf-8', 'utf8', 'ascii'):
            return False
        if normalized in ('utf-16', 'utf-8-sig'):
            return True  # BOM / multi-byte encodings always need conversion
        with open(self.csv_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return False
                if not chunk.isascii():
                    return True

    def _csv_read_options(self) -> Dict[str, Any]:
        """Polars read options shared by scan and batched read."""
        has_header = self._read_header_row is not None
        if has_header:
            skip_rows = self._read_header_row
            skip_rows_after_header = max(0, self._read_start_row - self._read_header_row - 1)
        else:
            skip_rows = self._read_start_row
            skip_rows_after_header = 0

        # infer_schema_length=0 reads every column as text; _to_numeric_series
        # converts it. Polars' own inference only samples the first rows and
        # silently turns later non-numeric values into null (then 0), which
        # would hide them from the non-numeric report.
        return dict(
            separator=self.delimiter,
            has_header=has_header,
            skip_rows=skip_rows,
            skip_rows_after_header=skip_rows_after_header,
            encoding='utf8-lossy',
            infer_schema_length=0,
            truncate_ragged_lines=True,
            low_memory=True,
        )

    def _get_cleaned_column_names(self, columns) -> Dict[str, str]:
        """Map old column names to clean ones."""
        old_to_new = {}
        used_names = set()
        
        for col in columns:
            clean_name = str(col).strip()
            # Prevent duplicates
            base_name = clean_name
            counter = 1
            while clean_name in used_names:
                clean_name = f"{base_name}_{counter}"
                counter += 1
            
            used_names.add(clean_name)
            old_to_new[col] = clean_name
            
        return old_to_new

    def _will_generate_time_column(self, schema: Dict[str, Any]) -> bool:
        """Check if a new time column will be added."""
        create_custom = self.settings.get('create_custom_time', False)
        time_col_name = self.settings.get('time_column')
        
        # If explicitly requested OR no valid time column exists
        if create_custom:
            return True
        if not time_col_name or time_col_name not in schema:
            return True
            
        return False

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

    def _handle_time_column_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle time column creation for batch, maintaining state."""
        create_custom = self.settings.get('create_custom_time', False)
        time_col_name = self.settings.get('time_column')
        new_col_name = self.settings.get('new_time_column_name', 'time_generated')
        
        # DEBUG: Log what we're looking for and what's available
        logger.info(f"[TIME TRACE] Looking for time_column='{time_col_name}' in columns: {df.columns[:5]}...")
        
        # Scenario A: Generate Custom Time
        # OR Scenario C: Fallback (No time column found)
        time_col_found = time_col_name and time_col_name in df.columns
        
        # If not found by exact name, try to find it by partial match
        if not time_col_found and time_col_name:
            # Try case-insensitive or partial match
            for col in df.columns:
                if col.lower() == time_col_name.lower() or \
                   time_col_name.lower() in col.lower() or \
                   col.lower() in time_col_name.lower():
                    logger.info(f"[TIME TRACE] Found by partial match: '{time_col_name}' -> '{col}'")
                    time_col_name = col
                    self.settings['time_column'] = col  # Update settings
                    time_col_found = True
                    break
        
        should_generate = create_custom or not time_col_found
        
        logger.info(f"[TIME TRACE] time_col_found={time_col_found}, should_generate={should_generate}")
        
        if should_generate:
            sampling_freq = self.settings.get('sampling_frequency', 1000.0)
            if sampling_freq <= 0: sampling_freq = 1000.0
            time_step = 1.0 / sampling_freq
            
            # Generate time array for this batch
            n_rows = df.height
            start = self.current_time_offset
            # Linspace is inclusive, arange is not. 
            # We want [start, start + step, ..., start + (n-1)*step]
            time_arr = np.linspace(start, start + (n_rows - 1) * time_step, n_rows, dtype=np.float64)
            
            # Update offset for next batch
            self.current_time_offset += n_rows * time_step
            
            # Add column
            target_name = new_col_name if create_custom else 'time'
            df = df.with_columns(pl.Series(target_name, time_arr))
            logger.info(f"[TIME TRACE] Generated time column '{target_name}' with {n_rows} rows")
            
        # Scenario B: Use/Fix Existing Time Column
        elif time_col_name in df.columns:
            # Ensure float64
            try:
                col = df[time_col_name]
                logger.info(f"[TIME TRACE] Using existing time column '{time_col_name}' (dtype={col.dtype})")
                
                # Log first few values for debugging
                if col.len() > 0:
                    sample_values = col.head(min(5, col.len())).to_list()
                    logger.info(f"[TIME TRACE] Sample values: {sample_values}")
                
                if col.dtype == pl.Utf8 or col.dtype == pl.String:
                    # String column - need special parsing
                    logger.info(f"[TIME TRACE] Time column is String, attempting conversion...")
                    
                    # Numbers (decimal comma aware) or date/time text -> epoch
                    # seconds; invalid values become 0 and are reported
                    converted_col = self._to_numeric_series(col)
                    if time_col_name in self._datetime_columns:
                        self.time_is_datetime = True
                        logger.info(f"[TIME TRACE] Parsed as datetime, converted to epoch seconds")
                    df = df.with_columns(converted_col.alias(time_col_name))
                    
                    # Log result
                    result_col = df[time_col_name]
                    if result_col.len() > 0:
                        sample_after = result_col.head(min(5, result_col.len())).to_list()
                        logger.info(f"[TIME TRACE] After conversion: {sample_after}")
                    
                elif col.dtype in (pl.Datetime, pl.Date):
                    # try_parse_dates gives Datetime/Date; a plain Float64 cast
                    # would yield microseconds, but the time axis expects epoch
                    # seconds (same as the string branch above)
                    logger.info(f"[TIME TRACE] Converting {col.dtype} to epoch seconds")
                    epoch_s = (col.dt.epoch("us").cast(pl.Float64) / 1e6) if col.dtype == pl.Datetime \
                        else col.dt.epoch("s").cast(pl.Float64)
                    df = df.with_columns(epoch_s.fill_null(0.0).alias(time_col_name))
                    self.time_is_datetime = True
                elif col.dtype not in [pl.Float64, pl.Float32]:
                    # Numeric but not float - simple cast
                    logger.info(f"[TIME TRACE] Casting {col.dtype} to Float64")
                    df = df.with_columns(col.cast(pl.Float64, strict=False).fill_null(0.0).alias(time_col_name))
                else:
                    # Already float - just fill nulls
                    logger.info(f"[TIME TRACE] Already Float64, filling nulls")
                    df = df.with_columns(col.fill_null(0.0).alias(time_col_name))

                # Time unit / Unix timestamp chosen in the import dialog
                df = df.with_columns(self._scale_time_column(df[time_col_name]).alias(time_col_name))
                    
            except Exception as e:
                logger.error(f"[TIME TRACE] Failed to process time column: {e}")
                import traceback
                traceback.print_exc()
                
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
    
    def _calculate_optimal_chunk_size(self, schema: Dict[str, Any]):
        """Calculate optimal chunk size."""
        try:
            mem = psutil.virtual_memory()
            target_ram_bytes = mem.total * (self.memory_limit_percent / 100.0)
            usable_ram_bytes = max(0, target_ram_bytes - (100 * 1024 * 1024))
            
            estimated_row_bytes = len(schema) * 16 # Rough estimate
            if estimated_row_bytes == 0: estimated_row_bytes = 100
                
            batch_ram_target = usable_ram_bytes * 0.5
            optimal_chunk_size = int(batch_ram_target / estimated_row_bytes)
            
            # Clamp limits
            self.chunk_size = max(1_000, min(optimal_chunk_size, 5_000_000))
            logger.info(f"Optimal Chunk Size: {self.chunk_size:,} rows")
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal chunk size: {e}")
            self.chunk_size = 50_000
    
    def _scan_csv(self):
        """Scan CSV file (metadata only)."""
        # Use working_csv_path (might be temp file)
        options = self._csv_read_options()
        logger.info(f"[CSV SCAN] header_row={self._read_header_row}, start_row={self._read_start_row}, "
                    f"separator={self.delimiter!r}, decimal_comma={self.decimal_comma}, "
                    f"skip_rows={options['skip_rows']}, skip_after_header={options['skip_rows_after_header']}")

        lazy_frame = pl.scan_csv(
            self.working_csv_path,
            rechunk=False,
            **options,
        )
        schema = lazy_frame.collect_schema()
        return lazy_frame, schema
    
    def _count_rows(self, lazy_frame: pl.LazyFrame) -> int:
        count_df = lazy_frame.select(pl.count()).collect(streaming=True)
        return count_df.item()
    
    def _get_default_statistics(self, schema: Dict[str, Any]) -> Dict[str, Dict]:
        column_stats = {}
        for col_name in schema.keys():
            column_stats[col_name] = {
                'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'rms': 0.0
            }
        return column_stats
    
    def _map_polars_type(self, pl_type) -> str:
        """Map Polars dtype to a string tag (was C++ DataType enum)."""
        if pl_type in [pl.Float64, pl.Float32]: return "FLOAT64"
        elif pl_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]: return "INT64"
        elif pl_type in [pl.Utf8, pl.String]: return "STRING"
        elif pl_type in [pl.Datetime, pl.Date]: return "DATETIME"
        else: return "FLOAT64"

    def _write_mpai_streaming(self, schema: Dict[str, Any], column_stats: Dict[str, Dict]) -> int:
        """Write MPAI file using Python MpaiStreamWriter (Zero-Copy Format)."""
        try:
            from src.data.data_engine import MpaiStreamWriter
        except ImportError:
            raise RuntimeError("MpaiStreamWriter not found in src.data.data_engine")
        
        # NOTE: We can't know exact row count beforehand easily with streaming + cleaning
        # So we write a placeholder or 0, and C++ writer handles it or we update later
        # However, MpaiWriter needs row_count for header.
        # We will estimate or count first if crucial. 
        # For now, let's scan-count first as in original code, but on input CSV
        scan_lf, _ = self._scan_csv()
        total_rows_input = self._count_rows(scan_lf)
        
        # Create MPAI writer
        writer = MpaiStreamWriter(self.mpai_path)
        
        # Register Column Metadata
        column_names = list(schema.keys())
        # Store mapping for cleaning
        # Need original column names from CSV to map
        _, original_schema = self._scan_csv()
        old_to_new = self._get_cleaned_column_names(original_schema.keys())
        
        # CRITICAL: Update time_column to use cleaned/renamed column name
        if 'time_column' in self.settings and self.settings['time_column']:
            original_time_col = self.settings['time_column']
            if original_time_col in old_to_new:
                cleaned_time_col = old_to_new[original_time_col]
                if cleaned_time_col != original_time_col:
                    logger.info(f"[TIME COLUMN] Mapping: '{original_time_col}' -> '{cleaned_time_col}'")
                    self.settings['time_column'] = cleaned_time_col
            else:
                new_names = set(old_to_new.values())
                if original_time_col in new_names:
                    logger.info(f"[TIME COLUMN] '{original_time_col}' found in cleaned column names")
                else:
                    logger.warning(f"[TIME COLUMN] '{original_time_col}' not found in any column names!")
        
        # Initialize Writer
        sampling_freq = self.settings.get('sampling_frequency', 1000.0)
        # Time of the first row: the reader builds plot times as
        # start_time + i * dt, so a log starting at 09:08:49 must not start at 0
        start_time = 0.0
        
        # AUTO-DETECT Sampling Frequency if Time Column exists
        if 'time_column' in self.settings:
            effective_time_col = self.settings['time_column']
            # Find matching column in original schema
             # We need to map cleaned name back or check scan_lf
            if effective_time_col in scan_lf.collect_schema().names() or effective_time_col in old_to_new.values():
                try:
                    # Get first few rows to calculate dt
                    sample_df = scan_lf.head(100).collect()
                    
                    # Determine which column in sample_df matches effective_time_col
                    target_col = effective_time_col
                    if effective_time_col not in sample_df.columns:
                        # Reverse lookup
                        for old, new in old_to_new.items():
                            if new == effective_time_col:
                                target_col = old
                                break
                    
                    if target_col in sample_df.columns:
                        time_vals = self._scale_time_column(
                            self._to_numeric_series(sample_df.get_column(target_col))).to_numpy()
                        # Sampling only: don't count these rows in the report,
                        # and the batches start again from the first row
                        self.non_numeric_report.clear()
                        self._clock_state.clear()
                        finite = time_vals[np.isfinite(time_vals)]
                        if len(finite):
                            start_time = float(finite[0])
                        # Calculate differences
                        if len(time_vals) > 5:
                            diffs = np.diff(time_vals)
                            median_diff = np.median(diffs)
                            if median_diff > 0:
                                detected_freq = 1.0 / median_diff
                                logger.info(f"[CSV AUTO-DETECT] Calculated Fs = {detected_freq:.2f} Hz (dt={median_diff:.6f}s)")
                                # Update if default or significantly different?
                                # Prefer detected if it looks valid
                                sampling_freq = detected_freq
                                self.settings['sampling_frequency'] = sampling_freq
                except Exception as e:
                    logger.warning(f"[CSV AUTO-DETECT] Failed to detect sampling rate: {e}")

        writer.initialize(column_names, sampling_freq, start_time=start_time, overwrite=True)

        # Read CSV in Batches using same options as _scan_csv
        options = self._csv_read_options()
        logger.info(f"[CSV BATCH READ] header_row={self._read_header_row}, start_row={self._read_start_row}, "
                    f"skip_rows={options['skip_rows']}, skip_after_header={options['skip_rows_after_header']}")

        reader = pl.read_csv_batched(
            self.working_csv_path,
            batch_size=self.chunk_size,
            **options,
        )
        
        chunk_id = 0
        rows_processed = 0
        
        while True:
            if self.cancelled: break
            
            batches = reader.next_batches(1)
            if not batches: break
            
            df_batch = batches[0]
            
            # --- APPLY CLEANING & TIME GENERATION ---
            df_batch = self._process_batch(df_batch, old_to_new)
            df_batch = self._numeric_batch(df_batch)
            
            # Validate schema consistency (important if cleaning changes schema)
            # Just ensure we have the columns we promised in header
            current_batch_size = df_batch.height
            
            # Progress update
            pct = 10 + int((rows_processed / max(total_rows_input, 1)) * 85)
            self.progress.emit(f"Writing chunk {chunk_id}... ({rows_processed:,} rows)", pct)
            
            # Prepare Chunk Data for Writer
            chunk_data = {}
            for col_name in column_names:
                if col_name in df_batch.columns:
                    series = self._to_numeric_series(df_batch.get_column(col_name))
                    data = series.to_numpy()  # nulls -> NaN -> 0 below

                    # Handle NaN/Inf
                    if data.dtype.kind in 'fi':
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Normalize to Float64
                    if data.dtype != np.float64:
                        try:
                            data = data.astype(np.float64)
                        except (ValueError, TypeError):
                            logger.warning(
                                "[CSV] Column '%s' cannot be cast to float64, filling with zeros",
                                col_name,
                            )
                            self._count_non_numeric(col_name, current_batch_size, current_batch_size)
                            data = np.zeros(current_batch_size, dtype=np.float64)
                         
                    chunk_data[col_name] = data
                else:
                    # Missing column fill
                    chunk_data[col_name] = np.zeros(current_batch_size, dtype=np.float64)
            
            # Write Chunk
            writer.write_chunk(chunk_data)
            
            rows_processed += current_batch_size
            chunk_id += 1
            del df_batch
            
        # Finalize
        writer.close()
        logger.info(f"MPAI directory written: {self.mpai_path}")
        
        return rows_processed
    
    def _numeric_expr(self, col: str) -> pl.Expr:
        """Text column -> Float64 expression (null where not a number)."""
        text = pl.col(col).str.strip_chars()
        if self.decimal_comma:
            text = text.str.replace_all(",", ".", literal=True)
        return text.cast(pl.Float64, strict=False)

    def _numeric_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert every text column of a batch to Float64 in one parallel pass.

        Values that are not numbers become null (stored as 0) and are counted
        in self.non_numeric_report, so one "N/A" no longer zeroes a whole
        column and the user is told which columns lost values.
        """
        text_cols = [c for c, dt in df.schema.items() if dt == pl.String]
        if not text_cols:
            return df
        converted = df.select([self._numeric_expr(c).alias(c) for c in text_cols])
        replacements = []
        for c in text_cols:
            # Empty cells are already null in the text column
            failed = converted[c].null_count() - df[c].null_count()
            if failed:
                replacements.append(self._resolve_non_numeric(df[c], converted[c], failed))
        if replacements:
            converted = converted.with_columns(replacements)
        return df.with_columns(converted)

    def _to_numeric_series(self, series: pl.Series) -> pl.Series:
        """Single-column variant of _numeric_batch (time column, sampling)."""
        if series.dtype == pl.String:
            numeric = series.to_frame().select(self._numeric_expr(series.name)).to_series()
            failed = numeric.null_count() - series.null_count()
            if failed:
                numeric = self._resolve_non_numeric(series, numeric, failed)
            return numeric
        if series.dtype == pl.Datetime:
            return series.dt.epoch("us").cast(pl.Float64) / 1e6
        if series.dtype == pl.Date:
            return series.dt.epoch("s").cast(pl.Float64)
        if series.dtype == pl.Boolean:
            return series.cast(pl.Float64)
        return series

    def _resolve_non_numeric(self, text: pl.Series, numeric: pl.Series, failed: int) -> pl.Series:
        """
        A text column had values that are not numbers. If most of them are
        dates or times, store them as seconds instead; count what remains
        invalid.
        """
        total = len(text) - text.null_count()
        if failed * 2 >= total:
            parsed = self._parse_time_text(text)
            if parsed is not None:
                numeric = parsed.alias(text.name)
                self._datetime_columns.add(text.name)
                failed = numeric.null_count() - text.null_count()
        if failed:
            self._count_non_numeric(text.name, failed, total)
        return numeric

    def _user_time_format(self, col_name: str) -> Optional[str]:
        """strftime format chosen in the import dialog for the time column."""
        if col_name != self.settings.get('time_column'):
            return None
        fmt = self.settings.get('time_format')
        return fmt if fmt and '%' in fmt else None

    def _parse_time_text(self, text: pl.Series) -> Optional[pl.Series]:
        """
        Read date/time text as seconds (Float64, null where unreadable), or
        None if the column does not hold dates or times.

        Dates with a time become epoch seconds; times without a date become
        seconds since midnight (elapsed seconds for durations). The format
        found in the first batch is tried first in the following ones.
        """
        name = text.name
        text = text.str.strip_chars()
        total = len(text) - text.null_count()
        if total == 0:
            return None

        def readable(result):
            return result is not None and (len(result) - result.null_count()) * 2 >= total

        cached = self._time_parsers.get(name)
        if cached is not None:
            result = self._apply_time_parser(text, name, cached)
            if readable(result):
                return result

        candidates = []
        user_fmt = self._user_time_format(name)
        if user_fmt:
            # The dialog's formats have whole seconds; %.f also accepts a fraction
            candidates.append(('datetime', user_fmt.replace('%S', '%S%.f')))
        candidates.append(('datetime', None))  # Polars infers the format
        candidates += [('datetime', fmt) for fmt in DATETIME_FORMATS]
        candidates += [('clock', None), ('min_sec', None), ('duration', None)]

        for parser in candidates:
            if parser == cached:
                continue
            result = self._apply_time_parser(text, name, parser)
            if readable(result):
                if user_fmt and parser != candidates[0] and name not in self._time_parsers:
                    logger.warning(f"[TIME] '{name}' does not match the chosen format "
                                   f"'{user_fmt}', read as {parser} instead")
                self._time_parsers[name] = parser
                logger.info(f"[TIME] Column '{name}' read as {parser[0]} {parser[1] or ''}")
                return result
        return None

    def _apply_time_parser(self, text: pl.Series, name: str, parser: tuple) -> Optional[pl.Series]:
        kind, fmt = parser
        try:
            if kind == 'datetime':
                dates = text.str.to_datetime(format=fmt, strict=False)
                return (dates.dt.epoch("us").cast(pl.Float64) / 1e6).alias(name)
            if kind == 'clock':
                return self._clock_seconds(text, name)
            if kind == 'min_sec':
                parts = text.str.extract_groups(MIN_SEC_PATTERN)
                minutes = parts.struct.field("1").cast(pl.Float64)
                seconds = parts.struct.field("2").str.replace(",", ".", literal=True).cast(pl.Float64)
                return (minutes * 60 + seconds).alias(name)
            if kind == 'duration':
                parts = text.str.extract_groups(DURATION_PATTERN)
                hours, minutes, seconds = (
                    parts.struct.field(str(i)).str.replace(",", ".", literal=True).cast(pl.Float64)
                    for i in (1, 2, 3))
                frame = pl.DataFrame({"h": hours, "m": minutes, "s": seconds})
                return frame.select(
                    # an empty match ("" or text without any unit) is not a duration
                    pl.when(pl.col("h").is_not_null() | pl.col("m").is_not_null() | pl.col("s").is_not_null())
                    .then(pl.col("h").fill_null(0) * 3600 + pl.col("m").fill_null(0) * 60 + pl.col("s").fill_null(0))
                ).to_series().alias(name)
        except pl.exceptions.PolarsError:
            return None
        return None

    def _clock_seconds(self, text: pl.Series, name: str) -> pl.Series:
        """h:m:s / h/m/s (AM/PM) text -> seconds, continuing past midnight."""
        parts = text.str.extract_groups(CLOCK_PATTERN)
        hours = parts.struct.field("1").cast(pl.Float64).to_numpy()
        minutes = parts.struct.field("2").cast(pl.Float64).to_numpy()
        seconds = parts.struct.field("3").str.replace(",", ".", literal=True).cast(pl.Float64).to_numpy()
        millis = parts.struct.field("4").cast(pl.Float64).fill_null(0).to_numpy()
        ampm = parts.struct.field("5").str.to_lowercase().str.slice(0, 1).fill_null("").to_numpy()

        with np.errstate(invalid='ignore'):
            valid = ~np.isnan(hours) & (minutes < 60) & (seconds < 61)
        is_pm = ampm == 'p'
        has_ampm = (ampm == 'a') | is_pm
        hours = np.where(has_ampm, hours % 12 + np.where(is_pm, 12, 0), hours)
        values = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        values[~valid] = np.nan

        # A clock that goes back by more than 12 h passed midnight: add a day
        days, last = self._clock_state.get(name, (0, None))
        idx = np.flatnonzero(valid)
        if len(idx):
            seq = values[idx]
            prev = np.concatenate(([seq[0] if last is None else last], seq[:-1]))
            day_steps = np.cumsum(seq < prev - 43200)
            values[idx] = seq + (days + day_steps) * 86400.0
            self._clock_state[name] = (days + int(day_steps[-1]), float(seq[-1]))
        return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)

    def _scale_time_column(self, series: pl.Series) -> pl.Series:
        """Apply the import dialog's time unit / Unix timestamp choice to the time column."""
        if series.name in self._datetime_columns:
            return series  # read from date/time text: already seconds
        factor = TIME_UNIT_FACTORS.get(self.settings.get('time_unit') or 'saniye', 1.0)
        if self.settings.get('time_format') == 'Unix Timestamp':
            if factor == 1.0 and len(series) and float(series.abs().median() or 0) > 1e11:
                factor = 1e-3  # milliseconds since 1970
            self.time_is_datetime = True
        return series * factor if factor != 1.0 else series

    def _count_non_numeric(self, col_name: str, failed: int, total: int):
        counts = self.non_numeric_report.setdefault(col_name, [0, 0])
        counts[0] += failed
        counts[1] += total

    def _generate_lod_pyramid(self, schema: Dict[str, Any], row_count: int):
        """
        Generate LOD pyramid files for fast visualization at any zoom level.
        
        Creates lod1_100.parquet, lod2_10k.parquet, lod3_100k.parquet
        with pre-computed min/max values per bucket.
        """
        try:
            from src.data.lod_generator import LodGenerator
            
            # SKIP LOD: New format uses .red files which are generated during write
            # No need for Parquet pyramid
            logger.info("[LOD] Skipping Parquet pyramid generation (Using .red files)")
            return
            
            # Determine container path (same as MPAI file without extension)
            container_path = os.path.splitext(self.mpai_path)[0] + '_lod'
            
            # Get column names
            time_column = self.settings.get('time_column')
            if not time_column:
                # Try to find time column from schema
                for col in schema.keys():
                    if 'time' in col.lower():
                        time_column = col
                        break
                if not time_column:
                    time_column = list(schema.keys())[0]
            
            signal_columns = [col for col in schema.keys() if col != time_column]
            
            if not signal_columns:
                logger.warning("[LOD] No signal columns found, skipping pyramid")
                return
            
            logger.info(f"[LOD] Generating pyramid: time={time_column}, signals={signal_columns[:3]}...")
            
            # Create generator with progress callback
            def lod_progress(msg, pct):
                self.progress.emit(msg, pct)
            
            generator = LodGenerator(container_path, progress_callback=lod_progress)
            
            # Generate LOD from MPAI directory using Python reader
            try:
                from src.data.data_loader import MpaiDirectoryReader
                reader = MpaiDirectoryReader(self.mpai_path)
                lod_files = generator.generate_from_mpai_reader(
                    reader, time_column, signal_columns
                )
                self.metrics['lod_files'] = len(lod_files)
                logger.info(f"[LOD] Generated {len(lod_files)} LOD files: {list(lod_files.keys())}")
            except Exception as e:
                logger.error(f"[LOD] Failed to read MPAI for LOD generation: {e}")
                # Continue without LOD - not fatal
                
        except ImportError as e:
            logger.warning(f"[LOD] LodGenerator not available: {e}")
        except Exception as e:
            logger.error(f"[LOD] Pyramid generation failed: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def _package_to_container(self):
        """
        Finalize conversion. 
        MpaiStreamWriter already handled file closing. 
        We rely on the directory structure, so no need to package into ZIP64.
        """
        logger.info(f"Conversion finalized. Output directory: {self.mpai_path}")
        # Clean up temp working csv if different from original
        if self.working_csv_path != self.csv_path and os.path.exists(self.working_csv_path):
            try:
                os.remove(self.working_csv_path)
                logger.info("Removed temporary preprocessed CSV")
            except:
                pass

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
