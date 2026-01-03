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
        
        self.cancelled = False
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
            encoding = self.settings.get('encoding', 'utf-8')
            header_row = self.settings.get('header_row')  # None means no header
            start_row = self.settings.get('start_row', 0)  # 0-indexed data start
            
            # Check for quote wrapping first
            needs_quote_fix = False
            with open(self.csv_path, 'r', encoding=encoding, errors='replace') as f:
                lines_checked = 0
                quote_wrapped_count = 0
                
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    if (line.startswith('"') and line.endswith('"') and 
                        ',' in line and line.count(',') > 0):
                        quote_wrapped_count += 1
                    
                    lines_checked += 1
                    if lines_checked >= 5: break
                
                if lines_checked > 0 and (quote_wrapped_count / lines_checked) > 0.5:
                    needs_quote_fix = True
                    logger.warning("Detected quote-wrapped CSV lines. Applying auto-fix.")
            
            # Check if we need row preprocessing (non-standard header/start positions)
            needs_row_preprocessing = (header_row is not None and header_row > 0) or \
                                      (start_row > 1) or \
                                      (header_row is not None and start_row != header_row + 1)
            
            if needs_quote_fix or needs_row_preprocessing:
                logger.info(f"[CSV PREPROCESS] header_row={header_row}, start_row={start_row}, "
                           f"quote_fix={needs_quote_fix}, row_preprocess={needs_row_preprocessing}")
                
                # Create temp directory if not exists
                if not self.temp_dir:
                    self.temp_dir = tempfile.mkdtemp()
                temp_csv = os.path.join(self.temp_dir, "preprocessed_data.csv")
                
                # Read all lines and preprocess
                with open(self.csv_path, 'r', encoding=encoding, errors='replace') as fin, \
                     open(temp_csv, 'w', encoding='utf-8', newline='') as fout:
                    
                    all_lines = []
                    for line in fin:
                        line = line.rstrip('\r\n')
                        
                        # Remove wrapping quotes if needed
                        if needs_quote_fix and line.startswith('"') and line.endswith('"') and len(line) > 1:
                            line = line[1:-1]
                        
                        all_lines.append(line)
                    
                    # Now select the right lines based on header_row and start_row
                    output_lines = []
                    
                    if header_row is not None:
                        # Include header line
                        if header_row < len(all_lines):
                            output_lines.append(all_lines[header_row])
                            logger.info(f"[CSV PREPROCESS] Header from line {header_row}: {all_lines[header_row][:50]}...")
                        else:
                            logger.error(f"[CSV PREPROCESS] Header row {header_row} exceeds file length {len(all_lines)}")
                        
                        # Include data lines starting from start_row
                        for i in range(start_row, len(all_lines)):
                            output_lines.append(all_lines[i])
                    else:
                        # No header - just skip to start_row
                        for i in range(start_row, len(all_lines)):
                            output_lines.append(all_lines[i])
                    
                    # Write preprocessed lines
                    for line in output_lines:
                        fout.write(line + '\n')
                    
                    logger.info(f"[CSV PREPROCESS] Wrote {len(output_lines)} lines to temp file "
                               f"(original: {len(all_lines)} lines)")
                
                # Switch to using preprocessed file
                self.working_csv_path = temp_csv
                
                # IMPORTANT: Since we've already extracted header and data,
                # clear the skip_rows settings for subsequent processing
                self.settings['header_row'] = 0  # Header is now at line 0
                self.settings['start_row'] = 1   # Data starts at line 1
                
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
                
        except Exception as e:
            logger.error(f"CSV preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with original file if preprocessing fails

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
                    
                    # Replace comma with dot for European decimal format (1,5 -> 1.5)
                    converted_col = col.str.replace(',', '.', literal=True)
                    
                    # Try to cast to float
                    converted_col = converted_col.cast(pl.Float64, strict=False)
                    
                    # Check how many nulls we got after conversion
                    null_count = converted_col.null_count()
                    total_count = converted_col.len()
                    
                    if null_count > total_count * 0.5:
                        # More than 50% failed - something is wrong
                        logger.error(f"[TIME TRACE] Conversion failed for {null_count}/{total_count} values!")
                        # Try to parse as datetime and convert to float
                        try:
                            # Maybe it's a datetime string
                            dt_col = col.str.to_datetime(strict=False)
                            if dt_col.null_count() < null_count:
                                # Datetime parsing worked better
                                # Convert to epoch seconds
                                converted_col = dt_col.dt.epoch("s").cast(pl.Float64)
                                logger.info(f"[TIME TRACE] Parsed as datetime, converted to epoch seconds")
                        except:
                            pass
                    
                    # Fill remaining nulls with interpolation or 0
                    df = df.with_columns(converted_col.fill_null(0.0).alias(time_col_name))
                    
                    # Log result
                    result_col = df[time_col_name]
                    if result_col.len() > 0:
                        sample_after = result_col.head(min(5, result_col.len())).to_list()
                        logger.info(f"[TIME TRACE] After conversion: {sample_after}")
                    
                elif col.dtype not in [pl.Float64, pl.Float32]:
                    # Numeric but not float - simple cast
                    logger.info(f"[TIME TRACE] Casting {col.dtype} to Float64")
                    df = df.with_columns(col.cast(pl.Float64, strict=False).fill_null(0.0).alias(time_col_name))
                else:
                    # Already float - just fill nulls
                    logger.info(f"[TIME TRACE] Already Float64, filling nulls")
                    df = df.with_columns(col.fill_null(0.0).alias(time_col_name))
                    
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
        
        # Get header and start row settings from import dialog
        header_row = self.settings.get('header_row')  # None means no header
        start_row = self.settings.get('start_row', 0)  # 0-indexed data start
        
        # Calculate skip_rows based on settings
        # header_row: row number containing column names (0-indexed)
        # start_row: row number where data starts (0-indexed)
        has_header = header_row is not None
        
        if has_header:
            # Skip rows before header
            skip_rows = header_row
            # Skip rows between header and data (after header is read)
            skip_rows_after_header = max(0, start_row - header_row - 1)
        else:
            # No header, skip directly to data start
            skip_rows = start_row
            skip_rows_after_header = 0
        
        logger.info(f"[CSV SCAN] header_row={header_row}, start_row={start_row}, "
                    f"skip_rows={skip_rows}, skip_after_header={skip_rows_after_header}")
        
        lazy_frame = pl.scan_csv(
            self.working_csv_path,
            has_header=has_header,
            skip_rows=skip_rows,
            skip_rows_after_header=skip_rows_after_header,
            try_parse_dates=True,
            ignore_errors=True,
            low_memory=True,
            rechunk=False,
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
    
    def _map_polars_type(self, pl_type) -> Any:
        import time_graph_cpp as tgcpp
        if pl_type in [pl.Float64, pl.Float32]: return tgcpp.DataType.FLOAT64
        elif pl_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]: return tgcpp.DataType.INT64
        elif pl_type in [pl.Utf8, pl.String]: return tgcpp.DataType.STRING
        elif pl_type in [pl.Datetime, pl.Date]: return tgcpp.DataType.DATETIME
        else: return tgcpp.DataType.FLOAT64

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
        
        # AUTO-DETECT Sampling Frequency if Time Column exists
        if 'time_column' in self.settings:
            effective_time_col = self.settings['time_column']
            # Find matching column in original schema
             # We need to map cleaned name back or check scan_lf
            if effective_time_col in scan_lf.columns or effective_time_col in old_to_new.values():
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
                        time_vals = sample_df.get_column(target_col).to_numpy()
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

        writer.initialize(column_names, sampling_freq, overwrite=True)

        # Read CSV in Batches using same skip settings as _scan_csv
        header_row = self.settings.get('header_row')  # None means no header
        start_row = self.settings.get('start_row', 0)  # 0-indexed data start
        
        has_header = header_row is not None
        
        if has_header:
            skip_rows = header_row
            skip_rows_after_header = max(0, start_row - header_row - 1)
        else:
            skip_rows = start_row
            skip_rows_after_header = 0
        
        logger.info(f"[CSV BATCH READ] header_row={header_row}, start_row={start_row}, "
                    f"skip_rows={skip_rows}, skip_after_header={skip_rows_after_header}")
        
        reader = pl.read_csv_batched(
            self.working_csv_path,
            batch_size=self.chunk_size,
            has_header=has_header,
            skip_rows=skip_rows,
            skip_rows_after_header=skip_rows_after_header,
            try_parse_dates=True,
            ignore_errors=True,
            low_memory=True
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
                    series = df_batch.get_column(col_name)
                    data = series.to_numpy()
                    
                    # Handle NaN/Inf
                    if data.dtype.kind in 'fi':
                         data = np.nan_to_num(data, nan=0.0)
                    
                    # Normalize to Float64
                    if data.dtype != np.float64:
                         data = data.astype(np.float64)
                         
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
            
            # For now, we'll generate LOD from the written MPAI file
            # This requires reading back from the MPAI file
            try:
                import time_graph_cpp as tgcpp
                reader = tgcpp.MpaiReader(self.mpai_path)
                
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
