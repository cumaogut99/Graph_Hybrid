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
                 compression_level: int = 3,
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
            
        except Exception as e:
            logger.exception("Conversion failed:")
            self.error.emit(str(e))
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
        If detected, create a cleaned temporary CSV.
        """
        try:
            encoding = self.settings.get('encoding', 'utf-8')
            needs_fix = False
            
            # Read first few lines to detect issue
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
                    needs_fix = True
                    logger.warning("Detected quote-wrapped CSV lines. Applying auto-fix.")
            
            if needs_fix:
                # Create temp file
                self.temp_dir = tempfile.mkdtemp()
                temp_csv = os.path.join(self.temp_dir, "cleaned_data.csv")
                
                # Process line by line (memory efficient)
                with open(self.csv_path, 'r', encoding=encoding, errors='replace') as fin, \
                     open(temp_csv, 'w', encoding='utf-8', newline='') as fout:
                    
                    for line in fin:
                        line = line.rstrip('\r\n')
                        if line.startswith('"') and line.endswith('"') and len(line) > 1:
                            line = line[1:-1]
                        fout.write(line + '\n')
                
                # Switch to using cleaned file
                self.working_csv_path = temp_csv
                logger.info(f"Using cleaned temp CSV: {temp_csv}")
                
        except Exception as e:
            logger.error(f"Quote fix check failed: {e}")
            # Continue with original file if check fails

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
        
        # Scenario A: Generate Custom Time
        # OR Scenario C: Fallback (No time column found)
        should_generate = create_custom or (not time_col_name) or (time_col_name not in df.columns)
        
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
            
        # Scenario B: Use/Fix Existing Time Column
        elif time_col_name in df.columns:
            # Ensure float64
            # TODO: Improve parsing for batch streaming if needed
            try:
                col = df[time_col_name]
                if col.dtype not in [pl.Float64, pl.Float32]:
                    # Try simple cast first
                    df = df.with_columns(col.cast(pl.Float64, strict=False).fill_null(0.0))
            except:
                pass # Keep as is if conversion fails
                
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
        lazy_frame = pl.scan_csv(
            self.working_csv_path,
            has_header=True,
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
        """Write MPAI file using C++ writer with BATCHED processing."""
        try:
            import sys
            # Attempt to find DLL
            dll_paths = [os.getcwd(), os.path.join(os.getcwd(), "build", "Release")]
            for p in dll_paths:
                if p not in sys.path: sys.path.append(p)
            import time_graph_cpp as tgcpp
        except ImportError:
            raise RuntimeError("C++ module 'time_graph_cpp' not available.")
        
        # NOTE: We can't know exact row count beforehand easily with streaming + cleaning
        # So we write a placeholder or 0, and C++ writer handles it or we update later
        # However, MpaiWriter needs row_count for header.
        # We will estimate or count first if crucial. 
        # For now, let's scan-count first as in original code, but on input CSV
        scan_lf, _ = self._scan_csv()
        total_rows_input = self._count_rows(scan_lf)
        
        # Create MPAI writer
        writer = tgcpp.MpaiWriter(self.mpai_path, self.compression_level)
        
        # Write header (Use input row count, assumption: cleaning preserves row count)
        writer.write_header(total_rows_input, len(schema), os.path.basename(self.csv_path))
        
        # Register Column Metadata
        column_names = list(schema.keys())
        # Store mapping for cleaning
        # Need original column names from CSV to map
        _, original_schema = self._scan_csv()
        old_to_new = self._get_cleaned_column_names(original_schema.keys())
        
        for col_name in column_names:
            stats = column_stats.get(col_name, {})
            # Default to float64 if not in original schema (generated columns)
            col_type = schema.get(col_name, pl.Float64)
            
            col_metadata = tgcpp.ColumnMetadata()
            col_metadata.name = col_name
            col_metadata.data_type = self._map_polars_type(col_type)
            col_metadata.unit = "" 
            
            # Zero stats for now
            col_metadata.statistics.mean = 0.0
            col_metadata.statistics.std_dev = 0.0
            col_metadata.statistics.min = 0.0
            col_metadata.statistics.max = 0.0
            
            writer.add_column_metadata(col_metadata)

        # Read CSV in Batches
        # Use working_csv_path
        reader = pl.read_csv_batched(
            self.working_csv_path,
            batch_size=self.chunk_size,
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
            
            # Write chunk for EACH column
            for col_idx, col_name in enumerate(column_names):
                try:
                    if col_name in df_batch.columns:
                        series = df_batch.get_column(col_name)
                        data = series.to_numpy()
                        
                        # Handle NaN for C++
                        if data.dtype.kind in 'fi':
                             data = np.nan_to_num(data, nan=0.0)
                        
                        # Ensure float64
                        if data.dtype != np.float64:
                             data = data.astype(np.float64)

                        writer.write_column_chunk(
                            col_idx, chunk_id,
                            data.tobytes(), len(data) * 8, len(data)
                        )
                    else:
                        # Missing column (should not happen if logic is correct)
                        # Fill with zeros
                        zeros = np.zeros(current_batch_size, dtype=np.float64)
                        writer.write_column_chunk(
                            col_idx, chunk_id,
                            zeros.tobytes(), len(zeros) * 8, len(zeros)
                        )
                except Exception as e:
                    logger.error(f"Error writing chunk {chunk_id} col {col_name}: {e}")
            
            rows_processed += current_batch_size
            chunk_id += 1
            del df_batch
            
        # Write application state
        app_state = tgcpp.ApplicationState()
        writer.write_application_state(app_state)
        
        # Finalize
        writer.finalize()
        logger.info(f"MPAI file written: {self.mpai_path}")
        
        return rows_processed
    
    def _generate_lod_pyramid(self, schema: Dict[str, Any], row_count: int):
        """
        Generate LOD pyramid files for fast visualization at any zoom level.
        
        Creates lod1_100.parquet, lod2_10k.parquet, lod3_100k.parquet
        with pre-computed min/max values per bucket.
        """
        try:
            from src.data.lod_generator import LodGenerator
            
            # Skip for small files (LOD not needed)
            if row_count < 200:
                logger.info(f"[LOD] Skipping pyramid for small file ({row_count} rows)")
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
        Package the binary MPAI file and LOD files into a ZIP64 container.
        
        This creates a single-file project format that includes:
        - The raw MPAI binary data (as raw_data.parquet equivalent)
        - All LOD pyramid files
        - Manifest and metadata for project management
        
        After packaging, the original binary files are cleaned up.
        """
        logger.info("[CONTAINER] Packaging into ZIP64 container...")
        
        # Create temp directory for container assembly
        container_temp = tempfile.mkdtemp(prefix="mpai_container_")
        
        try:
            # Read the binary MPAI file into a parquet-compatible format
            # Since C++ MPAI is a custom binary format, we need to read it and 
            # convert to parquet for the container
            import time_graph_cpp as tgcpp
            reader = tgcpp.MpaiReader(self.mpai_path)
            
            # Read all data from the binary MPAI
            column_names = reader.get_column_names()
            row_count = reader.get_row_count()
            
            # Build columns dict for Polars DataFrame
            columns_data = {}
            for col_name in column_names:
                try:
                    col_data = reader.load_column(col_name)
                    columns_data[col_name] = col_data
                except Exception as e:
                    logger.warning(f"[CONTAINER] Failed to load column {col_name}: {e}")
                    continue
            
            # Create Polars DataFrame
            if columns_data:
                df = pl.DataFrame(columns_data)
                
                # Prepare metadata
                time_col = self.settings.get('time_column', 'time')
                if time_col not in df.columns and df.columns:
                    time_col = df.columns[0]
                
                metadata = ProjectMetadata(
                    time_column=time_col,
                    sampling_frequency=self.settings.get('sampling_frequency', 0.0),
                    columns=[
                        {"name": col, "dtype": str(df[col].dtype)}
                        for col in df.columns
                    ],
                )
                
                # Collect LOD files if they exist
                lod_files = {}
                lod_dir = os.path.splitext(self.mpai_path)[0] + '_lod'
                if os.path.isdir(lod_dir):
                    for lod_file in os.listdir(lod_dir):
                        if lod_file.endswith('.parquet'):
                            lod_path = os.path.join(lod_dir, lod_file)
                            with open(lod_path, 'rb') as f:
                                lod_files[lod_file] = f.read()
                    logger.info(f"[CONTAINER] Found {len(lod_files)} LOD files")
                
                # Create ZIP64 container
                manager = MpaiProjectManager()
                
                # Determine container path (same name, but will replace binary)
                container_path = self.mpai_path + ".container.tmp"
                
                handle = manager.create_project(
                    df, 
                    container_path,
                    metadata=metadata,
                    lod_files=lod_files if lod_files else None,
                )
                
                # Replace original binary with container
                original_binary = self.mpai_path
                backup_path = self.mpai_path + ".binary"
                
                # Keep backup of binary format (optional, remove in production)
                shutil.move(original_binary, backup_path)
                shutil.move(container_path, original_binary)
                
                # Cleanup LOD directory (now inside container)
                if os.path.isdir(lod_dir):
                    shutil.rmtree(lod_dir)
                
                # Cleanup binary backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                # Update metrics
                self.metrics['container_format'] = 'zip64'
                self.metrics['mpai_size_mb'] = os.path.getsize(self.mpai_path) / (1024 * 1024)
                
                logger.info(f"[CONTAINER] Created ZIP64 container: {self.mpai_path}")
                logger.info(f"[CONTAINER] Final size: {self.metrics['mpai_size_mb']:.2f} MB")
            else:
                logger.warning("[CONTAINER] No column data available, skipping container packaging")
                
        except ImportError:
            logger.warning("[CONTAINER] time_graph_cpp not available, skipping container packaging")
        except Exception as e:
            logger.error(f"[CONTAINER] Packaging failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup temp directory
            if os.path.exists(container_temp):
                shutil.rmtree(container_temp, ignore_errors=True)

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
