"""
Writing the MPAI file: streaming the batches, column statistics, sample rate, LOD pyramid and container.

Part of CsvToMpaiConverter; moved from src/data/csv_to_mpai_converter.py without changes.
"""

import logging
import os
from typing import Dict, Any
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class MpaiWritingMixin:
    """Writing the MPAI file: streaming the batches, column statistics, sample rate, LOD pyramid and container."""

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
        # First and last valid time value with their row numbers: the real
        # sampling rate over the whole recording (see _correct_sample_rate)
        time_col = self.settings.get('time_column')
        time_first = time_last = None
        
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

                    if col_name == time_col and data.dtype.kind in 'fi':
                        valid = np.flatnonzero(np.isfinite(data))
                        if len(valid):
                            if time_first is None:
                                time_first = (rows_processed + valid[0], float(data[valid[0]]))
                            time_last = (rows_processed + valid[-1], float(data[valid[-1]]))

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
        if time_first and time_last:
            self._correct_sample_rate(writer, sampling_freq, time_first, time_last)
        logger.info(f"MPAI directory written: {self.mpai_path}")
        
        return rows_processed

    def _correct_sample_rate(self, writer, sampling_freq: float, first: tuple, last: tuple):
        """
        The rate detected from the first rows can be wrong: with a time column
        of whole seconds and several rows per second (09:38:15, 09:38:15, ...)
        the median step is 0 and the default 1000 Hz stays. The reader builds
        the time axis as T0 + i * dt, so 'View All' then showed only a few
        seconds. Use the rate over the whole recording instead.
        """
        (i_first, t_first), (i_last, t_last) = first, last
        if i_last <= i_first or t_last <= t_first:
            return
        real_freq = (i_last - i_first) / (t_last - t_first)
        if abs(real_freq - sampling_freq) <= sampling_freq * 0.01:
            return
        start_time = t_first - i_first / real_freq
        logger.info(f"[CSV AUTO-DETECT] Fs corrected from whole recording: "
                    f"{sampling_freq:.4f} Hz -> {real_freq:.4f} Hz")
        writer.update_timing(real_freq, start_time)
        self.settings['sampling_frequency'] = real_freq

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
