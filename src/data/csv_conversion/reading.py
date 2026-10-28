"""
Reading the CSV: quote unwrapping, encoding, read options, column names and chunking.

Part of CsvToMpaiConverter; moved from src/data/csv_to_mpai_converter.py without changes.
"""

import logging
import os
import psutil
import tempfile
from typing import Dict, Any
import polars as pl

logger = logging.getLogger(__name__)


class CsvReadingMixin:
    """Reading the CSV: quote unwrapping, encoding, read options, column names and chunking."""

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
