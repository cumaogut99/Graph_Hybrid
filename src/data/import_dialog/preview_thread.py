"""
Background thread that reads the preview rows of the file.

Moved from src/data/data_import_dialog.py without changes.
"""

import os
import io
import logging
import polars as pl

from PyQt5.QtCore import QThread, pyqtSignal as Signal

logger = logging.getLogger(__name__)


class DataPreviewThread(QThread):
    """Veri önizlemesi için background thread."""
    
    preview_ready = Signal(object, dict)  # DataFrame ve metadata
    error_occurred = Signal(str)
    
    def __init__(self, file_path: str, encoding: str = 'latin-1', delimiter: str = ',',
                 header_row: int = 0, start_row: int = 0, decimal_comma: bool = False):
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter
        self.header_row = header_row
        self.start_row = start_row
        self.decimal_comma = decimal_comma
        
        # Calculate skip_rows for Polars
        # Polars'ın skip_rows parametresi: dosyanın başından kaç satır atlanacak
        # has_header: header satırının olup olmadığı
        
        if header_row >= 0:
            # Header var
            self.has_header = True
            # Header satırına kadar atla (header satırı kendisi atlanmaz, okunur)
            self.skip_rows_before_header = header_row
            
            # Header'dan sonra veri başlangıcına kadar atlanacak satır sayısı
            if start_row > header_row:
                self.skip_rows_after_header = start_row - header_row - 1
            else:
                self.skip_rows_after_header = 0
        else:
            # Header yok
            self.has_header = False
            self.skip_rows_before_header = start_row
            self.skip_rows_after_header = 0
        
    def run(self):
        """Veri önizlemesi yükle."""
        try:
            # Dosya uzantısına göre okuma yöntemi belirle
            file_ext = os.path.splitext(self.file_path)[1].lower()

            if file_ext == '.csv':
                # Always use manual line reading to properly handle header_row and start_row
                df = self._read_csv_with_manual_skip()
            elif file_ext == '.parquet':
                # Parquet dosyası
                df = pl.read_parquet(self.file_path)
                df = df.head(100)  # İlk 100 satır
            elif file_ext == '.mpai':
                # MPAI dosyası — Python reader
                from src.data.data_reader import MpaiDirectoryReader
                reader = MpaiDirectoryReader(self.file_path)
                count = reader.get_row_count()
                preview_count = min(count, 100)
                cols = reader.get_column_names()
                data = {}
                for col in cols:
                    try:
                        data[col] = reader.load_column_slice(col, 0, preview_count)
                    except Exception as e:
                        logger.warning(f"Preview column '{col}' failed: {e}")
                        data[col] = [None] * preview_count
                df = pl.DataFrame(data)
            elif file_ext in ('.tdm', '.tdx'):
                df = self._read_ni_preview_tdm()
            elif file_ext == '.tdms':
                df = self._read_ni_preview_tdms()
            else:
                # Genel okuma denemesi - CSV gibi işle
                df = self._read_csv_with_manual_skip()
            
            # Metadata oluştur
            metadata = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                'file_size': os.path.getsize(self.file_path),
                'encoding': self.encoding,
                'delimiter': self.delimiter
            }
            
            self.preview_ready.emit(df, metadata)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _needs_line_quote_stripping(self):
        """Check if CSV file has entire lines wrapped in quotes."""
        try:
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                lines_checked = 0
                quote_wrapped_count = 0
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if (line.startswith('"') and line.endswith('"') and 
                        self.delimiter in line and line.count(self.delimiter) > 0):
                        quote_wrapped_count += 1
                    
                    lines_checked += 1
                    if lines_checked >= 5:
                        break
                
                return (quote_wrapped_count / max(lines_checked, 1)) > 0.5
                
        except Exception:
            return False
    
    def _read_csv_with_manual_skip(self):
        """
        Read CSV with manual line selection to properly handle header_row and start_row.
        This handles:
        - Quote-wrapped lines
        - Custom header row position
        - Custom data start row position
        """
        try:
            # Read all necessary lines from file
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                all_lines = []
                for i, line in enumerate(f):
                    line = line.rstrip('\r\n')
                    
                    # Remove wrapping quotes if present
                    if line.startswith('"') and line.endswith('"') and len(line) > 1:
                        line = line[1:-1]
                    
                    all_lines.append(line)
                    
                    # Read enough lines: start_row + 100 data rows
                    if i >= self.start_row + 100:
                        break
            
            # Select lines based on header_row and start_row
            selected_lines = []
            
            if self.has_header:
                # Add header line
                if self.header_row < len(all_lines):
                    selected_lines.append(all_lines[self.header_row])
                else:
                    raise ValueError(f"Header satırı ({self.header_row}) dosya boyutunu aşıyor")
                
                # Add data lines starting from start_row
                data_count = 0
                for i in range(self.start_row, len(all_lines)):
                    selected_lines.append(all_lines[i])
                    data_count += 1
                    if data_count >= 100:  # Preview için 100 satır yeterli
                        break
            else:
                # No header: just take lines from start_row onwards
                for i in range(self.start_row, min(self.start_row + 100, len(all_lines))):
                    selected_lines.append(all_lines[i])
            
            # Join and parse through StringIO
            cleaned_content = '\n'.join(selected_lines)

            # truncate_ragged_lines: while the user is still adjusting the
            # header/start rows, metadata lines may have extra fields; show a
            # preview for the chosen settings instead of failing.
            df = pl.read_csv(
                io.StringIO(cleaned_content),
                separator=self.delimiter,
                has_header=self.has_header,
                skip_rows=0,  # Already manually selected lines
                low_memory=False,
                infer_schema_length=100,
                ignore_errors=True,
                truncate_ragged_lines=True,
                decimal_comma=self.decimal_comma,
                try_parse_dates=False
            )
            # Blank lines come through as all-null rows
            if df.width > 0:
                df = df.filter(~pl.all_horizontal(pl.all().is_null()))

            logger.info(f"Preview with manual skip (header={self.header_row}, start={self.start_row}): {df.shape}")
            return df

        except Exception as e:
            # No fallback that re-reads from line 0: it would show a preview
            # that ignores the chosen header/start rows
            logger.error(f"Manual CSV read failed: {e}")
            raise

    def _read_ni_preview_tdm(self):
        """Load first 100 rows from a TDM/TDX file pair."""
        from src.data.tdm_reader import TdmReader
        reader = TdmReader(self.file_path)
        names  = reader.get_channel_names()
        n_rows = min(reader.get_row_count(), 100)
        data   = {}
        for name in names:
            try:
                arr = reader.read_channel(name, start=0, count=n_rows)
                data[name] = arr.tolist()
            except Exception as exc:
                logger.warning(f"TDM preview channel '{name}' failed: {exc}")
                data[name] = [None] * n_rows
        return pl.DataFrame(data)

    def _read_ni_preview_tdms(self):
        """Load first 100 rows from a TDMS file."""
        from src.data.tdms_reader import TdmsReader
        if not TdmsReader.is_available():
            raise ImportError(
                "TDMS önizleme için nptdms gereklidir.\n"
                "Kurulum: pip install nptdms"
            )
        reader = TdmsReader(self.file_path)
        names  = reader.get_channel_names()
        n_rows = min(reader.get_row_count(), 100)
        data   = {}
        for name in names:
            try:
                arr = reader.read_channel(name, start=0, count=n_rows)
                data[name] = arr.tolist()
            except Exception as exc:
                logger.warning(f"TDMS preview channel '{name}' failed: {exc}")
                data[name] = [None] * n_rows
        return pl.DataFrame(data)
