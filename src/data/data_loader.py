"""
Data Loader Worker for Time Graph Application

Handles threaded data loading via MPAI streaming.
Refactored to comply with ARROW_MIGRATION_ANALYSIS.md architecture.
"""

import logging
import os
import time
import hashlib
import json
import tempfile
from PyQt5.QtCore import QObject, pyqtSignal as Signal

from src.data.csv_to_mpai_converter import CsvToMpaiConverter
from src.data.excel_to_csv import is_excel_file, excel_to_temp_csv, remove_temp_csv
from src.data.data_reader import is_mpai_complete

logger = logging.getLogger(__name__)


class DataLoader(QObject):
    """
    Worker class for loading data in a separate thread.
    
    Architecture:
    - ALWAYS converts CSV to MPAI (if not exists/valid)
    - Loads data via C++ MpaiReader (Memory Mapped)
    - Returns MpaiReader object (not DataFrame)
    """
    
    finished = Signal(object, str)  # MpaiReader, time_column
    error = Signal(str)
    progress = Signal(str, int)  # message, percentage

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self._datetime_converted = False # Tracked during conversion now
        # Column -> [values stored as 0 because they were not numbers, non-empty values]
        self.non_numeric_report = {}
        self.converter = None # Active converter instance

    def cancel(self):
        """Cancel the current operation."""
        if self.converter:
            logger.info("Cancelling active converter...")
            self.converter.cancel()

    def run(self):
        """Start the data loading process."""
        try:
            reader = self._load_data()
            
            # Determine effective time column
            # If we created a custom time column during conversion, use that name
            if self.settings.get('create_custom_time', False):
                time_column = self.settings.get('new_time_column_name', 'time_generated')
            else:
                requested_time_col = self.settings.get('time_column', 'time')
                
                # Verify if this column actually exists in the reader
                # This handles case sensitivity issues (e.g. 'time' vs 'Time')
                actual_columns = reader.get_column_names()
                
                if requested_time_col in actual_columns:
                    time_column = requested_time_col
                else:
                    # Try case-insensitive match
                    found = False
                    for col in actual_columns:
                        if col.lower() == requested_time_col.lower():
                            time_column = col
                            found = True
                            logger.info(f"Resolved time column '{requested_time_col}' to '{col}'")
                            break
                    
                    if not found:
                        # Fallback to first column or default
                        logger.warning(f"Time column '{requested_time_col}' not found in file. Columns: {actual_columns[:5]}...")
                        time_column = requested_time_col # Pass it through, SignalProcessor handles missing
                
            self.finished.emit(reader, time_column)
            
        except FileNotFoundError as e:
            self.error.emit(f"Dosya bulunamadı: {str(e)}")
        except ValueError as e:
            self.error.emit(f"Veri hatası: {e}")
        except Exception as e:
            self.error.emit(f"Beklenmedik bir hata oluştu: {e}")
            logger.exception("Data loading error:")

    def _load_data(self):
        """Main data loading logic."""
        file_path = self.settings['file_path']
        file_ext = os.path.splitext(file_path)[1].lower()

        # 1. Direct MPAI Loading
        if file_ext == '.mpai':
            return self._load_mpai(file_path)

        # 2. CSV / delimited text Loading (Convert -> Load)
        if file_ext in ('.csv', '.txt'):
            return self._load_csv_as_mpai(file_path)

        # 3. NI TDM/TDX/TDMS Loading
        if file_ext in ('.tdm', '.tdx', '.tdms'):
            return self._load_ni_as_mpai(file_path)

        # 4. Excel: sheet -> temp CSV -> same CSV pipeline
        if is_excel_file(file_path):
            return self._load_excel_as_mpai(file_path)

        raise ValueError(f"Desteklenmeyen dosya formatı: {file_ext}")

    def _load_excel_as_mpai(self, file_path):
        """
        Load an Excel sheet through the CSV pipeline.

        The import dialog normally hands over the sheet already exported to a
        temp CSV (settings['_excel_csv_path']); it is exported here otherwise.
        The temp CSV is deleted once the MPAI exists.
        """
        csv_path = self.settings.get('_excel_csv_path')
        if not csv_path or not os.path.exists(csv_path):
            self.progress.emit("Excel sayfası okunuyor...", 2)
            csv_path = excel_to_temp_csv(file_path, self.settings.get('excel_sheet'))
        try:
            return self._load_csv_as_mpai(file_path, source_path=csv_path)
        finally:
            remove_temp_csv(csv_path)
            self.settings.pop('_excel_csv_path', None)

    def _load_csv_as_mpai(self, file_path, source_path=None):
        """
        Convert CSV to MPAI and load it. MPAI files are stored in temp directory.

        Args:
            file_path: Original file (names the cache entry, checked for changes)
            source_path: CSV actually converted, if different (Excel temp CSV)
        """
        source_path = source_path or file_path
        try:
            # === TEMP DIRECTORY SETUP ===
            # Use %LOCALAPPDATA%/TimeGraph/cache/ for temp MPAI files
            local_app_data = os.environ.get('LOCALAPPDATA', tempfile.gettempdir())
            temp_cache_dir = os.path.join(local_app_data, 'TimeGraph', 'cache')
            os.makedirs(temp_cache_dir, exist_ok=True)
            
            # Generate unique filename based on file path hash
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            mpai_path = os.path.join(temp_cache_dir, f"{file_name}_{file_hash}.mpai")
            settings_marker_path = os.path.join(temp_cache_dir, f"{file_name}_{file_hash}.mpai.settings")
            
            # Store temp paths in settings for cleanup tracking
            self.settings['_temp_mpai_path'] = mpai_path
            self.settings['_temp_settings_path'] = settings_marker_path
            self.settings['_is_temp_file'] = True
            
            logger.info(f"[TEMP] MPAI will be stored at: {mpai_path}")
            
            # Hash every import setting that affects the converted data, so
            # changing e.g. the delimiter or time column invalidates the cache
            cache_relevant = {
                k: v for k, v in self.settings.items()
                if not k.startswith('_') and k not in ('file_path', 'time_column_original')
            }
            # Conversion fixes must not be hidden by MPAIs cached by older code
            cache_relevant['_converter_version'] = CsvToMpaiConverter.CONVERTER_VERSION
            settings_key = hashlib.md5(
                json.dumps(cache_relevant, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check for existing valid cache
            should_regenerate = False
            if os.path.exists(mpai_path):
                # Simple check: if MPAI is newer than CSV, use it
                # Also verify the MPAI file is not corrupted by checking size
                if os.path.getmtime(mpai_path) > os.path.getmtime(file_path):
                    mpai_size = os.path.getsize(mpai_path) if os.path.isfile(mpai_path) else sum(
                        os.path.getsize(os.path.join(mpai_path, f)) 
                        for f in os.listdir(mpai_path) if os.path.isfile(os.path.join(mpai_path, f))
                    ) if os.path.isdir(mpai_path) else 0
                    csv_size = os.path.getsize(source_path)
                    # MPAI should be at least 5% of CSV size (compression)
                    # If too small, it's likely corrupted
                    if mpai_size > csv_size * 0.05 and not is_mpai_complete(mpai_path):
                        # A cleanup interrupted by a locked file leaves some
                        # channel files deleted; using it would plot empty data
                        logger.warning(f"Cached MPAI is incomplete, regenerating: {mpai_path}")
                        should_regenerate = True
                    elif mpai_size > csv_size * 0.05:
                        # Check if a settings marker file exists and matches current settings
                        # Marker (JSON): settings key + conversion results the
                        # UI needs again when loading from cache
                        marker = {}
                        if os.path.exists(settings_marker_path):
                            try:
                                with open(settings_marker_path, 'r', encoding='utf-8') as f:
                                    marker = json.load(f)
                            except Exception:
                                marker = {}  # old plain-text marker: regenerate
                        cached_settings = marker.get('key', '') if isinstance(marker, dict) else ''
                        
                        if cached_settings == settings_key:
                            self._datetime_converted = bool(marker.get('datetime', False))
                            self.non_numeric_report = marker.get('non_numeric', {})
                            logger.info(f"Using valid cached MPAI: {mpai_path} ({mpai_size/1024/1024:.1f} MB)")
                            self.progress.emit("Loading from cache...", 10)
                            return self._load_mpai(mpai_path)
                        else:
                            logger.info(f"Import settings changed ({cached_settings} -> {settings_key}), regenerating MPAI...")
                            should_regenerate = True
                    else:
                        logger.warning(f"Cached MPAI seems corrupted (too small), regenerating...")
                        should_regenerate = True
                else:
                    should_regenerate = True
                    
                if should_regenerate:
                    try:
                        import shutil
                        if os.path.isdir(mpai_path):
                            shutil.rmtree(mpai_path)
                        else:
                            os.remove(mpai_path)
                        if os.path.exists(settings_marker_path):
                            os.remove(settings_marker_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean old cache: {e}")
            
            # Perform Conversion
            logger.info(f"Converting CSV to MPAI: {file_path}")
            self.progress.emit("Converting CSV to MPAI for better performance", 0)
            
            def _progress_cb(msg: str, pct: int):
                # Relay conversion progress
                self.progress.emit(msg, pct)
            
            conversion_errors = []
            used_retry_path = False

            # Pass all settings (time creation, etc.) to converter
            # Use class directly to keep reference
            self.converter = CsvToMpaiConverter(
                source_path,
                mpai_path, 
                settings=self.settings
            )
            self.converter.progress.connect(_progress_cb)
            self.converter.error.connect(conversion_errors.append)
            
            # Run conversion
            success = self.converter.convert()
            self._datetime_converted = self.converter.time_is_datetime
            self.non_numeric_report = self.converter.non_numeric_report
            self.converter = None # Clear ref
            
            if not success:
                # If failed (likely due to file lock), try one more time with a unique suffix
                import time
                logger.warning("Conversion failed (likely locked). Retrying with unique path...")
                
                new_suffix = f"_{int(time.time())}"
                unique_mpai_path = mpai_path.replace('.mpai', f'{new_suffix}.mpai')
                
                # Update temp path in settings so cleaning works
                self.settings['_temp_mpai_path'] = unique_mpai_path
                
                self.progress.emit(f"Retrying with new cache path...", 5)
                
                self.converter = CsvToMpaiConverter(
                    source_path, 
                    unique_mpai_path, 
                    settings=self.settings
                )
                self.converter.progress.connect(_progress_cb)
                self.converter.error.connect(conversion_errors.append)
                
                success = self.converter.convert()
                self._datetime_converted = self.converter.time_is_datetime
                self.non_numeric_report = self.converter.non_numeric_report
                self.converter = None
                
                if success:
                    mpai_path = unique_mpai_path
                    used_retry_path = True
                else:
                    detail = conversion_errors[-1] if conversion_errors else "bilinmeyen hata"
                    raise ValueError(f"CSV dönüştürme başarısız: {detail}")

            # Save settings marker for cache validation
            try:
                # Ensure dir exists before writing marker
                marker_dir = os.path.dirname(settings_marker_path)
                if not os.path.exists(marker_dir):
                     os.makedirs(marker_dir, exist_ok=True)
                     
                if used_retry_path:
                    # Data is under a unique name; the regular cache path still
                    # holds old (possibly partial) data and must not validate
                    if os.path.exists(settings_marker_path):
                        os.remove(settings_marker_path)
                else:
                    with open(settings_marker_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'key': settings_key,
                            'datetime': self._datetime_converted,
                            'non_numeric': self.non_numeric_report,
                        }, f, ensure_ascii=False)
                    logger.info(f"Settings marker saved: {settings_key}")
            except Exception as e:
                logger.warning(f"Failed to save settings marker: {e}")
            
            self.progress.emit("Conversion complete, opening file...", 98)
            return self._load_mpai(mpai_path)
            
        except Exception as e:
            raise ValueError(f"CSV işlenemedi: {e}")

    def _load_ni_as_mpai(self, file_path: str):
        """Convert a NI TDM/TDX/TDMS file to MPAI directory and load it."""
        try:
            local_app_data = os.environ.get('LOCALAPPDATA', tempfile.gettempdir())
            temp_cache_dir = os.path.join(local_app_data, 'TimeGraph', 'cache')
            os.makedirs(temp_cache_dir, exist_ok=True)

            file_name  = os.path.splitext(os.path.basename(file_path))[0]
            file_hash  = hashlib.md5(file_path.encode()).hexdigest()[:8]
            mpai_path  = os.path.join(temp_cache_dir, f"{file_name}_{file_hash}.mpai")

            self.settings['_temp_mpai_path'] = mpai_path
            self.settings['_is_temp_file']   = True

            # Use valid cache when source file hasn't changed
            if os.path.exists(mpai_path):
                mpai_mtime = (
                    max(os.path.getmtime(os.path.join(mpai_path, f))
                        for f in os.listdir(mpai_path)
                        if os.path.isfile(os.path.join(mpai_path, f)))
                    if os.path.isdir(mpai_path) else
                    os.path.getmtime(mpai_path)
                )
                if mpai_mtime > os.path.getmtime(file_path) and is_mpai_complete(mpai_path):
                    logger.info("NI önbellek kullanılıyor: %s", mpai_path)
                    self.progress.emit("Önbellekten yükleniyor...", 10)
                    return self._load_mpai(mpai_path)
                import shutil
                try:
                    shutil.rmtree(mpai_path) if os.path.isdir(mpai_path) else os.remove(mpai_path)
                except Exception:
                    pass

            self.progress.emit("NI formatı MPAI'ye dönüştürülüyor...", 0)

            def _progress_cb(msg: str, pct: int):
                self.progress.emit(msg, pct)

            from src.data.ni_to_mpai_converter import NiToMpaiConverter
            self.converter = NiToMpaiConverter(file_path, mpai_path, settings=self.settings)
            self.converter.progress.connect(_progress_cb)
            success = self.converter.convert()
            self.converter = None

            if not success:
                raise ValueError("NI format dönüşümü başarısız oldu")

            self.progress.emit("Dönüşüm tamam, dosya açılıyor...", 98)
            return self._load_mpai(mpai_path)

        except Exception as exc:
            raise ValueError(f"NI dosyası işlenemedi: {exc}") from exc

    def _load_mpai(self, file_path):
        """Load MPAI file using appropriate reader (Directory-based or C++ Legacy)."""
        try:
            # Check if it's the new Directory-based MPAI
            if os.path.isdir(file_path):
                logger.info(f"Detected Directory-based MPAI: {file_path}")
                from src.data.data_reader import MpaiDirectoryReader
                reader = MpaiDirectoryReader(file_path)
                logger.info(f"MpaiDirectoryReader initialized: {file_path} ({reader.get_row_count()} rows)")
                return reader

            raise ValueError(
                f"Bu MPAI dosyası dizin formatında değil. "
                f"Lütfen dosyayı yeniden CSV'den içe aktarın: {file_path}"
            )
        except Exception as e:
            raise ValueError(f"MPAI okuma hatası: {e}")
