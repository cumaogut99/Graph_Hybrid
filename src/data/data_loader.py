"""
Data Loader Worker for Time Graph Application

Handles threaded data loading via MPAI streaming.
Refactored to comply with ARROW_MIGRATION_ANALYSIS.md architecture.
"""

import logging
import os
import time
import hashlib
import tempfile
from PyQt5.QtCore import QObject, pyqtSignal as Signal

from src.data.csv_to_mpai_converter import CsvToMpaiConverter

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

        # 2. CSV Loading (Convert -> Load)
        if file_ext == '.csv':
            return self._load_csv_as_mpai(file_path)
            
        # 3. Excel (Not Supported in Streaming Architecture)
        if file_ext in ['.xlsx', '.xls']:
            raise ValueError("Excel dosyaları performans mimarisinde desteklenmemektedir. Lütfen CSV'ye çevirin.")
            
        raise ValueError(f"Desteklenmeyen dosya formatı: {file_ext}")

    def _load_csv_as_mpai(self, file_path):
        """Convert CSV to MPAI and load it. MPAI files are stored in temp directory."""
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
            
            # Generate a simple hash based on import settings to invalidate cache when settings change
            header_row = self.settings.get('header_row')
            start_row = self.settings.get('start_row', 0)
            settings_key = f"h{header_row}_s{start_row}"
            
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
                    csv_size = os.path.getsize(file_path)
                    # MPAI should be at least 5% of CSV size (compression)
                    # If too small, it's likely corrupted
                    if mpai_size > csv_size * 0.05:
                        # Check if a settings marker file exists and matches current settings
                        cached_settings = ""
                        if os.path.exists(settings_marker_path):
                            try:
                                with open(settings_marker_path, 'r') as f:
                                    cached_settings = f.read().strip()
                            except:
                                pass
                        
                        if cached_settings == settings_key:
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
            
            # Pass all settings (time creation, etc.) to converter
            # Use class directly to keep reference
            self.converter = CsvToMpaiConverter(
                file_path, 
                mpai_path, 
                settings=self.settings
            )
            self.converter.progress.connect(_progress_cb)
            
            # Run conversion
            success = self.converter.convert()
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
                    file_path, 
                    unique_mpai_path, 
                    settings=self.settings
                )
                self.converter.progress.connect(_progress_cb)
                
                success = self.converter.convert()
                self.converter = None
                
                if success:
                    mpai_path = unique_mpai_path
                    # Update settings key marker path too?
                    # Ideally yes, but main concern is reading the data
                else:
                    raise ValueError("Conversion failed after retry")

            # Save settings marker for cache validation
            try:
                # Ensure dir exists before writing marker
                marker_dir = os.path.dirname(settings_marker_path)
                if not os.path.exists(marker_dir):
                     os.makedirs(marker_dir, exist_ok=True)
                     
                with open(settings_marker_path, 'w') as f:
                    f.write(settings_key)
                logger.info(f"Settings marker saved: {settings_key}")
            except Exception as e:
                logger.warning(f"Failed to save settings marker: {e}")
            
            self.progress.emit("Conversion complete, opening file...", 98)
            return self._load_mpai(mpai_path)
            
        except Exception as e:
            raise ValueError(f"CSV işlenemedi: {e}")

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

            # Fallback to C++ Legacy Reader (ZIP64) for files
            # Import C++ module
            import sys
            # Attempt to find DLL
            dll_paths = [os.getcwd(), os.path.join(os.getcwd(), "build", "Release")]
            for p in dll_paths:
                if p not in sys.path: sys.path.append(p)
                
            import time_graph_cpp
            
            reader = time_graph_cpp.MpaiReader(file_path)
            logger.info(f"Legacy MPAI loaded: {file_path} ({reader.get_row_count()} rows)")
            return reader
            
        except ImportError as e:
            if "time_graph_cpp" in str(e):
                 raise ValueError("Eski format için C++ motoru (time_graph_cpp) yüklenemedi. DLL eksik olabilir.")
            raise ValueError(f"MPAI okuyucu yüklenemedi: {e}")
        except Exception as e:
            raise ValueError(f"MPAI okuma hatası: {e}")
