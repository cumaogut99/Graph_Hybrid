"""
Data Loader Worker for Time Graph Application

Handles threaded data loading via MPAI streaming.
Refactored to comply with ARROW_MIGRATION_ANALYSIS.md architecture.
"""

import logging
import os
import time
from PyQt5.QtCore import QObject, pyqtSignal as Signal

from src.data.csv_to_mpai_converter import convert_csv_to_mpai

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

    def run(self):
        """Start the data loading process."""
        try:
            reader = self._load_data()
            
            # Determine effective time column
            # If we created a custom time column during conversion, use that name
            if self.settings.get('create_custom_time', False):
                time_column = self.settings.get('new_time_column_name', 'time_generated')
            else:
                time_column = self.settings.get('time_column', 'time')
                
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
        """Convert CSV to MPAI and load it."""
        try:
            base, _ = os.path.splitext(file_path)
            mpai_path = base + '.mpai'
            
            # Check for existing valid cache
            if os.path.exists(mpai_path):
                # Simple check: if MPAI is newer than CSV, use it
                # Also verify the MPAI file is not corrupted by checking size
                if os.path.getmtime(mpai_path) > os.path.getmtime(file_path):
                    mpai_size = os.path.getsize(mpai_path)
                    csv_size = os.path.getsize(file_path)
                    # MPAI should be at least 10% of CSV size (compression)
                    # If too small, it's likely corrupted
                    if mpai_size > csv_size * 0.05:
                        logger.info(f"Using valid cached MPAI: {mpai_path} ({mpai_size/1024/1024:.1f} MB)")
                        self.progress.emit("Önbellek yükleniyor...", 10)
                        return self._load_mpai(mpai_path)
                    else:
                        logger.warning(f"Cached MPAI seems corrupted (too small), regenerating...")
                        os.remove(mpai_path)
            
            # Perform Conversion
            logger.info(f"Converting CSV to MPAI: {file_path}")
            self.progress.emit("Veri optimize ediliyor (MPAI Dönüşümü)...", 0)
            
            def _progress_cb(msg: str, pct: int):
                # Relay conversion progress
                self.progress.emit(msg, pct)
            
            # Pass all settings (time creation, etc.) to converter
            convert_csv_to_mpai(
                file_path, 
                mpai_path, 
                progress_callback=_progress_cb,
                settings=self.settings
            )
            
            self.progress.emit("Dönüşüm tamamlandı, dosya açılıyor...", 98)
            return self._load_mpai(mpai_path)
            
        except Exception as e:
            raise ValueError(f"CSV işlenemedi: {e}")

    def _load_mpai(self, file_path):
        """Load MPAI file using C++ reader."""
        try:
            # Import C++ module
            import sys
            # Attempt to find DLL
            dll_paths = [os.getcwd(), os.path.join(os.getcwd(), "build", "Release")]
            for p in dll_paths:
                if p not in sys.path: sys.path.append(p)
                
            import time_graph_cpp
            
            reader = time_graph_cpp.MpaiReader(file_path)
            logger.info(f"MPAI loaded: {file_path} ({reader.get_row_count()} rows)")
            return reader
            
        except ImportError:
            raise ValueError("C++ motoru (time_graph_cpp) yüklenemedi. DLL eksik olabilir.")
        except Exception as e:
            raise ValueError(f"MPAI okuma hatası: {e}")
