"""
Data Operations Module

Veri yükleme ve kaydetme işlemlerini yöneten fonksiyonlar.

Bu modül şunları içerir:
- Dosya yükleme dialog'u ve işlemleri
- Veri kaydetme (CSV, Excel)
- Veri kalite kontrolü ve raporlama

Refactored from: app.py (TimeGraphApp sınıfı)
"""

import logging
import os
from typing import Optional, Dict, Any
import polars as pl
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from PyQt5.QtCore import QThread, pyqtSignal as Signal

# Import DataImportDialog ve DataLoader
from src.data.data_import_dialog import DataImportDialog
from src.data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataOperations:
    """
    Veri yükleme ve kaydetme işlemlerini yöneten sınıf.
    TimeGraphApp'den ayrıştırılmış, bağımsız çalışabilen modül.
    """
    
    def __init__(self, main_window):
        """
        Args:
            main_window: TimeGraphApp ana pencere referansı
        """
        self.main_window = main_window
        self.load_threads = []
        self.save_thread = None
        self.load_worker = None
        
    def open_file_dialog(self):
        """
        Dosya açma dialog'unu göster ve veri yükleme işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_file_open()
        """
        logger.info("Dosya açma işlemi başlatıldı")
        
        # Desteklenen dosya formatları
        file_filter = (
            "Veri Dosyaları (*.csv *.xlsx *.xls);;",
            "CSV Dosyaları (*.csv);;",
            "Excel Dosyaları (*.xlsx *.xls);;",
            "Tüm Dosyalar (*.*)"
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Veri Dosyası Seç",
            "",
            "".join(file_filter)
        )
        
        if file_path:
            # Dosya boyutunu kontrol et (bilgilendirme amaçlı)
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                # 25 MB limiti kaldırıldı - C++ engine büyük dosyaları destekliyor
                if file_size_mb > 5000:  # 5 GB üzeri için uyarı
                    reply = QMessageBox.question(
                        self.main_window,
                        "Çok Büyük Dosya",
                        f"Seçilen dosya çok büyük ({file_size_mb:.1f} MB / {file_size_mb/1024:.1f} GB).\n\n"
                        f"Yükleme uzun sürebilir ve çok fazla RAM kullanabilir.\n"
                        f"Devam etmek istiyor musunuz?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return False
                elif file_size_mb > 1000:  # 1 GB üzeri için bilgilendirme
                    QMessageBox.information(
                        self.main_window,
                        "Büyük Dosya",
                        f"Büyük dosya yükleniyor ({file_size_mb:.1f} MB).\n\n"
                        f"C++ hızlandırma motoru kullanılacak.\n"
                        f"Yükleme birkaç dakika sürebilir."
                    )
            except Exception as e:
                logger.error(f"Dosya boyutu kontrol edilirken hata: {e}")
                QMessageBox.critical(
                    self.main_window,
                    "Dosya Hatası",
                    f"Dosya bilgileri alınamadı:\n{str(e)}"
                )
                return False
            
            # Gelişmiş import dialog'unu aç
            import_dialog = DataImportDialog(file_path, self.main_window)
            if import_dialog.exec_() == QDialog.Accepted:
                # Import ayarlarını al
                settings = import_dialog.get_import_settings()
                self.load_data_with_settings(settings)
                return True
            else:
                logger.info("Dosya import işlemi iptal edildi")
                return False
        
        return False
    
    def load_data_with_settings(self, settings: Dict[str, Any]):
        """
        Verilen ayarlarla veri yükleme işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._load_data_with_settings()
        
        Args:
            settings: Import dialog'undan gelen ayarlar
        """
        try:
            file_path = settings['file_path']
            filename = os.path.basename(file_path)
            
            # Start loading operation
            if hasattr(self.main_window, 'loading_manager'):
                self.main_window.loading_manager.start_operation("file_loading", f"Loading {filename}...")
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.set_operation("File Loading", 0)
            
            # === MULTI-FILE CHECK ===
            file_manager = self.main_window.file_manager
            
            # Check if file already open
            existing_index = file_manager.is_file_already_open(file_path)
            if existing_index >= 0:
                file_manager.file_tab_widget.setCurrentIndex(existing_index)
                if hasattr(self.main_window, 'loading_manager'):
                    self.main_window.loading_manager.finish_operation("file_loading")
                QMessageBox.information(
                    self.main_window,
                    "Dosya Zaten Açık",
                    f"'{filename}' dosyası zaten açık.\nİlgili sekmeye geçildi."
                )
                return
            
            # Check file limit
            if not file_manager.can_add_file():
                current_count = len(file_manager.loaded_files)
                logger.warning(f"Cannot add file: {current_count}/{file_manager.max_files} files already loaded")
                QMessageBox.warning(
                    self.main_window,
                    "Maksimum Dosya Sayısı",
                    f"Maksimum {file_manager.max_files} dosya açık olabilir.\n"
                    f"Şu anda {current_count} dosya açık.\n"
                    f"Yeni dosya yüklemek için önce bir dosyayı kapatın."
                )
                if hasattr(self.main_window, 'loading_manager'):
                    self.main_window.loading_manager.finish_operation("file_loading")
                return
            
            # Dosya boyutunu kontrol et
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            # 25 MB limiti kaldırıldı - C++ engine büyük dosyaları destekliyor
            if file_size_mb > 1024:  # 1 GB üzeri
                if hasattr(self.main_window, 'loading_manager'):
                    self.main_window.loading_manager.update_operation("file_loading", f"Very large file: {filename} ({file_size_mb:.1f} MB) - C++ engine")
            elif file_size_mb > 100:
                if hasattr(self.main_window, 'loading_manager'):
                    self.main_window.loading_manager.update_operation("file_loading", f"Large file: {filename} ({file_size_mb:.1f} MB) - C++ accelerated")
            else:
                if hasattr(self.main_window, 'loading_manager'):
                    self.main_window.loading_manager.update_operation("file_loading", f"Loading: {filename}")
            
            # === THREAD YÖNETİMİ ===
            # Yeni thread oluştur
            load_thread = QThread()
            load_worker = DataLoader(settings)
            load_worker.moveToThread(load_thread)

            # Connect signals - main_window'daki callback'leri kullan
            load_thread.started.connect(load_worker.run)
            load_worker.finished.connect(self.main_window.on_loading_finished)
            load_worker.error.connect(self.main_window.on_loading_error)
            
            if hasattr(self.main_window, 'loading_manager'):
                load_worker.progress.connect(
                    lambda msg: self.main_window.loading_manager.update_operation("file_loading", msg)
                )
            
            # Cleanup
            load_worker.finished.connect(load_thread.quit)
            load_worker.error.connect(load_thread.quit)
            load_worker.finished.connect(load_worker.deleteLater)
            load_thread.finished.connect(load_thread.deleteLater)
            
            # Thread'i listede tut
            self.load_threads.append((load_thread, load_worker))
            self.load_worker = load_worker
            self.main_window.load_worker = load_worker  # Main window için referans

            # Start the thread
            load_thread.start()
            logger.debug(f"Started loading thread for {filename}")

        except Exception as e:
            error_msg = f"Dosya yükleme başlatılırken hata oluştu: {str(e)}"
            logger.error(error_msg)
            if hasattr(self.main_window, 'loading_manager'):
                self.main_window.loading_manager.finish_operation("file_loading")
            QMessageBox.critical(self.main_window, "Dosya Yükleme Hatası", error_msg)
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Dosya yükleme başarısız", 5000)
    
    def show_data_quality_summary(self, df: pl.DataFrame, filename: str = ""):
        """
        Yüklenen verinin kalite özetini göster.
        
        Refactored from: app.py -> TimeGraphApp._show_data_quality_summary()
        
        Args:
            df: Polars DataFrame or MpaiReader
            filename: Dosya adı
        """
        try:
            # Check for MpaiReader
            if hasattr(df, 'get_header'):
                logger.info(f"[QUALITY REPORT] MPAI Reader loaded: {filename}")
                logger.info(f"[QUALITY REPORT] {df.get_row_count():,} rows, {df.get_column_count()} columns")
                logger.info("[QUALITY REPORT] Skipping detailed quality check for large file performance.")
                return

            # Hızlı kalite kontrolü
            total_cols = len(df.columns)
            total_rows = df.height
            
            # NULL oranlarını hesapla
            high_null_cols = []
            for col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    null_pct = (null_count / total_rows) * 100
                    if null_pct > 20:
                        high_null_cols.append((str(col), null_pct, null_count))
            
            # Log raporu
            logger.info(f"[QUALITY REPORT] Veri Kalite Raporu - {filename}")
            logger.info(f"[QUALITY REPORT] Toplam: {total_rows} satir, {total_cols} kolon")
            
            if high_null_cols:
                logger.info(f"[QUALITY REPORT] Yuksek NULL orani olan kolonlar: {len(high_null_cols)}")
                for col, pct, count in high_null_cols[:5]:
                    logger.info(f"[QUALITY REPORT] - '{col}': {count} NULL (%{pct:.1f}) - otomatik duzeltildi")
            else:
                logger.info(f"[QUALITY REPORT] Tum kolonlar temiz (dusuk NULL orani)")
                
            logger.info(f"[QUALITY REPORT] Veri kullanima hazir!")
            
        except Exception as e:
            logger.debug(f"Data quality summary failed: {e}")
    
    def cleanup_threads(self):
        """Thread'leri güvenli şekilde temizle."""
        logger.info("Cleaning up data operations threads...")
        for thread, worker in self.load_threads:
            if thread:
                try:
                    if thread.isRunning():
                        thread.quit()
                        if not thread.wait(2000):
                            logger.warning(f"Thread did not finish, terminating...")
                            thread.terminate()
                            thread.wait(1000)
                except RuntimeError as e:
                    logger.debug(f"Thread already deleted: {e}")

