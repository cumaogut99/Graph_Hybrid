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
import hashlib
import tempfile
from typing import Optional, Dict, Any
import polars as pl
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from PyQt5.QtCore import QThread, pyqtSignal as Signal

# Import DataImportDialog ve DataLoader
from src.data.data_import_dialog import DataImportDialog
from src.data.data_loader import DataLoader
from src.data.scientific_data_importer import (
    is_scientific_file,
    ScientificFileConverter,
)

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
        self.is_loading_cancelled = False  # Flag to prevent callbacks after cancel
        # TDM/TDX/TDMS -> ara CSV dönüştürme için thread takibi
        self.convert_thread = None
        self.convert_worker = None
        
        # Connect to loading manager cancel signal
        if hasattr(main_window, 'loading_manager'):
            main_window.loading_manager.cancel_requested.connect(self._on_cancel_loading)
        
    def open_file_dialog(self):
        """
        Dosya açma dialog'unu göster ve veri yükleme işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_file_open()
        """
        logger.info("Dosya açma işlemi başlatıldı")

        # Desteklenen dosya formatları
        file_filter = (
            "Veri Dosyaları (*.csv *.xlsx *.xls *.tdms *.tdm *.tdx);;",
            "CSV Dosyaları (*.csv);;",
            "Excel Dosyaları (*.xlsx *.xls);;",
            "TDMS Dosyaları (*.tdms);;",
            "TDM/TDX Dosyaları (*.tdm *.tdx);;",
            "Tüm Dosyalar (*.*)"
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Veri Dosyası Seç",
            "",
            "".join(file_filter)
        )

        if not file_path:
            return False

        # TDM/TDX/TDMS dosyaları önce bir ara CSV'ye dönüştürülür; ardından
        # CSV ile aynı import dialog'u + MPAI boru hattı kullanılır.
        if is_scientific_file(file_path):
            self._start_scientific_conversion(file_path)
            return True

        # CSV / Excel: doğrudan import dialog'unu aç
        return self._open_import_dialog(file_path)

    def _open_import_dialog(self, file_path: str, original_path: Optional[str] = None) -> bool:
        """
        Import dialog'unu aç ve onaylanırsa yüklemeyi başlat.

        Args:
            file_path: Önizleme/okuma için kullanılacak yol (CSV).
            original_path: Kullanıcının seçtiği asıl dosya (TDM/TDX/TDMS ise),
                           başlıkta gösterilir ve önbellek anahtarı olarak kullanılır.
        """
        import_dialog = DataImportDialog(file_path, self.main_window)
        if original_path:
            # Başlıkta asıl dosya adını göster (ara CSV adı yerine) ve ara CSV
            # UTF-8 yazıldığı için encoding'i UTF-8'e ayarla.
            try:
                import_dialog.setWindowTitle(f"Veri Import - {os.path.basename(original_path)}")
                if hasattr(import_dialog, 'file_info_label'):
                    import_dialog.file_info_label.setText(f"📁 {os.path.basename(original_path)}")
                if hasattr(import_dialog, 'encoding_combo'):
                    import_dialog.encoding_combo.setCurrentText('utf-8')
            except Exception:
                pass

        if import_dialog.exec_() == QDialog.Accepted:
            settings = import_dialog.get_import_settings()
            if original_path:
                # Yükleme tamamlandığında silinmesi için ara CSV'yi işaretle
                settings['_intermediate_source_csv'] = file_path
                settings['_original_source_path'] = original_path
            self.load_data_with_settings(settings)
            return True

        logger.info("Dosya import işlemi iptal edildi")
        # İptal edilirse ara CSV'yi temizle
        if original_path:
            self._cleanup_intermediate_csv(file_path)
        return False

    def _start_scientific_conversion(self, file_path: str):
        """TDM/TDX/TDMS dosyasını ara CSV'ye dönüştürmeyi (overlay ile) başlat."""
        filename = os.path.basename(file_path)
        logger.info(f"[SCIENTIFIC] Converting {filename} to intermediate CSV...")

        # Ara CSV için benzersiz bir geçici yol oluştur
        local_app_data = os.environ.get('LOCALAPPDATA', tempfile.gettempdir())
        temp_cache_dir = os.path.join(local_app_data, 'TimeGraph', 'cache')
        os.makedirs(temp_cache_dir, exist_ok=True)
        stem = os.path.splitext(filename)[0]
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        csv_path = os.path.join(temp_cache_dir, f"{stem}_{file_hash}.import.csv")

        # İlerleme overlay'ini başlat
        self.is_loading_cancelled = False
        if hasattr(self.main_window, 'loading_manager'):
            self.main_window.loading_manager.start_operation(
                "file_loading",
                f"{filename} dönüştürülüyor...",
                subtitle="TDM/TDX/TDMS verisi içe aktarmaya hazırlanıyor",
            )

        # Dönüştürmeyi thread'de çalıştır
        thread = QThread()
        worker = ScientificFileConverter(file_path, csv_path)
        worker.moveToThread(thread)
        self.convert_thread = thread
        self.convert_worker = worker
        # İptal mekanizmasının erişebilmesi için aktif worker olarak işaretle
        self.load_worker = worker

        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda msg, pct: self._on_convert_progress(msg, pct)
        )
        worker.finished.connect(lambda out: self._on_scientific_converted(out, file_path))
        worker.error.connect(lambda err: self._on_scientific_error(err, filename))

        # Temizlik
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.start()

    def _on_convert_progress(self, msg: str, pct: int):
        if self.is_loading_cancelled:
            return
        if hasattr(self.main_window, 'loading_manager'):
            self.main_window.loading_manager.update_operation("file_loading", msg, pct)

    def _on_scientific_converted(self, csv_path: str, original_path: str):
        """Ara CSV hazır: overlay'i kapat ve import dialog'unu aç."""
        self.convert_worker = None
        self.convert_thread = None
        self.load_worker = None
        if self.is_loading_cancelled:
            self._cleanup_intermediate_csv(csv_path)
            return
        if hasattr(self.main_window, 'loading_manager'):
            self.main_window.loading_manager.finish_operation("file_loading")
        # Import dialog'unu asıl dosya adıyla aç
        self._open_import_dialog(csv_path, original_path=original_path)

    def _on_scientific_error(self, error_msg: str, filename: str):
        self.convert_worker = None
        self.convert_thread = None
        self.load_worker = None
        if hasattr(self.main_window, 'loading_manager'):
            self.main_window.loading_manager.finish_operation("file_loading")
        if self.is_loading_cancelled:
            return
        logger.error(f"[SCIENTIFIC] Conversion failed for {filename}: {error_msg}")
        QMessageBox.critical(
            self.main_window,
            "Dosya Dönüştürme Hatası",
            f"'{filename}' dosyası içe aktarılamadı:\n\n{error_msg}",
        )

    def _cleanup_intermediate_csv(self, csv_path: Optional[str]):
        """Ara (geçici) CSV dosyasını sessizce sil."""
        if csv_path and os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                logger.info(f"[SCIENTIFIC] Removed intermediate CSV: {csv_path}")
            except OSError as e:
                logger.debug(f"[SCIENTIFIC] Could not remove intermediate CSV: {e}")
    
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
            # Reset cancellation flag for new load
            self.is_loading_cancelled = False
            logger.info(f"[LOAD] Starting new file load: {filename}")
            
            if hasattr(self.main_window, 'loading_manager'):
                # Add subtitle for CSV files to indicate conversion
                subtitle = "Converting CSV to MPAI for better performance" if file_path.endswith('.csv') else ""
                self.main_window.loading_manager.start_operation("file_loading", f"Loading {filename}...", subtitle=subtitle)
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
                    lambda msg, pct: self.main_window.loading_manager.update_operation("file_loading", msg, pct)
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
    
    def _on_cancel_loading(self):
        """Handle cancel request from loading manager."""
        try:
            logger.info("[CANCEL] User requested to cancel loading operation")
            
            # Set cancellation flag to prevent callbacks
            self.is_loading_cancelled = True
            
            # Stop the current loading thread
            if self.load_worker:
                # Try to cancel the worker gracefully first
                if hasattr(self.load_worker, 'cancel'):
                    self.load_worker.cancel()
                
                # Get temp file paths before stopping
                temp_mpai = self.load_worker.settings.get('_temp_mpai_path')
                temp_settings = self.load_worker.settings.get('_temp_settings_path')
                
                # Stop the thread gracefully (with safety checks)
                for thread, worker in self.load_threads:
                    if worker == self.load_worker:
                        try:
                            if thread and thread.isRunning():
                                logger.info("[CANCEL] Stopping loading thread...")
                                thread.quit()
                                # Wait longer for graceful exit since we requested cancel
                                if not thread.wait(5000):  # Wait 5 seconds
                                    logger.info("[CANCEL] Thread did not stop gracefully, terminating...")
                                    thread.terminate()
                                    thread.wait(2000)  # Wait 2 more seconds after terminate
                                else:
                                    logger.info("[CANCEL] Thread stopped gracefully")
                        except RuntimeError:
                            # Thread already deleted - this is fine
                            logger.debug("[CANCEL] Thread already deleted")
                        break
                
                # Give the OS time to release file handles
                import time
                time.sleep(0.5)
                
                # Try to clean up temporary files (best effort - no warnings if fails)
                import shutil
                if temp_mpai and os.path.exists(temp_mpai):
                    try:
                        if os.path.isdir(temp_mpai):
                            shutil.rmtree(temp_mpai)
                        else:
                            os.remove(temp_mpai)
                        logger.info(f"[CANCEL] Deleted temp MPAI: {temp_mpai}")
                    except Exception:
                        # Silently fail - file will be cleaned up on next app start
                        # This is expected when C++ file handles are still open
                        logger.debug(f"[CANCEL] Temp file will be cleaned up on next app start: {temp_mpai}")
                
                if temp_settings and os.path.exists(temp_settings):
                    try:
                        os.remove(temp_settings)
                        logger.info(f"[CANCEL] Deleted temp settings: {temp_settings}")
                    except Exception:
                        logger.debug(f"[CANCEL] Settings file will be cleaned up on next app start")
                
                self.load_worker = None
            
            # Finish the loading operation
            if hasattr(self.main_window, 'loading_manager'):
                self.main_window.loading_manager.finish_operation("file_loading")
            
            # Update status bar
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Loading cancelled by user", 3000)
            
            logger.info("[CANCEL] Loading operation cancelled successfully")
            
        except Exception as e:
            logger.error(f"[CANCEL] Error during cancel: {e}")

