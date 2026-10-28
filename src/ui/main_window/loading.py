"""
Opening a file and handling the result of the loader thread.

Part of TimeGraphApp; moved from app.py without changes.
"""

import os
import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer, QThread
from src.data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class FileLoadingMixin:
    """Opening a file and handling the result of the loader thread."""

    def _on_file_open(self):
        """
        Dosya açma işlemi - Gelişmiş import dialog ile.
        
        ✅ REFACTORED: Delegate to DataOperations module
        """
        if self.data_ops:
            self.data_ops.open_file_dialog()
        else:
            logger.error("DataOperations module not initialized")

    def _load_data_with_settings(self, settings: dict):
        """Ayarlarla veri dosyasını yükle."""
        # Reset datetime conversion flag
        self._datetime_converted = False
        
        try:
            file_path = settings['file_path']
            filename = os.path.basename(file_path)
            
            # Start loading operation
            self.loading_manager.start_operation("file_loading", f"Loading {filename}...")
            self.status_bar.set_operation("File Loading", 0)
            
            # === MULTI-FILE CHECK ===
            # Check if file already open
            existing_index = self.file_manager.is_file_already_open(file_path)
            if existing_index >= 0:
                self.file_manager.file_tab_widget.setCurrentIndex(existing_index)
                self.loading_manager.finish_operation("file_loading")
                QMessageBox.information(
                    self,
                    "Dosya Zaten Açık",
                    f"'{filename}' dosyası zaten açık.\nİlgili sekmeye geçildi."
                )
                return
            
            # Check file limit
            if not self.file_manager.can_add_file():
                current_count = len(self.file_manager.loaded_files)
                logger.warning(f"Cannot add file: {current_count}/{self.file_manager.max_files} files already loaded")
                QMessageBox.warning(
                    self,
                    "Maksimum Dosya Sayısı",
                    f"Maksimum {self.file_manager.max_files} dosya açık olabilir.\n"
                    f"Şu anda {current_count} dosya açık.\n"
                    f"Yeni dosya yüklemek için önce bir dosyayı kapatın."
                )
                self.loading_manager.finish_operation("file_loading")
                return
            
            # Dosya boyutunu kontrol et
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            # 25 MB limiti kaldırıldı - büyük dosyalar için C++ kullanılıyor
            if file_size_mb > 1024:  # 1 GB üzeri için uyarı
                self.loading_manager.update_operation("file_loading", f"Very large file: {filename} ({file_size_mb:.1f} MB) - Using C++ engine")
            elif file_size_mb > 100:
                self.loading_manager.update_operation("file_loading", f"Large file: {filename} ({file_size_mb:.1f} MB) - C++ accelerated")
            else:
                self.loading_manager.update_operation("file_loading", f"Loading: {filename}")
            
            # === FİX: DÜZGÜN THREAD YÖNETİMİ ===
            # Yeni thread oluştur
            load_thread = QThread()
            load_worker = DataLoader(settings)
            load_worker.moveToThread(load_thread)

            # Connect signals
            load_thread.started.connect(load_worker.run)
            load_worker.finished.connect(self.on_loading_finished)
            load_worker.error.connect(self.on_loading_error)
            load_worker.progress.connect(lambda msg: self.loading_manager.update_operation("file_loading", msg))
            
            # Cleanup - BU KRİTİK!
            load_worker.finished.connect(load_thread.quit)
            load_worker.error.connect(load_thread.quit)  # Error durumunda da quit et
            load_worker.finished.connect(load_worker.deleteLater)
            load_thread.finished.connect(load_thread.deleteLater)
            
            # Thread'i listede tut (temizlik için)
            self.load_threads.append((load_thread, load_worker))
            self.load_worker = load_worker  # Current worker reference

            # Start the thread
            load_thread.start()
            logger.debug(f"Started loading thread for {filename}")

        except Exception as e:
             # This will now only catch errors from the pre-flight checks, not the loading itself
            error_msg = f"Dosya yükleme başlatılırken hata oluştu: {str(e)}"
            logger.error(error_msg)
            self.loading_manager.finish_operation("file_loading")
            QMessageBox.critical(self, "Dosya Yükleme Hatası", error_msg)
            self.status_bar.showMessage("Dosya yükleme başarısız", 5000)

    def on_loading_finished(self, df, time_column):
        """Worker thread'den veri yükleme bittiğinde çağrılır."""
        # Check if loading was cancelled
        if hasattr(self, 'data_ops') and self.data_ops and self.data_ops.is_loading_cancelled:
            logger.info("[CANCEL] ✓ Loading finished callback blocked - file loading prevented")
            return
        
        logger.info("[LOAD] Processing loaded data...")
        try:
            file_path = self.load_worker.settings.get('file_path', self.current_file_path)
            filename = os.path.basename(file_path)

            # Determine if data source is MpaiReader
            is_mpai = hasattr(df, 'get_header')
            
            if is_mpai:
                columns = df.get_column_names()
                row_count = df.get_row_count()
                col_count = df.get_column_count()
                logger.info("Loaded data type: MpaiReader")
            else:
                columns = df.columns
                row_count = df.height
                col_count = len(columns)
                logger.info("Loaded data type: Polars DataFrame")

            logger.debug(f"Widget'a gönderilen zaman kolonu: '{time_column}'")
            # logger.debug(f"DataFrame kolonları: {columns}") # Too verbose for large files
            
            if not is_mpai and time_column in columns:
                try:
                    time_data_sample = df.get_column(time_column).head(5).to_numpy()
                    logger.debug(f"Zaman kolonu '{time_column}' ilk 5 değer: {time_data_sample}")
                except Exception as e:
                    logger.debug(f"Could not sample time column: {e}")
            
            # ROBUST: Veri kalite özeti göster (MPAI için skip edilebilir)
            self._show_data_quality_summary(df, filename)
            
            # === MULTI-FILE: Dosyayı file manager'a ekle ===
            file_metadata = {
                'file_path': file_path,
                'filename': filename,
                'df': df,
                'time_column': time_column,
                'settings': self.load_worker.settings.copy(),
                'datetime_converted': getattr(self.load_worker, '_datetime_converted', False),
                'is_data_modified': False
            }
            
            # Dosyayı ekle ve index al
            file_index = self.file_manager.add_file(file_metadata)
            
            if file_index < 0:
                logger.error("File could not be added to manager")
                return
            
            # Bu dosya için yeni bir widget oluştur
            logger.info(f"[WIDGET CREATE] Creating new widget for file {file_index}")
            widget = self.widget_container_manager.create_widget_for_file(file_index)
            
            # Widget sinyallerini bağla
            self._connect_widget_signals(widget)
            
            # Widget'a geç
            self.widget_container_manager.switch_to_file_widget(file_index)
            
            # Veriyi widget'a yükle
            logger.info(f"[WIDGET CREATE] Loading data to widget for file {file_index}")
            widget.update_data(df, time_column=time_column)
            
            # ✅ LOD OPTIMIZATION: Store MPAI file path for LOD parquet lookup
            if is_mpai and hasattr(widget, 'signal_processor'):
                # For .mpai files, use the file_path directly
                # For converted CSVs, the MPAI path is base + .mpai
                if file_path.endswith('.mpai'):
                    widget.signal_processor.current_mpai_path = file_path
                else:
                    widget.signal_processor.current_mpai_path = os.path.splitext(file_path)[0] + '.mpai'
                logger.info(f"[LOD] Set current_mpai_path: {widget.signal_processor.current_mpai_path}")
            
            # CRITICAL FIX: Ensure full view range after loading data
            # Use QTimer to allow layout to settle before resetting view
            QTimer.singleShot(100, lambda: self._safe_reset_view(widget))
            
            # Datetime axis ayarını yap
            active_container = widget.get_active_graph_container()
            if active_container and hasattr(active_container.plot_manager, 'enable_datetime_axis'):
                if file_metadata['datetime_converted']:
                    active_container.plot_manager.enable_datetime_axis(True)
                    logger.info("Datetime axis formatting enabled for better readability")
                    if hasattr(widget, 'statistics_panel'):
                        widget.statistics_panel.set_datetime_axis(True)
                else:
                    active_container.plot_manager.enable_datetime_axis(False)
                    logger.info("Datetime axis formatting disabled for numeric data")
                    if hasattr(widget, 'statistics_panel'):
                        widget.statistics_panel.set_datetime_axis(False)
            
            # Metin değerler 0'a çevrildiyse kullanıcıya söyle (grafik çizildikten sonra)
            non_numeric = getattr(self.load_worker, 'non_numeric_report', None)
            if non_numeric and self.data_ops:
                QTimer.singleShot(300, lambda: self.data_ops.show_non_numeric_warning(filename, non_numeric))

            # Başarılı yükleme
            self.current_file_path = file_path
            self.is_data_modified = False
            
            self.setWindowTitle(f"Time Graph Widget - {filename}")
            
            self.status_bar.showMessage(
                f"Dosya yüklendi: {filename} ({row_count:,} satır, {col_count} sütun)", 
                10000
            )
            
            logger.info(f"Dosya başarıyla yüklendi: {file_path} ({row_count} satır, {col_count} sütun)")
            logger.info(f"Toplam açık dosya: {self.file_manager.get_file_count()}")
            
        except Exception as e:
            error_msg = f"Veri widget'a yüklenirken hata oluştu: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Veri İşleme Hatası", error_msg)

        finally:
            self.loading_manager.finish_operation("file_loading")

    def _safe_reset_view(self, widget):
        """Safely reset view of a widget if it has a plot manager."""
        try:
            active_container = widget.get_active_graph_container()
            if active_container and hasattr(active_container, 'plot_manager'):
                active_container.plot_manager.reset_view()
                logger.info("[AUTO-FIX] Reset view called after data load to ensure full visibility")
        except Exception as e:
            logger.warning(f"Could not auto-reset view: {e}")

    def on_loading_error(self, error_msg):
        """Worker thread'de bir hata oluştuğunda çağrılır."""
        # Check if loading was cancelled
        if hasattr(self, 'data_ops') and self.data_ops and self.data_ops.is_loading_cancelled:
            logger.info("[CANCEL] ✓ Loading error callback blocked - error suppressed")
            return
        
        logger.error(f"Dosya yüklenirken hata oluştu (worker'dan): {error_msg}")
        self.loading_manager.finish_operation("file_loading")
        QMessageBox.critical(
            self,
            "Dosya Yükleme Hatası",
            f"Dosya yüklenemedi:\n\n{error_msg}\n\nLütfen import ayarlarını kontrol edin."
        )

    def _show_data_quality_summary(self, df, filename=""):
        """
        Yüklenen verinin kalite özetini logla.
        
        ✅ REFACTORED: Delegate to DataOperations module
        """
        if self.data_ops:
            self.data_ops.show_data_quality_summary(df, filename)
        else:
            logger.error("DataOperations module not initialized")
