"""
Saving data, projects and layouts.

Part of TimeGraphApp; moved from app.py without changes.
"""

import os
import logging
import polars as pl
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QThread
from src.operations.data_saver import DataSaver

logger = logging.getLogger(__name__)


class SavingMixin:
    """Saving data, projects and layouts."""

    def _on_file_save(self):
        """Dosya kaydetme işlemi."""
        active_widget = self.widget_container_manager.get_active_widget()
        if not active_widget:
            return
            
        logger.info("Dosya kaydetme işlemi başlatıldı")
        
        # Kaydetme formatı seç
        file_filter = (
            "CSV Dosyası (*.csv);;",
            "Excel Dosyası (*.xlsx)"
        )
        
        default_name = "time_graph_data.csv"
        if self.current_file_path:
            default_name = os.path.splitext(os.path.basename(self.current_file_path))[0] + "_exported.csv"
            
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Veriyi Kaydet",
            default_name,
            "".join(file_filter)
        )
        
        if file_path:
            try:
                self.status_bar.showMessage(f"Dosya kaydediliyor: {os.path.basename(file_path)}...")
                
                # Mevcut veriyi al
                export_data = active_widget.export_data()
                
                if not export_data or 'signals' not in export_data:
                    raise ValueError("Kaydedilecek veri bulunamadı")
                    
                signals = export_data['signals']
                
                if not signals:
                    raise ValueError("Kaydedilecek sinyal bulunamadı")
                    
                first_signal = next(iter(signals.values()))
                time_data = first_signal.get('x_data', [])
                
                data_dict = {'time': time_data}
                for signal_name, signal_data in signals.items():
                    data_dict[signal_name] = signal_data.get('y_data', [])
                    
                df_to_save = pl.DataFrame(data_dict)

                # --- Threaded Data Saving ---
                self.save_thread = QThread()
                self.save_worker = DataSaver(df_to_save, file_path, selected_filter)
                self.save_worker.moveToThread(self.save_thread)

                # Connect signals
                self.save_thread.started.connect(self.save_worker.run)
                self.save_worker.finished.connect(self.on_saving_finished)
                self.save_worker.error.connect(self.on_saving_error)
                
                # Cleanup
                self.save_worker.finished.connect(self.save_thread.quit)
                self.save_worker.finished.connect(self.save_worker.deleteLater)
                self.save_thread.finished.connect(self.save_thread.deleteLater)

                # Start the thread
                self.save_thread.start()

            except (ValueError, KeyError) as e:
                error_msg = f"Kaydedilecek veri hazırlanırken hata oluştu: {str(e)}"
                logger.error(error_msg)
                QMessageBox.warning(self, "Veri Kaydetme Hatası", error_msg)
                self.status_bar.showMessage("Veri kaydetme başarısız", 5000)
            except Exception as e:
                error_msg = f"Dosya kaydetme işlemi başlatılamadı: {str(e)}"
                logger.error(error_msg)
                QMessageBox.critical(self, "Dosya Kaydetme Hatası", error_msg)

    def on_saving_finished(self, saved_path):
        """Worker thread'den dosya kaydetme bittiğinde çağrılır."""
        self.is_data_modified = False
        self.status_bar.showMessage(
            f"Dosya kaydedildi: {os.path.basename(saved_path)}", 
            5000
        )
        logger.info(f"Dosya başarıyla kaydedildi: {saved_path}")

    def on_saving_error(self, error_msg):
        """Worker thread'de kaydetme hatası oluştuğunda çağrılır."""
        logger.error(f"Dosya kaydedilirken hata oluştu (worker'dan): {error_msg}")
        QMessageBox.critical(
            self,
            "Dosya Kaydetme Hatası",
            f"Dosya kaydedilemedi:\n\n{error_msg}"
        )
        self.status_bar.showMessage("Dosya kaydetme başarısız", 5000)

    def _on_layout_import(self):
        """
        Layout'u bir dosyadan içe aktar.
        
        ✅ REFACTORED: Delegate to LayoutOperations module
        """
        if self.layout_ops:
            self.layout_ops.import_layout_dialog()
        else:
            logger.error("LayoutOperations module not initialized")

    def _on_layout_export(self):
        """
        Mevcut layout'u bir dosyaya aktar.
        
        ✅ REFACTORED: Delegate to LayoutOperations module
        """
        if self.layout_ops:
            self.layout_ops.export_layout_dialog()
        else:
            logger.error("LayoutOperations module not initialized")

    def _on_project_save(self):
        """
        Projeyi kaydet (.mpai format) - Veri + Layout + Metadata tek dosyada.
        
        ✅ REFACTORED: Delegate to ProjectOperations module
        """
        if self.project_ops:
            self.project_ops.save_project_dialog()
        else:
            logger.error("ProjectOperations module not initialized")

    def _on_project_open(self):
        """
        Proje aç (.mpai format) - Veri + Layout tek dosyadan hızlı yükleme.
        
        ✅ REFACTORED: Delegate to ProjectOperations module
        """
        if self.project_ops:
            self.project_ops.open_project_dialog()
        else:
            logger.error("ProjectOperations module not initialized")

    def _on_save_project_before_close(self, file_index: int):
        """
        Handle save project request before closing a file tab.
        
        Called when user clicks "Yes" on "Save project?" dialog.
        """
        file_data = self.file_manager.get_file_data(file_index)
        if not file_data:
            return
        
        # Make this file active (tab, file index and widget together):
        # save_project_dialog reads the active widget AND the active file's
        # metadata, so switching only the widget would mix two files
        self.file_manager.file_tab_widget.setCurrentIndex(file_index)

        # Trigger save dialog
        if self.project_ops:
            # After successful save, mark as saved and close
            success = self.project_ops.save_project_dialog()
            if success:
                # Mark as saved
                file_data['is_project_saved'] = True
                # Now close the tab and delete its temp MPAI cache
                self.file_manager.close_file_and_cleanup(file_index)
        else:
            logger.error("ProjectOperations module not initialized")
