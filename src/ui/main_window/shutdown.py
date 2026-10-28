"""
Closing the application: unsaved-work prompts, temp file cleanup and stopping threads.

Part of TimeGraphApp; moved from app.py without changes.
"""

import logging
from PyQt5.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class ShutdownMixin:
    """Closing the application: unsaved-work prompts, temp file cleanup and stopping threads."""

    def _on_file_exit(self):
        """Uygulamadan çıkış."""
        self.close()

    def closeEvent(self, event):
        """Pencere kapatma olayı."""
        # Check for unsaved changes in ANY file
        has_unsaved = False
        for file_data in self.file_manager.loaded_files:
            if file_data.get('is_data_modified', False):
                has_unsaved = True
                break
        
        if has_unsaved:
            reply = QMessageBox.question(
                self,
                "Kaydedilmemiş Değişiklikler",
                "Bir veya daha fazla dosyada kaydedilmemiş değişiklikler var.\nÇıkmak istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # === PROMPT TO SAVE UNSAVED PROJECTS ===
        unsaved_projects = [
            f for f in self.file_manager.loaded_files
            if f.get('settings', {}).get('_is_temp_file', False) and 
               not f.get('is_project_saved', False)
        ]
        
        if unsaved_projects:
            filenames = ", ".join([f.get('filename', 'Unknown') for f in unsaved_projects[:3]])
            if len(unsaved_projects) > 3:
                filenames += f" ve {len(unsaved_projects) - 3} dosya daha"
            
            reply = QMessageBox.question(
                self,
                "Save Projects",
                f"The following files have not been saved as projects yet:\n{filenames}\n\n"
                f"Are you sure you want to exit without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # === AUTO CLEANUP TEMP FILES (no prompt) ===
        for file_data in self.file_manager.loaded_files:
            if file_data.get('settings', {}).get('_is_temp_file', False):
                self.file_manager._cleanup_temp_files(file_data)
        logger.info("[CLEANUP] All temp files cleaned up on app close")
                
        # Temizlik işlemleri - tüm widget'ları temizle
        if self.widget_container_manager:
            self.widget_container_manager.cleanup_all()
        
        # Cleanup status bar and loading manager
        if self.status_bar_manager:
            self.status_bar_manager.cleanup()
        
        # Cleanup project manager temp files
        if hasattr(self, 'project_manager') and self.project_manager:
            self.project_manager.cleanup()
            logger.debug("Project manager cleaned up")
        
        # === FİX: TÜM LOAD THREAD'LERİNİ TEMİZLE ===
        logger.info("Cleaning up load threads...")
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
            
        if self.save_thread:
            try:
                if self.save_thread.isRunning():
                    self.save_thread.quit()
                    if not self.save_thread.wait(3000):  # Wait up to 3 seconds
                        logger.warning("Save thread did not finish, terminating...")
                        self.save_thread.terminate()
                        self.save_thread.wait(1000)
            except RuntimeError as e:
                logger.debug(f"Save thread already deleted: {e}")

        if self.loading_manager:
            # Finish any active operations
            for operation in self.loading_manager.get_active_operations():
                self.loading_manager.finish_operation(operation)
            
        # Debug: Thread sayısını logla
        self._log_active_threads()
        
        logger.info("Uygulama kapatılıyor")
        event.accept()

    def _log_active_threads(self):
        """Aktif thread'leri logla."""
        import threading
        active_threads = threading.active_count()
        logger.info(f"Aktif thread sayısı: {active_threads}")
        
        # QThread'leri kontrol et
        qthread_count = 0
        try:
            if hasattr(self, 'load_thread') and self.load_thread and self.load_thread.isRunning():
                qthread_count += 1
                logger.info("- Load thread hala çalışıyor")
        except RuntimeError:
            pass  # Thread already deleted
            
        try:
            if hasattr(self, 'save_thread') and self.save_thread and self.save_thread.isRunning():
                qthread_count += 1
                logger.info("- Save thread hala çalışıyor")
        except RuntimeError:
            pass  # Thread already deleted
            
        # Check all widgets for threads
        if hasattr(self, 'widget_container_manager') and self.widget_container_manager:
            for file_index, widget in self.widget_container_manager.widgets.items():
                if widget:
                    if hasattr(widget, 'processing_thread') and widget.processing_thread:
                        try:
                            if widget.processing_thread.isRunning():
                                qthread_count += 1
                                logger.info(f"- Processing thread for file {file_index} hala çalışıyor")
                        except RuntimeError:
                            pass  # Thread already deleted
        if hasattr(self, 'status_bar_manager') and self.status_bar_manager:
            if hasattr(self.status_bar_manager, 'monitor_thread') and self.status_bar_manager.monitor_thread:
                try:
                    if self.status_bar_manager.monitor_thread.isRunning():
                        qthread_count += 1
                        logger.info("- Monitor thread hala çalışıyor")
                except RuntimeError:
                    pass  # Thread already deleted
        
        logger.info(f"Toplam QThread sayısı: {qthread_count}")
