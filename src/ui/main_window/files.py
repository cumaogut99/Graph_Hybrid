"""
Switching between and closing the open files.

Part of TimeGraphApp; moved from app.py without changes.
"""

import logging
from PyQt5.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class OpenFilesMixin:
    """Switching between and closing the open files."""

    def _on_data_changed(self, data):
        """Veri değişikliği işlemi."""
        self.is_data_modified = True
        
        # Aktif dosyanın modified flag'ini güncelle
        active_file = self.file_manager.get_active_file_data()
        if active_file:
            active_file['is_data_modified'] = True
        
        # Pencere başlığına * ekle
        current_title = self.windowTitle()
        if not current_title.endswith('*'):
            self.setWindowTitle(current_title + '*')

    def _on_file_switched(self, new_index: int, old_index: int):
        """Dosya sekmesi değiştiğinde çağrılır - sadece widget'lar arası geçiş yap."""
        file_data = self.file_manager.get_file_data(new_index)
        if not file_data:
            return
        
        logger.info(f"[FILE SWITCH] Switching from file {old_index} to file {new_index}: {file_data['filename']}")
        
        try:
            # Sadece widget'a geç - her dosyanın kendi widget'ı var!
            self.widget_container_manager.switch_to_file_widget(new_index)
            
            # UI güncellemeleri
            self.current_file_path = file_data['file_path']
            self.is_data_modified = file_data['is_data_modified']
            self.setWindowTitle(f"Time Graph Widget - {file_data['filename']}")
            
            df = file_data['df']
            if hasattr(df, 'get_header'): # MpaiReader check
                row_count = df.get_row_count()
                col_count = df.get_column_count()
            else:
                row_count = df.height
                col_count = len(df.columns)
                
            self.status_bar.showMessage(
                f"Aktif dosya: {file_data['filename']} ({row_count:,} satır, {col_count} sütun)",
                5000
            )
            
            logger.info(f"Successfully switched to file: {file_data['filename']}")
            
        except Exception as e:
            error_msg = f"Dosya değiştirme sırasında hata: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Dosya Değiştirme Hatası", error_msg)

    def _on_file_closed(self, file_index: int):
        """Bir dosya kapatıldığında çağrılır."""
        logger.info(f"[FILE CLOSE] Closing file {file_index}")
        
        # Bu dosyanın widget'ını kaldır
        self.widget_container_manager.remove_widget_for_file(file_index)
        
        logger.info(f"[FILE CLOSE] Widget removed for file {file_index}")

    def _on_all_files_closed(self):
        """Tüm dosyalar kapatıldığında çağrılır."""
        self.setWindowTitle("Time Graph - Veri Analizi ve Görselleştirme")
        self.status_bar.showMessage("Tüm dosyalar kapatıldı", 3000)
        self.current_file_path = None
        self.is_data_modified = False
        
        # Tüm widget'ları temizle
        self.widget_container_manager.cleanup_all()
        
        # ✅ FIX: Initial widget'ın toolbar sinyallerini bağla (dosya açma için)
        initial_widget = self.widget_container_manager.initial_widget
        if initial_widget:
            self._connect_widget_signals(initial_widget)
            logger.info("[FILE CLOSE] Initial widget toolbar signals connected")
