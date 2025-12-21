"""
Layout Operations Module

Layout import/export işlemlerini yöneten fonksiyonlar.

Bu modül şunları içerir:
- Layout export (JSON)
- Layout import (JSON)
- Layout validasyonu

Refactored from: app.py (TimeGraphApp sınıfı)
"""

import logging
import json
import os
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)


class LayoutOperations:
    """
    Layout import/export işlemlerini yöneten sınıf.
    TimeGraphApp'den ayrıştırılmış, bağımsız çalışabilen modül.
    """
    
    def __init__(self, main_window):
        """
        Args:
            main_window: TimeGraphApp ana pencere referansı
        """
        self.main_window = main_window
        
    def import_layout_dialog(self):
        """
        Layout import dialog'unu göster ve import işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_layout_import()
        
        Returns:
            bool: İşlem başarılı ise True
        """
        active_widget = self.main_window.widget_container_manager.get_active_widget()
        if not active_widget:
            return False
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import Layout",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    layout_config = json.load(f)
                
                active_widget.set_layout_config(layout_config)
                
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(
                        f"Layout başarıyla içe aktarıldı: {os.path.basename(file_path)}", 
                        5000
                    )
                logger.info(f"Layout başarıyla içe aktarıldı: {file_path}")
                return True

            except Exception as e:
                error_msg = f"Layout içe aktarılırken hata oluştu: {str(e)}"
                logger.error(error_msg)
                QMessageBox.critical(
                    self.main_window,
                    "Layout Import Hatası",
                    error_msg
                )
                return False
        
        return False

    def export_layout_dialog(self):
        """
        Layout export dialog'unu göster ve export işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_layout_export()
        
        Returns:
            bool: İşlem başarılı ise True
        """
        active_widget = self.main_window.widget_container_manager.get_active_widget()
        if not active_widget:
            return False

        try:
            layout_config = active_widget.get_layout_config()

            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Export Layout",
                "layout.json",
                "JSON Files (*.json)"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(layout_config, f, indent=4, ensure_ascii=False)
                
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(
                        f"Layout başarıyla dışa aktarıldı: {os.path.basename(file_path)}", 
                        5000
                    )
                logger.info(f"Layout başarıyla dışa aktarıldı: {file_path}")
                return True

        except Exception as e:
            error_msg = f"Layout dışa aktarılırken hata oluştu: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(
                self.main_window,
                "Layout Export Hatası",
                error_msg
            )
            return False
        
        return False
