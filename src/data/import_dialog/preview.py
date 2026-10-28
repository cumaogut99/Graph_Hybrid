"""
Loading the preview, showing it in the table and checking its data quality.

Part of DataImportDialog; moved from src/data/data_import_dialog.py without changes.
"""

import logging
from src.data.data_validator import DataValidator
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
from PyQt5.QtGui import QColor
from src.data.import_dialog.preview_thread import DataPreviewThread

logger = logging.getLogger(__name__)


class ImportDialogPreviewMixin:
    """Loading the preview, showing it in the table and checking its data quality."""

    def _load_initial_preview(self):
        """İlk önizlemeyi yükle."""
        self._refresh_preview()

    def _refresh_preview(self):
        """Önizlemeyi yenile."""
        try:
            if self.preview_thread and self.preview_thread.isRunning():
                self.preview_thread.quit()
                if not self.preview_thread.wait(3000):  # 3 saniye bekle
                    logger.warning("Preview thread did not finish in _refresh_preview, terminating...")
                    self.preview_thread.terminate()
                    self.preview_thread.wait(1000)
        except RuntimeError as e:
            # Thread already deleted on C++ side
            logger.debug(f"Preview thread already deleted: {e}")
            self.preview_thread = None
            
        # Ayarları al
        encoding = self.encoding_combo.currentText()
        delimiter_text = self.delimiter_combo.currentText()
        
        # Delimiter'ı parse et
        if delimiter_text.startswith(','):
            delimiter = ','
        elif delimiter_text.startswith(';'):
            delimiter = ';'
        elif delimiter_text.startswith('\\t'):
            delimiter = '\t'
        elif delimiter_text.startswith('|'):
            delimiter = '|'
        elif delimiter_text.startswith(' '):
            delimiter = ' '
        elif delimiter_text.startswith('.'):
            delimiter = '.'
        else:
            delimiter = delimiter_text  # Kullanıcı girişi
            
        header_row = self.header_spinbox.value() if self.has_header_checkbox.isChecked() else -1
        start_row = self.start_row_spinbox.value()
        decimal_comma = self.decimal_comma_checkbox.isChecked() and delimiter != ','

        # Progress bar göster
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_label.setText("Veri önizlemesi yükleniyor...")
        
        # Thread başlat
        if not self._preview_path:
            return  # Excel sayfası okunamadı; hata zaten gösterildi
        
        self.preview_thread = DataPreviewThread(
            self._preview_path, encoding, delimiter, header_row, start_row, decimal_comma
        )
        self.preview_thread.preview_ready.connect(self._on_preview_ready)
        self.preview_thread.error_occurred.connect(self._on_preview_error)
        
        # Otomatik temizleme
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        
        self.preview_thread.start()

    def _on_preview_ready(self, df, metadata):
        """Önizleme hazır olduğunda çağrılır."""
        self.preview_df = df
        self.metadata = metadata
        
        # Progress bar gizle
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # ROBUST: Veri validasyonu yap
        self._validate_data_quality(df)
        
        # Tabloyu güncelle
        self._update_preview_table(df)
        
        # Zaman kolonu seçeneklerini güncelle
        self._update_time_column_options(df.columns)

    def _on_preview_error(self, error_msg):
        """Önizleme hatası olduğunda çağrılır."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        QMessageBox.warning(
            self, 
            "Önizleme Hatası", 
            f"Veri önizlemesi yüklenirken hata oluştu:\n\n{error_msg}\n\nLütfen format ayarlarını kontrol edin."
        )

    def _update_preview_table(self, df):
        """Önizleme tablosunu güncelle."""
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        
        # Sütun başlıkları
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        
        # Veriyi doldur
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                value = df.row(i)[j]
                if value is None:
                    item = QTableWidgetItem("NaN")
                    item.setForeground(QColor(128, 128, 128))
                else:
                    item = QTableWidgetItem(str(value))
                    
                self.preview_table.setItem(i, j, item)
                
        # Sütun genişliklerini ayarla
        self.preview_table.resizeColumnsToContents()

    def _validate_data_quality(self, df):
        """Veri kalitesini kontrol et ve kullanıcıyı bilgilendir."""
        try:
            import pandas as pd
            
            # Polars'ı pandas'a çevir validation için
            df_pd = df.to_pandas()
            
            # DataValidator kullan
            validator = DataValidator()
            validation_results = validator.validate_dataframe(df_pd)
            
            # Sorunları topla
            issues = []
            warnings = []
            
            for col_name, result in validation_results.items():
                # Yüksek null oranı
                null_pct = result.statistics.get('null_percentage', 0)
                if null_pct > 50:
                    issues.append(f"⚠️ '{col_name}': %{null_pct:.1f} boş değer")
                elif null_pct > 20:
                    warnings.append(f"💡 '{col_name}': %{null_pct:.1f} boş değer")
                
                # Mixed type uyarısı
                if result.data_type.value == 'mixed':
                    issues.append(f"⚠️ '{col_name}': Karışık veri tipleri tespit edildi")
                
                # Conversion sorunları
                if not result.is_valid:
                    for issue in result.issues:
                        if 'çevrilemedi' in issue:
                            warnings.append(f"💡 '{col_name}': {issue}")
            
            # Kullanıcıya bilgi ver (Kaldırıldı - log only)
            if issues:
                logger.info("Data Quality Issues found but dialog suppressed:")
                for issue in issues:
                    logger.info(issue)
                # msg = "📊 **Veri Kalite Uyarıları:**\n\n"
                # msg += "\n".join(issues[:5])  # İlk 5 sorunu göster
                # if len(issues) > 5:
                #     msg += f"\n\n... ve {len(issues) - 5} diğer sorun"
                # msg += "\n\n✅ Uygulama bu sorunları otomatik olarak düzeltmeye çalışacak."
                # msg += "\n\n💡 İsterseniz import ayarlarını değiştirebilirsiniz."
                
                # QMessageBox.information(self, "Veri Kalite Kontrolü", msg)
            
            logger.info(f"Veri validasyonu tamamlandı: {len(issues)} sorun, {len(warnings)} uyarı")
            
        except Exception as e:
            logger.warning(f"Veri validasyonu başarısız: {e}")
            # Validasyon hatası import'u engellemez
