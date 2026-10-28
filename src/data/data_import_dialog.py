#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Veri Import Dialog'u
=============================

Bu dialog, veri dosyalarını açmak için gelişmiş seçenekler sunar:
- Veri önizlemesi (ilk 100 satır)
- Zaman kolonu seçimi
- Header satırı belirleme
- Başlangıç satırı seçimi
- Encoding (Unicode) seçimi
- Delimiter seçimi
- Dosya formatı otomatik algılama
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List

from src.data.csv_format_detector import detect_csv_format
from src.data.excel_to_csv import is_excel_file, list_sheets, excel_to_temp_csv, remove_temp_csv
from src.utils.error_handler import ErrorHandler
from PyQt5.QtWidgets import QDialog, QTextEdit, QMessageBox, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette
from src.data.import_dialog.constants import TIME_FORMAT_CHOICES
from src.data.import_dialog.layout import ImportDialogLayoutMixin
from src.data.import_dialog.preview import ImportDialogPreviewMixin
from src.data.import_dialog.theme import ImportDialogThemeMixin

logger = logging.getLogger(__name__)


class DataImportDialog(ImportDialogLayoutMixin, ImportDialogPreviewMixin, ImportDialogThemeMixin, QDialog):
    """Gelişmiş veri import dialog'u."""
    
    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.preview_df = None
        self.metadata = None
        self.preview_thread = None

        self.selected_time_column = None
        self.selected_encoding = 'utf-8'
        self.selected_delimiter = ','
        self.selected_header_row = 0
        self.selected_start_row = 0

        ext = os.path.splitext(file_path)[1].lower()
        self._is_binary_format = ext in ('.tdm', '.tdx', '.tdms')
        self._is_excel = is_excel_file(file_path)
        # Excel sayfaları geçici CSV'ye çevrilip metin dosyası gibi işlenir
        self._is_text_format = self._is_excel or ext not in ('.tdm', '.tdx', '.tdms', '.mpai', '.parquet')
        self._file_format = ext.lstrip('.')  # e.g. 'tdm', 'tdx', 'tdms', 'csv'

        # Önizleme/algılama için okunan dosya (Excel'de seçili sayfanın CSV'si)
        self._preview_path = file_path
        self._excel_csv_path = None
        self._excel_csv_handed_over = False

        self._setup_ui()
        self._setup_connections()
        self._configure_for_file_type()
        if self._is_excel:
            self._export_excel_sheet()
        if self._is_text_format and self._preview_path:
            # Açıklama satırlı export'lar için header/veri satırını baştan algıla
            self._apply_detected_format(show_message=False)
        self._load_initial_preview()
        
    def _configure_for_file_type(self):
        """Adjust format controls for Excel and binary NI formats."""
        if self._is_excel:
            self._configure_for_excel()
            return
        if not self._is_binary_format:
            return

        fmt = self._file_format.upper()

        # Title
        self.setWindowTitle(
            f"Veri Import - {os.path.basename(self.file_path)} [{fmt}]"
        )

        # Encoding / delimiter / auto-detect — not applicable
        self.encoding_combo.setEnabled(False)
        self.delimiter_combo.setEnabled(False)
        self.decimal_comma_checkbox.setEnabled(False)
        self.auto_detect_btn.setEnabled(False)

        # Header & row settings — binary formats carry their own metadata
        self.header_spinbox.setEnabled(False)
        self.has_header_checkbox.setEnabled(False)
        self.has_header_checkbox.setChecked(False)
        self.start_row_spinbox.setEnabled(False)

        # Informational tooltip on disabled groups
        for w in (self.encoding_combo, self.delimiter_combo,
                  self.header_spinbox, self.start_row_spinbox):
            w.setToolTip(f"{fmt} ikili formatı için geçerli değil")

    def _configure_for_excel(self):
        """Excel: sayfa seçimi göster; encoding/ayırıcı CSV dönüşümünde sabit."""
        self.setWindowTitle(
            f"Veri Import - {os.path.basename(self.file_path)} [{self._file_format.upper()}]"
        )
        # Sayfa geçici CSV'ye UTF-8, virgül ayırıcı ve nokta ondalıkla yazılır
        for w in (self.encoding_combo, self.delimiter_combo, self.decimal_comma_checkbox):
            w.blockSignals(True)
        self.encoding_combo.setCurrentText('utf-8')
        self.delimiter_combo.setCurrentText(', (Virgül)')
        self.decimal_comma_checkbox.setChecked(False)
        for w in (self.encoding_combo, self.delimiter_combo, self.decimal_comma_checkbox):
            w.blockSignals(False)
            w.setEnabled(False)
            w.setToolTip("Excel dosyalarında otomatik ayarlanır")

        try:
            sheets = list_sheets(self.file_path)
        except Exception as e:
            logger.error(f"Excel sayfaları okunamadı: {e}")
            QMessageBox.warning(self, "Excel Hatası", f"Excel dosyası okunamadı:\n\n{e}")
            sheets = []
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.addItems(sheets)
        self.sheet_combo.blockSignals(False)
        self.sheet_label.setVisible(True)
        self.sheet_combo.setVisible(True)
        self.sheet_combo.setEnabled(len(sheets) > 1)
        self.format_group.setMaximumHeight(180)
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)

    def _export_excel_sheet(self) -> bool:
        """Seçili Excel sayfasını geçici CSV'ye yaz; önizleme bu CSV'den okunur."""
        remove_temp_csv(self._excel_csv_path)
        self._excel_csv_path = None
        self._preview_path = None
        sheet = self.sheet_combo.currentText()
        if not sheet:
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._excel_csv_path = excel_to_temp_csv(self.file_path, sheet)
            self._preview_path = self._excel_csv_path
            return True
        except Exception as e:
            logger.error(f"Excel sayfası okunamadı ({sheet}): {e}")
            QMessageBox.warning(self, "Excel Hatası", f"'{sheet}' sayfası okunamadı:\n\n{e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _on_sheet_changed(self, _sheet: str):
        """Excel sayfası değişince yeniden dönüştür, algıla ve önizle."""
        if self._export_excel_sheet():
            self._apply_detected_format(show_message=False)
            self._refresh_preview()

    def _discard_excel_csv(self):
        """Dialog iptal edilirse geçici CSV'yi sil (kabul edilirse loader siler)."""
        if not self._excel_csv_handed_over:
            remove_temp_csv(self._excel_csv_path)
            self._excel_csv_path = None

    def accept(self):
        if self._is_excel and not self._excel_csv_path:
            QMessageBox.warning(self, "Excel Hatası", "İçe aktarılacak bir sayfa okunamadı.")
            return
        self._excel_csv_handed_over = True
        super().accept()

    def _update_time_column_options(self, columns):
        """Zaman kolonu seçeneklerini güncelle."""
        self.time_column_combo.clear()
        
        column_names = [str(col) for col in columns]
        if column_names:
            self.time_column_combo.addItems(column_names)
            self.time_column_combo.setCurrentIndex(0)  # Set first item as default
                
    def _on_settings_changed(self):
        """Ayarlar değiştiğinde otomatik yenileme."""
        # Kısa bir gecikme ile yenile (kullanıcı yazmayı bitirsin diye)
        if hasattr(self, '_refresh_timer'):
            self._refresh_timer.stop()
        
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_preview)
        self._refresh_timer.start(1000)  # 1 saniye gecikme
        
    def _on_header_row_changed(self, header_row: int):
        """Header satırı değişince veri başlangıcı header'dan önce kalmasın."""
        if header_row >= 0 and self.start_row_spinbox.value() <= header_row:
            # start_row_spinbox.valueChanged zaten _on_settings_changed'i tetikler
            self.start_row_spinbox.setValue(header_row + 1)
        self._on_settings_changed()

    def _on_header_checkbox_changed(self, checked):
        """Header checkbox değiştiğinde."""
        self.header_spinbox.setEnabled(checked)
        if not checked:
            self.header_spinbox.setValue(-1)
        else:
            self.header_spinbox.setValue(0)
        self._on_settings_changed()
        
    def _on_column_selected(self):
        """Sütun seçildiğinde."""
        selected_columns = []
        for item in self.preview_table.selectedItems():
            col_index = item.column()
            if col_index not in selected_columns:
                selected_columns.append(col_index)
                
        if selected_columns and self.preview_df is not None:
            col_name = self.preview_df.columns[selected_columns[0]]
            # Sadece mevcut kolonu kullan modundayken sütun seçimini güncelle
            if self.time_mode_combo.currentText() == "Mevcut Kolonu Kullan":
                self.time_column_combo.setCurrentText(str(col_name))
                
    def _on_time_mode_changed(self, mode_text: str):
        """Zaman kolonu modu değiştiğinde."""
        is_existing_mode = (mode_text == "Mevcut Kolonu Kullan")
        
        # Mevcut zaman kolonu ayarlarını etkinleştir/devre dışı bırak
        self.time_column_combo.setEnabled(is_existing_mode)
        self.time_format_combo.setEnabled(is_existing_mode)
        
        # Yeni zaman kolonu widget'larını göster/gizle
        for widget in self.new_time_widgets:
            widget.setVisible(not is_existing_mode)
        self.time_unit_label.setVisible(is_existing_mode)
        self.time_unit_combo.setVisible(is_existing_mode)
        
        logger.debug(f"Zaman kolonu modu değişti: {mode_text}")
        
    def _on_start_time_changed(self, start_time_text: str):
        """Başlangıç zamanı seçimi değiştiğinde."""
        is_custom = (start_time_text == "Özel Zaman")
        self.custom_start_time.setEnabled(is_custom)
        
        if not is_custom:
            self.custom_start_time.clear()
            
    def _on_sampling_freq_changed(self, freq: int):
        """Örnekleme frekansı değiştiğinde."""
        # Frekans bilgisini göster
        period = 1.0 / freq if freq > 0 else 0
        if hasattr(self, 'sampling_freq_spinbox'):
            tooltip = f"Örnekleme periyodu: {period:.6f} saniye"
            self.sampling_freq_spinbox.setToolTip(tooltip)
            
    def _auto_detect_format(self):
        """Dosya formatını otomatik algıla (buton)."""
        if self._apply_detected_format(show_message=True):
            self._refresh_preview()

    def _apply_detected_format(self, show_message: bool) -> bool:
        """
        Ayırıcı, header satırı, veri başlangıç satırı ve ondalık ayırıcıyı
        algılayıp arayüze uygula.

        Returns:
            Algılama başarılıysa True
        """
        delimiter_map = {
            ',': ', (Virgül)',
            ';': '; (Noktalı virgül)',
            '\t': '\\t (Tab)',
            '|': '| (Pipe)',
            ' ': ' (Boşluk)'
        }
        try:
            fmt = detect_csv_format(self._preview_path, encoding=self.encoding_combo.currentText())
        except Exception as e:
            logger.warning(f"Format algılama başarısız: {e}")
            if show_message:
                QMessageBox.warning(self, "Algılama Hatası",
                                    f"Otomatik format algılama başarısız:\n{str(e)}")
            return False

        if fmt is None:
            if show_message:
                QMessageBox.warning(
                    self, "Otomatik Algılama",
                    "Dosyada tutarlı bir veri bloğu bulunamadı.\n"
                    "Lütfen ayarları elle girin."
                )
            return False

        # Tek seferde uygula: her değişiklik ayrı önizleme tetiklemesin
        widgets = (self.delimiter_combo, self.decimal_comma_checkbox, self.has_header_checkbox,
                   self.header_spinbox, self.start_row_spinbox)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.delimiter_combo.setCurrentText(delimiter_map.get(fmt.delimiter, fmt.delimiter))
            self.decimal_comma_checkbox.setChecked(fmt.decimal_comma)
            has_header = fmt.header_row is not None
            self.has_header_checkbox.setChecked(has_header)
            self.header_spinbox.setEnabled(has_header)
            self.header_spinbox.setValue(fmt.header_row if has_header else -1)
            self.start_row_spinbox.setValue(fmt.start_row)
        finally:
            for w in widgets:
                w.blockSignals(False)
        if hasattr(self, '_refresh_timer'):
            self._refresh_timer.stop()

        if show_message:
            delim_name = delimiter_map.get(fmt.delimiter, repr(fmt.delimiter))
            header_text = (f"{fmt.header_row}. satır" if fmt.header_row is not None else "yok")
            QMessageBox.information(
                self,
                "Otomatik Algılama",
                f"Ayırıcı: {delim_name}\n"
                f"Header satırı: {header_text}\n"
                f"Veri başlangıç satırı: {fmt.start_row}\n"
                f"Ondalık ayırıcı: {'virgül' if fmt.decimal_comma else 'nokta'}\n"
                f"Kolon sayısı: {fmt.column_count}"
            )
        return True
            
    def get_import_settings(self) -> Dict[str, Any]:
        """Import ayarlarını döndür."""
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
            delimiter = delimiter_text
            
        # Zaman kolonu ayarları
        time_mode = self.time_mode_combo.currentText()
        is_custom_time = (time_mode == "Yeni Zaman Kolonu Oluştur")
        
        # Zaman formatını parse et (kullanıcı dostu formattan gerçek formata)
        time_format = TIME_FORMAT_CHOICES.get(self.time_format_combo.currentText(), 'Otomatik')
        
        settings = {
            'file_path':    self.file_path,
            'file_format':  self._file_format,
            'encoding':     self.encoding_combo.currentText(),
            'delimiter':    delimiter,
            'decimal_comma': self.decimal_comma_checkbox.isChecked() and delimiter != ',',
            'header_row':   (self.header_spinbox.value()
                             if (self.has_header_checkbox.isChecked()
                                 and not self._is_binary_format)
                             else None),
            'start_row':    (self.start_row_spinbox.value()
                             if not self._is_binary_format else 0),
            'time_mode':    time_mode,
            'create_custom_time': is_custom_time,
        }
        
        if self._is_excel:
            settings['excel_sheet'] = self.sheet_combo.currentText()
            # Önizlemede oluşturulan CSV: loader tekrar dönüştürmez, işi bitince siler
            settings['_excel_csv_path'] = self._excel_csv_path
        
        if is_custom_time:
            # Yeni zaman kolonu ayarları
            settings.update({
                'sampling_frequency': self.sampling_freq_spinbox.value(),
                'start_time_mode': self.start_time_combo.currentText(),
                'custom_start_time': self.custom_start_time.text() if self.custom_start_time.isEnabled() else None,
                'time_unit': self.time_unit_combo.currentText(),
                'new_time_column_name': self.new_time_column_name.text()
            })
        else:
            # Mevcut zaman kolonu ayarları
            time_column_raw = self.time_column_combo.currentText()
            time_column_cleaned = time_column_raw.strip()  # Match converter's cleaning
            settings.update({
                'time_column': time_column_cleaned,  # Use cleaned name for consistency
                'time_column_original': time_column_raw,  # Keep original for debugging
                'time_format': time_format,  # Parsed format
                'time_unit': self.time_unit_combo.currentText()
            })
            
        return settings

    def closeEvent(self, event):
        """Dialog kapatılırken thread'i temizle."""
        try:
            if self.preview_thread and self.preview_thread.isRunning():
                self.preview_thread.quit()
                if not self.preview_thread.wait(3000):  # 3 saniye bekle
                    logger.warning("Preview thread did not finish, terminating...")
                    self.preview_thread.terminate()
                    self.preview_thread.wait(1000)
        except RuntimeError as e:
            # Thread already deleted on C++ side
            logger.debug(f"Preview thread already deleted in closeEvent: {e}")
        self._discard_excel_csv()
        super().closeEvent(event)

    def reject(self):
        """Dialog iptal edildiğinde thread'i temizle."""
        try:
            if self.preview_thread and self.preview_thread.isRunning():
                self.preview_thread.quit()
                if not self.preview_thread.wait(3000):
                    logger.warning("Preview thread did not finish, terminating...")
                    self.preview_thread.terminate()
                    self.preview_thread.wait(1000)
        except RuntimeError as e:
            # Thread already deleted on C++ side
            logger.debug(f"Preview thread already deleted in reject: {e}")
        self._discard_excel_csv()
        super().reject()

def test_dialog():
    """Test fonksiyonu."""
    from PyQt5.QtWidgets import QApplication, QFileDialog
    import sys
    
    app = QApplication(sys.argv)
    
    # Test dosyası seç
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Test için dosya seç",
        "",
        "CSV Dosyaları (*.csv);;Tüm Dosyalar (*.*)"
    )
    
    if file_path:
        dialog = DataImportDialog(file_path)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_import_settings()
            print("Import ayarları:", settings)
        else:
            print("İptal edildi")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_dialog()
