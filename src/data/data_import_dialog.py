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
import io
import logging
from typing import Optional, Dict, Any, List
import polars as pl

from src.data.data_validator import DataValidator
from src.utils.error_handler import ErrorHandler
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel, QComboBox, QSpinBox, QPushButton, QTextEdit, QCheckBox,
    QProgressBar, QMessageBox, QFrame, QScrollArea, QGridLayout, QLineEdit, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

logger = logging.getLogger(__name__)

class DataPreviewThread(QThread):
    """Veri önizlemesi için background thread."""
    
    preview_ready = Signal(object, dict)  # DataFrame ve metadata
    error_occurred = Signal(str)
    
    def __init__(self, file_path: str, encoding: str = 'latin-1', delimiter: str = ',', 
                 header_row: int = 0, start_row: int = 0):
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter
        self.header_row = header_row
        self.start_row = start_row
        
        # Calculate skip_rows for Polars
        # Polars'ın skip_rows parametresi: dosyanın başından kaç satır atlanacak
        # has_header: header satırının olup olmadığı
        
        if header_row >= 0:
            # Header var
            self.has_header = True
            # Header satırına kadar atla (header satırı kendisi atlanmaz, okunur)
            self.skip_rows_before_header = header_row
            
            # Header'dan sonra veri başlangıcına kadar atlanacak satır sayısı
            if start_row > header_row:
                self.skip_rows_after_header = start_row - header_row - 1
            else:
                self.skip_rows_after_header = 0
        else:
            # Header yok
            self.has_header = False
            self.skip_rows_before_header = start_row
            self.skip_rows_after_header = 0
        
    def run(self):
        """Veri önizlemesi yükle."""
        try:
            # Dosya uzantısına göre okuma yöntemi belirle
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                # Always use manual line reading to properly handle header_row and start_row
                df = self._read_csv_with_manual_skip()
            elif file_ext in ['.xlsx', '.xls']:
                # Excel dosyası
                df = pl.read_excel(
                    self.file_path,
                    has_header=self.has_header,
                    skip_rows=self.skip_rows,
                    n_rows=100
                )
            elif file_ext == '.parquet':
                # Parquet dosyası
                df = pl.read_parquet(self.file_path)
                df = df.head(100)  # İlk 100 satır
            elif file_ext == '.mpai':
                # MPAI dosyası
                import time_graph_cpp
                reader = time_graph_cpp.MpaiReader(self.file_path)
                
                # Metadata ve preview için ilk 100 satır
                count = reader.get_row_count()
                preview_count = min(count, 100)
                cols = reader.get_column_names()
                
                # Data dict oluştur
                data = {}
                for col in cols:
                    try:
                        data[col] = reader.load_column_slice(col, 0, preview_count)
                    except Exception as e:
                        logger.warning(f"Preview column '{col}' failed: {e}")
                        data[col] = [None] * preview_count
                
                df = pl.DataFrame(data)
            else:
                # Genel okuma denemesi - CSV gibi işle
                df = self._read_csv_with_manual_skip()
            
            # Metadata oluştur
            metadata = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                'file_size': os.path.getsize(self.file_path),
                'encoding': self.encoding,
                'delimiter': self.delimiter
            }
            
            self.preview_ready.emit(df, metadata)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _needs_line_quote_stripping(self):
        """Check if CSV file has entire lines wrapped in quotes."""
        try:
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                lines_checked = 0
                quote_wrapped_count = 0
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if (line.startswith('"') and line.endswith('"') and 
                        self.delimiter in line and line.count(self.delimiter) > 0):
                        quote_wrapped_count += 1
                    
                    lines_checked += 1
                    if lines_checked >= 5:
                        break
                
                return (quote_wrapped_count / max(lines_checked, 1)) > 0.5
                
        except Exception:
            return False
    
    def _read_csv_with_manual_skip(self):
        """
        Read CSV with manual line selection to properly handle header_row and start_row.
        This handles:
        - Quote-wrapped lines
        - Custom header row position
        - Custom data start row position
        """
        try:
            # Read all necessary lines from file
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                all_lines = []
                for i, line in enumerate(f):
                    line = line.rstrip('\r\n')
                    
                    # Remove wrapping quotes if present
                    if line.startswith('"') and line.endswith('"') and len(line) > 1:
                        line = line[1:-1]
                    
                    all_lines.append(line)
                    
                    # Read enough lines: start_row + 100 data rows
                    if i >= self.start_row + 100:
                        break
            
            # Select lines based on header_row and start_row
            selected_lines = []
            
            if self.has_header:
                # Add header line
                if self.header_row < len(all_lines):
                    selected_lines.append(all_lines[self.header_row])
                else:
                    raise ValueError(f"Header satırı ({self.header_row}) dosya boyutunu aşıyor")
                
                # Add data lines starting from start_row
                data_count = 0
                for i in range(self.start_row, len(all_lines)):
                    selected_lines.append(all_lines[i])
                    data_count += 1
                    if data_count >= 100:  # Preview için 100 satır yeterli
                        break
            else:
                # No header: just take lines from start_row onwards
                for i in range(self.start_row, min(self.start_row + 100, len(all_lines))):
                    selected_lines.append(all_lines[i])
            
            # Join and parse through StringIO
            cleaned_content = '\n'.join(selected_lines)
            
            df = pl.read_csv(
                io.StringIO(cleaned_content),
                separator=self.delimiter,
                has_header=self.has_header,
                skip_rows=0,  # Already manually selected lines
                low_memory=False,
                infer_schema_length=100,
                ignore_errors=True,
                try_parse_dates=False
            )
            
            logger.info(f"Preview with manual skip (header={self.header_row}, start={self.start_row}): {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Manual CSV read failed: {e}")
            # Fallback to simple read
            return pl.read_csv(
                self.file_path,
                encoding=self.encoding,
                separator=self.delimiter,
                n_rows=100,
                low_memory=False,
                quote_char='"',
                infer_schema_length=100,
                ignore_errors=True,
                try_parse_dates=False
            )

class DataImportDialog(QDialog):
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
        
        self._setup_ui()
        self._setup_connections()
        self._load_initial_preview()
        
    def _setup_ui(self):
        """Kullanıcı arayüzünü kur."""
        self.setWindowTitle(f"Veri Import - {os.path.basename(self.file_path)}")
        self.setMinimumSize(1150, 700)
        self.resize(1380, 800)
        
        # Ana layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Üst bilgi paneli - kompakt
        info_frame = self._create_info_panel()
        main_layout.addWidget(info_frame)
        
        # Ana splitter (sol: önizleme, sağ: ayarlar) - ana alanı kaplar
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1)  # Stretch factor = 1
        
        # Sol panel: Veri önizlemesi
        preview_panel = self._create_preview_panel()
        main_splitter.addWidget(preview_panel)
        
        # Sağ panel: Ayarlar
        settings_panel = self._create_settings_panel()
        main_splitter.addWidget(settings_panel)
        
        # Splitter oranları - sol panel daha geniş
        main_splitter.setSizes([650, 350])
        
        # Alt butonlar - kompakt
        button_layout = self._create_button_panel()
        main_layout.addLayout(button_layout)
        
        # Progress bar with status
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        main_layout.addWidget(progress_frame)
        
        # Tema uygula
        self._apply_theme()
        
    def _create_info_panel(self):
        """Üst bilgi paneli oluştur - kompakt."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setMaximumHeight(40)  # Maksimum yükseklik sınırla
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)  # Kompakt margin
        
        # Dosya bilgisi
        self.file_info_label = QLabel(f"📁 {os.path.basename(self.file_path)}")
        self.file_info_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(self.file_info_label)
        
        layout.addStretch()
        
        # Dosya boyutu - kompakt
        try:
            file_size = os.path.getsize(self.file_path)
            size_mb = file_size / (1024 * 1024)
            self.size_label = QLabel(f"📊 {size_mb:.1f} MB")
            self.size_label.setFont(QFont("Arial", 8))
            layout.addWidget(self.size_label)
        except:
            pass
            
        return frame
    def _create_preview_panel(self):
        """Veri önizleme paneli oluştur."""
        group = QGroupBox("📋 Veri Önizlemesi (İlk 100 Satır)")
        layout = QVBoxLayout(group)
        
        # Önizleme tablosu
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QTableWidget.SelectColumns)
        layout.addWidget(self.preview_table)
        
        return group
        
    def _create_settings_panel(self):
        """Ayarlar paneli oluştur."""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)  # Gruplar arası boşluk
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        
        # Dosya format ayarları
        format_group = self._create_format_settings()
        scroll_layout.addWidget(format_group)
        
        # Sütun ayarları
        column_group = self._create_column_settings()
        scroll_layout.addWidget(column_group)
        
        # Satır ayarları
        row_group = self._create_row_settings()
        scroll_layout.addWidget(row_group)
        
        # Zaman kolonu ayarları
        time_group = self._create_time_column_settings()
        scroll_layout.addWidget(time_group)
        
        # Büyük stretch ekle - kalan alanı doldur
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        return scroll_area
        
    def _create_format_settings(self):
        """Dosya format ayarları grubu."""
        group = QGroupBox("🔧 Dosya Format Ayarları")
        group.setMaximumHeight(120)  # Kompakt yükseklik
        layout = QGridLayout(group)
        layout.setSpacing(4)  # Kompakt spacing
        
        # Encoding seçimi
        layout.addWidget(QLabel("Encoding:"), 0, 0)
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems([
            'latin-1', 'utf-8', 'utf-8-sig', 'cp1252', 'cp1254', 'utf-16', 'ascii', 'iso-8859-1', 'iso-8859-9'
        ])
        self.encoding_combo.setCurrentText('latin-1')  # Varsayılan latin-1
        layout.addWidget(self.encoding_combo, 0, 1)
        
        # Delimiter seçimi
        layout.addWidget(QLabel("Ayırıcı:"), 1, 0)
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([
            ', (Virgül)', '; (Noktalı virgül)', '\\t (Tab)', '| (Pipe)', ' (Boşluk)', '. (Nokta)'
        ])
        self.delimiter_combo.setEditable(True)
        layout.addWidget(self.delimiter_combo, 1, 1)
        
        # Otomatik algıla butonu
        self.auto_detect_btn = QPushButton("🔍 Otomatik Algıla")
        self.auto_detect_btn.setToolTip("Dosya formatını otomatik olarak algıla")
        layout.addWidget(self.auto_detect_btn, 2, 0, 1, 2)
        
        return group
        
    def _create_column_settings(self):
        """Sütun ayarları grubu."""
        group = QGroupBox("📊 Header Ayarları")
        group.setMaximumHeight(120)  # Biraz daha yüksek
        layout = QGridLayout(group)
        layout.setSpacing(4)  # Kompakt spacing
        
        # Header satırı
        layout.addWidget(QLabel("Header Satır Numarası:"), 0, 0)
        self.header_spinbox = QSpinBox()
        self.header_spinbox.setRange(-1, 100)
        self.header_spinbox.setValue(0)
        self.header_spinbox.setSpecialValueText("Header Yok")
        self.header_spinbox.setToolTip("Kolon isimlerinin bulunduğu satır numarası (0=ilk satır)")
        layout.addWidget(self.header_spinbox, 0, 1)
        
        # Header var mı checkbox
        self.has_header_checkbox = QCheckBox("Dosyada header var")
        self.has_header_checkbox.setChecked(True)
        layout.addWidget(self.has_header_checkbox, 1, 0, 1, 2)
        
        # Açıklama
        info_label = QLabel("💡 0=İlk satır header, -1=Header yok")
        info_label.setStyleSheet("color: #7fb3d3; font-size: 9px;")
        layout.addWidget(info_label, 2, 0, 1, 2)
        
        return group
        
    def _create_row_settings(self):
        """Satır ayarları grubu."""
        group = QGroupBox("📝 Satır Ayarları")
        group.setMaximumHeight(100)  # Kompakt yükseklik
        layout = QGridLayout(group)
        layout.setSpacing(4)  # Kompakt spacing
        
        # Veri başlangıç satırı (header'dan sonra)
        layout.addWidget(QLabel("Veri Başlangıç Satırı:"), 0, 0)
        self.start_row_spinbox = QSpinBox()
        self.start_row_spinbox.setRange(0, 1000)
        self.start_row_spinbox.setValue(1)  # Header'dan sonra
        self.start_row_spinbox.setToolTip("Header satırından sonra veri hangi satırda başlıyor")
        layout.addWidget(self.start_row_spinbox, 0, 1)
        
        # Açıklama
        info_label = QLabel("💡 Header=0, Veri=1 → Normal CSV")
        info_label.setStyleSheet("color: #7fb3d3; font-size: 9px;")
        layout.addWidget(info_label, 1, 0, 1, 2)
        
        return group
        
    def _create_time_column_settings(self):
        """Zaman kolonu ayarları grubu."""
        group = QGroupBox("⏰ Zaman Kolonu Ayarları")
        # Time column settings daha büyük olabilir ama yine de sınırla
        group.setMaximumHeight(280)  # Biraz daha büyük ama sınırlı
        layout = QGridLayout(group)
        layout.setSpacing(4)  # Kompakt spacing
        
        # Zaman kolonu modu seçimi
        layout.addWidget(QLabel("Zaman Kolonu Modu:"), 0, 0)
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems([
            "Mevcut Kolonu Kullan", 
            "Yeni Zaman Kolonu Oluştur"
        ])
        layout.addWidget(self.time_mode_combo, 0, 1)
        
        # Mevcut zaman kolonu seçimi
        layout.addWidget(QLabel("Mevcut Zaman Kolonu:"), 1, 0)
        self.time_column_combo = QComboBox()
        layout.addWidget(self.time_column_combo, 1, 1)
        
        # Zaman formatı
        layout.addWidget(QLabel("Zaman Formatı:"), 2, 0)
        self.time_format_combo = QComboBox()
        self.time_format_combo.addItems([
            'Otomatik - Kendisi seçsin',
            '%Y-%m-%d %H:%M:%S (2024-01-15 14:30:45)',
            '%d/%m/%Y %H:%M:%S (15/01/2024 14:30:45)',
            '%Y-%m-%d (2024-01-15)',
            '%d/%m/%Y (15/01/2024)',
            'Unix Timestamp (1704110400)',
            'Saniyelik Index (0.0, 0.1, 0.2...)'
        ])
        layout.addWidget(self.time_format_combo, 2, 1)
        
<<<<<<< HEAD
        # Ayırıcı çizgi - yeni zaman kolonu için
        self.new_time_separator = QFrame()
        self.new_time_separator.setFrameShape(QFrame.HLine)
        self.new_time_separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.new_time_separator, 3, 0, 1, 2)
        
        # Yeni zaman kolonu ayarları başlığı
        self.new_time_header_label = QLabel("🕒 Yeni Zaman Kolonu Ayarları")
        layout.addWidget(self.new_time_header_label, 4, 0, 1, 2)
        
        # Örnekleme frekansı
        self.sampling_freq_label = QLabel("Örnekleme Frekansı (Hz):")
        layout.addWidget(self.sampling_freq_label, 5, 0)
=======
        # Ayırıcı çizgi
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator, 3, 0, 1, 2)
        
        # Yeni zaman kolonu ayarları
        layout.addWidget(QLabel("🕒 Yeni Zaman Kolonu Ayarları"), 4, 0, 1, 2)
        
        # Örnekleme frekansı
        layout.addWidget(QLabel("Örnekleme Frekansı (Hz):"), 5, 0)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        self.sampling_freq_spinbox = QSpinBox()
        self.sampling_freq_spinbox.setRange(1, 1000000)  # 1 Hz - 1 MHz
        self.sampling_freq_spinbox.setValue(1)  # Varsayılan 1 Hz
        self.sampling_freq_spinbox.setSuffix(" Hz")
        layout.addWidget(self.sampling_freq_spinbox, 5, 1)
        
        # Başlangıç zamanı
<<<<<<< HEAD
        self.start_time_label = QLabel("Başlangıç Zamanı:")
        layout.addWidget(self.start_time_label, 6, 0)
=======
        layout.addWidget(QLabel("Başlangıç Zamanı:"), 6, 0)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        self.start_time_combo = QComboBox()
        self.start_time_combo.addItems([
            "0 (Sıfırdan Başla)",
            "Şimdiki Zaman", 
            "Özel Zaman"
        ])
        layout.addWidget(self.start_time_combo, 6, 1)
        
        # Özel başlangıç zamanı
<<<<<<< HEAD
        self.custom_start_label = QLabel("Özel Başlangıç:")
        layout.addWidget(self.custom_start_label, 7, 0)
=======
        layout.addWidget(QLabel("Özel Başlangıç:"), 7, 0)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        self.custom_start_time = QLineEdit()
        self.custom_start_time.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.custom_start_time.setEnabled(False)
        layout.addWidget(self.custom_start_time, 7, 1)
        
        # Zaman birimi
<<<<<<< HEAD
        self.time_unit_label = QLabel("Zaman Birimi:")
        layout.addWidget(self.time_unit_label, 8, 0)
=======
        layout.addWidget(QLabel("Zaman Birimi:"), 8, 0)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(['saniye', 'milisaniye', 'mikrosaniye', 'nanosaniye'])
        layout.addWidget(self.time_unit_combo, 8, 1)
        
        # Zaman kolonu adı
<<<<<<< HEAD
        self.new_time_column_label = QLabel("Yeni Kolon Adı:")
        layout.addWidget(self.new_time_column_label, 9, 0)
=======
        layout.addWidget(QLabel("Yeni Kolon Adı:"), 9, 0)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        self.new_time_column_name = QLineEdit()
        self.new_time_column_name.setText("time_generated")
        layout.addWidget(self.new_time_column_name, 9, 1)
        
<<<<<<< HEAD
        # Yeni zaman kolonu widget'larını listeye kaydet (gizleme için)
        self.new_time_widgets = [
            self.new_time_separator,
            self.new_time_header_label,
            self.sampling_freq_label, self.sampling_freq_spinbox,
            self.start_time_label, self.start_time_combo,
            self.custom_start_label, self.custom_start_time,
            self.time_unit_label, self.time_unit_combo,
            self.new_time_column_label, self.new_time_column_name
        ]
        
        # Başlangıçta gizle (varsayılan mod "Mevcut Kolonu Kullan")
        for widget in self.new_time_widgets:
            widget.setVisible(False)
        
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        return group
        
    def _create_button_panel(self):
        """Alt buton paneli oluştur - kompakt."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)  # Kompakt margin
        
        # Önizleme yenile butonu
        self.refresh_btn = QPushButton("🔄 Yenile")
        self.refresh_btn.setToolTip("Ayarları uygulayarak önizlemeyi yenile")
        self.refresh_btn.setMaximumHeight(32)  # Kompakt yükseklik
        layout.addWidget(self.refresh_btn)
        
        layout.addStretch()
        
        # İptal butonu
        self.cancel_btn = QPushButton("❌ İptal")
        self.cancel_btn.setMaximumHeight(32)
        layout.addWidget(self.cancel_btn)
        
        # Import butonu
        self.import_btn = QPushButton("✅ Import Et")
        self.import_btn.setDefault(True)
        self.import_btn.setMaximumHeight(32)
        layout.addWidget(self.import_btn)
        
        return layout
        
    def _setup_connections(self):
        """Sinyal-slot bağlantılarını kur."""
        # Buton bağlantıları
        self.cancel_btn.clicked.connect(self.reject)
        self.import_btn.clicked.connect(self.accept)
        self.refresh_btn.clicked.connect(self._refresh_preview)
        self.auto_detect_btn.clicked.connect(self._auto_detect_format)
        
        # Ayar değişikliklerinde otomatik yenileme
        self.encoding_combo.currentTextChanged.connect(self._on_settings_changed)
        self.delimiter_combo.currentTextChanged.connect(self._on_settings_changed)
        self.header_spinbox.valueChanged.connect(self._on_settings_changed)
        self.start_row_spinbox.valueChanged.connect(self._on_settings_changed)  # start_row
        self.has_header_checkbox.toggled.connect(self._on_header_checkbox_changed)
        
        # Zaman kolonu ayarları
        self.time_mode_combo.currentTextChanged.connect(self._on_time_mode_changed)
        self.start_time_combo.currentTextChanged.connect(self._on_start_time_changed)
        self.sampling_freq_spinbox.valueChanged.connect(self._on_sampling_freq_changed)
        
        # Tablo seçimi
        self.preview_table.itemSelectionChanged.connect(self._on_column_selected)
        
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
        
        # Progress bar göster
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_label.setText("Veri önizlemesi yükleniyor...")
        
        # Thread başlat
        self.preview_thread = DataPreviewThread(
            self.file_path, encoding, delimiter, header_row, start_row
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
            
<<<<<<< HEAD
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
=======
            # Kullanıcıya bilgi ver
            if issues:
                msg = "📊 **Veri Kalite Uyarıları:**\n\n"
                msg += "\n".join(issues[:5])  # İlk 5 sorunu göster
                if len(issues) > 5:
                    msg += f"\n\n... ve {len(issues) - 5} diğer sorun"
                msg += "\n\n✅ Uygulama bu sorunları otomatik olarak düzeltmeye çalışacak."
                msg += "\n\n💡 İsterseniz import ayarlarını değiştirebilirsiniz."
                
                QMessageBox.information(self, "Veri Kalite Kontrolü", msg)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            
            logger.info(f"Veri validasyonu tamamlandı: {len(issues)} sorun, {len(warnings)} uyarı")
            
        except Exception as e:
            logger.warning(f"Veri validasyonu başarısız: {e}")
            # Validasyon hatası import'u engellemez
    
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
        
<<<<<<< HEAD
        # Yeni zaman kolonu widget'larını göster/gizle
        for widget in self.new_time_widgets:
            widget.setVisible(not is_existing_mode)
=======
        # Yeni zaman kolonu ayarlarını etkinleştir/devre dışı bırak
        self.sampling_freq_spinbox.setEnabled(not is_existing_mode)
        self.start_time_combo.setEnabled(not is_existing_mode)
        self.custom_start_time.setEnabled(not is_existing_mode and 
                                         self.start_time_combo.currentText() == "Özel Zaman")
        self.time_unit_combo.setEnabled(not is_existing_mode)
        self.new_time_column_name.setEnabled(not is_existing_mode)
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        
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
        """Dosya formatını otomatik algıla."""
        try:
            # Dosyanın ilk birkaç satırını oku
            with open(self.file_path, 'rb') as f:
                sample = f.read(1024).decode('utf-8', errors='ignore')
                
            # Delimiter algılama
            delimiters = [',', ';', '\t', '|', ' ']
            delimiter_counts = {}
            
            for delim in delimiters:
                count = sample.count(delim)
                delimiter_counts[delim] = count
                
            # En çok bulunan delimiter'ı seç
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            # Delimiter combo'yu güncelle
            delimiter_map = {
                ',': ', (Virgül)',
                ';': '; (Noktalı virgül)',
                '\t': '\\t (Tab)',
                '|': '| (Pipe)',
                ' ': ' (Boşluk)'
            }
            
            if best_delimiter in delimiter_map:
                self.delimiter_combo.setCurrentText(delimiter_map[best_delimiter])
                
            QMessageBox.information(
                self,
                "Otomatik Algılama",
                f"Algılanan ayırıcı: '{best_delimiter}'\nÖnizleme otomatik olarak güncellenecek."
            )
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Algılama Hatası",
                f"Otomatik format algılama başarısız:\n{str(e)}"
            )
            
    def _apply_theme(self):
        """Yumuşak uzay teması - göze rahat."""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2a3441, stop: 1 #1e2832);
                color: #e8eaed;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #5f7c8a;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 4px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3a4a5c, stop: 1 #2a3441);
                color: #e8eaed;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
<<<<<<< HEAD
                padding: 2px 6px 2px 6px;
                color: #1a5276;
                background-color: #e8eaed;
                border-radius: 3px;
                font-weight: 700;
                font-size: 12px;
=======
                padding: 0 4px 0 4px;
                color: #7fb3d3;
                font-weight: 600;
                font-size: 11px;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            }
            QLabel {
                color: #e8eaed;
                font-weight: normal;
                font-size: 11px;
                padding: 1px 2px;
            }
            QTableWidget {
                background-color: #2a3441;
                alternate-background-color: #3a4a5c;
                gridline-color: #5f7c8a;
                selection-background-color: #5f7c8a;
                color: #e8eaed;
                border: 1px solid #5f7c8a;
                border-radius: 4px;
            }
            QTableWidget::item {
                color: #e8eaed;
                padding: 3px 4px;
                font-size: 10px;
            }
            QTableWidget::item:selected {
                background-color: #5f7c8a;
                color: #ffffff;
            }
            QHeaderView::section {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5f7c8a, stop: 1 #4a6270);
                color: #ffffff;
                padding: 4px 6px;
                border: 1px solid #7fb3d3;
                font-weight: 600;
                font-size: 10px;
            }
            QComboBox, QSpinBox, QLineEdit {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3a4a5c, stop: 1 #2a3441);
                border: 1px solid #5f7c8a;
                border-radius: 4px;
                padding: 2px 4px;
                color: #e8eaed;
                font-weight: normal;
                font-size: 11px;
                min-height: 18px;
                max-height: 24px;
            }
            QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
                border-color: #7fb3d3;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a6270, stop: 1 #3a4a5c);
            }
            QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
                border-color: #9fc5e8;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a6270, stop: 1 #3a4a5c);
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border-left: 1px solid #5f7c8a;
            }
            QComboBox::down-arrow {
<<<<<<< HEAD
                image: url(icons/chevron-down.svg);
                width: 12px;
                height: 12px;
=======
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #e8eaed;
                margin-right: 6px;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            }
            QComboBox QAbstractItemView {
                background-color: #2a3441;
                border: 1px solid #5f7c8a;
                border-radius: 4px;
                selection-background-color: #5f7c8a;
                selection-color: #ffffff;
                color: #e8eaed;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5f7c8a, stop: 1 #4a6270);
                border: 1px solid #7fb3d3;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-weight: 600;
                font-size: 11px;
                min-width: 60px;
                min-height: 20px;
                max-height: 28px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #7fb3d3, stop: 1 #5f7c8a);
                border-color: #9fc5e8;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a6270, stop: 1 #3a4a5c);
            }
            QPushButton:default {
                border: 2px solid #7fb3d3;
            }
            QCheckBox {
                color: #e8eaed;
                font-weight: normal;
                font-size: 11px;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5f7c8a;
                border-radius: 2px;
                background-color: #2a3441;
            }
            QCheckBox::indicator:hover {
                border-color: #7fb3d3;
            }
            QCheckBox::indicator:checked {
                background-color: #5f7c8a;
                border: 1px solid #7fb3d3;
<<<<<<< HEAD
                image: url(icons/check.svg);
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            }
            QScrollArea {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2a3441, stop: 1 #1e2832);
                border: 1px solid #5f7c8a;
                border-radius: 4px;
            }
            QFrame[frameShape="4"] {
                color: #5f7c8a;
                background-color: #5f7c8a;
            }
            QProgressBar {
                border: 1px solid #5f7c8a;
                border-radius: 4px;
                background-color: #2a3441;
                color: #e8eaed;
                text-align: center;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #5f7c8a, stop: 1 #7fb3d3);
                border-radius: 3px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a6270, stop: 1 #3a4a5c);
                border: 1px solid #5f7c8a;
<<<<<<< HEAD
                width: 16px;
                height: 12px;
=======
                width: 14px;
                height: 10px;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #5f7c8a;
            }
<<<<<<< HEAD
            QSpinBox::up-arrow {
                image: url(icons/chevron-up.svg);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(icons/chevron-down.svg);
                width: 10px;
                height: 10px;
            }
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        """)
        
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
        time_format_display = self.time_format_combo.currentText()
        time_format_map = {
            'Otomatik - Kendisi seçsin': 'Otomatik',
            '%Y-%m-%d %H:%M:%S (2024-01-15 14:30:45)': '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S (15/01/2024 14:30:45)': '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d (2024-01-15)': '%Y-%m-%d',
            '%d/%m/%Y (15/01/2024)': '%d/%m/%Y',
            'Unix Timestamp (1704110400)': 'Unix Timestamp',
            'Saniyelik Index (0.0, 0.1, 0.2...)': 'Saniyelik Index'
        }
        time_format = time_format_map.get(time_format_display, 'Otomatik')
        
        settings = {
            'file_path': self.file_path,
            'encoding': self.encoding_combo.currentText(),
            'delimiter': delimiter,
            'header_row': self.header_spinbox.value() if self.has_header_checkbox.isChecked() else None,
            'start_row': self.start_row_spinbox.value(),
            'time_mode': time_mode,
            'create_custom_time': is_custom_time
        }
        
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
<<<<<<< HEAD
            time_column_raw = self.time_column_combo.currentText()
            time_column_cleaned = time_column_raw.strip()  # Match converter's cleaning
            settings.update({
                'time_column': time_column_cleaned,  # Use cleaned name for consistency
                'time_column_original': time_column_raw,  # Keep original for debugging
=======
            settings.update({
                'time_column': self.time_column_combo.currentText(),
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
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
