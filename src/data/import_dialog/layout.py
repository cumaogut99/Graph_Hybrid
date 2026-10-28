"""
Building the dialog: panels, setting groups, buttons and signal connections.

Part of DataImportDialog; moved from src/data/data_import_dialog.py without changes.
"""

import os
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSplitter, QTableWidget, QGroupBox, QLabel, QComboBox, QSpinBox, QPushButton, QCheckBox, QProgressBar, QFrame, QScrollArea, QGridLayout, QLineEdit, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from src.data.import_dialog.constants import TIME_FORMAT_CHOICES


class ImportDialogLayoutMixin:
    """Building the dialog: panels, setting groups, buttons and signal connections."""

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
        group.setMaximumHeight(150)  # Kompakt yükseklik
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

        # Ondalık ayırıcı virgül mü (0,25)
        self.decimal_comma_checkbox = QCheckBox("Ondalık ayırıcı virgül (0,25)")
        self.decimal_comma_checkbox.setToolTip(
            "Sayılar 0,25 gibi virgülle yazılmışsa işaretleyin.\n"
            "Ayırıcı virgül ise kullanılamaz."
        )
        layout.addWidget(self.decimal_comma_checkbox, 2, 0, 1, 2)

        # Otomatik algıla butonu
        self.auto_detect_btn = QPushButton("🔍 Otomatik Algıla")
        self.auto_detect_btn.setToolTip(
            "Ayırıcı, header satırı, veri başlangıç satırı ve ondalık ayırıcıyı algıla"
        )
        layout.addWidget(self.auto_detect_btn, 3, 0, 1, 2)

        # Excel sayfa seçimi (sadece Excel dosyalarında görünür)
        self.sheet_label = QLabel("Sayfa:")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setToolTip("İçe aktarılacak Excel sayfası")
        layout.addWidget(self.sheet_label, 4, 0)
        layout.addWidget(self.sheet_combo, 4, 1)
        self.sheet_label.setVisible(False)
        self.sheet_combo.setVisible(False)
        self.format_group = group

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
        self.header_spinbox.setObjectName("stepper")
        self.header_spinbox.setAlignment(Qt.AlignCenter)
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
        self.start_row_spinbox.setObjectName("stepper")
        self.start_row_spinbox.setAlignment(Qt.AlignCenter)
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
        self.time_format_combo.addItems(list(TIME_FORMAT_CHOICES))
        self.time_format_combo.setToolTip(
            "Otomatik: sayılar, tarih+saat ve tarihsiz saatler tanınır\n"
            "(14:30:45.123, 9:05:00 PM, 14/30/45, 05:30.5, 1h2m3s).\n"
            "Gün/ay sırası belirsiz tarihlerde (01/02/2024) formatı seçin.")
        layout.addWidget(self.time_format_combo, 2, 1)
        
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
        self.sampling_freq_spinbox = QSpinBox()
        self.sampling_freq_spinbox.setRange(1, 1000000)  # 1 Hz - 1 MHz
        self.sampling_freq_spinbox.setValue(1)  # Varsayılan 1 Hz
        self.sampling_freq_spinbox.setSuffix(" Hz")
        layout.addWidget(self.sampling_freq_spinbox, 5, 1)
        
        # Başlangıç zamanı
        self.start_time_label = QLabel("Başlangıç Zamanı:")
        layout.addWidget(self.start_time_label, 6, 0)
        self.start_time_combo = QComboBox()
        self.start_time_combo.addItems([
            "0 (Sıfırdan Başla)",
            "Şimdiki Zaman", 
            "Özel Zaman"
        ])
        layout.addWidget(self.start_time_combo, 6, 1)
        
        # Özel başlangıç zamanı
        self.custom_start_label = QLabel("Özel Başlangıç:")
        layout.addWidget(self.custom_start_label, 7, 0)
        self.custom_start_time = QLineEdit()
        self.custom_start_time.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.custom_start_time.setEnabled(False)
        layout.addWidget(self.custom_start_time, 7, 1)
        
        # Zaman birimi
        self.time_unit_label = QLabel("Zaman Birimi:")
        layout.addWidget(self.time_unit_label, 8, 0)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(['saniye', 'milisaniye', 'mikrosaniye', 'nanosaniye'])
        layout.addWidget(self.time_unit_combo, 8, 1)
        
        # Zaman kolonu adı
        self.new_time_column_label = QLabel("Yeni Kolon Adı:")
        layout.addWidget(self.new_time_column_label, 9, 0)
        self.new_time_column_name = QLineEdit()
        self.new_time_column_name.setText("time_generated")
        layout.addWidget(self.new_time_column_name, 9, 1)
        
        # Yeni zaman kolonu widget'larını listeye kaydet (gizleme için)
        self.new_time_widgets = [
            self.new_time_separator,
            self.new_time_header_label,
            self.sampling_freq_label, self.sampling_freq_spinbox,
            self.start_time_label, self.start_time_combo,
            self.custom_start_label, self.custom_start_time,
            self.new_time_column_label, self.new_time_column_name
        ]
        # The unit converts an existing numeric time column (e.g. ms -> s);
        # a generated column is always in seconds
        self.time_unit_label.setText("Zaman Kolonu Birimi:")
        self.time_unit_combo.setToolTip("Sayısal zaman kolonunun birimi; saniyeye çevrilir")
        
        # Başlangıçta gizle (varsayılan mod "Mevcut Kolonu Kullan")
        for widget in self.new_time_widgets:
            widget.setVisible(False)
        
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
        self.decimal_comma_checkbox.toggled.connect(self._on_settings_changed)
        self.header_spinbox.valueChanged.connect(self._on_header_row_changed)
        self.start_row_spinbox.valueChanged.connect(self._on_settings_changed)  # start_row
        self.has_header_checkbox.toggled.connect(self._on_header_checkbox_changed)
        
        # Zaman kolonu ayarları
        self.time_mode_combo.currentTextChanged.connect(self._on_time_mode_changed)
        self.start_time_combo.currentTextChanged.connect(self._on_start_time_changed)
        self.sampling_freq_spinbox.valueChanged.connect(self._on_sampling_freq_changed)
        
        # Tablo seçimi
        self.preview_table.itemSelectionChanged.connect(self._on_column_selected)
