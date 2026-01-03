#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Graph Widget - Bağımsız Uygulama
=====================================

Bu uygulama, time graph widget'ını bağımsız bir masaüstü uygulaması olarak çalıştırır.
Veri analizi ve görselleştirme için gelişmiş araçlar sunar.

Özellikler:
- Çoklu grafik desteği
- Gerçek zamanlı istatistikler
- Tema desteği
- Veri dışa/içe aktarma
- Gelişmiş cursor araçları
"""

import sys
import os

# FIX: Add Arrow DLLs to PATH before importing C++ module
# First, add local lib/ folder where we moved Arrow/Parquet DLLs
_app_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_app_dir, 'lib')
if os.path.exists(_lib_dir):
    os.environ['PATH'] = _lib_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_lib_dir)
        except Exception:
            pass

# Then add pyarrow's library directories
try:
    import pyarrow
    arrow_lib_dirs = pyarrow.get_library_dirs()
    if arrow_lib_dirs:
        for lib_dir in arrow_lib_dirs:
            if os.path.exists(lib_dir):
                # Add to PATH (legacy)
                os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
                # Add to DLL search path (Python 3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(lib_dir)
                    except Exception:
                        pass
except ImportError:
    pass  # PyArrow not installed, C++ module may still work without Arrow

import logging
import json
from typing import Optional
from datetime import datetime
import numpy as np
import polars as pl

# PyQt5 High DPI Desteği - QApplication oluşturulmadan ÖNCE
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# High DPI scaling etkinleştir (çift ekran desteği için kritik)
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# High DPI policy ayarla (Qt 5.14+, çift ekran DPI farklılıkları için)
try:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
except AttributeError:
    pass  # Qt 5.14'ten önceki versiyonlarda yok

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QMessageBox, 
    QFileDialog, QStatusBar, QMenuBar, QAction, QSplashScreen, QDialog
)
from PyQt5.QtCore import QTimer, pyqtSignal as Signal, QObject, QThread
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont

# Import our time graph widget and dialog
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable OpenGL for performance
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True)

# PyQtGraph patch'i uygula - autoRangeEnabled AttributeError sorununu çöz
try:
    from src.utils.pyqtgraph_patch import apply_pyqtgraph_patch
    apply_pyqtgraph_patch()
except ImportError as e:
    print(f"PyQtGraph patch yüklenemedi: {e}")
    # Patch yüklenemese bile devam et

try:
    # ORİJİNAL VERSİYON - Kararlı ve test edilmiş
    from time_graph_widget import TimeGraphWidget  # Orijinal stabil versiyon
    from src.data.data_import_dialog import DataImportDialog
    from src.managers.status_bar_manager import StatusBarManager
    from src.graphics.loading_overlay import LoadingManager
    # Multi-file support
    from src.managers.multi_file_manager import MultiFileManager
    from src.data.data_loader import DataLoader
    from src.managers.widget_container_manager import WidgetContainerManager
    # Project file support (.mpai)
    from src.managers.project_file_manager import ProjectFileManager
except ImportError as e:
    print(f"Import hatası: {e}")
    print("Lütfen tüm gerekli modüllerin mevcut olduğundan emin olun.")
    sys.exit(1)

# Setup logging - WARNING level for most modules, DEBUG only for cursor
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('time_graph_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cursor debug logging disabled - 4-cursor issue is resolved
# logging.getLogger('src.managers.cursor_manager').setLevel(logging.DEBUG)

# Enable INFO logging for data operations to track loading/cancellation
logging.getLogger('app.data_operations').setLevel(logging.INFO)
logging.getLogger('src.data.data_loader').setLevel(logging.INFO)
logging.getLogger('src.data.csv_to_mpai_converter').setLevel(logging.INFO)

# ✅ REFACTORED: OldDataLoader sınıfı kaldırıldı (445 satır)
# DataLoader artık src.data.data_loader modülünden import ediliyor (satır 69)
# Bu, kod tekrarını önler ve bakımı kolaylaştırır.

class DataSaver(QObject):
    """Veri kaydetme işlemlerini ayrı bir thread'de yürüten worker."""
    finished = Signal(str)  # Kaydedilen dosya yolunu gönderir
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, df, file_path, file_filter):
        super().__init__()
        self.df = df
        self.file_path = file_path
        self.file_filter = file_filter

    def run(self):
        """Veri kaydetme işlemini başlatır."""
        try:
            self._save_data()
            self.finished.emit(self.file_path)
        except Exception as e:
            self.error.emit(str(e))

    def _save_data(self):
        """Asıl veri kaydetme mantığı."""
        # Dosya formatına göre kaydet
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.csv' or 'CSV' in self.file_filter:
            self.df.write_csv(self.file_path)
        elif file_ext == '.xlsx' or 'Excel' in self.file_filter:
            self.df.write_excel(self.file_path)
        else:
            # Varsayılan CSV
            self.df.write_csv(self.file_path)


class TimeGraphApp(QMainWindow):
    """Ana uygulama penceresi."""
    
    def __init__(self):
        super().__init__()
        self.current_file_path = None
        self.is_data_modified = False
        
        # Threading members
        self.load_threads = []  # Track all threads for proper cleanup
        self.load_worker = None
        self.save_thread = None
        self.save_worker = None

        # Initialize managers
        self.status_bar_manager = None
        self.loading_manager = None
        
        # Multi-file manager
        self.file_manager = MultiFileManager(self, max_files=3)
        
        # Widget container manager - HER DOSYA İÇİN AYRI WİDGET
        self.widget_container_manager = None
        
        # Project file manager (.mpai)
        self.project_manager = ProjectFileManager(self)
        
        # ✅ REFACTORED: Operations modülleri (Delegation Pattern)
        # app/ klasöründen import edilecek
        self.data_ops = None
        self.project_ops = None
        self.layout_ops = None
        
        self._setup_loading_manager()
        self._setup_ui()
        self._setup_connections()
        self._setup_status_bar()
        
        # ✅ REFACTORED: Initialize operations modules
        self._setup_operations_modules()
        
        # Multi-file manager connections
        self._setup_file_manager_connections()
        
        # Uygulama başlangıç mesajı
        logger.info("Time Graph Uygulaması başlatıldı")
        
    def _setup_ui(self):
        """Kullanıcı arayüzünü kurulum."""
        self.setWindowTitle("Time Graph - Veri Analizi ve Görselleştirme")
        self.setMinimumSize(1200, 800)
        
        # Pencereyi ekrana sığacak şekilde boyutlandır (High DPI için uyarlanmış)
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            # Ekranın %85'ini kullan (taskbar ve kenar boşluklarına yer bırak)
            target_width = int(screen_geometry.width() * 0.85)
            target_height = int(screen_geometry.height() * 0.85)
            self.resize(target_width, target_height)
            
            # Pencereyi ekranın ortasına yerleştir
            self.move(
                screen_geometry.left() + (screen_geometry.width() - target_width) // 2,
                screen_geometry.top() + (screen_geometry.height() - target_height) // 2
            )
        else:
            # Fallback: Ekran bilgisi alınamazsa varsayılan boyut
            self.resize(1400, 900)
        
        # Widget container manager'ı oluştur - HER DOSYA İÇİN AYRI WİDGET
        self.widget_container_manager = WidgetContainerManager(self, self.loading_manager)
        
        # Stacked widget'ı ana widget olarak ayarla
        self.setCentralWidget(self.widget_container_manager.get_stacked_widget())
        
        # Initial widget için sinyal bağlantılarını kur
        if hasattr(self.widget_container_manager, 'initial_widget') and self.widget_container_manager.initial_widget:
            self._connect_widget_signals(self.widget_container_manager.initial_widget)
        
        # Pencere ikonunu ayarla (eğer varsa)
        try:
            # EXE için resource path'i kontrol et
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller ile paketlenmiş durumda
                icon_path = os.path.join(sys._MEIPASS, 'ikon.png')
            else:
                # Geliştirme ortamında
                icon_path = 'ikon.png'
            
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                # Fallback ikon yolu
                fallback_path = "icons/app_icon.png"
                if os.path.exists(fallback_path):
                    self.setWindowIcon(QIcon(fallback_path))
        except Exception as e:
            logger.debug(f"İkon yüklenemedi: {e}")
            pass  # İkon yoksa devam et
            
        # Pencereyi ekranın ortasına yerleştir
        self._center_window()
        
    def _center_window(self):
        """Pencereyi ekranın ortasına yerleştir."""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        
    def _setup_connections(self):
        """Sinyal-slot bağlantılarını kur."""
        # Widget container manager'dan widget oluşturulduğunda bağlantıları yapacağız
        # İlk widget oluşturulduğunda _connect_widget_signals çağrılacak
        pass
    
    def _connect_widget_signals(self, widget):
        """Bir widget için sinyal-slot bağlantılarını kur."""
        if widget and hasattr(widget, 'toolbar_manager'):
            toolbar = widget.toolbar_manager
            
            # File menü bağlantıları
            if hasattr(toolbar, 'file_open_requested'):
                toolbar.file_open_requested.connect(self._on_file_open)
            if hasattr(toolbar, 'file_save_requested'):
                toolbar.file_save_requested.connect(self._on_file_save)
            if hasattr(toolbar, 'file_exit_requested'):
                toolbar.file_exit_requested.connect(self._on_file_exit)
            if hasattr(toolbar, 'layout_import_requested'):
                toolbar.layout_import_requested.connect(self._on_layout_import)
            if hasattr(toolbar, 'layout_export_requested'):
                toolbar.layout_export_requested.connect(self._on_layout_export)
            # Project file operations (.mpai)
            if hasattr(toolbar, 'project_save_requested'):
                toolbar.project_save_requested.connect(self._on_project_save)
            if hasattr(toolbar, 'project_open_requested'):
                toolbar.project_open_requested.connect(self._on_project_open)
                
        # Veri değişikliği sinyali
        if widget:
            widget.data_changed.connect(self._on_data_changed)
            
            # Tema değişikliği sinyali - status bar'ı güncelle
            if hasattr(widget, 'theme_manager'):
                widget.theme_manager.theme_changed.connect(self._on_theme_changed)
        
        logger.debug("Widget signals connected")
    
    def _setup_file_manager_connections(self):
        """Multi-file manager sinyal bağlantılarını kur."""
        self.file_manager.file_switched.connect(self._on_file_switched)
        self.file_manager.file_closed.connect(self._on_file_closed)
        self.file_manager.all_files_closed.connect(self._on_all_files_closed)
        self.file_manager.save_project_requested.connect(self._on_save_project_before_close)
            
    def _setup_status_bar(self):
        """Durum çubuğunu kur."""
        # Create custom status bar with system monitoring
        self.status_bar_manager = StatusBarManager(self)
        self.setStatusBar(self.status_bar_manager)
        
        # Store reference for compatibility
        self.status_bar = self.status_bar_manager
        
        # === DOSYA SEKMELERİNİ STATUS BAR'A EKLE ===
        # Dosya sekmelerini oluştur
        file_tab_widget = self.file_manager.create_file_tab_widget()
        
        # Status bar'ın başına (soluna) ekle
        self.status_bar_manager.insertPermanentWidget(0, file_tab_widget)
        
        logger.debug("File tabs added to status bar")
    
    def _setup_operations_modules(self):
        """
        Operations modüllerini initialize et (Delegation Pattern).
        
        ✅ REFACTORED: app.py'deki metotlar artık bu modüllere delegate ediliyor.
        """
        from app.data_operations import DataOperations
        from app.project_operations import ProjectOperations
        from app.layout_operations import LayoutOperations
        
        self.data_ops = DataOperations(self)
        self.project_ops = ProjectOperations(self)
        self.layout_ops = LayoutOperations(self)
        
        logger.info("Operations modules initialized (DataOperations, ProjectOperations, LayoutOperations)")
    
    def _setup_loading_manager(self):
        """Loading manager'ı kur."""
        self.loading_manager = LoadingManager(self)
        
        # Başlangıçta tema renklerini ayarla
        self._update_status_bar_theme()
    
    def _update_status_bar_theme(self):
        """Status bar tema renklerini güncelle."""
        try:
            active_widget = self.widget_container_manager.get_active_widget()
            if self.status_bar_manager and active_widget and hasattr(active_widget, 'theme_manager'):
                theme_colors = active_widget.theme_manager.get_theme_colors()
                self.status_bar_manager.update_theme(theme_colors)
        except Exception as e:
            logger.debug(f"Could not update status bar theme at startup: {e}")
        
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
            
            # Başarılı yükleme
            self.current_file_path = file_path
            self.is_data_modified = False
            
            self.setWindowTitle(f"Time Graph Widget - {filename}")
            
            self.status_bar.showMessage(
                f"Dosya yüklendi: {filename} ({row_count:,} satır, {col_count} sütun)", 
                10000
            )
            
            logger.info(f"Dosya başarıyla yüklendi: {file_path} ({row_count} satır, {col_count} sütun)")
            logger.info(f"Toplam açık dosya: {self.file_manager.get_file_count()}/{self.file_manager.max_files}")
            
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
            
    def _on_file_exit(self):
        """Uygulamadan çıkış."""
        self.close()
        
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
    
    # Widget state save/restore artık gerekli değil - her dosyanın kendi widget instance'ı var
    
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
    
    def _on_theme_changed(self, theme_name: str):
        """Tema değişikliği olayı - status bar'ı güncelle."""
        try:
            active_widget = self.widget_container_manager.get_active_widget()
            if self.status_bar_manager and active_widget and hasattr(active_widget, 'theme_manager'):
                # Tema renklerini al ve status bar'ı güncelle
                theme_colors = active_widget.theme_manager.get_theme_colors()
                self.status_bar_manager.update_theme(theme_colors)
                logger.debug(f"Status bar theme updated to: {theme_name}")
        except Exception as e:
            logger.error(f"Error updating status bar theme: {e}")
            
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
        
        # Switch to this file's widget
        self.widget_container_manager.switch_to_file_widget(file_index)
        
        # Trigger save dialog
        if self.project_ops:
            # After successful save, mark as saved and close
            success = self.project_ops.save_project_dialog()
            if success:
                # Mark as saved
                file_data['is_project_saved'] = True
                # Now close the tab (temp cleanup will happen automatically)
                self.file_manager.close_file(file_index)
        else:
            logger.error("ProjectOperations module not initialized")
    
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
                        
                    if hasattr(widget, 'graph_renderer') and widget.graph_renderer:
                        try:
                            deviation_threads = [t for t in widget.graph_renderer.deviation_threads.values() if t and t.isRunning()]
                        except (RuntimeError, AttributeError):
                            deviation_threads = []
                        qthread_count += len(deviation_threads)
                        if deviation_threads:
                            logger.info(f"- {len(deviation_threads)} deviation thread for file {file_index} hala çalışıyor")
        if hasattr(self, 'status_bar_manager') and self.status_bar_manager:
            if hasattr(self.status_bar_manager, 'monitor_thread') and self.status_bar_manager.monitor_thread:
                try:
                    if self.status_bar_manager.monitor_thread.isRunning():
                        qthread_count += 1
                        logger.info("- Monitor thread hala çalışıyor")
                except RuntimeError:
                    pass  # Thread already deleted
        
        logger.info(f"Toplam QThread sayısı: {qthread_count}")

def create_splash_screen():
    """Başlangıç ekranı oluştur."""
    splash_pix = QPixmap(400, 300)
    splash_pix.fill(Qt.darkBlue)
    
    painter = QPainter(splash_pix)
    painter.setPen(Qt.white)
    
    # Başlık
    font = QFont("Arial", 16, QFont.Bold)
    painter.setFont(font)
    painter.drawText(splash_pix.rect(), Qt.AlignCenter, 
                    "Time Graph Widget\n\nVeri Analizi ve Görselleştirme\n\nYükleniyor...")
    
    painter.end()
    
    splash = QSplashScreen(splash_pix)
    splash.setMask(splash_pix.mask())
    return splash

def main():
    """Ana uygulama fonksiyonu."""
    # QApplication oluştur
    app = QApplication(sys.argv)
    app.setApplicationName("Time Graph Widget")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Data Analysis Tools")
    
    # Başlangıç ekranı göster
    splash = create_splash_screen()
    splash.show()
    app.processEvents()
    
    try:
        # Ana pencereyi oluştur
        splash.showMessage("Ana pencere oluşturuluyor...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        main_window = TimeGraphApp()
        
        splash.showMessage("Arayüz hazırlanıyor...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        # Pencereyi göster
        main_window.show()
        
        # Başlangıç ekranını kapat
        splash.finish(main_window)
        
        # Uygulama döngüsünü başlat
        sys.exit(app.exec_())
        
    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        logger.error(f"Uygulama başlatılırken hata oluştu: {e}")
        logger.error(f"Full traceback: {full_error}")
        print(f"FULL ERROR TRACEBACK:\n{full_error}")
        
        # Hata mesajı göster
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Başlatma Hatası")
        error_msg.setText(f"Uygulama başlatılamadı:\n\n{str(e)}")
        error_msg.exec_()
        
        sys.exit(1)

if __name__ == "__main__":
    main()
