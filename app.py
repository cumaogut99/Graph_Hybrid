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

# Load the polars binary that fits this CPU (the compatible one on
# Pentium/Celeron without AVX2) before anything imports polars
try:
    from src.utils.polars_runtime import select_polars_runtime
    select_polars_runtime()
except Exception:
    pass

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

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QMessageBox, QStatusBar, QMenuBar, QAction, QSplashScreen, QDialog
from PyQt5.QtCore import pyqtSignal as Signal, QObject
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
    from src.ui.time_graph_widget import TimeGraphWidget
    from src.data.data_import_dialog import DataImportDialog
    from src.managers.status_bar_manager import StatusBarManager
    from src.graphics.loading_overlay import LoadingManager
    # Multi-file support
    from src.managers.multi_file_manager import MultiFileManager
    from src.data.data_loader import DataLoader
    from src.managers.widget_container_manager import WidgetContainerManager
    # Project file support (.mpai)
    from src.managers.project_file_manager import ProjectFileManager
    # The toolbar doubles as the title bar
    from src.ui.frameless_window import FramelessWindowMixin
    # Parts of TimeGraphApp
    from src.ui.main_window.loading import FileLoadingMixin
    from src.ui.main_window.saving import SavingMixin
    from src.ui.main_window.files import OpenFilesMixin
    from src.ui.main_window.shutdown import ShutdownMixin
except ImportError as e:
    print(f"Import hatası: {e}")
    print("Lütfen tüm gerekli modüllerin mevcut olduğundan emin olun.")
    sys.exit(1)

# Setup logging - only errors are logged. Lower levels (DEBUG/INFO/WARNING)
# filled the log file with routine messages (one line per loaded signal etc.).
_log_handlers = [logging.FileHandler('time_graph_app.log', encoding='utf-8')]
if sys.stderr is not None:  # windowed PyInstaller builds have no stderr
    _log_handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers,
    force=True,  # replace any handlers an imported module may have set up
)
logger = logging.getLogger(__name__)


class TimeGraphApp(FileLoadingMixin, SavingMixin, OpenFilesMixin, ShutdownMixin, FramelessWindowMixin, QMainWindow):
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
        self.file_manager = MultiFileManager(self)
        
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
        # Never larger than the screen (e.g. 1280x800 at 200% scaling)
        available = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        if available is not None:
            self.setMinimumSize(min(1200, available.width()), min(800, available.height()))
        else:
            self.setMinimumSize(1200, 800)
        # Native title bar removed: the toolbar is the title bar (Windows)
        self._init_frameless()
        
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
        from src.operations.data_operations import DataOperations
        from src.operations.project_operations import ProjectOperations
        from src.operations.layout_operations import LayoutOperations
        
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
        
    # Widget state save/restore artık gerekli değil - her dosyanın kendi widget instance'ı var
    
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
