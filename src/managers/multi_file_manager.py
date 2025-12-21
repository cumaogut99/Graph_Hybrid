"""
Multi-File Manager for Time Graph Application

Handles multiple CSV file management with isolated settings and state.
Max 3 files can be open simultaneously for performance.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import QTabWidget, QWidget, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject

logger = logging.getLogger(__name__)


class MultiFileManager(QObject):
    """
    Manages multiple open files with isolated settings and widget states.
    
    Features:
    - Max 3 files simultaneously
    - Independent widget state per file
    - Automatic state save/restore on tab switching
    - File close with unsaved changes warning
    """
    
    # Signals
    file_loaded = Signal(int)  # file_index
    file_switched = Signal(int, int)  # new_index, old_index
    file_closed = Signal(int)  # file_index
    all_files_closed = Signal()
    
    def __init__(self, parent=None, max_files: int = 3):
        super().__init__(parent)
        self.parent = parent
        self.max_files = max_files
        
        # File storage
        self.loaded_files: List[Dict[str, Any]] = []
        self.active_file_index: int = -1
        
        # UI widget
        self.file_tab_widget: Optional[QTabWidget] = None
        
        logger.info(f"MultiFileManager initialized (max files: {max_files})")
    
    def create_file_tab_widget(self) -> QTabWidget:
        """Create and return the file tabs widget."""
        self.file_tab_widget = QTabWidget()
        self.file_tab_widget.setTabsClosable(True)
        self.file_tab_widget.setMovable(False)
        # Status bar yüksekliği ile uyumlu hale getirildi (34 pixel)
        self.file_tab_widget.setMaximumHeight(34)
        self.file_tab_widget.setMinimumHeight(34)
        
        # FİX: Tüm sekmelerin görünür kalmasını sağla
        tab_bar = self.file_tab_widget.tabBar()
        tab_bar.setUsesScrollButtons(False)  # Scroll button'ları devre dışı
        tab_bar.setExpanding(False)  # Sekmelerin genişlemesini engelle
        
        # Initially hidden until first file is loaded
        self.file_tab_widget.setVisible(False)
        
        # Connect signals
        self.file_tab_widget.currentChanged.connect(self._on_tab_changed)
        self.file_tab_widget.tabCloseRequested.connect(self._on_close_requested)
        
        # Styling
        # NOT: MaximumWidth yok - sekmelerin hepsi görünür kalacak
        self.file_tab_widget.setMinimumWidth(400)  # Minimum genişlik garantisi
        
        self.file_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #424242, stop: 1 #353535);
                color: #b0b0b0;
                padding: 6px 14px;
                margin-right: 3px;
                margin-top: 2px;
                border: 1px solid #2a2a2a;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                width: 140px;
                font-size: 10pt;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #6ba3e8, stop: 1 #4a90e2);
                color: #ffffff;
                font-weight: 600;
                border: 1px solid #3d7ec9;
                border-bottom: none;
                margin-top: 0px;
                padding-top: 8px;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #505050, stop: 1 #424242);
                color: #e0e0e0;
                border-color: #555555;
            }
            QTabBar::tab:hover {
                color: #ffffff;
            }
            QTabBar::close-button {
                subcontrol-position: right;
                subcontrol-origin: padding;
                margin: 2px;
                padding: 2px;
                border-radius: 3px;
            }
            QTabBar::close-button:hover {
                background-color: rgba(232, 17, 35, 0.8);
                border-radius: 3px;
            }
            QTabBar::close-button:pressed {
                background-color: rgba(241, 112, 122, 0.9);
            }
        """)
        
        logger.debug("File tab widget created")
        return self.file_tab_widget
    
    def can_add_file(self) -> bool:
        """Check if another file can be added."""
        return len(self.loaded_files) < self.max_files
    
    def is_file_already_open(self, file_path: str) -> int:
        """
        Check if file is already open.
        
        Returns:
            File index if open, -1 otherwise
        """
        for i, file_data in enumerate(self.loaded_files):
            if file_data['file_path'] == file_path:
                return i
        return -1
    
    def _truncate_filename(self, filename: str, max_length: int = 20) -> str:
        """
        Truncate long filenames for display in tabs.
        
        Args:
            filename: Original filename
            max_length: Maximum character length
            
        Returns:
            Truncated filename with ellipsis if needed
        """
        if len(filename) <= max_length:
            return filename
        
        # Dosya uzantısını koru
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        
        # Uzantı ile birlikte max_length'e sığdır
        if ext:
            available = max_length - len(ext) - 4  # 4 = "..." + "."
            if available > 0:
                return f"{name[:available]}...{ext}"
        
        # Uzantısız veya çok kısa ise basit kesme
        return filename[:max_length-3] + "..."
    
    def add_file(self, file_metadata: Dict[str, Any]) -> int:
        """
        Add a new file to the manager.
        
        Args:
            file_metadata: Dict containing file info (file_path, filename, df, etc.)
            
        Returns:
            Index of the added file
        """
        if not self.can_add_file():
            logger.warning(f"Cannot add file: limit reached ({self.max_files})")
            return -1
        
        # Add to list
        self.loaded_files.append(file_metadata)
        new_index = len(self.loaded_files) - 1
        
        # Add tab
        filename = file_metadata.get('filename', f"File {new_index + 1}")
        display_name = self._truncate_filename(filename, max_length=16)
        
        # FİX: currentChanged sinyalini geçici olarak devre dışı bırak
        self.file_tab_widget.blockSignals(True)
        self.file_tab_widget.addTab(QWidget(), display_name)
        
        # Tooltip ile tam dosya adını göster
        self.file_tab_widget.setTabToolTip(new_index, filename)
        
        self.file_tab_widget.setCurrentIndex(new_index)
        self.file_tab_widget.blockSignals(False)
        
        # Show tabs if hidden
        if not self.file_tab_widget.isVisible():
            self.file_tab_widget.setVisible(True)
        
        # Update active index
        old_index = self.active_file_index
        self.active_file_index = new_index
        
        logger.info(f"File added: {filename} (index: {new_index}, total: {len(self.loaded_files)}/{self.max_files})")
        self.file_loaded.emit(new_index)
        
        # Manuel olarak file_switched emit et (sadece değiştiğinde)
        if old_index != new_index:
            logger.info(f"File tab switched (after add): {old_index} -> {new_index}")
            self.file_switched.emit(new_index, old_index)
        
        return new_index
    
    def get_file_data(self, index: int) -> Optional[Dict[str, Any]]:
        """Get file data by index."""
        if 0 <= index < len(self.loaded_files):
            return self.loaded_files[index]
        return None
    
    def get_active_file_data(self) -> Optional[Dict[str, Any]]:
        """Get currently active file data."""
        return self.get_file_data(self.active_file_index)
    
    def update_file_data(self, index: int, key: str, value: Any):
        """Update a specific field in file metadata."""
        if 0 <= index < len(self.loaded_files):
            self.loaded_files[index][key] = value
            logger.debug(f"File {index} updated: {key}")
    
    def save_widget_state(self, index: int, widget_state: Dict[str, Any]):
        """Save widget state for a file."""
        self.update_file_data(index, 'widget_state', widget_state)
    
    def get_widget_state(self, index: int) -> Optional[Dict[str, Any]]:
        """Get saved widget state for a file."""
        file_data = self.get_file_data(index)
        if file_data:
            return file_data.get('widget_state')
        return None
    
    def _on_tab_changed(self, new_index: int):
        """Handle tab change event."""
        if new_index < 0 or new_index >= len(self.loaded_files):
            return
        
        # SORUN: Her zaman file_switched emit ediyordu, sadece değiştiğinde emit etmeli
        if new_index != self.active_file_index:
            old_index = self.active_file_index
            self.active_file_index = new_index
            
            logger.info(f"File tab switched: {old_index} -> {new_index}")
            self.file_switched.emit(new_index, old_index)  # Both indices
        else:
            logger.debug(f"Tab changed but already active: {new_index}")
    
    def _on_close_requested(self, index: int):
        """Handle tab close request."""
        if index < 0 or index >= len(self.loaded_files):
            return
        
        file_data = self.loaded_files[index]
        filename = file_data.get('filename', 'Unknown')
        
        # Check for unsaved changes
        if file_data.get('is_data_modified', False):
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                None,
                "Kaydedilmemiş Değişiklikler",
                f"'{filename}' dosyasında kaydedilmemiş değişiklikler var.\n"
                f"Kapatmak istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Close the file
        self.close_file(index)
    
    def close_file(self, index: int):
        """Close a file by index."""
        if index < 0 or index >= len(self.loaded_files):
            return
        
        filename = self.loaded_files[index].get('filename', 'Unknown')
        logger.info(f"Closing file: {filename} (index: {index})")
        
        # Remove tab
        self.file_tab_widget.removeTab(index)
        
        # Remove from list
        del self.loaded_files[index]
        
        # Update active index
        if len(self.loaded_files) == 0:
            # No files left
            self.active_file_index = -1
            self.file_tab_widget.setVisible(False)
            self.all_files_closed.emit()
            
        elif index == self.active_file_index:
            # Active file was closed, switch to another
            new_index = min(index, len(self.loaded_files) - 1)
            self.file_tab_widget.setCurrentIndex(new_index)
            # _on_tab_changed will be called automatically
        else:
            # Non-active file was closed
            if index < self.active_file_index:
                self.active_file_index -= 1
        
        self.file_closed.emit(index)
        logger.info(f"File closed. Remaining: {len(self.loaded_files)}/{self.max_files}")
    
    def close_all_files(self):
        """Close all open files."""
        while len(self.loaded_files) > 0:
            self.close_file(0)
    
    def get_file_count(self) -> int:
        """Get number of open files."""
        return len(self.loaded_files)
    
    def get_active_index(self) -> int:
        """Get active file index."""
        return self.active_file_index

