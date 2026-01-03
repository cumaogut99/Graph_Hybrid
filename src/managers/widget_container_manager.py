"""
Widget Container Manager - Her dosya için ayrı TimeGraphWidget instance'ı yönetir
"""

import logging
from typing import Dict, Optional
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)


class WidgetContainerManager(QObject):
    """
    Her dosya için ayrı TimeGraphWidget instance'ı yönetir.
    
    Bu sayede:
    - Her dosyanın tamamen bağımsız grafik sekmeleri olur
    - Her dosyanın kendi signal processor'u olur
    - Dosyalar arası hiçbir state paylaşımı olmaz
    - Performans optimize edilir (sadece aktif widget görünür)
    """
    
    # Signals
    widget_switched = Signal(object, object)  # new_widget, old_widget
    
    def __init__(self, parent=None, loading_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.loading_manager = loading_manager
        
        # Widget storage: file_index -> TimeGraphWidget
        self.widgets: Dict[int, object] = {}
        
        # Stacked widget for switching between file widgets
        self.stacked_widget = QStackedWidget()
        
        # Currently active widget
        self.active_widget_index = -1
        
        # Filtre durumu koruma - her widget için ayrı filtre state'i
        self.widget_filter_states: Dict[int, dict] = {}
        
        # Başlangıçta boş bir TimeGraphWidget oluştur (file_index = -1)
        self._create_initial_widget()
        
        logger.info("WidgetContainerManager initialized")
    
    def _create_initial_widget(self):
        """Başlangıçta boş bir TimeGraphWidget oluştur."""
        from time_graph_widget import TimeGraphWidget
        
        # Initial widget (file_index = -1 for initial state)
        self.initial_widget = TimeGraphWidget(parent=self.parent, loading_manager=self.loading_manager)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(self.initial_widget)
        
        logger.info("Initial empty widget created and shown")
    
    def get_stacked_widget(self):
        """Return the QStackedWidget for embedding in main window."""
        return self.stacked_widget
    
    def create_widget_for_file(self, file_index: int):
        """
        Belirtilen dosya için yeni bir TimeGraphWidget instance'ı oluştur.
        
        Args:
            file_index: Dosya indeksi
            
        Returns:
            Oluşturulan widget instance'ı
        """
        # Check if widget already exists
        if file_index in self.widgets:
            logger.warning(f"Widget already exists for file {file_index}, returning existing")
            return self.widgets[file_index]
        
        # Import here to avoid circular dependency
        from time_graph_widget import TimeGraphWidget
        
        # Create new widget instance
        logger.info(f"Creating new TimeGraphWidget for file {file_index}")
        widget = TimeGraphWidget(parent=self.parent, loading_manager=self.loading_manager)
        
        # Store widget
        self.widgets[file_index] = widget
        
        # Add to stacked widget
        self.stacked_widget.addWidget(widget)
        
        logger.info(f"Widget created and added for file {file_index}")
        return widget
    
    def get_widget_for_file(self, file_index: int):
        """
        Belirtilen dosya için widget'ı döndür.
        
        Args:
            file_index: Dosya indeksi
            
        Returns:
            Widget instance veya None
        """
        return self.widgets.get(file_index)
    
    def switch_to_file_widget(self, file_index: int):
        """
        Belirtilen dosyanın widget'ına geç.
        
        Args:
            file_index: Dosya indeksi
        """
        if file_index not in self.widgets:
            logger.error(f"No widget found for file {file_index}")
            return
        
        old_index = self.active_widget_index
        old_widget = self.widgets.get(old_index) if old_index >= 0 else None
        new_widget = self.widgets[file_index]
        
        # İlk widget oluşturulduğunda initial widget'ı gizle
        if hasattr(self, 'initial_widget') and self.initial_widget:
            if self.stacked_widget.indexOf(self.initial_widget) >= 0:
                self.stacked_widget.removeWidget(self.initial_widget)
                self.initial_widget.cleanup()
                self.initial_widget.deleteLater()
                self.initial_widget = None
                logger.debug("Initial widget removed")
        
        # Filtre durumunu koruma: Eski widget'ın filtre durumunu kaydet
        if old_widget and old_index >= 0:
            self._save_widget_filter_state(old_index, old_widget)
        
        # Switch to the widget
        self.stacked_widget.setCurrentWidget(new_widget)
        self.active_widget_index = file_index
        
        # Filtre durumunu geri yükle: Yeni widget'ın filtre durumunu yükle
        self._restore_widget_filter_state(file_index, new_widget)
        
        logger.info(f"Switched from file {old_index} to file {file_index}")
        
        # Emit signal
        self.widget_switched.emit(new_widget, old_widget)
    
    def remove_widget_for_file(self, file_index: int):
        """
        Belirtilen dosyanın widget'ını kaldır ve temizle.
        
        Args:
            file_index: Dosya indeksi
        """
        if file_index not in self.widgets:
            logger.warning(f"No widget to remove for file {file_index}")
            return
        
        widget = self.widgets[file_index]
        
        # Cleanup widget - CRITICAL: Thread temizliği için önce cleanup yap
        if hasattr(widget, 'cleanup'):
            try:
                logger.debug(f"Starting widget cleanup for file {file_index}")
                widget.cleanup()
                
                # Thread'lerin tamamen temizlendiğinden emin ol
                import time
                time.sleep(0.1)  # Kısa bekleme süresi
                
                logger.debug(f"Widget cleanup completed for file {file_index}")
            except Exception as e:
                logger.warning(f"Error during widget cleanup for file {file_index}: {e}")
        
        # Remove from stacked widget
        widget_index = self.stacked_widget.indexOf(widget)
        if widget_index >= 0:
            self.stacked_widget.removeWidget(widget)
        
        # Delete widget - CRITICAL: deleteLater() çağrıldıktan sonra widget'a erişmeyin
        if widget is not None:
            widget.deleteLater()
        
        # Remove from storage
        del self.widgets[file_index]
        
        # Filtre durumunu da temizle
        if file_index in self.widget_filter_states:
            del self.widget_filter_states[file_index]
        
        logger.info(f"Widget removed for file {file_index}")
        
        # Update active index if needed
        if self.active_widget_index == file_index:
            self.active_widget_index = -1
    
    def get_active_widget(self):
        """
        Şu anda aktif olan widget'ı döndür.
        
        Returns:
            Aktif widget veya None
        """
        if self.active_widget_index >= 0:
            return self.widgets.get(self.active_widget_index)
        return None
    
    def cleanup_all(self):
        """Tüm widget'ları temizle."""
        logger.info("Cleaning up all widgets")
        
        # Make a copy of keys to avoid modification during iteration
        file_indices = list(self.widgets.keys())
        
        for file_index in file_indices:
            self.remove_widget_for_file(file_index)
        
        # Filtre durumlarını da temizle
        self.widget_filter_states.clear()
        
        # Tüm widget'lar temizlendikten sonra initial widget'ı tekrar göster
        if not hasattr(self, 'initial_widget') or not self.initial_widget:
            self._create_initial_widget()
        else:
            self.stacked_widget.setCurrentWidget(self.initial_widget)
        
        logger.info("All widgets cleaned up")
    
    def _save_widget_filter_state(self, file_index: int, widget):
        """
        Widget'ın filtre durumunu kaydet.
        
        Args:
            file_index: Dosya indeksi
            widget: Widget instance'ı
        """
        try:
            if hasattr(widget, 'filter_manager'):
                filter_manager = widget.filter_manager
                if hasattr(filter_manager, 'get_active_filters'):
                    active_filters = filter_manager.get_active_filters()
                    self.widget_filter_states[file_index] = active_filters.copy()
                    logger.debug(f"Saved filter state for file {file_index}: {len(active_filters)} filters")
                else:
                    logger.debug(f"No get_active_filters method for file {file_index}")
            else:
                logger.debug(f"No filter_manager for file {file_index}")
        except Exception as e:
            logger.warning(f"Error saving filter state for file {file_index}: {e}")
    
    def _restore_widget_filter_state(self, file_index: int, widget):
        """
        Widget'ın filtre durumunu geri yükle.
        
        Args:
            file_index: Dosya indeksi
            widget: Widget instance'ı
        """
        try:
            if file_index not in self.widget_filter_states:
                logger.debug(f"No saved filter state for file {file_index}")
                return
            
            saved_state = self.widget_filter_states[file_index]
            if not saved_state:
                logger.debug(f"Empty filter state for file {file_index}")
                return
            
            if hasattr(widget, 'filter_manager'):
                filter_manager = widget.filter_manager
                if hasattr(filter_manager, 'active_filters'):
                    # Filtre durumunu geri yükle
                    filter_manager.active_filters = saved_state.copy()
                    filter_manager.filter_applied = bool(saved_state)
                    
                    # UI'da filtre durumunu güncelle
                    if hasattr(widget, '_restore_filter_ui_state'):
                        widget._restore_filter_ui_state(saved_state)
                    
                    logger.info(f"Restored filter state for file {file_index}: {len(saved_state)} filters")
                else:
                    logger.debug(f"No active_filters attribute for file {file_index}")
            else:
                logger.debug(f"No filter_manager for file {file_index}")
        except Exception as e:
            logger.warning(f"Error restoring filter state for file {file_index}: {e}")
    
    def save_current_filter_state(self):
        """Şu anda aktif olan widget'ın filtre durumunu kaydet."""
        if self.active_widget_index >= 0:
            active_widget = self.widgets.get(self.active_widget_index)
            if active_widget:
                self._save_widget_filter_state(self.active_widget_index, active_widget)
    
    def get_widget_count(self):
        """Açık widget sayısını döndür."""
        return len(self.widgets)

