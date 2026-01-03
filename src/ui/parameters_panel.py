"""
Parameters Panel - Signal selection and visibility management
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QCheckBox, QGroupBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class ParametersPanel(QWidget):
    """Panel for parameter/signal selection and visibility management with deferred loading."""
    
    signals_changed = pyqtSignal(list)  # Emits list of selected signals

    def __init__(self, all_signals: List[str], visible_signals: List[str] = None, parent=None):
        super().__init__(parent)
        self.all_signals = all_signals if all_signals else []
        self.visible_signals = visible_signals if visible_signals else []
        
        # PERFORMANCE: Deferred loading için
        self._initial_load_count = 50  # İlk yüklenen item sayısı
        self._items_fully_loaded = False
        
        # PERFORMANCE: Search debouncing
        from PyQt5.QtCore import QTimer
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)  # 200ms debounce
        self._search_timer.timeout.connect(self._apply_search_filter)
        self._pending_search_text = ""
        
        self._setup_ui()
        self._setup_connections()
        
        # PERFORMANCE: Geri kalan itemleri arka planda yükle (100ms sonra)
        if len(self.all_signals) > self._initial_load_count:
            QTimer.singleShot(100, self._load_remaining_items)
            logger.info(f"Deferred loading: {self._initial_load_count} items now, {len(self.all_signals) - self._initial_load_count} items in 100ms")
        
    def _setup_ui(self):
        """Setup the parameters panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Parameters")
        title.setStyleSheet("""
            QLabel {
                color: #e6f3ff;
                font-size: 18px;
                font-weight: 700;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.1));
                border-radius: 6px;
                border: 1px solid rgba(74, 144, 226, 0.3);
            }
        """)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select which parameters/signals to display in the graph. Use the search bar to quickly find specific signals.")
        desc.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                margin-bottom: 10px;
                padding: 8px;
            }
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search signals...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.1), stop:1 transparent);
                color: #e6f3ff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #4a90e2;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 rgba(74, 144, 226, 0.05));
            }
        """)
        layout.addWidget(self.search_bar)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.invert_selection_btn = QPushButton("Invert Selection")
        
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 transparent);
                color: #e6f3ff;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.1));
                border-color: #4a90e2;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 rgba(74, 144, 226, 0.3));
            }
        """
        
        for btn in [self.select_all_btn, self.deselect_all_btn, self.invert_selection_btn]:
            btn.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.select_all_btn)
        controls_layout.addWidget(self.deselect_all_btn)
        controls_layout.addWidget(self.invert_selection_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Signal list
        self.signal_list = QListWidget()
        self.signal_list.setStyleSheet("""
            QListWidget {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 4px;
                color: #e6f3ff;
                selection-background-color: rgba(74, 144, 226, 0.5);
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(74, 144, 226, 0.2);
            }
            QListWidget::item:hover {
                background: rgba(74, 144, 226, 0.2);
            }
        """)
        
        # Populate signal list
        self._populate_signal_list()
        
        layout.addWidget(self.signal_list)
        
        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 11px;
                padding: 5px;
            }
        """)
        self._update_stats()
        layout.addWidget(self.stats_label)
        
    def _populate_signal_list(self):
        """Populate the signal list with DEFERRED LOADING for large datasets."""
        # PERFORMANCE: Disable updates during batch operation
        self.signal_list.setUpdatesEnabled(False)
        self.signal_list.clear()
        
        # PERFORMANCE: Batch add items - much faster
        visible_set = set(self.visible_signals)  # O(1) lookup instead of O(n)
        
        # PERFORMANCE: İlk sadece N item yükle (hızlı açılış için)
        signals_to_load = self.all_signals[:self._initial_load_count] if not self._items_fully_loaded else self.all_signals
        
        for signal in signals_to_load:
            item = QListWidgetItem(signal)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if signal in visible_set else Qt.Unchecked)
            self.signal_list.addItem(item)
        
        # PERFORMANCE: Re-enable updates and refresh once
        self.signal_list.setUpdatesEnabled(True)
        
        # Show loading indicator if more items to load
        if not self._items_fully_loaded and len(self.all_signals) > self._initial_load_count:
            loading_item = QListWidgetItem(f"[Loading {len(self.all_signals) - self._initial_load_count} more items...]")
            loading_item.setFlags(Qt.ItemIsEnabled)  # Not checkable
            loading_item.setForeground(Qt.yellow)
            self.signal_list.addItem(loading_item)
    
    def _load_remaining_items(self):
        """Load remaining items after initial display (deferred loading)."""
        try:
            logger.info(f"Loading remaining {len(self.all_signals) - self._initial_load_count} items...")
            
            # Remove loading indicator
            last_item = self.signal_list.item(self.signal_list.count() - 1)
            if last_item and "[Loading" in last_item.text():
                self.signal_list.takeItem(self.signal_list.count() - 1)
            
            # Disable updates for batch operation
            self.signal_list.setUpdatesEnabled(False)
            
            visible_set = set(self.visible_signals)
            
            # Add remaining items
            for signal in self.all_signals[self._initial_load_count:]:
                item = QListWidgetItem(signal)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if signal in visible_set else Qt.Unchecked)
                self.signal_list.addItem(item)
            
            self._items_fully_loaded = True
            
            # Re-enable updates
            self.signal_list.setUpdatesEnabled(True)
            
            # Update stats
            self._update_stats()
            
            logger.info(f"Successfully loaded all {len(self.all_signals)} items")
            
        except Exception as e:
            logger.error(f"Error loading remaining items: {e}")
            
    def _setup_connections(self):
        """Setup signal connections."""
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        self.invert_selection_btn.clicked.connect(self._invert_selection)
        self.signal_list.itemChanged.connect(self._on_item_changed)
        self.search_bar.textChanged.connect(self._filter_signals)
        
    def _select_all(self):
        """Select all signals."""
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setCheckState(Qt.Checked)
        self._update_stats()
        self._emit_selection_changed()
        
    def _deselect_all(self):
        """Deselect all signals."""
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self._update_stats()
        self._emit_selection_changed()
        
    def _invert_selection(self):
        """Invert signal selection."""
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            current_state = item.checkState()
            new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
        self._update_stats()
        self._emit_selection_changed()
        
    def _on_item_changed(self, item):
        """Handle item check state change."""
        self._update_stats()
        self._emit_selection_changed()
        
    def _update_stats(self):
        """Update selection statistics."""
        total = self.signal_list.count()
        selected = len(self.get_selected_signals())
        self.stats_label.setText(f"Selected: {selected} / {total} signals")
        
    def _emit_selection_changed(self):
        """Emit signal when selection changes."""
        selected = self.get_selected_signals()
        self.signals_changed.emit(selected)
        
    def _filter_signals(self, text: str):
        """Filter signals based on search text with debouncing."""
        # PERFORMANCE: Debounce - sadece kullanıcı yazmayı bıraktıktan sonra filtrele
        self._pending_search_text = text
        self._search_timer.stop()
        self._search_timer.start()
    
    def _apply_search_filter(self):
        """Apply the search filter (called after debounce delay)."""
        text = self._pending_search_text
        
        if not text:
            # Boş search - tümünü göster (hızlı yol)
            for i in range(self.signal_list.count()):
                self.signal_list.item(i).setHidden(False)
            self._update_stats()
            return
        
        # PERFORMANCE: Batch updates - UI güncellemeyi devre dışı bırak
        self.signal_list.setUpdatesEnabled(False)
        
        text_lower = text.lower()
        visible_count = 0
        
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            matches = text_lower in item.text().lower()
            item.setHidden(not matches)
            if matches:
                visible_count += 1
        
        # PERFORMANCE: UI güncellemeyi tekrar aç
        self.signal_list.setUpdatesEnabled(True)
        
        # Update stats for filtered view
        self.stats_label.setText(f"Showing {visible_count} of {len(self.all_signals)} signals (filtered)")
        
    def get_selected_signals(self) -> List[str]:
        """Get list of selected signals."""
        selected = []
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected
        
    def set_selected_signals(self, signals: List[str]):
        """Set which signals are selected."""
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in signals else Qt.Unchecked)
        self._update_stats()
        
    def update_available_signals(self, signals: List[str]):
        """Update the list of available signals."""
        self.all_signals = signals
        selected = self.get_selected_signals()  # Preserve current selection
        self._populate_signal_list()
        self.set_selected_signals(selected)  # Restore selection where possible
