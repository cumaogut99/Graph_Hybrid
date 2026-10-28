"""
Graph tabs and the number and order of graphs in a tab.

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging
from typing import Optional
from PyQt5.QtCore import QTimer
from src.graphics.graph_container import GraphContainer
from src.utils.resources import icon_url

logger = logging.getLogger(__name__)


class TabsMixin:
    """Graph tabs and the number and order of graphs in a tab."""

    def _on_graph_count_changed(self, count: int):
        """Delegates graph count change to the active tab's container."""
        active_container = self.get_active_graph_container()
        if active_container:
            # ✅ FIX: Grafik sayısı değiştiğinde graph_signal_mapping'i ÖNCE güncelle
            # Yeni grafikler için boş mapping ekle, mevcut eşlemeleri koru
            # ✅ FIX: Get tab index from GraphContainer instead of tab_widget
            current_tab = active_container.tab_index if hasattr(active_container, 'tab_index') else self.tab_widget.currentIndex()
            if current_tab not in self.graph_signal_mapping:
                self.graph_signal_mapping[current_tab] = {}
            
            # Yeni grafikler için boş liste ekle (mevcut grafikleri koru)
            for i in range(count):
                if i not in self.graph_signal_mapping[current_tab]:
                    self.graph_signal_mapping[current_tab][i] = []
            
            # Mevcut grafik sayısından fazla olan mapping'leri temizle
            # UYARI: Bu sinyalleri mapping'den kaldırıyor ama veriyi kaybetmiyor
            # Çünkü veriler self.signal_processor içinde tutuluyor
            to_remove = [k for k in self.graph_signal_mapping[current_tab].keys() if k >= count]
            for key in to_remove:
                del self.graph_signal_mapping[current_tab][key]
            
            # Graph signal mapping updated for current tab
            
            # SONRA grafik sayısını değiştir (bu işlem graph_signal_mapping'i kullanacak)
            active_container.set_graph_count(count)
            
            # Use delayed initialization to ensure plots are fully ready
            QTimer.singleShot(100, self._delayed_post_graph_change)
            # Rebuild the graph settings panel immediately
            self.graph_settings_panel_manager.rebuild_controls(count)

    def _delayed_post_graph_change(self):
        """Delayed initialization after graph count change to ensure plots are ready."""
        active_container = self.get_active_graph_container()
        if not active_container:
            return
        count = active_container.plot_manager.get_subplot_count()
        
        # Store current cursor mode before reinitializing
        current_mode = getattr(self, 'current_cursor_mode', 'dual')
        
        # Re-initialize cursors after plots are fully ready
        self._initialize_cursor_manager()
        
        # Ensure cursor mode is preserved and applied
        if self.cursor_manager and current_mode:
            self.cursor_manager.set_mode(current_mode)
            logger.info(f"Restored cursor mode to '{current_mode}' after graph count change")
        
        # Force cursor mode to match toolbar selection
        self._force_cursor_mode_sync()
        
        # Apply saved graph settings after graph count change
        self._apply_saved_graph_settings()
        
        # Recreate the stats panel to match the new number of graphs
        self._recreate_statistics_panel()
        # DON'T update statistics here - wait for cursor movement for performance

    def _apply_tab_stylesheet(self):
        """Apply a modern stylesheet to the tab widget."""
        colors = self.theme_manager.get_theme_colors()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border-top: 2px solid {colors.get('primary', '#4a90e2')};
                background: {colors.get('surface', '#2d2d2d')};
            }}
            QTabBar::tab {{
                background: {colors.get('surface_variant', '#3c3c3c')};
                color: {colors.get('text_secondary', '#e0e0e0')};
                border: 1px solid {colors.get('surface', '#2d2d2d')};
                border-bottom-color: {colors.get('primary', '#4a90e2')}; 
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 100px;
                padding: 8px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background: {colors.get('surface', '#2d2d2d')};
                color: {colors.get('text_primary', '#ffffff')};
                font-weight: bold;
            }}
            QTabBar::close-button {{
                image: url({icon_url('x')});
                subcontrol-position: right;
                subcontrol-origin: padding;
                margin: 1px;
                padding: 2px;
                border-radius: 3px;
                width: 14px;
                height: 14px;
            }}
            QTabBar::close-button:hover {{
                background-color: rgba(232, 17, 35, 0.8);
            }}
            QTabBar::close-button:pressed {{
                background-color: rgba(241, 112, 122, 0.9);
            }}
            QToolButton {{
                background-color: {colors.get('surface_variant', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {colors.get('surface', '#2d2d2d')};
            }}
            QToolButton:hover {{
                background-color: {colors.get('primary', '#4a90e2')};
            }}
        """)

    def _on_tab_count_changed(self, count: int):
        """Handle tab count changes from toolbar."""
        # This method is now deprecated and should not be used.
        pass

    def _on_tab_changed(self, index: int):
        """Handle tab switching."""
        #logger.debug(f"Switched to tab {index}")
        # Update graph count on toolbar to reflect the new active tab
        active_container = self.get_active_graph_container()
        if active_container:
            graph_count = active_container.plot_manager.get_subplot_count()
            self.toolbar_manager.set_graph_count(graph_count)
            # Rebuild graph settings panel when tab changes
            self.graph_settings_panel_manager.rebuild_controls(graph_count)
        self.settings_panel_manager.update_marker_list()

        # Use delayed initialization for tab changes too
        QTimer.singleShot(50, self._delayed_post_tab_change)

    def _delayed_post_tab_change(self):
        """Delayed initialization after tab change to ensure plots are ready."""
        # Store current cursor mode before reinitializing
        current_mode = getattr(self, 'current_cursor_mode', 'dual')
        
        # Cursors are initialized when tabs change to handle different plot widgets
        self._initialize_cursor_manager()
        
        # Ensure cursor mode is preserved and applied
        if self.cursor_manager and current_mode:
            self.cursor_manager.set_mode(current_mode)
            logger.info(f"Restored cursor mode to '{current_mode}' after tab change")
        
        # Force cursor mode to match toolbar selection
        self._force_cursor_mode_sync()
        
        # Apply saved graph settings for the new tab
        self._apply_saved_graph_settings()
        
        # Update statistics for the new tab
        self._recreate_statistics_panel()

    def _add_tab(self, name: Optional[str] = None):
        """Add a new tab with a GraphContainer."""
        if not name:
            name = f"Tab {self.tab_widget.count() + 1}"
        
        # Get the tab index (will be the current count before adding)
        tab_index = self.tab_widget.count()
        
        # Create a new graph container for the tab, passing self as the main_widget and tab_index
        graph_container = GraphContainer(self.theme_manager, main_widget=self, tab_index=tab_index)
        graph_container.signal_processor = self.signal_processor  # Assign signal processor
        
        # Connect the settings button signal from the new container's plot manager
        
        # Connect drag-drop signal to plot column on graph
        graph_container.column_dropped.connect(self._on_column_dropped_on_graph)

        # Markers of the active tab are listed in the general settings panel
        graph_container.plot_manager.markers_changed.connect(
            self.settings_panel_manager.update_marker_list)
        
        self.graph_containers.append(graph_container)
        
        # Add the container as a new tab
        self.tab_widget.addTab(graph_container, name)
        self.tab_widget.setCurrentWidget(graph_container)
        
        # Update tab count in toolbar
        self.toolbar_manager.set_graph_count(self.tab_widget.count())
        
        # Apply the current theme to the new container's plots
        graph_container.apply_theme()
        
        return graph_container

    def _rename_tab(self, index):
        """Rename a tab when it is double-clicked."""
        from PyQt5.QtWidgets import QInputDialog, QLineEdit

        old_name = self.tab_widget.tabText(index)
        
        # Create styled input dialog
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Sekmeyi Yeniden Adlandır")
        dialog.setLabelText("Yeni sekme adı:")
        dialog.setTextValue(old_name)
        dialog.setInputMode(QInputDialog.TextInput)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QInputDialog {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5ba0f2;
            }
            QPushButton:pressed {
                background-color: #3a80d2;
            }
        """)
        
        ok = dialog.exec_()
        new_name = dialog.textValue()
        
        if ok and new_name and new_name.strip():
            self.tab_widget.setTabText(index, new_name.strip())
            logger.info(f"Tab {index} renamed from '{old_name}' to '{new_name.strip()}'")

    def _remove_tab(self, index: int):
        """Remove a tab at a given index."""
        if self.tab_widget.count() > 1:
            widget_to_remove = self.tab_widget.widget(index)
            self.tab_widget.removeTab(index)
            
            if widget_to_remove in self.graph_containers:
                self.graph_containers.remove(widget_to_remove)
            
            # Clean up the removed widget
            widget_to_remove.deleteLater()

            # Update remaining tab titles
            for i in range(self.tab_widget.count()):
                self.tab_widget.setTabText(i, f"Tab {i + 1}")

    def get_active_graph_container(self) -> Optional['GraphContainer']:
        """Gets the GraphContainer from the currently active tab."""
        if not hasattr(self, 'tab_widget') or not self.tab_widget:
            return None
        current_index = self.tab_widget.currentIndex()
        if 0 <= current_index < len(self.graph_containers):
            return self.graph_containers[current_index]
        return None

    def _on_graph_reorder_requested(self, from_index: int, to_index: int):
        """Handle graph reorder request from statistics panel."""
        logger.info(f"Graph reorder requested: Graph {from_index + 1} -> Graph {to_index + 1}")
        tab_index = self.tab_widget.currentIndex()
        if tab_index < 0:
            return
        
        # Get active container
        active_container = self.get_active_graph_container()
        if not active_container:
            logger.warning("No active container found for graph reordering")
            return
        
        count = active_container.plot_manager.get_subplot_count()
        if from_index == to_index or not (0 <= from_index < count and 0 <= to_index < count):
            return

        # The plot widgets stay where they are; their CONTENTS are swapped.
        # Everything per graph (callbacks, axis labels, cursors, statistics,
        # legends) is addressed by graph index, so swapping the per-index
        # state and redrawing keeps all of it consistent. Moving the widgets
        # instead left those index-bound parts pointing at the wrong graph.
        def swap(store: dict):
            if store is None:
                return
            a, b = store.pop(from_index, None), store.pop(to_index, None)
            if b is not None:
                store[from_index] = b
            if a is not None:
                store[to_index] = a

        swap(self.graph_signal_mapping.setdefault(tab_index, {}))
        swap(self.graph_settings.get(tab_index))

        swap(active_container.plot_manager.original_data_ranges)

        self.is_data_modified = True

        cursor_positions = (self.cursor_manager.get_cursor_positions()
                            if self.cursor_manager else {})

        # Redraws signals, legends, limit lines and filters, restores the
        # cursors and rebuilds the statistics panel
        self._redraw_all_signals()

        # The rebuilt statistics panel is empty until the next cursor move;
        # the values existed before the swap, so recompute them once the
        # redraw re-enables statistics (~350 ms, see _restore_cursors_after_redraw)
        if cursor_positions:
            QTimer.singleShot(450, lambda: self._refresh_statistics(cursor_positions))

        logger.info(f"Graphs reordered successfully: Graph {from_index + 1} <-> Graph {to_index + 1}")

    def _on_add_tab_clicked(self):
        """Handle the click event for adding a new tab."""
        self._add_tab()

    def _on_remove_tab_clicked(self):
        """Handle the click event for removing the current tab."""
        self._remove_tab()
