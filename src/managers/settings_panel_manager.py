"""
Settings Panel Manager for Time Graph Widget

Manages the settings panel interface including:
- Graph display settings
- Analysis parameters
- Export options
- Theme settings
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGroupBox, QComboBox, QListWidget, QLineEdit, QCheckBox, QRadioButton,
    QButtonGroup, QPushButton, QListWidgetItem, QSpinBox, QSlider, QFormLayout,
    QMenu
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QFont
import polars as pl

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class SettingsPanelManager(QObject):
    """Manages the settings panel interface for the Time Graph Widget."""
    
    # Signals
    theme_changed = Signal(str)
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.settings_panel = None
        
        self._setup_settings_panel()
    
    def _setup_settings_panel(self):
        """Create the main settings panel."""
        self.settings_panel = QFrame()
        self.settings_panel.setMinimumWidth(280)
        self.settings_panel.setMaximumWidth(350)
        # Apply theme-based styling - will be updated dynamically
        self._apply_theme_styling()
        
        # Main layout in a scroll area for scalability
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Create settings sections
        self._create_display_settings(main_layout)
        self._create_marker_settings(main_layout)
        self._create_data_export_settings(main_layout)
        
        main_layout.addStretch()
        
        scroll_area.setWidget(container_widget)
        
        # Set the scroll area as the main layout for the panel
        panel_layout = QVBoxLayout(self.settings_panel)
        panel_layout.setContentsMargins(0,0,0,0)
        panel_layout.addWidget(scroll_area)
    
    def _create_display_settings(self, parent_layout):
        """Create display settings section."""
        group = QGroupBox("📊 Display")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignLeft)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Space"])
        self.theme_combo.setCurrentText("Space")  # Space is default
        self.theme_combo.currentTextChanged.connect(self.theme_changed.emit)
        layout.addRow("Theme:", self.theme_combo)

        parent_layout.addWidget(group)

    def _create_marker_settings(self, parent_layout):
        """Markers of the active graph tab (added from the plot's right-click menu)."""
        group = QGroupBox("📍 Markers")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.marker_hint_label = QLabel("Right-click a graph → Add Marker")
        self.marker_hint_label.setStyleSheet("font-style: italic; color: #888888;")
        layout.addWidget(self.marker_hint_label)

        self.marker_list = QListWidget()
        self.marker_list.setMaximumHeight(160)
        self.marker_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.marker_list.customContextMenuRequested.connect(self._on_marker_context_menu)
        self.marker_list.setVisible(False)  # shown once a marker exists
        layout.addWidget(self.marker_list)

        parent_layout.addWidget(group)

    def update_marker_list(self):
        """Show the markers of the active graph tab."""
        self.marker_list.clear()
        container = self.parent.get_active_graph_container()
        plot_manager = getattr(container, 'plot_manager', None)
        markers = plot_manager.get_markers() if plot_manager else []
        for marker in markers:
            text = f"Marker {marker['number']}  —  {plot_manager.format_x_value(marker['x'])}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, marker['number'])
            self.marker_list.addItem(item)
        self.marker_list.setVisible(bool(markers))
        self.marker_hint_label.setVisible(not markers)

    def _on_marker_context_menu(self, pos):
        item = self.marker_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self.marker_list)
        remove_action = menu.addAction("Remove")
        if menu.exec_(self.marker_list.mapToGlobal(pos)) == remove_action:
            container = self.parent.get_active_graph_container()
            if container is not None:
                # The plot manager's markers_changed signal refreshes the list
                container.plot_manager.remove_marker(item.data(Qt.UserRole))

    # Cursor and Performance settings removed - functionality moved to Graph Settings panel
    
    def _create_data_export_settings(self, parent_layout):
        """Create the cursor-range data export section."""
        data_group = QGroupBox("📊 Data Export")
        data_layout = QFormLayout(data_group)
        data_layout.setSpacing(8)
        
        # Info label for cursor range
        cursor_info_label = QLabel("Exports data between two cursors")
        cursor_info_label.setStyleSheet("font-style: italic; color: #888888;")
        data_layout.addRow(cursor_info_label)
        
        export_data_btn = QPushButton("Export CSV Data")
        export_data_btn.clicked.connect(self._export_data)
        data_layout.addRow(export_data_btn)
        
        parent_layout.addWidget(data_group)
    
    def _export_data(self):
        """Export data between two cursors as CSV."""
        logger.info("Export data requested")
        
        # Get current active tab
        if not hasattr(self.parent, 'tab_widget') or self.parent.tab_widget.count() == 0:
            logger.warning("No active tabs found for data export")
            return
            
        current_index = self.parent.tab_widget.currentIndex()
        container = self.parent.graph_containers[current_index]
        
        # Check if cursor manager exists and has dual cursors
        if not hasattr(container, 'cursor_manager') or not container.cursor_manager:
            logger.warning("No cursor manager found")
            return
            
        cursor_positions = container.cursor_manager.get_cursor_positions()
        
        if len(cursor_positions) < 2:
            self._show_styled_message_box(
                "Cursor Hatası",
                "İki cursor arası veri export etmek için dual cursor modunda iki cursor yerleştirin.",
                "warning"
            )
            return
            
        # Get cursor positions
        cursor1_pos = cursor_positions.get('c1') or cursor_positions.get('cursor1')
        cursor2_pos = cursor_positions.get('c2') or cursor_positions.get('cursor2')
        
        if cursor1_pos is None or cursor2_pos is None:
            logger.warning("Could not get both cursor positions")
            return
            
        # Ensure cursor1 is the smaller value
        start_pos = min(cursor1_pos, cursor2_pos)
        end_pos = max(cursor1_pos, cursor2_pos)
        
        # Get file path for saving
        from PyQt5.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self.parent,
            "CSV Export Dosyası",
            f"cursor_data_{start_pos:.3f}_{end_pos:.3f}.csv",
            "CSV Files (*.csv)"
        )
        
        if not filepath:
            return
            
        # Export data between cursors
        self._export_cursor_range_data(container, start_pos, end_pos, filepath)
    
    def _export_cursor_range_data(self, container, start_pos, end_pos, filepath):
        """Export data between cursor positions to CSV."""
        try:
            import pandas as pd
            import numpy as np
            
            # Get signal data from the container
            if not hasattr(container, 'signal_processor') or not container.signal_processor:
                logger.warning("No signal processor found")
                return
                
            all_signals = container.signal_processor.get_all_signals()
            
            if not all_signals:
                logger.warning("No signals found for export")
                return
                
            # Prepare data dictionary
            export_data = {}
            
            # Process each signal
            for signal_name, signal_data in all_signals.items():
                x_data = np.array(signal_data.get('x_data', []))
                y_data = np.array(signal_data.get('y_data', []))
                
                if len(x_data) == 0 or len(y_data) == 0:
                    continue
                    
                # Find indices within cursor range
                mask = (x_data >= start_pos) & (x_data <= end_pos)
                
                if not np.any(mask):
                    continue
                    
                # Extract data within range
                x_range = x_data[mask]
                y_range = y_data[mask]
                
                # Add to export data
                if len(export_data) == 0:
                    export_data['time'] = x_range
                    
                export_data[signal_name] = y_range
            
            if not export_data:
                self._show_styled_message_box(
                    "Veri Hatası",
                    "Seçilen cursor aralığında veri bulunamadı.",
                    "warning"
                )
                return
                
            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Data exported to {filepath}")
            
            # Show success message
            self._show_styled_message_box(
                "Export Başarılı",
                f"Cursor aralığı ({start_pos:.3f} - {end_pos:.3f}) verileri başarıyla export edildi:\n{filepath}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting cursor range data: {e}")
            self._show_styled_message_box(
                "Export Hatası",
                f"Veri export edilirken hata oluştu:\n{str(e)}",
                "critical"
            )

    def _show_styled_message_box(self, title: str, text: str, icon_type: str = "information"):
        """Display a QMessageBox with theme-aware styling."""
        from PyQt5.QtWidgets import QMessageBox

        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(text)

        if icon_type == "information":
            msg_box.setIcon(QMessageBox.Information)
        elif icon_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
        elif icon_type == "critical":
            msg_box.setIcon(QMessageBox.Critical)

        # Get theme colors for styling
        theme_colors = {}
        if hasattr(self.parent, 'theme_manager'):
            theme_colors = self.parent.theme_manager.get_theme_colors()
        
        background = theme_colors.get('surface', '#333333')
        text_color = theme_colors.get('text_primary', '#FFFFFF')
        button_bg = theme_colors.get('primary', '#4a90e2')
        button_border = theme_colors.get('border', '#5a5a5a')

        msg_box.setStyleSheet(f"""
            QMessageBox {{
                background-color: {background};
            }}
            QMessageBox QLabel {{
                color: {text_color};
                font-size: 11px;
            }}
            QMessageBox QPushButton {{
                background-color: {button_bg};
                color: white;
                border: 1px solid {button_border};
                padding: 5px 15px;
                border-radius: 4px;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {theme_colors.get('hover', '#5a9eee')};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: {theme_colors.get('selected', '#3a80d2')};
            }}
        """)
        
        msg_box.exec_()
    
    def get_settings_panel(self) -> QWidget:
        """Get the settings panel widget."""
        return self.settings_panel
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update settings from external source."""
        if 'theme' in settings:
            self.theme_combo.setCurrentText(settings['theme'])
        
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings values."""
        return {
            'theme': self.theme_combo.currentText()
        }
    
    def _apply_theme_styling(self):
        """Apply current theme styling to the settings panel."""
        # Get theme colors from parent's theme manager
        if hasattr(self.parent, 'theme_manager'):
            theme_colors = self.parent.theme_manager.get_theme_colors()
        else:
            # Fallback to space theme colors
            theme_colors = {
                'background': '#1a2332',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'primary': '#4a90e2',
                'text_primary': '#e6f3ff',
                'text_secondary': '#ffffff',
                'border': '#4a90e2'
            }
        
        self.settings_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_colors['surface']};
                border: 1px solid {theme_colors['border']};
                border-radius: 8px;
            }}
            QGroupBox {{
                color: {theme_colors['text_primary']};
                font-weight: bold;
                font-size: 11px;
                border: 1px solid {theme_colors['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: {theme_colors['surface_variant']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {theme_colors['primary']};
            }}
            QLabel {{
                color: {theme_colors['text_secondary']};
                font-size: 10px;
            }}
            QCheckBox, QRadioButton {{
                color: {theme_colors['text_secondary']};
                font-size: 10px;
                spacing: 5px;
            }}
            QComboBox, QSpinBox, QLineEdit, QListWidget {{
                border: 1px solid {theme_colors['border']};
                border-radius: 4px;
                padding: 4px;
                background-color: {theme_colors['background']};
                color: {theme_colors['text_primary']};
                font-size: 10px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QPushButton {{
                background-color: {theme_colors['primary']};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme_colors.get('hover', theme_colors['primary'])};
            }}
            QPushButton:pressed {{
                background-color: {theme_colors.get('selected', theme_colors['primary'])};
            }}
        """)
    
    def update_theme(self):
        """Update the panel styling when theme changes."""
        self._apply_theme_styling()
