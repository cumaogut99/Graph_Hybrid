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
    QButtonGroup, QPushButton, QListWidgetItem, QSpinBox, QSlider, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.exporters
import polars as pl

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class SettingsPanelManager(QObject):
    """Manages the settings panel interface for the Time Graph Widget."""
    
    # Signals
    theme_changed = Signal(str)
    export_format_changed = Signal(str)
    
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
        self._create_export_settings(main_layout)
        
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

    # Cursor and Performance settings removed - functionality moved to Graph Settings panel
    
    def _create_export_settings(self, parent_layout):
        """Create export settings section."""
        # Plot Export Section
        plot_group = QGroupBox("📈 Plot Export")
        plot_layout = QFormLayout(plot_group)
        plot_layout.setSpacing(8)
        
        self.plot_format_combo = QComboBox()
        self.plot_format_combo.addItems(["PNG", "PDF"])  # SVG removed
        self.plot_format_combo.setCurrentText("PNG")
        self.plot_format_combo.currentTextChanged.connect(self.export_format_changed.emit)
        plot_layout.addRow("Format:", self.plot_format_combo)
        
        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self._export_plot)
        plot_layout.addRow(export_plot_btn)
        
        parent_layout.addWidget(plot_group)
        
        # Data Export Section
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
    
    def _export_plot(self):
        """Export plot to file."""
        logger.info("Export plot requested")
        
        format_type = self.plot_format_combo.currentText().lower()
        
        # Get active tab container from parent
        if hasattr(self.parent, 'tab_widget') and self.parent.tab_widget.count() > 0:
            if format_type == "png":
                self._export_png_per_tab()
            elif format_type == "pdf":
                self._export_pdf_multi_page()
        else:
            logger.warning("No active tabs found for export")
    
    def _export_png_per_tab(self):
        """Export PNG for each tab separately."""
        from PyQt5.QtWidgets import QFileDialog
        import os
        
        # Get directory to save files
        directory = QFileDialog.getExistingDirectory(
            self.parent, 
            "PNG Export Klasörü Seçin",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not directory:
            return
            
        tab_count = self.parent.tab_widget.count()
        for i in range(tab_count):
            tab_name = self.parent.tab_widget.tabText(i)
            # Clean tab name for filename
            safe_name = "".join(c for c in tab_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_tab_{i+1}.png"
            filepath = os.path.join(directory, filename)
            
            # Switch to tab and export
            self.parent.tab_widget.setCurrentIndex(i)
            container = self.parent.graph_containers[i]
            
            if hasattr(container, 'plot_manager'):
                # Pass the entire plot_manager to the export function
                self._export_tab_plots_as_png(container.plot_manager, filepath)
                
        logger.info(f"PNG files exported to {directory}")
    
    def _export_pdf_multi_page(self):
        """Export PDF with each tab as separate page."""
        from PyQt5.QtWidgets import QFileDialog
        from PyQt5.QtPrintSupport import QPrinter
        from PyQt5.QtGui import QPainter
        from PyQt5.QtCore import QRectF
        
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self.parent,
            "PDF Export Dosyası",
            "time_graph_export.pdf",
            "PDF Files (*.pdf)"
        )
        
        if not filepath:
            return
            
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filepath)
        printer.setPageSize(QPrinter.A4)
        
        painter = QPainter()
        painter.begin(printer)
        
        tab_count = self.parent.tab_widget.count()
        for i in range(tab_count):
            if i > 0:
                printer.newPage()
                
            # Switch to tab
            self.parent.tab_widget.setCurrentIndex(i)
            container = self.parent.graph_containers[i]
            
            if hasattr(container, 'plot_manager') and container.plot_manager.plot_widgets:
                self._render_tab_to_pdf(container.plot_manager.plot_widgets, painter, printer)
                
        painter.end()
        logger.info(f"PDF exported to {filepath}")
    
    def _export_tab_plots_as_png(self, plot_manager, filepath):
        """Export the entire plot container as a single PNG."""
        if not plot_manager or not hasattr(plot_manager, 'plot_container'):
            logger.warning("Plot manager or plot container not found for PNG export")
            return

        # The widget to export is the container of all plot widgets
        widget_to_export = plot_manager.plot_container
        
        if not widget_to_export:
            logger.warning("Plot container widget is not available.")
            return

        # Grab the widget's contents into a QPixmap
        pixmap = widget_to_export.grab()
        
        # Save the pixmap to a file
        if not pixmap.save(filepath, "PNG"):
            logger.error(f"Failed to save PNG to {filepath}")
    
    def _render_tab_to_pdf(self, plot_widgets, painter, printer):
        """Render tab plots to PDF page."""
        if not plot_widgets:
            return
            
        # For now, render the first plot widget
        # TODO: Layout multiple plot widgets on page
        plot_widget = plot_widgets[0]
        
        # Get the plot widget's pixmap
        pixmap = plot_widget.grab()
        
        # Scale to fit page
        page_rect = printer.pageRect()
        scaled_pixmap = pixmap.scaled(
            page_rect.size().toSize(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Center on page
        x = (page_rect.width() - scaled_pixmap.width()) // 2
        y = (page_rect.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x, y, scaled_pixmap)
    
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
        
        if 'plot_format' in settings:
            self.plot_format_combo.setCurrentText(settings['plot_format'])
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings values."""
        return {
            'theme': self.theme_combo.currentText(),
            'plot_format': self.plot_format_combo.currentText()
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
            QComboBox, QSpinBox, QLineEdit {{
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
