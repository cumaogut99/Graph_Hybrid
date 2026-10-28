"""
Dialog stylesheet for the current theme.

Part of DataImportDialog; moved from src/data/data_import_dialog.py without changes.
"""

from src.utils.resources import resolve_icon_urls
from src.data.import_dialog.constants import STEPPER_STYLE


class ImportDialogThemeMixin:
    """Dialog stylesheet for the current theme."""

    def _apply_theme(self):
        """Yumuşak uzay teması - göze rahat."""
        self.setStyleSheet(resolve_icon_urls("""
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
                padding: 2px 6px 2px 6px;
                color: #1a5276;
                background-color: #e8eaed;
                border-radius: 3px;
                font-weight: 700;
                font-size: 12px;
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
                image: url(icons/chevron-down.svg);
                width: 12px;
                height: 12px;
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
                image: url(icons/check.svg);
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
                width: 16px;
                height: 12px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #5f7c8a;
            }
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
        """) + STEPPER_STYLE)
