"""
Advanced Settings Dialog for Time Graph Widget

Opened from the toolbar "Filters" button. Holds the range filters, which
apply to all graphs (concatenated display: the time ranges where all
conditions hold are joined together).
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
from PyQt5.QtCore import Qt, pyqtSignal

from src.ui.parameter_filters_panel import ParameterFiltersPanel

logger = logging.getLogger(__name__)


class GraphAdvancedSettingsDialog(QDialog):
    """Range filter settings for all graphs."""

    # Emits filter data: {'conditions': [...], 'mode': 'concatenated'};
    # empty conditions clear the filter
    range_filter_applied = pyqtSignal(dict)

    def __init__(self, all_signals: List[str], saved_filter_data: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.all_signals = all_signals if all_signals else []
        self.saved_filter_data = saved_filter_data
        # Last filter sent to the graphs, so OK doesn't recompute an unchanged filter
        self._last_applied = self._normalized(saved_filter_data) if saved_filter_data else None

        self._setup_dialog()
        self._setup_ui()

    def _setup_dialog(self):
        self.setWindowTitle("Advanced Settings - Range Filters")
        self.setMinimumSize(640, 520)
        self.resize(760, 640)

        # Taskbar entry with minimize/maximize buttons
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint |
                            Qt.WindowTitleHint | Qt.WindowSystemMenuHint)

        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:0.5 #2d4a66, stop:1 #1a2332);
                color: #e6f3ff;
            }
        """)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        self.parameter_filters_panel = ParameterFiltersPanel(self.all_signals, self)
        # The panel's own Apply / Reset buttons take effect immediately
        self.parameter_filters_panel.range_filter_applied.connect(self._emit_filter)
        layout.addWidget(self.parameter_filters_panel, 1)

        if self.saved_filter_data:
            self.parameter_filters_panel.set_range_filter_conditions(self.saved_filter_data)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setStyleSheet("""
            QDialogButtonBox QPushButton {
                padding: 6px 16px;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 transparent);
                color: #e6f3ff;
                font-size: 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QDialogButtonBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.1));
                border-color: #4a90e2;
            }
        """)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _normalized(filter_data: Optional[dict]) -> Dict[str, Any]:
        """Comparable form of filter data (ignores bookkeeping keys)."""
        return {'conditions': (filter_data or {}).get('conditions', [])}

    def _emit_filter(self, filter_data: dict):
        filter_data = dict(filter_data)
        filter_data['mode'] = 'concatenated'
        self._last_applied = self._normalized(filter_data)
        logger.info(f"[FILTER DIALOG] Applying {len(filter_data.get('conditions', []))} condition(s) to all graphs")
        self.range_filter_applied.emit(filter_data)

    def get_range_filter_conditions(self) -> Dict[str, Any]:
        """Current range filter conditions in the panel."""
        return self.parameter_filters_panel.get_range_filter_conditions()

    def accept(self):
        """Apply the filter shown in the panel (if it changed) and close."""
        current = self.get_range_filter_conditions()
        previous = self._last_applied or {'conditions': []}
        if self._normalized(current) != previous:
            self._emit_filter(current)
        super().accept()
