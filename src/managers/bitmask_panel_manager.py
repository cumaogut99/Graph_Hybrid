import os
import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QGroupBox
from PyQt5.QtCore import QObject
import numpy as np
import logging

from src.data.excel_to_csv import list_sheets, read_sheet_rows

logger = logging.getLogger(__name__)

# A bit number cell: "3", "3.0" or "Bit 3"
_BIT_CELL = re.compile(r'^\s*(?:bit\s*)?(\d+)(?:\.0+)?\s*$', re.IGNORECASE)

FILTER_NOTICE = ("Bitmask results are intentionally hidden while a range filter is active: "
                 "the filtered view joins separate time ranges, so the cursor time does not "
                 "point to the recorded sample. Clear the filter to see them again.")


class BitmaskPanelManager(QObject):
    """
    Decodes bitmask (status word) signals at cursor 1.

    The definitions come from an Excel file: one sheet per signal, named like
    the signal, with the bit number in the first column and its description
    in the second.
    """

    def __init__(self, signal_processor, theme_manager, parent=None):
        super().__init__(parent)
        self.widget = QWidget()
        self.signal_processor = signal_processor
        self.theme_manager = theme_manager
        self.graph_sections = []
        self._bitmask_data = {}
        self._filter_active = False
        self._last_time = None
        self._setup_ui()
        self.update_theme()

    def get_widget(self):
        return self.widget

    def update_theme(self):
        """Apply the current theme to the panel."""
        panel_stylesheet = self.theme_manager.get_widget_stylesheet('panel')
        self.widget.setStyleSheet(panel_stylesheet)

    def _setup_ui(self):
        layout = QVBoxLayout(self.widget)

        self.load_excel_button = QPushButton("Load Bitmask Excel File")
        self.load_excel_button.clicked.connect(self._load_excel_file)
        layout.addWidget(self.load_excel_button)

        self.status_label = QLabel("Please load an Excel file.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.filter_notice = QLabel(FILTER_NOTICE)
        self.filter_notice.setWordWrap(True)
        self.filter_notice.setStyleSheet(
            "QLabel { color: #f0c36d; background-color: rgba(240, 195, 109, 0.12);"
            " border: 1px solid #8a6d2f; border-radius: 4px; padding: 6px; }")
        self.filter_notice.setVisible(False)
        layout.addWidget(self.filter_notice)

        # Graph-specific sections will be added here dynamically
        self.graphs_layout = QVBoxLayout()
        layout.addLayout(self.graphs_layout)

        layout.addStretch()

    def _load_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Open Bitmask Excel File",
            "",
            "Excel Files (*.xlsx *.xlsm *.xls)"
        )
        if file_path:
            self.load_definitions(file_path)

    def load_definitions(self, file_path: str):
        """Read the bit definitions from every sheet of an Excel file."""
        try:
            logger.info(f"Loading bitmask Excel file: {file_path}")
            bitmask_data = {}
            for sheet_name in list_sheets(file_path):
                bits = {}
                # Header rows (e.g. "Bit | Description") are skipped because
                # their first cell is not a bit number
                for row in read_sheet_rows(file_path, sheet_name):
                    if len(row) < 2:
                        continue
                    match = _BIT_CELL.match(str(row[0]))
                    if match:
                        bits[int(match.group(1))] = str(row[1]).strip()
                if bits:
                    bitmask_data[sheet_name] = bits
                    logger.info(f"Bitmask sheet '{sheet_name}': {len(bits)} bits")
                else:
                    logger.warning(f"Bitmask sheet '{sheet_name}' has no bit definitions")

            if not bitmask_data:
                raise ValueError("no sheet has bit definitions (bit number in column A, description in column B)")

            self._bitmask_data = bitmask_data
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)} ({len(bitmask_data)} parameters)")
            self.update_all_comboboxes()
            self._refresh()

        except Exception as e:
            logger.error(f"Error loading bitmask file: {e}")
            self.status_label.setText(f"Error loading file: {e}")
            self._bitmask_data = {}
            self.update_all_comboboxes()

    def update_graph_sections(self, num_graphs):
        # Clear existing sections
        for section in self.graph_sections:
            section['widget'].setParent(None)
        self.graph_sections = []

        # Create a new section for each graph
        for i in range(num_graphs):
            graph_section_widget, combo, result_display = self._create_graph_section(i + 1)
            self.graphs_layout.addWidget(graph_section_widget)
            self.graph_sections.append({
                "widget": graph_section_widget,
                "combo": combo,
                "result_display": result_display
            })
        self.update_all_comboboxes()
        self._refresh()

    def update_all_comboboxes(self):
        """Update all parameter selection comboboxes with loaded sheet names."""
        parameter_names = [""] + sorted(self._bitmask_data.keys())
        for section in self.graph_sections:
            combo = section['combo']
            current_selection = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(parameter_names)
            if current_selection in parameter_names:
                combo.setCurrentText(current_selection)
            combo.blockSignals(False)

    def _create_graph_section(self, graph_number):
        section_widget = QGroupBox(f"Graph {graph_number} Analysis")
        section_layout = QVBoxLayout(section_widget)

        param_label = QLabel("Select Parameter:")
        param_combo = QComboBox()
        param_combo.currentTextChanged.connect(lambda _text: self._refresh())

        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setText("Move cursor over graph to see bitmask details.")
        result_display.setFixedHeight(100)

        section_layout.addWidget(param_label)
        section_layout.addWidget(param_combo)
        section_layout.addWidget(result_display)

        return section_widget, param_combo, result_display

    def set_filter_active(self, active: bool):
        """Hide the results while a range filter is active (see FILTER_NOTICE)."""
        active = bool(active)
        if active == self._filter_active:
            return
        self._filter_active = active
        self.filter_notice.setVisible(active)
        self._refresh()

    def _refresh(self):
        """Re-evaluate the sections at the last known cursor time."""
        if self._filter_active:
            for section in self.graph_sections:
                section['result_display'].setText("Hidden: range filter is active.")
        elif self._last_time is not None:
            self._update_sections(self._last_time)

    def on_cursor_position_changed(self, cursor_positions: dict):
        if 'c1' not in cursor_positions:
            return
        self._last_time = cursor_positions['c1']
        if self._filter_active:
            return
        self._update_sections(self._last_time)

    def _update_sections(self, time_pos: float):
        for section in self.graph_sections:
            param_name = section['combo'].currentText()
            result_display = section['result_display']

            if not param_name or param_name not in self._bitmask_data:
                result_display.setText("Select a valid parameter.")
                continue

            signal_name = self._resolve_signal(param_name)
            if signal_name is None:
                result_display.setText(f"No signal named '{param_name}' in the loaded data.")
                continue

            value = self._value_at(signal_name, time_pos)
            if value is None or not np.isfinite(value):
                result_display.setText(f"{param_name} @ {time_pos:.2f}s: No data")
                continue

            # Status words are integers; negative values are read as 64-bit two's complement
            int_value = int(value)
            bits = int_value & ((1 << 64) - 1)
            bit_definitions = self._bitmask_data[param_name]
            active_bits = [f"Bit {bit}: {bit_definitions.get(bit, 'Undefined')}"
                           for bit in range(64) if (bits >> bit) & 1]

            result_text = f"{param_name} @ {time_pos:.2f}s = {int_value}\n"
            result_text += "\n".join(active_bits) if active_bits else "No active bits."
            result_display.setText(result_text)

    def _resolve_signal(self, param_name: str):
        """Signal for a sheet name: exact match, else case-insensitive."""
        names = list(self.signal_processor.signal_data.keys())
        if param_name in names:
            return param_name
        wanted = param_name.strip().lower()
        for name in names:
            if name.strip().lower() == wanted:
                return name
        return None

    def _value_at(self, signal_name: str, time_pos: float):
        """Value of the sample nearest to time_pos (no interpolation: bits must not mix)."""
        info = self.signal_processor.signal_data.get(signal_name)
        if not info:
            return None
        x_data = info.get('x_data')
        time_range = info.get('metadata', {}).get('full_time_range')
        if not time_range and x_data is not None and len(x_data):
            time_range = (float(x_data[0]), float(x_data[-1]))
        if time_range and not (time_range[0] <= time_pos <= time_range[1]):
            return None

        if self.signal_processor._is_file_backed(info):
            # Nearest sample read from the memory-mapped file
            return self.signal_processor.get_signals_at_time([signal_name], time_pos).get(signal_name)

        y_data = info.get('y_data')
        if x_data is None or y_data is None or len(x_data) == 0:
            return None
        idx = int(np.searchsorted(x_data, time_pos))
        idx = min(max(idx, 0), len(x_data) - 1)
        if idx > 0 and abs(x_data[idx - 1] - time_pos) <= abs(x_data[idx] - time_pos):
            idx -= 1
        return float(y_data[idx])
