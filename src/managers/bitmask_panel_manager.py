from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QGroupBox
from PyQt5.QtCore import QObject, pyqtSignal as Signal, Qt
import polars as pl
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BitmaskPanelManager(QObject):
    def __init__(self, data_manager, theme_manager, parent=None):
        super().__init__(parent)
        self.widget = QWidget()
        self.data_manager = data_manager
        self.theme_manager = theme_manager
        self.graph_sections = []
        self._bitmask_data = {}
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
        layout.addWidget(self.status_label)

        # Graph-specific sections will be added here dynamically
        self.graphs_layout = QVBoxLayout()
        layout.addLayout(self.graphs_layout)
        
        layout.addStretch()

    def _load_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Open Bitmask Excel File",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            try:
                # ✅ FIX: Read all sheets properly
                # Polars doesn't support sheet_name=None, so use openpyxl to get sheet names first
                import openpyxl
                
                logger.info(f"Loading Excel file: {file_path}")
                
                # Get all sheet names using openpyxl
                workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                sheet_names = workbook.sheetnames
                workbook.close()
                
                logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
                
                # Read each sheet with Polars
                raw_data = {}
                for sheet_name in sheet_names:
                    try:
                        df = pl.read_excel(file_path, sheet_name=sheet_name)
                        raw_data[sheet_name] = df
                        logger.info(f"Loaded sheet '{sheet_name}': {df.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to load sheet '{sheet_name}': {e}")
                        continue
                
                # Standardize the data format
                self._bitmask_data = {}
                for sheet_name, df in raw_data.items():
                    if df.shape[1] >= 2:
                        # Assume first col is bit number, second is description
                        df.columns = ['bit', 'description']
                        # Set bit number as index for fast lookup
                        bit_desc_dict = {}
                        for row in df.iter_rows(named=True):
                            bit_desc_dict[row['bit']] = row['description']
                        self._bitmask_data[sheet_name] = bit_desc_dict
                        logger.info(f"Processed sheet '{sheet_name}': {len(bit_desc_dict)} bits")
                    
                self.status_label.setText(f"Loaded: {file_path.split('/')[-1]}")
                logger.info(f"Successfully loaded {len(self._bitmask_data)} sheets")
                self.update_all_comboboxes()

            except Exception as e:
                error_msg = f"Error loading file: {e}"
                logger.error(error_msg)
                self.status_label.setText(error_msg)
                self._bitmask_data = {}

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
        
    def update_all_comboboxes(self):
        """Update all parameter selection comboboxes with loaded sheet names."""
        if not self._bitmask_data:
            return
        
        parameter_names = [""] + sorted(list(self._bitmask_data.keys()))
        for section in self.graph_sections:
            combo = section['combo']
            current_selection = combo.currentText()
            combo.clear()
            combo.addItems(parameter_names)
            
            if current_selection in parameter_names:
                combo.setCurrentText(current_selection)

    def _create_graph_section(self, graph_number):
        section_widget = QGroupBox(f"Graph {graph_number} Analysis")
        section_layout = QVBoxLayout(section_widget)
        
        param_label = QLabel("Select Parameter:")
        param_combo = QComboBox()
        
        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setText("Move cursor over graph to see bitmask details.")
        result_display.setFixedHeight(100)

        section_layout.addWidget(param_label)
        section_layout.addWidget(param_combo)
        section_layout.addWidget(result_display)
        
        return section_widget, param_combo, result_display

    def on_cursor_position_changed(self, cursor_positions: dict):
        if 'c1' not in cursor_positions:
            return
            
        time_pos = cursor_positions['c1']

        for section in self.graph_sections:
            param_name = section['combo'].currentText()
            result_display = section['result_display']

            if not param_name or not self._bitmask_data or param_name not in self._bitmask_data:
                result_display.setText("Select a valid parameter.")
                continue

            value = self.data_manager.get_value_at_time(param_name, time_pos)

            if value is None or np.isnan(value):
                result_display.setText(f"{param_name} @ {time_pos:.2f}s: No data")
                continue

            try:
                int_value = int(value)
                active_bits = []
                bit_definitions = self._bitmask_data[param_name]

                for bit in range(64): # Check up to 64 bits
                    if (int_value >> bit) & 1:
                        error_msg = bit_definitions.get(bit, f"Undefined Bit {bit}")
                        active_bits.append(f"Bit {bit}: {error_msg}")
                
                if active_bits:
                    result_text = f"{param_name} @ {time_pos:.2f}s = {int_value}\n"
                    result_text += "\n".join(active_bits)
                    result_display.setText(result_text)
                else:
                    result_display.setText(f"{param_name} @ {time_pos:.2f}s = {int_value}\nNo active bits.")

            except (ValueError, TypeError) as e:
                result_display.setText(f"Error processing value: {value}.\n{e}")
