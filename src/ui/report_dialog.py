"""
Report window: configure and create a PowerPoint / Word report from the
active file's data (opened from the toolbar's Report button).
"""

import logging
import os
import re
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QStackedWidget, QTableWidget, QVBoxLayout, QWidget,
)

from src.reporting.config import (
    FORMAT_DOCX, FORMAT_PPTX, LAYOUT_AUTO, LAYOUT_OVERLAY, LAYOUT_STACKED, SCOPE_ALL,
    SCOPE_CURSORS, TIME_AXIS, OperatingPointFilter, OperatingPoints, ReportConfig,
    TrendChart,
)
from src.reporting.data_source import ReportContext
from src.reporting.texts import LANGUAGES, texts
from src.reporting.worker import ReportWorker

logger = logging.getLogger(__name__)

SETTINGS_DIR = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'TimeGraph')
LAST_CONFIG_PATH = os.path.join(SETTINGS_DIR, 'report_last.json')

LAYOUTS = [(LAYOUT_AUTO, "Auto"), (LAYOUT_OVERLAY, "Shared axis"), (LAYOUT_STACKED, "Separate panels")]

STYLE = """
QDialog {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a2332, stop:0.5 #22384f, stop:1 #1a2332);
    color: #e6f3ff;
}
QLabel, QCheckBox, QRadioButton { color: #e6f3ff; font-size: 12px; background: transparent; }
QLabel#hint { color: #9fb3c8; font-size: 11px; }
QLabel#header { font-size: 16px; font-weight: 600; }
QGroupBox {
    color: #e6f3ff; font-size: 12px; font-weight: 600;
    border: 1px solid rgba(74, 144, 226, 0.35); border-radius: 8px; margin-top: 10px; padding: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: rgba(0, 0, 0, 0.28); color: #e6f3ff; border: 1px solid #1f6b73;
    border-radius: 5px; padding: 3px 6px; font-size: 12px; selection-background-color: #2c99a5;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 20px; max-height: 24px; }
QTableWidget QLineEdit, QTableWidget QComboBox, QTableWidget QDoubleSpinBox, QTableWidget QPushButton {
    border-radius: 0px; margin: 0px;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #35b6c4; }
QComboBox QAbstractItemView { background: #1a2332; color: #e6f3ff; selection-background-color: #1f6b73; }
QListWidget, QTableWidget {
    background: rgba(0, 0, 0, 0.22); color: #e6f3ff; border: 1px solid rgba(74, 144, 226, 0.35);
    border-radius: 6px; font-size: 12px; gridline-color: rgba(255, 255, 255, 0.08);
}
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected, QTableWidget::item:selected { background: rgba(53, 182, 196, 0.25); color: #ffffff; }
QListWidget#nav { font-size: 13px; outline: 0; }
QListWidget#nav::item { padding: 9px 8px; }
QHeaderView::section { background: #1f3346; color: #c9d8e6; border: none; padding: 5px; font-size: 11px; }
QPushButton {
    background: rgba(0, 0, 0, 0.28); color: #d4dade; border: 1px solid #1f6b73;
    border-radius: 6px; padding: 5px 12px; font-size: 12px;
}
QPushButton:hover { border-color: #2c99a5; color: #ffffff; background: rgba(44, 153, 165, 0.12); }
QPushButton:disabled { color: #6c7a86; border-color: #2a3d4f; }
QPushButton#primary { background: #1f6b73; color: #ffffff; border-color: #2c99a5; font-weight: 600; padding: 6px 18px; }
QPushButton#primary:hover { background: #2c8a94; }
QProgressBar {
    background: rgba(0, 0, 0, 0.28); border: 1px solid #1f6b73; border-radius: 5px;
    color: #e6f3ff; text-align: center; font-size: 11px; max-height: 16px;
}
QProgressBar::chunk { background: #2c99a5; border-radius: 4px; }
"""


class ParameterChecklist(QWidget):
    """Searchable list of parameters with checkboxes."""

    changed = Signal()

    def __init__(self, parameters: List[str], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parameters...")
        self.search.textChanged.connect(self._filter)
        top.addWidget(self.search, 1)
        for text, state in (("All", True), ("None", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _, s=state: self._set_all_visible(s))
            top.addWidget(button)
        layout.addLayout(top)

        self.list = QListWidget()
        for name in parameters:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list.addItem(item)
        self.list.itemChanged.connect(lambda _: self._on_changed())
        layout.addWidget(self.list, 1)

        self.count_label = QLabel()
        self.count_label.setObjectName("hint")
        layout.addWidget(self.count_label)
        self._on_changed()

    def _filter(self, text: str):
        text = text.strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _set_all_visible(self, checked: bool):
        """(Un)check the parameters the search shows."""
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.list.blockSignals(False)
        self._on_changed()

    def _on_changed(self):
        self.count_label.setText(f"{len(self.checked())} of {self.list.count()} selected")
        self.changed.emit()

    def checked(self) -> List[str]:
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def set_checked(self, names: List[str]):
        wanted = set(names)
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        self.list.blockSignals(False)
        self._on_changed()


class ParameterPickerDialog(QDialog):
    """Pick the Y parameters of a trend chart."""

    def __init__(self, parameters: List[str], selected: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Y parameters")
        self.setStyleSheet(STYLE)
        self.resize(360, 460)
        layout = QVBoxLayout(self)
        self.checklist = ParameterChecklist(parameters)
        self.checklist.set_checked(selected)
        layout.addWidget(self.checklist, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected(self) -> List[str]:
        return self.checklist.checked()


class ReportDialog(QDialog):
    """The report window of one data file (TimeGraphWidget)."""

    NAV_GENERAL, NAV_STATISTICS, NAV_TRENDS, NAV_HISTOGRAMS = range(4)

    def __init__(self, graph_widget, file_name: str = "", parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.file_name = file_name
        self.parameters = sorted(graph_widget.signal_processor.signal_data.keys())
        self.worker: Optional[ReportWorker] = None

        self.setWindowTitle(f"Report - {file_name}" if file_name else "Report")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint |
                            Qt.WindowCloseButtonHint | Qt.WindowTitleHint | Qt.WindowSystemMenuHint)
        self.setStyleSheet(STYLE)
        self.setMinimumSize(860, 600)
        self.resize(980, 680)

        self._build_ui()
        self._apply_config(self._load_last_config())

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        header = QLabel(f"Report from: {self.file_name or 'active data'}")
        header.setObjectName("header")
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        for text, checkable in (("General", False), ("Statistics", True), ("Trends", True), ("Histograms", True)):
            item = QListWidgetItem(text)
            if checkable:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip("Tick to include this section in the report")
            self.nav.addItem(item)
        body.addWidget(self.nav)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._general_page())
        self.pages.addWidget(self._statistics_page())
        self.pages.addWidget(self._trends_page())
        self.pages.addWidget(self._histograms_page())
        body.addWidget(self.pages, 1)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        root.addLayout(body, 1)

        bottom = QHBoxLayout()
        for text, slot, tip in (("Load template...", self._load_template, "Load report settings from a JSON template"),
                                ("Save template...", self._save_template, "Save these report settings as a JSON template")):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bottom.addWidget(button)
        bottom.addStretch()
        self.status_label = QLabel("")
        self.status_label.setObjectName("hint")
        bottom.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        bottom.addWidget(self.progress)
        self.create_button = QPushButton("Create report")
        self.create_button.setObjectName("primary")
        self.create_button.clicked.connect(self._on_create_clicked)
        bottom.addWidget(self.create_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom.addWidget(close_button)
        root.addLayout(bottom)

    def _page(self, title: str, hint: str = "") -> (QWidget, QVBoxLayout):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("header")
        layout.addWidget(label)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("hint")
            hint_label.setWordWrap(True)
            layout.addWidget(hint_label)
        return page, layout

    def _general_page(self) -> QWidget:
        page, layout = self._page("General")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.title_edit = QLineEdit()
        self.project_edit = QLineEdit()
        self.author_edit = QLineEdit()
        form.addRow("Title:", self.title_edit)
        form.addRow("Project:", self.project_edit)
        form.addRow("Prepared by:", self.author_edit)

        self.language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.language_combo.addItem(name, code)
        self.language_combo.currentIndexChanged.connect(self._update_title_placeholder)
        form.addRow("Report language:", self.language_combo)

        format_row = QHBoxLayout()
        self.format_group = QButtonGroup(self)
        self.pptx_radio = QRadioButton("PowerPoint (.pptx)")
        self.docx_radio = QRadioButton("Word (.docx)")
        for radio in (self.pptx_radio, self.docx_radio):
            self.format_group.addButton(radio)
            format_row.addWidget(radio)
        format_row.addStretch()
        form.addRow("Format:", format_row)

        scope_row = QVBoxLayout()
        self.scope_group = QButtonGroup(self)
        self.cursor_radio = QRadioButton()
        self.all_radio = QRadioButton("All data of the file")
        for radio in (self.cursor_radio, self.all_radio):
            self.scope_group.addButton(radio)
            scope_row.addWidget(radio)
        form.addRow("Data range:", scope_row)
        self._update_cursor_option()

        self.summary_check = QCheckBox("Include a summary (file, range, sample rate, notes)")
        form.addRow("", self.summary_check)
        layout.addLayout(form)

        notes_label = QLabel("Notes / comments (shown in the summary):")
        layout.addWidget(notes_label)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("e.g. test conditions, observations, conclusions")
        layout.addWidget(self.notes_edit, 1)
        return page

    def _statistics_page(self) -> QWidget:
        page, layout = self._page(
            "Statistics",
            "Min, mean, max, standard deviation and RMS of the selected parameters, "
            "calculated on the full-resolution data of the report range.")
        self.stats_checklist = ParameterChecklist(self.parameters)
        self.stats_checklist.list.setMinimumHeight(150)
        layout.addWidget(self.stats_checklist, 1)

        self.op_group = QGroupBox("Statistics per operating point")
        self.op_group.setCheckable(True)
        self.op_group.setChecked(False)
        op_layout = QVBoxLayout(self.op_group)
        op_hint = QLabel("For each value, the rows where |control parameter − value| ≤ tolerance "
                         "(and the extra conditions) are used. The selected parameters above are reported.")
        op_hint.setObjectName("hint")
        op_hint.setWordWrap(True)
        op_layout.addWidget(op_hint)

        form = QFormLayout()
        form.setVerticalSpacing(6)
        self.op_control_combo = QComboBox()
        self.op_control_combo.addItems(self.parameters)
        self.op_values_edit = QLineEdit()
        self.op_values_edit.setPlaceholderText("e.g. 1000; 1500; 2000")
        self.op_tolerance_spin = QDoubleSpinBox()
        self.op_tolerance_spin.setDecimals(4)
        self.op_tolerance_spin.setRange(0.0, 1e9)
        self.op_tolerance_spin.setValue(0.5)
        form.addRow("Control parameter:", self.op_control_combo)
        form.addRow("Values:", self.op_values_edit)
        form.addRow("Tolerance (±):", self.op_tolerance_spin)
        op_layout.addLayout(form)

        op_layout.addWidget(QLabel("Extra conditions (all must hold):"))
        self.op_filters_table = QTableWidget(0, 3)
        self.op_filters_table.setHorizontalHeaderLabels(["Parameter", "Condition", "Value"])
        self.op_filters_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.op_filters_table.verticalHeader().setVisible(False)
        self.op_filters_table.verticalHeader().setDefaultSectionSize(30)
        self.op_filters_table.setColumnWidth(1, 90)
        self.op_filters_table.setColumnWidth(2, 130)
        self.op_filters_table.setMaximumHeight(100)
        op_layout.addWidget(self.op_filters_table)
        filter_buttons = QHBoxLayout()
        add = QPushButton("Add condition")
        add.clicked.connect(lambda: self._add_filter_row())
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self._remove_selected_rows(self.op_filters_table))
        filter_buttons.addWidget(add)
        filter_buttons.addWidget(remove)
        filter_buttons.addStretch()
        op_layout.addLayout(filter_buttons)
        layout.addWidget(self.op_group)
        return page

    def _trends_page(self) -> QWidget:
        page, layout = self._page(
            "Trends",
            "One chart per row. X = Time draws the parameters over time; another X parameter draws "
            "a scatter (X–Y) chart. Layout 'Auto' uses separate panels when the scales differ.")
        self.trends_table = QTableWidget(0, 4)
        self.trends_table.setHorizontalHeaderLabels(["Title (optional)", "X axis", "Y parameters", "Layout"])
        header = self.trends_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.trends_table.verticalHeader().setVisible(False)
        self.trends_table.verticalHeader().setDefaultSectionSize(32)
        self.trends_table.setColumnWidth(1, 160)
        self.trends_table.setColumnWidth(3, 130)
        self.trends_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.trends_table, 1)

        buttons = QHBoxLayout()
        for text, slot, tip in (
                ("Add chart", lambda: self._add_trend_row(TrendChart()), "Add an empty chart row"),
                ("From current graphs", self._trends_from_graphs,
                 "Add one chart per graph of the active tab, with the signals plotted on it"),
                ("Remove", lambda: self._remove_selected_rows(self.trends_table), "Remove the selected rows"),
                ("Move up", lambda: self._move_trend_row(-1), ""),
                ("Move down", lambda: self._move_trend_row(1), "")):
            button = QPushButton(text)
            if tip:
                button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return page

    def _histograms_page(self) -> QWidget:
        page, layout = self._page(
            "Histograms",
            "Distribution of each selected parameter with its mean and ±1 standard deviation.")
        self.hist_checklist = ParameterChecklist(self.parameters)
        layout.addWidget(self.hist_checklist, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(0, 500)
        self.bins_spin.setSpecialValueText("Auto")
        row.addWidget(self.bins_spin)
        row.addStretch()
        layout.addLayout(row)
        return page

    # ------------------------------------------------------------------
    # Table rows
    # ------------------------------------------------------------------
    def _combo(self, items: List[str], current: str = "") -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
        return combo

    def _add_filter_row(self, f: Optional[OperatingPointFilter] = None):
        f = f or OperatingPointFilter(param=self.parameters[0] if self.parameters else "")
        row = self.op_filters_table.rowCount()
        self.op_filters_table.insertRow(row)
        self.op_filters_table.setCellWidget(row, 0, self._combo(self.parameters, f.param))
        self.op_filters_table.setCellWidget(row, 1, self._combo(["<", ">"], f.op))
        value = QDoubleSpinBox()
        value.setDecimals(4)
        value.setRange(-1e12, 1e12)
        value.setValue(f.value)
        self.op_filters_table.setCellWidget(row, 2, value)

    def _add_trend_row(self, chart: TrendChart):
        row = self.trends_table.rowCount()
        self.trends_table.insertRow(row)
        title = QLineEdit(chart.title)
        title.setPlaceholderText("automatic")
        self.trends_table.setCellWidget(row, 0, title)
        self.trends_table.setCellWidget(row, 1, self._combo([TIME_AXIS] + self.parameters, chart.x))
        y_button = QPushButton()
        y_button.setProperty("names", list(chart.y))
        self._update_y_button(y_button)
        y_button.clicked.connect(lambda _, b=y_button: self._pick_y(b))
        self.trends_table.setCellWidget(row, 2, y_button)
        layout_combo = QComboBox()
        for code, name in LAYOUTS:
            layout_combo.addItem(name, code)
        layout_combo.setCurrentIndex(max(0, layout_combo.findData(chart.layout)))
        self.trends_table.setCellWidget(row, 3, layout_combo)

    @staticmethod
    def _update_y_button(button: QPushButton):
        names = button.property("names") or []
        button.setText(", ".join(names) if names else "Select...")
        button.setToolTip("\n".join(names))

    def _pick_y(self, button: QPushButton):
        picker = ParameterPickerDialog(self.parameters, button.property("names") or [], self)
        if picker.exec_() == QDialog.Accepted:
            button.setProperty("names", picker.selected())
            self._update_y_button(button)

    def _trend_rows(self) -> List[TrendChart]:
        charts = []
        for row in range(self.trends_table.rowCount()):
            charts.append(TrendChart(
                title=self.trends_table.cellWidget(row, 0).text().strip(),
                x=self.trends_table.cellWidget(row, 1).currentText(),
                y=list(self.trends_table.cellWidget(row, 2).property("names") or []),
                layout=self.trends_table.cellWidget(row, 3).currentData(),
            ))
        return charts

    def _set_trend_rows(self, charts: List[TrendChart]):
        self.trends_table.setRowCount(0)
        for chart in charts:
            self._add_trend_row(chart)

    def _move_trend_row(self, step: int):
        row = self.trends_table.currentRow()
        charts = self._trend_rows()
        target = row + step
        if row < 0 or not (0 <= target < len(charts)):
            return
        charts[row], charts[target] = charts[target], charts[row]
        self._set_trend_rows(charts)
        self.trends_table.selectRow(target)

    @staticmethod
    def _remove_selected_rows(table: QTableWidget):
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        if not rows and table.currentRow() >= 0:
            rows = [table.currentRow()]
        for row in rows:
            table.removeRow(row)

    def _trends_from_graphs(self):
        """One chart per graph of the active tab, with its plotted signals."""
        widget = self.graph_widget
        tab = widget.tab_widget.currentIndex()
        tab_name = widget.tab_widget.tabText(tab)
        mapping = widget.graph_signal_mapping.get(tab, {})
        added = 0
        for graph_index in sorted(mapping):
            names = [n for n in mapping[graph_index] if n in widget.signal_processor.signal_data]
            if names:
                self._add_trend_row(TrendChart(x=TIME_AXIS, y=names, title=f"{tab_name} – Graph {graph_index + 1}"))
                added += 1
        if added:
            self._set_section_checked(self.NAV_TRENDS, True)
        else:
            QMessageBox.information(self, "Report", "The graphs of the active tab show no signals.")

    # ==================================================================
    # Config <-> UI
    # ==================================================================
    def _section_checked(self, row: int) -> bool:
        return self.nav.item(row).checkState() == Qt.Checked

    def _set_section_checked(self, row: int, checked: bool):
        self.nav.item(row).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _update_title_placeholder(self):
        self.title_edit.setPlaceholderText(texts(self.language_combo.currentData())["default_title"])

    def _cursor_range(self):
        cursor_manager = getattr(self.graph_widget, 'cursor_manager', None)
        try:
            if cursor_manager and cursor_manager.can_zoom_to_cursors():
                a = cursor_manager.dual_cursors_1[0].value()
                b = cursor_manager.dual_cursors_2[0].value()
                return min(a, b), max(a, b)
        except Exception as e:
            logger.debug(f"Cursor range not available: {e}")
        return None

    def _update_cursor_option(self):
        cursor_range = self._cursor_range()
        if cursor_range is None:
            self.cursor_radio.setText("Between the cursors (no cursors)")
            self.cursor_radio.setEnabled(False)
            self.all_radio.setChecked(True)
        else:
            self.cursor_radio.setEnabled(True)
            container = self.graph_widget.get_active_graph_container()
            fmt = container.plot_manager.format_x_value if container else (lambda x: f"{x:.6g}")
            self.cursor_radio.setText(f"Between the cursors ({fmt(cursor_range[0])} – {fmt(cursor_range[1])})")

    def _apply_config(self, config: ReportConfig):
        self.title_edit.setText(config.title)
        self.project_edit.setText(config.project)
        self.author_edit.setText(config.author)
        self.notes_edit.setPlainText(config.notes)
        index = self.language_combo.findData(config.language)
        self.language_combo.setCurrentIndex(max(index, 0))
        self._update_title_placeholder()
        (self.docx_radio if config.output_format == FORMAT_DOCX else self.pptx_radio).setChecked(True)
        if config.scope == SCOPE_CURSORS and self.cursor_radio.isEnabled():
            self.cursor_radio.setChecked(True)
        else:
            self.all_radio.setChecked(True)
        self.summary_check.setChecked(config.summary_enabled)

        self._set_section_checked(self.NAV_STATISTICS, config.statistics_enabled)
        self.stats_checklist.set_checked(config.statistics_params)
        op = config.operating_points
        self.op_group.setChecked(op.enabled)
        if op.control_param in self.parameters:
            self.op_control_combo.setCurrentText(op.control_param)
        self.op_values_edit.setText("; ".join(f"{v:g}" for v in op.values))
        self.op_tolerance_spin.setValue(op.tolerance)
        self.op_filters_table.setRowCount(0)
        for f in op.filters:
            self._add_filter_row(f)

        self._set_section_checked(self.NAV_TRENDS, config.trends_enabled)
        self._set_trend_rows(config.trends)

        self._set_section_checked(self.NAV_HISTOGRAMS, config.histograms_enabled)
        self.hist_checklist.set_checked(config.histogram_params)
        self.bins_spin.setValue(config.histogram_bins)

    def _read_config(self) -> ReportConfig:
        filters = []
        for row in range(self.op_filters_table.rowCount()):
            filters.append(OperatingPointFilter(
                param=self.op_filters_table.cellWidget(row, 0).currentText(),
                op=self.op_filters_table.cellWidget(row, 1).currentText(),
                value=self.op_filters_table.cellWidget(row, 2).value()))
        return ReportConfig(
            title=self.title_edit.text().strip(),
            project=self.project_edit.text().strip(),
            author=self.author_edit.text().strip(),
            notes=self.notes_edit.toPlainText().strip(),
            output_format=FORMAT_DOCX if self.docx_radio.isChecked() else FORMAT_PPTX,
            language=self.language_combo.currentData(),
            scope=SCOPE_CURSORS if self.cursor_radio.isChecked() else SCOPE_ALL,
            summary_enabled=self.summary_check.isChecked(),
            statistics_enabled=self._section_checked(self.NAV_STATISTICS),
            statistics_params=self.stats_checklist.checked(),
            operating_points=OperatingPoints(
                enabled=self.op_group.isChecked(),
                control_param=self.op_control_combo.currentText(),
                values=self._parse_values(self.op_values_edit.text()) or [],
                tolerance=self.op_tolerance_spin.value(),
                filters=filters),
            trends_enabled=self._section_checked(self.NAV_TRENDS),
            trends=self._trend_rows(),
            histograms_enabled=self._section_checked(self.NAV_HISTOGRAMS),
            histogram_params=self.hist_checklist.checked(),
            histogram_bins=self.bins_spin.value(),
        )

    @staticmethod
    def _parse_values(text: str) -> Optional[List[float]]:
        """'1000; 1500 2000' -> [1000, 1500, 2000]; None if a value is not a number."""
        parts = [p for p in re.split(r"[;\s]+", text.strip()) if p]
        try:
            return [float(p.replace(",", ".")) for p in parts]
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------
    def _load_last_config(self) -> ReportConfig:
        try:
            if os.path.exists(LAST_CONFIG_PATH):
                return ReportConfig.load(LAST_CONFIG_PATH)
        except Exception as e:
            logger.warning(f"Could not read the last report settings: {e}")
        return ReportConfig()

    def _save_last_config(self, config: ReportConfig):
        try:
            os.makedirs(SETTINGS_DIR, exist_ok=True)
            config.save(LAST_CONFIG_PATH)
        except Exception as e:
            logger.warning(f"Could not save the report settings: {e}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load report template", "", "Report template (*.json)")
        if not path:
            return
        try:
            config = ReportConfig.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Report", f"The template could not be read:\n{e}")
            return
        self._apply_config(config)
        self._report_unknown_parameters(config)

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save report template", "report_template.json",
                                              "Report template (*.json)")
        if not path:
            return
        try:
            self._read_config().save(path)
            self.status_label.setText(f"Template saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Report", f"The template could not be saved:\n{e}")

    def _report_unknown_parameters(self, config: ReportConfig):
        unknown = [p for p in config.used_parameters() if p not in self.parameters]
        if unknown:
            QMessageBox.warning(
                self, "Report",
                "These parameters of the template are not in this file and were skipped:\n\n"
                + "\n".join(unknown[:30]) + ("\n..." if len(unknown) > 30 else ""))

    # ==================================================================
    # Create
    # ==================================================================
    def _validate(self, config: ReportConfig) -> Optional[str]:
        if config.statistics_enabled and not config.statistics_params:
            return "Statistics is ticked but no parameter is selected."
        if config.statistics_enabled and config.operating_points.enabled:
            if self._parse_values(self.op_values_edit.text()) is None or not config.operating_points.values:
                return "Enter the operating point values as numbers, e.g. 1000; 1500; 2000."
        if config.trends_enabled and not any(c.y for c in config.trends):
            return "Trends is ticked but no chart has Y parameters."
        if config.histograms_enabled and not config.histogram_params:
            return "Histograms is ticked but no parameter is selected."
        if not (config.statistics_enabled or config.trends_enabled or config.histograms_enabled):
            return "Tick at least one section (Statistics, Trends or Histograms) on the left."
        return None

    def _on_create_clicked(self):
        if self.worker is not None:
            self.worker.cancel()
            self.create_button.setEnabled(False)
            self.status_label.setText("Cancelling...")
            return

        self._update_cursor_option()
        config = self._read_config()
        problem = self._validate(config)
        if problem:
            QMessageBox.warning(self, "Report", problem)
            return
        self._save_last_config(config)

        extension = ".docx" if config.output_format == FORMAT_DOCX else ".pptx"
        stem = os.path.splitext(self.file_name)[0] if self.file_name else "report"
        default_dir = getattr(self, '_last_output_dir', os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save report", os.path.join(default_dir, f"{stem}_report{extension}"),
            "PowerPoint (*.pptx)" if extension == ".pptx" else "Word (*.docx)")
        if not path:
            return
        if not path.lower().endswith(extension):
            path += extension
        self._last_output_dir = os.path.dirname(path)

        cursor_range = self._cursor_range() if config.scope == SCOPE_CURSORS else None
        container = self.graph_widget.get_active_graph_container()
        context = ReportContext(
            signal_processor=self.graph_widget.signal_processor,
            file_name=self.file_name,
            parameters=list(self.parameters),
            start=cursor_range[0] if cursor_range else None,
            end=cursor_range[1] if cursor_range else None,
            datetime_axis=bool(container and getattr(container.plot_manager, 'datetime_axis_enabled', False)),
        )

        self.worker = ReportWorker(config, context, path, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_succeeded)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_worker_finished)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.create_button.setText("Cancel")
        self.status_label.setText("Creating report...")
        self.worker.start()

    def _on_progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        if message:
            self.status_label.setText(message[:60])

    def _on_succeeded(self, path: str):
        self.status_label.setText(f"Saved: {os.path.basename(path)}")
        box = QMessageBox(QMessageBox.Information, "Report", f"The report was created:\n{path}", parent=self)
        open_button = box.addButton("Open", QMessageBox.AcceptRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is open_button:
            try:
                os.startfile(path)
            except Exception as e:
                QMessageBox.warning(self, "Report", f"The file could not be opened:\n{e}")

    def _on_failed(self, message: str):
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Report", f"The report could not be created:\n{message}")

    def _on_cancelled(self):
        self.status_label.setText("Cancelled")

    def _on_worker_finished(self):
        self.worker = None
        self.progress.setVisible(False)
        self.create_button.setEnabled(True)
        self.create_button.setText("Create report")

    def showEvent(self, event):
        # The cursors may have moved since the window was last shown
        self._update_cursor_option()
        super().showEvent(event)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
        self._save_last_config(self._read_config())
        super().closeEvent(event)
