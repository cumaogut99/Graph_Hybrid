"""
Report window: configure and create a PowerPoint / Word report from the
active file's data (opened from the toolbar's Report button).
"""

import logging
import os
from typing import List, Optional

from PyQt5.QtCore import Qt, QRegExp, pyqtSignal as Signal
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QStackedWidget, QTableWidget, QVBoxLayout, QWidget,
)

from src.reporting.config import (
    FORMAT_DOCX, FORMAT_PPTX, LAYOUT_AUTO, LAYOUT_OVERLAY, LAYOUT_STACKED, SCOPE_ALL,
    SCOPE_CURSORS, TIME_AXIS, ConditionGroup, RangeCondition, ReportConfig, TrendChart,
)
from src.reporting.charts import MAX_SERIES
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

    # Same order as the sections in the report
    NAV_GENERAL, NAV_HISTOGRAMS, NAV_TRENDS, NAV_STATISTICS, NAV_FILTERED = range(5)

    def __init__(self, graph_widget, file_name: str = "", parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.file_name = file_name
        self.parameters = sorted(graph_widget.signal_processor.signal_data.keys())
        self.worker: Optional[ReportWorker] = None
        self.groups: List[ConditionGroup] = []
        self._group_row = -1   # group shown in the conditions table

        self.setWindowTitle(f"Report - {file_name}" if file_name else "Report")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint |
                            Qt.WindowCloseButtonHint | Qt.WindowTitleHint | Qt.WindowSystemMenuHint)
        self.setStyleSheet(STYLE)
        self.setMinimumSize(860, 600)
        self.resize(1020, 720)

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
        for text, checkable in (("General", False), ("Histograms", True), ("Trends", True),
                                ("Statistics", True), ("Filtered charts", True)):
            item = QListWidgetItem(text)
            if checkable:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip("Tick to include this section in the report")
            self.nav.addItem(item)
        body.addWidget(self.nav)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._general_page())
        self.pages.addWidget(self._histograms_page())
        self.pages.addWidget(self._trends_page())
        self.pages.addWidget(self._statistics_page())
        self.pages.addWidget(self._filtered_page())
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
            "Min, mean, max and standard deviation of the selected parameters, calculated on the "
            "full-resolution data of the report range. Each condition group gets its own table, "
            "computed on the rows where all conditions of the group hold.")
        self.stats_checklist = ParameterChecklist(self.parameters)
        self.stats_checklist.list.setMinimumHeight(100)
        layout.addWidget(self.stats_checklist, 1)
        self.whole_range_check = QCheckBox("Also a table for the whole range (without conditions)")
        layout.addWidget(self.whole_range_check)

        groups_box = QGroupBox("Condition groups")
        groups_layout = QHBoxLayout(groups_box)
        groups_layout.setSpacing(10)

        left = QVBoxLayout()
        self.groups_list = QListWidget()
        self.groups_list.setFixedWidth(200)
        self.groups_list.setToolTip("Double-click to rename")
        self.groups_list.currentRowChanged.connect(self._on_group_selected)
        self.groups_list.itemChanged.connect(self._on_group_renamed)
        left.addWidget(self.groups_list, 1)
        group_buttons = QHBoxLayout()
        for text, slot, tip in (("Add", self._add_group, "Add an empty condition group"),
                                ("Copy", self._copy_group, "Add a copy of the selected group"),
                                ("Remove", self._remove_group, "Remove the selected group")):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            group_buttons.addWidget(button)
        left.addLayout(group_buttons)
        groups_layout.addLayout(left)

        right = QVBoxLayout()
        self.conditions_label = QLabel()
        right.addWidget(self.conditions_label)
        self.conditions_table = QTableWidget(0, 3)
        self.conditions_table.setHorizontalHeaderLabels(["Parameter", "Min", "Max"])
        self.conditions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.conditions_table.verticalHeader().setVisible(False)
        self.conditions_table.verticalHeader().setDefaultSectionSize(30)
        self.conditions_table.setColumnWidth(1, 120)
        self.conditions_table.setColumnWidth(2, 120)
        self.conditions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        right.addWidget(self.conditions_table, 1)
        condition_buttons = QHBoxLayout()
        self.add_condition_button = QPushButton("Add condition")
        self.add_condition_button.clicked.connect(lambda: self._add_condition_row())
        self.remove_condition_button = QPushButton("Remove")
        self.remove_condition_button.clicked.connect(
            lambda: self._remove_selected_rows(self.conditions_table))
        condition_buttons.addWidget(self.add_condition_button)
        condition_buttons.addWidget(self.remove_condition_button)
        condition_buttons.addStretch()
        right.addLayout(condition_buttons)
        bounds_hint = QLabel("Rows with Min ≤ value ≤ Max are used; leave Min or Max empty for no limit.")
        bounds_hint.setObjectName("hint")
        bounds_hint.setWordWrap(True)
        right.addWidget(bounds_hint)
        groups_layout.addLayout(right, 1)
        layout.addWidget(groups_box, 2)
        self._show_group(-1)
        return page

    def _filtered_page(self) -> QWidget:
        page, layout = self._page(
            "Filtered charts",
            "One chart per condition group (defined on the Statistics page): the rows that meet "
            "the group's conditions are joined end to end and drawn against the joined time, "
            "so the time axis is shifted. Dashed lines mark where the segments meet. "
            f"At most {MAX_SERIES} parameters per chart.")
        self.filtered_checklist = ParameterChecklist(self.parameters)
        layout.addWidget(self.filtered_checklist, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Layout:"))
        self.filtered_layout_combo = QComboBox()
        for code, name in LAYOUTS:
            self.filtered_layout_combo.addItem(name, code)
        row.addWidget(self.filtered_layout_combo)
        row.addStretch()
        layout.addLayout(row)
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

    def _bound_edit(self, value: Optional[float]) -> QLineEdit:
        edit = QLineEdit("" if value is None else f"{value:g}")
        edit.setPlaceholderText("no limit")
        edit.setValidator(QRegExpValidator(QRegExp(r"-?[0-9]*([.,][0-9]*)?([eE][-+]?[0-9]*)?"), edit))
        return edit

    @staticmethod
    def _bound_value(edit: QLineEdit) -> Optional[float]:
        text = edit.text().strip().replace(",", ".")
        try:
            return float(text) if text else None
        except ValueError:
            return None

    def _add_condition_row(self, condition: Optional[RangeCondition] = None):
        if self._group_row < 0:
            return
        condition = condition or RangeCondition(param=self.parameters[0] if self.parameters else "")
        row = self.conditions_table.rowCount()
        self.conditions_table.insertRow(row)
        self.conditions_table.setCellWidget(row, 0, self._combo(self.parameters, condition.param))
        self.conditions_table.setCellWidget(row, 1, self._bound_edit(condition.minimum))
        self.conditions_table.setCellWidget(row, 2, self._bound_edit(condition.maximum))

    def _condition_rows(self) -> List[RangeCondition]:
        return [RangeCondition(param=self.conditions_table.cellWidget(row, 0).currentText(),
                               minimum=self._bound_value(self.conditions_table.cellWidget(row, 1)),
                               maximum=self._bound_value(self.conditions_table.cellWidget(row, 2)))
                for row in range(self.conditions_table.rowCount())]

    # ------------------------------------------------------------------
    # Condition groups (the table shows the conditions of the selected group)
    # ------------------------------------------------------------------
    def _commit_group(self):
        """Store the conditions table into the group it shows."""
        if 0 <= self._group_row < len(self.groups):
            self.groups[self._group_row].conditions = self._condition_rows()

    def _show_group(self, row: int):
        self._group_row = row if 0 <= row < len(self.groups) else -1
        self.conditions_table.setRowCount(0)
        has_group = self._group_row >= 0
        if has_group:
            for condition in self.groups[self._group_row].conditions:
                self._add_condition_row(condition)
            self.conditions_label.setText(
                f"Conditions of '{self.groups[self._group_row].name}' (all must hold):")
        else:
            self.conditions_label.setText("Add a condition group to enter conditions.")
        for widget in (self.conditions_table, self.add_condition_button, self.remove_condition_button):
            widget.setEnabled(has_group)

    def _on_group_selected(self, row: int):
        self._commit_group()
        self._show_group(row)

    def _on_group_renamed(self, item: QListWidgetItem):
        row = self.groups_list.row(item)
        if 0 <= row < len(self.groups):
            self.groups[row].name = item.text().strip()
            if row == self._group_row:
                self.conditions_label.setText(f"Conditions of '{self.groups[row].name}' (all must hold):")

    def _set_groups(self, groups: List[ConditionGroup], current: int = 0):
        self.groups = groups
        self._group_row = -1
        self.groups_list.blockSignals(True)
        self.groups_list.clear()
        for group in groups:
            item = QListWidgetItem(group.name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.groups_list.addItem(item)
        self.groups_list.blockSignals(False)
        current = min(current, len(groups) - 1)
        self.groups_list.blockSignals(True)
        self.groups_list.setCurrentRow(current)
        self.groups_list.blockSignals(False)
        self._show_group(current)

    def _default_group_name(self) -> str:
        template = texts(self.language_combo.currentData())["group_default"]
        used = {g.name for g in self.groups}
        n = len(self.groups) + 1
        while template.format(n=n) in used:
            n += 1
        return template.format(n=n)

    def _add_group(self):
        self._commit_group()
        groups = self.groups + [ConditionGroup(name=self._default_group_name())]
        self._set_groups(groups, len(groups) - 1)
        self._add_condition_row()

    def _copy_group(self):
        if self._group_row < 0:
            return
        self._commit_group()
        source = self.groups[self._group_row]
        copy = ConditionGroup(name=self._default_group_name(),
                              conditions=[RangeCondition(c.param, c.minimum, c.maximum)
                                          for c in source.conditions])
        groups = self.groups + [copy]
        self._set_groups(groups, len(groups) - 1)

    def _remove_group(self):
        row = self._group_row
        if row < 0:
            return
        groups = self.groups[:row] + self.groups[row + 1:]
        self._set_groups(groups, max(0, row - 1) if groups else -1)

    def _read_groups(self) -> List[ConditionGroup]:
        self._commit_group()
        return [ConditionGroup(name=g.name, conditions=list(g.conditions)) for g in self.groups]

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

        self._set_section_checked(self.NAV_HISTOGRAMS, config.histograms_enabled)
        self.hist_checklist.set_checked(config.histogram_params)
        self.bins_spin.setValue(config.histogram_bins)

        self._set_section_checked(self.NAV_TRENDS, config.trends_enabled)
        self._set_trend_rows(config.trends)

        self._set_section_checked(self.NAV_STATISTICS, config.statistics_enabled)
        self.stats_checklist.set_checked(config.statistics_params)
        self.whole_range_check.setChecked(config.statistics_whole_range)
        self._set_groups([ConditionGroup(name=g.name, conditions=list(g.conditions))
                          for g in config.condition_groups])

        self._set_section_checked(self.NAV_FILTERED, config.filtered_charts_enabled)
        self.filtered_checklist.set_checked(config.filtered_chart_params)
        self.filtered_layout_combo.setCurrentIndex(
            max(0, self.filtered_layout_combo.findData(config.filtered_chart_layout)))

    def _read_config(self) -> ReportConfig:
        return ReportConfig(
            title=self.title_edit.text().strip(),
            project=self.project_edit.text().strip(),
            author=self.author_edit.text().strip(),
            notes=self.notes_edit.toPlainText().strip(),
            output_format=FORMAT_DOCX if self.docx_radio.isChecked() else FORMAT_PPTX,
            language=self.language_combo.currentData(),
            scope=SCOPE_CURSORS if self.cursor_radio.isChecked() else SCOPE_ALL,
            summary_enabled=self.summary_check.isChecked(),
            histograms_enabled=self._section_checked(self.NAV_HISTOGRAMS),
            histogram_params=self.hist_checklist.checked(),
            histogram_bins=self.bins_spin.value(),
            trends_enabled=self._section_checked(self.NAV_TRENDS),
            trends=self._trend_rows(),
            statistics_enabled=self._section_checked(self.NAV_STATISTICS),
            statistics_params=self.stats_checklist.checked(),
            statistics_whole_range=self.whole_range_check.isChecked(),
            condition_groups=self._read_groups(),
            filtered_charts_enabled=self._section_checked(self.NAV_FILTERED),
            filtered_chart_params=self.filtered_checklist.checked(),
            filtered_chart_layout=self.filtered_layout_combo.currentData(),
        )

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
        if config.histograms_enabled and not config.histogram_params:
            return "Histograms is ticked but no parameter is selected."
        if config.trends_enabled and not any(c.y for c in config.trends):
            return "Trends is ticked but no chart has Y parameters."
        if config.statistics_enabled and not config.statistics_params:
            return "Statistics is ticked but no parameter is selected."
        if config.filtered_charts_enabled:
            if not config.filtered_chart_params:
                return "Filtered charts is ticked but no parameter is selected."
            if len(config.filtered_chart_params) > MAX_SERIES:
                return f"Filtered charts: select at most {MAX_SERIES} parameters."
            if not config.condition_groups:
                return "Filtered charts is ticked but there is no condition group (Statistics page)."
        if config.statistics_enabled or config.filtered_charts_enabled:
            for group in config.condition_groups:
                for c in group.conditions:
                    if c.minimum is not None and c.maximum is not None and c.minimum > c.maximum:
                        return (f"Condition group '{group.name}': the Min of {c.param} "
                                f"is greater than its Max.")
        if not (config.histograms_enabled or config.trends_enabled or config.statistics_enabled
                or config.filtered_charts_enabled):
            return "Tick at least one section (Histograms, Trends, Statistics or Filtered charts) on the left."
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
