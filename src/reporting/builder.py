"""
Builds the report content (format independent) from a ReportConfig and the
active file's data. Writers turn the result into .pptx or .docx.
"""

import datetime
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from . import charts
from .charts import format_number
from .config import ReportConfig, TIME_AXIS
from .data_source import ReportContext, ReportData, envelope, statistics, thin
from .texts import texts


# ----------------------------------------------------------------------
# Content model
# ----------------------------------------------------------------------
@dataclass
class Paragraph:
    text: str


@dataclass
class KeyValues:
    rows: List[Tuple[str, str]]


@dataclass
class Table:
    caption: str
    headers: List[str]
    rows: List[List[str]]


@dataclass
class Figure:
    caption: str
    png: bytes


Block = Union[Paragraph, KeyValues, Table, Figure]


@dataclass
class Section:
    title: str
    blocks: List[Block] = field(default_factory=list)


@dataclass
class Report:
    title: str
    project: str
    author: str
    date: str
    file_name: str
    range_text: str
    texts: dict
    sections: List[Section] = field(default_factory=list)


class BuildCancelled(Exception):
    pass


class ReportBuilder:
    """Computes statistics and charts; progress(percent, message) is called along the way."""

    def __init__(self, config: ReportConfig, context: ReportContext,
                 progress: Optional[Callable[[int, str], None]] = None,
                 is_cancelled: Optional[Callable[[], bool]] = None):
        self.config = config
        self.context = context
        self.data = ReportData(context)
        self.t = texts(config.language)
        self._progress = progress or (lambda p, m: None)
        self._is_cancelled = is_cancelled or (lambda: False)
        self._steps_done = 0
        self._steps_total = 1

    # ------------------------------------------------------------------
    def build(self) -> Report:
        c = self.config
        op_points = len(c.operating_points.values) if c.statistics_enabled and c.operating_points.enabled else 0
        self._steps_total = 3 + op_points \
            + (len(c.trends) if c.trends_enabled else 0) \
            + (len(c.histogram_params) if c.histograms_enabled else 0)

        report = Report(
            title=c.title.strip() or self.t["default_title"],
            project=c.project.strip(),
            author=c.author.strip(),
            date=datetime.date.today().strftime("%d.%m.%Y"),
            file_name=self.context.file_name,
            range_text=self._range_text(),
            texts=self.t,
        )

        content: List[Section] = []
        if c.statistics_enabled and c.statistics_params:
            content.append(self._statistics_section())
        if c.trends_enabled and c.trends:
            content.append(self._trends_section())
        if c.histograms_enabled and c.histogram_params:
            content.append(self._histograms_section())
        content = [s for s in content if s.blocks]

        if c.summary_enabled:
            report.sections.append(self._summary_section(content))
        report.sections += content
        self._step("")
        return report

    # ------------------------------------------------------------------
    def _step(self, message: str):
        if self._is_cancelled():
            raise BuildCancelled()
        self._steps_done += 1
        self._progress(min(90, int(90 * self._steps_done / max(self._steps_total, 1))), message)

    def _format_time(self, t: float) -> str:
        if self.context.datetime_axis:
            return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%d.%m.%Y %H:%M:%S")
        return f"{format_number(t)} s"

    def _span(self):
        return self.data.time_span(self.config.used_parameters() or self.context.parameters)

    def _range_text(self) -> str:
        span = self._span()
        if span is None:
            return self.t["no_data_in_range"]
        start, end, _ = span
        scope = self.t["cursor_range"] if self.context.start is not None else self.t["all_data"]
        return f"{self._format_time(start)} – {self._format_time(end)} ({scope})"

    def _stat_headers(self) -> List[str]:
        t = self.t
        return [t["h_parameter"], t["h_samples"], t["h_min"], t["h_mean"], t["h_max"], t["h_std"], t["h_rms"]]

    @staticmethod
    def _stat_row(name: str, st) -> List[str]:
        return [name, f"{st.count:,}", format_number(st.minimum), format_number(st.mean),
                format_number(st.maximum), format_number(st.std), format_number(st.rms)]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _summary_section(self, content: List[Section]) -> Section:
        t = self.t
        section = Section(t["summary"])
        rows = []
        if self.config.project.strip():
            rows.append((t["project"], self.config.project.strip()))
        if self.context.file_name:
            rows.append((t["data_file"], self.context.file_name))
        span = self._span()
        if span is not None:
            start, end, count = span
            rows.append((t["range"], self._range_text()))
            rows.append((t["duration"], f"{format_number(end - start)} s"))
            rows.append((t["samples_in_range"], f"{count:,}"))
            if count > 1 and end > start:
                rows.append((t["sample_rate"], f"{format_number((count - 1) / (end - start))} Hz"))
        rows.append((t["params_in_file"], str(len(self.context.parameters))))
        in_report = [p for p in self.config.used_parameters() if p in self.context.parameters]
        rows.append((t["params_in_report"], str(len(in_report))))
        section.blocks.append(KeyValues(rows))

        if content:
            section.blocks.append(Paragraph(t["report_contains"].format(
                sections=", ".join(s.title for s in content))))
        if self.config.notes.strip():
            section.blocks.append(Paragraph(f"{t['notes']}: {self.config.notes.strip()}"))
        self._step(t["summary"])
        return section

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _statistics_section(self) -> Section:
        t = self.t
        section = Section(t["statistics"])
        rows, missing = [], []
        for name in self.config.statistics_params:
            s = self.data.series(name)
            st = statistics(s[1]) if s is not None else None
            if st is None:
                missing.append(name)
            else:
                rows.append(self._stat_row(name, st))
        if rows:
            section.blocks.append(Table(t["stats_table"], self._stat_headers(), rows))
        if missing:
            section.blocks.append(Paragraph(t["no_data_for"].format(names=", ".join(missing))))
        self._step(t["statistics"])

        if self.config.operating_points.enabled:
            section.blocks += self._operating_point_blocks()
        return section

    def _operating_point_blocks(self) -> List[Block]:
        t = self.t
        op = self.config.operating_points
        params = [p for p in self.config.statistics_params if p != op.control_param]
        filter_params = [f.param for f in op.filters if f.param]
        others = [p for p in dict.fromkeys(params + filter_params) if p != op.control_param]
        aligned = self.data.aligned(op.control_param, others) if op.control_param else None
        if aligned is None:
            return [Paragraph(t["op_missing"].format(param=op.control_param))]

        control = aligned[op.control_param]
        base_mask = np.ones(len(control), dtype=bool)
        conditions = []
        for f in op.filters:
            if f.param in aligned:
                values = aligned[f.param]
                base_mask &= (values < f.value) if f.op == "<" else (values > f.value)
                conditions.append(f"{f.param} {f.op} {format_number(f.value)}")

        blocks: List[Block] = [Paragraph(t["op_intro"].format(
            param=op.control_param, tol=format_number(op.tolerance),
            conditions=(t["op_and"] + t["op_and"].join(conditions)) if conditions else ""))]

        means = {p: [] for p in params}
        point_headers, empty, details = [], [], []
        for value in op.values:
            mask = base_mask & (np.abs(control - value) <= op.tolerance)
            count = int(np.count_nonzero(mask))
            label = f"{op.control_param} ≈ {format_number(value)}"
            if count == 0:
                empty.append(format_number(value))
            else:
                point_headers.append(f"≈ {format_number(value)} (n={count:,})")
                rows = []
                for p in params:
                    st = statistics(aligned[p][mask])
                    means[p].append(format_number(st.mean) if st else "–")
                    if st:
                        rows.append(self._stat_row(p, st))
                if rows:
                    details.append(Table(t["op_detail"].format(label=label, count=f"{count:,}"),
                                         self._stat_headers(), rows))
            self._step(label)

        if point_headers and params:
            blocks.append(Table(t["op_means"].format(param=op.control_param),
                                [t["h_parameter"]] + point_headers,
                                [[p] + means[p] for p in params]))
        blocks += details
        if empty:
            blocks.append(Paragraph(t["op_empty"].format(param=op.control_param, values=", ".join(empty))))
        return blocks

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------
    def _trends_section(self) -> Section:
        t = self.t
        section = Section(t["trends"])
        skipped = []
        for chart in self.config.trends:
            ys = [y for y in chart.y if y]
            if not ys:
                continue
            x_label = t["time"] if chart.x == TIME_AXIS else chart.x
            title = chart.title.strip() or t["vs"].format(y=", ".join(ys), x=x_label)
            png = self._trend_png(chart.x, ys, chart.layout)
            if png is None:
                skipped.append(title)
            else:
                section.blocks.append(Figure(title, png))
            self._step(title)
        if skipped:
            section.blocks.append(Paragraph(t["not_drawn"].format(titles="; ".join(skipped))))
        return section

    def _trend_png(self, x: str, ys: List[str], layout: str) -> Optional[bytes]:
        if x == TIME_AXIS:
            names, series = [], []
            for name in ys:
                s = self.data.series(name)
                if s is not None:
                    names.append(name)
                    series.append(envelope(*s))
            if not series:
                return None
            return charts.time_trend(names, series, self.context.datetime_axis, self.t, layout)

        aligned = self.data.aligned(x, ys)
        if aligned is None or len(aligned[x]) == 0:
            return None
        arrays = thin(aligned[x], *[aligned[y] for y in ys])
        return charts.xy_scatter(x, ys, arrays[0], list(arrays[1:]))

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------
    def _histograms_section(self) -> Section:
        t = self.t
        section = Section(t["distributions"])
        for name in self.config.histogram_params:
            s = self.data.series(name)
            png = charts.histogram(name, s[1], t, self.config.histogram_bins) if s is not None else None
            if png is not None:
                st = statistics(s[1])
                section.blocks.append(Figure(t["hist_caption"].format(
                    name=name, mean=format_number(st.mean), std=format_number(st.std),
                    min=format_number(st.minimum), max=format_number(st.maximum),
                    count=f"{st.count:,}"), png))
            self._step(name)
        return section
