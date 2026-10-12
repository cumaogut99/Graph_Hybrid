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
from .config import ConditionGroup, RangeCondition, ReportConfig, TIME_AXIS
from .data_source import ReportContext, ReportData, envelope, join_segments, statistics, thin
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
        groups = len(c.condition_groups)
        self._steps_total = 3 \
            + (len(c.histogram_params) if c.histograms_enabled else 0) \
            + (len(c.trends) if c.trends_enabled else 0) \
            + (groups if c.statistics_enabled else 0) \
            + (groups if c.filtered_charts_enabled else 0)

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
        if c.histograms_enabled and c.histogram_params:
            content.append(self._histograms_section())
        if c.trends_enabled and c.trends:
            content.append(self._trends_section())
        if c.statistics_enabled and c.statistics_params:
            content.append(self._statistics_section())
        if c.filtered_charts_enabled and c.filtered_chart_params and c.condition_groups:
            content.append(self._filtered_charts_section())
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
        return [t["h_parameter"], t["h_samples"], t["h_min"], t["h_mean"], t["h_max"], t["h_std"]]

    @staticmethod
    def _stat_row(name: str, st) -> List[str]:
        return [name, f"{st.count:,}", format_number(st.minimum), format_number(st.mean),
                format_number(st.maximum), format_number(st.std)]

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
    # Condition groups
    # ------------------------------------------------------------------
    def _group_name(self, index: int, group: ConditionGroup) -> str:
        return group.name.strip() or self.t["group_default"].format(n=index + 1)

    @staticmethod
    def _active(group: ConditionGroup) -> List[RangeCondition]:
        """Conditions that limit something (a parameter and at least one bound)."""
        return [c for c in group.conditions
                if c.param and (c.minimum is not None or c.maximum is not None)]

    def _conditions_text(self, group: ConditionGroup) -> str:
        t = self.t
        parts = []
        for c in self._active(group):
            if c.minimum is not None and c.maximum is not None:
                parts.append(f"{format_number(c.minimum)} ≤ {c.param} ≤ {format_number(c.maximum)}")
            elif c.minimum is not None:
                parts.append(f"{c.param} ≥ {format_number(c.minimum)}")
            else:
                parts.append(f"{c.param} ≤ {format_number(c.maximum)}")
        if not parts:
            return t["group_no_conditions"]
        return t["group_conditions"].format(conditions=t["group_and"].join(parts))

    def _group_rows(self, group: ConditionGroup, params: List[str]):
        """
        (aligned columns, row mask) of `params` for the rows that meet the
        group's conditions, on the time base of the first parameter;
        (None, missing names) if a parameter has no data in range.
        """
        conditions = self._active(group)
        names = list(dict.fromkeys(params + [c.param for c in conditions]))
        missing = [n for n in names if self.data.series(n) is None]
        if missing:
            return None, missing
        aligned = self.data.aligned(names[0], names[1:])
        if aligned is None:
            return None, names
        mask = np.ones(len(aligned['__time__']), dtype=bool)
        for c in conditions:
            values = aligned[c.param]
            if c.minimum is not None:
                mask &= values >= c.minimum
            if c.maximum is not None:
                mask &= values <= c.maximum
        return aligned, mask

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _statistics_section(self) -> Section:
        t = self.t
        section = Section(t["statistics"])
        params = self.config.statistics_params
        groups = self.config.condition_groups

        # Tables per condition group; without groups one table over the whole range
        if not groups:
            rows, missing = [], []
            for name in params:
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

        for index, group in enumerate(groups):
            name = self._group_name(index, group)
            section.blocks.append(Paragraph(f"{name}: {self._conditions_text(group)}"))
            aligned, mask = self._group_rows(group, params)
            if aligned is None:
                section.blocks.append(Paragraph(t["group_missing"].format(name=name, names=", ".join(mask))))
            else:
                count = int(np.count_nonzero(mask))
                rows = []
                for p in params:
                    st = statistics(aligned[p][mask]) if count else None
                    if st:
                        rows.append(self._stat_row(p, st))
                if rows:
                    section.blocks.append(Table(t["group_table"].format(name=name, count=f"{count:,}"),
                                                self._stat_headers(), rows))
                else:
                    section.blocks.append(Paragraph(t["group_empty"].format(name=name)))
            self._step(name)
        return section

    # ------------------------------------------------------------------
    # Charts of the condition groups
    # ------------------------------------------------------------------
    def _filtered_charts_section(self) -> Section:
        t = self.t
        section = Section(t["filtered_charts"])
        params = self.config.filtered_chart_params
        for index, group in enumerate(self.config.condition_groups):
            name = self._group_name(index, group)
            section.blocks.append(Paragraph(f"{name}: {self._conditions_text(group)}"))
            aligned, mask = self._group_rows(group, params)
            if aligned is None:
                section.blocks.append(Paragraph(t["group_missing"].format(name=name, names=", ".join(mask))))
            elif not np.any(mask):
                section.blocks.append(Paragraph(t["group_empty"].format(name=name)))
            else:
                joined, starts = join_segments(aligned['__time__'], mask)
                series = [envelope(joined, aligned[p][mask]) for p in params]
                png = charts.time_trend(params, series, False, t, self.config.filtered_chart_layout,
                                        x_label=t["axis_joined_time"], boundaries=starts)
                section.blocks.append(Figure(t["filtered_caption"].format(
                    name=name, count=f"{len(joined):,}", segments=f"{len(starts) + 1:,}"), png))
            self._step(name)
        return section

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
