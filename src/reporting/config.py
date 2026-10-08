"""
Report configuration: what goes into a report and how it is labelled.

Edited in the report window, saved/loaded as a JSON template.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List

FORMAT_PPTX = "pptx"
FORMAT_DOCX = "docx"

SCOPE_CURSORS = "cursors"   # data between the two cursors
SCOPE_ALL = "all"           # all data of the file

TIME_AXIS = "Time"          # X axis choice meaning "time" in a trend chart

# Trend chart layouts: one panel only if the scales fit (auto), always one
# panel, or one panel per parameter on a shared time axis
LAYOUT_AUTO = "auto"
LAYOUT_OVERLAY = "overlay"
LAYOUT_STACKED = "stacked"


@dataclass
class OperatingPointFilter:
    """Extra condition on the rows of an operating point: param <op> value."""
    param: str = ""
    op: str = "<"            # '<' or '>'
    value: float = 0.0


@dataclass
class OperatingPoints:
    """Statistics per operating point: rows where |control - value| <= tolerance."""
    enabled: bool = False
    control_param: str = ""
    values: List[float] = field(default_factory=list)
    tolerance: float = 0.5
    filters: List[OperatingPointFilter] = field(default_factory=list)


@dataclass
class TrendChart:
    """One chart: Y parameters against time (x == TIME_AXIS) or against a parameter."""
    x: str = TIME_AXIS
    y: List[str] = field(default_factory=list)
    title: str = ""
    layout: str = LAYOUT_AUTO


@dataclass
class ReportConfig:
    title: str = ""          # empty: the default title of the report language
    project: str = ""
    author: str = ""
    notes: str = ""
    output_format: str = FORMAT_PPTX
    language: str = "tr"     # see texts.LANGUAGES
    scope: str = SCOPE_CURSORS

    summary_enabled: bool = True

    statistics_enabled: bool = True
    statistics_params: List[str] = field(default_factory=list)
    operating_points: OperatingPoints = field(default_factory=OperatingPoints)

    trends_enabled: bool = True
    trends: List[TrendChart] = field(default_factory=list)

    histograms_enabled: bool = False
    histogram_params: List[str] = field(default_factory=list)
    histogram_bins: int = 0  # 0: automatic

    # ------------------------------------------------------------------
    # JSON templates
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReportConfig":
        """Build from a template dict; unknown keys are ignored, missing keys keep defaults."""
        config = cls()
        for key, value in (data or {}).items():
            if key == "operating_points" and isinstance(value, dict):
                op = OperatingPoints()
                for k, v in value.items():
                    if k == "filters":
                        op.filters = [OperatingPointFilter(**_known(OperatingPointFilter, f)) for f in v or []]
                    elif hasattr(op, k):
                        setattr(op, k, v)
                op.values = [float(v) for v in op.values]
                config.operating_points = op
            elif key == "trends":
                config.trends = [TrendChart(**_known(TrendChart, t)) for t in value or []]
            elif hasattr(config, key):
                setattr(config, key, value)
        return config

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReportConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Parameters the report needs
    # ------------------------------------------------------------------
    def used_parameters(self) -> List[str]:
        """Every parameter referenced by an enabled section."""
        names = []
        if self.statistics_enabled:
            names += self.statistics_params
            if self.operating_points.enabled:
                names.append(self.operating_points.control_param)
                names += [f.param for f in self.operating_points.filters]
        if self.trends_enabled:
            for chart in self.trends:
                names += chart.y
                if chart.x != TIME_AXIS:
                    names.append(chart.x)
        if self.histograms_enabled:
            names += self.histogram_params
        return [n for n in dict.fromkeys(names) if n]


def _known(cls, data: dict) -> dict:
    """Only the keys the dataclass has."""
    return {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
