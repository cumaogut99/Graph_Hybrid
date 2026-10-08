"""
Report charts rendered with matplotlib (Agg, no GUI) into PNG bytes.

Light report style: recessive grid and axes, thin lines, categorical colors in
a fixed order. One Y scale per panel: parameters of very different ranges get
stacked panels on a shared time axis instead of a second Y axis.
"""

import io
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.dates as mdates

# Categorical order (validated palette, light surface); never cycled past 8
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

MAX_SERIES = len(SERIES_COLORS)
WIDTH_IN = 10.0
DPI = 200


def _figure(height_in: float) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, height_in), dpi=DPI, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax, ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def _png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=DPI, facecolor=SURFACE, bbox_inches='tight', pad_inches=0.15)
    return buffer.getvalue()


def _time_axis(ax, t: np.ndarray, datetime_axis: bool, labels: dict) -> np.ndarray:
    """X values for matplotlib; epoch seconds become dates (UTC, as in the app)."""
    if not datetime_axis:
        ax.set_xlabel(labels["axis_time_s"], color=INK_SECONDARY, fontsize=10)
        return t
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel(labels["axis_time_utc"], color=INK_SECONDARY, fontsize=10)
    return t / 86400.0  # matplotlib dates: days since 1970-01-01


def _scales_compatible(series: Sequence[Tuple[np.ndarray, np.ndarray]]) -> bool:
    """True if the curves can share one Y scale without flattening any of them."""
    spans, lows, highs = [], [], []
    for _, y in series:
        finite = y[np.isfinite(y)]
        if len(finite) == 0:
            continue
        lo, hi = float(np.min(finite)), float(np.max(finite))
        lows.append(lo); highs.append(hi); spans.append(max(hi - lo, 1e-12))
    if len(spans) < 2:
        return True
    union = max(highs) - min(lows)
    # Each curve must use at least a quarter of the shared axis height
    return min(spans) / max(union, 1e-12) >= 0.25


def time_trend(names: List[str], series: List[Tuple[np.ndarray, np.ndarray]],
               datetime_axis: bool, labels: dict, layout: str = "auto") -> bytes:
    """
    Parameters over time. layout "overlay": one panel; "stacked": a panel per
    parameter; "auto": one panel only if the scales fit, else stacked.
    """
    names, series = names[:MAX_SERIES], series[:MAX_SERIES]
    if layout == "overlay":
        shared = True
    elif layout == "stacked":
        shared = len(series) == 1
    else:
        shared = _scales_compatible(series)
    panels = 1 if shared else len(series)
    height = 4.6 if panels == 1 else min(1.9 * panels + 0.6, 9.0)
    fig = _figure(height)
    axes = fig.subplots(panels, 1, sharex=True, squeeze=False)[:, 0]

    _time_axis(axes[-1], np.empty(0), datetime_axis, labels)  # labels the shared X axis
    for i, (name, (t, y)) in enumerate(zip(names, series)):
        ax = axes[0] if shared else axes[i]
        ax.plot(t / 86400.0 if datetime_axis else t, y, color=SERIES_COLORS[i], linewidth=1.2, label=name)
        if not shared:
            _style_axes(ax, name)
    if shared:
        _style_axes(axes[0], names[0] if len(names) == 1 else "")
        if len(names) > 1:
            axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                           loc='upper left', bbox_to_anchor=(0, 1.12), ncol=min(len(names), 4))
    else:
        for ax in axes[:-1]:
            ax.set_xlabel("")
    fig.align_ylabels(axes)
    return _png(fig)


def xy_scatter(x_name: str, y_names: List[str], x: np.ndarray, ys: List[np.ndarray]) -> bytes:
    """Y parameters against an X parameter (one dot per sample)."""
    y_names, ys = y_names[:3], ys[:3]  # dots overlap: at most 3 colors stay distinguishable
    fig = _figure(5.0)
    ax = fig.add_subplot(111)
    for i, (name, y) in enumerate(zip(y_names, ys)):
        ax.scatter(x, y, s=4, color=SERIES_COLORS[i], alpha=0.45, linewidths=0, label=name, rasterized=True)
    _style_axes(ax, y_names[0] if len(y_names) == 1 else "")
    ax.set_xlabel(x_name, color=INK_SECONDARY, fontsize=10)
    if len(y_names) > 1:
        legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, markerscale=3,
                           loc='upper left', bbox_to_anchor=(0, 1.12), ncol=len(y_names))
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    return _png(fig)


def histogram(name: str, values: np.ndarray, labels: dict, bins: int = 0) -> Optional[bytes]:
    """Distribution of a parameter with its mean and +/-1 std marked."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    fig = _figure(4.6)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins if bins > 0 else 'auto', color=SERIES_COLORS[0],
            edgecolor=SURFACE, linewidth=0.6)
    mean, std = float(np.mean(values)), float(np.std(values))
    ax.axvline(mean, color=INK, linewidth=1.2, label=f"{labels['mean']} {format_number(mean)}")
    ax.axvline(mean - std, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)),
               label=f"±1σ  [{format_number(mean - std)}, {format_number(mean + std)}]")
    ax.axvline(mean + std, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    # Legend above the plot: labels never collide with the bars
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    _style_axes(ax, labels["axis_samples"])
    ax.set_xlabel(name, color=INK_SECONDARY, fontsize=10)
    return _png(fig)


def format_number(value: float) -> str:
    """Compact engineering format for tables and labels."""
    if value is None or not np.isfinite(value):
        return "–"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e6 or magnitude < 1e-3):
        return f"{value:.3e}"
    if magnitude >= 100:
        return f"{value:,.1f}"
    return f"{value:.4g}"
