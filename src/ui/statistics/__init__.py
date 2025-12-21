# type: ignore
"""
Statistics UI Components

Modular components for the statistics panel:
- ClickableColorLabel: Color selection widget
- ClickableGroupBox: Draggable group box for graphs
- SignalRowWidget: Signal row with context menu
- PanelThemeApplier: Theme styling helper
"""

from .color_label import ClickableColorLabel
from .clickable_groupbox import ClickableGroupBox
from .signal_row import SignalRowWidget
from .panel_theme_applier import PanelThemeApplier

__all__ = [
    'ClickableColorLabel',
    'ClickableGroupBox',
    'SignalRowWidget',
    'PanelThemeApplier'
]

