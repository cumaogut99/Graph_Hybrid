# type: ignore
"""
Toolbar Manager for Time Graph Widget

Manages the toolbar interface including:
- Panel toggles
- Graph and Tab count controls
- Normalization controls
- File operations

Note: Cursor mode is permanently set to 'dual' (no UI control)
"""

import logging
import os  # ✅ FIX: Import os for absolute path handling
from typing import TYPE_CHECKING, Optional
from PyQt5.QtWidgets import (
    QToolBar, QToolButton, QButtonGroup, QLabel, QSpinBox, QPushButton, QFrame, QHBoxLayout, QComboBox, QMenu, QAction,
    QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QFont

from src.ui.frameless_window import DRAG_AREA_PROPERTY, WindowControls

if TYPE_CHECKING:
    from ..widgets.time_graph_widget_refactored import TimeGraphWidget

logger = logging.getLogger(__name__)

class ToolbarManager(QObject):
    """Manages the toolbar and its controls for the Time Graph Widget."""
    
    # Signals
    # NOTE: cursor_mode_changed signal removed - cursor mode is now permanently set to 'dual'
    panel_toggled = Signal()
    settings_toggled = Signal()
    graph_settings_toggled = Signal()
    statistics_settings_toggled = Signal()
    parameters_toggled = Signal()  # New parameters panel toggle
    correlations_toggled = Signal()
    bitmask_toggled = Signal()
    filters_requested = Signal()  # open the range filter (Advanced Settings) dialog
    report_requested = Signal()  # open the report window
    graph_count_changed = Signal(int)
    tab_count_changed = Signal(int)
    file_open_requested = Signal()
    file_save_requested = Signal()
    file_exit_requested = Signal()
    layout_import_requested = Signal()
    layout_export_requested = Signal()
    # Project file signals (.mpai)
    project_save_requested = Signal()
    project_open_requested = Signal()
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.toolbar = None
        self.cursor_group = None
        self.normalize_btn = None
        
        self.current_tab_count = 1
        self.current_graph_count = 1
        
        self._setup_toolbar()
        self._setup_connections()
    
    def _setup_toolbar(self):
        """Setup the toolbar with panel toggles and actions."""
        self.toolbar = QToolBar("Analysis Tools")
        self.toolbar.setMovable(False)
        self.toolbar.setFixedHeight(39)

        # The toolbar is also the window's title bar: its empty space moves the
        # window and the caption buttons sit at its right end (frameless window)
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setAttribute(Qt.WA_StyledBackground, True)
        self.title_bar.setProperty(DRAG_AREA_PROPERTY, True)
        self.title_bar.setFixedHeight(40)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 1)  # bottom: border line
        title_layout.setSpacing(0)
        title_layout.addWidget(self.toolbar, 1)
        self.window_controls = WindowControls()
        title_layout.addWidget(self.window_controls, 0, Qt.AlignTop)

        # Apply theme-based styling
        self._apply_theme_styling()

        # File button on the far left
        self._create_file_control()

        # Settings / panel buttons
        self._create_settings_control()

        # Cursor mode controls (no UI: always dual cursor)
        self._create_cursor_controls()

        # Range filters (Advanced Settings dialog, applies to all graphs)
        self.filters_btn = self._make_button("Filters", "Range filters for all graphs")
        self.filters_btn.clicked.connect(self.filters_requested.emit)
        self.toolbar.addWidget(self.filters_btn)

        # Graph count controls
        graph_count_label = QLabel("Graphs:")
        graph_count_label.setObjectName("groupLabel")
        self.toolbar.addWidget(graph_count_label)
        self.graph_count_spinbox = QSpinBox()
        self.graph_count_spinbox.setRange(1, 10)
        self.graph_count_spinbox.setValue(1)
        self.graph_count_spinbox.setAlignment(Qt.AlignCenter)
        self.graph_count_spinbox.setToolTip("Number of graphs to display in the current tab")
        self.toolbar.addWidget(self.graph_count_spinbox)

        # Statistics panel toggle
        self.panel_toggle_btn = self._make_button("Statistics", "Toggle signal statistics panel", checked=True)
        self.panel_toggle_btn.clicked.connect(self.panel_toggled.emit)
        self.toolbar.addWidget(self.panel_toggle_btn)

        # Statistics settings button
        self.statistics_settings_btn = self._make_button("Statistics Settings", "Configure which statistics to display", checked=False)
        self.statistics_settings_btn.clicked.connect(self.statistics_settings_toggled.emit)
        self.toolbar.addWidget(self.statistics_settings_btn)

        # Correlations button
        self.correlations_btn = self._make_button("Correlations", "Open correlations analysis panel", checked=False)
        self.correlations_btn.clicked.connect(self.correlations_toggled.emit)
        self.toolbar.addWidget(self.correlations_btn)

        # Bitmask button
        self.bitmask_btn = self._make_button("Bitmask", "Open bitmask analysis panel", checked=False)
        self.bitmask_btn.clicked.connect(self.bitmask_toggled.emit)
        self.toolbar.addWidget(self.bitmask_btn)

        # Report window (PowerPoint / Word report of the active file)
        self.report_btn = self._make_button("Report", "Create a PowerPoint or Word report from the active file")
        self.report_btn.clicked.connect(self.report_requested.emit)
        self.toolbar.addWidget(self.report_btn)

    def _make_button(self, text: str, tooltip: str, checked: Optional[bool] = None) -> QToolButton:
        """Create a toolbar button; checked=None makes a plain (non-toggle) button."""
        btn = QToolButton()
        btn.setObjectName("textButton")
        btn.setText(text)
        btn.setToolTip(tooltip)
        if checked is not None:
            btn.setCheckable(True)
            btn.setChecked(checked)
        return btn

    def set_filter_active(self, active: bool):
        """Highlight the Filters button while a range filter is applied."""
        if getattr(self, 'filters_btn', None) is None:
            return
        self.filters_btn.setProperty("active", active)
        self.filters_btn.setToolTip(
            "Range filter is active (all graphs) - click to edit" if active
            else "Range filters for all graphs"
        )
        # Re-evaluate [active="true"] in the stylesheet
        self.filters_btn.style().unpolish(self.filters_btn)
        self.filters_btn.style().polish(self.filters_btn)

    def _apply_theme_styling(self):
        """Apply theme-based styling to the toolbar."""
        # Get theme colors from parent's theme manager
        if hasattr(self.parent, 'theme_manager'):
            colors = self.parent.theme_manager.get_theme_colors()
        else:
            # Fallback colors for space theme
            colors = {
                'surface': '#1a2332',
                'surface_variant': '#2d4a66',
                'primary': '#4a90e2',
                'primary_variant': '#6bb6ff',
                'text_primary': '#ffffff',
                'text_secondary': '#e6f3ff',
                'border': '#4a90e2',
                'hover': '#3a5f7a'
            }

        # Button look: dark fill, thin teal outline, rounded corners,
        # regular-weight light-gray text
        accent = '#1f6b73'        # outline
        accent_hover = '#2c99a5'
        accent_active = '#35b6c4'
        button_bg = 'rgba(0, 0, 0, 0.28)'
        button_text = '#d4dade'

        self.title_bar.setStyleSheet(f"""
            QWidget#titleBar {{
                background: {colors['surface']};
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }}
        """)
        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                background: transparent;
                border: none;
                spacing: 4px;
                padding: 3px 6px;
            }}
            QToolBar::separator {{
                width: 0px;
                margin: 0px;
            }}
            QToolButton {{
                background: {button_bg};
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 3px 8px;
                margin: 0px;
                color: {button_text};
                font-size: 13px;
                font-weight: 400;
                min-height: 22px;
            }}
            QToolButton:hover {{
                border-color: {accent_hover};
                background: rgba(44, 153, 165, 0.10);
                color: #ffffff;
            }}
            QToolButton:pressed {{
                background: rgba(44, 153, 165, 0.20);
            }}
            QToolButton:checked, QToolButton[active="true"] {{
                border-color: {accent_active};
                background: rgba(53, 182, 196, 0.18);
                color: #ffffff;
            }}
            QToolButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
            QLabel#groupLabel {{
                background: transparent;
                border: none;
                color: {button_text};
                font-size: 13px;
                font-weight: 400;
                margin: 0px 0px 0px 4px;
            }}
            QComboBox {{
                background: {button_bg};
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 3px 8px;
                color: {button_text};
                font-size: 13px;
                min-width: 80px;
                min-height: 22px;
            }}
            QComboBox:hover {{
                border-color: {accent_hover};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['surface']};
                border: 1px solid {accent};
                selection-background-color: {accent};
                color: {button_text};
                font-size: 13px;
            }}
            QSpinBox {{
                background: {button_bg};
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 3px 2px;  /* Qt reserves the -/+ button widths itself */
                color: {button_text};
                font-size: 13px;
                min-width: 80px;
                min-height: 22px;
            }}
            QSpinBox:hover {{
                border-color: {accent_hover};
            }}
            /* Stepper layout: [-] value [+], full-height buttons that are easy to hit */
            QSpinBox::up-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                background: transparent;
                border: none;
                border-radius: 5px;
                width: 26px;
                height: 31px;  /* full spinbox height inside the border */
            }}
            QSpinBox::up-button {{
                subcontrol-position: right;
            }}
            QSpinBox::down-button {{
                subcontrol-position: left;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background: rgba(44, 153, 165, 0.25);
            }}
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{
                background: rgba(53, 182, 196, 0.4);
            }}
            QSpinBox::up-arrow {{
                image: url({os.path.abspath('icons/spin-plus.svg').replace(os.sep, '/')});
                width: 13px;
                height: 13px;
            }}
            QSpinBox::down-arrow {{
                image: url({os.path.abspath('icons/spin-minus.svg').replace(os.sep, '/')});
                width: 13px;
                height: 13px;
            }}
        """)

    def _create_file_control(self):
        """Create the file menu button."""
        self.file_btn = QToolButton()
        self.file_btn.setObjectName("textButton")
        self.file_btn.setText("File")
        self.file_btn.setToolTip("File operations")
        self.file_btn.setPopupMode(QToolButton.InstantPopup)
        
        # Create file menu
        file_menu = QMenu()
        
        # Project file operations (.mpai)
        open_project_action = QAction("Open Project (.mpai)", self)
        open_project_action.setShortcut("Ctrl+Shift+O")
        open_project_action.setToolTip("Open a complete project file (data + layout)")
        open_project_action.triggered.connect(self.project_open_requested.emit)
        file_menu.addAction(open_project_action)
        
        save_project_action = QAction("Save Project (.mpai)", self)
        save_project_action.setShortcut("Ctrl+Shift+S")
        save_project_action.setToolTip("Save complete project (data + layout) to a single file")
        save_project_action.triggered.connect(self.project_save_requested.emit)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        # Legacy CSV operations
        open_action = QAction("Open Data File", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.file_open_requested.emit)
        file_menu.addAction(open_action)
        


        file_menu.addSeparator()

        # Layout operations
        import_layout_action = QAction("Import Layout", self)
        import_layout_action.triggered.connect(self.layout_import_requested.emit)
        file_menu.addAction(import_layout_action)

        export_layout_action = QAction("Export Layout", self)
        export_layout_action.triggered.connect(self.layout_export_requested.emit)
        file_menu.addAction(export_layout_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.file_exit_requested.emit)
        file_menu.addAction(exit_action)
        
        self.file_btn.setMenu(file_menu)
        self.toolbar.addWidget(self.file_btn)

    def _create_settings_control(self):
        """Create the settings panel toggle button."""
        self.settings_btn = QToolButton()
        self.settings_btn.setObjectName("textButton")
        self.settings_btn.setText("General")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setChecked(False)
        self.settings_btn.setToolTip("Toggle general settings panel")
        self.settings_btn.clicked.connect(self.settings_toggled.emit)
        self.toolbar.addWidget(self.settings_btn)

        # Add the new Graph Settings button
        self.graph_settings_btn = QToolButton()
        self.graph_settings_btn.setObjectName("textButton")
        self.graph_settings_btn.setText("Graph Settings")
        self.graph_settings_btn.setCheckable(True)
        self.graph_settings_btn.setChecked(False)
        self.graph_settings_btn.setToolTip("Toggle graph settings panel")
        self.graph_settings_btn.clicked.connect(self.graph_settings_toggled.emit)
        self.toolbar.addWidget(self.graph_settings_btn)

        # Add Parameters button
        self.parameters_btn = QToolButton()
        self.parameters_btn.setObjectName("textButton")
        self.parameters_btn.setText("Parameters")
        self.parameters_btn.setCheckable(True)
        self.parameters_btn.setChecked(False)
        self.parameters_btn.setToolTip("Toggle parameters panel")
        self.parameters_btn.clicked.connect(self.parameters_toggled.emit)
        self.toolbar.addWidget(self.parameters_btn)

    def _create_cursor_controls(self):
        """Cursor mode is now permanently set to 'dual' - no UI control needed."""
        # Cursor mode removed from toolbar - always uses dual cursor mode
        pass


    def _setup_connections(self):
        """Connect signals and slots for toolbar widgets."""
        # Cursor mode combo box removed - always dual mode
        
        # Connect graph controls
        self.graph_count_spinbox.valueChanged.connect(self.graph_count_changed.emit)
        
    def get_toolbar(self) -> QToolBar:
        """Get the configured toolbar."""
        return self.toolbar

    def get_title_bar(self) -> QWidget:
        """The toolbar together with the window's caption buttons."""
        return self.title_bar

    def set_graph_count(self, count: int):
        self.graph_count_spinbox.setValue(count)

    def set_tab_count(self, count: int):
        """Set the value of the tab count spinbox."""
        pass # This is now deprecated

    def update_theme(self):
        """Update toolbar styling when theme changes."""
        self._apply_theme_styling()
