"""
Graph Settings Panel Manager for Time Graph Widget
"""

import logging
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QScrollArea,
    QGroupBox, QFormLayout, QLabel, QPushButton, QCheckBox, QHBoxLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal as Signal

if TYPE_CHECKING:
    from time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class GraphSettingsPanelManager(QObject):
    """Manages the graph settings panel interface."""
    
    # Per-graph signals (for right-click context menu)
    normalization_toggled = Signal(int, bool) # graph_index, is_checked
    view_reset_requested = Signal(int) # graph_index
    grid_visibility_changed = Signal(int, bool) # graph_index, is_visible
    autoscale_changed = Signal(int, bool) # graph_index, is_enabled
    
    # Global signals (for panel settings applied to all graphs)
    global_normalization_toggled = Signal(bool) # is_checked
    global_view_reset_requested = Signal() # no args
    global_grid_visibility_changed = Signal(bool) # is_visible
    global_autoscale_changed = Signal(bool) # is_enabled
    global_legend_visibility_changed = Signal(bool) # is_visible
    global_tooltips_changed = Signal(bool) # is_enabled
    global_snap_to_data_changed = Signal(bool) # is_enabled
    global_line_width_changed = Signal(int) # width
    global_x_axis_mouse_changed = Signal(bool) # is_enabled
    global_y_axis_mouse_changed = Signal(bool) # is_enabled
    global_secondary_axis_changed = Signal(bool) # is_enabled

    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.settings_panel = None
        
        # Store global settings state
        self.global_settings = {
            'normalize': False,
            'show_grid': True,
            'autoscale': True,
            'show_legend': True,
            'show_tooltips': False,  # Default to False - user can enable if needed
            'snap_to_data': False,
            'line_width': 1,
            'x_axis_mouse': True,
            'y_axis_mouse': True,
            'secondary_axis': False,
        }
        
        self._setup_settings_panel()
    
    def _setup_settings_panel(self):
        """Create the main settings panel."""
        self.settings_panel = QFrame()
        self.settings_panel.setMinimumWidth(280)
        self.settings_panel.setMaximumWidth(350)
        self.settings_panel.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #444;
                border-radius: 8px;
            }
            QGroupBox {
                color: #e0e0e0;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: #383838;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #5d9cec;
            }
            QLabel {
                color: #d0d0d0;
                font-size: 10px;
            }
        """)
        
        # Create main container layout
        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Create global settings group
        self._create_global_settings_group(main_layout)
        
        main_layout.addStretch()
        
        panel_layout = QVBoxLayout(self.settings_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(container_widget)

    def _create_global_settings_group(self, parent_layout):
        """Create global settings group that applies to all graphs."""
        group = QGroupBox("🌐 Global Graph Settings")
        
        # Apply theme-appropriate styling
        self._apply_group_styling(group)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(8)
        
        # Create all controls in single column layout
        
        # Normalize checkbox
        self.global_normalize_check = QCheckBox("Normalize All Graphs")
        self.global_normalize_check.setChecked(self.global_settings['normalize'])
        self.global_normalize_check.toggled.connect(self._on_global_normalize_toggled)
        layout.addWidget(self.global_normalize_check)
        
        # Grid checkbox
        self.global_grid_check = QCheckBox("Show Grid (All)")
        self.global_grid_check.setChecked(self.global_settings['show_grid'])
        self.global_grid_check.toggled.connect(self._on_global_grid_toggled)
        layout.addWidget(self.global_grid_check)
        
        # Autoscale checkbox
        self.global_autoscale_check = QCheckBox("Auto-scale Y-Axis (All)")
        self.global_autoscale_check.setChecked(self.global_settings['autoscale'])
        self.global_autoscale_check.toggled.connect(self._on_global_autoscale_toggled)
        layout.addWidget(self.global_autoscale_check)
        
        # Legend checkbox
        self.global_legend_check = QCheckBox("Show Legend (All)")
        self.global_legend_check.setChecked(self.global_settings['show_legend'])
        self.global_legend_check.toggled.connect(self._on_global_legend_toggled)
        layout.addWidget(self.global_legend_check)
        
        # Show Tooltips checkbox
        self.global_tooltips_check = QCheckBox("Show Tooltips")
        self.global_tooltips_check.setChecked(self.global_settings['show_tooltips'])
        self.global_tooltips_check.toggled.connect(self._on_global_tooltips_toggled)
        layout.addWidget(self.global_tooltips_check)
        
        # Cursor Snap to Data Points checkbox
        self.global_snap_check = QCheckBox("Cursor Snap to Data Points")
        self.global_snap_check.setChecked(self.global_settings['snap_to_data'])
        self.global_snap_check.toggled.connect(self._on_global_snap_toggled)
        layout.addWidget(self.global_snap_check)

        # Line Width control
        line_width_layout = QHBoxLayout()
        line_width_label = QLabel("Line Width:")
        self.global_line_width_spin = QSpinBox()
        self.global_line_width_spin.setRange(1, 10)
        self.global_line_width_spin.setValue(self.global_settings['line_width'])
        self.global_line_width_spin.valueChanged.connect(self._on_global_line_width_changed)
        line_width_layout.addWidget(line_width_label)
        line_width_layout.addWidget(self.global_line_width_spin)
        line_width_layout.addStretch()
        layout.addLayout(line_width_layout)
        
        # X Axis Mouse Enabled checkbox
        self.global_x_mouse_check = QCheckBox("X Axis Mouse Enabled")
        self.global_x_mouse_check.setChecked(self.global_settings['x_axis_mouse'])
        self.global_x_mouse_check.toggled.connect(self._on_global_x_mouse_toggled)
        layout.addWidget(self.global_x_mouse_check)
        
        # Y Axis Mouse Enabled checkbox
        self.global_y_mouse_check = QCheckBox("Y Axis Mouse Enabled")
        self.global_y_mouse_check.setChecked(self.global_settings['y_axis_mouse'])
        self.global_y_mouse_check.toggled.connect(self._on_global_y_mouse_toggled)
        layout.addWidget(self.global_y_mouse_check)
        
        # Secondary Axis checkbox
        self.global_secondary_axis_check = QCheckBox("Enable Secondary Y-Axis")
        self.global_secondary_axis_check.setChecked(self.global_settings['secondary_axis'])
        self.global_secondary_axis_check.toggled.connect(self._on_global_secondary_axis_toggled)
        layout.addWidget(self.global_secondary_axis_check)
        
        # Zoom to Cursors button
        self.zoom_to_cursors_button = QPushButton("🎯 Zoom to Cursors")
        self.zoom_to_cursors_button.setToolTip("Zoom to the range between dual cursors")
        self.zoom_to_cursors_button.clicked.connect(self._on_zoom_to_cursors_clicked)
        layout.addWidget(self.zoom_to_cursors_button)
        
        # Reset button
        self.global_reset_button = QPushButton("Reset All Views")
        self.global_reset_button.clicked.connect(self._on_global_reset_clicked)
        layout.addWidget(self.global_reset_button)
        
        # Info label
        info_label = QLabel("💡 These settings apply to all graphs simultaneously")
        info_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-style: italic;
                font-size: 9px;
                padding: 5px;
            }
        """)
        layout.addWidget(info_label)
        
        parent_layout.addWidget(group)

    def _apply_group_styling(self, group):
        """Apply theme-appropriate styling to a group box."""
        # Get theme colors from parent or use fallback
        if hasattr(self.parent, 'theme_manager'):
            theme_colors = self.parent.theme_manager.get_theme_colors()
        else:
            # Fallback to space theme colors
            theme_colors = {
                'text_primary': '#e6f3ff',
                'text_secondary': '#ffffff',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'border': '#4a90e2',
                'primary': '#4a90e2'
            }
        
        # Determine if this is a light theme
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        
        if is_light_theme:
            title_bg_color = 'rgba(255, 255, 255, 0.8)'
        else:
            title_bg_color = 'rgba(0, 0, 0, 0.3)'
        
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {theme_colors.get('border', '#4a90e2')};
                border-radius: 12px;
                margin-top: 10px;
                padding: 10px;
                background-color: {theme_colors.get('surface', '#2d4a66')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
                background-color: {title_bg_color};
                border-radius: 6px;
                color: {theme_colors.get('primary', '#4a90e2')};
            }}
            QCheckBox {{
                color: {theme_colors.get('text_primary', '#e6f3ff')};
                font-size: 12px;
                font-weight: 500;
                spacing: 8px;
            }}
            QPushButton {{
                background-color: {theme_colors.get('primary', '#4a90e2')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme_colors.get('hover', theme_colors.get('primary', '#4a90e2'))};
            }}
            QSpinBox {{
                background-color: {theme_colors.get('surface_variant', '#3a5f7a')};
                border: 1px solid {theme_colors.get('border', '#4a90e2')};
                border-radius: 4px;
                padding: 4px;
                color: {theme_colors.get('text_primary', '#e6f3ff')};
                font-size: 12px;
                min-width: 60px;
            }}
            QSpinBox:hover {{
                border-color: {theme_colors.get('primary', '#4a90e2')};
            }}
            QLabel {{
                color: {theme_colors.get('text_secondary', '#ffffff')};
                font-size: 12px;
                font-weight: 500;
            }}
        """)

    def _on_global_normalize_toggled(self, checked):
        """Handle global normalization toggle."""
        self.global_settings['normalize'] = checked
        self.global_normalization_toggled.emit(checked)
        logger.info(f"Global normalization {'enabled' if checked else 'disabled'}")

    def _on_global_reset_clicked(self):
        """Handle global reset button click."""
        self.global_view_reset_requested.emit()
        logger.info("Global view reset requested")

    def _on_global_grid_toggled(self, checked):
        """Handle global grid visibility toggle."""
        self.global_settings['show_grid'] = checked
        self.global_grid_visibility_changed.emit(checked)
        logger.info(f"Global grid {'shown' if checked else 'hidden'}")

    def _on_global_autoscale_toggled(self, checked):
        """Handle global autoscale toggle."""
        self.global_settings['autoscale'] = checked
        self.global_autoscale_changed.emit(checked)
        logger.info(f"Global autoscale {'enabled' if checked else 'disabled'}")

    def _on_global_legend_toggled(self, checked):
        """Handle global legend visibility toggle."""
        self.global_settings['show_legend'] = checked
        self.global_legend_visibility_changed.emit(checked)
        logger.info(f"Global legend {'shown' if checked else 'hidden'}")

    def _on_global_tooltips_toggled(self, checked):
        """Handle global tooltips toggle."""
        self.global_settings['show_tooltips'] = checked
        self.global_tooltips_changed.emit(checked)
        logger.info(f"Global tooltips {'enabled' if checked else 'disabled'}")

    def _on_global_snap_toggled(self, checked):
        """Handle global snap to data toggle."""
        self.global_settings['snap_to_data'] = checked
        self.global_snap_to_data_changed.emit(checked)
        logger.info(f"Global snap to data {'enabled' if checked else 'disabled'}")
        
    def _on_global_line_width_changed(self, width):
        """Handle global line width change."""
        self.global_settings['line_width'] = width
        self.global_line_width_changed.emit(width)
        logger.info(f"Global line width set to {width}")

    def _on_global_x_mouse_toggled(self, checked):
        """Handle global X axis mouse toggle."""
        self.global_settings['x_axis_mouse'] = checked
        self.global_x_axis_mouse_changed.emit(checked)
        logger.info(f"Global X axis mouse {'enabled' if checked else 'disabled'}")

    def _on_global_y_mouse_toggled(self, checked):
        """Handle global Y axis mouse toggle."""
        self.global_settings['y_axis_mouse'] = checked
        self.global_y_axis_mouse_changed.emit(checked)
        logger.info(f"Global Y axis mouse {'enabled' if checked else 'disabled'}")
    
    def _on_global_secondary_axis_toggled(self, checked):
        """Handle global secondary axis toggle."""
        self.global_settings['secondary_axis'] = checked
        self.global_secondary_axis_changed.emit(checked)
        logger.info(f"Global secondary axis {'enabled' if checked else 'disabled'}")

    def rebuild_controls(self, graph_count: int):
        """No longer needed - using global controls only."""
        # Global controls don't need rebuilding based on graph count
        logger.info(f"Graph count changed to {graph_count}, but using global controls")

    def update_theme(self):
        """Update the panel styling when theme changes."""
        # Re-apply styling to the global settings group
        if hasattr(self, 'global_normalize_check'):
            # Find the parent group box and re-apply styling
            group = self.global_normalize_check.parent()
            while group and not isinstance(group, QGroupBox):
                group = group.parent()
            if group:
                self._apply_group_styling(group)

    def get_settings_panel(self) -> QWidget:
        """Get the settings panel widget."""
        return self.settings_panel

    def sync_global_settings_from_right_click(self, settings: dict):
        """Synchronize global settings when changed via right-click menu."""
        # Update global settings state
        if 'normalize' in settings:
            self.global_settings['normalize'] = settings['normalize']
            if hasattr(self, 'global_normalize_check'):
                self.global_normalize_check.blockSignals(True)
                self.global_normalize_check.setChecked(settings['normalize'])
                self.global_normalize_check.blockSignals(False)
        
        if 'show_grid' in settings:
            self.global_settings['show_grid'] = settings['show_grid']
            if hasattr(self, 'global_grid_check'):
                self.global_grid_check.blockSignals(True)
                self.global_grid_check.setChecked(settings['show_grid'])
                self.global_grid_check.blockSignals(False)
        
        if 'autoscale' in settings:
            self.global_settings['autoscale'] = settings['autoscale']
            if hasattr(self, 'global_autoscale_check'):
                self.global_autoscale_check.blockSignals(True)
                self.global_autoscale_check.setChecked(settings['autoscale'])
                self.global_autoscale_check.blockSignals(False)
        
        if 'show_legend' in settings:
            self.global_settings['show_legend'] = settings['show_legend']
            if hasattr(self, 'global_legend_check'):
                self.global_legend_check.blockSignals(True)
                self.global_legend_check.setChecked(settings['show_legend'])
                self.global_legend_check.blockSignals(False)
        
        if 'show_tooltips' in settings:
            self.global_settings['show_tooltips'] = settings['show_tooltips']
            if hasattr(self, 'global_tooltips_check'):
                self.global_tooltips_check.blockSignals(True)
                self.global_tooltips_check.setChecked(settings['show_tooltips'])
                self.global_tooltips_check.blockSignals(False)
        
        if 'snap_to_data' in settings:
            self.global_settings['snap_to_data'] = settings['snap_to_data']
            if hasattr(self, 'global_snap_check'):
                self.global_snap_check.blockSignals(True)
                self.global_snap_check.setChecked(settings['snap_to_data'])
                self.global_snap_check.blockSignals(False)
        
        if 'line_width' in settings:
            self.global_settings['line_width'] = settings['line_width']
            if hasattr(self, 'global_line_width_spin'):
                self.global_line_width_spin.blockSignals(True)
                self.global_line_width_spin.setValue(settings['line_width'])
                self.global_line_width_spin.blockSignals(False)
        
        if 'x_axis_mouse' in settings:
            self.global_settings['x_axis_mouse'] = settings['x_axis_mouse']
            if hasattr(self, 'global_x_mouse_check'):
                self.global_x_mouse_check.blockSignals(True)
                self.global_x_mouse_check.setChecked(settings['x_axis_mouse'])
                self.global_x_mouse_check.blockSignals(False)
        
        if 'y_axis_mouse' in settings:
            self.global_settings['y_axis_mouse'] = settings['y_axis_mouse']
            if hasattr(self, 'global_y_mouse_check'):
                self.global_y_mouse_check.blockSignals(True)
                self.global_y_mouse_check.setChecked(settings['y_axis_mouse'])
                self.global_y_mouse_check.blockSignals(False)
        
        
        logger.info(f"Global settings synced from right-click: {settings}")

    def get_global_settings(self) -> dict:
        """Get current global settings."""
        return self.global_settings.copy()
    
    def _on_zoom_to_cursors_clicked(self):
        """Handle zoom to cursors button click."""
        try:
            # Get cursor manager from parent
            cursor_manager = getattr(self.parent, 'cursor_manager', None)
            if not cursor_manager:
                logger.warning("No cursor manager available for zoom operation")
                return
            
            # Check if cursors are available
            if not cursor_manager.can_zoom_to_cursors():
                logger.warning("Cannot zoom to cursors - dual cursors not available or not positioned")
                return
            
            # Perform the zoom
            success = cursor_manager.zoom_to_cursors()
            if success:
                logger.info("Successfully zoomed to cursor range from settings panel")
            else:
                logger.warning("Failed to zoom to cursors - check cursor positions")
                
        except Exception as e:
            logger.error(f"Error during zoom to cursors operation: {e}")
    
    def update_zoom_button_state(self):
        """Update the zoom to cursors button enabled state."""
        try:
            if hasattr(self, 'zoom_to_cursors_button'):
                cursor_manager = getattr(self.parent, 'cursor_manager', None)
                can_zoom = False
                if cursor_manager and hasattr(cursor_manager, 'can_zoom_to_cursors'):
                    try:
                        can_zoom = cursor_manager.can_zoom_to_cursors()
                        can_zoom = bool(can_zoom) if can_zoom is not None else False
                    except Exception as e:
                        logger.debug(f"Error checking cursor zoom state: {e}")
                        can_zoom = False
                self.zoom_to_cursors_button.setEnabled(can_zoom)
        except Exception as e:
            logger.debug(f"Error updating zoom button state: {e}")