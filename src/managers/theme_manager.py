# type: ignore
"""
Theme Manager for Time Graph Widget

Manages visual themes and styling including:
- Dark/Light theme support
- Professional color schemes
- Consistent styling across components
- Performance-optimized style application
"""

import logging
from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from PyQt5.QtGui import QColor, QPalette

logger = logging.getLogger(__name__)

class ThemeManager(QObject):
    """
    Manages themes and styling for the Time Graph Widget.
    
    Features:
    - Multiple theme presets
    - Dynamic theme switching
    - Consistent color schemes
    - Performance-optimized style caching
    """
    
    # Signals
    theme_changed = Signal(str)  # theme_name
    
    # Theme definitions
    THEMES = {
        
        'light_professional': {
            'name': 'Light Professional',
            'background': '#ffffff',
            'surface': '#f5f5f5',
            'surface_variant': '#e0e0e0',
            'primary': '#2196f3',
            'primary_variant': '#1976d2',
            'secondary': '#90caf9',
            'text_primary': '#212121',
            'text_secondary': '#757575',
            'text_disabled': '#bdbdbd',
            'accent': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'success': '#4caf50',
            'grid': '#e0e0e0',
            'border': '#cccccc',
            'hover': '#e3f2fd',
            'selected': '#bbdefb',
            'plot_background': '#fafafa',
            'axis_color': '#212121',
            'axis_text': '#212121'
        },
        
        'space': {
            'name': 'Space',
            'background': '#1a2332',
            'surface': '#2d4a66',
            'surface_variant': '#3a5f7a',
            'primary': '#4a90e2',
            'primary_variant': '#6bb6ff',
            'secondary': '#5ba0f2',
            'text_primary': '#e6f3ff',
            'text_secondary': '#ffffff',
            'text_disabled': '#7a8a9a',
            'accent': '#4a90e2',
            'warning': '#ffab00',
            'error': '#ff5252',
            'success': '#69f0ae',
            'grid': '#2d4a66',
            'border': '#4a90e2',
            'hover': '#5ba0f2',
            'selected': '#6bb6ff',
            'plot_background': '#1a2332',
            'axis_color': '#e6f3ff',
            'axis_text': '#ffffff'
        }
    }
    
    # Signal color palettes
    SIGNAL_COLORS = {
        'professional': [
            '#ff4444', '#4444ff', '#44ff44', '#ff8800', 
            '#8844ff', '#ff4488', '#44ffff', '#ffff44'
        ],
        'vibrant': [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
            '#9b59b6', '#e91e63', '#1abc9c', '#f1c40f'
        ],
        'pastel': [
            '#ff9999', '#9999ff', '#99ff99', '#ffcc99',
            '#cc99ff', '#ff99cc', '#99ffff', '#ffffcc'
        ]
    }
    
    def __init__(self):
        super().__init__()
        self.current_theme = 'space'  # Changed default theme to space
        self.current_signal_palette = 'professional'
        self.style_cache = {}  # Cache for computed styles
        self.color_overrides = {}  # To store user-defined signal colors {signal_index: color_hex}
        
    def set_theme(self, theme_name: str):
        """Set the current theme."""
        theme_key = theme_name.lower().replace(' ', '_')
        # Map common names to theme keys
        theme_map = {
            'light': 'light_professional',
            'space': 'space'
        }
        
        resolved_theme_key = theme_map.get(theme_key, 'space')

        if resolved_theme_key in self.THEMES:
            self.current_theme = resolved_theme_key
            self.style_cache.clear()  # Clear cache when theme changes
            self.theme_changed.emit(resolved_theme_key)
            logger.info(f"Theme changed to: {resolved_theme_key}")
        else:
            logger.warning(f"Unknown theme: {theme_name}")
    
    def get_current_theme(self) -> str:
        """Get the current theme name."""
        return self.current_theme
    
    def get_theme_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Get color definitions for a theme."""
        theme = theme_name or self.current_theme
        return self.THEMES.get(theme, self.THEMES['space']).copy()
    
    def get_color(self, color_key: str, theme_name: Optional[str] = None) -> str:
        """Get a specific color from the current or specified theme."""
        theme = theme_name or self.current_theme
        colors = self.THEMES.get(theme, self.THEMES['space'])
        return colors.get(color_key, '#ffffff')
    
    def get_signal_colors(self, palette_name: Optional[str] = None) -> List[str]:
        """Get signal color palette."""
        palette = palette_name or self.current_signal_palette
        return self.SIGNAL_COLORS.get(palette, self.SIGNAL_COLORS['professional']).copy()
    
    def get_signal_color(self, index: int, palette_name: Optional[str] = None) -> str:
        """Get a signal color by index, checking for overrides first."""
        if index in self.color_overrides:
            return self.color_overrides[index]
        
        colors = self.get_signal_colors(palette_name)
        return colors[index % len(colors)]
    
    def set_signal_color_override(self, signal_index: int, color_hex: str):
        """Set a custom color for a specific signal index."""
        self.color_overrides[signal_index] = color_hex
        logger.debug(f"Color override set for signal index {signal_index} to {color_hex}")

    def clear_signal_color_overrides(self):
        """Clear all custom signal colors."""
        self.color_overrides.clear()
        logger.debug("All signal color overrides cleared")

    def set_signal_palette(self, palette_name: str):
        """Set the signal color palette."""
        if palette_name in self.SIGNAL_COLORS:
            self.current_signal_palette = palette_name
            logger.info(f"Signal palette changed to: {palette_name}")
        else:
            logger.warning(f"Unknown signal palette: {palette_name}")
    
    def get_widget_stylesheet(self, widget_type: str, theme_name: Optional[str] = None) -> str:
        """
        Get optimized stylesheet for specific widget types.
        
        Args:
            widget_type: Type of widget ('toolbar', 'legend', 'plot', etc.)
            theme_name: Theme to use (current theme if None)
            
        Returns:
            CSS stylesheet string
        """
        theme = theme_name or self.current_theme
        cache_key = f"{theme}_{widget_type}"
        
        # Check cache first
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]
        
        colors = self.get_theme_colors(theme)
        stylesheet = ""
        
        if widget_type == 'toolbar':
            stylesheet = self._get_toolbar_stylesheet(colors)
        elif widget_type == 'legend':
            stylesheet = self._get_legend_stylesheet(colors)
        elif widget_type == 'plot':
            stylesheet = self._get_plot_stylesheet(colors)
        elif widget_type == 'dialog':
            stylesheet = self._get_dialog_stylesheet(colors)
        elif widget_type == 'button':
            stylesheet = self._get_button_stylesheet(colors)
        elif widget_type == 'panel':
            stylesheet = self._get_panel_stylesheet(colors)
        
        # Cache the result
        self.style_cache[cache_key] = stylesheet
        return stylesheet
    
    def _get_toolbar_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate toolbar stylesheet."""
        return f"""
            QToolBar {{
                background-color: {colors['surface']};
                border: none;
                spacing: 2px;
            }}
            QToolButton {{
                background-color: {colors['surface_variant']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
                margin: 1px;
                color: {colors['text_primary']};
                font-size: 10px;
                min-width: 60px;
            }}
            QToolButton:hover {{
                background-color: {colors['hover']};
                border-color: {colors['primary_variant']};
            }}
            QToolButton:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary_variant']};
                font-weight: bold;
            }}
            QSpinBox {{
                background-color: {colors['surface_variant']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 2px 4px;
                color: {colors['text_primary']};
                font-size: 10px;
                min-width: 40px;
                max-width: 60px;
            }}
            QLabel {{
                color: {colors['text_primary']};
                font-size: 10px;
                margin: 0px 4px;
            }}
        """
    
    def _get_legend_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate legend panel stylesheet."""
        return f"""
            QWidget {{
                background-color: {colors['background']};
                color: {colors['text_primary']};
            }}
            QFrame {{
                background-color: {colors['secondary']};
                border: 1px solid {colors['primary']};
                border-radius: 4px;
                margin: 1px;
            }}
            QFrame:hover {{
                background-color: {colors['hover']};
                border-color: {colors['primary_variant']};
            }}
            QScrollArea {{
                background-color: {colors['surface']};
                border: 1px solid {colors['secondary']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background-color: {colors['secondary']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors['primary']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {colors['primary_variant']};
            }}
        """
    
    def _get_plot_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate plot widget stylesheet."""
        return f"""
            QWidget {{
                background-color: {colors['plot_background']};
                border: 1px solid {colors['border']};
            }}
        """
    
    def _get_dialog_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate dialog stylesheet."""
        return f"""
            QDialog {{
                background-color: {colors['background']};
                color: {colors['text_primary']};
            }}
        """
    
    def _get_button_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate button stylesheet."""
        return f"""
            QPushButton {{
                background-color: {colors['surface_variant']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
                color: {colors['text_primary']};
                font-size: 9px;
            }}
            QPushButton:hover {{
                background-color: {colors['hover']};
                border-color: {colors['primary_variant']};
            }}
            QPushButton:pressed {{
                background-color: {colors['primary']};
            }}
        """
    
    def _get_panel_stylesheet(self, colors: Dict[str, str]) -> str:
        """Generate panel stylesheet."""
        return f"""
            QWidget {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
            }}
            QLabel {{
                color: {colors['text_primary']};
            }}
            QGroupBox {{
                background-color: {colors['surface_variant']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                color: {colors['text_primary']};
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }}
            QComboBox {{
                background-color: {colors['surface_variant']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 4px;
                color: {colors['text_primary']};
            }}
            QComboBox:hover {{
                border-color: {colors['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                selection-background-color: {colors['primary']};
            }}
            QTextEdit {{
                background-color: {colors['background']};
                color: {colors['text_secondary']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 4px;
            }}
        """
    
    def apply_theme_to_widget(self, widget: QWidget, widget_type: str, theme_name: Optional[str] = None):
        """Apply theme styling to a widget."""
        stylesheet = self.get_widget_stylesheet(widget_type, theme_name)
        widget.setStyleSheet(stylesheet)
    
    def get_plot_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Get plot-specific colors."""
        colors = self.get_theme_colors(theme_name)
        return {
            'background': colors['plot_background'],
            'axis': colors['axis_color'],
            'axis_text': colors['axis_text'],
            'grid': colors['grid'],
            'border': colors['border']
        }
    
    def clear_cache(self):
        """Clear the style cache."""
        self.style_cache.clear()
        logger.debug("Theme style cache cleared")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self.THEMES.keys())
    
    def get_available_signal_palettes(self) -> List[str]:
        """Get list of available signal color palettes."""
        return list(self.SIGNAL_COLORS.keys())
    
    def create_custom_theme(self, name: str, colors: Dict[str, str]):
        """Create a custom theme."""
        # Validate required color keys
        required_keys = ['background', 'surface', 'primary', 'text_primary']
        if not all(key in colors for key in required_keys):
            logger.error(f"Custom theme '{name}' missing required color keys")
            return False
        
        self.THEMES[name] = colors.copy()
        logger.info(f"Created custom theme: {name}")
        return True
    
    def export_theme(self, theme_name: str) -> Optional[Dict[str, str]]:
        """Export theme colors for external use."""
        return self.THEMES.get(theme_name, {}).copy()
    
    def get_contrast_color(self, background_color: str) -> str:
        """Get contrasting text color for a background."""
        # Simple contrast calculation
        color = QColor(background_color)
        luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
        return '#000000' if luminance > 0.5 else '#ffffff'
