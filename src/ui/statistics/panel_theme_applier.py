# type: ignore
"""
Theme Applier Helper for Statistics Panel

Handles all theme-related styling operations:
- Main panel styling
- Table styling
- Header table styling  
- Cursor info panel styling
- Light/dark theme support
"""

import logging
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QTableWidget

logger = logging.getLogger(__name__)


class PanelThemeApplier:
    """Helper class for applying theme styling to statistics panel components."""
    
    def __init__(self, panel):
        """
        Initialize theme applier.
        
        Args:
            panel: Parent StatisticsPanel instance
        """
        self.panel = panel
        logger.debug("PanelThemeApplier initialized")
    
    def apply_initial_theme_styling(self):
        """Apply initial theme styling when panel is created."""
        theme_colors = self._get_theme_colors()
        self.apply_theme_styling(theme_colors)
    
    def update_theme(self, theme_colors=None):
        """
        Update the panel styling when theme changes.
        
        Args:
            theme_colors: Optional theme color dictionary
        """
        # Apply theme styling to main panel
        self.apply_theme_styling(theme_colors)
        
        # Update all graph tables with new theme
        for table in self.panel.graph_tables.values():
            self.apply_table_styling_to_single_table(table)
        
        # Update header table styling with new theme
        if hasattr(self.panel, 'header_table') and self.panel.header_table:
            self.update_header_table_theme(theme_colors)
        
        # Update cursor info panel if it exists
        if hasattr(self.panel, 'cursor_info_panel') and self.panel.cursor_info_panel:
            self.update_cursor_info_panel_theme(theme_colors)
        
        # Update all graph sections styling
        for graph_section in self.panel.graph_sections.values():
            self.panel._apply_graph_section_styling(graph_section)
        
        logger.debug("Theme updated for statistics panel")
    
    def apply_theme_styling(self, theme_colors=None):
        """
        Apply theme-based styling to the statistics panel.
        
        Args:
            theme_colors: Optional theme color dictionary
        """
        theme_colors = self._get_theme_colors(theme_colors)
        logger.debug(f"Statistics panel applying theme with text_primary: {theme_colors.get('text_primary')}")
        
        # Determine if this is a light theme
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        
        # Get theme-specific colors
        colors = self._get_theme_specific_colors(theme_colors, is_light_theme)
        
        # Update title label styling
        self._update_title_label_styling(colors)
        
        # Apply main panel stylesheet
        self._apply_main_panel_stylesheet(theme_colors, colors)
        
        # Apply table styling to all graph tables
        self._apply_all_graph_tables_styling(colors)
    
    def update_header_table_theme(self, theme_colors=None):
        """
        Update header table theme styling.
        
        Args:
            theme_colors: Optional theme color dictionary
        """
        theme_colors = self._get_theme_colors(theme_colors)
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        colors = self._get_basic_theme_colors(theme_colors, is_light_theme)
        
        # Update header table style
        header_table_style = f"""
            QTableWidget {{
                background-color: rgba({colors['bg_color_base']}, {colors['bg_opacity']});
                border: 1px solid rgba({colors['border_color_base']}, {colors['border_opacity']});
                border-radius: 8px;
                gridline-color: rgba({colors['border_color_base']}, 0.3);
                color: {colors['text_color']};
            }}
            QHeaderView::section {{
                background-color: rgba({colors['bg_color_base']}, 0.15);
                border: 1px solid rgba({colors['border_color_base']}, 0.3);
                padding: 5px 8px;
                font-weight: bold;
                font-size: 12px;
                color: {colors['text_color']};
            }}
            QHeaderView::section:first {{
                font-size: 13px;
                color: {colors['text_color']};
            }}
            QTableWidget::item {{
                color: {colors['text_color']};
                background-color: transparent;
            }}
        """
        self.panel.header_table.setStyleSheet(header_table_style)
    
    def update_cursor_info_panel_theme(self, theme_colors=None):
        """
        Update cursor info panel theme styling.
        
        Args:
            theme_colors: Optional theme color dictionary
        """
        theme_colors = self._get_theme_colors(theme_colors)
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        text_color = '#212121' if is_light_theme else '#ffffff'
        
        # Update all cursor labels
        for label in [self.panel.cursor1_time_label, self.panel.cursor2_time_label, 
                     self.panel.delta_time_label, self.panel.frequency_label]:
            if hasattr(self.panel, label.objectName()) or label:
                bg_color = self._extract_background_color(label)
                
                label.setStyleSheet(f"""
                    QLabel {{
                        color: {text_color};
                        font-size: 13px;
                        font-weight: 600;
                        padding: 6px 10px;
                        border-radius: 4px;
                        background-color: {bg_color};
                        min-height: 20px;
                    }}
                """)
    
    def apply_table_styling_to_single_table(self, table: QTableWidget):
        """
        Apply current theme styling to a single table.
        
        Args:
            table: QTableWidget to style
        """
        # Set row height for compact display
        table.verticalHeader().setDefaultSectionSize(24)
        
        theme_colors = self._get_theme_colors()
        is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
        colors = self._get_basic_theme_colors(theme_colors, is_light_theme)
        
        # Apply table styling
        table_style = f"""
                QTableWidget {{
                    background-color: rgba({colors['bg_color_base']}, {colors['bg_opacity']});
                    border: 1px solid rgba({colors['border_color_base']}, {colors['border_opacity']});
                    border-radius: 8px;
                    color: {colors['text_color']};
                    gridline-color: rgba({colors['border_color_base']}, 0.1);
                    selection-background-color: rgba(74, 144, 226, 0.3);
                    alternate-background-color: rgba({colors['bg_color_base']}, 0.08);
                }}
                
                QTableWidget::item {{
                    padding: 1px 8px;
                    border: none;
                    color: {colors['text_color']};
                    background-color: transparent;
                }}
                
                QTableWidget::item:alternate {{
                    background-color: rgba({colors['bg_color_base']}, 0.08);
                    color: {colors['text_color']};
                }}
                
                QTableWidget::item:selected {{
                    background-color: rgba(74, 144, 226, 0.3);
                    color: {colors['text_color']};
                }}
                
                QTableWidget::item:hover {{
                    background-color: rgba({colors['border_color_base']}, 0.15);
                    color: {colors['text_color']};
                }}
                
                QHeaderView::section {{
                    background-color: rgba({colors['bg_color_base']}, 0.2);
                    color: {colors['text_color']};
                    padding: 4px 8px;
                    border: 1px solid rgba({colors['border_color_base']}, 0.2);
                    font-weight: bold;
                }}
                
                QHeaderView::section:hover {{
                    background-color: rgba(74, 144, 226, 0.2);
                    color: {colors['text_color']};
                }}
            """
        table.setStyleSheet(table_style)
    
    # ==================== PRIVATE HELPER METHODS ====================
    
    def _get_theme_colors(self, theme_colors=None) -> Dict[str, Any]:
        """Get theme colors from parent or use fallback."""
        if theme_colors is None and hasattr(self.panel.parent(), 'theme_manager'):
            theme_colors = self.panel.parent().theme_manager.get_theme_colors()
        
        if theme_colors is None:
            # Fallback colors for space theme
            theme_colors = {
                'text_primary': '#e6f3ff',
                'text_secondary': '#ffffff',
                'surface': '#2d4a66',
                'surface_variant': '#3a5f7a',
                'border': '#4a90e2',
                'primary': '#4a90e2'
            }
        
        return theme_colors
    
    def _get_basic_theme_colors(self, theme_colors: Dict, is_light_theme: bool) -> Dict[str, str]:
        """Get basic color settings for theme."""
        if is_light_theme:
            return {
                'text_color': '#212121',
                'border_color_base': '0, 0, 0',
                'bg_color_base': '0, 0, 0',
                'border_opacity': '0.3',
                'bg_opacity': '0.1'
            }
        else:
            return {
                'text_color': theme_colors.get('text_primary', '#ffffff'),
                'border_color_base': '255, 255, 255',
                'bg_color_base': '255, 255, 255',
                'border_opacity': '0.2',
                'bg_opacity': '0.05'
            }
    
    def _get_theme_specific_colors(self, theme_colors: Dict, is_light_theme: bool) -> Dict[str, str]:
        """Get comprehensive theme-specific colors."""
        if is_light_theme:
            return {
                'text_color': '#212121',
                'secondary_text_color': '#757575',
                'border_color_base': '0, 0, 0',
                'bg_color_base': '0, 0, 0',
                'border_opacity': '0.3',
                'bg_opacity': '0.1',
                'scrollbar_bg': 'rgba(0, 0, 0, 0.15)',
                'scrollbar_handle': 'rgba(0, 0, 0, 0.4)',
                'scrollbar_handle_hover': 'rgba(0, 0, 0, 0.6)',
                'title_bg_opacity': '0.1',
                'title_bg_color': 'rgba(255, 255, 255, 0.9)'
            }
        else:
            return {
                'text_color': theme_colors.get('text_primary', '#ffffff'),
                'secondary_text_color': theme_colors.get('text_secondary', '#e0e0e0'),
                'border_color_base': '255, 255, 255',
                'bg_color_base': '255, 255, 255',
                'border_opacity': '0.2',
                'bg_opacity': '0.05',
                'scrollbar_bg': 'rgba(255, 255, 255, 0.1)',
                'scrollbar_handle': 'rgba(255, 255, 255, 0.3)',
                'scrollbar_handle_hover': 'rgba(255, 255, 255, 0.5)',
                'title_bg_opacity': '0.1',
                'title_bg_color': 'rgba(0, 0, 0, 0.3)'
            }
    
    def _update_title_label_styling(self, colors: Dict):
        """Update title label with theme colors."""
        if hasattr(self.panel, 'title_label'):
            self.panel.title_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['text_color']};
                    font-size: 18px;
                    font-weight: bold;
                    padding: 2px;
                    background-color: rgba({colors['bg_color_base']}, {colors['title_bg_opacity']});
                    border-radius: 8px;
                    margin-bottom: 2px;
                }}
            """)
    
    def _apply_main_panel_stylesheet(self, theme_colors: Dict, colors: Dict):
        """Apply main stylesheet to panel."""
        self.panel.setStyleSheet(f"""
            StatisticsPanel {{
                background-color: transparent;
                border: none;
            }}
            
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: 1px solid rgba({colors['border_color_base']}, {colors['border_opacity']});
                border-radius: 8px;
                margin-top: 0px;
                margin-bottom: 1px;
                padding: 1px;
                background-color: rgba({colors['bg_color_base']}, {colors['bg_opacity']});
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
                background-color: {colors['title_bg_color']};
                border-radius: 6px;
                font-size: 15px;
                font-weight: bold;
                color: {theme_colors.get('primary', '#4a90e2')};
            }}
            
            QLabel {{
                color: {colors['text_color']};
                font-size: 16px;
                font-weight: 500;
                padding: 3px 0px;
            }}

            QFormLayout QLabel {{
                font-weight: bold;
                color: {colors['secondary_text_color']};
                font-size: 14px;
                min-width: 80px;
            }}
            
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QScrollArea QScrollBar:vertical {{
                background-color: {colors['scrollbar_bg']};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollArea QScrollBar::handle:vertical {{
                background-color: {colors['scrollbar_handle']};
                border-radius: 6px;
                min-height: 20px;
            }}
            
            QScrollArea QScrollBar::handle:vertical:hover {{
                background-color: {colors['scrollbar_handle_hover']};
            }}
            
            /* Graph section scroll bars */
            QGroupBox QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QGroupBox QScrollArea QScrollBar:vertical {{
                background-color: {colors['scrollbar_bg']};
                width: 8px;
                border-radius: 4px;
            }}
            
            QGroupBox QScrollArea QScrollBar::handle:vertical {{
                background-color: {colors['scrollbar_handle']};
                border-radius: 4px;
                min-height: 15px;
            }}
            
            QGroupBox QScrollArea QScrollBar::handle:vertical:hover {{
                background-color: {colors['scrollbar_handle_hover']};
            }}
        """)
    
    def _apply_all_graph_tables_styling(self, colors: Dict):
        """Apply styling to all graph tables."""
        table_style = f"""
                QTableWidget {{
                    background-color: rgba({colors['bg_color_base']}, {colors['bg_opacity']});
                    border: 1px solid rgba({colors['border_color_base']}, {colors['border_opacity']});
                    border-radius: 8px;
                    color: {colors['text_color']};
                    gridline-color: rgba({colors['border_color_base']}, 0.1);
                    selection-background-color: rgba(74, 144, 226, 0.3);
                    alternate-background-color: rgba({colors['bg_color_base']}, 0.08);
                }}
                
                QTableWidget::item {{
                    padding: 1px 8px;
                    border: none;
                    color: {colors['text_color']};
                    background-color: transparent;
                }}
                
                QTableWidget::item:alternate {{
                    background-color: rgba({colors['bg_color_base']}, 0.08);
                    color: {colors['text_color']};
                }}
                
                QTableWidget::item:selected {{
                    background-color: rgba(74, 144, 226, 0.3);
                    color: {colors['text_color']};
                }}
                
                QTableWidget::item:hover {{
                    background-color: rgba({colors['border_color_base']}, 0.15);
                    color: {colors['text_color']};
                }}
                
                QHeaderView::section {{
                    background-color: rgba({colors['bg_color_base']}, 0.2);
                    color: {colors['text_color']};
                    padding: 4px 8px;
                    border: 1px solid rgba({colors['border_color_base']}, 0.2);
                    font-weight: bold;
                }}
                
                QHeaderView::section:hover {{
                    background-color: rgba(74, 144, 226, 0.2);
                    color: {colors['text_color']};
                }}
            """
        
        # Apply styling to all graph tables (performance-optimized: single loop)
        for table in self.panel.graph_tables.values():
            table.setStyleSheet(table_style)
            table.verticalHeader().setDefaultSectionSize(24)
    
    def _extract_background_color(self, label) -> str:
        """Extract background color from label's current stylesheet."""
        current_style = label.styleSheet()
        
        # Map color patterns to background colors
        color_map = {
            'rgba(74, 144, 226': 'rgba(74, 144, 226, 0.15)',
            'rgba(226, 74, 144': 'rgba(226, 74, 144, 0.15)',
            'rgba(144, 226, 74': 'rgba(144, 226, 74, 0.15)',
            'rgba(226, 144, 74': 'rgba(226, 144, 74, 0.15)'
        }
        
        for pattern, bg_color in color_map.items():
            if pattern in current_style:
                return bg_color
        
        return 'rgba(74, 144, 226, 0.15)'  # Default

