# type: ignore
"""
Legend Manager for Time Graph Widget

Manages the legend interface including:
- ATI Vision-style compact legend
- Real-time signal value display
- Signal visibility controls
- Color management
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QScrollArea, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal as Signal, QTimer, QObject
from PyQt5.QtGui import QColor, QFont, QPalette

if TYPE_CHECKING:
    from .time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)

class LegendManager(QObject):
    """Manages the legend interface for the Time Graph Widget."""
    
    # Signals
    signal_visibility_changed = Signal(str, bool)  # signal_name, visible
    signal_selected = Signal(str)  # signal_name
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.legend_panel = None
        self.legend_items = {}  # Dict of signal_name -> legend_widget
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_values)
        self.update_timer.start(100)  # Update every 100ms for smooth real-time display
        
        self._setup_legend_panel()
    
    def _setup_legend_panel(self):
        """Create ATI Vision-style compact legend panel with signal values."""
        self.legend_panel = QWidget()
        self.legend_panel.setMaximumWidth(280)
        self.legend_panel.setMinimumWidth(250)
        
        # Main layout
        layout = QVBoxLayout(self.legend_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Title
        title = QLabel("📊 Signal Legend")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2d4a66;
                padding: 6px;
                border-radius: 4px;
                border: 1px solid #4a90e2;
            }
        """)
        layout.addWidget(title)
        
        # Scrollable area for legend items
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1a2332;
                border: 1px solid #2d4a66;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #2d4a66;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a90e2;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6bb6ff;
            }
        """)
        
        # Container for legend items
        self.legend_container = QWidget()
        self.legend_layout = QVBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend_layout.setSpacing(2)
        self.legend_layout.addStretch()  # Push items to top
        
        scroll_area.setWidget(self.legend_container)
        layout.addWidget(scroll_area)
        
        # Control buttons
        self._create_control_buttons(layout)
    
    def _create_control_buttons(self, layout: QVBoxLayout):
        """Create control buttons for legend management."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        
        # Show/Hide All button
        show_all_btn = QPushButton("Show All")
        show_all_btn.setMaximumHeight(24)
        show_all_btn.clicked.connect(self._show_all_signals)
        show_all_btn.setStyleSheet(self._get_button_style())
        button_layout.addWidget(show_all_btn)
        
        # Hide All button
        hide_all_btn = QPushButton("Hide All")
        hide_all_btn.setMaximumHeight(24)
        hide_all_btn.clicked.connect(self._hide_all_signals)
        hide_all_btn.setStyleSheet(self._get_button_style())
        button_layout.addWidget(hide_all_btn)
        
        layout.addLayout(button_layout)
    
    def _get_button_style(self) -> str:
        """Get consistent button styling."""
        return """
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 9px;
            }
            QPushButton:hover {
                background-color: #4a90e2;
                border-color: #6bb6ff;
            }
            QPushButton:pressed {
                background-color: #3a7bc8;
            }
        """
    
    def add_legend_item(self, name: str, color: str, current_value: float = 0.0):
        """Add a signal to the ATI Vision style legend panel."""
        if name in self.legend_items:
            # Update existing item
            self.update_legend_value(name, current_value)
            return
        
        # Create legend item widget
        item_widget = QFrame()
        item_widget.setFrameStyle(QFrame.Box)
        item_widget.setMaximumHeight(32)
        item_widget.setStyleSheet(f"""
            QFrame {{
                background-color: #2d4a66;
                border: 1px solid #4a90e2;
                border-radius: 4px;
                margin: 1px;
            }}
            QFrame:hover {{
                background-color: #3a5f7a;
                border-color: #6bb6ff;
            }}
        """)
        
        # Item layout
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(6, 4, 6, 4)
        item_layout.setSpacing(6)
        
        # Visibility checkbox
        visibility_cb = QCheckBox()
        visibility_cb.setChecked(True)
        visibility_cb.setMaximumWidth(20)
        visibility_cb.toggled.connect(lambda checked: self.signal_visibility_changed.emit(name, checked))
        visibility_cb.setStyleSheet("""
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                border: 1px solid #6bb6ff;
                border-radius: 2px;
            }
        """)
        item_layout.addWidget(visibility_cb)
        
        # Color indicator
        color_label = QLabel()
        color_label.setMaximumSize(12, 12)
        color_label.setMinimumSize(12, 12)
        color_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                border: 1px solid #ffffff;
                border-radius: 6px;
            }}
        """)
        item_layout.addWidget(color_label)
        
        # Signal name
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 8, QFont.Bold))
        name_label.setStyleSheet("color: #ffffff;")
        name_label.setMinimumWidth(80)
        item_layout.addWidget(name_label)
        
        # Current value
        value_label = QLabel(f"{current_value:.3f}")
        value_label.setFont(QFont("Courier", 8))  # Monospace for aligned numbers
        value_label.setStyleSheet("color: #00ff00; background-color: #1a1a1a; padding: 2px; border-radius: 2px;")
        value_label.setMinimumWidth(60)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        item_layout.addWidget(value_label)
        
        # Store references
        self.legend_items[name] = {
            'widget': item_widget,
            'visibility_cb': visibility_cb,
            'color_label': color_label,
            'name_label': name_label,
            'value_label': value_label,
            'color': color,
            'visible': True
        }
        
        # Add to layout (insert before stretch)
        self.legend_layout.insertWidget(self.legend_layout.count() - 1, item_widget)
        
        logger.debug(f"Added legend item for signal: {name}")
    
    def has_item(self, signal_name: str) -> bool:
        """Check if legend item exists for the given signal."""
        return signal_name in self.legend_items
    
    def remove_legend_item(self, name: str):
        """Remove a signal from the legend."""
        if name in self.legend_items:
            item = self.legend_items[name]
            item['widget'].setParent(None)
            del self.legend_items[name]
            logger.info(f"Removed legend item for signal: {name}")
    
    def update_legend_value(self, name: str, value: float):
        """Update the current value display for a signal."""
        if name in self.legend_items:
            item = self.legend_items[name]
            try:
                # Ensure value is a Python float, not PyArrow scalar
                float_value = float(value)
                new_text = f"{float_value:.3f}"
                
                # OPTIMIZATION: Only update text if it changed to avoid repaint/layout
                if item['value_label'].text() != new_text:
                    item['value_label'].setText(new_text)
            except (ValueError, TypeError):
                if item['value_label'].text() != "N/A":
                    item['value_label'].setText("N/A")
    
    def set_signal_visibility(self, name: str, visible: bool):
        """Set the visibility state of a signal."""
        if name in self.legend_items:
            item = self.legend_items[name]
            item['visibility_cb'].setChecked(visible)
            item['visible'] = visible
    
    def get_signal_visibility(self, name: str) -> bool:
        """Get the visibility state of a signal."""
        if name in self.legend_items:
            return self.legend_items[name]['visible']
        return True
    
    def _show_all_signals(self):
        """Show all signals."""
        for name, item in self.legend_items.items():
            item['visibility_cb'].setChecked(True)
    
    def _hide_all_signals(self):
        """Hide all signals."""
        for name, item in self.legend_items.items():
            item['visibility_cb'].setChecked(False)
    
    def _update_values(self):
        """Update legend values based on current cursor position or latest data."""
<<<<<<< HEAD
        if not self.parent:
            return

        try:
            # Get cursor position
            cursor_pos = None
            if hasattr(self.parent, 'cursor_manager') and self.parent.cursor_manager:
                cursor_pos = self.parent.cursor_manager.get_cursor_position()

            # Get signal data if available
            if hasattr(self.parent, 'signal_processor'):
                signal_data = self.parent.signal_processor.get_all_signals()
                if signal_data:
                    self.update_values_from_data(signal_data, cursor_pos)
                    
        except Exception as e:
            # Don't log every 100ms to avoid spamming, but catch to prevent crash
            pass
=======
        # This will be called by the parent widget with actual data
        pass
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    
    def update_values_from_data(self, signal_data: Dict[str, Any], cursor_position: Optional[float] = None):
        """Update legend values from signal data."""
        for name, data in signal_data.items():
            if name in self.legend_items and 'y_data' in data:
                y_data = data['y_data']
                
                if cursor_position is not None and 'x_data' in data:
                    # Find value at cursor position
                    x_data = data['x_data']
                    if len(x_data) > 0 and len(y_data) > 0:
                        # Find closest index to cursor position
                        idx = np.argmin(np.abs(x_data - cursor_position))
                        value = y_data[idx]
                        self.update_legend_value(name, value)
                elif len(y_data) > 0:
                    # Use latest value
                    value = y_data[-1]
                    self.update_legend_value(name, value)
    
    def get_legend_panel(self) -> QWidget:
        """Get the legend panel widget."""
        return self.legend_panel
    
    def clear_all_items(self):
        """Clear all legend items."""
        for name in list(self.legend_items.keys()):
            self.remove_legend_item(name)
    
    def get_signal_colors(self) -> Dict[str, str]:
        """Get all signal colors."""
        return {name: item['color'] for name, item in self.legend_items.items()}
    
    def set_signal_color(self, name: str, color: str):
        """Update signal color."""
        if name in self.legend_items:
            item = self.legend_items[name]
            item['color'] = color
            item['color_label'].setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border: 1px solid #ffffff;
                    border-radius: 6px;
                }}
            """)
    
    def get_visible_signals(self) -> List[str]:
        """Get list of visible signal names."""
        return [name for name, item in self.legend_items.items() if item['visible']]
    
    def stop_updates(self):
        """Stop the update timer."""
        if self.update_timer.isActive():
            self.update_timer.stop()
    
    def start_updates(self):
        """Start the update timer."""
        if not self.update_timer.isActive():
            self.update_timer.start(100)
