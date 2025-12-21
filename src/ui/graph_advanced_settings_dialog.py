"""
Advanced Graph Settings Dialog for Time Graph Widget
Modern, professional interface with comprehensive configuration options
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QDialogButtonBox,
    QListWidgetItem, QLabel, QFrame, QLineEdit, QWidget, QSplitter,
    QPushButton, QScrollArea, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QSlider,
    QColorDialog, QSizePolicy, QStackedWidget, QButtonGroup, QMessageBox
)
from src.ui.parameter_filters_panel import ParameterFiltersPanel
# NOTE: ParametersPanel removed - now in separate Parameters panel (toolbar)
from src.ui.static_limits_panel import StaticLimitsPanel
from src.ui.deviation_panel import DeviationPanel
from src.ui.basic_deviation_panel import BasicDeviationPanel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPainter, QPixmap

logger = logging.getLogger(__name__)

class ModernSidebarButton(QPushButton):
    """Modern sidebar navigation button with hover effects and selection state."""
    
    def __init__(self, text: str, icon: str = "", parent=None):
        super().__init__(parent)
        self.setText(f"{icon} {text}" if icon else text)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(38)
        self.setMaximumHeight(38)
        
        # Modern space theme styling
        self.setStyleSheet("""
            ModernSidebarButton {
                text-align: left;
                padding: 10px 16px;
                border: 1px solid rgba(74, 144, 226, 0.2);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.1), stop:1 transparent);
                color: #e6f3ff;
                font-size: 13px;
                font-weight: 500;
            }
            ModernSidebarButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.1));
                color: #ffffff;
                border-color: rgba(74, 144, 226, 0.5);
            }
            ModernSidebarButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 #5ba0f2);
                color: #ffffff;
                font-weight: 600;
                border-color: #4a90e2;
            }
            ModernSidebarButton:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ba0f2, stop:1 #6bb0ff);
            }
        """)


class GraphAdvancedSettingsDialog(QDialog):
    """
    Advanced Graph Settings Dialog - Modern, Professional Interface
    
    Provides comprehensive configuration options for graph display and filtering.
    """
    
    # Signals for filter application
    range_filter_applied = pyqtSignal(dict)  # Emits filter conditions
    basic_deviation_applied = pyqtSignal(int, dict)  # Emits graph_index, deviation_settings
    limits_applied = pyqtSignal(int, dict) # Emits graph_index, limits_settings
    
    def __init__(self, graph_index: int, all_signals: List[str], visible_signals: List[str] = None, 
                 saved_filter_data: dict = None, saved_limits_data: dict = None, 
                 saved_basic_deviation_data: dict = None, parent=None):
        super().__init__(parent)
        
        self.graph_index = graph_index
        self.all_signals = all_signals if all_signals else []
        self.visible_signals = visible_signals if visible_signals else []
        self.saved_filter_data = saved_filter_data  # Store saved filter data
        self.saved_limits_data = saved_limits_data  # Store saved limits data
        self.saved_basic_deviation_data = saved_basic_deviation_data  # Store saved basic deviation data
        
        # Debug logging
        logger.debug(f"GraphAdvancedSettingsDialog constructor:")
        logger.debug(f"  - graph_index: {graph_index}")
        logger.debug(f"  - all_signals count: {len(all_signals) if all_signals else 0}")
        logger.debug(f"  - visible_signals count: {len(visible_signals) if visible_signals else 0}")
        logger.debug(f"  - saved_filter_data: {saved_filter_data is not None}")
        
        self._setup_dialog()
        self._setup_ui()
        self._setup_connections()
        
        # PERFORMANCE: Saved data yükleme deferred creation'dan sonra yapılacak
        # (_create_deferred_panels içinde çağrılıyor)
        
        logger.debug(f"GraphAdvancedSettingsDialog initialized for graph {graph_index}")
        
    def _setup_dialog(self):
        """Setup dialog properties."""
        self.setWindowTitle(f"Graph {self.graph_index + 1} - Advanced Settings")
        self.setMinimumSize(900, 600)
        self.resize(1200, 650)  # Sağ panel için genişlik artırıldı
        
        # Window flags - taskbar'da görünmesi ve minimize/maximize butonları için
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | 
                          Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint |
                          Qt.WindowTitleHint | Qt.WindowSystemMenuHint)
        
        # Taskbar'da görünmesi için
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)
        self.setWindowModality(Qt.NonModal)  # Non-modal dialog
        
        # Modern space theme
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:0.5 #2d4a66, stop:1 #1a2332);
                color: #e6f3ff;
            }
        """)
        
    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left sidebar
        self._create_sidebar(splitter)
        
        # Middle content area
        self._create_main_content(splitter)
        
        # Right summary panel
        self._create_summary_panel(splitter)
        
        # Set splitter proportions: sidebar(220) + content(650) + summary(330)
        splitter.setSizes([220, 650, 330])
        splitter.setCollapsible(0, False)  # Sidebar
        splitter.setCollapsible(1, False)  # Main content
        splitter.setCollapsible(2, True)   # Summary panel (collapsible)
        
        main_layout.addWidget(splitter)
        
        # Bottom buttons
        self._create_bottom_buttons(main_layout)
        
    def _create_sidebar(self, parent):
        """Create the left sidebar with navigation buttons."""
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QWidget {
                background: #2d4a66;
                border-right: 2px solid rgba(74, 144, 226, 0.5);
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(10)
        
        # Title
        title = QLabel(f"Graph {self.graph_index + 1}")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #e6f3ff;
                padding: 8px 0;
                border-bottom: 2px solid rgba(74, 144, 226, 0.5);
                margin-bottom: 12px;
            }
        """)
        layout.addWidget(title)
        
        # Navigation buttons
        self.nav_buttons = QButtonGroup(self)
        
        # NOTE: Parameters button removed - now in separate Parameters panel
        self.filters_btn = ModernSidebarButton("Range Filters", "🔍")
        self.limits_btn = ModernSidebarButton("Static Limits", "⚠️")
        self.basic_deviation_btn = ModernSidebarButton("Basic Deviation", "📊")
        self.advanced_deviation_btn = ModernSidebarButton("Advanced Deviation", "📈")
        self.advanced_filters_btn = ModernSidebarButton("Advanced Filters", "🔧")
        
        buttons = [self.filters_btn, self.limits_btn, self.basic_deviation_btn, self.advanced_deviation_btn, self.advanced_filters_btn]
        
        for i, btn in enumerate(buttons):
            self.nav_buttons.addButton(btn, i)
            layout.addWidget(btn)
            
        # Set default selection to Range Filters (first button now)
        self.filters_btn.setChecked(True)
        
        layout.addStretch()
        parent.addWidget(sidebar)
        
    def _create_main_content(self, parent):
        """Create the main content area with stacked panels."""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Stacked widget for different panels
        self.stacked_widget = QStackedWidget()
        
        # NOTE: ParametersPanel removed - now in separate Parameters panel (toolbar)
        # PERFORMANCE: Deferred creation - panelleri sakla ama widget'ları QTimer ile oluştur
        
        # Diğer paneller için placeholder ekle
        self._placeholder_filters = QWidget()
        self._placeholder_limits = QWidget()
        self._placeholder_deviation = QWidget()
        self._placeholder_adv_deviation = self._create_placeholder_panel("Advanced Deviation", "📈")
        self._placeholder_adv_filters = self._create_placeholder_panel("Advanced Filters", "🔧")
        
        self.stacked_widget.addWidget(self._placeholder_filters)
        self.stacked_widget.addWidget(self._placeholder_limits)
        self.stacked_widget.addWidget(self._placeholder_deviation)
        self.stacked_widget.addWidget(self._placeholder_adv_deviation)
        self.stacked_widget.addWidget(self._placeholder_adv_filters)
        
        # Gerçek paneller None olarak başlar
        self.parameter_filters_panel = None
        self.static_limits_panel = None
        self.basic_deviation_panel = None
        self.advanced_deviation_panel = None
        self.advanced_filters_panel = None
        
        # PERFORMANCE: Dialog açıldıktan sonra diğer panelleri oluştur
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._create_deferred_panels)
        
        content_layout.addWidget(self.stacked_widget)
        
        # Set default selection to first button (Range Filters)
        self.filters_btn.setChecked(True)
        self.stacked_widget.setCurrentIndex(0)
        
        parent.addWidget(content_widget)
        
    def _create_placeholder_panel(self, title: str, icon: str) -> QWidget:
        """Create a placeholder panel with 'content will be created soon' message."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title with icon
        title_label = QLabel(f"{icon} {title}")
        title_label.setStyleSheet("""
            QLabel {
                color: #e6f3ff;
                font-size: 24px;
                font-weight: 700;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(74, 144, 226, 0.3), stop:1 rgba(74, 144, 226, 0.1));
                border-radius: 12px;
                border: 2px solid rgba(74, 144, 226, 0.3);
                text-align: center;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Placeholder message
        message_label = QLabel("Content will be created soon")
        message_label.setStyleSheet("""
            QLabel {
                color: rgba(230, 243, 255, 0.7);
                font-size: 16px;
                font-style: italic;
                padding: 30px;
                background: rgba(74, 144, 226, 0.05);
                border-radius: 8px;
                border: 1px solid rgba(74, 144, 226, 0.2);
                text-align: center;
            }
        """)
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)
        
        layout.addStretch()
        return panel
        
    def _create_summary_panel(self, parent):
        """Create the right summary panel."""
        summary_widget = QWidget()
        summary_widget.setFixedWidth(330)
        summary_widget.setStyleSheet("""
            QWidget {
                background: #2d4a66;
                border-left: 2px solid rgba(74, 144, 226, 0.5);
            }
        """)
        
        layout = QVBoxLayout(summary_widget)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📋 Current Settings")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #e6f3ff;
                padding: 8px 0;
                border-bottom: 2px solid rgba(74, 144, 226, 0.5);
                margin-bottom: 12px;
            }
        """)
        layout.addWidget(title)
        
        # Scroll area for summary content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; }")
        
        # Summary content widget
        self.summary_content = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_content)
        self.summary_layout.setContentsMargins(5, 5, 5, 5)
        self.summary_layout.setSpacing(10)
        
        # Initial empty state
        self._update_summary_content()
        
        scroll_area.setWidget(self.summary_content)
        layout.addWidget(scroll_area)
        
        parent.addWidget(summary_widget)
        
    def _update_summary_content(self):
        """Update the summary panel content based on current tab."""
        # Clear existing content
        for i in reversed(range(self.summary_layout.count())):
            child = self.summary_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        current_index = self.stacked_widget.currentIndex()
        
        if current_index == 0:  # Parameters
            self._show_parameters_summary()
        elif current_index == 1:  # Range Filters
            self._show_filters_summary()
        elif current_index == 2:  # Static Limits
            self._show_limits_summary()
        elif current_index == 3:  # Basic Deviation
            self._show_basic_deviation_summary()
        elif current_index == 4:  # Advanced Deviation
            self._show_placeholder_summary("Advanced Deviation", "📈")
        elif current_index == 5:  # Advanced Filters
            self._show_placeholder_summary("Advanced Filters", "🔧")
        
        # Add stretch at the end
        self.summary_layout.addStretch()
        
    def _show_parameters_summary(self):
        """Show parameters summary."""
        group = QGroupBox("📊 Selected Parameters")
        group.setStyleSheet(self._get_summary_group_style())
        layout = QVBoxLayout(group)
        
        # NOTE: Parameters panel removed - signals now managed in separate Parameters panel
        selected_signals = []
        
        if selected_signals:
            for signal in selected_signals:
                label = QLabel(f"• {signal}")
                label.setStyleSheet("color: #e6f3ff; font-size: 13px; padding: 2px;")
                label.setWordWrap(True)
                layout.addWidget(label)
        else:
            label = QLabel("No parameters selected")
            label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
            layout.addWidget(label)
        
        self.summary_layout.addWidget(group)
        
    def _show_filters_summary(self):
        """Show range filters summary."""
        group = QGroupBox("🔍 Active Filters")
        group.setStyleSheet(self._get_summary_group_style())
        layout = QVBoxLayout(group)
        
        try:
            filter_conditions = self.parameter_filters_panel.get_range_filter_conditions()
            conditions = filter_conditions.get('conditions', [])
            
            if conditions:
                for i, condition in enumerate(conditions):
                    param_name = condition.get('parameter', 'Unknown')
                    ranges = condition.get('ranges', [])
                    
                    # Parameter name
                    param_label = QLabel(f"📋 {param_name}")
                    param_label.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 13px; margin-top: 5px;")
                    layout.addWidget(param_label)
                    
                    # Ranges
                    for range_info in ranges:
                        range_text = f"  {range_info.get('operator', '')} {range_info.get('value', '')}"
                        range_label = QLabel(range_text)
                        range_label.setStyleSheet("color: #e6f3ff; font-size: 12px; padding-left: 10px;")
                        layout.addWidget(range_label)
                        
                # Mode info
                mode = filter_conditions.get('mode', 'segmented')
                mode_label = QLabel(f"Mode: {mode.title()}")
                mode_label.setStyleSheet("color: #888; font-size: 12px; margin-top: 8px; font-style: italic;")
                layout.addWidget(mode_label)
            else:
                label = QLabel("No filters applied")
                label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
                layout.addWidget(label)
        except:
            label = QLabel("Filter info unavailable")
            label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
            layout.addWidget(label)
        
        self.summary_layout.addWidget(group)
        
    def _show_limits_summary(self):
        """Show static limits summary."""
        group = QGroupBox("⚠️ Static Limits")
        group.setStyleSheet(self._get_summary_group_style())
        layout = QVBoxLayout(group)
        
        try:
            limits_config = self.static_limits_panel.get_limits_configuration()
            
            if limits_config:
                for signal_name, limits in limits_config.items():
                    # Signal name
                    signal_label = QLabel(f"📋 {signal_name}")
                    signal_label.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 13px; margin-top: 5px;")
                    layout.addWidget(signal_label)
                    
                    # Warning limits
                    if limits.get('warning_min') != 0 or limits.get('warning_max') != 0:
                        warning_text = f"  ⚠️ Warning: {limits.get('warning_min', 0):.2f} - {limits.get('warning_max', 0):.2f}"
                        warning_label = QLabel(warning_text)
                        warning_label.setStyleSheet("color: #ffa500; font-size: 12px; padding-left: 10px;")
                        layout.addWidget(warning_label)
            else:
                label = QLabel("No limits configured")
                label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
                layout.addWidget(label)
        except Exception as e:
            label = QLabel("Limits info unavailable")
            label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
            layout.addWidget(label)
        
        self.summary_layout.addWidget(group)
        
    def _show_basic_deviation_summary(self):
        """Show basic deviation summary."""
        group = QGroupBox("📊 Basic Deviation Settings")
        group.setStyleSheet(self._get_summary_group_style())
        layout = QVBoxLayout(group)
        
        try:
            deviation_settings = self.basic_deviation_panel.get_deviation_settings()
            
            # Selected Parameters
            selected_params = deviation_settings.get('selected_parameters', [])
            if selected_params:
                param_count = len(selected_params)
                if param_count <= 3:
                    param_text = f"🎯 Selected Parameters: {', '.join(selected_params)}"
                else:
                    param_text = f"🎯 Selected Parameters: {', '.join(selected_params[:3])} and {param_count - 3} more"
            else:
                param_text = "🎯 Selected Parameters: None (will apply to all visible signals)"
                
            param_label = QLabel(param_text)
            param_label.setStyleSheet("color: #4a90e2; font-size: 13px; font-weight: bold; margin-bottom: 5px;")
            param_label.setWordWrap(True)
            layout.addWidget(param_label)
            
            # Trend Analysis
            trend_settings = deviation_settings.get('trend_analysis', {})
            if trend_settings.get('enabled', False):
                trend_label = QLabel(f"✅ Trend Following: Sensitivity {trend_settings.get('sensitivity', 0)}")
                trend_label.setStyleSheet("color: #4a90e2; font-size: 13px; font-weight: bold;")
                layout.addWidget(trend_label)
            else:
                trend_label = QLabel("❌ Trend Following: Disabled")
                trend_label.setStyleSheet("color: #888; font-size: 13px;")
                layout.addWidget(trend_label)
            
            # Fluctuation Detection
            fluctuation_settings = deviation_settings.get('fluctuation_detection', {})
            if fluctuation_settings.get('enabled', False):
                window = fluctuation_settings.get('sample_window', 0)
                threshold = fluctuation_settings.get('threshold_percent', 0)
                fluct_label = QLabel(f"✅ Fluctuation Detection: {window} samples, {threshold}% threshold")
                fluct_label.setStyleSheet("color: #4a90e2; font-size: 13px; font-weight: bold;")
                layout.addWidget(fluct_label)
                
                if fluctuation_settings.get('highlight_on_graph', False):
                    highlight_label = QLabel("  📍 Highlighting enabled")
                    highlight_label.setStyleSheet("color: #e6f3ff; font-size: 12px; padding-left: 10px;")
                    layout.addWidget(highlight_label)
                    
                if fluctuation_settings.get('red_highlighting', False):
                    red_label = QLabel("  🔴 Red highlighting enabled")
                    red_label.setStyleSheet("color: #ff4444; font-size: 12px; padding-left: 10px;")
                    layout.addWidget(red_label)
            else:
                fluct_label = QLabel("❌ Fluctuation Detection: Disabled")
                fluct_label.setStyleSheet("color: #888; font-size: 13px;")
                layout.addWidget(fluct_label)
            
            # Visual Settings
            visual_settings = deviation_settings.get('visual_settings', {})
            if visual_settings.get('show_bands', False):
                transparency = visual_settings.get('band_transparency', 30)
                visual_label = QLabel(f"📈 Deviation Bands: {transparency}% transparency")
                visual_label.setStyleSheet("color: #e6f3ff; font-size: 12px;")
                layout.addWidget(visual_label)
                
        except Exception as e:
            label = QLabel("Basic deviation info unavailable")
            label.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
            layout.addWidget(label)
        
        self.summary_layout.addWidget(group)
        
    def _show_placeholder_summary(self, title: str, icon: str):
        """Show placeholder summary for panels under development."""
        group = QGroupBox(f"{icon} {title}")
        group.setStyleSheet(self._get_summary_group_style())
        layout = QVBoxLayout(group)
        
        # Placeholder message
        label = QLabel("Content will be created soon")
        label.setStyleSheet("""
            color: rgba(230, 243, 255, 0.7);
            font-style: italic;
            font-size: 13px;
            padding: 10px;
            text-align: center;
        """)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        self.summary_layout.addWidget(group)
        
    def _get_summary_group_style(self):
        """Get styling for summary group boxes."""
        return """
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #e6f3ff;
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.1), stop:1 transparent);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px 0 6px;
                color: #4a90e2;
                font-weight: 700;
            }
        """
        
    def _create_bottom_buttons(self, parent_layout):
        """Create bottom dialog buttons - sadece OK ve Cancel."""
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel  # Apply butonu kaldırıldı
        )
        
        button_box.setStyleSheet("""
            QDialogButtonBox QPushButton {
                padding: 8px 16px;
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.2), stop:1 transparent);
                color: #e6f3ff;
                font-size: 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QDialogButtonBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 144, 226, 0.4), stop:1 rgba(74, 144, 226, 0.1));
                border-color: #4a90e2;
            }
        """)
        
        parent_layout.addWidget(button_box)
        
        # Store references
        self.ok_btn = button_box.button(QDialogButtonBox.Ok)
        self.cancel_btn = button_box.button(QDialogButtonBox.Cancel)
        # Apply butonu yok artık
        
    def _setup_connections(self):
        """Setup signal connections."""
        # Navigation
        self.nav_buttons.buttonClicked.connect(self._on_nav_button_clicked)
        
        # Dialog buttons - Apply butonu kaldırıldı, sadece OK ile uygulama
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # NOTE: Parameters panel signal removed - now in separate Parameters panel
        
        # Connect other panel signals if they exist - REAL-TIME güncelleme için
        if hasattr(self.static_limits_panel, 'limits_changed'):
            self.static_limits_panel.limits_changed.connect(self._on_limits_changed)
        if hasattr(self.basic_deviation_panel, 'deviation_settings_changed'):
            self.basic_deviation_panel.deviation_settings_changed.connect(self._on_basic_deviation_changed)
        if hasattr(self.advanced_deviation_panel, 'deviation_settings_changed'):
            self.advanced_deviation_panel.deviation_settings_changed.connect(self._on_advanced_deviation_changed)
        
        # Range filters için de real-time güncelleme - condition eklenince/silinince
        from PyQt5.QtCore import QTimer
        self._filter_check_timer = QTimer()
        self._filter_check_timer.timeout.connect(self._check_filter_changes)
        self._filter_check_timer.start(500)  # Her 500ms'de kontrol et
        self._last_filter_state = None
        
    def _create_deferred_panels(self):
        """PERFORMANCE: Create heavy panels after dialog is shown (deferred creation)."""
        try:
            logger.debug("Creating deferred panels...")
            
            # Filter panel
            self.parameter_filters_panel = ParameterFiltersPanel(self.graph_index, self.all_signals, self)
            self.stacked_widget.removeWidget(self._placeholder_filters)
            self.stacked_widget.insertWidget(1, self.parameter_filters_panel)
            
            # Static limits panel
            self.static_limits_panel = StaticLimitsPanel(self.all_signals, self)
            self.stacked_widget.removeWidget(self._placeholder_limits)
            self.stacked_widget.insertWidget(2, self.static_limits_panel)
            if hasattr(self.static_limits_panel, 'limits_changed'):
                self.static_limits_panel.limits_changed.connect(self._on_limits_changed)
            
            # Basic deviation panel
            self.basic_deviation_panel = BasicDeviationPanel(self.all_signals, self)
            self.stacked_widget.removeWidget(self._placeholder_deviation)
            self.stacked_widget.insertWidget(3, self.basic_deviation_panel)
            if hasattr(self.basic_deviation_panel, 'deviation_settings_changed'):
                self.basic_deviation_panel.deviation_settings_changed.connect(self._on_basic_deviation_changed)
            
            # Advanced panels (placeholders kalabilir)
            self.advanced_deviation_panel = self._placeholder_adv_deviation
            self.advanced_filters_panel = self._placeholder_adv_filters
            
            # Load saved data
            self._load_saved_filter_data()
            self._load_saved_limits_data()
            self._load_saved_basic_deviation_data()
            
            logger.debug("Deferred panels created successfully")
        except Exception as e:
            logger.error(f"Error creating deferred panels: {e}")
    
    def _on_nav_button_clicked(self, button):
        """Handle navigation button clicks."""
        button_id = self.nav_buttons.id(button)
        
        # Update summary ÖNCE - önceki panelin son durumunu görmek için
        self._update_summary_content()
        
        # Sonra panel değiştir
        self.stacked_widget.setCurrentIndex(button_id)
        
        # Yeni panelin durumunu da güncelle
        self._update_summary_content()
        
        # Log navigation for debugging
        panel_names = ["Parameters", "Range Filters", "Static Limits", "Basic Deviation", "Advanced Deviation", "Advanced Filters"]
        if button_id < len(panel_names):
            logger.debug(f"Switched to {panel_names[button_id]} panel")
        
    def _on_parameters_changed(self, selected_signals):
        """Handle parameters panel changes."""
        # Update summary if parameters panel is active
        if self.stacked_widget.currentIndex() == 0:
            self._update_summary_content()
            
    def _on_limits_changed(self, limits_data):
        """Handle static limits panel changes."""
        # Update summary if limits panel is active
        if self.stacked_widget.currentIndex() == 2:
            self._update_summary_content()
            
        # Emit signal to apply settings immediately - ASIL SORUN BURADAYDI!
        # logger.info(f"[DIALOG] Static limits settings changed, emitting signal: {limits_data}")
        # self.limits_applied.emit(self.graph_index, limits_data)
            
    def _on_basic_deviation_changed(self, deviation_data):
        """Handle basic deviation panel changes."""
        # Update summary if basic deviation panel is active
        if self.stacked_widget.currentIndex() == 3:
            self._update_summary_content()
            
        # Emit signal to apply settings immediately
        # logger.info(f"[DIALOG] Basic deviation settings changed, emitting signal: {deviation_data}")
        # self.basic_deviation_applied.emit(self.graph_index, deviation_data)
            
    def _on_advanced_deviation_changed(self, deviation_data):
        """Handle advanced deviation panel changes."""
        # Update summary if advanced deviation panel is active
        if self.stacked_widget.currentIndex() == 4:
            self._update_summary_content()
    
    def _check_filter_changes(self):
        """Check if filter conditions have changed - real-time update için."""
        try:
            # Sadece Range Filters paneli aktifse kontrol et
            if self.stacked_widget.currentIndex() != 1:
                return
            
            current_filter_state = self.parameter_filters_panel.get_range_filter_conditions()
            
            # İlk kez çalışıyorsa veya değişti ise güncelle
            if self._last_filter_state != current_filter_state:
                self._last_filter_state = current_filter_state
                self._update_summary_content()
        except Exception as e:
            logger.debug(f"Error checking filter changes: {e}")
            
    def _load_saved_filter_data(self):
        """Load saved filter data into the panels."""
        if self.saved_filter_data:
            try:
                # ✅ FIX Problem #1: Panel null check - deferred creation may not be complete
                if not self.parameter_filters_panel:
                    logger.warning(f"[DIALOG STATE] Parameter filters panel not yet created, cannot load saved data")
                    return
                
                # Load range filter data
                if 'conditions' in self.saved_filter_data or 'mode' in self.saved_filter_data:
                    self.parameter_filters_panel.set_range_filter_conditions(self.saved_filter_data)
                    logger.info(f"[DIALOG STATE] Loaded saved filter data for graph {self.graph_index}")
                    logger.debug(f"[DIALOG STATE] Filter data: {self.saved_filter_data}")
                    
                    # Update summary panel to reflect loaded data
                    self._update_summary_content()
                    
            except Exception as e:
                logger.error(f"Error loading saved filter data: {e}", exc_info=True)
    
    def _load_saved_limits_data(self):
        """Load saved limits data into the static limits panel."""
        if self.saved_limits_data:
            try:
                # ✅ FIX Problem #1: Panel null check - deferred creation may not be complete
                if not self.static_limits_panel:
                    logger.warning(f"[DIALOG STATE] Static limits panel not yet created, cannot load saved data")
                    return
                
                self.static_limits_panel.set_limits_configuration(self.saved_limits_data)
                logger.info(f"[DIALOG STATE] Loaded saved limits data for graph {self.graph_index}")
                logger.debug(f"[DIALOG STATE] Limits data: {self.saved_limits_data}")
                
                # Update summary panel to reflect loaded data
                self._update_summary_content()
                
            except Exception as e:
                logger.error(f"Error loading saved limits data: {e}", exc_info=True)
                
    def _load_saved_basic_deviation_data(self):
        """Load saved basic deviation data into the basic deviation panel."""
        if self.saved_basic_deviation_data:
            try:
                # ✅ FIX Problem #1: Panel null check - deferred creation may not be complete
                if not self.basic_deviation_panel:
                    logger.warning(f"[DIALOG STATE] Basic deviation panel not yet created, cannot load saved data")
                    return
                
                self.basic_deviation_panel.set_deviation_settings(self.saved_basic_deviation_data)
                logger.info(f"[DIALOG STATE] Loaded saved basic deviation data for graph {self.graph_index}")
                logger.debug(f"[DIALOG STATE] Deviation data: {self.saved_basic_deviation_data}")
                
                # Update summary panel to reflect loaded data
                self._update_summary_content()
                
            except Exception as e:
                logger.error(f"Error loading saved basic deviation data: {e}", exc_info=True)
        
    def get_selected_signals(self):
        """Get list of selected signals from parameters panel."""
        # NOTE: Parameters panel removed - return the visible signals that were passed in
        # This preserves the current signal selection when dialog is opened/closed
        return self.visible_signals.copy() if self.visible_signals else []
        
    def get_range_filter_conditions(self) -> Dict[str, Any]:
        """Get the current range filter conditions from the panel."""
        return self.parameter_filters_panel.get_range_filter_conditions()

    def get_limits_configuration(self) -> Dict[str, Any]:
        """Get the current static limits configuration from the panel."""
        return self.static_limits_panel.get_limits_configuration()

    def get_basic_deviation_settings(self) -> Dict[str, Any]:
        """Get the current basic deviation settings from the panel."""
        return self.basic_deviation_panel.get_deviation_settings()

    def _apply_settings(self):
        """Apply ALL settings from ALL panels (not just current panel)."""
        logger.info(f"[DIALOG] Applying ALL settings for graph {self.graph_index}...")
        
        # Apply Range Filters (regardless of which panel is active)
        # HER ZAMAN emit et - boş conditions bile, çünkü filter temizlenmiş olabilir
        try:
            filter_conditions = self.parameter_filters_panel.get_range_filter_conditions()
            # graph_index ekle
            filter_conditions['graph_index'] = self.graph_index
            
            num_conditions = len(filter_conditions.get('conditions', []))
            logger.info(f"[DIALOG] Emitting Range Filter signal: {num_conditions} conditions, mode: {filter_conditions.get('mode', 'N/A')}")
            logger.info(f"[DIALOG] Filter data: {filter_conditions}")
            
            # Boş conditions bile emit et - bu filter temizlendi demektir
            self.range_filter_applied.emit(filter_conditions)
            logger.info(f"[DIALOG] Range filter signal EMITTED successfully")
        except Exception as e:
            logger.error(f"Error applying range filters: {e}", exc_info=True)
        
        # Apply Static Limits (regardless of which panel is active)
        # HER ZAMAN emit et - boş config bile, çünkü limitler temizlenmiş olabilir
        try:
            limits_config = self.static_limits_panel.get_limits_configuration()
            logger.debug(f"Applying Static Limits: {len(limits_config) if limits_config else 0} signals")
            # Boş config bile emit et - bu limitler temizlendi demektir
            self.limits_applied.emit(self.graph_index, limits_config if limits_config else {})
        except Exception as e:
            logger.error(f"Error applying static limits: {e}")
        
        # Apply Basic Deviation (regardless of which panel is active)
        try:
            deviation_settings = self.basic_deviation_panel.get_deviation_settings()
            # Check if any meaningful settings are configured
            has_settings = (
                deviation_settings.get('trend_analysis', {}).get('enabled', False) or
                deviation_settings.get('fluctuation_detection', {}).get('enabled', False)
            )
            if has_settings:
                logger.debug(f"Applying Basic Deviation settings")
                self.basic_deviation_applied.emit(self.graph_index, deviation_settings)
            else:
                logger.debug("No basic deviation settings enabled")
        except Exception as e:
            logger.error(f"Error applying basic deviation: {e}")
        
        # Update Current Settings panel to show applied changes
        self._update_summary_content()
        
        logger.info(f"[DIALOG] Finished applying all settings for graph {self.graph_index}")

    def accept(self):
        """Apply ALL settings and close the dialog."""
        logger.info("[DIALOG] ==================== OK BUTTON CLICKED ====================")
        logger.info(f"[DIALOG] Graph index: {self.graph_index}")
        logger.info("[DIALOG] Starting _apply_settings()...")
        self._apply_settings()
        logger.info("[DIALOG] _apply_settings() completed")
        logger.info("[DIALOG] Calling super().accept() to close dialog")
        super().accept()
        logger.info("[DIALOG] Dialog closed")
        
    def get_all_settings(self):
        """Get all settings from all panels."""
        settings = {
            'graph_index': self.graph_index,
            'selected_signals': self.get_selected_signals(),
            'filters': self._get_filter_settings(),
            'limits': self._get_limit_settings(),
            'deviation': self._get_deviation_settings()
        }
        return settings
        
    def _get_filter_settings(self):
        """Get filter settings from the filters panel."""
        try:
            return self.parameter_filters_panel.get_range_filter_conditions()
        except:
            return {}
            
    def _get_limit_settings(self):
        """Get limit settings from the limits panel."""
        try:
            return self.static_limits_panel.get_limits_configuration()
        except:
            return {}
            
    def _get_deviation_settings(self):
        """Get deviation settings from both deviation panels."""
        try:
            basic_settings = self.basic_deviation_panel.get_deviation_settings()
            advanced_settings = self.advanced_deviation_panel.get_deviation_settings()
            return {
                'basic': basic_settings,
                'advanced': advanced_settings
            }
        except:
            return {}
