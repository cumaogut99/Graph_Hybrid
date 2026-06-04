"""
Loading Overlay with Lottie Animation for Time Graph Widget
Shows loading animation when operations are in progress
"""

import logging
import json
import os
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QGraphicsView, QGraphicsScene, QApplication, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QPalette

logger = logging.getLogger(__name__)

class LottieWidget(QWidget):
    """Simple Lottie animation widget using basic animation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation_data = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 25
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._next_frame)
        
        # Rotation animation for fallback
        self.rotation_angle = 0
        self.rotation_animation = QPropertyAnimation(self, b"rotation")
        self.rotation_animation.setDuration(2000)  # 2 seconds
        self.rotation_animation.setStartValue(0)
        self.rotation_animation.setEndValue(360)
        self.rotation_animation.setLoopCount(-1)  # Infinite loop
        
        # Progress percentage for ring
        self._progress_percentage = 0
        
        self.setFixedSize(120, 120)
        
    def load_animation(self, file_path: str) -> bool:
        """Load Lottie animation from JSON file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Animation file not found: {file_path}")
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                self.animation_data = json.load(f)
                
            # Extract animation properties
            self.total_frames = self.animation_data.get('op', 100)  # out point
            self.fps = self.animation_data.get('fr', 25)  # frame rate
            
            logger.info(f"Loaded animation: {self.total_frames} frames at {self.fps} fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load animation: {e}")
            return False
    
    def start_animation(self):
        """Start the animation."""
        if self.animation_data:
            # Start Lottie animation
            interval = int(1000 / self.fps)  # Convert to milliseconds
            self.animation_timer.start(interval)
            logger.debug("Lottie animation started")
        else:
            # Fallback to rotation animation
            self.rotation_animation.start()
            logger.debug("Fallback rotation animation started")
    
    def stop_animation(self):
        """Stop the animation."""
        self.animation_timer.stop()
        self.rotation_animation.stop()
        self.current_frame = 0
        self.rotation_angle = 0
        self.update()
        
    def _next_frame(self):
        """Advance to next frame."""
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.update()
    
    @pyqtProperty(int)
    def rotation(self):
        return self.rotation_angle
    
    @rotation.setter
    def rotation(self, angle):
        self.rotation_angle = angle
        self.update()
    
    @pyqtProperty(int)
    def progress_percentage(self):
        return self._progress_percentage
    
    @progress_percentage.setter
    def progress_percentage(self, value):
        self._progress_percentage = max(0, min(100, value))
        self.update()
        
    def paintEvent(self, event):
        """Paint the animation frame."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget center
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        if self.animation_data and self.animation_timer.isActive():
            # Draw Lottie animation (simplified - just a spinning circle for now)
            self._draw_lottie_frame(painter, center_x, center_y)
        else:
            # Draw fallback spinning animation
            self._draw_fallback_animation(painter, center_x, center_y)
    
    def _draw_lottie_frame(self, painter, center_x, center_y):
        """Draw current Lottie animation frame (simplified implementation)."""
        painter.save()
        painter.translate(center_x, center_y)
        
        # Draw progress ring first (behind the gear)
        self._draw_progress_ring(painter)
        
        # Calculate rotation based on current frame
        rotation = (self.current_frame / self.total_frames) * 360
        painter.rotate(rotation)
        
        # Draw gear-like shape
        painter.setPen(QColor(74, 144, 226))  # Blue
        painter.setBrush(QColor(74, 144, 226, 100))
        
        # Draw outer circle
        painter.drawEllipse(-40, -40, 80, 80)
        
        # Draw inner spokes
        for i in range(8):
            angle = i * 45
            painter.save()
            painter.rotate(angle)
            painter.drawRect(-3, -50, 6, 20)
            painter.restore()
        
        # Draw inner circle
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(-15, -15, 30, 30)
        
        painter.restore()
    
    def _draw_fallback_animation(self, painter, center_x, center_y):
        """Draw fallback spinning animation."""
        painter.save()
        painter.translate(center_x, center_y)
        
        # Draw progress ring first (behind the animation)
        self._draw_progress_ring(painter)
        
        painter.rotate(self.rotation_angle)
        
        # Draw spinning circle with dots
        painter.setPen(QColor(74, 144, 226))
        painter.setBrush(QColor(74, 144, 226, 150))
        
        # Draw dots around circle
        for i in range(12):
            angle = i * 30
            painter.save()
            painter.rotate(angle)
            
            # Fade effect
            alpha = int(255 * (1 - i / 12))
            painter.setBrush(QColor(74, 144, 226, alpha))
            painter.drawEllipse(0, -35, 8, 8)
            
            painter.restore()
        
        painter.restore()
    
    def _draw_progress_ring(self, painter):
        """Draw circular progress ring around the animation."""
        painter.save()
        
        ring_radius = 55
        ring_width = 6
        
        # Draw background ring (gray)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(60, 60, 70, 100))
        painter.drawEllipse(-ring_radius, -ring_radius, ring_radius * 2, ring_radius * 2)
        
        # Draw inner circle to create ring effect
        inner_radius = ring_radius - ring_width
        painter.setBrush(QColor(20, 20, 30, 240))  # Match card background
        painter.drawEllipse(-inner_radius, -inner_radius, inner_radius * 2, inner_radius * 2)
        
        # Draw progress arc (blue)
        if self._progress_percentage > 0:
            from PyQt5.QtGui import QPen, QPainterPath
            from PyQt5.QtCore import QRectF
            
            # Calculate span angle (in 16th of a degree)
            span_angle = int((self._progress_percentage / 100.0) * 360 * 16)
            
            # Create path for progress arc
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(74, 144, 226, 200))  # Blue
            
            # Draw outer arc
            outer_rect = QRectF(-ring_radius, -ring_radius, ring_radius * 2, ring_radius * 2)
            path = QPainterPath()
            path.arcMoveTo(outer_rect, 90)  # Start at top (12 o'clock)
            path.arcTo(outer_rect, 90, -self._progress_percentage / 100.0 * 360)
            
            # Draw inner arc (reverse direction)
            inner_rect = QRectF(-inner_radius, -inner_radius, inner_radius * 2, inner_radius * 2)
            path.arcTo(inner_rect, 90 - self._progress_percentage / 100.0 * 360, self._progress_percentage / 100.0 * 360)
            path.closeSubpath()
            
            painter.drawPath(path)
        
        painter.restore()

class LoadingOverlay(QWidget):
    """Loading overlay that covers the entire application."""
    
    cancel_requested = Signal()  # Signal emitted when cancel button is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Make it cover the entire parent
        if parent:
            self.setGeometry(parent.geometry())
        
        self._setup_ui()
        self.hide()  # Hidden by default
        
    def _setup_ui(self):
        """Setup the loading overlay UI."""
        # Main layout - centered
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        
        # Compact loading card (not full screen)
        loading_card = QFrame()
        loading_card.setFixedSize(400, 300)  # Compact size
        loading_card.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 30, 240);
                border: 2px solid rgba(74, 144, 226, 180);
                border-radius: 12px;
            }
        """)
        layout.addWidget(loading_card)
        
        # Content layout inside card
        content_layout = QVBoxLayout(loading_card)
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(12)  # Reduced from 20 to 12
        content_layout.setContentsMargins(20, 20, 20, 20)  # Reduced from 30 to 20
        
        # Loading animation
        self.animation_widget = LottieWidget()
        
        # Try to load the Engine Animation.json
        animation_path = os.path.join(os.path.dirname(__file__), "Engine Animation.json")
        if not self.animation_widget.load_animation(animation_path):
            logger.warning("Could not load Engine Animation.json, using fallback animation")
        
        content_layout.addWidget(self.animation_widget, 0, Qt.AlignCenter)
        
        # Add spacing between animation and percentage
        content_layout.addSpacing(15)
        
        # Percentage label (large, bold, prominent)
        self.percentage_label = QLabel("0%")
        self.percentage_label.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Large and bold
        self.percentage_label.setStyleSheet("""
            QLabel {
                color: #4A90E2;
                background: transparent;
                padding: 0px;
            }
        """)
        self.percentage_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.percentage_label, 0, Qt.AlignCenter)
        
        # Subtitle label for conversion message
        self.subtitle_label = QLabel("")
        self.subtitle_label.setFont(QFont("Segoe UI", 9))
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                background: transparent;
                padding: 2px;
            }
        """)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setVisible(False)
        content_layout.addWidget(self.subtitle_label, 0, Qt.AlignCenter)
        
        # Progress info (message without percentage)
        self.progress_label = QLabel("")
        self.progress_label.setFont(QFont("Segoe UI", 9))
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background: transparent;
                padding: 2px;
            }
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setVisible(False)
        content_layout.addWidget(self.progress_label, 0, Qt.AlignCenter)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont("Segoe UI", 9))
        self.cancel_button.setFixedSize(80, 28)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(232, 17, 35, 200);
                color: white;
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: rgba(241, 112, 122, 220);
                border: 1px solid rgba(255, 255, 255, 150);
            }
            QPushButton:pressed {
                background-color: rgba(200, 10, 25, 230);
            }
        """)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        content_layout.addWidget(self.cancel_button, 0, Qt.AlignCenter)
    
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        logger.info("Cancel button clicked")
        self.cancel_requested.emit()
        
    def show_loading(self, message: str = "Processing...", progress_info: str = "", subtitle: str = ""):
        """Show the loading overlay."""
        try:
            # Update parent geometry if needed
            if self.parent():
                self.setGeometry(self.parent().geometry())
            
            # Update subtitle (e.g., conversion message)
            if subtitle:
                self.subtitle_label.setText(subtitle)
                self.subtitle_label.setVisible(True)
            else:
                self.subtitle_label.setVisible(False)
            
            if progress_info:
                self.progress_label.setText(progress_info)
                self.progress_label.setVisible(True)
            else:
                self.progress_label.setVisible(False)
            
            # Start animation and show
            self.animation_widget.start_animation()
            self.show()
            self.raise_()  # Bring to front
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            logger.info(f"Loading overlay shown: {message}")
            
        except Exception as e:
            logger.error(f"Error showing loading overlay: {e}")
    
    def hide_loading(self):
        """Hide the loading overlay."""
        try:
            self.animation_widget.stop_animation()
            self.hide()
            logger.info("Loading overlay hidden")
        except Exception as e:
            logger.error(f"Error hiding loading overlay: {e}")
    
    def update_progress(self, progress_info: str, progress_percent: int = None):
        """Update progress information."""
        try:
            # Update percentage label and animation ring
            if progress_percent is not None:
                self.percentage_label.setText(f"{progress_percent}%")
                self.animation_widget.progress_percentage = progress_percent
            
            # Update progress message (without percentage)
            if progress_info:
                self.progress_label.setText(progress_info)
                self.progress_label.setVisible(True)
            else:
                self.progress_label.setVisible(False)
            
            QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def resizeEvent(self, event):
        """Handle resize events to maintain full coverage."""
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().geometry())

class LoadingManager(QObject):
    """Manages loading states and overlays for the application."""
    
    cancel_requested = Signal()  # Signal emitted when user cancels loading
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.loading_overlay = LoadingOverlay(main_window)
        self.active_operations = set()
        
        # Connect overlay cancel signal
        self.loading_overlay.cancel_requested.connect(self._on_cancel_requested)
    
    def _on_cancel_requested(self):
        """Handle cancel request from overlay."""
        logger.info("Loading operation cancelled by user")
        self.cancel_requested.emit()
        
    def start_operation(self, operation_name: str, message: str = None, subtitle: str = ""):
        """Start a loading operation."""
        try:
            self.active_operations.add(operation_name)
            
            if not message:
                message = f"Processing {operation_name}..."
            
            self.loading_overlay.show_loading(message, subtitle=subtitle)
            
            # Update status bar if available
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.set_operation(operation_name, 0)
            
            logger.info(f"Started operation: {operation_name}")
            
        except Exception as e:
            logger.error(f"Error starting operation {operation_name}: {e}")
    
    def update_operation(self, operation_name: str, progress_info: str = "", progress_percent: int = None):
        """Update an active operation."""
        try:
            if operation_name in self.active_operations:
                self.loading_overlay.update_progress(progress_info, progress_percent)
                
                # Update status bar if available
                if hasattr(self.main_window, 'status_bar') and progress_percent is not None:
                    self.main_window.status_bar.update_progress(progress_percent)
                    
        except Exception as e:
            logger.error(f"Error updating operation {operation_name}: {e}")
    
    def finish_operation(self, operation_name: str):
        """Finish a loading operation."""
        try:
            if operation_name in self.active_operations:
                self.active_operations.remove(operation_name)
                
                # If no more operations, hide overlay
                if not self.active_operations:
                    self.loading_overlay.hide_loading()
                    
                    # Update status bar
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.set_operation("Ready")
                
                logger.info(f"Finished operation: {operation_name}")
                
        except Exception as e:
            logger.error(f"Error finishing operation {operation_name}: {e}")
    
    def is_loading(self) -> bool:
        """Check if any operations are active."""
        return len(self.active_operations) > 0
    
    def get_active_operations(self) -> list:
        """Get list of active operations."""
        return list(self.active_operations)
