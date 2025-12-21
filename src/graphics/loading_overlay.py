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
    QFrame, QGraphicsView, QGraphicsScene, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
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
        # For now, draw a spinning gear-like animation
        painter.save()
        painter.translate(center_x, center_y)
        
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

class LoadingOverlay(QWidget):
    """Loading overlay that covers the entire application."""
    
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
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Semi-transparent background
        background = QFrame()
        background.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 150);
                border: none;
            }
        """)
        layout.addWidget(background)
        
        # Content layout
        content_layout = QVBoxLayout(background)
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(20)
        
        # Loading animation
        self.animation_widget = LottieWidget()
        
        # Try to load the Engine Animation.json
        animation_path = os.path.join(os.path.dirname(__file__), "Engine Animation.json")
        if not self.animation_widget.load_animation(animation_path):
            logger.warning("Could not load Engine Animation.json, using fallback animation")
        
        content_layout.addWidget(self.animation_widget, 0, Qt.AlignCenter)
        
        # Loading text
        self.loading_label = QLabel("Loading...")
        self.loading_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.loading_label.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                padding: 10px;
            }
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.loading_label, 0, Qt.AlignCenter)
        
        # Progress info (optional)
        self.progress_label = QLabel("")
        self.progress_label.setFont(QFont("Arial", 12))
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background: transparent;
                padding: 5px;
            }
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setVisible(False)
        content_layout.addWidget(self.progress_label, 0, Qt.AlignCenter)
        
    def show_loading(self, message: str = "Loading...", progress_info: str = ""):
        """Show the loading overlay."""
        try:
            # Update parent geometry if needed
            if self.parent():
                self.setGeometry(self.parent().geometry())
            
            # Update messages
            self.loading_label.setText(message)
            
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
    
    def update_progress(self, progress_info: str):
        """Update progress information."""
        try:
            self.progress_label.setText(progress_info)
            self.progress_label.setVisible(bool(progress_info))
            QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def resizeEvent(self, event):
        """Handle resize events to maintain full coverage."""
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().geometry())

class LoadingManager:
    """Manages loading states and overlays for the application."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.loading_overlay = LoadingOverlay(main_window)
        self.active_operations = set()
        
    def start_operation(self, operation_name: str, message: str = None):
        """Start a loading operation."""
        try:
            self.active_operations.add(operation_name)
            
            if not message:
                message = f"Processing {operation_name}..."
            
            self.loading_overlay.show_loading(message)
            
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
                self.loading_overlay.update_progress(progress_info)
                
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
