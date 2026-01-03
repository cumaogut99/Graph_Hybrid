# type: ignore
"""
Parameters Panel Manager

Manages the parameters panel with drag-and-drop functionality.
Parameters can be dragged onto graphs to visualize them.
"""

import logging
from typing import TYPE_CHECKING, List, Callable
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import QObject, pyqtSignal as Signal
import numpy as np

from src.ui.parameter_list import ParameterList
from src.ui.parameter_calculator_dialog import ParameterCalculatorDialog
from src.ui.parameter_item import ParameterItem

if TYPE_CHECKING:
    from ..widgets.time_graph_widget_refactored import TimeGraphWidget

logger = logging.getLogger(__name__)


class ParametersPanelManager(QObject):
    """
    Manages the parameters panel with drag-drop functionality.
    
    Features:
    - List of available parameters
    - Drag parameters onto graphs to plot them
    - Search and filter parameters
    - Custom parameter support
    """
    
    # Signals
    parameter_drag_started = Signal(str)  # parameter_name
    parameter_dropped = Signal(int, str)  # graph_index, parameter_name
    
    def __init__(self, parent_widget: "TimeGraphWidget"):
        super().__init__()
        self.parent = parent_widget
        self.panel = None
        self.parameter_list = None
        
        self._setup_panel()
    
    def _setup_panel(self):
        """Setup the parameters panel UI with drag-drop functionality."""
        # Create main panel widget
        self.panel = QWidget()
        self.panel.setObjectName("parametersPanel")
        
        # Create main layout
        main_layout = QVBoxLayout(self.panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title section
        title_container = QWidget()
        title_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d4a66, stop:1 #1a2332);
                border-bottom: 2px solid rgba(74, 144, 226, 0.5);
            }
        """)
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("🔧 Parameters")
        title_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: #e6f3ff;
            padding: 5px;
        """)
        title_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Drag parameters onto graphs to plot them")
        desc_label.setStyleSheet("""
            font-size: 9pt;
            color: #a0c0e0;
            padding: 2px 5px;
        """)
        title_layout.addWidget(desc_label)
        
        # Add New Parameter button
        self.add_param_btn = QPushButton("➕ Add New Parameter")
        self.add_param_btn.setStyleSheet("""
            QPushButton {
                background: rgba(74, 144, 226, 0.3);
                border: 1px solid rgba(74, 144, 226, 0.5);
                border-radius: 4px;
                padding: 8px;
                color: #e6f3ff;
                font-size: 10pt;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(74, 144, 226, 0.5);
            }
            QPushButton:pressed {
                background: rgba(74, 144, 226, 0.7);
            }
        """)
        self.add_param_btn.clicked.connect(self._on_add_parameter_clicked)
        title_layout.addWidget(self.add_param_btn)
        
        main_layout.addWidget(title_container)
        
        # Parameter list
        self.parameter_list = ParameterList(self.panel)
        self.parameter_list.parameter_drag_started.connect(self._on_parameter_drag_started)
        main_layout.addWidget(self.parameter_list, 1)
        
        logger.debug("Parameters panel initialized with drag-drop support")
    
    def _on_parameter_drag_started(self, parameter_name: str):
        """Handle when a parameter drag starts."""
        logger.info(f"Parameter drag started: {parameter_name}")
        self.parameter_drag_started.emit(parameter_name)
    
    def get_panel(self) -> QWidget:
        """Get the parameters panel widget."""
        return self.panel
    
    def update_columns(self, column_names: list, time_column: str = None):
        """
        Update the parameter list with columns from loaded data.
        
        Args:
            column_names: List of column names from the dataframe
            time_column: Name of the time column (will be excluded)
        """
        if self.parameter_list:
            self.parameter_list.update_columns(column_names, time_column)
            logger.info(f"Parameters panel updated with {len(column_names)} columns")
    
    def update_theme(self):
        """Update panel styling when theme changes."""
        # Theme update logic can be added here if needed
        pass
    
    def _on_add_parameter_clicked(self):
        """Handle Add New Parameter button click."""
        # Get available parameters
        if not self.parameter_list:
            return
        
        available_params = list(self.parameter_list.parameter_items.keys())
        if not available_params:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self.panel,
                "No Parameters",
                "Please load data first to see available parameters."
            )
            return
        
        # Get signal data getter function
        def get_signal_data(param_name: str):
            """
            Get FULL signal data for a parameter.
            
            For MPAI files, this will load the complete data (not just preview).
            This is necessary for parameter calculations to work on the entire dataset.
            """
            if not hasattr(self.parent, 'signal_processor'):
                raise ValueError("Signal processor not available")
            
<<<<<<< HEAD
            signal_processor = self.parent.signal_processor
            
            # Check for MPAI reader first (MpaiDirectoryReader)
            raw_df = getattr(signal_processor, 'raw_dataframe', None)
            is_mpai_reader = raw_df and hasattr(raw_df, 'load_column_slice')
            
            if is_mpai_reader:
                # MPAI file - load full data directly from reader
                logger.info(f"Loading full MPAI data for parameter calculation: {param_name}")
                try:
                    reader = raw_df
                    row_count = reader.get_row_count()
                    
                    # Load signal data
                    y_data = reader.load_column_slice(param_name, 0, row_count)
                    if isinstance(y_data, list):
                        y_data = np.array(y_data, dtype=np.float64)
                    
                    # Get time data
                    time_col = getattr(signal_processor, 'time_column_name', 'time')
                    try:
                        x_data = reader.load_column_slice(time_col, 0, row_count)
                        if isinstance(x_data, list):
                            x_data = np.array(x_data, dtype=np.float64)
                    except Exception:
                        # Generate synthetic time
                        t_start, t_end = reader.get_time_range()
                        x_data = np.linspace(t_start, t_end, row_count, dtype=np.float64)
                    
                    logger.info(f"Loaded {len(y_data)} points for {param_name}")
                    return x_data, y_data
                    
                except Exception as e:
                    logger.error(f"Failed to load MPAI data for {param_name}: {e}")
                    raise ValueError(f"Could not load data for '{param_name}': {e}")
            
            # Non-MPAI: Get from signal_processor cache
            signal_data = signal_processor.get_signal_data(param_name)
            if not signal_data:
                raise ValueError(f"Signal '{param_name}' not found")
            
            # Check if x_data and y_data exist
            if 'x_data' not in signal_data or 'y_data' not in signal_data:
                raise ValueError(f"Signal '{param_name}' missing x_data or y_data")
            
            return np.array(signal_data['x_data']), np.array(signal_data['y_data'])
=======
            signal_data = self.parent.signal_processor.get_signal_data(param_name)
            if not signal_data:
                raise ValueError(f"Signal '{param_name}' not found")
            
            # Check if this is MPAI data (preview mode)
            metadata = signal_data.get('metadata', {})
            is_mpai = metadata.get('mpai', False)
            
            if is_mpai:
                # MPAI file - need to load full data for parameter calculation
                logger.info(f"Loading full MPAI data for parameter calculation: {param_name}")
                
                # Get the MpaiReader from data_manager
                if hasattr(self.parent, 'data_manager') and hasattr(self.parent.data_manager, 'raw_data'):
                    reader = self.parent.data_manager.raw_data
                    if hasattr(reader, 'load_column'):
                        try:
                            # Load complete column data
                            full_y_data = reader.load_column(param_name)
                            
                            # Generate corresponding time data
                            row_count = reader.get_row_count()
                            # Assume 1 Hz sampling for now (TODO: get actual sample rate)
                            full_x_data = np.arange(row_count, dtype=np.float64)
                            
                            logger.info(f"Loaded {len(full_y_data)} points for {param_name}")
                            return full_x_data, full_y_data
                        except Exception as e:
                            logger.error(f"Failed to load full MPAI data: {e}")
                            # Fallback to preview data
                            logger.warning("Falling back to preview data (first 10k points)")
                            return signal_data['x_data'], signal_data['y_data']
            
            # Normal data (already in memory)
            return signal_data['x_data'], signal_data['y_data']
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        
        # Create and show dialog
        dialog = ParameterCalculatorDialog(available_params, get_signal_data, self.panel)
        dialog.parameter_created.connect(self._on_parameter_created)
        
        # Center dialog on parent window
        if self.parent and hasattr(self.parent, 'parent') and self.parent.parent():
            parent_geometry = self.parent.parent().geometry()
            dialog.move(
                parent_geometry.center().x() - dialog.width() // 2,
                parent_geometry.center().y() - dialog.height() // 2
            )
        
        dialog.exec_()
    
<<<<<<< HEAD
    def _on_parameter_created(self, param_name: str, formula: str, used_params: list):
        """
        Handle new parameter creation - STREAMING DISK-BASED calculation.
        
        Instead of keeping calculated parameters in RAM, we:
        1. Start a background StreamingParamWorker
        2. Process data in chunks (100K rows)
        3. Write results to MPAI directory format
        4. Register with MpaiDirectoryReader for zero-copy access
        """
        logger.info(f"[STREAM CALC] Starting disk-based parameter: {param_name}")
        logger.info(f"[STREAM CALC] Formula: {formula}")
        logger.info(f"[STREAM CALC] Uses: {used_params}")
        
        try:
            import os
            from pathlib import Path
            from src.data.streaming_param_worker import StreamingParamWorker
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            
            # Get signal processor
            if not hasattr(self.parent, 'signal_processor'):
                logger.error("Signal processor not available")
                return
            
            signal_processor = self.parent.signal_processor
            
            # Get input reader (MpaiDirectoryReader)
            raw_df = getattr(signal_processor, 'raw_dataframe', None)
            if not raw_df or not hasattr(raw_df, 'load_column_slice'):
                logger.error("No MPAI reader available for streaming calculation")
                # Fallback to legacy (not implemented yet)
                return
            
            # Determine output directory
            base_path = getattr(raw_df, 'mpai_path', None)
            if base_path:
                calc_dir = Path(base_path).parent
            else:
                import tempfile
                calc_dir = Path(tempfile.gettempdir())
            
            # Get time column name
            time_column = getattr(signal_processor, 'time_column_name', 'Time')
            
            # Create progress dialog
            self._progress_dialog = QProgressDialog(
                f"Calculating '{param_name}'...", 
                "Cancel", 
                0, 100,
                self.panel
            )
            self._progress_dialog.setWindowTitle("Streaming Calculation")
            self._progress_dialog.setWindowModality(Qt.WindowModal)
            self._progress_dialog.setMinimumDuration(0)
            self._progress_dialog.setValue(0)
            
            # Create and start worker
            self._calc_worker = StreamingParamWorker(
                param_name=param_name,
                formula=formula,
                used_params=used_params,
                input_reader=raw_df,
                time_column=time_column,
                output_dir=str(calc_dir)
            )
            
            # Connect signals
            self._calc_worker.progress.connect(self._on_calc_progress)
            self._calc_worker.finished.connect(self._on_calc_finished)
            self._calc_worker.error.connect(self._on_calc_error)
            self._progress_dialog.canceled.connect(self._calc_worker.cancel)
            
            # Start background calculation
            self._calc_worker.start()
            
        except Exception as e:
            logger.error(f"[STREAM CALC] Error starting calculation: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.panel, "Error", f"Failed to start calculation: {e}")
    
    def _on_calc_progress(self, percentage: int, message: str):
        """Update progress dialog during streaming calculation."""
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.setValue(percentage)
            self._progress_dialog.setLabelText(message)
    
    def _on_calc_finished(self, param_name: str, output_path: str):
        """Handle successful completion of streaming calculation."""
        logger.info(f"[STREAM CALC] Completed: {param_name} -> {output_path}")
        
        # Close progress dialog
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        
        try:
            from src.data.data_reader import MpaiDirectoryReader
            
            # Open the calculated parameter as MpaiDirectoryReader
            reader = MpaiDirectoryReader(output_path)
            
            # Register with signal processor
            if hasattr(self.parent, 'signal_processor'):
                signal_processor = self.parent.signal_processor
                
                # Get metadata from reader
                row_count = reader.get_row_count()
                t_min, t_max = reader.get_time_range()
                
                # Register the calculated parameter exactly like a regular MPAI signal
                signal_processor.signal_data[param_name] = {
                    'mpai_reader': reader,
                    'column_name': param_name,
                    'time_column': 'Time',  # Synthetic time
                    'row_count': row_count,
                    'time_range': (t_min, t_max),
                    'metadata': {
                        'memory_mapped': True,
                        'calculated': True,
                        'mpai': True,
                        'full_count': row_count,
                        'full_time_range': (t_min, t_max)
                    }
                }
                
                # Store reader reference for cleanup
                if not hasattr(signal_processor, '_calc_param_readers'):
                    signal_processor._calc_param_readers = {}
                signal_processor._calc_param_readers[param_name] = reader
                
                logger.info(f"[STREAM CALC] Registered '{param_name}' ({row_count:,} rows, memory-mapped)")
            
            # Add to parameter list UI
            if self.parameter_list:
                self.parameter_list.add_parameter(param_name)
            
            # Show success message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self.panel,
                "Success",
                f"Parameter '{param_name}' created successfully!\n\n"
                f"Rows: {row_count:,}\n"
                f"Location: {output_path}"
            )
            
        except Exception as e:
            logger.error(f"[STREAM CALC] Failed to register result: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.panel, "Error", f"Failed to register calculated parameter: {e}")
    
    def _on_calc_error(self, error_msg: str):
        """Handle streaming calculation error."""
        logger.error(f"[STREAM CALC] Error: {error_msg}")
        
        # Close progress dialog
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self.panel, "Calculation Error", f"Failed to calculate parameter:\n{error_msg}")
=======
    def _on_parameter_created(self, param_name: str, x_data: np.ndarray, y_data: np.ndarray):
        """Handle new parameter creation."""
        logger.info(f"New parameter created: {param_name}")
        
        # Add to signal processor
        if hasattr(self.parent, 'signal_processor'):
            self.parent.signal_processor.add_signal(param_name, x_data, y_data)
            logger.info(f"Added '{param_name}' to signal processor")
        
        # Add to parameter list
        if self.parameter_list:
            self.parameter_list.add_parameter(param_name)
            logger.info(f"Added '{param_name}' to parameter list")
        
        # Update parent widget if needed
        # Note: _on_processing_finished expects 'all_signals' parameter
        # We don't need to call it here - parameter is already added to signal_processor
        # The UI will update automatically when the parameter is plotted
        logger.debug("Parameter created successfully, no need to trigger full processing refresh")
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

