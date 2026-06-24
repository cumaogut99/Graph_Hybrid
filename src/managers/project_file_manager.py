"""
Project File Manager for Time Graph Application
================================================

Manages .mpai project files (Motor Performance Analysis Interface format)
Single format: ZIP64 Container with manifest.json

Features:
- Single-file project container (.mpai)
- Fast loading with parquet format
- Append-only saves (O(1) time)
- Crash-safe with manifest versioning
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import polars as pl
from PyQt5.QtCore import QObject, pyqtSignal as Signal

from src.data.mpai_project_manager import (
    MpaiProjectManager, 
    ProjectMetadata,
    ProjectHandle
)

logger = logging.getLogger(__name__)


class ProjectFileManager(QObject):
    """
    Manages .mpai project files using ZIP64 Container format.
    
    Features:
    - Save complete project state (data + layout + metadata)
    - Load project with fast parquet data loading
    - Append-only saves for large files
    - Single-file container for easy sharing
    """
    
    # Signals
    save_progress = Signal(str, int)  # message, percentage
    load_progress = Signal(str, int)  # message, percentage
    save_completed = Signal(str)  # file_path
    load_completed = Signal(dict)  # project_data
    error_occurred = Signal(str)  # error_message
    
    # Project format version
    PROJECT_VERSION = "2.0"
    PROJECT_EXTENSION = ".mpai"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager = MpaiProjectManager()
        self._current_handle: Optional[ProjectHandle] = None
    
    def save_project(
        self, 
        file_path: str,
        dataframe: pl.DataFrame,
        layout_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a complete project to .mpai file (ZIP64 Container).
        
        Args:
            file_path: Path to save the .mpai file
            dataframe: Polars DataFrame with the data
            layout_config: Layout configuration dict (graphs, cursors, etc.)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure .mpai extension
            if not file_path.endswith(self.PROJECT_EXTENSION):
                file_path += self.PROJECT_EXTENSION
            
            logger.info(f"Saving project to: {file_path}")
            self.save_progress.emit("Proje kaydediliyor...", 0)
            
            # Check if we have an existing handle (append mode)
            if (self._current_handle and 
                self._current_handle.path == file_path):
                # Append-only save
                return self._save_append(layout_config, metadata)
            else:
                # New project save
                return self._save_new(file_path, dataframe, layout_config, metadata)
                
        except Exception as e:
            error_msg = f"Proje kaydedilemedi: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return False
    
    def _save_new(
        self,
        file_path: str,
        dataframe: pl.DataFrame,
        layout_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new project file."""
        self.save_progress.emit("Yeni proje oluşturuluyor...", 10)
        
        # Prepare project metadata
        proj_metadata = ProjectMetadata(
            time_column=metadata.get('time_column', 'time') if metadata else 'time',
            sampling_frequency=metadata.get('sampling_frequency', 0.0) if metadata else 0.0,
            layout_config=layout_config,
            columns=[
                {"name": col, "dtype": str(dataframe[col].dtype)}
                for col in dataframe.columns
            ],
        )
        
        # Create ZIP64 container
        self.save_progress.emit("Veri yazılıyor...", 30)
        
        def progress_cb(msg, pct):
            adjusted_pct = 30 + int(pct * 0.6)  # Map 0-100 to 30-90
            self.save_progress.emit(msg, adjusted_pct)
        
        handle = self._manager.create_project(
            dataframe, 
            file_path,
            metadata=proj_metadata,
            progress_callback=progress_cb
        )
        
        self._current_handle = handle
        
        self.save_progress.emit("Proje başarıyla kaydedildi!", 100)
        self.save_completed.emit(file_path)
        logger.info(f"Project saved: {file_path}")
        
        return True
    
    def _save_append(
        self,
        layout_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Append-only save to existing project (fast!)."""
        logger.info("Using append-only save (fast mode)")
        self.save_progress.emit("Değişiklikler kaydediliyor (hızlı mod)...", 30)
        
        # Update metadata
        self._current_handle.metadata.layout_config = layout_config
        
        if metadata:
            self._current_handle.metadata.preferences = metadata.get('preferences', {})
        
        def progress_cb(msg, pct):
            adjusted_pct = 30 + int(pct * 0.6)
            self.save_progress.emit(msg, adjusted_pct)
        
        self._manager.save_project(
            self._current_handle,
            metadata=self._current_handle.metadata,
            progress_callback=progress_cb
        )
        
        self.save_progress.emit("Proje başarıyla kaydedildi!", 100)
        self.save_completed.emit(self._current_handle.path)
        logger.info(f"Project saved (append): {self._current_handle.path}")
        
        return True
    
    def save_directory_project(
        self,
        source_dir: str,
        file_path: str,
        layout_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a project by packing an existing MPAI directory into a .mpai file.
        
        Args:
            source_dir: Path to source directory (unpacked MPAI)
            file_path: Path to save the .mpai file (packed ZIP)
            layout_config: Layout configuration to save
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Ensure .mpai extension
            if not file_path.endswith(self.PROJECT_EXTENSION):
                file_path += self.PROJECT_EXTENSION
            
            logger.info(f"Saving directory project: {source_dir} -> {file_path}")
            self.save_progress.emit("Paketleme işlemi başlatılıyor...", 0)
            
            # Prepare project metadata
            # We assume row/col counts are already in the source directory's manifest
            # or will be read by create_project_from_directory
            original_filename = "unknown"
            if metadata:
                original_filename = metadata.get('original_file', 'unknown')
                
            proj_metadata = ProjectMetadata(
                time_column=metadata.get('time_column', 'time') if metadata else 'time',
                sampling_frequency=metadata.get('sampling_frequency', 0.0) if metadata else 0.0,
                layout_config=layout_config,
                preferences={'original_file': original_filename} # Store original filename
            )
            
            def progress_cb(msg, pct):
                self.save_progress.emit(msg, pct)
            
            handle = self._manager.create_project_from_directory(
                source_dir,
                file_path,
                metadata=proj_metadata,
                progress_callback=progress_cb
            )
            
            self._current_handle = handle
            
            self.save_progress.emit("Proje başarıyla paketlendi!", 100)
            self.save_completed.emit(file_path)
            
            return True
            
        except Exception as e:
            error_msg = f"Proje paketlenemedi: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return False
    
    def load_project(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a project from .mpai file (ZIP64 Container).
        
        Args:
            file_path: Path to the .mpai file
            
        Returns:
            Dict containing:
                - 'dataframe': Polars DataFrame
                - 'layout_config': Layout configuration
                - 'metadata': Project metadata
            None if failed
        """
        try:
            logger.info(f"Loading project from: {file_path}")
            self.load_progress.emit("Proje açılıyor...", 0)
            
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Proje dosyası bulunamadı: {file_path}")
            
            if not file_path.endswith(self.PROJECT_EXTENSION):
                raise ValueError(f"Geçersiz dosya uzantısı. .mpai dosyası bekleniyor.")
            
            # Open ZIP64 container
            self.load_progress.emit("ZIP64 container açılıyor...", 10)
            
            handle = self._manager.open_project(file_path)
            self._current_handle = handle
            
            # Check if this is a directory-based container (has setup.xml)
            is_directory_container = False
            import zipfile
            import tempfile
            import shutil
            
            with zipfile.ZipFile(file_path, 'r') as zf:
                if 'setup.xml' in zf.namelist():
                    is_directory_container = True
                    logger.info("Detected directory-based MPAI container")
            
            if is_directory_container:
                # Extract to temp directory and use MpaiDirectoryReader
                self.load_progress.emit("Klasör tabanlı proje açılıyor...", 30)
                
                temp_dir = tempfile.mkdtemp(prefix="mpai_project_")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        zf.extractall(temp_dir)
                    
                    from src.data.data_reader import MpaiDirectoryReader
                    reader = MpaiDirectoryReader(temp_dir)
                    
                    # ✅ NEW: Detect calculated parameters in the extracted directory
                    # They are stored as calculated_<param_name> subdirectories
                    calc_param_dirs = {}
                    logger.info(f"[LOAD] Scanning for calculated params in: {temp_dir}")
                    
                    dir_contents = os.listdir(temp_dir)
                    logger.info(f"[LOAD] Directory contents: {dir_contents}")
                    
                    for item in dir_contents:
                        item_path = os.path.join(temp_dir, item)
                        if os.path.isdir(item_path):
                            logger.info(f"[LOAD] Found directory: {item}")
                            if item.startswith("calculated_"):
                                param_name = item[len("calculated_"):]  # Remove prefix
                                # Check if it's a valid MPAI directory (has setup.xml)
                                setup_path = os.path.join(item_path, "setup.xml")
                                if os.path.exists(setup_path):
                                    calc_param_dirs[param_name] = item_path
                                    logger.info(f"[LOAD] ✓ Found calculated parameter: {param_name} at {item_path}")
                                else:
                                    logger.warning(f"[LOAD] ✗ Directory {item} has no setup.xml")
                    
                    logger.info(f"[LOAD] Total calculated params found: {len(calc_param_dirs)}")
                    
                    # For directory-based, we return the reader instead of DataFrame
                    # The caller (TimeGraphWidget) should handle this
                    layout_config = handle.metadata.layout_config if handle.metadata.layout_config else {}
                    
                    metadata = {
                        'version': handle.manifest.format_version,
                        'format': 'directory_container',
                        'created_at': handle.manifest.created_at,
                        'modified_at': handle.manifest.modified_at,
                        'row_count': reader.get_row_count(),
                        'column_count': reader.get_column_count(),
                        'time_column': handle.metadata.time_column,
                        'sampling_frequency': handle.metadata.sampling_frequency or reader.sample_rate,
                        'columns': reader.get_column_names(),
                        'has_lod': False,
                        'temp_dir': temp_dir,  # Store for cleanup later
                        'calculated_params': calc_param_dirs,  # ✅ NEW: Include calc param paths
                    }
                    
                    project_data = {
                        'reader': reader,  # Return reader instead of dataframe
                        'layout_config': layout_config,
                        'metadata': metadata,
                        'file_path': file_path,
                        'is_directory_based': True,
                    }
                    
                    calc_count = len(calc_param_dirs)
                    logger.info(f"Directory-based project loaded: {reader.get_row_count()} rows, {reader.get_column_count()} columns, {calc_count} calculated params")
                    self.load_progress.emit("Proje başarıyla yüklendi!", 100)
                    self.load_completed.emit(project_data)
                    
                    return project_data

                    
                except Exception as e:
                    # Cleanup temp dir on failure
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    raise
            else:
                # Standard parquet-based container
                self.load_progress.emit("Veri yükleniyor...", 30)
                dataframe = self._manager.get_raw_data_polars(handle)
                logger.info(f"Loaded: {len(dataframe)} rows, {len(dataframe.columns)} columns")
                
                # Get layout config
                self.load_progress.emit("Layout yükleniyor...", 70)
                layout_config = handle.metadata.layout_config if handle.metadata.layout_config else {}
                
                # Build metadata dict
                self.load_progress.emit("Metadata yükleniyor...", 85)
                metadata = {
                    'version': handle.manifest.format_version,
                    'format': 'zip64_container',
                    'created_at': handle.manifest.created_at,
                    'modified_at': handle.manifest.modified_at,
                    'row_count': handle.manifest.row_count,
                    'column_count': handle.manifest.column_count,
                    'time_column': handle.metadata.time_column,
                    'sampling_frequency': handle.metadata.sampling_frequency,
                    'columns': handle.metadata.columns,
                    'has_lod': len(handle.manifest.lod_files) > 0,
                }
                
                project_data = {
                    'dataframe': dataframe,
                    'layout_config': layout_config,
                    'metadata': metadata,
                    'file_path': file_path,
                }
                
                logger.info(f"Project loaded successfully: {file_path}")
                self.load_progress.emit("Proje başarıyla yüklendi!", 100)
                self.load_completed.emit(project_data)
            
            return project_data
            
        except Exception as e:
            error_msg = f"Proje yüklenemedi: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return None
    
    def validate_project(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate a .mpai project file.
        
        Args:
            file_path: Path to the .mpai file
            
        Returns:
            (is_valid, message) tuple
        """
        is_valid, format_type = self._manager.is_mpai_project(file_path)
        
        if not is_valid:
            return False, "Geçersiz veya bozuk MPAI dosyası"
        
        if format_type == "zip":
            return True, "Geçerli MPAI proje dosyası"
        else:
            return False, f"Bilinmeyen format: {format_type}"
    
    def get_project_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get project information without loading the full data.
        Useful for file browser preview.
        
        Args:
            file_path: Path to .mpai file
            
        Returns:
            Project info dict or None
        """
        info = self._manager.get_project_info(file_path)
        
        if info:
            # Add file stats
            file_stats = os.stat(file_path)
            info['file_info'] = {
                'size_bytes': file_stats.st_size,
                'size_mb': file_stats.st_size / (1024 * 1024),
                'modified_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
        
        return info
    
    def cleanup(self):
        """Clean up resources."""
        if self._current_handle:
            try:
                self._current_handle.close()
            except:
                pass
            self._current_handle = None
    
    def __del__(self):
        """Destructor - cleanup resources."""
        self.cleanup()


def create_project_from_csv(
    csv_path: str,
    mpai_path: str,
    layout_config: Dict[str, Any],
    import_settings: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Helper function to create a .mpai project from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        mpai_path: Path to save .mpai file
        layout_config: Layout configuration
        import_settings: CSV import settings (encoding, delimiter, etc.)
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Creating project from CSV: {csv_path}")
        
        # Load CSV
        df = pl.read_csv(csv_path)
        
        # Create metadata
        metadata = {
            'original_file': os.path.basename(csv_path),
            'original_file_path': csv_path,
            'import_settings': import_settings or {}
        }
        
        # Save project
        manager = ProjectFileManager()
        success = manager.save_project(mpai_path, df, layout_config, metadata)
        manager.cleanup()
        
        return success
        
    except Exception as e:
        logger.error(f"Error creating project from CSV: {e}")
        return False
