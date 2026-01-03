"""
MPAI Project Manager - Single-File Container Architecture

Manages .mpai project files using ZIP64 format with manifest-based versioning.

Features:
- ZIP_STORED mode (no compression, fast random access)
- Manifest-based versioning (crash-safe)
- Append-only saves (O(1) save time)
- Stream-based reading (zero-copy when possible)

Architecture:
    project.mpai (ZIP64 container)
    ├── manifest.json              # Current version pointers
    ├── raw_data.parquet           # Immutable raw data
    ├── lod/
    │   ├── lod1_100.parquet
    │   ├── lod2_10k.parquet
    │   └── lod3_100k.parquet
    ├── metadata_v001.json         # Version 1 metadata
    ├── calculations_v001.parquet  # Version 1 calculations
    └── ...

Performance Target: Zero-lag for 50GB files (ATI Vision / Dewesoft standard)
"""

import io
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ManifestInfo:
    """Manifest structure stored in manifest.json"""
    version: int = 1
    format_version: str = "2.0.0"
    created_at: str = ""
    modified_at: str = ""
    
    # Current version pointers
    current_metadata_version: int = 1
    current_calculations_version: int = 0  # 0 = no calculations
    
    # File references
    raw_data_file: str = "raw_data.parquet"
    lod_files: List[str] = field(default_factory=list)
    
    # Statistics
    row_count: int = 0
    column_count: int = 0
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "current_metadata_version": self.current_metadata_version,
            "current_calculations_version": self.current_calculations_version,
            "raw_data_file": self.raw_data_file,
            "lod_files": self.lod_files,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_size_bytes": self.file_size_bytes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestInfo":
        return cls(
            version=data.get("version", 1),
            format_version=data.get("format_version", "2.0.0"),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            current_metadata_version=data.get("current_metadata_version", 1),
            current_calculations_version=data.get("current_calculations_version", 0),
            raw_data_file=data.get("raw_data_file", "raw_data.parquet"),
            lod_files=data.get("lod_files", []),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            file_size_bytes=data.get("file_size_bytes", 0),
        )


@dataclass 
class ProjectMetadata:
    """Project metadata stored in metadata_vXXX.json"""
    version: int = 1
    
    # Time column info
    time_column: str = "time"
    sampling_frequency: float = 0.0
    
    # Column information
    columns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Application state
    layout_config: Dict[str, Any] = field(default_factory=dict)
    cursors: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "time_column": self.time_column,
            "sampling_frequency": self.sampling_frequency,
            "columns": self.columns,
            "layout_config": self.layout_config,
            "cursors": self.cursors,
            "filters": self.filters,
            "annotations": self.annotations,
            "preferences": self.preferences,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectMetadata":
        return cls(
            version=data.get("version", 1),
            time_column=data.get("time_column", "time"),
            sampling_frequency=data.get("sampling_frequency", 0.0),
            columns=data.get("columns", []),
            layout_config=data.get("layout_config", {}),
            cursors=data.get("cursors", []),
            filters=data.get("filters", []),
            annotations=data.get("annotations", []),
            preferences=data.get("preferences", {}),
        )


@dataclass
class ProjectHandle:
    """Handle for an opened project"""
    path: str
    manifest: ManifestInfo
    metadata: ProjectMetadata
    _zip_file: Optional[zipfile.ZipFile] = None
    _temp_dir: Optional[str] = None
    
    def close(self):
        """Close the project and cleanup resources"""
        if self._zip_file:
            try:
                self._zip_file.close()
            except:
                pass
            self._zip_file = None
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
            except:
                pass
            self._temp_dir = None


# =============================================================================
# Main Class
# =============================================================================

class MpaiProjectManager:
    """
    Single-file project container (.mpai) using ZIP64 format.
    
    Key Features:
    - ZIP_STORED mode: No compression overhead, OS-level fast random access
    - Manifest versioning: Crash-safe, never overwrite existing data
    - Append-only saves: 50GB file saves in milliseconds (only new data appended)
    - Stream reading: Direct BytesIO access without disk extraction
    
    Usage:
        # Create new project from CSV
        manager = MpaiProjectManager()
        manager.create_project("data.csv", "project.mpai")
        
        # Open existing project
        handle = manager.open_project("project.mpai")
        df = manager.get_raw_data_stream(handle)
        
        # Save changes (append-only)
        manager.save_project(handle, new_metadata, calculations_df)
        
        # Close
        handle.close()
    """
    
    # ZIP configuration
    COMPRESSION = zipfile.ZIP_STORED  # No compression for speed
    ALLOW_ZIP64 = True  # Support files > 4GB
    
    # File naming conventions
    MANIFEST_FILE = "manifest.json"
    RAW_DATA_FILE = "raw_data.parquet"
    METADATA_PATTERN = "metadata_v{:03d}.json"
    CALCULATIONS_PATTERN = "calculations_v{:03d}.parquet"
    LOD_DIR = "lod/"
    
    # Performance thresholds
    STREAM_SIZE_LIMIT = 1024 * 1024 * 1024  # 1GB - use temp file above this
    
    def __init__(self):
        self._current_handle: Optional[ProjectHandle] = None
    
    # =========================================================================
    # Project Creation
    # =========================================================================
    
    def create_project(
        self,
        parquet_data: Union[str, pl.DataFrame, bytes],
        output_path: str,
        metadata: Optional[ProjectMetadata] = None,
        lod_files: Optional[Dict[str, bytes]] = None,
        progress_callback: Optional[callable] = None
    ) -> ProjectHandle:
        """
        Create a new .mpai project file.
        
        Args:
            parquet_data: Source data - can be:
                - Path to existing parquet file
                - Polars DataFrame
                - Raw parquet bytes
            output_path: Path for the new .mpai file
            metadata: Optional project metadata
            lod_files: Optional dict of LOD file name -> bytes
            progress_callback: Optional callback(message, percentage)
        
        Returns:
            ProjectHandle for the created project
        """
        logger.info(f"Creating new MPAI project: {output_path}")
        
        if progress_callback:
            progress_callback("Creating project container...", 0)
        
        # Prepare manifest
        now = datetime.now().isoformat()
        manifest = ManifestInfo(
            created_at=now,
            modified_at=now,
            current_metadata_version=1,
            current_calculations_version=0,
        )
        
        # Prepare metadata
        if metadata is None:
            metadata = ProjectMetadata()
        
        # Create ZIP container
        with zipfile.ZipFile(
            output_path, 
            mode='w',
            compression=self.COMPRESSION,
            allowZip64=self.ALLOW_ZIP64
        ) as zf:
            
            # 1. Write raw data
            if progress_callback:
                progress_callback("Writing raw data...", 10)
            
            if isinstance(parquet_data, str):
                # Path to existing parquet
                if os.path.exists(parquet_data):
                    zf.write(parquet_data, self.RAW_DATA_FILE)
                    manifest.file_size_bytes = os.path.getsize(parquet_data)
                else:
                    raise FileNotFoundError(f"Parquet file not found: {parquet_data}")
                    
            elif isinstance(parquet_data, pl.DataFrame):
                # DataFrame - write to bytes buffer
                buffer = io.BytesIO()
                parquet_data.write_parquet(buffer)
                buffer.seek(0)
                zf.writestr(self.RAW_DATA_FILE, buffer.getvalue())
                manifest.row_count = len(parquet_data)
                manifest.column_count = len(parquet_data.columns)
                manifest.file_size_bytes = buffer.tell()
                
                # Update metadata with column info
                if not metadata.columns:
                    metadata.columns = [
                        {"name": col, "dtype": str(parquet_data[col].dtype)}
                        for col in parquet_data.columns
                    ]
                    
            elif isinstance(parquet_data, bytes):
                # Raw bytes
                zf.writestr(self.RAW_DATA_FILE, parquet_data)
                manifest.file_size_bytes = len(parquet_data)
            else:
                raise TypeError(f"Unsupported parquet_data type: {type(parquet_data)}")
            
            # 2. Write LOD files
            if lod_files:
                if progress_callback:
                    progress_callback("Writing LOD pyramid...", 50)
                
                for lod_name, lod_bytes in lod_files.items():
                    lod_path = f"{self.LOD_DIR}{lod_name}"
                    zf.writestr(lod_path, lod_bytes)
                    manifest.lod_files.append(lod_path)
            
            # 3. Write metadata v001
            if progress_callback:
                progress_callback("Writing metadata...", 80)
            
            metadata_file = self.METADATA_PATTERN.format(1)
            metadata_json = json.dumps(metadata.to_dict(), indent=2)
            zf.writestr(metadata_file, metadata_json)
            
            # 4. Write manifest (last, so it's only written if everything else succeeded)
            if progress_callback:
                progress_callback("Finalizing...", 95)
            
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            zf.writestr(self.MANIFEST_FILE, manifest_json)
        
        logger.info(f"Project created successfully: {output_path}")
        
        if progress_callback:
            progress_callback("Complete!", 100)
        
        # Return handle
        return ProjectHandle(
            path=output_path,
            manifest=manifest,
            metadata=metadata,
        )
    
<<<<<<< HEAD
    def create_project_from_directory(
        self,
        source_dir: str,
        output_path: str,
        metadata: Optional[ProjectMetadata] = None,
        progress_callback: Optional[callable] = None
    ) -> ProjectHandle:
        """
        Create a single-file .mpai project from an existing MPAI directory.
        
        Optimized for large datasets:
        1. Uses ZIP_STORED (no compression) for speed
        2. Streams files directly from disk to ZIP without RAM loading
        
        Args:
            source_dir: Path to source MPAI directory
            output_path: Path for the new .mpai file
            metadata: Project metadata to save
            progress_callback: Optional callback(message, percentage)
            
        Returns:
            ProjectHandle for the created project
        """
        logger.info(f"Packing MPAI directory to single file: {source_dir} -> {output_path}")
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        if progress_callback:
            progress_callback("Scanning directory...", 0)
            
        # Collect all files to zip
        files_to_zip = []
        total_size = 0
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                
                # Skip temp files or lock files
                if file.endswith('.lock') or file.endswith('.tmp'):
                    continue
                    
                size = os.path.getsize(full_path)
                files_to_zip.append((full_path, rel_path, size))
                total_size += size
                
        if progress_callback:
            progress_callback(f"Found {len(files_to_zip)} files ({total_size / 1024 / 1024:.1f} MB)", 5)
            
        # Create ZIP container
        with zipfile.ZipFile(
            output_path, 
            mode='w',
            compression=self.COMPRESSION,
            allowZip64=self.ALLOW_ZIP64
        ) as zf:
            
            processed_size = 0
            last_progress = 0
            
            # Write all files from directory
            for full_path, rel_path, size in files_to_zip:
                # Special handling for metadata files if we are providing new metadata
                if metadata and rel_path == self.MANIFEST_FILE:
                    continue # We will write a new manifest
                if metadata and rel_path.startswith("metadata_v"):
                    continue # We will write new metadata
                    
                zf.write(full_path, rel_path)
                
                # Update progress
                processed_size += size
                if total_size > 0:
                    current_progress = 5 + int((processed_size / total_size) * 85) # 5% to 90%
                    if current_progress > last_progress + 1: # Only emit if changed by > 1%
                        if progress_callback:
                            progress_callback(f"Packing: {rel_path}...", current_progress)
                        last_progress = current_progress
            
            # Write updated metadata and manifest
            if metadata:
                if progress_callback:
                    progress_callback("Writing metadata...", 92)
                
                # Ensure metadata has correct format version
                manifest_path = os.path.join(source_dir, self.MANIFEST_FILE)
                row_count = 0
                column_count = 0
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        old_manifest = json.load(f)
                        row_count = old_manifest.get('row_count', 0)
                        column_count = old_manifest.get('column_count', 0)
                
                # Create new manifest
                now = datetime.now().isoformat()
                manifest = ManifestInfo(
                    created_at=now,
                    modified_at=now,
                    current_metadata_version=1,
                    current_calculations_version=0,
                    row_count=row_count,
                    column_count=column_count,
                    file_size_bytes=total_size,
                    format_version="2.1.0" # Directory-sourced container
                )
                
                # Write metadata v001
                metadata_file = self.METADATA_PATTERN.format(1)
                metadata_json = json.dumps(metadata.to_dict(), indent=2)
                zf.writestr(metadata_file, metadata_json)
                
                # Write manifest
                manifest_json = json.dumps(manifest.to_dict(), indent=2)
                zf.writestr(self.MANIFEST_FILE, manifest_json)
                
        logger.info(f"Project packed successfully: {output_path}")
        
        if progress_callback:
            progress_callback("Complete!", 100)
            
        # Return handle
        return ProjectHandle(
            path=output_path,
            manifest=manifest if metadata else ManifestInfo(),
            metadata=metadata if metadata else ProjectMetadata(),
        )

=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    # =========================================================================
    # Project Opening
    # =========================================================================
    
    def open_project(self, mpai_path: str) -> ProjectHandle:
        """
        Open an existing .mpai project.
        
        Args:
            mpai_path: Path to the .mpai file
        
        Returns:
            ProjectHandle for accessing project data
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(mpai_path):
            raise FileNotFoundError(f"MPAI file not found: {mpai_path}")
        
        logger.info(f"Opening MPAI project: {mpai_path}")
        
        # Check if it's a valid ZIP file
        if not zipfile.is_zipfile(mpai_path):
            # Might be legacy binary MPAI format - let caller handle
            raise ValueError(f"Not a ZIP-based MPAI file: {mpai_path}. May be legacy binary format.")
        
        # Open ZIP and read manifest
        zf = zipfile.ZipFile(mpai_path, mode='r')
        
        try:
            # Read manifest
            if self.MANIFEST_FILE not in zf.namelist():
                zf.close()
                raise ValueError(f"Invalid MPAI project: missing {self.MANIFEST_FILE}")
            
            manifest_data = json.loads(zf.read(self.MANIFEST_FILE).decode('utf-8'))
            manifest = ManifestInfo.from_dict(manifest_data)
            
            # Read current metadata version
            metadata_file = self.METADATA_PATTERN.format(manifest.current_metadata_version)
            if metadata_file not in zf.namelist():
                # Fallback: find latest metadata file
                metadata_file = self._find_latest_metadata(zf)
            
            if metadata_file:
                metadata_data = json.loads(zf.read(metadata_file).decode('utf-8'))
                metadata = ProjectMetadata.from_dict(metadata_data)
            else:
                metadata = ProjectMetadata()
            
            # Create handle
            handle = ProjectHandle(
                path=mpai_path,
                manifest=manifest,
                metadata=metadata,
                _zip_file=zf,
            )
            
            self._current_handle = handle
            logger.info(f"Project opened: {manifest.row_count} rows, {manifest.column_count} columns")
            
            return handle
            
        except Exception as e:
            zf.close()
            raise
    
    def _find_latest_metadata(self, zf: zipfile.ZipFile) -> Optional[str]:
        """Find the latest metadata file in the ZIP"""
        metadata_files = [
            name for name in zf.namelist() 
            if name.startswith("metadata_v") and name.endswith(".json")
        ]
        if metadata_files:
            # Sort by version number and return latest
            metadata_files.sort(reverse=True)
            return metadata_files[0]
        return None
    
    # =========================================================================
    # Data Access (Stream-based)
    # =========================================================================
    
    def get_raw_data_stream(self, handle: ProjectHandle) -> BinaryIO:
        """
        Get a stream to the raw data parquet file.
        
        For files < 1GB: Returns BytesIO (in-memory)
        For files >= 1GB: Extracts to temp file and returns file handle
        
        Args:
            handle: Project handle from open_project()
        
        Returns:
            File-like object containing parquet data
        """
        if handle._zip_file is None:
            handle._zip_file = zipfile.ZipFile(handle.path, 'r')
        
        zf = handle._zip_file
        raw_file = handle.manifest.raw_data_file
        
        if raw_file not in zf.namelist():
            raise ValueError(f"Raw data file not found in project: {raw_file}")
        
        # Check file size
        info = zf.getinfo(raw_file)
        file_size = info.file_size
        
        if file_size < self.STREAM_SIZE_LIMIT:
            # Small file: load into BytesIO
            logger.debug(f"Loading raw data into memory: {file_size / 1024 / 1024:.1f} MB")
            data = zf.read(raw_file)
            return io.BytesIO(data)
        else:
            # Large file: extract to temp file
            logger.info(f"Large file detected ({file_size / 1024 / 1024 / 1024:.2f} GB), using temp file")
            
            if handle._temp_dir is None:
                handle._temp_dir = tempfile.mkdtemp(prefix="mpai_")
            
            temp_path = os.path.join(handle._temp_dir, raw_file)
            zf.extract(raw_file, handle._temp_dir)
            
            return open(temp_path, 'rb')
    
    def get_raw_data_polars(self, handle: ProjectHandle) -> pl.DataFrame:
        """
        Read raw data as Polars DataFrame.
        
        Args:
            handle: Project handle
        
        Returns:
            Polars DataFrame with raw data
        """
        stream = self.get_raw_data_stream(handle)
        try:
            return pl.read_parquet(stream)
        finally:
            stream.close()
    
    def get_lod_stream(self, handle: ProjectHandle, lod_name: str) -> Optional[BinaryIO]:
        """
        Get a stream to a LOD parquet file.
        
        Args:
            handle: Project handle
            lod_name: Name of the LOD file (e.g., "lod1_100.parquet")
        
        Returns:
            File-like object or None if not found
        """
        if handle._zip_file is None:
            handle._zip_file = zipfile.ZipFile(handle.path, 'r')
        
        zf = handle._zip_file
        lod_path = f"{self.LOD_DIR}{lod_name}"
        
        if lod_path not in zf.namelist():
            return None
        
        data = zf.read(lod_path)
        return io.BytesIO(data)
    
    def get_calculations_stream(self, handle: ProjectHandle) -> Optional[BinaryIO]:
        """
        Get a stream to the current calculations parquet file.
        
        Args:
            handle: Project handle
        
        Returns:
            File-like object or None if no calculations exist
        """
        if handle.manifest.current_calculations_version == 0:
            return None
        
        if handle._zip_file is None:
            handle._zip_file = zipfile.ZipFile(handle.path, 'r')
        
        zf = handle._zip_file
        calc_file = self.CALCULATIONS_PATTERN.format(handle.manifest.current_calculations_version)
        
        if calc_file not in zf.namelist():
            return None
        
        data = zf.read(calc_file)
        return io.BytesIO(data)
    
    # =========================================================================
    # Project Saving (Append-Only)
    # =========================================================================
    
    def save_project(
        self,
        handle: ProjectHandle,
        metadata: Optional[ProjectMetadata] = None,
        calculations: Optional[Union[pl.DataFrame, bytes]] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Save project changes using append-only strategy.
        
        CRITICAL: Never rewrites the 50GB raw data. Only appends new metadata
        and calculations to the ZIP file.
        
        Args:
            handle: Project handle
            metadata: Updated metadata (None = keep current)
            calculations: New calculations DataFrame or parquet bytes (None = no change)
            progress_callback: Optional callback(message, percentage)
        """
        logger.info(f"Saving project: {handle.path}")
        
        if progress_callback:
            progress_callback("Preparing save...", 0)
        
        # Close current ZIP if open (required for append mode)
        if handle._zip_file:
            handle._zip_file.close()
            handle._zip_file = None
        
        # Prepare new version numbers
        new_metadata_version = handle.manifest.current_metadata_version + 1
        new_calculations_version = handle.manifest.current_calculations_version
        
        if calculations is not None:
            new_calculations_version += 1
        
        # Update manifest
        handle.manifest.modified_at = datetime.now().isoformat()
        handle.manifest.current_metadata_version = new_metadata_version
        handle.manifest.current_calculations_version = new_calculations_version
        
        # Open ZIP in append mode
        with zipfile.ZipFile(
            handle.path,
            mode='a',
            compression=self.COMPRESSION,
            allowZip64=self.ALLOW_ZIP64
        ) as zf:
            
            # 1. Write new metadata version
            if progress_callback:
                progress_callback("Writing metadata...", 30)
            
            if metadata is not None:
                handle.metadata = metadata
            
            metadata_file = self.METADATA_PATTERN.format(new_metadata_version)
            metadata_json = json.dumps(handle.metadata.to_dict(), indent=2)
            zf.writestr(metadata_file, metadata_json)
            
            # 2. Write new calculations version (if provided)
            if calculations is not None:
                if progress_callback:
                    progress_callback("Writing calculations...", 60)
                
                calc_file = self.CALCULATIONS_PATTERN.format(new_calculations_version)
                
                if isinstance(calculations, pl.DataFrame):
                    buffer = io.BytesIO()
                    calculations.write_parquet(buffer)
                    zf.writestr(calc_file, buffer.getvalue())
                elif isinstance(calculations, bytes):
                    zf.writestr(calc_file, calculations)
            
            # 3. Write updated manifest
            if progress_callback:
                progress_callback("Updating manifest...", 90)
            
            manifest_json = json.dumps(handle.manifest.to_dict(), indent=2)
            zf.writestr(self.MANIFEST_FILE, manifest_json)
        
        logger.info(f"Project saved: metadata v{new_metadata_version}, calculations v{new_calculations_version}")
        
        if progress_callback:
            progress_callback("Save complete!", 100)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def is_mpai_project(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file is a valid MPAI project.
        
        Returns:
            (is_valid, format_type) where format_type is:
            - "zip": New ZIP-based format
            - "binary": Legacy C++ binary format
            - "unknown": Not a valid MPAI file
        """
        if not os.path.exists(file_path):
            return False, "unknown"
        
        # Check for ZIP format
        if zipfile.is_zipfile(file_path):
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if self.MANIFEST_FILE in zf.namelist():
                        return True, "zip"
            except:
                pass
        
        # Check for legacy binary format (MPAI magic number)
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'MPAI':
                    return True, "binary"
        except:
            pass
        
        return False, "unknown"
    
    def get_project_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get project information without fully loading the data.
        Useful for file browser preview.
        
        Args:
            file_path: Path to .mpai file
        
        Returns:
            Dict with project info or None
        """
        is_valid, format_type = self.is_mpai_project(file_path)
        
        if not is_valid:
            return None
        
        if format_type == "zip":
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if self.MANIFEST_FILE in zf.namelist():
                        manifest_data = json.loads(zf.read(self.MANIFEST_FILE).decode('utf-8'))
                        return {
                            "format": "zip",
                            "row_count": manifest_data.get("row_count", 0),
                            "column_count": manifest_data.get("column_count", 0),
                            "created_at": manifest_data.get("created_at", ""),
                            "modified_at": manifest_data.get("modified_at", ""),
                            "file_size_bytes": manifest_data.get("file_size_bytes", 0),
                            "has_lod": len(manifest_data.get("lod_files", [])) > 0,
                            "has_calculations": manifest_data.get("current_calculations_version", 0) > 0,
                        }
            except Exception as e:
                logger.warning(f"Failed to read project info: {e}")
        
        elif format_type == "binary":
            # For binary format, return minimal info
            return {
                "format": "binary",
                "file_size_bytes": os.path.getsize(file_path),
            }
        
        return None
    
    def compact_project(self, handle: ProjectHandle, output_path: Optional[str] = None) -> str:
        """
        Compact a project by removing old versions (vacuum operation).
        
        Creates a new clean ZIP with only the latest versions of all files.
        This is useful after many saves to reduce file bloat.
        
        Args:
            handle: Project handle
            output_path: Output path (None = overwrite original)
        
        Returns:
            Path to the compacted file
        """
        if output_path is None:
            output_path = handle.path + ".compact.mpai"
        
        logger.info(f"Compacting project: {handle.path} -> {output_path}")
        
        # Close current ZIP
        if handle._zip_file:
            handle._zip_file.close()
            handle._zip_file = None
        
        # Read current data
        with zipfile.ZipFile(handle.path, 'r') as src_zf:
            # Create new compact ZIP
            with zipfile.ZipFile(
                output_path,
                mode='w',
                compression=self.COMPRESSION,
                allowZip64=self.ALLOW_ZIP64
            ) as dst_zf:
                
                # Copy raw data
                if self.RAW_DATA_FILE in src_zf.namelist():
                    dst_zf.writestr(self.RAW_DATA_FILE, src_zf.read(self.RAW_DATA_FILE))
                
                # Copy LOD files
                for lod_file in handle.manifest.lod_files:
                    if lod_file in src_zf.namelist():
                        dst_zf.writestr(lod_file, src_zf.read(lod_file))
                
                # Write only latest metadata
                metadata_file = self.METADATA_PATTERN.format(1)  # Reset to v001
                dst_zf.writestr(metadata_file, json.dumps(handle.metadata.to_dict(), indent=2))
                
                # Write only latest calculations (if any)
                if handle.manifest.current_calculations_version > 0:
                    src_calc = self.CALCULATIONS_PATTERN.format(handle.manifest.current_calculations_version)
                    if src_calc in src_zf.namelist():
                        dst_zf.writestr(
                            self.CALCULATIONS_PATTERN.format(1),  # Reset to v001
                            src_zf.read(src_calc)
                        )
                
                # Write updated manifest
                new_manifest = ManifestInfo(
                    version=1,
                    format_version=handle.manifest.format_version,
                    created_at=handle.manifest.created_at,
                    modified_at=datetime.now().isoformat(),
                    current_metadata_version=1,
                    current_calculations_version=1 if handle.manifest.current_calculations_version > 0 else 0,
                    raw_data_file=handle.manifest.raw_data_file,
                    lod_files=handle.manifest.lod_files,
                    row_count=handle.manifest.row_count,
                    column_count=handle.manifest.column_count,
                    file_size_bytes=handle.manifest.file_size_bytes,
                )
                dst_zf.writestr(self.MANIFEST_FILE, json.dumps(new_manifest.to_dict(), indent=2))
        
        logger.info(f"Project compacted: {output_path}")
        
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mpai_from_parquet(parquet_path: str, mpai_path: str) -> ProjectHandle:
    """
    Create a new .mpai project from an existing parquet file.
    
    Args:
        parquet_path: Path to source parquet file
        mpai_path: Path for the new .mpai file
    
    Returns:
        ProjectHandle
    """
    manager = MpaiProjectManager()
    return manager.create_project(parquet_path, mpai_path)


def create_mpai_from_dataframe(df: pl.DataFrame, mpai_path: str) -> ProjectHandle:
    """
    Create a new .mpai project from a Polars DataFrame.
    
    Args:
        df: Polars DataFrame with the data
        mpai_path: Path for the new .mpai file
    
    Returns:
        ProjectHandle
    """
    manager = MpaiProjectManager()
    return manager.create_project(df, mpai_path)
