#!/usr/bin/env python3
"""
Unit Tests for MpaiProjectManager

Tests the ZIP64 container implementation including:
- Project creation from DataFrame and parquet files
- Manifest-based versioning
- Append-only saves
- Stream-based data access
- Large file handling

Usage:
    python -m pytest test_code/test_mpai_project_manager.py -v
    
    Or run directly:
    python test_code/test_mpai_project_manager.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import polars as pl
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mpai_project_manager import (
    MpaiProjectManager,
    ManifestInfo,
    ProjectMetadata,
    ProjectHandle,
    create_mpai_from_dataframe,
    create_mpai_from_parquet,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    d = tempfile.mkdtemp(prefix="mpai_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    import numpy as np
    np.random.seed(42)
    
    n_rows = 10000
    return pl.DataFrame({
        "time": pl.Series(range(n_rows)).cast(pl.Float64) / 1000.0,
        "voltage": pl.Series(np.random.normal(230, 10, n_rows)),
        "current": pl.Series(np.random.normal(15, 2, n_rows)),
        "temperature": pl.Series(np.random.normal(25, 5, n_rows)),
    })


@pytest.fixture
def sample_parquet(temp_dir, sample_dataframe):
    """Create a sample parquet file for testing"""
    path = os.path.join(temp_dir, "test_data.parquet")
    sample_dataframe.write_parquet(path)
    return path


@pytest.fixture
def manager():
    """Create a MpaiProjectManager instance"""
    return MpaiProjectManager()


# =============================================================================
# Test: Project Creation
# =============================================================================

class TestProjectCreation:
    
    def test_create_from_dataframe(self, manager, temp_dir, sample_dataframe):
        """Test creating a project from a Polars DataFrame"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        handle = manager.create_project(sample_dataframe, mpai_path)
        
        # Verify file was created
        assert os.path.exists(mpai_path)
        
        # Verify it's a valid ZIP
        assert zipfile.is_zipfile(mpai_path)
        
        # Verify manifest was created
        with zipfile.ZipFile(mpai_path, 'r') as zf:
            assert "manifest.json" in zf.namelist()
            assert "raw_data.parquet" in zf.namelist()
            assert "metadata_v001.json" in zf.namelist()
        
        # Verify handle info
        assert handle.manifest.row_count == len(sample_dataframe)
        assert handle.manifest.column_count == len(sample_dataframe.columns)
        
        print(f"✓ Created project with {handle.manifest.row_count} rows")
    
    def test_create_from_parquet(self, manager, temp_dir, sample_parquet):
        """Test creating a project from an existing parquet file"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        handle = manager.create_project(sample_parquet, mpai_path)
        
        assert os.path.exists(mpai_path)
        assert handle.manifest.file_size_bytes > 0
        
        print(f"✓ Created project from parquet: {handle.manifest.file_size_bytes} bytes")
    
    def test_create_with_lod_files(self, manager, temp_dir, sample_dataframe):
        """Test creating a project with LOD files"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        # Create fake LOD data
        lod_df = sample_dataframe.group_by_dynamic("time", every="0.1s").agg([
            pl.col("voltage").min().alias("voltage_min"),
            pl.col("voltage").max().alias("voltage_max"),
        ])
        
        lod_buffer = io.BytesIO()
        lod_df.write_parquet(lod_buffer)
        lod_files = {"lod1_100.parquet": lod_buffer.getvalue()}
        
        handle = manager.create_project(sample_dataframe, mpai_path, lod_files=lod_files)
        
        # Verify LOD file was included
        assert len(handle.manifest.lod_files) == 1
        
        with zipfile.ZipFile(mpai_path, 'r') as zf:
            assert "lod/lod1_100.parquet" in zf.namelist()
        
        print(f"✓ Created project with {len(handle.manifest.lod_files)} LOD files")
    
    def test_create_with_metadata(self, manager, temp_dir, sample_dataframe):
        """Test creating a project with custom metadata"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        metadata = ProjectMetadata(
            time_column="time",
            sampling_frequency=1000.0,
            preferences={"dark_mode": True},
        )
        
        handle = manager.create_project(sample_dataframe, mpai_path, metadata=metadata)
        
        assert handle.metadata.time_column == "time"
        assert handle.metadata.sampling_frequency == 1000.0
        assert handle.metadata.preferences["dark_mode"] == True
        
        print("✓ Created project with custom metadata")


# =============================================================================
# Test: Project Opening
# =============================================================================

class TestProjectOpening:
    
    def test_open_project(self, manager, temp_dir, sample_dataframe):
        """Test opening an existing project"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        # Create project
        manager.create_project(sample_dataframe, mpai_path)
        
        # Open project
        handle = manager.open_project(mpai_path)
        
        assert handle.manifest.row_count == len(sample_dataframe)
        assert handle.metadata is not None
        
        # Cleanup
        handle.close()
        
        print("✓ Opened project successfully")
    
    def test_open_nonexistent_file(self, manager):
        """Test opening a nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            manager.open_project("/nonexistent/path/test.mpai")
        
        print("✓ FileNotFoundError raised for nonexistent file")
    
    def test_open_invalid_file(self, manager, temp_dir):
        """Test opening an invalid file raises error"""
        # Create a non-ZIP file
        invalid_path = os.path.join(temp_dir, "invalid.mpai")
        with open(invalid_path, 'w') as f:
            f.write("This is not a ZIP file")
        
        with pytest.raises(ValueError):
            manager.open_project(invalid_path)
        
        print("✓ ValueError raised for invalid file format")


# =============================================================================
# Test: Data Access
# =============================================================================

class TestDataAccess:
    
    def test_get_raw_data_stream(self, manager, temp_dir, sample_dataframe):
        """Test getting raw data as a stream"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        handle = manager.open_project(mpai_path)
        stream = manager.get_raw_data_stream(handle)
        
        assert stream is not None
        
        # Read as parquet
        df = pl.read_parquet(stream)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
        
        stream.close()
        handle.close()
        
        print(f"✓ Read {len(df)} rows from stream")
    
    def test_get_raw_data_polars(self, manager, temp_dir, sample_dataframe):
        """Test getting raw data as Polars DataFrame"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        handle = manager.open_project(mpai_path)
        df = manager.get_raw_data_polars(handle)
        
        assert len(df) == len(sample_dataframe)
        
        # Verify data integrity
        assert df["time"][0] == sample_dataframe["time"][0]
        assert df["voltage"].mean() == pytest.approx(sample_dataframe["voltage"].mean(), rel=0.01)
        
        handle.close()
        
        print("✓ Data integrity verified")
    
    def test_get_lod_stream(self, manager, temp_dir, sample_dataframe):
        """Test getting LOD data as a stream"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        # Create with LOD
        lod_buffer = io.BytesIO()
        sample_dataframe.head(100).write_parquet(lod_buffer)
        lod_files = {"lod1_100.parquet": lod_buffer.getvalue()}
        
        manager.create_project(sample_dataframe, mpai_path, lod_files=lod_files)
        
        handle = manager.open_project(mpai_path)
        lod_stream = manager.get_lod_stream(handle, "lod1_100.parquet")
        
        assert lod_stream is not None
        
        lod_df = pl.read_parquet(lod_stream)
        assert len(lod_df) == 100
        
        lod_stream.close()
        handle.close()
        
        print("✓ LOD stream access works")


# =============================================================================
# Test: Append-Only Saving
# =============================================================================

class TestAppendOnlySaving:
    
    def test_save_increments_version(self, manager, temp_dir, sample_dataframe):
        """Test that save increments metadata version"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        handle = manager.open_project(mpai_path)
        
        # Initial version
        assert handle.manifest.current_metadata_version == 1
        
        # Save with updated metadata
        handle.metadata.preferences["test_key"] = "test_value"
        manager.save_project(handle, metadata=handle.metadata)
        
        # Re-open and check version
        handle.close()
        handle = manager.open_project(mpai_path)
        
        assert handle.manifest.current_metadata_version == 2
        assert handle.metadata.preferences.get("test_key") == "test_value"
        
        handle.close()
        
        print("✓ Save correctly incremented version")
    
    def test_save_appends_not_rewrites(self, manager, temp_dir, sample_dataframe):
        """Test that save appends to file, not rewrites"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        
        # Create large-ish project
        large_df = pl.concat([sample_dataframe] * 10)  # 100k rows
        manager.create_project(large_df, mpai_path)
        
        # Get initial file size
        initial_size = os.path.getsize(mpai_path)
        
        # Save with new metadata (should only add a few KB)
        handle = manager.open_project(mpai_path)
        handle.metadata.preferences["test"] = "value"
        manager.save_project(handle, metadata=handle.metadata)
        handle.close()
        
        # Check file size increased only slightly
        new_size = os.path.getsize(mpai_path)
        size_increase = new_size - initial_size
        
        # Should be less than 10KB for metadata only
        assert size_increase < 10 * 1024, f"File grew too much: {size_increase} bytes"
        
        print(f"✓ Append save added only {size_increase} bytes (not rewriting {initial_size} bytes)")
    
    def test_save_with_calculations(self, manager, temp_dir, sample_dataframe):
        """Test saving with calculations DataFrame"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        handle = manager.open_project(mpai_path)
        
        # Create calculations
        calc_df = pl.DataFrame({
            "power": sample_dataframe["voltage"] * sample_dataframe["current"]
        })
        
        manager.save_project(handle, calculations=calc_df)
        handle.close()
        
        # Re-open and check calculations
        handle = manager.open_project(mpai_path)
        
        assert handle.manifest.current_calculations_version == 1
        
        calc_stream = manager.get_calculations_stream(handle)
        assert calc_stream is not None
        
        loaded_calc = pl.read_parquet(calc_stream)
        assert "power" in loaded_calc.columns
        
        calc_stream.close()
        handle.close()
        
        print("✓ Calculations saved and loaded successfully")


# =============================================================================
# Test: Utility Methods
# =============================================================================

class TestUtilityMethods:
    
    def test_is_mpai_project(self, manager, temp_dir, sample_dataframe):
        """Test is_mpai_project detection"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        is_valid, format_type = manager.is_mpai_project(mpai_path)
        
        assert is_valid == True
        assert format_type == "zip"
        
        print("✓ Project format detection works")
    
    def test_get_project_info(self, manager, temp_dir, sample_dataframe):
        """Test getting project info without full load"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        info = manager.get_project_info(mpai_path)
        
        assert info is not None
        assert info["format"] == "zip"
        assert info["row_count"] == len(sample_dataframe)
        assert info["column_count"] == len(sample_dataframe.columns)
        
        print(f"✓ Got project info: {info['row_count']} rows, {info['column_count']} columns")
    
    def test_compact_project(self, manager, temp_dir, sample_dataframe):
        """Test project compaction (vacuum)"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        handle = manager.open_project(mpai_path)
        
        # Do multiple saves to create bloat
        for i in range(5):
            handle.metadata.preferences[f"key_{i}"] = f"value_{i}"
            manager.save_project(handle, metadata=handle.metadata)
            handle = manager.open_project(mpai_path)
        
        # Check for multiple metadata versions
        with zipfile.ZipFile(mpai_path, 'r') as zf:
            metadata_files = [n for n in zf.namelist() if n.startswith("metadata_v")]
            assert len(metadata_files) == 6  # v001 + 5 saves
        
        # Compact
        compact_path = manager.compact_project(handle)
        handle.close()
        
        # Verify compacted file has only one version
        with zipfile.ZipFile(compact_path, 'r') as zf:
            metadata_files = [n for n in zf.namelist() if n.startswith("metadata_v")]
            assert len(metadata_files) == 1
        
        # Verify data is still accessible
        handle = manager.open_project(compact_path)
        df = manager.get_raw_data_polars(handle)
        assert len(df) == len(sample_dataframe)
        
        # Verify all preferences were preserved
        assert handle.metadata.preferences.get("key_4") == "value_4"
        
        handle.close()
        
        print("✓ Project compaction works")


# =============================================================================
# Test: ZIP64 Compliance
# =============================================================================

class TestZIP64Compliance:
    
    def test_zip_stored_mode(self, manager, temp_dir, sample_dataframe):
        """Test that ZIP uses STORED mode (no compression)"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        with zipfile.ZipFile(mpai_path, 'r') as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, \
                    f"File {info.filename} uses compression type {info.compress_type}"
        
        print("✓ All files use ZIP_STORED mode")
    
    def test_manifest_structure(self, manager, temp_dir, sample_dataframe):
        """Test manifest JSON structure"""
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        manager.create_project(sample_dataframe, mpai_path)
        
        with zipfile.ZipFile(mpai_path, 'r') as zf:
            manifest = json.loads(zf.read("manifest.json").decode('utf-8'))
        
        # Required fields
        assert "version" in manifest
        assert "format_version" in manifest
        assert "created_at" in manifest
        assert "modified_at" in manifest
        assert "current_metadata_version" in manifest
        assert "raw_data_file" in manifest
        assert "row_count" in manifest
        assert "column_count" in manifest
        
        print("✓ Manifest structure is correct")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MPAI PROJECT MANAGER TESTS")
    print("=" * 80)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="mpai_test_")
    
    try:
        # Create sample data
        import numpy as np
        np.random.seed(42)
        n_rows = 10000
        sample_df = pl.DataFrame({
            "time": pl.Series(range(n_rows)).cast(pl.Float64) / 1000.0,
            "voltage": pl.Series(np.random.normal(230, 10, n_rows)),
            "current": pl.Series(np.random.normal(15, 2, n_rows)),
            "temperature": pl.Series(np.random.normal(25, 5, n_rows)),
        })
        
        manager = MpaiProjectManager()
        
        print("\n[1] Testing Project Creation...")
        mpai_path = os.path.join(temp_dir, "test_project.mpai")
        handle = manager.create_project(sample_df, mpai_path)
        print(f"    ✓ Created: {mpai_path}")
        print(f"    ✓ Rows: {handle.manifest.row_count}")
        print(f"    ✓ Columns: {handle.manifest.column_count}")
        
        print("\n[2] Testing Project Opening...")
        handle = manager.open_project(mpai_path)
        print(f"    ✓ Opened successfully")
        print(f"    ✓ Manifest version: {handle.manifest.version}")
        
        print("\n[3] Testing Data Access...")
        df = manager.get_raw_data_polars(handle)
        print(f"    ✓ Read {len(df)} rows")
        print(f"    ✓ Columns: {df.columns}")
        
        print("\n[4] Testing Append-Only Save...")
        initial_size = os.path.getsize(mpai_path)
        handle.metadata.preferences["test"] = "value"
        manager.save_project(handle, metadata=handle.metadata)
        new_size = os.path.getsize(mpai_path)
        print(f"    ✓ Initial size: {initial_size:,} bytes")
        print(f"    ✓ After save: {new_size:,} bytes")
        print(f"    ✓ Increase: {new_size - initial_size:,} bytes (append-only)")
        
        handle.close()
        
        print("\n[5] Testing Project Info...")
        info = manager.get_project_info(mpai_path)
        print(f"    ✓ Format: {info['format']}")
        print(f"    ✓ Row count: {info['row_count']}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
