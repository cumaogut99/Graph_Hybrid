"""
MPAI Data Reader - Zero-Copy Directory Reader
---------------------------------------------
Reads the MPAI Directory Format using OS-level Memory Mapping (mmap).
Designed for Dewesoft-like performance on huge datasets (50GB+).
"""

import os
import mmap
import struct
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Arrow Bridge: Zero-copy data sharing between Python and C++
try:
    import pyarrow as pa
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

logger = logging.getLogger(__name__)

class MpaiDirectoryReader:
    """
    High-Performance Reader for MPAI Directory Format.
    
    Key Features:
    - Zero-Copy: Uses numpy.memmap to map files directly into virtual memory.
    - Dual-Stream Access: Automatically switches between .red and .bin based on zoom level.
    - Lazy Loading: Opens file handles but reads bytes only on demand.
    - Compatible with SignalProcessor API.
    """
    
    def __init__(self, mpai_dir_path: str):
        self.mpai_path = Path(mpai_dir_path)
        if not self.mpai_path.exists():
            raise FileNotFoundError(f"MPAI Directory not found: {self.mpai_path}")
            
        self.setup_path = self.mpai_path / "setup.xml"
        if not self.setup_path.exists():
            raise FileNotFoundError("Invalid MPAI: setup.xml missing.")

        # Metadata Storage
        self.channels: Dict[int, Dict] = {}
        # Map name to ID for name-based lookup
        self.name_to_id: Dict[str, int] = {}
        
        self.sample_rate = 0.0
        self.dt = 0.0
        self.t0 = 0.0
        
        # Load Metadata
        self._parse_metadata()
        
        # Open Memory Maps
        self._open_streams()

    # =========================================================================
    # SignalProcessor Compatibility API
    # =========================================================================
    
    def get_header(self) -> Dict:
        """Dummy method to satisfy is_mpai check."""
        return {"type": "directory_mpai"}
        
    def get_column_names(self) -> List[str]:
        """
        Returns list of PHYSICAL channel names only (NO synthetic time).
        
        This prevents filter errors when filtering on non-existent 'time' column.
        Use get_all_column_names_with_time() for UI display that includes time.
        """
        names = []
        # Sort by ID to ensure consistent order
        for ch_id in sorted(self.channels.keys()):
            names.append(self.channels[ch_id]["name"])
        return names
    
    def get_all_column_names_with_time(self) -> List[str]:
        """
        Returns column names INCLUDING 'Time' (for UI parameter display).
        
        If no physical time column exists, synthetic 'Time' is prepended.
        """
        names = self.get_column_names()
        if not self.has_physical_time_column():
            names.insert(0, "Time")  # Virtual time column for display
        return names
        
    def get_column_count(self) -> int:
        return len(self.channels)  # Physical columns only
        
    def get_row_count(self) -> int:
        if not self.channels:
            return 0
        # Return max row count
        return max(ch["sample_count"] for ch in self.channels.values())
        
    def load_column_slice(self, col_name: str, start: int, length: int) -> Union[List[float], np.ndarray]:
        """
        Loads a slice of data for a specific column.
        
        Args:
            col_name: Name of the column ('time' or channel name)
            start: Start index
            length: Number of samples to load
            
        Returns:
            List or Array of values.
        """
        if length <= 0:
            return []
            
        row_count = self.get_row_count()
        start = max(0, start)
        end = min(row_count, start + length)
        actual_len = end - start
        
        if actual_len <= 0:
            return []

        # Case 1: Check if it's a known physical channel FIRST (Priority to real data)
        # This handles user-selected "Time" column that is actually recorded in the file
        if col_name in self.name_to_id:
            pass # Fall through to Case 2 (Data Channel Logic)
            
        elif col_name.lower() == "time":
            # Synthetic Time Generation (Only if no physical "Time" channel exists)
            # Generate time vector efficiently
            # t = t0 + (start + i) * dt
            # Using np.linspace is safer for floating point
            t_start = self.t0 + start * self.dt
            t_end = self.t0 + (end - 1) * self.dt
            # Generate
            return np.linspace(t_start, t_end, actual_len)
            
        # Case 2: Data Channel
        if col_name not in self.name_to_id:
            # Try case insensitive lookup to find physical channel
            found = False
            for name, ch_id in self.name_to_id.items():
                if name.lower() == col_name.lower():
                    col_name = name
                    found = True
                    break
            
            # Use found name for ID lookup
            if not found:
                 if col_name.lower() != "time": # Avoid double warning if it was "time" check above
                     logger.warning(f"Column '{col_name}' not found in MPAI")
                 return np.zeros(actual_len)
                
        if col_name in self.name_to_id:
            ch_id = self.name_to_id[col_name]
            ch = self.channels[ch_id]
            
            # Read from mmap (Raw)
            # Verify range
            if start >= ch["sample_count"]:
                return []
                
            c_end = min(ch["sample_count"], end)
            
            # Return slice (returns numpy array view)
            return ch["mmap_bin"][start:c_end]
            
        return np.zeros(actual_len)

        
    def _parse_metadata(self):
        """Parses setup.xml to populate channel info."""
        tree = ET.parse(self.setup_path)
        root = tree.getroot()
        
        # Global Info
        info_node = root.find("Info")
        self.sample_rate = float(info_node.find("SampleRate").text)
        self.dt = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        
        # Channels
        for ch_node in root.find("Channels").findall("Channel"):
            ch_id = int(ch_node.get("id"))
            
            # Determine Numpy Dtype
            dtype_str = ch_node.find("DataType").text
            dtype = np.float64 if dtype_str == "Float64" else np.float32
            bytes_per_sample = int(ch_node.find("BytesPerSample").text)
            
            self.channels[ch_id] = {
                "name": ch_node.find("Name").text,
                "unit": ch_node.find("Unit").text,
                "dtype": dtype,
                "bytes_per_sample": bytes_per_sample,
                "bin_file": ch_node.find("BinFile").text,
                "red_file": ch_node.find("RedFile").text,
                # Store full paths for easier access
                "bin_path": self.mpai_path / ch_node.find("BinFile").text,
                "red_path": self.mpai_path / ch_node.find("RedFile").text,
                # Runtime handles (populated later)
                "mmap_bin": None,
                "mmap_red": None,
                "sample_count": 0
            }
            # Populate name map
            self.name_to_id[self.channels[ch_id]["name"]] = ch_id
            
    def _open_streams(self):
        """
        Opens memory maps for all channels.
        CRITICAL: This does NOT load data into RAM. It just maps virtual address space.
        """
        for ch_id, ch_info in self.channels.items():
            # 1. Map Raw Binary File
            bin_path = ch_info["bin_path"]
            if bin_path.exists():
                # Calculate sample count from file size
                file_size = bin_path.stat().st_size
                n_samples = file_size // ch_info["bytes_per_sample"]
                ch_info["sample_count"] = n_samples
                
                # Create memory map (Read-Only, Copy-On-Write is default for 'r' usually, here 'r' means read-only)
                # We use 'r' mode to ensure we don't accidentally write
                ch_info["mmap_bin"] = np.memmap(
                    bin_path, 
                    dtype=ch_info["dtype"], 
                    mode='r', 
                    shape=(n_samples,)
                )
            else:
                logger.error(f"Bin file missing for Ch {ch_id}: {bin_path}")
                
            # 2. Map Reduced File
            red_path = ch_info["red_path"]
            if red_path.exists():
                file_size_red = red_path.stat().st_size
                # Each struct is 4 doubles (32 bytes)
                n_blocks = file_size_red // 32
                
                # Map as (N, 4) array of float64
                ch_info["mmap_red"] = np.memmap(
                    red_path,
                    dtype=np.float64,
                    mode='r',
                    shape=(n_blocks, 4) # Columns: Min, Max, Avg, RMS
                )
                
    def get_time_range(self) -> Tuple[float, float]:
        """Returns total (start, end) time of the recording."""
        if not self.channels:
            return 0.0, 0.0
        
        # Get max sample count across all channels
        max_samples = max(c["sample_count"] for c in self.channels.values())
        duration = max_samples * self.dt
        return self.t0, self.t0 + duration

    # =========================================================================
    # Arrow Bridge API - Zero-Copy Data Sharing
    # =========================================================================
    
    def get_arrow_array(self, column_name: str):
        """
        Get column data as Arrow array (zero-copy from mmap).
        
        Args:
            column_name: Name of the column to retrieve
            
        Returns:
            pyarrow.Array wrapping the mmap buffer (no copy!)
            
        Raises:
            ImportError: If pyarrow is not available
            KeyError: If column not found
        """
        if not ARROW_AVAILABLE:
            raise ImportError("pyarrow is required for Arrow bridge. Install with: pip install pyarrow")
        
        # Handle synthetic time column
        col_lower = column_name.lower()
        if col_lower == "time" and column_name not in self.name_to_id:
            # Generate synthetic time array
            row_count = self.get_row_count()
            t_start, t_end = self.get_time_range()
            time_data = np.linspace(t_start, t_end, row_count, dtype=np.float64)
            # Note: This creates a copy since we're generating data
            return pa.array(time_data)
        
        # Look up physical column
        if column_name not in self.name_to_id:
            # Try case-insensitive lookup
            for name in self.name_to_id.keys():
                if name.lower() == col_lower:
                    column_name = name
                    break
            else:
                raise KeyError(f"Column '{column_name}' not found in MPAI. Available: {list(self.name_to_id.keys())}")
        
        ch_id = self.name_to_id[column_name]
        mmap_data = self.channels[ch_id]["mmap_bin"]
        
        if mmap_data is None:
            raise ValueError(f"Channel '{column_name}' has no mmap data")
        
        # Zero-copy: Arrow wraps the mmap buffer directly
        # The mmap is already a numpy array, Arrow can use its buffer
        return pa.array(mmap_data, from_pandas=False)
    
    def get_arrow_batch(self, column_names: List[str]):
        """
        Get multiple columns as Arrow RecordBatch (zero-copy).
        
        Args:
            column_names: List of column names to include
            
        Returns:
            pyarrow.RecordBatch with requested columns
        """
        if not ARROW_AVAILABLE:
            raise ImportError("pyarrow is required for Arrow bridge")
        
        arrays = []
        fields = []
        
        for name in column_names:
            try:
                arr = self.get_arrow_array(name)
                arrays.append(arr)
                fields.append(pa.field(name, arr.type))
            except KeyError as e:
                logger.warning(f"Skipping column in batch: {e}")
                continue
        
        if not arrays:
            raise ValueError("No valid columns found for Arrow batch")
        
        schema = pa.schema(fields)
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    
    def get_real_column_names(self) -> List[str]:
        """
        Returns only REAL/physical column names (excludes synthetic time).
        Use this for operations that need actual data columns.
        """
        return [self.channels[ch_id]["name"] for ch_id in sorted(self.channels.keys())]
    
    def has_physical_time_column(self) -> bool:
        """
        Check if a physical time column exists in the data.
        
        Returns:
            True if a column with 'time' in its name exists
        """
        for name in self.get_real_column_names():
            if 'time' in name.lower() or 'zaman' in name.lower():
                return True
        return False
    
    def get_time_column_name(self) -> Optional[str]:
        """
        Get the name of the time column (physical or synthetic).
        
        Returns:
            Column name if physical time exists, 'Time' for synthetic, None if unclear
        """
        # First check for physical time column
        for name in self.get_real_column_names():
            if 'time' in name.lower() or 'zaman' in name.lower():
                return name
        
        # Return synthetic time indicator
        return "Time"  # Virtual/generated time


    def get_render_data(self, ch_id: int, t_min: float, t_max: float, width_pixels: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        High-Speed Vectorized Downsampling for Visualization.
        
        Args:
            ch_id: Channel ID to render
            t_min, t_max: Time range to render
            width_pixels: Target number of visual points (e.g. screen width)
            
        Returns:
            (x_data, y_data, load_count)
        """
        # 1. Calculate Indices
        start_idx = int((t_min - self.t0) / self.dt)
        end_idx = int((t_max - self.t0) / self.dt)
        
        # Clamp
        row_count = self.get_row_count()
        start_idx = max(0, min(row_count, start_idx))
        end_idx = max(0, min(row_count, end_idx))
        
        if end_idx <= start_idx:
            return np.array([]), np.array([]), 0
            
        count = end_idx - start_idx
        ch_info = self.channels.get(ch_id)
        
        if not ch_info or ch_info["mmap_bin"] is None:
            return np.array([]), np.array([]), 0

        # Optimization: Use RED file if available and appropriate
        # Assume RED file is 100x downsampled (or check ratio)
        # For now, stick to robust BIN processing as its vectorized memory map is extremely fast
        
        # 2. Vectorized Downsampling on BIN file
        target_points = width_pixels * 2 
        
        if count <= target_points * 2:
             # Small enough, return raw
             t_start = self.t0 + start_idx * self.dt
             t_end = self.t0 + (end_idx - 1) * self.dt  # Precise end
             x_out = np.linspace(t_start, t_end, count)
             y_raw = ch_info["mmap_bin"][start_idx:end_idx]
             return x_out, y_raw, count
        
        # Bucketize
        y_raw = ch_info["mmap_bin"][start_idx:end_idx]
        bucket_size = count // width_pixels
        n_processable = width_pixels * bucket_size
        y_processable = y_raw[:n_processable]
        
        # Reshape & MinMax
        y_reshaped = y_processable.reshape(width_pixels, bucket_size)
        min_vals = np.min(y_reshaped, axis=1)
        max_vals = np.max(y_reshaped, axis=1)
        y_out = np.column_stack((min_vals, max_vals)).flatten()
        
        # Generate X
        # Correctly calculate duration based on count and dt
        t_start = self.t0 + start_idx * self.dt
        current_duration = count * self.dt # Use explicit name to avoid conflict
        x_out = np.linspace(t_start, t_start + current_duration, len(y_out))
        
        return x_out, y_out, count

    def close(self):
        """Closes file handles (if necessary). Numpy memmap handles closing automatically usually."""
        # Explicit closing isn't strictly needed for memmap in read mode, 
        # but good practice if we want to release file locks on Windows.
        # However, np.memmap doesn't have a close() method. 
        # We can delete the reference.
        for ch in self.channels.values():
            if ch["mmap_bin"] is not None:
                del ch["mmap_bin"]
                ch["mmap_bin"] = None
            if ch["mmap_red"] is not None:
                del ch["mmap_red"]
                ch["mmap_red"] = None

    # =========================================================================
    # High-Performance Data Access
    # =========================================================================

    def get_render_data(self, 
                       ch_id: int, 
                       t_start: float, 
                       t_end: float, 
                       pixel_width: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Optimized data fetcher for UI Rendering.
        Decides intelligently between Raw and Reduced streams.
        
        Returns:
            (time_array, value_array, source_type)
            source_type is "raw" or "reduced"
        """
        if ch_id not in self.channels:
            raise ValueError(f"Channel {ch_id} not found.")
            
        ch = self.channels[ch_id]
        
        # 1. Convert Time to Indices
        idx_start = int((t_start - self.t0) / self.dt)
        idx_end = int((t_end - self.t0) / self.dt)
        
        # Clamp
        idx_start = max(0, idx_start)
        idx_end = min(ch["sample_count"], idx_end)
        
        if idx_start >= idx_end:
            return np.array([]), np.array([]), "empty"
            
        n_samples_requested = idx_end - idx_start
        
        # 2. Decision Logic
        # If we have more samples than pixels by a factor of X, use Reduced.
        # Each reduced block is 500 samples.
        # If pixels * 500 < samples, it means even reduced might be too detailed?
        # Actually, standard logic: If "Samples per Pixel" > BlockSize (500), use Reduced.
        
        samples_per_pixel = n_samples_requested / pixel_width if pixel_width > 0 else 999999
        
        USE_REDUCED_THRESHOLD = 250 # If > 250 samples fit in 1 pixel, use reduced
        # Or strictly if we can simply map 1 Reduced Block to < 2 Pixels.
        
        if samples_per_pixel > USE_REDUCED_THRESHOLD and ch["mmap_red"] is not None:
            return self._get_reduced_window(ch, idx_start, idx_end)
        else:
            return self._get_raw_window(ch, idx_start, idx_end)

    def _get_raw_window(self, ch: Dict, idx_start: int, idx_end: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Reads from Raw Binary Stream via mmap."""
        # Slicing a memmap returns a new memmap (view), which is instant.
        # Converting to array triggers the read.
        data_view = ch["mmap_bin"][idx_start:idx_end]
        
        # Create Time Vector on-the-fly (Linear generation is fast)
        # t = t0 + (idx_start + i) * dt
        # We can use np.linspace or arange
        t_view = np.linspace(
            self.t0 + idx_start * self.dt,
            self.t0 + (idx_end - 1) * self.dt,
            len(data_view)
        )
        
        return t_view, data_view, "raw"

    def _get_reduced_window(self, ch: Dict, idx_start: int, idx_end: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Reads from Reduced Stream (Min/Max)."""
        # Block size constraint
        BLOCK_SIZE = 500
        
        # Convert sample indices to block indices
        blk_start = idx_start // BLOCK_SIZE
        blk_end = (idx_end + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Clamp block indices
        n_blocks_total = ch["mmap_red"].shape[0]
        blk_end = min(n_blocks_total, blk_end)
        
        if blk_start >= blk_end:
            return np.array([]), np.array([]), "empty"
            
        # Read blocks: Shape (M, 4) -> [Min, Max, Avg, RMS]
        blocks = ch["mmap_red"][blk_start:blk_end]
        
        # For visualization (Min/Max Envelope), we need Min and Max interleaved
        # to draw the vertical lines correctly.
        # Construct specific geometry: T_i -> Min, T_i -> Max
        
        mins = blocks[:, 0]
        maxs = blocks[:, 1]
        
        # Create interleaved array for "Range Plot"
        # [Min0, Max0, Min1, Max1, ...]
        values = np.empty(mins.size + maxs.size, dtype=mins.dtype)
        values[0::2] = mins
        values[1::2] = maxs
        
        # Time needs to be interleaved too, or simple step
        # T_block_center = t0 + (blk_idx * BlockSize + BlockSize/2) * dt
        # Actually simpler: T_start_block, T_end_block?
        # Usually for Min/Max graph, we put Min and Max at the SAME time point (visual artifact),
        # OR slightly offset. DeweSoft often draws a vertical line at T_center.
        
        # Generate block times
        block_indices = np.arange(blk_start, blk_end)
        block_times_center = self.t0 + (block_indices * BLOCK_SIZE + BLOCK_SIZE/2) * self.dt
        
        # Interleave times: [T0, T0, T1, T1...]
        times = np.repeat(block_times_center, 2)
        
        return times, values, "reduced"

    def get_cursor_value(self, ch_id: int, t: float) -> float:
        """
        Gets a single scalar value at time t. 
        Uses O(1) random access on raw file.
        """
        if ch_id not in self.channels:
            return 0.0
            
        ch = self.channels[ch_id]
        idx = int((t - self.t0) / self.dt)
        
        # Boundary check
        if 0 <= idx < ch["sample_count"]:
            return float(ch["mmap_bin"][idx])
        return 0.0

    def get_statistics_snapshot(self, ch_id: int, t_start: float, t_end: float) -> Dict[str, float]:
        """
        Calculates statistics for a specific window.
        - If window is small (Raw mode): Calculates from raw data.
        - If window is large (Reduced mode): Calculates from Reduced stats (Approximate but fast).
        
        Returns:
            {min, max, avg, rms, std}
        """
        # For now, let's just use get_render_data logic to decide source, 
        # but here we need accurate stats so maybe always pref Raw?
        # No, for 50GB file, calculating Avg on 50GB raw is slow.
        # We MUST use reduced stream for global stats.
        
        # Let's trust the Dual logic, but maybe be more aggressive on using Reduced.
        
        # Temporary: Just fetch raw if small (< 1M samples), else Reduced
        ch = self.channels[ch_id]
        idx_start = max(0, int((t_start - self.t0) / self.dt))
        idx_end = min(ch["sample_count"], int((t_end - self.t0) / self.dt))
        count = idx_end - idx_start
        
        if count <= 0:
            return {"min":0, "max":0, "avg":0, "rms":0, "std":0}
            
        if count < 1_000_000: # 1M samples = 8MB -> Safe for Raw Calc
            raw_data = ch["mmap_bin"][idx_start:idx_end]
            _min = float(np.min(raw_data))
            _max = float(np.max(raw_data))
            _avg = float(np.mean(raw_data))
            _rms = float(np.sqrt(np.mean(raw_data**2)))
            _std = float(np.std(raw_data))
            
            # Calculate Duty Cycle (using Mean as threshold)
            # Count samples > mean
            # Ideally this should be time-based crossing, but sample count is good approx for uniform sampling
            above_threshold_count = np.count_nonzero(raw_data > _avg)
            _duty_cycle = (above_threshold_count / count) * 100.0 if count > 0 else 50.0
            
            return {
                "min": _min, "max": _max, "avg": _avg, "rms": _rms, "std": _std, 
                "duty_cycle": _duty_cycle
            }
            
        else:
            # Aggregate from Reduced Stream
            # Logic: Weighted average of blocks?
            # Ideally verify alignment.
            
            BLOCK_SIZE = 500
            blk_start = idx_start // BLOCK_SIZE
            blk_end = idx_end // BLOCK_SIZE
            
            if blk_start >= blk_end: # Window smaller than 1 block but > 1M samples? Impossible.
                return {"min":0, "max":0, "avg":0, "rms":0, "std":0}
                
            blocks = ch["mmap_red"][blk_start:blk_end]
            
            # Global Min/Max
            _min = float(np.min(blocks[:, 0]))
            _max = float(np.max(blocks[:, 1]))
            
            # Global Avg = Mean of Avgs
            _avg = float(np.mean(blocks[:, 2]))
            
            # Global RMS = Sqrt(Mean(RMS^2))
            # Because RMS of block is sqrt(mean(x^2))
            # Square it back to get mean(x^2) of block
            # Then mean of those means
            ms_blocks = blocks[:, 3] ** 2
            global_ms = np.mean(ms_blocks)
            _rms = float(np.sqrt(global_ms))
            
            # Global Std = Sqrt(RMS^2 - Avg^2)
            # Standard statistical identity
            try:
                var = max(0.0, _rms**2 - _avg**2)
                _std = float(np.sqrt(var))
            except:
                _std = 0.0
                
            return {"min": _min, "max": _max, "avg": _avg, "rms": _rms, "std": _std}

