"""
MPAI Data Engine - Directory-Based Dual-Stream Storage
------------------------------------------------------
Implements the specific directory-based format required for MachinePulseAI.
Focuses on High-Performance I/O using Memory Mapping and Raw Binary streams.
"""

import os
import shutil
import logging
import struct
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DataEngine:
    """
    Core engine for managing MPAI Directory-Based Data Storage.
    
    Architecture:
        Record_Name.mpai/          (Directory)
            ├── setup.xml          (Metadata)
            ├── channel_0.bin      (Raw Data Stream - Float64/32)
            ├── channel_0.red      (Reduced Data Stream - Min/Max/Avg/RMS)
            └── ...
            
    Key Features:
    - OS-Level Memory Mapping (mmap) for Zero-Copy access.
    - Explicit Dual-Stream recording (Raw + Reduced).
    - No Compression (Speed > Storage).
    """
    
    REDUCED_BLOCK_SIZE = 500  # Number of samples per reduced block
    
    # Struct format for Reduced Data: 4 x double (min, max, avg, rms)
    # < = Little Endian, d = double (8 bytes)
    REDUCED_STRUCT_FMT = "<dddd" 
    REDUCED_STRUCT_SIZE = struct.calcsize(REDUCED_STRUCT_FMT) # 32 bytes

    def __init__(self, base_dir: str):
        """
        Initialize the Data Engine.
        
        Args:
            base_dir: Base directory where records will be stored/created.
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
    def import_numpy_data(self, 
                          data: np.ndarray, 
                          sample_rate: float, 
                          record_name: str,
                          channel_names: Optional[List[str]] = None,
                          units: Optional[List[str]] = None,
                          chunk_size_mb: int = 50) -> str:
        """
        High-Performance Import: Numpy Array -> MPAI Directory Structure.
        
        Args:
            data: Numpy array (Shape: [N_Samples, N_Channels] or [N_Samples])
            sample_rate: Sampling frequency in Hz
            record_name: Name of the record (folder name will be {record_name}.mpai)
            channel_names: List of channel names.
            units: List of units correspoding to channels.
            chunk_size_mb: Chunk size for writing (in MB) to optimize I/O.
        
        Returns:
            Absolute path to the created .mpai directory.
        """
        # 1. Normalize Input Data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_channels = data.shape
        dtype = data.dtype
        
        # Ensure float32 or float64
        if dtype not in [np.float32, np.float64]:
            logger.info(f"Converting data from {dtype} to float64 for MPAI storage.")
            data = data.astype(np.float64)
            dtype = np.float64

        # Defaults for Metadata
        if channel_names is None:
            channel_names = [f"Channel_{i}" for i in range(n_channels)]
        if units is None:
            units = ["V" for _ in range(n_channels)]
            
        if len(channel_names) != n_channels:
            raise ValueError("Length of channel_names must match number of columns in data.")
            
        # 2. Create Directory Structure
        record_dir_name = f"{record_name}.mpai"
        mpai_dir = self.base_dir / record_dir_name
        
        if mpai_dir.exists():
            logger.warning(f"Overwriting existing MPAI record: {mpai_dir}")
            shutil.rmtree(mpai_dir)
        
        mpai_dir.mkdir()
        logger.info(f"Created MPAI directory: {mpai_dir}")
        
        # 3. Write Metadata (setup.xml)
        self._write_setup_xml(mpai_dir, channel_names, sample_rate, dtype, units)
        
        # 4. Write Data Streams (Dual-Stream: Raw & Reduced)
        # We process channel by channel to minimize file handle switching and maximize throughput
        
        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]
            raw_filename = mpai_dir / f"channel_{ch_idx}.bin"
            red_filename = mpai_dir / f"channel_{ch_idx}.red"
            
            # Extract channel data
            ch_data = data[:, ch_idx]
            
            logger.info(f"Processing Channel {ch_idx}: {ch_name} | {n_samples} samples")
            
            # Write Raw Stream
            self._write_raw_stream(raw_filename, ch_data, chunk_size_mb)
            
            # Write Reduced Stream
            self._write_reduced_stream(red_filename, ch_data)
            
        logger.info("Import completed successfully.")
        return str(mpai_dir)

    def _write_setup_xml(self, 
                         mpai_dir: Path, 
                         channel_names: List[str], 
                         sample_rate: float, 
                         dtype: np.dtype,
                         units: List[str]):
        """Creates the setup.xml metadata file."""
        root = ET.Element("Setup")
        
        # General Info
        info_elem = ET.SubElement(root, "Info")
        ET.SubElement(info_elem, "SampleRate").text = str(sample_rate)
        ET.SubElement(info_elem, "Created").text = datetime.now().isoformat()
        
        # Channels
        channels_elem = ET.SubElement(root, "Channels")
        
        bytes_per_sample = 8 if dtype == np.float64 else 4
        
        for i, name in enumerate(channel_names):
            ch_elem = ET.SubElement(channels_elem, "Channel", id=str(i))
            ET.SubElement(ch_elem, "Name").text = name
            ET.SubElement(ch_elem, "Unit").text = units[i] if i < len(units) else "-"
            ET.SubElement(ch_elem, "DataType").text = "Float64" if dtype == np.float64 else "Float32"
            ET.SubElement(ch_elem, "BytesPerSample").text = str(bytes_per_sample)
            ET.SubElement(ch_elem, "BinFile").text = f"channel_{i}.bin"
            ET.SubElement(ch_elem, "RedFile").text = f"channel_{i}.red"
            
            # Implicit Timing Formula Constants
            ET.SubElement(ch_elem, "T0").text = "0.0" 
            ET.SubElement(ch_elem, "Dt").text = str(1.0 / sample_rate)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(mpai_dir / "setup.xml", encoding="utf-8", xml_declaration=True)

    def _write_raw_stream(self, filepath: Path, data: np.ndarray, chunk_size_mb: int):
        """
        Writes raw binary data efficiently.
        Uses mmap for larger files or direct binary write.
        """
        # For simplicity and robustness during import, we use direct binary write with large chunks.
        # mmap is excellent for reading or random access updates, but for sequential write,
        # standard buffered IO is usually sufficient and simpler to manage sequentially.
        
        # However, the spec mentions: "Import phase... Write raw.bin using memory mapping OR direct binary write."
        # Using Direct Binary Write (tofile) is extremely fast for sequential Numpy arrays.
        
        # Optimization: If array is contiguous, tofile is C-speed.
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        with open(filepath, "wb") as f:
            data.tofile(f)
            
    def _write_reduced_stream(self, filepath: Path, data: np.ndarray):
        """
        Calculates and writes the Reduced Data Stream (Min/Max/Avg/RMS).
        Uses Vectorized NumPY operations for speed (No Python Loops per sample).
        """
        n_samples = len(data)
        block_size = self.REDUCED_BLOCK_SIZE
        
        # Truncate to nearest block multiple
        n_blocks = n_samples // block_size
        trunc_length = n_blocks * block_size
        
        if n_blocks > 0:
            # Reshape to (n_blocks, block_size) for vectorized calc
            reshaped = data[:trunc_length].reshape(n_blocks, block_size)
            
            # 1. Calculate Statistics Vectorized
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
            avgs = reshaped.mean(axis=1)
            
            # RMS
            squares = reshaped.astype(np.float64) ** 2
            means_sq = squares.mean(axis=1)
            rmss = np.sqrt(means_sq)
            
            # 2. Interleave
            reduced_matrix = np.column_stack((mins, maxs, avgs, rmss)).astype(np.float64)
            flat_reduced = reduced_matrix.flatten()
            
            # Write to file
            with open(filepath, "wb") as f:
                flat_reduced.tofile(f)

        # Handle remainder (if any, typically ignored in high-speed reduced views or appended)
        remainder = data[trunc_length:]
        if len(remainder) > 0:
            r_min = float(np.min(remainder))
            r_max = float(np.max(remainder))
            r_avg = float(np.mean(remainder))
            r_rms = float(np.sqrt(np.mean(remainder.astype(np.float64)**2)))
            
            # Pack single struct
            packed = struct.pack(self.REDUCED_STRUCT_FMT, r_min, r_max, r_avg, r_rms)
            with open(filepath, "ab") as f:
                f.write(packed)


class MpaiStreamWriter:
    """
    Stateful writer for streaming data into MPAI directory format.
    Handles buffering for Reduced Data stream generation.
    """
    
    def __init__(self, output_path: str):
        """
        Args:
            output_path: Full path to the .mpai directory to be created.
        """
        self.output_path = Path(output_path)
        self.bin_files = {} # ch_idx -> file_handle
        self.red_files = {} # ch_idx -> file_handle
        self.red_buffers = {} # ch_idx -> list of values (buffer)
        
        self.channel_names = []
        self.sample_rate = 1.0
        
        # Engine constants
        self.REDUCED_BLOCK_SIZE = DataEngine.REDUCED_BLOCK_SIZE
        self.REDUCED_STRUCT_FMT = DataEngine.REDUCED_STRUCT_FMT
        
    def initialize(self, channel_names: List[str], sample_rate: float, 
                   start_time: float = 0.0, overwrite: bool = True):
        """Setup directory and open files."""
        self.channel_names = channel_names
        self.sample_rate = sample_rate
        
        # Create Directory
        if self.output_path.exists():
            if overwrite:
                shutil.rmtree(self.output_path)
            else:
                raise FileExistsError(f"{self.output_path} already exists")
        
        self.output_path.mkdir(parents=True)
        
        # Create Setup.xml
        # We assume Float64 for simplicity in streaming, or we could infer
        # Creating a minimal engine to re-use xml logic
        # We can create a dummy DataEngine instance or static method?
        # Just manually call _write_setup_xml from a dummy instance
        engine = DataEngine(str(self.output_path.parent))
        engine._write_setup_xml(
            self.output_path, 
            channel_names, 
            sample_rate, 
            np.float64, # Defaulting to Float64 for safety
            ["" for _ in channel_names] # Empty units
        )
        
        # Open Handles
        for i, _ in enumerate(channel_names):
            # Binary Stream (Append mode, though we start fresh)
            bin_path = self.output_path / f"channel_{i}.bin"
            self.bin_files[i] = open(bin_path, "wb")
            
            # Reduced Stream
            red_path = self.output_path / f"channel_{i}.red"
            self.red_files[i] = open(red_path, "wb")
            
            # Buffer
            self.red_buffers[i] = []
            
    def write_chunk(self, chunk_data: Dict[str, np.ndarray]):
        """
        Write a chunk of data.
        chunk_data: Dict mapping {column_name: numpy_array}
        """
        # Map names to indices
        name_to_idx = {name: i for i, name in enumerate(self.channel_names)}
        
        for name, data in chunk_data.items():
            if name not in name_to_idx:
                continue
                
            ch_idx = name_to_idx[name]
            
            # 1. Write Raw (.bin)
            if data.dtype != np.float64:
                data = data.astype(np.float64)
            data.tofile(self.bin_files[ch_idx])
            
            # 2. Process Reduced (.red)
            self._process_reduced_chunk(ch_idx, data)
            
    def _process_reduced_chunk(self, ch_idx: int, data: np.ndarray):
        """Accumulate buffer and write reduced blocks."""
        buffer = self.red_buffers[ch_idx]
        
        # Combine [Buffer] + [New Data]
        if len(buffer) > 0:
            combined = np.concatenate((np.array(buffer, dtype=np.float64), data))
        else:
            combined = data
            
        n_samples = len(combined)
        n_blocks = n_samples // self.REDUCED_BLOCK_SIZE
        
        if n_blocks > 0:
            trunc_len = n_blocks * self.REDUCED_BLOCK_SIZE
            to_process = combined[:trunc_len]
            remainder = combined[trunc_len:]
            
            # Vectorized calc
            reshaped = to_process.reshape(n_blocks, self.REDUCED_BLOCK_SIZE)
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
            avgs = reshaped.mean(axis=1)
            squares = reshaped ** 2
            rmss = np.sqrt(squares.mean(axis=1))
            
            # Interleave and Write
            reduced = np.column_stack((mins, maxs, avgs, rmss)).astype(np.float64)
            reduced.flatten().tofile(self.red_files[ch_idx])
            
            # Update buffer
            self.red_buffers[ch_idx] = remainder.tolist()
        else:
            # Not enough for a block, just update buffer
            self.red_buffers[ch_idx] = combined.tolist()

    def close(self):
        """Flush buffers and close files."""
        # Flush remaining buffers
        for ch_idx, buffer in self.red_buffers.items():
            if len(buffer) > 0:
                # Write partial
                arr = np.array(buffer, dtype=np.float64)
                _min = float(np.min(arr))
                _max = float(np.max(arr))
                _avg = float(np.mean(arr))
                _rms = float(np.sqrt(np.mean(arr**2)))
                
                packed = struct.pack(self.REDUCED_STRUCT_FMT, _min, _max, _avg, _rms)
                self.red_files[ch_idx].write(packed)
        
        # Close all files
        for f in self.bin_files.values():
            try: f.close()
            except: pass
        for f in self.red_files.values():
            try: f.close()
            except: pass
            
        self.bin_files.clear()
        self.red_files.clear()


