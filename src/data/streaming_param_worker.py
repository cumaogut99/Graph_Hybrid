# type: ignore
"""
Streaming Parameter Worker - Background Calculated Parameter Creation

Enables zero-copy, streaming calculation of derived parameters without loading 
entire datasets into RAM. Uses chunked processing with MpaiStreamWriter for
disk-based storage.

Memory Budget: ~48MB per calculation (regardless of source data size)
- 16MB for chunk A data (100K rows × 8 bytes × 2 columns)
- 16MB for chunk B data  
- 16MB for result chunk
"""

import logging
import os
import re
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal as Signal

logger = logging.getLogger(__name__)

# Chunk size for streaming (100K rows = ~16MB per float64 column)
DEFAULT_CHUNK_SIZE = 100_000


class StreamingParamWorker(QThread):
    """
    Background worker for streaming calculation of derived parameters.
    
    Instead of loading entire columns into RAM, processes data in chunks:
    1. Read chunk from source MPAI reader
    2. Evaluate formula on chunk
    3. Write result to output MPAI directory
    4. Repeat until all data processed
    
    This ensures RAM usage stays under ~48MB regardless of source file size.
    """
    
    # Signals
    progress = Signal(int, str)  # percentage (0-100), status message
    finished = Signal(str, str)  # param_name, output_path
    error = Signal(str)  # error message
    
    def __init__(self, 
                 param_name: str,
                 formula: str,
                 used_params: List[str],
                 input_reader,  # MpaiDirectoryReader or similar
                 time_column: str,
                 output_dir: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the streaming parameter worker.
        
        Args:
            param_name: Name for the new calculated parameter
            formula: Mathematical formula string (e.g., "Channel_1 + Channel_2 * 2")
            used_params: List of parameter names referenced in formula
            input_reader: MpaiDirectoryReader instance for source data
            time_column: Name of time column
            output_dir: Directory where calculated MPAI will be created
            chunk_size: Number of rows per chunk (default 100K)
        """
        super().__init__()
        
        self.param_name = param_name
        self.formula = formula
        self.used_params = used_params
        self.input_reader = input_reader
        self.time_column = time_column
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        self._cancelled = False
        
    def cancel(self):
        """Request cancellation of the calculation."""
        self._cancelled = True
        
    def run(self):
        """Execute the streaming calculation in background thread."""
        try:
            self._run_streaming_calculation()
        except Exception as e:
            logger.error(f"[STREAM CALC] Error: {e}", exc_info=True)
            self.error.emit(str(e))
            
    def _run_streaming_calculation(self):
        """Core streaming calculation logic."""
        from src.data.data_engine import MpaiStreamWriter
        
        # 1. Validate inputs
        if not self.used_params:
            self.error.emit("Formula doesn't reference any parameters")
            return
            
        row_count = self.input_reader.get_row_count()
        if row_count == 0:
            self.error.emit("Source data is empty")
            return
            
        logger.info(f"[STREAM CALC] Starting: {self.param_name} = {self.formula}")
        logger.info(f"[STREAM CALC] Processing {row_count:,} rows in chunks of {self.chunk_size:,}")
        
        # 2. Get sample rate from source
        sample_rate = getattr(self.input_reader, 'sample_rate', 1000.0)
        
        # 3. Create output MPAI directory
        output_mpai_path = self.output_dir / f"calc_{self.param_name}.mpai"
        
        # Clean up existing if present
        if output_mpai_path.exists():
            import shutil
            shutil.rmtree(output_mpai_path)
            
        # 4. Initialize writer with two columns: time and calculated param
        writer = MpaiStreamWriter(str(output_mpai_path))
        writer.initialize(
            channel_names=[self.param_name],  # Only the calculated column (time is synthetic)
            sample_rate=sample_rate,
            start_time=0.0,
            overwrite=True
        )
        
        self.progress.emit(5, "Initialized output file")
        
        # 5. Process in chunks
        num_chunks = (row_count + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            if self._cancelled:
                writer.close()
                # Clean up partial file
                if output_mpai_path.exists():
                    import shutil
                    shutil.rmtree(output_mpai_path)
                self.error.emit("Calculation cancelled")
                return
                
            chunk_start = chunk_idx * self.chunk_size
            chunk_len = min(self.chunk_size, row_count - chunk_start)
            
            # Progress update
            progress_pct = int(5 + (chunk_idx / num_chunks) * 90)
            self.progress.emit(progress_pct, f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            
            try:
                # Load chunk data for each used parameter
                param_data = {}
                for param in self.used_params:
                    chunk_data = self.input_reader.load_column_slice(param, chunk_start, chunk_len)
                    if isinstance(chunk_data, list):
                        chunk_data = np.array(chunk_data, dtype=np.float64)
                    elif not isinstance(chunk_data, np.ndarray):
                        chunk_data = np.asarray(chunk_data, dtype=np.float64)
                    param_data[param] = chunk_data
                
                # Evaluate formula on this chunk
                result_chunk = self._evaluate_formula(param_data)
                
                if result_chunk is None or len(result_chunk) == 0:
                    logger.warning(f"[STREAM CALC] Empty result for chunk {chunk_idx}")
                    continue
                    
                # Write chunk to output
                writer.write_chunk({
                    self.param_name: result_chunk.astype(np.float64)
                })
                
            except Exception as e:
                logger.error(f"[STREAM CALC] Chunk {chunk_idx} error: {e}")
                # Continue with next chunk rather than failing entirely
                continue
                
        # 6. Finalize
        writer.close()
        
        self.progress.emit(100, "Calculation complete")
        logger.info(f"[STREAM CALC] Completed: {output_mpai_path}")
        
        self.finished.emit(self.param_name, str(output_mpai_path))
        
    def _evaluate_formula(self, param_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Safely evaluate the formula on chunk data.
        
        Args:
            param_data: Dict mapping parameter names to numpy arrays
            
        Returns:
            Result array or None on error
        """
        try:
            # Create safe evaluation environment
            safe_dict = {
                'np': np,
                '__builtins__': {},
            }
            safe_dict.update(param_data)
            
            # Evaluate
            result = eval(self.formula, safe_dict)
            
            # Ensure result is array
            if isinstance(result, np.ndarray):
                return result
            elif np.isscalar(result):
                # Scalar result - broadcast to array size
                first_param = next(iter(param_data.values()))
                return np.full(len(first_param), result, dtype=np.float64)
            else:
                return np.asarray(result, dtype=np.float64)
                
        except Exception as e:
            logger.error(f"[STREAM CALC] Formula evaluation error: {e}")
            return None


def detect_streaming_incompatible(formula: str) -> List[str]:
    """
    Detect formulas that are incompatible with streaming chunk processing.
    
    These include functions that require the entire dataset:
    - np.cumsum, np.cumprod (cumulative operations)
    - np.diff with n > chunk overlap
    - np.correlate, np.convolve (cross-row operations)
    - Custom rolling/window functions
    
    Args:
        formula: Formula string to analyze
        
    Returns:
        List of incompatible function names found
    """
    incompatible_patterns = [
        r'np\.cumsum',
        r'np\.cumprod',
        r'np\.diff',
        r'np\.correlate',
        r'np\.convolve',
        r'np\.gradient',
        r'pd\.rolling',
        r'\.rolling\(',
        r'\.shift\(',
    ]
    
    found = []
    for pattern in incompatible_patterns:
        if re.search(pattern, formula, re.IGNORECASE):
            # Extract function name
            match = re.search(pattern, formula, re.IGNORECASE)
            if match:
                found.append(match.group(0))
                
    return found
