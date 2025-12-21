#!/usr/bin/env python3
"""
MPAI Reader Diagnostic Test

This script tests the MPAI reader's load_column_slice() function to identify
why only 14M out of 50M rows are being loaded.

Usage:
    python test_code/test_mpai_chunk_loading.py path/to/your/file.mpai
"""

import sys
import time_graph_cpp as tg

def test_mpai_chunk_loading(mpai_path):
    """Test MPAI reader chunk loading and identify truncation issue."""
    
    print("=" * 80)
    print("MPAI READER DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Open MPAI file
    print(f"\n[1] Opening MPAI file: {mpai_path}")
    try:
        reader = tg.MpaiReader(mpai_path)
        print("✓ File opened successfully")
    except Exception as e:
        print(f"✗ Failed to open file: {e}")
        return False
    
    # Get header information
    print("\n[2] Reading header information...")
    header = reader.get_header()
    row_count = reader.get_row_count()
    col_count = reader.get_column_count()
    col_names = reader.get_column_names()
    
    print(f"  Total rows (from header): {row_count:,}")
    print(f"  Total columns: {col_count}")
    print(f"  Column names: {col_names[:5]}{'...' if len(col_names) > 5 else ''}")
    
    # Inspect chunk metadata for first column
    print("\n[3] Inspecting chunk metadata...")
    time_col = col_names[0]  # Assume first column is time
    
    try:
        col_meta = reader.get_column_metadata(0)  # Get metadata by index
        chunk_count = len(col_meta.chunks)
        print(f"  Column '{time_col}' has {chunk_count} chunks")
        
        total_chunk_rows = 0
        print(f"\n  Chunk Details:")
        print(f"  {'Chunk':<8} {'Rows':<12} {'Offset':<15} {'Compressed':<15} {'Uncompressed':<15}")
        print(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
        
        for i, chunk in enumerate(col_meta.chunks):
            total_chunk_rows += chunk.row_count
            if i < 10 or i >= chunk_count - 5:  # Show first 10 and last 5
                print(f"  {i:<8} {chunk.row_count:<12,} {chunk.offset:<15,} {chunk.compressed_size:<15,} {chunk.uncompressed_size:<15,}")
            elif i == 10:
                print(f"  ... ({chunk_count - 15} more chunks) ...")
        
        print(f"\n  Total rows from all chunks: {total_chunk_rows:,}")
        
        if total_chunk_rows != row_count:
            print(f"  ⚠️  WARNING: Chunk rows ({total_chunk_rows:,}) != Header rows ({row_count:,})")
            print(f"  ⚠️  Difference: {abs(total_chunk_rows - row_count):,} rows")
        else:
            print(f"  ✓ Chunk rows match header row count")
            
    except Exception as e:
        print(f"  ✗ Failed to inspect chunks: {e}")
        return False
    
    # Test loading full column
    print(f"\n[4] Testing load_column_slice('{time_col}', 0, {row_count})...")
    try:
        import time
        start = time.perf_counter()
        data = reader.load_column_slice(time_col, 0, row_count)
        elapsed = time.perf_counter() - start
        
        actual_loaded = len(data)
        print(f"  ✓ Loaded {actual_loaded:,} rows in {elapsed:.2f}s")
        
        if actual_loaded != row_count:
            print(f"\n  ❌ DATA TRUNCATION DETECTED!")
            print(f"  Expected: {row_count:,} rows")
            print(f"  Actual:   {actual_loaded:,} rows")
            print(f"  Missing:  {row_count - actual_loaded:,} rows ({(1 - actual_loaded/row_count)*100:.1f}% loss)")
            
            # Try to identify which chunks are missing
            print(f"\n  Analyzing missing data...")
            if actual_loaded < total_chunk_rows:
                # Calculate expected rows from chunks
                rows_per_chunk = total_chunk_rows / chunk_count
                loaded_chunks_estimate = int(actual_loaded / rows_per_chunk)
                print(f"  Estimated chunks loaded: {loaded_chunks_estimate} / {chunk_count}")
                print(f"  Missing chunks: ~{chunk_count - loaded_chunks_estimate}")
            
            return False
        else:
            print(f"  ✓ All data loaded successfully!")
            
            # Verify data integrity
            if len(data) > 1:
                print(f"\n  Data integrity check:")
                print(f"    First value: {data[0]}")
                print(f"    Last value:  {data[-1]}")
                print(f"    Time range:  {data[-1] - data[0]:.2f}")
            
            return True
            
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mpai_chunk_loading.py <path_to_mpai_file>")
        print("\nExample:")
        print("  python test_code/test_mpai_chunk_loading.py data/large_file.mpai")
        sys.exit(1)
    
    mpai_path = sys.argv[1]
    success = test_mpai_chunk_loading(mpai_path)
    
    print("\n" + "=" * 80)
    if success:
        print("RESULT: ✓ All tests passed - No truncation detected")
    else:
        print("RESULT: ✗ Data truncation detected - See details above")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
