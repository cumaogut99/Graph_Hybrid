/**
 * MPAI Reader - Read data and application state from MPAI format
 *
 * Features:
 * - Memory-mapped I/O (zero-copy)
 * - Lazy chunk loading
 * - Instant statistics access
 * - Application state restoration
 *
 * Target: 300 MB RAM for 50 GB files
 */

#pragma once

#include "mpai_format.hpp"
#include <ctime>
#include <memory>
#include <unordered_map>
#include <utility>

namespace timegraph {
namespace mpai {

class MpaiReader {
public:
  /**
   * Open MPAI file
   *
   * @param filename MPAI file path
   * @param use_mmap Use memory-mapped I/O (default: true)
   */
  explicit MpaiReader(const std::string &filename, bool use_mmap = true);

  ~MpaiReader();

  // Disable copy
  MpaiReader(const MpaiReader &) = delete;
  MpaiReader &operator=(const MpaiReader &) = delete;

  /**
   * Get header (instant, already loaded)
   */
  const MpaiHeader &get_header() const;

  /**
   * Get data metadata (instant, already loaded)
   */
  const DataMetadata &get_data_metadata() const;

  /**
   * Get application state (instant, already loaded)
   */
  const ApplicationState &get_application_state() const;

  /**
   * Get column metadata by name
   */
  const ColumnMetadata *
  get_column_metadata(const std::string &column_name) const;

  /**
   * Get column metadata by index
   */
  const ColumnMetadata &get_column_metadata(uint32_t column_index) const;

  /**
   * Get column statistics (instant, pre-computed)
   *
   * @param column_name Column name
   * @return Pre-computed statistics
   */
  const ColumnStatistics &get_statistics(const std::string &column_name) const;

  /**
   * Load column chunk
   *
   * @param column_index Column index
   * @param chunk_id Chunk ID
   * @return Decompressed chunk data
   *
   * Memory: Only this chunk loaded (~8 MB per 1M rows)
   */
  std::vector<double> load_column_chunk(uint32_t column_index,
                                        uint32_t chunk_id);

  /**
   * Load entire column (use with caution!)
   *
   * @param column_name Column name
   * @return All data for this column
   *
   * Memory: Entire column in memory
   * Warning: For large files, use load_column_chunk() instead
   */
  std::vector<double> load_column(const std::string &column_name);

  /**
   * Load column slice
   *
   * @param column_name Column name
   * @param start_row Start row index
   * @param row_count Number of rows to load
   * @return Requested data slice
   *
   * Memory: Only requested rows loaded
   */
  std::vector<double> load_column_slice(const std::string &column_name,
                                        uint64_t start_row, uint64_t row_count);

  /**
   * Get row count
   */
  uint64_t get_row_count() const;

  /**
   * Get column count
   */
  uint32_t get_column_count() const;

  /**
   * Get column names
   */
  std::vector<std::string> get_column_names() const;

  /**
   * Check if column exists
   */
  bool has_column(const std::string &column_name) const;

  /**
   * Get file size
   */
  uint64_t get_file_size() const;

  /**
   * Get compression ratio
   */
  double get_compression_ratio() const;

  /**
   * Get memory usage (current)
   */
  size_t get_memory_usage() const;

  /**
   * Get interpolated values for multiple signals at a specific timestamp
   *
   * @param timestamp Time point to query
   * @param time_col Name of the time column
   * @param signal_cols Names of signal columns to query
   * @return Map of signal name -> interpolated value
   *
   * Optimization: Loads minimal data chunks and performs batch lookup
   */
  std::unordered_map<std::string, double>
  get_batch_values(double timestamp, const std::string &time_col,
                   const std::vector<std::string> &signal_cols);

  /**
   * Get range of chunk indices [start, end] overlapping with row range.
   * Uses binary search (O(log chunks)).
   *
   * @param column_index Column index
   * @param start_row Start row
   * @param end_row End row
   * @return pair of chunk indices [start_chunk_idx, end_chunk_idx]
   */
  std::pair<uint32_t, uint32_t> get_chunk_range(uint32_t column_index,
                                                uint64_t start_row,
                                                uint64_t end_row) const;

  /**
   * Get approximate row index for a value in a column.
   * Used for time-to-row conversion.
   * Uses binary search on chunk min/max (O(log chunks)).
   *
   * @param column_name Column name
   * @param value Value to search for
   * @return Approximate row index
   */
  uint64_t get_row_for_value(const std::string &column_name,
                             double value) const;

private:
  std::string filename_;
  bool use_mmap_;

  // File handle
  int fd_;           // File descriptor (for mmap)
  void *mmap_data_;  // Memory-mapped data
  size_t mmap_size_; // Mapped size

  // Metadata (loaded at open, ~100 KB)
  MpaiHeader header_;
  DataMetadata data_metadata_;
  ApplicationState app_state_;

  // Column name → index mapping
  std::unordered_map<std::string, uint32_t> column_index_map_;

  // Chunk cache (LRU, max 10 chunks =  // Cache
  struct ChunkCacheEntry {
    uint32_t column_index;
    uint32_t chunk_id;
    std::vector<double> data;
    std::time_t last_access_time;
  };

  std::vector<ChunkCacheEntry> chunk_cache_;
  size_t max_cache_size_;

  // Optimization: Pre-calculated start rows for each chunk in each column
  // Map column_index -> vector of start rows (one per chunk)
  std::vector<std::vector<uint64_t>> chunk_start_rows_;

  // Helper methods
  void open_file();
  void load_header();
  void load_data_metadata();
  void load_application_state();
  void build_column_index();
  void build_chunk_index(); // New optimization method

  std::vector<uint8_t> read_compressed_chunk(uint64_t offset, uint32_t size);
  std::vector<double>
  decompress_chunk(const std::vector<uint8_t> &compressed_data,
                   uint32_t uncompressed_size);

  void add_to_cache(uint32_t column_index, uint32_t chunk_id,
                    const std::vector<double> &data);
  const std::vector<double> *get_from_cache(uint32_t column_index,
                                            uint32_t chunk_id);
  void evict_oldest_cache_entry();
};

} // namespace mpai
} // namespace timegraph
