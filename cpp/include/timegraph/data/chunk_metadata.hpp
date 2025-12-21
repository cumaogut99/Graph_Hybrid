/**
 * @file chunk_metadata.hpp
 * @brief Pre-aggregated chunk metadata for O(1) statistics calculations
 *
 * Memory-efficient metadata structure that enables instant statistics
 * on any range without loading raw data.
 *
 * Memory overhead: 56 bytes per chunk
 * Example: 50M rows ÷ 4096 rows/chunk = 12,208 chunks × 56 bytes = 683 KB
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace timegraph {
namespace mpai {

/**
 * Pre-computed statistics for a single chunk
 *
 * Enables O(1) aggregation for complete chunks:
 * - Mean = Σ(chunk.sum) / Σ(chunk.count)
 * - Variance = Σ(chunk.sum_squares) / Σ(chunk.count) - mean²
 * - Min = min(chunk.min_value)
 * - Max = max(chunk.max_value)
 */
struct ChunkMetadata {
  uint64_t start_row; ///< Global row index (start of chunk)
  uint64_t row_count; ///< Number of rows in this chunk

  // Pre-computed statistics (for O(1) aggregation)
  double sum;         ///< Σx (sum of all values)
  double sum_squares; ///< Σx² (sum of squares, for variance/std)
  double min_value;   ///< Minimum value in chunk
  double max_value;   ///< Maximum value in chunk
  uint32_t count;     ///< Valid data points (excludes NaN/Inf)

  // Padding to align to 64 bytes (cache line)
  uint32_t _padding;

  ChunkMetadata()
      : start_row(0), row_count(0), sum(0.0), sum_squares(0.0),
        min_value(std::numeric_limits<double>::infinity()),
        max_value(-std::numeric_limits<double>::infinity()), count(0),
        _padding(0) {}
};

// Verify size is as expected
static_assert(sizeof(ChunkMetadata) == 56, "ChunkMetadata must be 56 bytes");

/**
 * Metadata for all chunks in a column
 */
struct ColumnChunkMetadata {
  std::string column_name;
  std::vector<ChunkMetadata> chunks;

  /**
   * Get total memory usage for this column's metadata
   */
  size_t memory_usage() const {
    return sizeof(ColumnChunkMetadata) + column_name.capacity() +
           chunks.capacity() * sizeof(ChunkMetadata);
  }
};

/**
 * Statistics result from range query
 */
struct RangeStatistics {
  double min;
  double max;
  double mean;
  double variance;
  double std_dev;
  double rms;
  uint64_t count;

  // Metadata about the calculation
  uint32_t complete_chunks; ///< Number of complete chunks (O(1))
  uint32_t partial_chunks;  ///< Number of partial chunks (loaded)
  uint64_t rows_loaded;     ///< Actual rows loaded from disk

  RangeStatistics()
      : min(std::numeric_limits<double>::infinity()),
        max(-std::numeric_limits<double>::infinity()), mean(0.0), variance(0.0),
        std_dev(0.0), rms(0.0), count(0), complete_chunks(0), partial_chunks(0),
        rows_loaded(0) {}
};

} // namespace mpai
} // namespace timegraph
