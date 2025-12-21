/**
 * @file fast_stats_calculator.hpp
 * @brief O(1) statistics calculator using pre-aggregated chunk metadata
 *
 * Enables instant statistics on any range without loading raw data.
 * Performance: < 1ms for 100M points (complete chunks) + ~5ms for edge data
 */

#pragma once

#include "timegraph/data/chunk_metadata.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include <cmath>
#include <vector>


namespace timegraph {
namespace mpai {

class FastStatsCalculator {
public:
  /**
   * Calculate statistics for a time range using pre-aggregated metadata
   *
   * Algorithm:
   * 1. For complete chunks: Use pre-computed sum/sum_squares (O(1))
   * 2. For partial chunks: Load only edge data and calculate
   *
   * @param reader MPAI reader instance
   * @param column_name Column to calculate stats for
   * @param start_row Start row index (inclusive)
   * @param end_row End row index (exclusive)
   * @return RangeStatistics with min, max, mean, std, rms, etc.
   */
  static RangeStatistics
  calculate_range_statistics(MpaiReader &reader, const std::string &column_name,
                             uint64_t start_row, uint64_t end_row);

  /**
   * Calculate statistics for a time range (time-based indexing)
   *
   * @param reader MPAI reader instance
   * @param column_name Column to calculate stats for
   * @param start_time Start time value
   * @param end_time End time value
   * @param time_column Name of time column (for indexing)
   * @return RangeStatistics
   */
  static RangeStatistics calculate_time_range_statistics(
      MpaiReader &reader, const std::string &column_name, double start_time,
      double end_time, const std::string &time_column = "time");

  /**
   * Find rows that violate limit thresholds using Skip Logic
   *
   * Algorithm (Skip Logic):
   * 1. Check chunk min/max against limits
   * 2. If chunk.max < lower_limit OR chunk.min > upper_limit: SKIP (all values outside)
   * 3. If chunk.min >= lower_limit AND chunk.max <= upper_limit: SKIP (all values inside)
   * 4. Otherwise: Load chunk and find exact violations
   *
   * Performance: O(relevant_chunks) instead of O(all_chunks)
   * For sparse violations (~1%), achieves ~100x speedup
   *
   * @param reader MPAI reader instance
   * @param column_name Column to check for violations
   * @param lower_limit Lower threshold (use -INFINITY to disable)
   * @param upper_limit Upper threshold (use +INFINITY to disable)
   * @param start_row Start row index (inclusive)
   * @param end_row End row index (exclusive, use UINT64_MAX for all)
   * @return Vector of (start_row, end_row) pairs for violation regions
   */
  static std::vector<std::pair<uint64_t, uint64_t>> find_limit_violations(
      MpaiReader &reader, const std::string &column_name,
      double lower_limit, double upper_limit,
      uint64_t start_row = 0, uint64_t end_row = UINT64_MAX);

  /**
   * Count rows violating limits using Skip Logic
   *
   * @param reader MPAI reader instance  
   * @param column_name Column to check
   * @param lower_limit Lower threshold
   * @param upper_limit Upper threshold
   * @return Number of rows with values outside [lower_limit, upper_limit]
   */
  static uint64_t count_limit_violations(
      MpaiReader &reader, const std::string &column_name,
      double lower_limit, double upper_limit);

private:
  /**
   * Find row index for a given time value
   * Uses binary search on chunk metadata
   */
  static uint64_t time_to_row(MpaiReader &reader,
                              const std::string &time_column,
                              double time_value);

  /**
   * Calculate statistics for edge chunk (partial overlap)
   */
  static void accumulate_partial_chunk(MpaiReader &reader,
                                       const std::string &column_name,
                                       uint64_t chunk_start, uint64_t chunk_end,
                                       uint64_t range_start, uint64_t range_end,
                                       RangeStatistics &stats);
};

} // namespace mpai
} // namespace timegraph
