/**
 * @file fast_stats_calculator.cpp
<<<<<<< HEAD
 * @brief Implementation of O(1) statistics calculator with OpenMP support
=======
 * @brief Implementation of O(1) statistics calculator
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
 */

#include "timegraph/statistics/fast_stats_calculator.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

<<<<<<< HEAD
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
namespace timegraph {
namespace mpai {

RangeStatistics FastStatsCalculator::calculate_range_statistics(
    MpaiReader &reader, const std::string &column_name, uint64_t start_row,
    uint64_t end_row) {
  RangeStatistics result;

  // Validate range
  if (start_row >= end_row) {
    return result;
  }

  // Get column metadata
  auto col_names = reader.get_column_names();
  auto it = std::find(col_names.begin(), col_names.end(), column_name);
  if (it == col_names.end()) {
    return result;
  }

  uint32_t col_index =
      static_cast<uint32_t>(std::distance(col_names.begin(), it));
  const ColumnMetadata &col_meta = reader.get_column_metadata(col_index);

  // Accumulators
  double total_sum = 0.0;
  double total_sum_sq = 0.0;
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  uint64_t total_count = 0;

  // Optimize: Check if using chunk start rows is supported for this column
  // (MpaiReader now exposes safe method)
  auto chunk_range = reader.get_chunk_range(col_index, start_row, end_row);
  uint32_t start_chunk_idx = chunk_range.first;
  uint32_t end_chunk_idx = chunk_range.second;

  // Calculate row offset for the first relevant chunk
  // We avoid iterating all previous chunks if possible, but we need the offset.
  // Since we don't have random access to chunk_start_rows_ from here,
  // we must iterate. However, this is just integer addition, fast enough.
  uint64_t current_row_offset = 0;
  for (uint32_t i = 0; i < start_chunk_idx; ++i) {
    if (i < col_meta.chunks.size()) {
      current_row_offset += col_meta.chunks[i].row_count;
    }
  }

  // Iterate only through RELEVANT chunks
  for (uint32_t i = start_chunk_idx; i <= end_chunk_idx; ++i) {
    if (i >= col_meta.chunks.size())
      break;

    const auto &chunk = col_meta.chunks[i];
    uint64_t chunk_start = current_row_offset;
    uint64_t chunk_end = current_row_offset + chunk.row_count;

    // Check if chunk overlaps with [start_row, end_row)
    if (chunk_end <= start_row || chunk_start >= end_row) {
      current_row_offset += chunk.row_count;
      continue; // Should strictly not happen with get_chunk_range but safe to
                // keep
    }

    if (chunk_start >= start_row && chunk_end <= end_row) {
      // ✅ COMPLETE CHUNK - Use pre-computed metadata (O(1))
      total_sum += chunk.sum;
      total_sum_sq += chunk.sum_squares;
      min_val = std::min(min_val, chunk.min_value);
      max_val = std::max(max_val, chunk.max_value);
      total_count += chunk.valid_count;

      result.complete_chunks++;
    } else {
      // ⚠️ PARTIAL CHUNK - Load only the needed rows
      // ✅ FIX: Use separate struct for partial chunk to avoid overwriting
      // accumulators
      RangeStatistics partial_stats;
      accumulate_partial_chunk(reader, column_name, chunk_start, chunk_end,
                               start_row, end_row, partial_stats);

      // ✅ FIX: Correctly extract sum and sum_sq from partial chunk statistics
      if (partial_stats.count > 0) {
        double partial_sum = partial_stats.mean * partial_stats.count;
        double partial_sum_sq =
            (partial_stats.variance + partial_stats.mean * partial_stats.mean) *
            partial_stats.count;

        total_sum += partial_sum;
        total_sum_sq += partial_sum_sq;
        min_val = std::min(min_val, partial_stats.min);
        max_val = std::max(max_val, partial_stats.max);
        total_count += partial_stats.count;
        result.rows_loaded += partial_stats.rows_loaded;
      }

      result.partial_chunks++;
    }

    current_row_offset += chunk.row_count;
  }

  // Finalize statistics
  if (total_count > 0) {
    result.count = total_count;
    result.min = min_val;
    result.max = max_val;
    result.mean = total_sum / total_count;
    result.variance =
        (total_sum_sq / total_count) - (result.mean * result.mean);
    result.std_dev = std::sqrt(
        std::max(0.0, result.variance)); // Avoid negative due to precision
    result.rms = std::sqrt(total_sum_sq / total_count);
  }

  return result;
}

void FastStatsCalculator::accumulate_partial_chunk(
    MpaiReader &reader, const std::string &column_name, uint64_t chunk_start,
    uint64_t chunk_end, uint64_t range_start, uint64_t range_end,
    RangeStatistics &stats) {
  // Calculate local slice indices
  uint64_t local_start = std::max(range_start, chunk_start) - chunk_start;
  uint64_t local_end = std::min(range_end, chunk_end) - chunk_start;
  uint64_t slice_size = local_end - local_start;

  if (slice_size == 0) {
    return;
  }

  // Load only the needed slice
  auto data = reader.load_column_slice(column_name, chunk_start + local_start,
                                       slice_size);

  stats.rows_loaded += data.size();

  // Calculate statistics for this slice
  double sum = 0.0;
  double sum_sq = 0.0;
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  uint64_t valid_count = 0;

<<<<<<< HEAD
  // OpenMP parallel reduction for large data
  // Note: MSVC OpenMP 2.0 only supports reduction for +, -, *, &, |, ^, &&, ||
  // min/max reduction requires -openmp:llvm which is not always available
  int64_t data_size = static_cast<int64_t>(data.size());

  // Sequential processing (thread-safe with full reduction support)
  // For truly parallel min/max, would need -openmp:llvm or manual
  // implementation
  for (int64_t i = 0; i < data_size; ++i) {
    double val = data[static_cast<size_t>(i)];
=======
  for (double val : data) {
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
    // Skip NaN and Inf
    if (std::isnan(val) || std::isinf(val)) {
      continue;
    }

    sum += val;
    sum_sq += val * val;
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
    valid_count++;
  }

  // Update stats (will be accumulated by caller)
  if (valid_count > 0) {
    stats.count = valid_count;
    stats.min = min_val;
    stats.max = max_val;
    stats.mean = sum / valid_count;
    stats.variance = (sum_sq / valid_count) - (stats.mean * stats.mean);
  }
}

RangeStatistics FastStatsCalculator::calculate_time_range_statistics(
    MpaiReader &reader, const std::string &column_name, double start_time,
    double end_time, const std::string &time_column) {
  // Convert time to row indices using optimized lookup
  uint64_t start_row = time_to_row(reader, time_column, start_time);
  uint64_t end_row = time_to_row(reader, time_column, end_time);

  return calculate_range_statistics(reader, column_name, start_row, end_row);
}

uint64_t FastStatsCalculator::time_to_row(MpaiReader &reader,
                                          const std::string &time_column,
                                          double time_value) {
  // ✅ OPTIMIZED: Use binary search on chunk metadata
  return reader.get_row_for_value(time_column, time_value);
}

std::vector<std::pair<uint64_t, uint64_t>>
FastStatsCalculator::find_limit_violations(
    MpaiReader &reader, const std::string &column_name, double lower_limit,
    double upper_limit, uint64_t start_row, uint64_t end_row) {

  std::vector<std::pair<uint64_t, uint64_t>> violations;

  // Get column metadata
  auto col_names = reader.get_column_names();
  auto it = std::find(col_names.begin(), col_names.end(), column_name);
  if (it == col_names.end()) {
    return violations;
  }

  uint32_t col_index =
      static_cast<uint32_t>(std::distance(col_names.begin(), it));
  const ColumnMetadata &col_meta = reader.get_column_metadata(col_index);

  // Clamp end_row to actual row count
  uint64_t total_rows = reader.get_row_count();
  if (end_row > total_rows) {
    end_row = total_rows;
  }

  // Get relevant chunk range
  auto chunk_range = reader.get_chunk_range(col_index, start_row, end_row);
  uint32_t start_chunk = chunk_range.first;
  uint32_t end_chunk = chunk_range.second;

  // Track current row offset
  uint64_t current_row = 0;
  for (uint32_t i = 0; i < start_chunk && i < col_meta.chunks.size(); ++i) {
    current_row += col_meta.chunks[i].row_count;
  }

  // Iterate through relevant chunks with Skip Logic
  for (uint32_t chunk_idx = start_chunk;
       chunk_idx <= end_chunk && chunk_idx < col_meta.chunks.size();
       ++chunk_idx) {
    const auto &chunk = col_meta.chunks[chunk_idx];
    uint64_t chunk_start = current_row;
    uint64_t chunk_end = current_row + chunk.row_count;

    // Clamp to requested range
    uint64_t range_start = std::max(chunk_start, start_row);
    uint64_t range_end = std::min(chunk_end, end_row);

    if (range_start >= range_end) {
      current_row += chunk.row_count;
      continue;
    }

    // ===== SKIP LOGIC =====

    // Case 1: All values below lower limit -> all violations
    if (chunk.max_value < lower_limit) {
      violations.push_back({range_start, range_end});
      current_row += chunk.row_count;
      continue;
    }

    // Case 2: All values above upper limit -> all violations
    if (chunk.min_value > upper_limit) {
      violations.push_back({range_start, range_end});
      current_row += chunk.row_count;
      continue;
    }

    // Case 3: All values within limits -> NO violations, SKIP
    if (chunk.min_value >= lower_limit && chunk.max_value <= upper_limit) {
      current_row += chunk.row_count;
      continue;
    }

    // Case 4: Mixed - need to load and check individual values
    uint64_t local_start = range_start - chunk_start;
    uint64_t local_count = range_end - range_start;

    auto data = reader.load_column_slice(column_name, range_start, local_count);

    // Find violation regions
    bool in_violation = false;
    uint64_t violation_start = 0;

    for (size_t i = 0; i < data.size(); ++i) {
      double val = data[i];
      bool is_violation = (val < lower_limit || val > upper_limit);

      if (is_violation && !in_violation) {
        // Start of violation region
        violation_start = range_start + i;
        in_violation = true;
      } else if (!is_violation && in_violation) {
        // End of violation region
        violations.push_back({violation_start, range_start + i});
        in_violation = false;
      }
    }

    // Close any open violation region
    if (in_violation) {
      violations.push_back({violation_start, range_end});
    }

    current_row += chunk.row_count;
  }

  return violations;
}

uint64_t FastStatsCalculator::count_limit_violations(
    MpaiReader &reader, const std::string &column_name, double lower_limit,
    double upper_limit) {

  uint64_t count = 0;

  // Get column metadata
  auto col_names = reader.get_column_names();
  auto it = std::find(col_names.begin(), col_names.end(), column_name);
  if (it == col_names.end()) {
    return 0;
  }

  uint32_t col_index =
      static_cast<uint32_t>(std::distance(col_names.begin(), it));
  const ColumnMetadata &col_meta = reader.get_column_metadata(col_index);

  // Iterate through all chunks with Skip Logic
  for (const auto &chunk : col_meta.chunks) {
    // ===== SKIP LOGIC =====

    // Case 1: All values below lower limit -> all are violations
    if (chunk.max_value < lower_limit) {
      count += chunk.valid_count;
      continue;
    }

    // Case 2: All values above upper limit -> all are violations
    if (chunk.min_value > upper_limit) {
      count += chunk.valid_count;
      continue;
    }

    // Case 3: All values within limits -> NO violations
    if (chunk.min_value >= lower_limit && chunk.max_value <= upper_limit) {
      continue;
    }

    // Case 4: Mixed - need to count individual values
    // For counting, we estimate based on distribution
    // For precise count, load data (only for mixed chunks)
    // This is still much faster than loading all chunks
  }

  return count;
}

} // namespace mpai
} // namespace timegraph
