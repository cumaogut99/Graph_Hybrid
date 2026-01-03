#include "timegraph/processing/statistics_engine.hpp"
<<<<<<< HEAD
#include "timegraph/processing/arrow_utils.hpp"
#include "timegraph/processing/simd_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

=======
#include "timegraph/processing/simd_utils.hpp"
#include "timegraph/processing/arrow_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <cstdio>
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

#ifdef HAVE_ARROW
#include <arrow/api.h>
#include <arrow/compute/api.h>
#endif

namespace timegraph {

<<<<<<< HEAD
double StatisticsEngine::calculate_mean(const double *data, size_t length) {
  if (length == 0) {
    return 0.0;
  }

  // Priority: Arrow Compute > SIMD > Scalar
  const size_t arrow_threshold =
      10000; // Arrow overhead worth it for large data
  const size_t simd_threshold = 1000;

#ifdef HAVE_ARROW
  // Use Arrow Compute for large datasets (SIMD-optimized + numerically stable)
  if (length >= arrow_threshold && arrow_utils::is_arrow_available()) {
    try {
      // Create Arrow array directly from pointer (zero-copy when possible)
      arrow::DoubleBuilder builder;
      auto status = builder.AppendValues(data, length);
      if (status.ok()) {
        auto maybe_array = builder.Finish();
        if (maybe_array.ok()) {
          arrow::compute::ExecContext ctx;
          auto result = arrow::compute::CallFunction(
              "mean", {maybe_array.ValueOrDie()}, &ctx);
          if (result.ok()) {
            return result.ValueOrDie().scalar_as<arrow::DoubleScalar>().value;
          }
        }
      }
    } catch (...) {
      // Fall through to SIMD/scalar
    }
  }
#endif

  // Use SIMD if available
  if (length >= simd_threshold && simd::has_avx2()) {
#ifdef TIMEGRAPH_HAS_AVX2
    return simd::stats::mean_avx2(data, length);
#else
    return simd::scalar::mean_scalar(data, length);
#endif
  } else {
    // Scalar fallback
    return simd::scalar::mean_scalar(data, length);
  }
}

double StatisticsEngine::calculate_std_dev(const double *data, size_t length,
                                           double mean) {
  if (length == 0) {
    return 0.0;
  }

  // Use SIMD if available and data is large enough
  const size_t simd_threshold = 1000;

  if (length >= simd_threshold && simd::has_avx2()) {
#ifdef TIMEGRAPH_HAS_AVX2
    return simd::stats::stddev_avx2(data, length, mean);
#else
    double var = simd::scalar::variance_scalar(data, length, mean);
    return std::sqrt(var);
#endif
  } else {
    // Scalar fallback
    double var = simd::scalar::variance_scalar(data, length, mean);
    return std::sqrt(var);
  }
}

void StatisticsEngine::calculate_min_max(const double *data, size_t length,
                                         double &min, double &max) {
  if (length == 0) {
    min = max = std::numeric_limits<double>::quiet_NaN();
    return;
  }

  const size_t arrow_threshold = 10000;

#ifdef HAVE_ARROW
  // Use Arrow Compute for large datasets (SIMD-optimized)
  if (length >= arrow_threshold && arrow_utils::is_arrow_available()) {
    try {
      arrow::DoubleBuilder builder;
      auto status = builder.AppendValues(data, length);
      if (status.ok()) {
        auto maybe_array = builder.Finish();
        if (maybe_array.ok()) {
          arrow::compute::ExecContext ctx;
          auto result = arrow::compute::CallFunction(
              "min_max", {maybe_array.ValueOrDie()}, &ctx);
          if (result.ok()) {
            auto minmax_scalar =
                result.ValueOrDie().scalar_as<arrow::StructScalar>();
            min = std::static_pointer_cast<arrow::DoubleScalar>(
                      minmax_scalar.value[0])
                      ->value;
            max = std::static_pointer_cast<arrow::DoubleScalar>(
                      minmax_scalar.value[1])
                      ->value;
            return;
          }
        }
      }
    } catch (...) {
      // Fall through to scalar
    }
  }
#endif

  // Scalar implementation
  min = std::numeric_limits<double>::infinity();
  max = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(data[i])) {
      if (data[i] < min)
        min = data[i];
      if (data[i] > max)
        max = data[i];
    }
  }

  // Handle case where all values are NaN
  if (std::isinf(min))
    min = 0.0;
  if (std::isinf(max))
    max = 0.0;
}

ColumnStatistics StatisticsEngine::calculate(const DataFrame &df,
                                             const std::string &column_name) {
  return calculate_range(df, column_name, 0, df.row_count());
}

ColumnStatistics
StatisticsEngine::calculate_range(const DataFrame &df,
                                  const std::string &column_name,
                                  size_t start_index, size_t end_index) {
  ColumnStatistics stats;

  // Validate inputs
  if (!df.has_column(column_name)) {
    throw std::runtime_error("Column not found: " + column_name);
  }

  if (start_index >= end_index || end_index > df.row_count()) {
    throw std::runtime_error("Invalid index range");
  }

  // Get column data
  const double *data = df.get_column_ptr_f64(column_name);
  size_t length = end_index - start_index;
  const double *range_data = data + start_index;

  // Count valid values
  stats.count = length;
  stats.valid_count = 0;
  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(range_data[i])) {
      stats.valid_count++;
    }
  }

  if (stats.valid_count == 0) {
    return stats; // All NaN, return zeros
  }

  // Calculate statistics
  stats.mean = calculate_mean(range_data, length);
  stats.std_dev = calculate_std_dev(range_data, length, stats.mean);
  calculate_min_max(range_data, length, stats.min, stats.max);

  // Calculate sum and RMS
  stats.sum = 0.0;
  double sum_squares = 0.0;
  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(range_data[i])) {
      stats.sum += range_data[i];
      sum_squares += range_data[i] * range_data[i];
    }
  }

  // Calculate RMS
  if (stats.valid_count > 0) {
    stats.rms = std::sqrt(sum_squares / stats.valid_count);
  }

  // Calculate peak-to-peak
  stats.peak_to_peak = stats.max - stats.min;

  // Calculate median (simple implementation - not optimized)
  std::vector<double> valid_values;
  valid_values.reserve(stats.valid_count);
  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(range_data[i])) {
      valid_values.push_back(range_data[i]);
    }
  }

  if (!valid_values.empty()) {
    std::nth_element(valid_values.begin(),
                     valid_values.begin() + valid_values.size() / 2,
                     valid_values.end());
    stats.median = valid_values[valid_values.size() / 2];
  }

  return stats;
}

ThresholdStatistics StatisticsEngine::calculate_with_threshold(
    const DataFrame &df, const std::string &column_name,
    const std::string &time_column, double threshold) {
  ThresholdStatistics stats;

  // First calculate basic statistics
  ColumnStatistics basic_stats = calculate(df, column_name);
  stats.mean = basic_stats.mean;
  stats.std_dev = basic_stats.std_dev;
  stats.min = basic_stats.min;
  stats.max = basic_stats.max;
  stats.median = basic_stats.median;
  stats.sum = basic_stats.sum;
  stats.count = basic_stats.count;
  stats.valid_count = basic_stats.valid_count;

  // Set threshold
  stats.threshold = threshold;

  // Get data
  const double *data = df.get_column_ptr_f64(column_name);
  const double *time_data = df.get_column_ptr_f64(time_column);
  size_t length = df.row_count();

  // Count above/below threshold
  stats.above_count = 0;
  stats.below_count = 0;
  stats.time_above = 0.0;
  stats.time_below = 0.0;

  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(data[i])) {
      if (data[i] > threshold) {
        stats.above_count++;
        // Calculate time duration (assuming uniform sampling)
        if (i > 0) {
          stats.time_above += time_data[i] - time_data[i - 1];
        }
      } else {
        stats.below_count++;
        if (i > 0) {
          stats.time_below += time_data[i] - time_data[i - 1];
        }
      }
    }
  }

  // Calculate percentages
  if (stats.valid_count > 0) {
    stats.above_percentage = (100.0 * stats.above_count) / stats.valid_count;
    stats.below_percentage = (100.0 * stats.below_count) / stats.valid_count;
  }

  return stats;
}

std::vector<ColumnStatistics> StatisticsEngine::calculate_rolling(
    const DataFrame &df, const std::string &column_name, size_t window_size) {
  if (window_size == 0) {
    throw std::runtime_error("Window size must be > 0");
  }

  size_t length = df.row_count();
  std::vector<ColumnStatistics> results;

  if (length < window_size) {
    return results; // Not enough data
  }

  // Calculate rolling statistics
  for (size_t i = 0; i <= length - window_size; ++i) {
    results.push_back(calculate_range(df, column_name, i, i + window_size));
  }

  return results;
}

double StatisticsEngine::percentile(const DataFrame &df,
                                    const std::string &column_name,
                                    double percentile_value) {
  if (percentile_value < 0.0 || percentile_value > 100.0) {
    throw std::runtime_error("Percentile must be between 0 and 100");
  }

  // Get data
  const double *data = df.get_column_ptr_f64(column_name);
  size_t length = df.row_count();

  // Collect valid values
  std::vector<double> valid_values;
  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(data[i])) {
      valid_values.push_back(data[i]);
    }
  }

  if (valid_values.empty()) {
    return 0.0;
  }

  // Sort and find percentile
  std::sort(valid_values.begin(), valid_values.end());
  size_t index = static_cast<size_t>((percentile_value / 100.0) *
                                     (valid_values.size() - 1));
  return valid_values[index];
}

std::vector<size_t> StatisticsEngine::histogram(const DataFrame &df,
                                                const std::string &column_name,
                                                size_t num_bins) {
  if (num_bins == 0) {
    throw std::runtime_error("Number of bins must be > 0");
  }

  // Get min/max
  double min_val, max_val;
  const double *data = df.get_column_ptr_f64(column_name);
  size_t length = df.row_count();
  calculate_min_max(data, length, min_val, max_val);

  // Create bins
  std::vector<size_t> bins(num_bins, 0);
  double bin_width = (max_val - min_val) / num_bins;

  if (bin_width == 0.0) {
    // All values are the same
    bins[0] = length;
    return bins;
  }

  // Fill bins
  for (size_t i = 0; i < length; ++i) {
    if (!std::isnan(data[i])) {
      size_t bin_index = static_cast<size_t>((data[i] - min_val) / bin_width);
      // Handle edge case where value == max_val
      if (bin_index >= num_bins)
        bin_index = num_bins - 1;
      bins[bin_index]++;
    }
  }

  return bins;
=======
double StatisticsEngine::calculate_mean(const double* data, size_t length) {
    if (length == 0) {
        return 0.0;
    }
    
    // Use SIMD if available and data is large enough
    const size_t simd_threshold = 1000;
    
    if (length >= simd_threshold && simd::has_avx2()) {
#ifdef TIMEGRAPH_HAS_AVX2
        return simd::stats::mean_avx2(data, length);
#else
        return simd::scalar::mean_scalar(data, length);
#endif
    } else {
        // Scalar fallback
        return simd::scalar::mean_scalar(data, length);
    }
}

double StatisticsEngine::calculate_std_dev(const double* data, size_t length, double mean) {
    if (length == 0) {
        return 0.0;
    }
    
    // Use SIMD if available and data is large enough
    const size_t simd_threshold = 1000;
    
    if (length >= simd_threshold && simd::has_avx2()) {
#ifdef TIMEGRAPH_HAS_AVX2
        return simd::stats::stddev_avx2(data, length, mean);
#else
        double var = simd::scalar::variance_scalar(data, length, mean);
        return std::sqrt(var);
#endif
    } else {
        // Scalar fallback
        double var = simd::scalar::variance_scalar(data, length, mean);
        return std::sqrt(var);
    }
}

void StatisticsEngine::calculate_min_max(const double* data, size_t length, double& min, double& max) {
    if (length == 0) {
        min = max = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // TODO: SIMD optimization (AVX2/AVX512)
    min = std::numeric_limits<double>::infinity();
    max = -std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            if (data[i] < min) min = data[i];
            if (data[i] > max) max = data[i];
        }
    }
    
    // Handle case where all values are NaN
    if (std::isinf(min)) min = 0.0;
    if (std::isinf(max)) max = 0.0;
}

ColumnStatistics StatisticsEngine::calculate(
    const DataFrame& df,
    const std::string& column_name
) {
    return calculate_range(df, column_name, 0, df.row_count());
}

ColumnStatistics StatisticsEngine::calculate_range(
    const DataFrame& df,
    const std::string& column_name,
    size_t start_index,
    size_t end_index
) {
    ColumnStatistics stats;
    
    // Validate inputs
    if (!df.has_column(column_name)) {
        throw std::runtime_error("Column not found: " + column_name);
    }
    
    if (start_index >= end_index || end_index > df.row_count()) {
        throw std::runtime_error("Invalid index range");
    }
    
    // Get column data
    const double* data = df.get_column_ptr_f64(column_name);
    size_t length = end_index - start_index;
    const double* range_data = data + start_index;
    
    // Count valid values
    stats.count = length;
    stats.valid_count = 0;
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(range_data[i])) {
            stats.valid_count++;
        }
    }
    
    if (stats.valid_count == 0) {
        return stats;  // All NaN, return zeros
    }
    
    // Calculate statistics
    stats.mean = calculate_mean(range_data, length);
    stats.std_dev = calculate_std_dev(range_data, length, stats.mean);
    calculate_min_max(range_data, length, stats.min, stats.max);
    
    // Calculate sum and RMS
    stats.sum = 0.0;
    double sum_squares = 0.0;
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(range_data[i])) {
            stats.sum += range_data[i];
            sum_squares += range_data[i] * range_data[i];
        }
    }
    
    // Calculate RMS
    if (stats.valid_count > 0) {
        stats.rms = std::sqrt(sum_squares / stats.valid_count);
    }
    
    // Calculate peak-to-peak
    stats.peak_to_peak = stats.max - stats.min;
    
    // Calculate median (simple implementation - not optimized)
    std::vector<double> valid_values;
    valid_values.reserve(stats.valid_count);
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(range_data[i])) {
            valid_values.push_back(range_data[i]);
        }
    }
    
    if (!valid_values.empty()) {
        std::nth_element(valid_values.begin(), 
                        valid_values.begin() + valid_values.size() / 2,
                        valid_values.end());
        stats.median = valid_values[valid_values.size() / 2];
    }
    
    return stats;
}

ThresholdStatistics StatisticsEngine::calculate_with_threshold(
    const DataFrame& df,
    const std::string& column_name,
    const std::string& time_column,
    double threshold
) {
    ThresholdStatistics stats;
    
    // First calculate basic statistics
    ColumnStatistics basic_stats = calculate(df, column_name);
    stats.mean = basic_stats.mean;
    stats.std_dev = basic_stats.std_dev;
    stats.min = basic_stats.min;
    stats.max = basic_stats.max;
    stats.median = basic_stats.median;
    stats.sum = basic_stats.sum;
    stats.count = basic_stats.count;
    stats.valid_count = basic_stats.valid_count;
    
    // Set threshold
    stats.threshold = threshold;
    
    // Get data
    const double* data = df.get_column_ptr_f64(column_name);
    const double* time_data = df.get_column_ptr_f64(time_column);
    size_t length = df.row_count();
    
    // Count above/below threshold
    stats.above_count = 0;
    stats.below_count = 0;
    stats.time_above = 0.0;
    stats.time_below = 0.0;
    
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            if (data[i] > threshold) {
                stats.above_count++;
                // Calculate time duration (assuming uniform sampling)
                if (i > 0) {
                    stats.time_above += time_data[i] - time_data[i-1];
                }
            } else {
                stats.below_count++;
                if (i > 0) {
                    stats.time_below += time_data[i] - time_data[i-1];
                }
            }
        }
    }
    
    // Calculate percentages
    if (stats.valid_count > 0) {
        stats.above_percentage = (100.0 * stats.above_count) / stats.valid_count;
        stats.below_percentage = (100.0 * stats.below_count) / stats.valid_count;
    }
    
    return stats;
}

std::vector<ColumnStatistics> StatisticsEngine::calculate_rolling(
    const DataFrame& df,
    const std::string& column_name,
    size_t window_size
) {
    if (window_size == 0) {
        throw std::runtime_error("Window size must be > 0");
    }
    
    size_t length = df.row_count();
    std::vector<ColumnStatistics> results;
    
    if (length < window_size) {
        return results;  // Not enough data
    }
    
    // Calculate rolling statistics
    for (size_t i = 0; i <= length - window_size; ++i) {
        results.push_back(calculate_range(df, column_name, i, i + window_size));
    }
    
    return results;
}

double StatisticsEngine::percentile(
    const DataFrame& df,
    const std::string& column_name,
    double percentile_value
) {
    if (percentile_value < 0.0 || percentile_value > 100.0) {
        throw std::runtime_error("Percentile must be between 0 and 100");
    }
    
    // Get data
    const double* data = df.get_column_ptr_f64(column_name);
    size_t length = df.row_count();
    
    // Collect valid values
    std::vector<double> valid_values;
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            valid_values.push_back(data[i]);
        }
    }
    
    if (valid_values.empty()) {
        return 0.0;
    }
    
    // Sort and find percentile
    std::sort(valid_values.begin(), valid_values.end());
    size_t index = static_cast<size_t>((percentile_value / 100.0) * (valid_values.size() - 1));
    return valid_values[index];
}

std::vector<size_t> StatisticsEngine::histogram(
    const DataFrame& df,
    const std::string& column_name,
    size_t num_bins
) {
    if (num_bins == 0) {
        throw std::runtime_error("Number of bins must be > 0");
    }
    
    // Get min/max
    double min_val, max_val;
    const double* data = df.get_column_ptr_f64(column_name);
    size_t length = df.row_count();
    calculate_min_max(data, length, min_val, max_val);
    
    // Create bins
    std::vector<size_t> bins(num_bins, 0);
    double bin_width = (max_val - min_val) / num_bins;
    
    if (bin_width == 0.0) {
        // All values are the same
        bins[0] = length;
        return bins;
    }
    
    // Fill bins
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            size_t bin_index = static_cast<size_t>((data[i] - min_val) / bin_width);
            // Handle edge case where value == max_val
            if (bin_index >= num_bins) bin_index = num_bins - 1;
            bins[bin_index]++;
        }
    }
    
    return bins;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
}

// ========== MPAI STREAMING STATISTICS IMPLEMENTATION ==========

<<<<<<< HEAD
ColumnStatistics
StatisticsEngine::calculate_streaming(mpai::MpaiReader &reader,
                                      const std::string &column_name,
                                      size_t start_row, size_t row_count) {
  ColumnStatistics stats;

  // Get total row count
  size_t total_rows = reader.get_row_count();

  // Validate range
  if (start_row >= total_rows) {
    return stats; // Empty stats
  }

  // Determine actual row count
  size_t actual_row_count = row_count;
  if (row_count == 0 || start_row + row_count > total_rows) {
    actual_row_count = total_rows - start_row;
  }

  stats.count = actual_row_count;

  // ========================================================================
  // SINGLE-PASS STREAMING CALCULATION (Welford's Algorithm)
  // ========================================================================
  // This calculates mean, variance, min, max, sum, RMS in ONE pass through the
  // data. Benefits:
  // - 2x faster than two-pass algorithm (reads data only once)
  // - Same numerical accuracy
  // - Memory efficient (10k chunk size)
  // ========================================================================

  const size_t chunk_size = 10000;
  size_t rows_processed = 0;

  // Welford's online algorithm variables
  double mean = 0.0;
  double M2 = 0.0; // Sum of squared differences from mean
  double sum = 0.0;
  double sum_squares = 0.0;
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  size_t valid_count = 0;

  while (rows_processed < actual_row_count) {
    size_t current_chunk_size =
        std::min(chunk_size, actual_row_count - rows_processed);

    // Load chunk from MPAI
    std::vector<double> chunk_data = reader.load_column_slice(
        column_name, start_row + rows_processed, current_chunk_size);

    // Process chunk with Welford's algorithm
    for (size_t i = 0; i < chunk_data.size(); ++i) {
      double val = chunk_data[i];
      if (!std::isnan(val) && !std::isinf(val)) {
        valid_count++;

        // Welford's online algorithm for mean and variance
        double delta = val - mean;
        mean += delta / valid_count;
        double delta2 = val - mean;
        M2 += delta * delta2;

        // Sum and sum of squares for RMS
        sum += val;
        sum_squares += val * val;

        // Min/Max tracking
        if (val < min_val)
          min_val = val;
        if (val > max_val)
          max_val = val;
      }
    }

    rows_processed += current_chunk_size;
  }

  stats.valid_count = valid_count;

  if (valid_count == 0) {
    return stats; // All NaN/Inf
  }

  // Final statistics calculation
  stats.mean = mean;
  stats.sum = sum;
  stats.min = min_val;
  stats.max = max_val;
  stats.rms = std::sqrt(sum_squares / valid_count);
  stats.peak_to_peak = max_val - min_val;

  // Standard deviation from Welford's M2
  // variance = M2 / valid_count (population variance)
  // For sample variance, use M2 / (valid_count - 1)
  stats.std_dev = std::sqrt(M2 / valid_count);

  // Note: Median calculation skipped for streaming (requires sorting all data)
  // For large datasets, exact median would require O(n) memory
  stats.median = 0.0;

  return stats;
}

ColumnStatistics StatisticsEngine::calculate_time_range_streaming(
    mpai::MpaiReader &reader, const std::string &column_name,
    const std::string &time_column, double start_time, double end_time) {
  // Load time column to find row indices
  // For efficiency, we could binary search if time is monotonic
  // For now, simple approach: load time column in chunks

  size_t total_rows = reader.get_row_count();
  const size_t chunk_size = 10000;

  size_t start_row = 0;
  size_t end_row = total_rows;
  bool found_start = false;
  bool found_end = false;

  // Find start and end rows
  for (size_t offset = 0; offset < total_rows && !found_end;
       offset += chunk_size) {
    size_t current_chunk_size = std::min(chunk_size, total_rows - offset);
    std::vector<double> time_chunk_data =
        reader.load_column_slice(time_column, offset, current_chunk_size);

    for (size_t i = 0; i < time_chunk_data.size(); ++i) {
      double t = time_chunk_data[i];

      if (!found_start && t >= start_time) {
        start_row = offset + i;
        found_start = true;
      }

      if (found_start && t > end_time) {
        end_row = offset + i;
        found_end = true;
        break;
      }
    }
  }

  // Calculate statistics for the found range
  size_t row_count = end_row - start_row;
  return calculate_streaming(reader, column_name, start_row, row_count);
=======
ColumnStatistics StatisticsEngine::calculate_streaming(
    mpai::MpaiReader& reader,
    const std::string& column_name,
    size_t start_row,
    size_t row_count
) {
    ColumnStatistics stats;
    
    // Get total row count
    size_t total_rows = reader.get_row_count();
    
    // Validate range
    if (start_row >= total_rows) {
        return stats;  // Empty stats
    }
    
    // Determine actual row count
    size_t actual_row_count = row_count;
    if (row_count == 0 || start_row + row_count > total_rows) {
        actual_row_count = total_rows - start_row;
    }
    
    stats.count = actual_row_count;
    
    // ========================================================================
    // SINGLE-PASS STREAMING CALCULATION (Welford's Algorithm)
    // ========================================================================
    // This calculates mean, variance, min, max, sum, RMS in ONE pass through the data.
    // Benefits:
    // - 2x faster than two-pass algorithm (reads data only once)
    // - Same numerical accuracy
    // - Memory efficient (10k chunk size)
    // ========================================================================
    
    const size_t chunk_size = 10000;
    size_t rows_processed = 0;
    
    // Welford's online algorithm variables
    double mean = 0.0;
    double M2 = 0.0;  // Sum of squared differences from mean
    double sum = 0.0;
    double sum_squares = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    size_t valid_count = 0;
    
    while (rows_processed < actual_row_count) {
        size_t current_chunk_size = std::min(chunk_size, actual_row_count - rows_processed);
        
        // Load chunk from MPAI
        std::vector<double> chunk_data = reader.load_column_slice(
            column_name, 
            start_row + rows_processed, 
            current_chunk_size
        );
        
        // Process chunk with Welford's algorithm
        for (size_t i = 0; i < chunk_data.size(); ++i) {
            double val = chunk_data[i];
            if (!std::isnan(val) && !std::isinf(val)) {
                valid_count++;
                
                // Welford's online algorithm for mean and variance
                double delta = val - mean;
                mean += delta / valid_count;
                double delta2 = val - mean;
                M2 += delta * delta2;
                
                // Sum and sum of squares for RMS
                sum += val;
                sum_squares += val * val;
                
                // Min/Max tracking
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        
        rows_processed += current_chunk_size;
    }
    
    stats.valid_count = valid_count;
    
    if (valid_count == 0) {
        return stats;  // All NaN/Inf
    }
    
    // Final statistics calculation
    stats.mean = mean;
    stats.sum = sum;
    stats.min = min_val;
    stats.max = max_val;
    stats.rms = std::sqrt(sum_squares / valid_count);
    stats.peak_to_peak = max_val - min_val;
    
    // Standard deviation from Welford's M2
    // variance = M2 / valid_count (population variance)
    // For sample variance, use M2 / (valid_count - 1)
    stats.std_dev = std::sqrt(M2 / valid_count);
    
    // Note: Median calculation skipped for streaming (requires sorting all data)
    // For large datasets, exact median would require O(n) memory
    stats.median = 0.0;
    
    return stats;
}

ColumnStatistics StatisticsEngine::calculate_time_range_streaming(
    mpai::MpaiReader& reader,
    const std::string& column_name,
    const std::string& time_column,
    double start_time,
    double end_time
) {
    // Load time column to find row indices
    // For efficiency, we could binary search if time is monotonic
    // For now, simple approach: load time column in chunks
    
    size_t total_rows = reader.get_row_count();
    const size_t chunk_size = 10000;
    
    size_t start_row = 0;
    size_t end_row = total_rows;
    bool found_start = false;
    bool found_end = false;
    
    // Find start and end rows
    for (size_t offset = 0; offset < total_rows && !found_end; offset += chunk_size) {
        size_t current_chunk_size = std::min(chunk_size, total_rows - offset);
        std::vector<double> time_chunk_data = reader.load_column_slice(time_column, offset, current_chunk_size);
        
        for (size_t i = 0; i < time_chunk_data.size(); ++i) {
            double t = time_chunk_data[i];
            
            if (!found_start && t >= start_time) {
                start_row = offset + i;
                found_start = true;
            }
            
            if (found_start && t > end_time) {
                end_row = offset + i;
                found_end = true;
                break;
            }
        }
    }
    
    // Calculate statistics for the found range
    size_t row_count = end_row - start_row;
    return calculate_streaming(reader, column_name, start_row, row_count);
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
}

// ========== ARROW COMPUTE IMPLEMENTATIONS ==========

<<<<<<< HEAD
ColumnStatistics
StatisticsEngine::calculate_arrow(const std::vector<double> &data) {
#ifdef HAVE_ARROW
  // TEMPORARY FIX: Arrow Compute functions not available at runtime
  // Fallback to native SIMD which works perfectly
  // TODO: Fix Arrow Compute library loading issue

  if (false) { // Force native SIMD path DISABLED - Enable Arrow Compute
    // Use native SIMD - fast and reliable
    ColumnStatistics stats;
    stats.count = data.size();
    stats.valid_count = data.size();
    stats.mean = calculate_mean(data.data(), data.size());
    stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
    calculate_min_max(data.data(), data.size(), stats.min, stats.max);
    stats.peak_to_peak = stats.max - stats.min;

    // RMS calculation
    double sum_sq = 0.0;
    for (double val : data) {
      sum_sq += val * val;
    }
    stats.rms = std::sqrt(sum_sq / data.size());
    stats.sum = stats.mean * data.size();

    return stats;
  }

  try {
    // NOTE: Arrow Compute path disabled due to runtime issues
    // Zero-copy wrap as Arrow array
    printf("[DEBUG] Using ARROW COMPUTE for statistics (Zero-Copy)\n");
    auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);

    ColumnStatistics stats;
    stats.count = data.size();
    stats.valid_count = data.size();

    // Execute context for Arrow compute
    arrow::compute::ExecContext ctx;

    // Mean
    auto mean_result =
        arrow::compute::CallFunction("mean", {arrow_array}, &ctx);
    if (mean_result.ok()) {
      auto mean_scalar =
          mean_result.ValueOrDie().scalar_as<arrow::DoubleScalar>();
      stats.mean = mean_scalar.value;
      stats.sum = stats.mean * data.size();
    }

    // Stddev
    auto stddev_result =
        arrow::compute::CallFunction("stddev", {arrow_array}, &ctx);
    if (stddev_result.ok()) {
      auto stddev_scalar =
          stddev_result.ValueOrDie().scalar_as<arrow::DoubleScalar>();
      stats.std_dev = stddev_scalar.value;
    }

    // Min/Max
    auto minmax_result =
        arrow::compute::CallFunction("min_max", {arrow_array}, &ctx);
    if (minmax_result.ok()) {
      auto minmax_scalar =
          minmax_result.ValueOrDie().scalar_as<arrow::StructScalar>();
      stats.min =
          std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[0])
              ->value;
      stats.max =
          std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[1])
              ->value;
      stats.peak_to_peak = stats.max - stats.min;
    }

    // RMS (manual calculation, Arrow doesn't have built-in RMS)
    double sum_sq = 0.0;
    for (double val : data) {
      sum_sq += val * val;
    }
    stats.rms = std::sqrt(sum_sq / data.size());

    // Median (optional, expensive)
    stats.median = 0.0; // Skip for performance

    return stats;

  } catch (const std::exception &e) {
    // Arrow compute failed, fallback to native
    printf("[DEBUG] Arrow compute FAILED: %s. Falling back to Native SIMD.\n",
           e.what());
    ColumnStatistics stats;
    stats.count = data.size();
    stats.valid_count = data.size();
    stats.mean = calculate_mean(data.data(), data.size());
    stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
    calculate_min_max(data.data(), data.size(), stats.min, stats.max);
    stats.peak_to_peak = stats.max - stats.min;

    double sum_sq = 0.0;
    for (double val : data) {
      sum_sq += val * val;
    }
    stats.rms = std::sqrt(sum_sq / data.size());
    stats.sum = stats.mean * data.size();

    return stats;
  }

#else
  // Arrow not available, use native implementation
  ColumnStatistics stats;
  stats.count = data.size();
  stats.valid_count = data.size();
  stats.mean = calculate_mean(data.data(), data.size());
  stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
  calculate_min_max(data.data(), data.size(), stats.min, stats.max);
  stats.peak_to_peak = stats.max - stats.min;

  double sum_sq = 0.0;
  for (double val : data) {
    sum_sq += val * val;
  }
  stats.rms = std::sqrt(sum_sq / data.size());
  stats.sum = stats.mean * data.size();

  return stats;
#endif
}

ColumnStatistics
StatisticsEngine::calculate_chunk_arrow(const std::vector<double> &chunk_data) {
  // Just delegate to calculate_arrow (same implementation)
  return calculate_arrow(chunk_data);
}

double StatisticsEngine::mean_arrow(const std::vector<double> &data) {
#ifdef HAVE_ARROW
  if (!arrow_utils::is_arrow_available() || data.empty()) {
    return calculate_mean(data.data(), data.size());
  }

  try {
    auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
    arrow::compute::ExecContext ctx;

    auto result = arrow::compute::CallFunction("mean", {arrow_array}, &ctx);
    if (result.ok()) {
      return result.ValueOrDie().scalar_as<arrow::DoubleScalar>().value;
    }
  } catch (...) {
    // Fallback
  }
#endif
  return calculate_mean(data.data(), data.size());
}

double StatisticsEngine::stddev_arrow(const std::vector<double> &data) {
#ifdef HAVE_ARROW
  if (!arrow_utils::is_arrow_available() || data.empty()) {
    double mean = calculate_mean(data.data(), data.size());
    return calculate_std_dev(data.data(), data.size(), mean);
  }

  try {
    auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
    arrow::compute::ExecContext ctx;

    auto result = arrow::compute::CallFunction("stddev", {arrow_array}, &ctx);
    if (result.ok()) {
      return result.ValueOrDie().scalar_as<arrow::DoubleScalar>().value;
    }
  } catch (...) {
    // Fallback
  }
#endif
  double mean = calculate_mean(data.data(), data.size());
  return calculate_std_dev(data.data(), data.size(), mean);
}

void StatisticsEngine::minmax_arrow(const std::vector<double> &data,
                                    double &min, double &max) {
#ifdef HAVE_ARROW
  if (!arrow_utils::is_arrow_available() || data.empty()) {
    calculate_min_max(data.data(), data.size(), min, max);
    return;
  }

  try {
    auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
    arrow::compute::ExecContext ctx;

    auto result = arrow::compute::CallFunction("min_max", {arrow_array}, &ctx);
    if (result.ok()) {
      auto minmax_scalar = result.ValueOrDie().scalar_as<arrow::StructScalar>();
      min =
          std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[0])
              ->value;
      max =
          std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[1])
              ->value;
      return;
    }
  } catch (...) {
    // Fallback
  }
#endif
  calculate_min_max(data.data(), data.size(), min, max);
}

} // namespace timegraph
=======
ColumnStatistics StatisticsEngine::calculate_arrow(const std::vector<double>& data) {
#ifdef HAVE_ARROW
    // TEMPORARY FIX: Arrow Compute functions not available at runtime
    // Fallback to native SIMD which works perfectly
    // TODO: Fix Arrow Compute library loading issue
    
    if (false) {  // Force native SIMD path DISABLED - Enable Arrow Compute
        // Use native SIMD - fast and reliable
        ColumnStatistics stats;
        stats.count = data.size();
        stats.valid_count = data.size();
        stats.mean = calculate_mean(data.data(), data.size());
        stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
        calculate_min_max(data.data(), data.size(), stats.min, stats.max);
        stats.peak_to_peak = stats.max - stats.min;
        
        // RMS calculation
        double sum_sq = 0.0;
        for (double val : data) {
            sum_sq += val * val;
        }
        stats.rms = std::sqrt(sum_sq / data.size());
        stats.sum = stats.mean * data.size();
        
        return stats;
    }
    
    try {
        // NOTE: Arrow Compute path disabled due to runtime issues
        // Zero-copy wrap as Arrow array
        printf("[DEBUG] Using ARROW COMPUTE for statistics (Zero-Copy)\n");
        auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
        
        ColumnStatistics stats;
        stats.count = data.size();
        stats.valid_count = data.size();
        
        // Execute context for Arrow compute
        arrow::compute::ExecContext ctx;
        
        // Mean
        auto mean_result = arrow::compute::CallFunction("mean", {arrow_array}, &ctx);
        if (mean_result.ok()) {
            auto mean_scalar = mean_result.ValueOrDie().scalar_as<arrow::DoubleScalar>();
            stats.mean = mean_scalar.value;
            stats.sum = stats.mean * data.size();
        }
        
        // Stddev
        auto stddev_result = arrow::compute::CallFunction("stddev", {arrow_array}, &ctx);
        if (stddev_result.ok()) {
            auto stddev_scalar = stddev_result.ValueOrDie().scalar_as<arrow::DoubleScalar>();
            stats.std_dev = stddev_scalar.value;
        }
        
        // Min/Max
        auto minmax_result = arrow::compute::CallFunction("min_max", {arrow_array}, &ctx);
        if (minmax_result.ok()) {
            auto minmax_scalar = minmax_result.ValueOrDie().scalar_as<arrow::StructScalar>();
            stats.min = std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[0])->value;
            stats.max = std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[1])->value;
            stats.peak_to_peak = stats.max - stats.min;
        }
        
        // RMS (manual calculation, Arrow doesn't have built-in RMS)
        double sum_sq = 0.0;
        for (double val : data) {
            sum_sq += val * val;
        }
        stats.rms = std::sqrt(sum_sq / data.size());
        
        // Median (optional, expensive)
        stats.median = 0.0;  // Skip for performance
        
        return stats;
        
    } catch (const std::exception& e) {
        // Arrow compute failed, fallback to native
        printf("[DEBUG] Arrow compute FAILED: %s. Falling back to Native SIMD.\n", e.what());
        ColumnStatistics stats;
        stats.count = data.size();
        stats.valid_count = data.size();
        stats.mean = calculate_mean(data.data(), data.size());
        stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
        calculate_min_max(data.data(), data.size(), stats.min, stats.max);
        stats.peak_to_peak = stats.max - stats.min;
        
        double sum_sq = 0.0;
        for (double val : data) {
            sum_sq += val * val;
        }
        stats.rms = std::sqrt(sum_sq / data.size());
        stats.sum = stats.mean * data.size();
        
        return stats;
    }
    
#else
    // Arrow not available, use native implementation
    ColumnStatistics stats;
    stats.count = data.size();
    stats.valid_count = data.size();
    stats.mean = calculate_mean(data.data(), data.size());
    stats.std_dev = calculate_std_dev(data.data(), data.size(), stats.mean);
    calculate_min_max(data.data(), data.size(), stats.min, stats.max);
    stats.peak_to_peak = stats.max - stats.min;
    
    double sum_sq = 0.0;
    for (double val : data) {
        sum_sq += val * val;
    }
    stats.rms = std::sqrt(sum_sq / data.size());
    stats.sum = stats.mean * data.size();
    
    return stats;
#endif
}

ColumnStatistics StatisticsEngine::calculate_chunk_arrow(const std::vector<double>& chunk_data) {
    // Just delegate to calculate_arrow (same implementation)
    return calculate_arrow(chunk_data);
}

double StatisticsEngine::mean_arrow(const std::vector<double>& data) {
#ifdef HAVE_ARROW
    if (!arrow_utils::is_arrow_available() || data.empty()) {
        return calculate_mean(data.data(), data.size());
    }
    
    try {
        auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
        arrow::compute::ExecContext ctx;
        
        auto result = arrow::compute::CallFunction("mean", {arrow_array}, &ctx);
        if (result.ok()) {
            return result.ValueOrDie().scalar_as<arrow::DoubleScalar>().value;
        }
    } catch (...) {
        // Fallback
    }
#endif
    return calculate_mean(data.data(), data.size());
}

double StatisticsEngine::stddev_arrow(const std::vector<double>& data) {
#ifdef HAVE_ARROW
    if (!arrow_utils::is_arrow_available() || data.empty()) {
        double mean = calculate_mean(data.data(), data.size());
        return calculate_std_dev(data.data(), data.size(), mean);
    }
    
    try {
        auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
        arrow::compute::ExecContext ctx;
        
        auto result = arrow::compute::CallFunction("stddev", {arrow_array}, &ctx);
        if (result.ok()) {
            return result.ValueOrDie().scalar_as<arrow::DoubleScalar>().value;
        }
    } catch (...) {
        // Fallback
    }
#endif
    double mean = calculate_mean(data.data(), data.size());
    return calculate_std_dev(data.data(), data.size(), mean);
}

void StatisticsEngine::minmax_arrow(const std::vector<double>& data, double& min, double& max) {
#ifdef HAVE_ARROW
    if (!arrow_utils::is_arrow_available() || data.empty()) {
        calculate_min_max(data.data(), data.size(), min, max);
        return;
    }
    
    try {
        auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
        arrow::compute::ExecContext ctx;
        
        auto result = arrow::compute::CallFunction("min_max", {arrow_array}, &ctx);
        if (result.ok()) {
            auto minmax_scalar = result.ValueOrDie().scalar_as<arrow::StructScalar>();
            min = std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[0])->value;
            max = std::static_pointer_cast<arrow::DoubleScalar>(minmax_scalar.value[1])->value;
            return;
        }
    } catch (...) {
        // Fallback
    }
#endif
    calculate_min_max(data.data(), data.size(), min, max);
}

} // namespace timegraph

>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
