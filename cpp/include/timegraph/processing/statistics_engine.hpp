#pragma once

#include "timegraph/data/dataframe.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include <string>
#include <vector>
#include <optional>
#include <cstddef>

namespace timegraph {

/// Statistical results for a data column
struct ColumnStatistics {
    double mean;        ///< Arithmetic mean
    double std_dev;     ///< Standard deviation
    double min;         ///< Minimum value
    double max;         ///< Maximum value
    double median;      ///< Median value
    double sum;         ///< Sum of all values
    double rms;         ///< Root mean square
    double peak_to_peak; ///< Peak-to-peak value
    size_t count;       ///< Number of values
    size_t valid_count; ///< Number of non-NaN values
    
    ColumnStatistics()
        : mean(0.0), std_dev(0.0), min(0.0), max(0.0)
        , median(0.0), sum(0.0), rms(0.0), peak_to_peak(0.0)
        , count(0), valid_count(0)
    {}
};

/// Statistical results with threshold analysis
struct ThresholdStatistics : public ColumnStatistics {
    double threshold;           ///< Threshold value
    size_t above_count;         ///< Count of values above threshold
    size_t below_count;         ///< Count of values below threshold
    double above_percentage;    ///< Percentage above threshold
    double below_percentage;    ///< Percentage below threshold
    double time_above;          ///< Total time above threshold (seconds)
    double time_below;          ///< Total time below threshold (seconds)
    
    ThresholdStatistics()
        : threshold(0.0), above_count(0), below_count(0)
        , above_percentage(0.0), below_percentage(0.0)
        , time_above(0.0), time_below(0.0)
    {}
};

/// High-performance statistics engine
class StatisticsEngine {
public:
    /// Constructor
    StatisticsEngine() = default;
    
    /// Calculate basic statistics for a column
    /// Uses SIMD optimizations (AVX2/AVX512) when available
    static ColumnStatistics calculate(
        const DataFrame& df,
        const std::string& column_name
    );
    
    /// Calculate statistics for a column with index range
    static ColumnStatistics calculate_range(
        const DataFrame& df,
        const std::string& column_name,
        size_t start_index,
        size_t end_index
    );
    
    /// Calculate statistics with threshold analysis
    static ThresholdStatistics calculate_with_threshold(
        const DataFrame& df,
        const std::string& column_name,
        const std::string& time_column,
        double threshold
    );
    
    /// Calculate rolling statistics (moving window)
    static std::vector<ColumnStatistics> calculate_rolling(
        const DataFrame& df,
        const std::string& column_name,
        size_t window_size
    );
    
    /// Calculate percentile value
    static double percentile(
        const DataFrame& df,
        const std::string& column_name,
        double percentile_value  // 0.0 to 100.0
    );
    
    /// Calculate histogram bins
    static std::vector<size_t> histogram(
        const DataFrame& df,
        const std::string& column_name,
        size_t num_bins
    );
    
    // ========== ARROW COMPUTE STATISTICS (HYBRID) ==========
    
    /// Calculate statistics using Arrow Compute (SIMD-optimized, zero-copy)
    /// 20-30x faster than Python NumPy for large datasets
    /// Falls back to native SIMD if Arrow not available
    ///
    /// @param data Input data (will be wrapped as Arrow array, zero-copy!)
    /// @return Statistics struct with mean, std, min, max, rms
    static ColumnStatistics calculate_arrow(
        const std::vector<double>& data
    );
    
    /// Calculate statistics for MPAI chunk using Arrow Compute
    /// Wraps MPAI chunk data as Arrow array (zero-copy) and uses Arrow compute
    ///
    /// @param chunk_data MPAI chunk data (from MpaiReader)
    /// @return Statistics for this chunk
    static ColumnStatistics calculate_chunk_arrow(
        const std::vector<double>& chunk_data
    );
    
    /// Fast mean calculation using Arrow Compute
    /// @param data Input data
    /// @return Mean value
    static double mean_arrow(const std::vector<double>& data);
    
    /// Fast standard deviation using Arrow Compute
    /// @param data Input data
    /// @return Standard deviation
    static double stddev_arrow(const std::vector<double>& data);
    
    /// Fast min/max using Arrow Compute
    /// @param data Input data
    /// @param[out] min Minimum value
    /// @param[out] max Maximum value
    static void minmax_arrow(const std::vector<double>& data, double& min, double& max);
    
    // ========== MPAI STREAMING STATISTICS (NEW) ==========
    
    /// Calculate statistics from MPAI file (streaming, low RAM)
    /// Uses chunk-based loading to minimize memory usage
    static ColumnStatistics calculate_streaming(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        size_t start_row = 0,
        size_t row_count = 0  // 0 = all rows
    );
    
    /// Calculate statistics for time range (MPAI streaming)
    static ColumnStatistics calculate_time_range_streaming(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        const std::string& time_column,
        double start_time,
        double end_time
    );
    
    // ========== LOW-LEVEL STATISTICS (PUBLIC FOR UTILITY) ==========
    
    /// Fast mean calculation (SIMD optimized)
    static double calculate_mean(const double* data, size_t length);
    
    /// Fast std dev calculation (SIMD optimized)
    static double calculate_std_dev(const double* data, size_t length, double mean);
    
    /// Fast min/max calculation (SIMD optimized)
    static void calculate_min_max(const double* data, size_t length, double& min, double& max);
};

} // namespace timegraph

