#pragma once

#include "timegraph/data/dataframe.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cstddef>

namespace timegraph {

/// Filter condition types
enum class FilterType {
    RANGE,      ///< Value in range [min, max]
    GREATER,    ///< Value > threshold
    LESS,       ///< Value < threshold
    EQUAL,      ///< Value == threshold
    NOT_EQUAL   ///< Value != threshold
};

/// Filter operator for combining conditions
enum class FilterOperator {
    AND,  ///< All conditions must be satisfied
    OR    ///< Any condition can be satisfied
};

/// Single filter condition
struct FilterCondition {
    std::string column_name;    ///< Column to filter
    FilterType type;            ///< Filter type
    double min_value;           ///< Minimum value (for RANGE, GREATER)
    double max_value;           ///< Maximum value (for RANGE, LESS)
    double threshold;           ///< Threshold value (for GREATER, LESS, EQUAL)
    FilterOperator op;          ///< Operator for combining with other conditions
    
    FilterCondition()
        : type(FilterType::RANGE)
        , min_value(0.0)
        , max_value(0.0)
        , threshold(0.0)
        , op(FilterOperator::AND)
    {}
};

/// Time segment that satisfies filter conditions
struct TimeSegment {
    double start_time;  ///< Segment start time
    double end_time;    ///< Segment end time
    size_t start_index; ///< Start index in data
    size_t end_index;   ///< End index in data
    
    TimeSegment(double start, double end, size_t start_idx, size_t end_idx)
        : start_time(start), end_time(end)
        , start_index(start_idx), end_index(end_idx)
    {}
};

/// High-performance filter engine for time-series data
class FilterEngine {
public:
    /// Constructor
    FilterEngine() = default;
    
    /// Apply filter conditions to DataFrame and return matching segments
    /// Thread-safe, uses SIMD optimizations when available
    std::vector<TimeSegment> calculate_segments(
        const DataFrame& df,
        const std::string& time_column,
        const std::vector<FilterCondition>& conditions
    );
    
    /// Apply filter and return boolean mask (1 = pass, 0 = fail)
    /// Optimized for large datasets using AVX2/AVX512
    std::vector<bool> calculate_mask(
        const DataFrame& df,
        const std::vector<FilterCondition>& conditions
    );
    
    /// Get filtered DataFrame (only rows that pass filter)
    DataFrame apply_filter(
        const DataFrame& df,
        const std::vector<FilterCondition>& conditions
    );
    
    /// Check if a single value passes filter condition
    static bool check_condition(double value, const FilterCondition& cond);

    // ========== ARROW COMPUTE INTEGRATION (HYBRID) ==========
    /// Calculate mask using Arrow Compute (SIMD-optimized, zero-copy)
    /// If Arrow is available, uses Arrow compute for 8-15x speedup
    /// Falls back to native implementation if Arrow not available
    ///
    /// @param data     Input data (will be wrapped as Arrow array, zero-copy!)
    /// @param cond     Filter condition
    /// @return Boolean mask (true = passes filter)
    std::vector<bool> calculate_mask_arrow(
        const std::vector<double>& data,
        const FilterCondition& cond
    );
    
    /// Apply filter to MPAI chunk using Arrow Compute
    /// Wraps MPAI chunk data as Arrow array (zero-copy) and uses Arrow compute
    ///
    /// @param chunk_data MPAI chunk data (from MpaiReader)
    /// @param conditions Filter conditions
    /// @return Boolean mask for this chunk
    std::vector<bool> apply_filter_to_chunk_arrow(
        const std::vector<double>& chunk_data,
        const std::vector<FilterCondition>& conditions
    );

    // ========== MPAI STREAMING FILTERING (NEW) ==========
    /// Calculate segments from MPAI file using streaming (low RAM)
    /// Loads data in chunks from MpaiReader and never materializes
    /// the full boolean mask in memory.
    ///
    /// @param reader       MPAI reader (binary format)
    /// @param time_column  Name of time column
    /// @param conditions   Filter conditions
    /// @param start_row    First row to process (default: 0)
    /// @param row_count    Number of rows to process (0 = until end)
    std::vector<TimeSegment> calculate_streaming(
        mpai::MpaiReader& reader,
        const std::string& time_column,
        const std::vector<FilterCondition>& conditions,
        uint64_t start_row = 0,
        uint64_t row_count = 0
    );
    
private:
    /// Apply single condition and return mask
    std::vector<bool> apply_single_condition(
        const DataFrame& df,
        const FilterCondition& cond
    );
    
    /// Combine masks using AND/OR logic
    std::vector<bool> combine_masks(
        const std::vector<bool>& mask1,
        const std::vector<bool>& mask2,
        FilterOperator op
    );
    
    /// Convert boolean mask to time segments
    std::vector<TimeSegment> mask_to_segments(
        const std::vector<bool>& mask,
        const double* time_data,
        size_t length
    );
};

} // namespace timegraph

