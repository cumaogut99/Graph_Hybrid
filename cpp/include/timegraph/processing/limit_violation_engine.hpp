#pragma once

#include "timegraph/data/mpai_reader.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace timegraph {

/// Limit violation segment (start_idx, end_idx)
struct ViolationSegment {
    uint64_t start_index;
    uint64_t end_index;
    double start_time;
    double end_time;
    double min_value;  // Minimum value in this segment
    double max_value;  // Maximum value in this segment
    
    ViolationSegment(uint64_t start_idx, uint64_t end_idx, 
                     double start_t, double end_t,
                     double min_val, double max_val)
        : start_index(start_idx), end_index(end_idx)
        , start_time(start_t), end_time(end_t)
        , min_value(min_val), max_value(max_val)
    {}
};

/// Limit violation result
struct LimitViolationResult {
    std::vector<ViolationSegment> violations;  // List of violation segments
    uint64_t total_violation_points;           // Total number of violation points
    uint64_t total_data_points;                // Total data points checked
    double min_violation_value;                // Minimum value that violated
    double max_violation_value;                // Maximum value that violated
    
    LimitViolationResult()
        : total_violation_points(0)
        , total_data_points(0)
        , min_violation_value(0.0)
        , max_violation_value(0.0)
    {}
};

/// High-performance limit violation detector
/// Uses SIMD optimizations and streaming for large datasets
class LimitViolationEngine {
public:
    /// Constructor
    LimitViolationEngine() = default;
    
    /// Find all violations in MPAI data using streaming (low RAM)
    /// This is the CRITICAL function - it MUST check ALL data points
    /// to ensure no violations are missed.
    ///
    /// @param reader        MPAI reader (binary format)
    /// @param signal_name   Name of signal column to check
    /// @param time_column   Name of time column
    /// @param warning_min   Minimum warning threshold
    /// @param warning_max   Maximum warning threshold
    /// @param start_row     First row to process (default: 0)
    /// @param row_count     Number of rows to process (0 = until end)
    /// @return Violation result with all violation segments
    LimitViolationResult calculate_violations_streaming(
        mpai::MpaiReader& reader,
        const std::string& signal_name,
        const std::string& time_column,
        double warning_min,
        double warning_max,
        uint64_t start_row = 0,
        uint64_t row_count = 0
    );
    
    /// Find violations in already-loaded NumPy arrays (for small datasets)
    /// @param signal_data   Signal values
    /// @param time_data     Time values
    /// @param data_length   Number of data points
    /// @param warning_min   Minimum warning threshold
    /// @param warning_max   Maximum warning threshold
    /// @return Violation result with all violation segments
    LimitViolationResult calculate_violations_arrays(
        const double* signal_data,
        const double* time_data,
        size_t data_length,
        double warning_min,
        double warning_max
    );

private:
    /// Group consecutive violation indices into segments
    std::vector<ViolationSegment> group_consecutive_violations(
        const std::vector<uint64_t>& violation_indices,
        const double* time_data,
        const double* signal_data,
        size_t data_length
    );
};

} // namespace timegraph

