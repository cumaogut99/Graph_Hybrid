#include "timegraph/processing/limit_violation_engine.hpp"
#include "timegraph/processing/simd_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace timegraph {

LimitViolationResult LimitViolationEngine::calculate_violations_streaming(
    mpai::MpaiReader& reader,
    const std::string& signal_name,
    const std::string& time_column,
    double warning_min,
    double warning_max,
    uint64_t start_row,
    uint64_t row_count
) {
    LimitViolationResult result;
    
    // Get total row count from reader
    const uint64_t actual_row_count = (row_count == 0)
        ? reader.get_header().row_count - start_row
        : std::min(row_count, reader.get_header().row_count - start_row);
    
    if (actual_row_count == 0) {
        return result;
    }
    
    // Process data in chunks (1M rows at a time, aligned with MPAI format)
    const uint64_t chunk_size = 1000000;
    uint64_t rows_processed = 0;
    
    // Track global violation state across chunks
    std::vector<uint64_t> all_violation_indices;
    std::vector<double> all_time_data;
    std::vector<double> all_signal_data;
    
    // Reserve space (avoid reallocations)
    all_violation_indices.reserve(actual_row_count / 10);  // Assume ~10% violations
    all_time_data.reserve(actual_row_count / 10);
    all_signal_data.reserve(actual_row_count / 10);
    
    while (rows_processed < actual_row_count) {
        const uint64_t current_chunk_size = std::min(
            chunk_size,
            actual_row_count - rows_processed
        );
        
        const uint64_t chunk_start_row = start_row + rows_processed;
        
        // Load signal and time data for this chunk
        std::vector<double> signal_chunk;
        std::vector<double> time_chunk;
        
        try {
            signal_chunk = reader.load_column_slice(
                signal_name,
                chunk_start_row,
                static_cast<uint64_t>(current_chunk_size)
            );
            time_chunk = reader.load_column_slice(
                time_column,
                chunk_start_row,
                static_cast<uint64_t>(current_chunk_size)
            );
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Failed to load data for violation detection: ") + e.what()
            );
        }
        
        if (signal_chunk.empty() || time_chunk.empty()) {
            break;
        }
        
        const size_t local_size = signal_chunk.size();
        result.total_data_points += local_size;
        
        // Find violations in this chunk using SIMD
        const double* signal_ptr = signal_chunk.data();
        
        // Use SIMD for large chunks
        if (local_size >= 1000 && simd::has_avx2()) {
            // Allocate mask for SIMD
            std::vector<uint8_t> violation_mask(local_size, 0);
            
            // Check both min and max violations using SIMD
            #ifdef TIMEGRAPH_HAS_AVX2
            // Check min violations: signal < warning_min
            std::vector<uint8_t> min_mask(local_size, 0);
            simd::filter::apply_less_avx2(signal_ptr, local_size, warning_min, min_mask.data());
            
            // Check max violations: signal > warning_max
            std::vector<uint8_t> max_mask(local_size, 0);
            simd::filter::apply_greater_avx2(signal_ptr, local_size, warning_max, max_mask.data());
            
            // Combine masks (OR operation)
            for (size_t i = 0; i < local_size; ++i) {
                violation_mask[i] = (min_mask[i] != 0) || (max_mask[i] != 0);
            }
            #else
            // Scalar fallback
            for (size_t i = 0; i < local_size; ++i) {
                double val = signal_ptr[i];
                if (std::isnan(val)) continue;
                violation_mask[i] = (val < warning_min) || (val > warning_max);
            }
            #endif
            
            // Collect violation indices and data
            for (size_t i = 0; i < local_size; ++i) {
                if (violation_mask[i]) {
                    uint64_t global_idx = chunk_start_row + i;
                    all_violation_indices.push_back(global_idx);
                    all_time_data.push_back(time_chunk[i]);
                    all_signal_data.push_back(signal_chunk[i]);
                    
                    // Track min/max violations
                    double val = signal_chunk[i];
                    if (result.total_violation_points == 0) {
                        result.min_violation_value = val;
                        result.max_violation_value = val;
                    } else {
                        if (val < result.min_violation_value) {
                            result.min_violation_value = val;
                        }
                        if (val > result.max_violation_value) {
                            result.max_violation_value = val;
                        }
                    }
                    result.total_violation_points++;
                }
            }
        } else {
            // Scalar path for small chunks
            for (size_t i = 0; i < local_size; ++i) {
                double val = signal_ptr[i];
                if (std::isnan(val)) continue;
                
                if (val < warning_min || val > warning_max) {
                    uint64_t global_idx = chunk_start_row + i;
                    all_violation_indices.push_back(global_idx);
                    all_time_data.push_back(time_chunk[i]);
                    all_signal_data.push_back(val);
                    
                    // Track min/max violations
                    if (result.total_violation_points == 0) {
                        result.min_violation_value = val;
                        result.max_violation_value = val;
                    } else {
                        if (val < result.min_violation_value) {
                            result.min_violation_value = val;
                        }
                        if (val > result.max_violation_value) {
                            result.max_violation_value = val;
                        }
                    }
                    result.total_violation_points++;
                }
            }
        }
        
        rows_processed += current_chunk_size;
    }
    
    // Group consecutive violations into segments
    if (!all_violation_indices.empty()) {
        result.violations = group_consecutive_violations(
            all_violation_indices,
            all_time_data.data(),
            all_signal_data.data(),
            all_violation_indices.size()
        );
    }
    
    return result;
}

LimitViolationResult LimitViolationEngine::calculate_violations_arrays(
    const double* signal_data,
    const double* time_data,
    size_t data_length,
    double warning_min,
    double warning_max
) {
    LimitViolationResult result;
    result.total_data_points = data_length;
    
    if (data_length == 0) {
        return result;
    }
    
    std::vector<uint64_t> violation_indices;
    violation_indices.reserve(data_length / 10);  // Assume ~10% violations
    
    // Use SIMD for large arrays
    if (data_length >= 1000 && simd::has_avx2()) {
        std::vector<uint8_t> violation_mask(data_length, 0);
        
        #ifdef TIMEGRAPH_HAS_AVX2
        // Check min violations
        std::vector<uint8_t> min_mask(data_length, 0);
        simd::filter::apply_less_avx2(signal_data, data_length, warning_min, min_mask.data());
        
        // Check max violations
        std::vector<uint8_t> max_mask(data_length, 0);
        simd::filter::apply_greater_avx2(signal_data, data_length, warning_max, max_mask.data());
        
        // Combine masks
        for (size_t i = 0; i < data_length; ++i) {
            violation_mask[i] = (min_mask[i] != 0) || (max_mask[i] != 0);
        }
        #else
        // Scalar fallback
        for (size_t i = 0; i < data_length; ++i) {
            double val = signal_data[i];
            if (std::isnan(val)) continue;
            violation_mask[i] = (val < warning_min) || (val > warning_max);
        }
        #endif
        
        // Collect violations
        for (size_t i = 0; i < data_length; ++i) {
            if (violation_mask[i]) {
                violation_indices.push_back(i);
                double val = signal_data[i];
                if (result.total_violation_points == 0) {
                    result.min_violation_value = val;
                    result.max_violation_value = val;
                } else {
                    if (val < result.min_violation_value) {
                        result.min_violation_value = val;
                    }
                    if (val > result.max_violation_value) {
                        result.max_violation_value = val;
                    }
                }
                result.total_violation_points++;
            }
        }
    } else {
        // Scalar path
        for (size_t i = 0; i < data_length; ++i) {
            double val = signal_data[i];
            if (std::isnan(val)) continue;
            
            if (val < warning_min || val > warning_max) {
                violation_indices.push_back(i);
                if (result.total_violation_points == 0) {
                    result.min_violation_value = val;
                    result.max_violation_value = val;
                } else {
                    if (val < result.min_violation_value) {
                        result.min_violation_value = val;
                    }
                    if (val > result.max_violation_value) {
                        result.max_violation_value = val;
                    }
                }
                result.total_violation_points++;
            }
        }
    }
    
    // Group into segments
    if (!violation_indices.empty()) {
        result.violations = group_consecutive_violations(
            violation_indices,
            time_data,
            signal_data,
            data_length
        );
    }
    
    return result;
}

std::vector<ViolationSegment> LimitViolationEngine::group_consecutive_violations(
    const std::vector<uint64_t>& violation_indices,
    const double* time_data,
    const double* signal_data,
    size_t data_length
) {
    std::vector<ViolationSegment> segments;
    
    if (violation_indices.empty()) {
        return segments;
    }
    
    // Group consecutive indices
    // We iterate through the vector indices (0 to size-1)
    // violation_indices[i] gives the global row index
    size_t seg_start_vec_idx = 0;
    
    double min_val = signal_data[0];
    double max_val = signal_data[0];
    
    for (size_t i = 1; i < violation_indices.size(); ++i) {
        uint64_t curr_global_idx = violation_indices[i];
        uint64_t prev_global_idx = violation_indices[i-1];
        
        // Check if consecutive (difference is 1)
        if (curr_global_idx == prev_global_idx + 1) {
            // Continue current segment
            double val = signal_data[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        } else {
            // Gap detected - close current segment
            size_t seg_end_vec_idx = i - 1;
            
            segments.emplace_back(
                violation_indices[seg_start_vec_idx], // start_index (global)
                violation_indices[seg_end_vec_idx],   // end_index (global)
                time_data[seg_start_vec_idx],         // start_time
                time_data[seg_end_vec_idx],           // end_time
                min_val,
                max_val
            );
            
            // Start new segment
            seg_start_vec_idx = i;
            min_val = signal_data[i];
            max_val = signal_data[i];
        }
    }
    
    // Add last segment
    size_t last_vec_idx = violation_indices.size() - 1;
    segments.emplace_back(
        violation_indices[seg_start_vec_idx],
        violation_indices[last_vec_idx],
        time_data[seg_start_vec_idx],
        time_data[last_vec_idx],
        min_val,
        max_val
    );
    
    return segments;
}

} // namespace timegraph

