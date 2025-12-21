#pragma once

#include <cstddef>
#include <cstdint>

// SIMD intrinsics
#if defined(__AVX2__)
    #include <immintrin.h>
    #define TIMEGRAPH_HAS_AVX2 1
#elif defined(__SSE4_2__)
    #include <nmmintrin.h>
    #define TIMEGRAPH_HAS_SSE42 1
#endif

namespace timegraph {
namespace simd {

/// Check if AVX2 is available at runtime
bool has_avx2();

/// Check if SSE4.2 is available at runtime
bool has_sse42();

/// SIMD-optimized operations for filter masks
namespace filter {

#ifdef TIMEGRAPH_HAS_AVX2
    /// Apply RANGE filter using AVX2 (value >= min && value <= max)
    /// Returns number of elements that pass the filter
    size_t apply_range_avx2(
        const double* data,
        size_t length,
        double min_val,
        double max_val,
        uint8_t* mask  // Output: 1 = pass, 0 = fail
    );
    
    /// Apply GREATER filter using AVX2 (value > threshold)
    size_t apply_greater_avx2(
        const double* data,
        size_t length,
        double threshold,
        uint8_t* mask
    );
    
    /// Apply LESS filter using AVX2 (value < threshold)
    size_t apply_less_avx2(
        const double* data,
        size_t length,
        double threshold,
        uint8_t* mask
    );
    
    /// Combine two masks using AND operation (AVX2)
    void combine_and_avx2(
        const uint8_t* mask1,
        const uint8_t* mask2,
        size_t length,
        uint8_t* result
    );
    
    /// Combine two masks using OR operation (AVX2)
    void combine_or_avx2(
        const uint8_t* mask1,
        const uint8_t* mask2,
        size_t length,
        uint8_t* result
    );
#endif

} // namespace filter

/// SIMD-optimized operations for statistics
namespace stats {

#ifdef TIMEGRAPH_HAS_AVX2
    /// Calculate sum using AVX2
    double sum_avx2(const double* data, size_t length);
    
    /// Calculate mean using AVX2
    double mean_avx2(const double* data, size_t length);
    
    /// Calculate min using AVX2
    double min_avx2(const double* data, size_t length);
    
    /// Calculate max using AVX2
    double max_avx2(const double* data, size_t length);
    
    /// Calculate sum of squares using AVX2 (for variance/std)
    double sum_squares_avx2(const double* data, size_t length, double mean);
    
    /// Calculate variance using AVX2
    double variance_avx2(const double* data, size_t length, double mean);
    
    /// Calculate standard deviation using AVX2
    double stddev_avx2(const double* data, size_t length, double mean);
    
    /// Count values above threshold using AVX2
    size_t count_above_avx2(const double* data, size_t length, double threshold);
    
    /// Count values below threshold using AVX2
    size_t count_below_avx2(const double* data, size_t length, double threshold);
#endif

} // namespace stats

/// Scalar fallback implementations (no SIMD)
namespace scalar {

    /// Apply RANGE filter (scalar)
    size_t apply_range_scalar(
        const double* data,
        size_t length,
        double min_val,
        double max_val,
        uint8_t* mask
    );
    
    /// Apply GREATER filter (scalar)
    size_t apply_greater_scalar(
        const double* data,
        size_t length,
        double threshold,
        uint8_t* mask
    );
    
    /// Apply LESS filter (scalar)
    size_t apply_less_scalar(
        const double* data,
        size_t length,
        double threshold,
        uint8_t* mask
    );
    
    /// Calculate sum (scalar)
    double sum_scalar(const double* data, size_t length);
    
    /// Calculate mean (scalar)
    double mean_scalar(const double* data, size_t length);
    
    /// Calculate min (scalar)
    double min_scalar(const double* data, size_t length);
    
    /// Calculate max (scalar)
    double max_scalar(const double* data, size_t length);
    
    /// Calculate variance (scalar)
    double variance_scalar(const double* data, size_t length, double mean);

} // namespace scalar

} // namespace simd
} // namespace timegraph

