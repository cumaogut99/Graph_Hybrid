#include "timegraph/processing/simd_utils.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace timegraph {
namespace simd {

// CPU feature detection
bool has_avx2() {
#ifdef TIMEGRAPH_HAS_AVX2
    return true;
#else
    return false;
#endif
}

bool has_sse42() {
#ifdef TIMEGRAPH_HAS_SSE42
    return true;
#else
    return false;
#endif
}

namespace filter {

#ifdef TIMEGRAPH_HAS_AVX2

size_t apply_range_avx2(
    const double* data,
    size_t length,
    double min_val,
    double max_val,
    uint8_t* mask
) {
    const size_t simd_width = 4; // AVX2 processes 4 doubles at a time
    const size_t simd_end = (length / simd_width) * simd_width;
    size_t pass_count = 0;
    
    // Broadcast min and max values to all lanes
    __m256d min_vec = _mm256_set1_pd(min_val);
    __m256d max_vec = _mm256_set1_pd(max_val);
    
    // Process 4 doubles at a time
    for (size_t i = 0; i < simd_end; i += simd_width) {
        // Load 4 doubles
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // Check: value >= min_val
        __m256d ge_min = _mm256_cmp_pd(values, min_vec, _CMP_GE_OQ);
        
        // Check: value <= max_val
        __m256d le_max = _mm256_cmp_pd(values, max_vec, _CMP_LE_OQ);
        
        // Combine: (value >= min) AND (value <= max)
        __m256d result = _mm256_and_pd(ge_min, le_max);
        
        // Extract results (each lane is 0xFFFFFFFFFFFFFFFF if true, 0 if false)
        double result_array[4];
        _mm256_storeu_pd(result_array, result);
        
        // Convert to uint8_t mask
        for (size_t j = 0; j < simd_width; ++j) {
            // Check if all bits are set (true)
            mask[i + j] = (*reinterpret_cast<uint64_t*>(&result_array[j]) != 0) ? 1 : 0;
            pass_count += mask[i + j];
        }
    }
    
    // Handle remaining elements (scalar)
    for (size_t i = simd_end; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val >= min_val && val <= max_val) ? 1 : 0;
        pass_count += mask[i];
    }
    
    return pass_count;
}

size_t apply_greater_avx2(
    const double* data,
    size_t length,
    double threshold,
    uint8_t* mask
) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    size_t pass_count = 0;
    
    __m256d threshold_vec = _mm256_set1_pd(threshold);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // Check: value > threshold
        __m256d result = _mm256_cmp_pd(values, threshold_vec, _CMP_GT_OQ);
        
        double result_array[4];
        _mm256_storeu_pd(result_array, result);
        
        for (size_t j = 0; j < simd_width; ++j) {
            mask[i + j] = (*reinterpret_cast<uint64_t*>(&result_array[j]) != 0) ? 1 : 0;
            pass_count += mask[i + j];
        }
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val > threshold) ? 1 : 0;
        pass_count += mask[i];
    }
    
    return pass_count;
}

size_t apply_less_avx2(
    const double* data,
    size_t length,
    double threshold,
    uint8_t* mask
) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    size_t pass_count = 0;
    
    __m256d threshold_vec = _mm256_set1_pd(threshold);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // Check: value < threshold
        __m256d result = _mm256_cmp_pd(values, threshold_vec, _CMP_LT_OQ);
        
        double result_array[4];
        _mm256_storeu_pd(result_array, result);
        
        for (size_t j = 0; j < simd_width; ++j) {
            mask[i + j] = (*reinterpret_cast<uint64_t*>(&result_array[j]) != 0) ? 1 : 0;
            pass_count += mask[i + j];
        }
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val < threshold) ? 1 : 0;
        pass_count += mask[i];
    }
    
    return pass_count;
}

void combine_and_avx2(
    const uint8_t* mask1,
    const uint8_t* mask2,
    size_t length,
    uint8_t* result
) {
    const size_t simd_width = 32; // AVX2 can process 32 bytes at a time
    const size_t simd_end = (length / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        // Load 32 bytes from each mask
        __m256i m1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask1[i]));
        __m256i m2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask2[i]));
        
        // AND operation
        __m256i res = _mm256_and_si256(m1, m2);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), res);
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        result[i] = mask1[i] & mask2[i];
    }
}

void combine_or_avx2(
    const uint8_t* mask1,
    const uint8_t* mask2,
    size_t length,
    uint8_t* result
) {
    const size_t simd_width = 32;
    const size_t simd_end = (length / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256i m1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask1[i]));
        __m256i m2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask2[i]));
        
        // OR operation
        __m256i res = _mm256_or_si256(m1, m2);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), res);
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        result[i] = mask1[i] | mask2[i];
    }
}

#endif // TIMEGRAPH_HAS_AVX2

} // namespace filter

// Scalar fallback implementations
namespace scalar {

size_t apply_range_scalar(
    const double* data,
    size_t length,
    double min_val,
    double max_val,
    uint8_t* mask
) {
    size_t pass_count = 0;
    for (size_t i = 0; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val >= min_val && val <= max_val) ? 1 : 0;
        pass_count += mask[i];
    }
    return pass_count;
}

size_t apply_greater_scalar(
    const double* data,
    size_t length,
    double threshold,
    uint8_t* mask
) {
    size_t pass_count = 0;
    for (size_t i = 0; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val > threshold) ? 1 : 0;
        pass_count += mask[i];
    }
    return pass_count;
}

size_t apply_less_scalar(
    const double* data,
    size_t length,
    double threshold,
    uint8_t* mask
) {
    size_t pass_count = 0;
    for (size_t i = 0; i < length; ++i) {
        double val = data[i];
        mask[i] = (!std::isnan(val) && val < threshold) ? 1 : 0;
        pass_count += mask[i];
    }
    return pass_count;
}

} // namespace scalar

} // namespace simd
} // namespace timegraph

