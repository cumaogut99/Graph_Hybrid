#include "timegraph/processing/simd_utils.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace timegraph {
namespace simd {
namespace stats {

#ifdef TIMEGRAPH_HAS_AVX2

double sum_avx2(const double* data, size_t length) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    
    // Initialize accumulator with zeros
    __m256d sum_vec = _mm256_setzero_pd();
    
    // Process 4 doubles at a time
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, values);
    }
    
    // Horizontal sum of the 4 lanes
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Add remaining elements (scalar)
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i])) {
            sum += data[i];
        }
    }
    
    return sum;
}

double mean_avx2(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum = sum_avx2(data, length);
    return sum / static_cast<double>(length);
}

double min_avx2(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    
    // Initialize with maximum value
    __m256d min_vec = _mm256_set1_pd(std::numeric_limits<double>::max());
    
    // Process 4 doubles at a time
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        min_vec = _mm256_min_pd(min_vec, values);
    }
    
    // Extract minimum from 4 lanes
    double min_array[4];
    _mm256_storeu_pd(min_array, min_vec);
    double min_val = std::min({min_array[0], min_array[1], min_array[2], min_array[3]});
    
    // Check remaining elements (scalar)
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i])) {
            min_val = std::min(min_val, data[i]);
        }
    }
    
    return min_val;
}

double max_avx2(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    
    // Initialize with minimum value
    __m256d max_vec = _mm256_set1_pd(std::numeric_limits<double>::lowest());
    
    // Process 4 doubles at a time
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        max_vec = _mm256_max_pd(max_vec, values);
    }
    
    // Extract maximum from 4 lanes
    double max_array[4];
    _mm256_storeu_pd(max_array, max_vec);
    double max_val = std::max({max_array[0], max_array[1], max_array[2], max_array[3]});
    
    // Check remaining elements (scalar)
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i])) {
            max_val = std::max(max_val, data[i]);
        }
    }
    
    return max_val;
}

double sum_squares_avx2(const double* data, size_t length, double mean) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    
    // Process 4 doubles at a time
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // diff = value - mean
        __m256d diff = _mm256_sub_pd(values, mean_vec);
        
        // diff^2
        __m256d sq = _mm256_mul_pd(diff, diff);
        
        // Accumulate
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);
    }
    
    // Horizontal sum
    double sum_sq_array[4];
    _mm256_storeu_pd(sum_sq_array, sum_sq_vec);
    double sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i])) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
    }
    
    return sum_sq;
}

double variance_avx2(const double* data, size_t length, double mean) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum_sq = sum_squares_avx2(data, length, mean);
    return sum_sq / static_cast<double>(length);
}

double stddev_avx2(const double* data, size_t length, double mean) {
    double var = variance_avx2(data, length, mean);
    return std::sqrt(var);
}

size_t count_above_avx2(const double* data, size_t length, double threshold) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    size_t count = 0;
    
    __m256d threshold_vec = _mm256_set1_pd(threshold);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // Check: value > threshold
        __m256d result = _mm256_cmp_pd(values, threshold_vec, _CMP_GT_OQ);
        
        // Count set bits
        double result_array[4];
        _mm256_storeu_pd(result_array, result);
        
        for (size_t j = 0; j < simd_width; ++j) {
            if (*reinterpret_cast<uint64_t*>(&result_array[j]) != 0) {
                count++;
            }
        }
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i]) && data[i] > threshold) {
            count++;
        }
    }
    
    return count;
}

size_t count_below_avx2(const double* data, size_t length, double threshold) {
    const size_t simd_width = 4;
    const size_t simd_end = (length / simd_width) * simd_width;
    size_t count = 0;
    
    __m256d threshold_vec = _mm256_set1_pd(threshold);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d values = _mm256_loadu_pd(&data[i]);
        
        // Check: value < threshold
        __m256d result = _mm256_cmp_pd(values, threshold_vec, _CMP_LT_OQ);
        
        // Count set bits
        double result_array[4];
        _mm256_storeu_pd(result_array, result);
        
        for (size_t j = 0; j < simd_width; ++j) {
            if (*reinterpret_cast<uint64_t*>(&result_array[j]) != 0) {
                count++;
            }
        }
    }
    
    // Scalar remainder
    for (size_t i = simd_end; i < length; ++i) {
        if (!std::isnan(data[i]) && data[i] < threshold) {
            count++;
        }
    }
    
    return count;
}

#endif // TIMEGRAPH_HAS_AVX2

} // namespace stats

// Scalar fallback implementations
namespace scalar {

double sum_scalar(const double* data, size_t length) {
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            sum += data[i];
        }
    }
    return sum;
}

double mean_scalar(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum_scalar(data, length) / static_cast<double>(length);
}

double min_scalar(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double min_val = std::numeric_limits<double>::max();
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            min_val = std::min(min_val, data[i]);
        }
    }
    return min_val;
}

double max_scalar(const double* data, size_t length) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double max_val = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            max_val = std::max(max_val, data[i]);
        }
    }
    return max_val;
}

double variance_scalar(const double* data, size_t length, double mean) {
    if (length == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < length; ++i) {
        if (!std::isnan(data[i])) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
    }
    return sum_sq / static_cast<double>(length);
}

} // namespace scalar

} // namespace simd
} // namespace timegraph

