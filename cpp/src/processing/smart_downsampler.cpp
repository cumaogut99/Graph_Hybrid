/**
 * @file smart_downsampler.cpp
 * @brief High-performance Smart Downsampling Implementation
 * 
 * Implements hybrid LTTB + Critical Points downsampling with SIMD optimization.
 * 
 * Performance Targets:
 * - 1M points → 4K points in <50ms
 * - Zero spike loss guarantee
 * - AVX2 acceleration where available
 */

#include "timegraph/processing/smart_downsampler.hpp"
#include "timegraph/data/mpai_reader.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>
#include <stdexcept>

// SIMD intrinsics
#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#if defined(__AVX2__) || defined(_MSC_VER)
    #include <immintrin.h>
    #define SMART_DS_HAS_AVX2 1
#else
    #define SMART_DS_HAS_AVX2 0
#endif

// OpenMP for parallel processing
#ifdef _OPENMP
    #include <omp.h>
    #define SMART_DS_HAS_OMP 1
#else
    #define SMART_DS_HAS_OMP 0
#endif

namespace timegraph {

// ============================================================================
// SIMD Helper Functions
// ============================================================================

#if SMART_DS_HAS_AVX2

/**
 * @brief AVX2-optimized sum of double array
 */
static double avx2_sum(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    
    size_t i = 0;
    // Process 4 doubles at a time
    for (; i + 4 <= size; i += 4) {
        __m256d vals = _mm256_loadu_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, vals);
    }
    
    // Horizontal sum of vector
    __m128d low  = _mm256_castpd256_pd128(sum_vec);
    __m128d high = _mm256_extractf128_pd(sum_vec, 1);
    low = _mm_add_pd(low, high);
    __m128d shuf = _mm_shuffle_pd(low, low, 1);
    low = _mm_add_pd(low, shuf);
    
    double sum = _mm_cvtsd_f64(low);
    
    // Handle remaining elements
    for (; i < size; ++i) {
        sum += data[i];
    }
    
    return sum;
}

/**
 * @brief AVX2-optimized find values above threshold
 * Returns count and populates indices
 */
static size_t avx2_find_above(const double* data, size_t size, double threshold, 
                               std::vector<size_t>& indices) {
    __m256d thresh_vec = _mm256_set1_pd(threshold);
    size_t count = 0;
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d vals = _mm256_loadu_pd(data + i);
        __m256d cmp = _mm256_cmp_pd(vals, thresh_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        
        if (mask) {
            // At least one value above threshold
            if (mask & 1) { indices.push_back(i);     ++count; }
            if (mask & 2) { indices.push_back(i + 1); ++count; }
            if (mask & 4) { indices.push_back(i + 2); ++count; }
            if (mask & 8) { indices.push_back(i + 3); ++count; }
        }
    }
    
    // Scalar remainder
    for (; i < size; ++i) {
        if (data[i] > threshold) {
            indices.push_back(i);
            ++count;
        }
    }
    
    return count;
}

/**
 * @brief AVX2-optimized find values below threshold
 */
static size_t avx2_find_below(const double* data, size_t size, double threshold,
                               std::vector<size_t>& indices) {
    __m256d thresh_vec = _mm256_set1_pd(threshold);
    size_t count = 0;
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d vals = _mm256_loadu_pd(data + i);
        __m256d cmp = _mm256_cmp_pd(vals, thresh_vec, _CMP_LT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        
        if (mask) {
            if (mask & 1) { indices.push_back(i);     ++count; }
            if (mask & 2) { indices.push_back(i + 1); ++count; }
            if (mask & 4) { indices.push_back(i + 2); ++count; }
            if (mask & 8) { indices.push_back(i + 3); ++count; }
        }
    }
    
    for (; i < size; ++i) {
        if (data[i] < threshold) {
            indices.push_back(i);
            ++count;
        }
    }
    
    return count;
}

/**
 * @brief AVX2-optimized min/max finding
 */
static void avx2_minmax(const double* data, size_t size, double& out_min, double& out_max) {
    if (size == 0) {
        out_min = out_max = 0.0;
        return;
    }
    
    __m256d min_vec = _mm256_set1_pd(data[0]);
    __m256d max_vec = _mm256_set1_pd(data[0]);
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d vals = _mm256_loadu_pd(data + i);
        min_vec = _mm256_min_pd(min_vec, vals);
        max_vec = _mm256_max_pd(max_vec, vals);
    }
    
    // Reduce vectors to scalars
    alignas(32) double min_arr[4], max_arr[4];
    _mm256_store_pd(min_arr, min_vec);
    _mm256_store_pd(max_arr, max_vec);
    
    out_min = std::min({min_arr[0], min_arr[1], min_arr[2], min_arr[3]});
    out_max = std::max({max_arr[0], max_arr[1], max_arr[2], max_arr[3]});
    
    // Handle remainder
    for (; i < size; ++i) {
        out_min = std::min(out_min, data[i]);
        out_max = std::max(out_max, data[i]);
    }
}

#endif // SMART_DS_HAS_AVX2

// ============================================================================
// Scalar Fallback Functions
// ============================================================================

static double scalar_sum(const double* data, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

static void scalar_minmax(const double* data, size_t size, double& out_min, double& out_max) {
    if (size == 0) {
        out_min = out_max = 0.0;
        return;
    }
    out_min = out_max = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] < out_min) out_min = data[i];
        if (data[i] > out_max) out_max = data[i];
    }
}

static size_t scalar_find_above(const double* data, size_t size, double threshold,
                                 std::vector<size_t>& indices) {
    size_t count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] > threshold) {
            indices.push_back(i);
            ++count;
        }
    }
    return count;
}

static size_t scalar_find_below(const double* data, size_t size, double threshold,
                                 std::vector<size_t>& indices) {
    size_t count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < threshold) {
            indices.push_back(i);
            ++count;
        }
    }
    return count;
}

// ============================================================================
// SmartDownsampler Implementation
// ============================================================================

double SmartDownsampler::calculate_mean_simd(const double* data, size_t size) {
    if (size == 0) return 0.0;
    
#if SMART_DS_HAS_AVX2
    return avx2_sum(data, size) / static_cast<double>(size);
#else
    return scalar_sum(data, size) / static_cast<double>(size);
#endif
}

double SmartDownsampler::calculate_stddev_simd(const double* data, size_t size, double mean) {
    if (size < 2) return 0.0;
    
    double sum_sq = 0.0;
    
#if SMART_DS_HAS_AVX2
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_vec = _mm256_setzero_pd();
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d vals = _mm256_loadu_pd(data + i);
        __m256d diff = _mm256_sub_pd(vals, mean_vec);
        __m256d sq = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, sq);
    }
    
    // Horizontal sum
    __m128d low  = _mm256_castpd256_pd128(sum_vec);
    __m128d high = _mm256_extractf128_pd(sum_vec, 1);
    low = _mm_add_pd(low, high);
    __m128d shuf = _mm_shuffle_pd(low, low, 1);
    low = _mm_add_pd(low, shuf);
    sum_sq = _mm_cvtsd_f64(low);
    
    // Remainder
    for (; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
#endif
    
    return std::sqrt(sum_sq / static_cast<double>(size - 1));
}

std::pair<double, double> SmartDownsampler::auto_calculate_thresholds(
    const double* data, size_t size, double sigma) 
{
    double mean = calculate_mean_simd(data, size);
    double stddev = calculate_stddev_simd(data, size, mean);
    
    double high = mean + sigma * stddev;
    double low = mean - sigma * stddev;
    
    return {high, low};
}

// ============================================================================
// Step 1: Critical Points Detection
// ============================================================================

std::vector<size_t> SmartDownsampler::detect_spikes_simd(
    const double* data, size_t size, double threshold_high, double threshold_low)
{
    std::vector<size_t> indices;
    indices.reserve(size / 100);  // Estimate ~1% are spikes
    
    bool check_high = !std::isnan(threshold_high);
    bool check_low = !std::isnan(threshold_low);
    
    if (!check_high && !check_low) return indices;
    
#if SMART_DS_HAS_AVX2
    if (check_high) {
        avx2_find_above(data, size, threshold_high, indices);
    }
    if (check_low) {
        avx2_find_below(data, size, threshold_low, indices);
    }
#else
    if (check_high) {
        scalar_find_above(data, size, threshold_high, indices);
    }
    if (check_low) {
        scalar_find_below(data, size, threshold_low, indices);
    }
#endif
    
    // Sort and remove duplicates (in case both thresholds hit same point)
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    
    return indices;
}

std::vector<size_t> SmartDownsampler::detect_extrema(
    const double* data, size_t size, size_t window_size, double min_prominence)
{
    std::vector<size_t> indices;
    
    if (size < window_size * 2 + 1) return indices;
    
    // Auto-calculate prominence if not set
    if (min_prominence <= 0.0) {
        double mean = calculate_mean_simd(data, size);
        double stddev = calculate_stddev_simd(data, size, mean);
        min_prominence = stddev * 0.3;  // 30% of std deviation
    }
    
    indices.reserve(size / (window_size * 2));
    
    // Detect local maxima and minima
#if SMART_DS_HAS_OMP
    #pragma omp parallel for schedule(dynamic, 1000) if(size > 10000)
#endif
    for (int64_t i = static_cast<int64_t>(window_size); i < static_cast<int64_t>(size - window_size); ++i) {
        double current = data[i];
        bool is_max = true;
        bool is_min = true;
        
        // Check neighborhood
        for (size_t j = i - window_size; j <= i + window_size; ++j) {
            if (j == i) continue;
            if (data[j] >= current) is_max = false;
            if (data[j] <= current) is_min = false;
            if (!is_max && !is_min) break;
        }
        
        if (is_max || is_min) {
            // Calculate prominence (simple version)
            double left_depth = 0, right_depth = 0;
            
            // Find depth on left
            for (size_t j = i - 1; j > 0; --j) {
                if (is_max && data[j] > current) break;
                if (is_min && data[j] < current) break;
                double diff = std::abs(current - data[j]);
                left_depth = std::max(left_depth, diff);
            }
            
            // Find depth on right
            for (size_t j = i + 1; j < size; ++j) {
                if (is_max && data[j] > current) break;
                if (is_min && data[j] < current) break;
                double diff = std::abs(current - data[j]);
                right_depth = std::max(right_depth, diff);
            }
            
            double prominence = std::min(left_depth, right_depth);
            
            if (prominence >= min_prominence) {
#if SMART_DS_HAS_OMP
                #pragma omp critical
#endif
                {
                    indices.push_back(i);
                }
            }
        }
    }
    
    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<size_t> SmartDownsampler::detect_changes(
    const double* x_data, const double* y_data, size_t size, double sigma_threshold)
{
    std::vector<size_t> indices;
    
    if (size < 3) return indices;
    
    // Calculate derivatives
    std::vector<double> derivatives;
    derivatives.reserve(size - 1);
    
    for (size_t i = 1; i < size; ++i) {
        double dt = x_data[i] - x_data[i-1];
        double dy = y_data[i] - y_data[i-1];
        double deriv = (dt > 0.0) ? dy / dt : 0.0;
        derivatives.push_back(deriv);
    }
    
    // Calculate mean and std of derivatives
    double mean = calculate_mean_simd(derivatives.data(), derivatives.size());
    double stddev = calculate_stddev_simd(derivatives.data(), derivatives.size(), mean);
    
    if (stddev <= 0.0) return indices;
    
    double threshold = sigma_threshold * stddev;
    
    // Find sudden changes
    indices.reserve(derivatives.size() / 50);
    
    for (size_t i = 0; i < derivatives.size(); ++i) {
        if (std::abs(derivatives[i] - mean) > threshold) {
            indices.push_back(i + 1);  // +1 because derivative is offset
        }
    }
    
    return indices;
}

std::vector<size_t> SmartDownsampler::detect_critical_points_simd(
    const double* x_data, const double* y_data, size_t size,
    const SmartDownsampleConfig& config, SmartDownsampleResult& stats)
{
    std::vector<size_t> critical_indices;
    
    // 1. Threshold-based spike detection (highest priority)
    double thresh_high = config.spike_threshold_high;
    double thresh_low = config.spike_threshold_low;
    
    if (config.use_auto_threshold && std::isnan(thresh_high) && std::isnan(thresh_low)) {
        auto [auto_high, auto_low] = auto_calculate_thresholds(
            y_data, size, config.auto_threshold_sigma);
        thresh_high = auto_high;
        thresh_low = auto_low;
    }
    
    auto spike_indices = detect_spikes_simd(y_data, size, thresh_high, thresh_low);
    stats.spike_count = spike_indices.size();
    critical_indices.insert(critical_indices.end(), spike_indices.begin(), spike_indices.end());
    
    // 2. Local extrema detection
    if (config.detect_local_extrema) {
        auto extrema_indices = detect_extrema(
            y_data, size, config.extrema_window, config.min_prominence);
        stats.peak_count = extrema_indices.size() / 2;  // Approximate
        stats.valley_count = extrema_indices.size() - stats.peak_count;
        critical_indices.insert(critical_indices.end(), extrema_indices.begin(), extrema_indices.end());
    }
    
    // 3. Sudden change detection
    if (config.detect_sudden_changes) {
        auto change_indices = detect_changes(x_data, y_data, size, config.change_sigma);
        stats.change_count = change_indices.size();
        critical_indices.insert(critical_indices.end(), change_indices.begin(), change_indices.end());
    }
    
    // Sort and deduplicate
    std::sort(critical_indices.begin(), critical_indices.end());
    critical_indices.erase(
        std::unique(critical_indices.begin(), critical_indices.end()),
        critical_indices.end()
    );
    
    stats.critical_points_count = critical_indices.size();
    return critical_indices;
}

// ============================================================================
// Step 2: LTTB Algorithm
// ============================================================================

std::vector<size_t> SmartDownsampler::apply_lttb(
    const double* x_data, const double* y_data, size_t size, size_t target_points)
{
    std::vector<size_t> indices;
    
    if (size <= target_points || target_points < 3) {
        // No downsampling needed, return all indices
        indices.resize(size);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    
    indices.reserve(target_points);
    
    // Bucket size
    double bucket_size = static_cast<double>(size - 2) / (target_points - 2);
    
    // Always include first point
    indices.push_back(0);
    
    size_t a = 0;  // Previous selected point
    
    for (size_t i = 0; i < target_points - 2; ++i) {
        // Calculate average point for next bucket (point C)
        size_t avg_start = static_cast<size_t>((i + 2) * bucket_size) + 1;
        size_t avg_end = static_cast<size_t>((i + 3) * bucket_size) + 1;
        avg_end = std::min(avg_end, size);
        
        double avg_x = 0.0, avg_y = 0.0;
        size_t avg_count = avg_end - avg_start;
        
        if (avg_count > 0) {
            for (size_t j = avg_start; j < avg_end; ++j) {
                avg_x += x_data[j];
                avg_y += y_data[j];
            }
            avg_x /= avg_count;
            avg_y /= avg_count;
        } else {
            avg_x = x_data[size - 1];
            avg_y = y_data[size - 1];
        }
        
        // Current bucket range
        size_t range_start = static_cast<size_t>((i + 1) * bucket_size) + 1;
        size_t range_end = static_cast<size_t>((i + 2) * bucket_size) + 1;
        range_end = std::min(range_end, size);
        
        // Find point with max triangle area
        double max_area = -1.0;
        size_t max_idx = range_start;
        
        double ax = x_data[a];
        double ay = y_data[a];
        
        for (size_t j = range_start; j < range_end; ++j) {
            // Triangle area formula (simplified)
            double area = std::abs(
                (ax - avg_x) * (y_data[j] - ay) -
                (ax - x_data[j]) * (avg_y - ay)
            ) * 0.5;
            
            if (area > max_area) {
                max_area = area;
                max_idx = j;
            }
        }
        
        indices.push_back(max_idx);
        a = max_idx;
    }
    
    // Always include last point
    indices.push_back(size - 1);
    
    return indices;
}

// ============================================================================
// Step 3: Merge and Deduplicate
// ============================================================================

std::vector<size_t> SmartDownsampler::merge_and_deduplicate(
    std::vector<size_t>& critical_indices,
    std::vector<size_t>& lttb_indices,
    size_t dedup_distance)
{
    // Use set for automatic sorting and deduplication
    std::set<size_t> merged_set;
    
    // Add all critical points (these have priority)
    for (size_t idx : critical_indices) {
        merged_set.insert(idx);
    }
    
    // Add LTTB points if not too close to existing points
    for (size_t idx : lttb_indices) {
        // Check if there's already a point within dedup_distance
        bool too_close = false;
        
        auto it = merged_set.lower_bound(idx > dedup_distance ? idx - dedup_distance : 0);
        auto end = merged_set.upper_bound(idx + dedup_distance);
        
        if (it != end) {
            // There's at least one point within range
            too_close = true;
        }
        
        if (!too_close) {
            merged_set.insert(idx);
        }
    }
    
    // Convert to vector
    std::vector<size_t> result(merged_set.begin(), merged_set.end());
    return result;
}

// ============================================================================
// Build Final Result
// ============================================================================

SmartDownsampleResult SmartDownsampler::build_result(
    const double* x_data, const double* y_data, size_t original_size,
    const std::vector<size_t>& indices, SmartDownsampleResult& stats)
{
    stats.input_size = original_size;
    stats.output_size = indices.size();
    
    stats.x.reserve(indices.size());
    stats.y.reserve(indices.size());
    stats.original_indices = indices;
    
    for (size_t idx : indices) {
        stats.x.push_back(x_data[idx]);
        stats.y.push_back(y_data[idx]);
    }
    
    return stats;
}

// ============================================================================
// Main Downsample Functions
// ============================================================================

SmartDownsampleResult SmartDownsampler::downsample(
    const std::vector<double>& x_data,
    const std::vector<double>& y_data,
    const SmartDownsampleConfig& config)
{
    return downsample(x_data.data(), y_data.data(), x_data.size(), config);
}

SmartDownsampleResult SmartDownsampler::downsample(
    const double* x_data,
    const double* y_data,
    size_t size,
    const SmartDownsampleConfig& config)
{
    SmartDownsampleResult result;
    
    // Validation
    if (size == 0 || x_data == nullptr || y_data == nullptr) {
        return result;
    }
    
    // If data is already small enough, return as-is
    if (size <= config.target_points) {
        result.x.assign(x_data, x_data + size);
        result.y.assign(y_data, y_data + size);
        result.original_indices.resize(size);
        std::iota(result.original_indices.begin(), result.original_indices.end(), 0);
        result.input_size = size;
        result.output_size = size;
        return result;
    }
    
    // Step 1: Detect critical points
    auto critical_indices = detect_critical_points_simd(x_data, y_data, size, config, result);
    
    // Step 2: Apply LTTB
    std::vector<size_t> lttb_indices;
    if (config.use_lttb) {
        // Calculate LTTB target (accounting for critical points)
        size_t lttb_target = static_cast<size_t>(config.target_points * config.lttb_ratio);
        lttb_target = std::max(lttb_target, size_t(100));  // Minimum LTTB points
        
        lttb_indices = apply_lttb(x_data, y_data, size, lttb_target);
        result.lttb_points_count = lttb_indices.size();
    }
    
    // Step 3: Merge and deduplicate
    auto final_indices = merge_and_deduplicate(
        critical_indices, lttb_indices, config.dedup_distance);
    
    // Build result
    return build_result(x_data, y_data, size, final_indices, result);
}

SmartDownsampleResult SmartDownsampler::downsample_streaming(
    timegraph::mpai::MpaiReader& reader,
    const std::string& time_column,
    const std::string& signal_column,
    const SmartDownsampleConfig& config)
{
    SmartDownsampleResult result;
    
    uint64_t total_rows = reader.get_row_count();
    if (total_rows == 0) return result;
    
    // For streaming, we use chunk-based processing
    const uint64_t chunk_size = 500000;  // 500K per chunk
    
    // Collect all critical indices globally
    std::vector<size_t> global_critical;
    std::vector<double> global_x, global_y;
    
    // First pass: collect critical points and global statistics
    for (uint64_t start = 0; start < total_rows; start += chunk_size) {
        uint64_t count = std::min(chunk_size, total_rows - start);
        
        auto x_chunk = reader.load_column_slice(time_column, start, count);
        auto y_chunk = reader.load_column_slice(signal_column, start, count);
        
        // Detect spikes in this chunk
        auto chunk_spikes = detect_spikes_simd(
            y_chunk.data(), y_chunk.size(),
            config.spike_threshold_high, config.spike_threshold_low);
        
        // Offset indices to global space
        for (size_t idx : chunk_spikes) {
            global_critical.push_back(start + idx);
        }
        
        // Store chunk data for LTTB
        global_x.insert(global_x.end(), x_chunk.begin(), x_chunk.end());
        global_y.insert(global_y.end(), y_chunk.begin(), y_chunk.end());
    }
    
    result.spike_count = global_critical.size();
    result.critical_points_count = global_critical.size();
    
    // Apply LTTB on collected data
    std::vector<size_t> lttb_indices;
    if (config.use_lttb && !global_x.empty()) {
        size_t lttb_target = static_cast<size_t>(config.target_points * config.lttb_ratio);
        lttb_indices = apply_lttb(global_x.data(), global_y.data(), global_x.size(), lttb_target);
        result.lttb_points_count = lttb_indices.size();
    }
    
    // Merge
    auto final_indices = merge_and_deduplicate(
        global_critical, lttb_indices, config.dedup_distance);
    
    // Build result
    return build_result(global_x.data(), global_y.data(), global_x.size(), final_indices, result);
}

SmartDownsampleResult SmartDownsampler::quick_downsample(
    const std::vector<double>& x_data,
    const std::vector<double>& y_data,
    size_t target_points,
    std::optional<double> threshold)
{
    SmartDownsampler ds;
    SmartDownsampleConfig config;
    config.target_points = target_points;
    
    if (threshold.has_value()) {
        config.spike_threshold_high = threshold.value();
        config.spike_threshold_low = -threshold.value();
        config.use_auto_threshold = false;
    }
    
    return ds.downsample(x_data, y_data, config);
}

// ============================================================================
// Convenience Function
// ============================================================================

SmartDownsampleResult smart_downsample(
    const std::vector<double>& x_data,
    const std::vector<double>& y_data,
    size_t target_points,
    double threshold_high,
    double threshold_low)
{
    SmartDownsampler ds;
    SmartDownsampleConfig config;
    config.target_points = target_points;
    config.spike_threshold_high = threshold_high;
    config.spike_threshold_low = threshold_low;
    config.use_auto_threshold = std::isnan(threshold_high) && std::isnan(threshold_low);
    
    return ds.downsample(x_data, y_data, config);
}

} // namespace timegraph
