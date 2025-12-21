#pragma once

/**
 * @file smart_downsampler.hpp
 * @brief High-performance Smart Downsampling with Critical Point Preservation
 * 
 * This module implements a hybrid downsampling strategy that:
 * 1. Detects critical points (spikes, peaks, valleys) using SIMD/AVX2
 * 2. Applies LTTB for visual fidelity
 * 3. Merges both sets with deduplication
 * 
 * Guarantees: NO spike/anomaly is ever lost during visualization.
 * 
 * @author MachinePulseAI Team
 * @date 2024
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <limits>
#include <span>

// Forward declarations
namespace timegraph {
namespace mpai {
    class MpaiReader;
} // namespace mpai
} // namespace timegraph

namespace timegraph {

/**
 * @brief Configuration for Smart Downsampling
 */
struct SmartDownsampleConfig {
    // Target output
    size_t target_points = 4000;           ///< Target number of output points
    
    // Critical point detection thresholds
    double spike_threshold_high = std::numeric_limits<double>::quiet_NaN();  ///< Upper threshold for spike detection
    double spike_threshold_low  = std::numeric_limits<double>::quiet_NaN();  ///< Lower threshold for spike detection
    
    // Auto-threshold calculation (if thresholds not set)
    double auto_threshold_sigma = 3.0;     ///< Sigma multiplier for auto-threshold (mean ± sigma*std)
    bool use_auto_threshold = true;        ///< Calculate threshold from data statistics
    
    // Peak/Valley detection
    bool detect_local_extrema = true;      ///< Detect local min/max points
    size_t extrema_window = 10;            ///< Window size for local extrema detection
    double min_prominence = 0.0;           ///< Minimum prominence (0 = auto-calculate)
    
    // Sudden change detection (derivative-based)
    bool detect_sudden_changes = true;     ///< Detect rapid value changes
    double change_sigma = 2.5;             ///< Derivative threshold in std deviations
    
    // LTTB configuration
    bool use_lttb = true;                  ///< Use LTTB for visual preservation
    double lttb_ratio = 0.7;               ///< Ratio of target_points for LTTB (rest for critical)
    
    // Performance tuning
    size_t simd_chunk_size = 1024;         ///< Chunk size for SIMD processing
    bool use_simd = true;                  ///< Enable SIMD/AVX2 optimization
    bool parallel = true;                  ///< Enable OpenMP parallel processing
    
    // Merge settings
    size_t dedup_distance = 3;             ///< Minimum index distance to consider points unique
    
    SmartDownsampleConfig() = default;
};

/**
 * @brief Result of smart downsampling operation
 */
struct SmartDownsampleResult {
    std::vector<double> x;                 ///< Downsampled X (time) values
    std::vector<double> y;                 ///< Downsampled Y (signal) values
    std::vector<size_t> original_indices;  ///< Original indices in source data
    
    // Statistics
    size_t input_size = 0;                 ///< Original data size
    size_t output_size = 0;                ///< Final output size
    size_t critical_points_count = 0;      ///< Number of critical points preserved
    size_t lttb_points_count = 0;          ///< Number of LTTB points
    size_t spike_count = 0;                ///< Number of threshold violations
    size_t peak_count = 0;                 ///< Number of local maxima
    size_t valley_count = 0;               ///< Number of local minima
    size_t change_count = 0;               ///< Number of sudden changes
    
    double compression_ratio() const {
        return input_size > 0 ? static_cast<double>(output_size) / input_size : 1.0;
    }
    
    bool is_valid() const { return !x.empty() && x.size() == y.size(); }
};

/**
 * @brief High-performance Smart Downsampler
 * 
 * Implements a hybrid downsampling strategy:
 * 1. Fast critical point detection (SIMD-optimized)
 * 2. LTTB visual preservation
 * 3. Intelligent merge with deduplication
 * 
 * Usage:
 * @code
 * SmartDownsampler ds;
 * SmartDownsampleConfig config;
 * config.target_points = 4000;
 * config.spike_threshold_high = 100.0;
 * 
 * auto result = ds.downsample(time_data, signal_data, config);
 * @endcode
 */
class SmartDownsampler {
public:
    SmartDownsampler() = default;
    ~SmartDownsampler() = default;
    
    // Non-copyable (for safety with internal buffers)
    SmartDownsampler(const SmartDownsampler&) = delete;
    SmartDownsampler& operator=(const SmartDownsampler&) = delete;
    
    // Movable
    SmartDownsampler(SmartDownsampler&&) = default;
    SmartDownsampler& operator=(SmartDownsampler&&) = default;
    
    /**
     * @brief Main downsampling function
     * 
     * @param x_data Time/X values (must be sorted)
     * @param y_data Signal/Y values
     * @param config Downsampling configuration
     * @return SmartDownsampleResult containing reduced data
     */
    SmartDownsampleResult downsample(
        const std::vector<double>& x_data,
        const std::vector<double>& y_data,
        const SmartDownsampleConfig& config = SmartDownsampleConfig()
    );
    
    /**
     * @brief Downsample using raw pointers (zero-copy from NumPy)
     * 
     * @param x_data Pointer to time data
     * @param y_data Pointer to signal data
     * @param size Data length
     * @param config Downsampling configuration
     * @return SmartDownsampleResult
     */
    SmartDownsampleResult downsample(
        const double* x_data,
        const double* y_data,
        size_t size,
        const SmartDownsampleConfig& config = SmartDownsampleConfig()
    );
    
    /**
     * @brief Streaming downsample from MPAI file (low memory)
     * 
     * @param reader MPAI reader instance
     * @param time_column Time column name
     * @param signal_column Signal column name
     * @param config Downsampling configuration
     * @return SmartDownsampleResult
     */
    SmartDownsampleResult downsample_streaming(
        timegraph::mpai::MpaiReader& reader,
        const std::string& time_column,
        const std::string& signal_column,
        const SmartDownsampleConfig& config = SmartDownsampleConfig()
    );
    
    // ==================== Static Utility Functions ====================
    
    /**
     * @brief Quick downsample (convenience wrapper)
     * 
     * @param x_data Time data
     * @param y_data Signal data
     * @param target_points Target output size
     * @param threshold Optional threshold for spike detection
     * @return SmartDownsampleResult
     */
    static SmartDownsampleResult quick_downsample(
        const std::vector<double>& x_data,
        const std::vector<double>& y_data,
        size_t target_points = 4000,
        std::optional<double> threshold = std::nullopt
    );
    
private:
    // ==================== Internal Algorithms ====================
    
    /**
     * @brief Step 1: Detect critical points using SIMD
     * Returns indices of critical points
     */
    std::vector<size_t> detect_critical_points_simd(
        const double* x_data,
        const double* y_data,
        size_t size,
        const SmartDownsampleConfig& config,
        SmartDownsampleResult& stats
    );
    
    /**
     * @brief Step 1a: Detect threshold violations (spikes)
     * Uses AVX2 for parallel comparison
     */
    std::vector<size_t> detect_spikes_simd(
        const double* data,
        size_t size,
        double threshold_high,
        double threshold_low
    );
    
    /**
     * @brief Step 1b: Detect local extrema (peaks/valleys)
     */
    std::vector<size_t> detect_extrema(
        const double* data,
        size_t size,
        size_t window_size,
        double min_prominence
    );
    
    /**
     * @brief Step 1c: Detect sudden changes (derivative spikes)
     */
    std::vector<size_t> detect_changes(
        const double* x_data,
        const double* y_data,
        size_t size,
        double sigma_threshold
    );
    
    /**
     * @brief Step 2: Apply LTTB algorithm
     */
    std::vector<size_t> apply_lttb(
        const double* x_data,
        const double* y_data,
        size_t size,
        size_t target_points
    );
    
    /**
     * @brief Step 3: Merge and deduplicate indices
     */
    std::vector<size_t> merge_and_deduplicate(
        std::vector<size_t>& critical_indices,
        std::vector<size_t>& lttb_indices,
        size_t dedup_distance
    );
    
    /**
     * @brief Build final result from selected indices
     */
    SmartDownsampleResult build_result(
        const double* x_data,
        const double* y_data,
        size_t original_size,
        const std::vector<size_t>& indices,
        SmartDownsampleResult& stats
    );
    
    // ==================== Statistics Helpers ====================
    
    /**
     * @brief Calculate mean using SIMD
     */
    static double calculate_mean_simd(const double* data, size_t size);
    
    /**
     * @brief Calculate standard deviation using SIMD
     */
    static double calculate_stddev_simd(const double* data, size_t size, double mean);
    
    /**
     * @brief Auto-calculate thresholds from data statistics
     */
    static std::pair<double, double> auto_calculate_thresholds(
        const double* data,
        size_t size,
        double sigma
    );
};

// ==================== Convenience Functions ====================

/**
 * @brief Simple downsample function (for Python binding)
 * 
 * @param x_data Time values
 * @param y_data Signal values
 * @param target_points Target output size (default: 4000)
 * @param threshold_high Upper spike threshold (NaN = auto)
 * @param threshold_low Lower spike threshold (NaN = auto)
 * @return SmartDownsampleResult
 */
SmartDownsampleResult smart_downsample(
    const std::vector<double>& x_data,
    const std::vector<double>& y_data,
    size_t target_points = 4000,
    double threshold_high = std::numeric_limits<double>::quiet_NaN(),
    double threshold_low = std::numeric_limits<double>::quiet_NaN()
);

} // namespace timegraph
