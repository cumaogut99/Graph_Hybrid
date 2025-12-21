#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <limits>

#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/processing/critical_points.hpp"

namespace timegraph {

struct DownsampleResult {
    std::vector<double> time;
    std::vector<double> value;
    std::vector<size_t> indices;  ///< Original indices (optional)
    size_t critical_count = 0;    ///< Number of critical points preserved
};

/**
 * Min/Max + First + Threshold-aware downsample for visualization.
 * Guarantees per-bucket first, min, max; optionally first threshold hit.
 * Streaming over MPAI reader, low RAM.
 */
DownsampleResult downsample_minmax_streaming(
    timegraph::mpai::MpaiReader& reader,
    const std::string& time_column,
    const std::string& signal_column,
    uint64_t max_points,
    double warning_min = std::numeric_limits<double>::quiet_NaN(),
    double warning_max = std::numeric_limits<double>::quiet_NaN()
);

/**
 * LTTB (Largest Triangle Three Buckets) downsampling algorithm
 * Preserves visual characteristics while reducing point count
 * 
 * @param time_data Time values
 * @param signal_data Signal values
 * @param max_points Target number of points (typically 2000-4000)
 * @return Downsampled result
 */
DownsampleResult downsample_lttb(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t max_points
);

/**
 * Smart LTTB downsampling with critical points preservation
 * Combines LTTB with critical points detection to ensure no data loss
 * 
 * @param time_data Time values
 * @param signal_data Signal values
 * @param max_points Target number of points
 * @param config Critical points detection configuration
 * @return Downsampled result with critical points preserved
 */
DownsampleResult downsample_lttb_with_critical(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t max_points,
    const CriticalPointsConfig& config = CriticalPointsConfig()
);

/**
 * Auto-adaptive downsampling strategy
 * Chooses between LTTB, LTTB+Critical, or no downsampling based on:
 * - Data size
 * - Presence of limits
 * - Available screen space
 * 
 * @param time_data Time values
 * @param signal_data Signal values
 * @param screen_width Screen width in pixels (default: 1920)
 * @param has_limits Whether static limits are active
 * @return Downsampled result (or original if no downsampling needed)
 */
DownsampleResult downsample_auto(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t screen_width = 1920,
    bool has_limits = false
);

}  // namespace timegraph


