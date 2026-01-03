#include "timegraph/processing/downsample.hpp"
#include "timegraph/processing/critical_points.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <set>

namespace timegraph {

struct BucketState {
    bool has_first{false};
    double first_time{0.0};
    double first_value{0.0};

    bool has_min{false};
    double min_time{0.0};
    double min_value{0.0};

    bool has_max{false};
    double max_time{0.0};
    double max_value{0.0};

    bool has_thresh{false};
    double thresh_time{0.0};
    double thresh_value{0.0};
};

DownsampleResult downsample_minmax_streaming(
    timegraph::mpai::MpaiReader& reader,
    const std::string& time_column,
    const std::string& signal_column,
    uint64_t max_points,
    double warning_min,
    double warning_max
) {
    DownsampleResult out;
    const uint64_t total_rows = reader.get_row_count();
    if (total_rows == 0 || max_points == 0) {
        return out;
    }

    const uint64_t interval = std::max<uint64_t>(1, (total_rows + max_points - 1) / max_points);
    const uint64_t chunk_size = 1'000'000;  // reuse large streaming chunk

    BucketState bucket;
    uint64_t current_bucket = 0;

    auto flush_bucket = [&](BucketState& b) {
        // Emit points in time order, avoid duplicates
        struct Pt { double t; double v; };
        std::vector<Pt> pts;
        pts.reserve(4);
        if (b.has_first)  pts.push_back({b.first_time,  b.first_value});
        if (b.has_min)    pts.push_back({b.min_time,    b.min_value});
        if (b.has_max)    pts.push_back({b.max_time,    b.max_value});
        if (b.has_thresh) pts.push_back({b.thresh_time, b.thresh_value});
        std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b){ return a.t < b.t; });
        // Deduplicate same time/value pairs
        for (size_t i = 0; i < pts.size(); ++i) {
            if (i > 0 && pts[i].t == pts[i-1].t && pts[i].v == pts[i-1].v) continue;
            out.time.push_back(pts[i].t);
            out.value.push_back(pts[i].v);
        }
        b = BucketState{};  // reset
    };

    uint64_t global_index = 0;
    for (uint64_t start = 0; start < total_rows; start += chunk_size) {
        uint64_t count = std::min<uint64_t>(chunk_size, total_rows - start);
        auto time_chunk = reader.load_column_slice(time_column, start, count);
        auto data_chunk = reader.load_column_slice(signal_column, start, count);
        for (uint64_t i = 0; i < count; ++i, ++global_index) {
            const uint64_t bucket_id = global_index / interval;
            if (bucket_id != current_bucket) {
                flush_bucket(bucket);
                current_bucket = bucket_id;
            }
            double t = time_chunk[i];
            double v = data_chunk[i];

            if (!bucket.has_first) {
                bucket.has_first = true;
                bucket.first_time = t;
                bucket.first_value = v;
            }
            if (!bucket.has_min || v < bucket.min_value) {
                bucket.has_min = true;
                bucket.min_time = t;
                bucket.min_value = v;
            }
            if (!bucket.has_max || v > bucket.max_value) {
                bucket.has_max = true;
                bucket.max_time = t;
                bucket.max_value = v;
            }

            // Threshold hit (first occurrence in bucket)
            const bool has_warn_min = !std::isnan(warning_min);
            const bool has_warn_max = !std::isnan(warning_max);
            if (!bucket.has_thresh && (has_warn_min || has_warn_max)) {
                if ((has_warn_min && v < warning_min) || (has_warn_max && v > warning_max)) {
                    bucket.has_thresh = true;
                    bucket.thresh_time = t;
                    bucket.thresh_value = v;
                }
            }
        }
    }

    // flush last bucket
    flush_bucket(bucket);

    return out;
}

// ========== LTTB IMPLEMENTATION ==========

DownsampleResult downsample_lttb(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t max_points
) {
    DownsampleResult result;
    
    size_t data_length = time_data.size();
    
    if (data_length <= max_points || max_points < 3) {
        // No downsampling needed or invalid max_points
        result.time = time_data;
        result.value = signal_data;
        result.indices.resize(data_length);
        for (size_t i = 0; i < data_length; ++i) {
            result.indices[i] = i;
        }
        return result;
    }
    
    result.time.reserve(max_points);
    result.value.reserve(max_points);
    result.indices.reserve(max_points);
    
    // Bucket size (floating point)
    double bucket_size = static_cast<double>(data_length - 2) / (max_points - 2);
    
    // Always add first point
    result.time.push_back(time_data[0]);
    result.value.push_back(signal_data[0]);
    result.indices.push_back(0);
    
    size_t a = 0;  // Point A (last selected point)
    
    for (size_t i = 0; i < max_points - 2; ++i) {
        // Calculate point average for next bucket (point C)
        size_t avg_range_start = static_cast<size_t>((i + 2) * bucket_size) + 1;
        size_t avg_range_end = static_cast<size_t>((i + 3) * bucket_size) + 1;
        
        if (avg_range_end >= data_length) {
            avg_range_end = data_length;
        }
        
        double avg_time = 0.0;
        double avg_value = 0.0;
        size_t avg_range_length = avg_range_end - avg_range_start;
        
        if (avg_range_length > 0) {
            for (size_t j = avg_range_start; j < avg_range_end; ++j) {
                avg_time += time_data[j];
                avg_value += signal_data[j];
            }
            avg_time /= avg_range_length;
            avg_value /= avg_range_length;
        } else {
            // Edge case: use last point
            avg_time = time_data[data_length - 1];
            avg_value = signal_data[data_length - 1];
        }
        
        // Get current bucket range
        size_t range_start = static_cast<size_t>((i + 1) * bucket_size) + 1;
        size_t range_end = static_cast<size_t>((i + 2) * bucket_size) + 1;
        
        if (range_end >= data_length) {
            range_end = data_length;
        }
        
        // Point A coordinates
        double point_a_time = time_data[a];
        double point_a_value = signal_data[a];
        
        // Find point with maximum triangle area
        double max_area = -1.0;
        size_t max_area_point = range_start;
        
        for (size_t j = range_start; j < range_end; ++j) {
            // Calculate triangle area
            double area = std::abs(
                (point_a_time - avg_time) * (signal_data[j] - point_a_value) -
                (point_a_time - time_data[j]) * (avg_value - point_a_value)
            ) * 0.5;
            
            if (area > max_area) {
                max_area = area;
                max_area_point = j;
            }
        }
        
        // Add selected point
        result.time.push_back(time_data[max_area_point]);
        result.value.push_back(signal_data[max_area_point]);
        result.indices.push_back(max_area_point);
        
        a = max_area_point;  // This becomes the next point A
    }
    
    // Always add last point
    result.time.push_back(time_data[data_length - 1]);
    result.value.push_back(signal_data[data_length - 1]);
    result.indices.push_back(data_length - 1);
    
    return result;
}

DownsampleResult downsample_lttb_with_critical(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t max_points,
    const CriticalPointsConfig& config
) {
    size_t data_length = time_data.size();
    
    if (data_length <= max_points) {
        // No downsampling needed
        DownsampleResult result;
        result.time = time_data;
        result.value = signal_data;
        result.indices.resize(data_length);
        for (size_t i = 0; i < data_length; ++i) {
            result.indices[i] = i;
        }
        return result;
    }
    
    // Step 1: Detect critical points
    auto critical_points = CriticalPointsDetector::detect(
        time_data, signal_data, config
    );
    
    // Step 2: Calculate points to reserve for LTTB
    size_t critical_count = critical_points.size();
    size_t lttb_points = (max_points > critical_count) ? 
                        (max_points - critical_count) : max_points / 2;
    
    // Step 3: Run LTTB on full data
    auto lttb_result = downsample_lttb(time_data, signal_data, lttb_points);
    
    // Step 4: Merge LTTB points with critical points
    std::set<size_t> selected_indices;
    
    // Add all LTTB points
    for (size_t idx : lttb_result.indices) {
        selected_indices.insert(idx);
    }
    
    // Add all critical points
    for (const auto& cp : critical_points) {
        selected_indices.insert(cp.index);
    }
    
    // Step 5: Build final result (sorted by index)
    DownsampleResult result;
    result.time.reserve(selected_indices.size());
    result.value.reserve(selected_indices.size());
    result.indices.reserve(selected_indices.size());
    result.critical_count = critical_count;
    
    for (size_t idx : selected_indices) {
        result.time.push_back(time_data[idx]);
        result.value.push_back(signal_data[idx]);
        result.indices.push_back(idx);
    }
    
    return result;
}

DownsampleResult downsample_auto(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t screen_width,
    bool has_limits
) {
    size_t data_length = time_data.size();
    
    // Strategy: Use 2x screen width for target points
    size_t target_points = screen_width * 2;
    
    if (data_length <= target_points) {
        // No downsampling needed
        DownsampleResult result;
        result.time = time_data;
        result.value = signal_data;
        result.indices.resize(data_length);
        for (size_t i = 0; i < data_length; ++i) {
            result.indices[i] = i;
        }
        return result;
    }
    
    // If limits are active, use critical-aware downsampling
    if (has_limits) {
        CriticalPointsConfig config;
        config.detect_limit_violations = true;
        config.detect_peaks = true;
        config.detect_valleys = true;
        config.detect_sudden_changes = true;
        config.max_points = 500;  // Max 500 critical points
        
        return downsample_lttb_with_critical(
            time_data, signal_data, target_points, config
        );
    } else {
        // Simple LTTB for browsing mode
        return downsample_lttb(time_data, signal_data, target_points);
    }
}

}  // namespace timegraph

