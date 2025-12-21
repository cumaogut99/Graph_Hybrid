#include "timegraph/processing/critical_points.hpp"
#include "timegraph/processing/arrow_utils.hpp"
#include "timegraph/processing/statistics_engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

#ifdef HAVE_ARROW
#include <arrow/api.h>
#include <arrow/compute/api.h>
#endif

namespace timegraph {

std::vector<CriticalPoint> CriticalPointsDetector::detect(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    const CriticalPointsConfig& config
) {
    if (time_data.size() != signal_data.size() || time_data.empty()) {
        return {};
    }
    
    std::vector<std::vector<CriticalPoint>> all_points;
    
    // 1. Detect local extrema (peaks/valleys)
    if (config.detect_peaks || config.detect_valleys) {
        auto extrema = detect_local_extrema(
            time_data, signal_data,
            config.window_size,
            config.min_prominence
        );
        all_points.push_back(extrema);
    }
    
    // 2. Detect sudden changes
    if (config.detect_sudden_changes) {
        auto changes = detect_sudden_changes(
            time_data, signal_data,
            config.change_threshold
        );
        all_points.push_back(changes);
    }
    
    // 3. Detect limit violations
    if (config.detect_limit_violations) {
        std::vector<double> all_limits;
        all_limits.insert(all_limits.end(), 
            config.warning_limits.begin(), config.warning_limits.end());
        all_limits.insert(all_limits.end(), 
            config.error_limits.begin(), config.error_limits.end());
        
        if (!all_limits.empty()) {
            auto violations = detect_limit_violations(
                time_data, signal_data, all_limits
            );
            all_points.push_back(violations);
        }
    }
    
    // 4. Merge and deduplicate
    auto merged = merge_and_deduplicate(all_points, 5);
    
    // 5. Filter by significance and max points
    auto filtered = filter_by_significance(
        merged,
        config.min_significance,
        config.max_points
    );
    
    return filtered;
}

std::vector<CriticalPoint> CriticalPointsDetector::detect_arrow(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    const CriticalPointsConfig& config
) {
    // For now, just use the native implementation
    // Arrow acceleration can be added for derivative calculations
    return detect(time_data, signal_data, config);
}

std::vector<CriticalPoint> CriticalPointsDetector::detect_local_extrema(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    size_t window_size,
    double min_prominence
) {
    std::vector<CriticalPoint> extrema;
    
    if (signal_data.size() < window_size * 2 + 1) {
        return extrema;
    }
    
    // Auto-calculate prominence if not specified
    if (min_prominence <= 0.0) {
        double mean = StatisticsEngine::calculate_mean(
            signal_data.data(), signal_data.size()
        );
        double std_dev = StatisticsEngine::calculate_std_dev(
            signal_data.data(), signal_data.size(), mean
        );
        min_prominence = std_dev * 0.5;  // Half std dev
    }
    
    // Find local maxima and minima
    for (size_t i = window_size; i < signal_data.size() - window_size; ++i) {
        double current = signal_data[i];
        
        // Check if local maximum
        bool is_max = true;
        bool is_min = true;
        
        for (size_t j = i - window_size; j <= i + window_size; ++j) {
            if (j == i) continue;
            
            if (signal_data[j] >= current) {
                is_max = false;
            }
            if (signal_data[j] <= current) {
                is_min = false;
            }
            
            if (!is_max && !is_min) break;
        }
        
        if (is_max) {
            double prominence = calculate_prominence(signal_data, i, true);
            if (prominence >= min_prominence) {
                CriticalPoint pt(
                    i, time_data[i], current,
                    CriticalPoint::Type::LOCAL_MAX,
                    std::min(1.0, prominence / (min_prominence * 3.0))
                );
                extrema.push_back(pt);
            }
        } else if (is_min) {
            double prominence = calculate_prominence(signal_data, i, false);
            if (prominence >= min_prominence) {
                CriticalPoint pt(
                    i, time_data[i], current,
                    CriticalPoint::Type::LOCAL_MIN,
                    std::min(1.0, prominence / (min_prominence * 3.0))
                );
                extrema.push_back(pt);
            }
        }
    }
    
    return extrema;
}

std::vector<CriticalPoint> CriticalPointsDetector::detect_sudden_changes(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    double threshold_sigma
) {
    std::vector<CriticalPoint> changes;
    
    if (signal_data.size() < 3) {
        return changes;
    }
    
    // Calculate first derivative
    auto derivative = calculate_derivative(time_data, signal_data);
    
    // Calculate mean and std of derivative
    double mean = StatisticsEngine::calculate_mean(
        derivative.data(), derivative.size()
    );
    double std_dev = StatisticsEngine::calculate_std_dev(
        derivative.data(), derivative.size(), mean
    );
    
    if (std_dev <= 0.0) {
        return changes;
    }
    
    double threshold = threshold_sigma * std_dev;
    
    // Find points where derivative exceeds threshold
    for (size_t i = 0; i < derivative.size(); ++i) {
        double abs_deriv = std::abs(derivative[i] - mean);
        
        if (abs_deriv > threshold) {
            size_t data_idx = i + 1;  // Derivative is offset by 1
            
            CriticalPoint pt(
                data_idx,
                time_data[data_idx],
                signal_data[data_idx],
                CriticalPoint::Type::SUDDEN_CHANGE,
                std::min(1.0, abs_deriv / (threshold * 2.0))
            );
            changes.push_back(pt);
        }
    }
    
    return changes;
}

std::vector<CriticalPoint> CriticalPointsDetector::detect_limit_violations(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data,
    const std::vector<double>& limits
) {
    std::vector<CriticalPoint> violations;
    
    if (signal_data.size() < 2 || limits.empty()) {
        return violations;
    }
    
    // For each limit, find crossing points
    for (double limit : limits) {
        for (size_t i = 1; i < signal_data.size(); ++i) {
            double prev = signal_data[i - 1];
            double curr = signal_data[i];
            
            // Check for crossing (either direction)
            bool crosses = (prev < limit && curr >= limit) ||
                          (prev > limit && curr <= limit) ||
                          (prev <= limit && curr > limit) ||
                          (prev >= limit && curr < limit);
            
            if (crosses) {
                CriticalPoint pt(
                    i,
                    time_data[i],
                    curr,
                    CriticalPoint::Type::LIMIT_VIOLATION,
                    1.0  // High significance
                );
                violations.push_back(pt);
            }
        }
    }
    
    return violations;
}

std::vector<CriticalPoint> CriticalPointsDetector::merge_and_deduplicate(
    const std::vector<std::vector<CriticalPoint>>& points_list,
    size_t max_distance
) {
    std::vector<CriticalPoint> merged;
    
    // Collect all points
    for (const auto& points : points_list) {
        merged.insert(merged.end(), points.begin(), points.end());
    }
    
    if (merged.empty()) {
        return merged;
    }
    
    // Sort by index
    std::sort(merged.begin(), merged.end(),
        [](const CriticalPoint& a, const CriticalPoint& b) {
            return a.index < b.index;
        });
    
    // Remove duplicates (points within max_distance)
    std::vector<CriticalPoint> deduplicated;
    deduplicated.push_back(merged[0]);
    
    for (size_t i = 1; i < merged.size(); ++i) {
        const auto& current = merged[i];
        const auto& last = deduplicated.back();
        
        if (current.index - last.index > max_distance) {
            // Far enough, keep it
            deduplicated.push_back(current);
        } else {
            // Too close, keep the one with higher significance
            if (current.significance > last.significance) {
                deduplicated.back() = current;
            }
        }
    }
    
    return deduplicated;
}

std::vector<CriticalPoint> CriticalPointsDetector::filter_by_significance(
    const std::vector<CriticalPoint>& points,
    double min_significance,
    size_t max_points
) {
    std::vector<CriticalPoint> filtered;
    
    // Filter by minimum significance
    for (const auto& pt : points) {
        if (pt.significance >= min_significance) {
            filtered.push_back(pt);
        }
    }
    
    // Sort by significance (descending)
    std::sort(filtered.begin(), filtered.end(),
        [](const CriticalPoint& a, const CriticalPoint& b) {
            return a.significance > b.significance;
        });
    
    // Limit to max_points
    if (max_points > 0 && filtered.size() > max_points) {
        filtered.resize(max_points);
    }
    
    // Re-sort by index for output
    std::sort(filtered.begin(), filtered.end(),
        [](const CriticalPoint& a, const CriticalPoint& b) {
            return a.index < b.index;
        });
    
    return filtered;
}

// Private helper functions

double CriticalPointsDetector::calculate_prominence(
    const std::vector<double>& data,
    size_t index,
    bool is_max
) {
    if (index == 0 || index >= data.size() - 1) {
        return 0.0;
    }
    
    double peak_value = data[index];
    
    // Find the lowest/highest contour around the peak
    double contour = peak_value;
    
    // Search left
    for (size_t i = index - 1; i > 0; --i) {
        if (is_max) {
            if (data[i] > peak_value) break;
            contour = std::min(contour, data[i]);
        } else {
            if (data[i] < peak_value) break;
            contour = std::max(contour, data[i]);
        }
    }
    
    // Search right
    for (size_t i = index + 1; i < data.size(); ++i) {
        if (is_max) {
            if (data[i] > peak_value) break;
            contour = std::min(contour, data[i]);
        } else {
            if (data[i] < peak_value) break;
            contour = std::max(contour, data[i]);
        }
    }
    
    return std::abs(peak_value - contour);
}

std::vector<double> CriticalPointsDetector::calculate_derivative(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data
) {
    std::vector<double> derivative;
    derivative.reserve(signal_data.size() - 1);
    
    for (size_t i = 1; i < signal_data.size(); ++i) {
        double dt = time_data[i] - time_data[i - 1];
        double dy = signal_data[i] - signal_data[i - 1];
        
        if (dt > 0.0) {
            derivative.push_back(dy / dt);
        } else {
            derivative.push_back(0.0);
        }
    }
    
    return derivative;
}

std::vector<double> CriticalPointsDetector::calculate_second_derivative(
    const std::vector<double>& time_data,
    const std::vector<double>& signal_data
) {
    auto first_deriv = calculate_derivative(time_data, signal_data);
    
    std::vector<double> second_deriv;
    second_deriv.reserve(first_deriv.size() - 1);
    
    for (size_t i = 1; i < first_deriv.size(); ++i) {
        double dt = (time_data[i + 1] - time_data[i]) +
                    (time_data[i] - time_data[i - 1]);
        dt /= 2.0;
        
        double dy = first_deriv[i] - first_deriv[i - 1];
        
        if (dt > 0.0) {
            second_deriv.push_back(dy / dt);
        } else {
            second_deriv.push_back(0.0);
        }
    }
    
    return second_deriv;
}

std::vector<size_t> CriticalPointsDetector::find_zero_crossings(
    const std::vector<double>& data
) {
    std::vector<size_t> crossings;
    
    for (size_t i = 1; i < data.size(); ++i) {
        if ((data[i - 1] < 0.0 && data[i] >= 0.0) ||
            (data[i - 1] > 0.0 && data[i] <= 0.0)) {
            crossings.push_back(i);
        }
    }
    
    return crossings;
}

} // namespace timegraph
