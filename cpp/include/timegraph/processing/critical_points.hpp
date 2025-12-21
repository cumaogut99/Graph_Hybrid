#pragma once

#include <vector>
#include <cstddef>
#include <optional>

namespace timegraph {

/// A critical point in time series data
struct CriticalPoint {
    size_t index;           ///< Index in original data
    double time;            ///< Time value
    double value;           ///< Data value
    
    enum class Type {
        LOCAL_MAX,          ///< Local maximum (peak)
        LOCAL_MIN,          ///< Local minimum (valley)
        SUDDEN_CHANGE,      ///< Sudden change (derivative spike)
        LIMIT_VIOLATION,    ///< Crosses warning/error limit
        INFLECTION          ///< Inflection point (curvature change)
    };
    
    Type type;              ///< Type of critical point
    double significance;    ///< Significance score (0.0 to 1.0)
    
    CriticalPoint()
        : index(0), time(0.0), value(0.0)
        , type(Type::LOCAL_MAX), significance(0.0)
    {}
    
    CriticalPoint(size_t idx, double t, double v, Type tp, double sig = 1.0)
        : index(idx), time(t), value(v), type(tp), significance(sig)
    {}
};

/// Configuration for critical points detection
struct CriticalPointsConfig {
    // Peak/Valley detection
    bool detect_peaks = true;           ///< Detect local maxima
    bool detect_valleys = true;         ///< Detect local minima
    size_t window_size = 10;            ///< Window size for local extrema
    double min_prominence = 0.0;        ///< Minimum prominence (0 = auto)
    
    // Sudden change detection
    bool detect_sudden_changes = true;  ///< Detect derivative spikes
    double change_threshold = 3.0;      ///< Threshold in std devs
    
    // Limit violation detection
    bool detect_limit_violations = true; ///< Detect limit crossings
    std::vector<double> warning_limits;  ///< Warning limits to check
    std::vector<double> error_limits;    ///< Error limits to check
    
    // Inflection point detection
    bool detect_inflections = false;     ///< Detect curvature changes
    
    // General
    size_t max_points = 1000;           ///< Maximum critical points to return
    double min_significance = 0.1;      ///< Minimum significance (0-1)
    
    CriticalPointsConfig() = default;
};

/// High-performance critical points detector
class CriticalPointsDetector {
public:
    /// Constructor
    CriticalPointsDetector() = default;
    
    /// Detect critical points in time series data
    /// @param time_data Time values
    /// @param signal_data Signal values
    /// @param config Detection configuration
    /// @return Vector of critical points (sorted by index)
    static std::vector<CriticalPoint> detect(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data,
        const CriticalPointsConfig& config = CriticalPointsConfig()
    );
    
    /// Detect critical points with Arrow Compute acceleration
    /// Same as detect() but uses Arrow for intermediate calculations
    static std::vector<CriticalPoint> detect_arrow(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data,
        const CriticalPointsConfig& config = CriticalPointsConfig()
    );
    
    /// Detect local extrema (peaks and valleys)
    static std::vector<CriticalPoint> detect_local_extrema(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data,
        size_t window_size,
        double min_prominence = 0.0
    );
    
    /// Detect sudden changes (derivative spikes)
    static std::vector<CriticalPoint> detect_sudden_changes(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data,
        double threshold_sigma = 3.0
    );
    
    /// Detect limit violations (warning/error crossings)
    static std::vector<CriticalPoint> detect_limit_violations(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data,
        const std::vector<double>& limits
    );
    
    /// Merge multiple critical point vectors and remove duplicates
    /// @param points_list List of critical point vectors
    /// @param max_distance Maximum distance to consider points as duplicates
    /// @return Merged and deduplicated vector
    static std::vector<CriticalPoint> merge_and_deduplicate(
        const std::vector<std::vector<CriticalPoint>>& points_list,
        size_t max_distance = 5
    );
    
    /// Filter critical points by significance
    static std::vector<CriticalPoint> filter_by_significance(
        const std::vector<CriticalPoint>& points,
        double min_significance,
        size_t max_points = 0  // 0 = unlimited
    );

private:
    /// Calculate prominence of a peak/valley
    static double calculate_prominence(
        const std::vector<double>& data,
        size_t index,
        bool is_max
    );
    
    /// Calculate first derivative (finite difference)
    static std::vector<double> calculate_derivative(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data
    );
    
    /// Calculate second derivative (curvature)
    static std::vector<double> calculate_second_derivative(
        const std::vector<double>& time_data,
        const std::vector<double>& signal_data
    );
    
    /// Find zero crossings in data
    static std::vector<size_t> find_zero_crossings(
        const std::vector<double>& data
    );
};

} // namespace timegraph
