#pragma once

/**
 * @file types.hpp
 * @brief Core data types for TimeGraph data layer
 * 
 * This file defines fundamental data structures used throughout the
 * TimeGraph C++ engine for data management and caching.
 */

#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <functional>

namespace timegraph {

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Contiguous chunk of signal data
 * 
 * Represents a slice of data from a signal, used for efficient
 * data transfer between C++ and Python, and for caching.
 */
struct DataChunk {
    std::vector<double> data;       ///< Signal values
    std::vector<double> time_data;  ///< Time values (optional, for interpolation)
    uint64_t start_row;             ///< Starting row index in source
    uint64_t row_count;             ///< Number of rows in this chunk
    std::string signal_name;        ///< Signal identifier
    
    /// Check if chunk is empty
    bool empty() const { return data.empty(); }
    
    /// Get number of samples
    size_t size() const { return data.size(); }
    
    /// Array access operator
    double operator[](size_t i) const { return data[i]; }
    
    /// Get time at index (returns index if no time data)
    double time_at(size_t i) const {
        if (!time_data.empty() && i < time_data.size()) {
            return time_data[i];
        }
        return static_cast<double>(start_row + i);
    }
    
    /// Default constructor
    DataChunk() : start_row(0), row_count(0) {}
    
    /// Constructor with signal name
    explicit DataChunk(const std::string& name) 
        : start_row(0), row_count(0), signal_name(name) {}
    
    /// Constructor with data
    DataChunk(const std::string& name, std::vector<double> d, 
              uint64_t start, uint64_t count)
        : data(std::move(d)), start_row(start), row_count(count), signal_name(name) {}
};

/**
 * @brief Metadata for a signal/channel
 * 
 * Contains descriptive information about a signal including
 * its sampling characteristics and value range.
 */
struct SignalMetadata {
    std::string name;               ///< Signal name/identifier
    std::string unit;               ///< Physical unit (e.g., "V", "m/s")
    std::string description;        ///< Human-readable description
    double sample_rate;             ///< Samples per second (Hz)
    uint64_t total_samples;         ///< Total number of samples
    double min_value;               ///< Minimum value in dataset
    double max_value;               ///< Maximum value in dataset
    double start_time;              ///< Start time (seconds)
    double end_time;                ///< End time (seconds)
    
    /// Custom attributes (key-value pairs)
    std::map<std::string, std::string> attributes;
    
    /// Default constructor
    SignalMetadata() 
        : sample_rate(1.0), total_samples(0)
        , min_value(0.0), max_value(0.0)
        , start_time(0.0), end_time(0.0) {}
    
    /// Constructor with name
    explicit SignalMetadata(const std::string& n) 
        : name(n), sample_rate(1.0), total_samples(0)
        , min_value(0.0), max_value(0.0)
        , start_time(0.0), end_time(0.0) {}
    
    /// Get duration in seconds
    double duration() const { return end_time - start_time; }
    
    /// Get value range
    double range() const { return max_value - min_value; }
};

// ============================================================================
// Cache Types
// ============================================================================

/**
 * @brief Key for data cache lookups
 * 
 * Uniquely identifies a cached data chunk by signal name and row range.
 */
struct CacheKey {
    std::string signal_name;
    uint64_t start_row;
    uint64_t row_count;
    
    /// Equality comparison
    bool operator==(const CacheKey& other) const {
        return signal_name == other.signal_name &&
               start_row == other.start_row &&
               row_count == other.row_count;
    }
    
    /// Less-than comparison (for std::map)
    bool operator<(const CacheKey& other) const {
        if (signal_name != other.signal_name) return signal_name < other.signal_name;
        if (start_row != other.start_row) return start_row < other.start_row;
        return row_count < other.row_count;
    }
    
    /// Default constructor
    CacheKey() : start_row(0), row_count(0) {}
    
    /// Constructor
    CacheKey(const std::string& name, uint64_t start, uint64_t count)
        : signal_name(name), start_row(start), row_count(count) {}
};

/**
 * @brief Statistics about cache performance
 */
struct CacheStats {
    size_t hits;            ///< Number of cache hits
    size_t misses;          ///< Number of cache misses
    size_t entries;         ///< Current number of entries
    size_t memory_bytes;    ///< Current memory usage
    size_t max_memory;      ///< Maximum memory limit
    double hit_rate;        ///< Hit rate (0.0 - 1.0)
    
    /// Default constructor
    CacheStats() 
        : hits(0), misses(0), entries(0)
        , memory_bytes(0), max_memory(0), hit_rate(0.0) {}
    
    /// Calculate hit rate
    void update_hit_rate() {
        size_t total = hits + misses;
        hit_rate = total > 0 ? static_cast<double>(hits) / total : 0.0;
    }
};

// ============================================================================
// Result Types
// ============================================================================

/**
 * @brief Result of an interpolation operation
 */
struct InterpolationResult {
    double value;           ///< Interpolated value
    bool valid;             ///< Whether interpolation succeeded
    uint64_t left_index;    ///< Index of left neighbor
    uint64_t right_index;   ///< Index of right neighbor
    double weight;          ///< Interpolation weight (0.0 = left, 1.0 = right)
    
    /// Default constructor (invalid result)
    InterpolationResult() 
        : value(0.0), valid(false), left_index(0), right_index(0), weight(0.0) {}
    
    /// Constructor for valid result
    InterpolationResult(double v, uint64_t left, uint64_t right, double w)
        : value(v), valid(true), left_index(left), right_index(right), weight(w) {}
    
    /// Create invalid result
    static InterpolationResult invalid() { return InterpolationResult(); }
};

// ============================================================================
// Callback Types
// ============================================================================

/// Progress callback: receives progress (0.0 to 1.0)
using ProgressCallback = std::function<void(float)>;

/// Data callback: receives signal name and data chunk
using DataCallback = std::function<void(const std::string&, const DataChunk&)>;

/// Error callback: receives error message
using ErrorCallback = std::function<void(const std::string&)>;

// ============================================================================
// Constants
// ============================================================================

namespace constants {
    /// Default tile size for caching (rows per tile)
    constexpr size_t DEFAULT_TILE_SIZE = 10000;
    
    /// Default cache memory limit (512 MB)
    constexpr size_t DEFAULT_CACHE_MEMORY_MB = 512;
    
    /// Default statistics cache size (entries)
    constexpr size_t DEFAULT_STATS_CACHE_SIZE = 100;
    
    /// Maximum signals for batch operations
    constexpr size_t MAX_BATCH_SIGNALS = 100;
    
    /// Interpolation window size (samples around target)
    constexpr size_t INTERPOLATION_WINDOW = 100;
}

} // namespace timegraph

// ============================================================================
// Hash Functions (for std::unordered_map)
// ============================================================================

namespace std {
    template<>
    struct hash<timegraph::CacheKey> {
        size_t operator()(const timegraph::CacheKey& k) const {
            size_t h1 = hash<string>()(k.signal_name);
            size_t h2 = hash<uint64_t>()(k.start_row);
            size_t h3 = hash<uint64_t>()(k.row_count);
            // Combine hashes using FNV-1a style mixing
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

