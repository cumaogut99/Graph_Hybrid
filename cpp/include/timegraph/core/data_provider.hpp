#pragma once

/**
 * @file data_provider.hpp
 * @brief Unified data access API for TimeGraph
 * 
 * DataProvider is the main entry point for all data access operations.
 * It combines data source abstraction, caching, and statistics calculation
 * into a single, easy-to-use interface.
 * 
 * Features:
 * - Unified API for all data sources (file, live, network)
 * - Automatic caching with LRU eviction
 * - Efficient statistics calculation
 * - Value interpolation for cursor support
 * - Batch operations for multiple signals
 * - Thread-safe operations
 * 
 * Usage:
 * @code
 *   auto source = std::make_shared<MpaiDataSource>("data.mpai");
 *   DataProvider provider(source, 512);  // 512 MB cache
 *   
 *   // Get statistics
 *   auto stats = provider.get_statistics("Signal1", 0, 100000);
 *   
 *   // Get value at time (for cursor)
 *   double value = provider.interpolate_at("Signal1", 12345.678);
 *   
 *   // Batch operations
 *   auto all_stats = provider.get_statistics_batch(
 *       {"Signal1", "Signal2", "Signal3"}, 0, 100000
 *   );
 * @endcode
 */

#include "timegraph/core/types.hpp"
#include "timegraph/core/data_source.hpp"
#include "timegraph/core/data_cache.hpp"
#include "timegraph/processing/statistics_engine.hpp"
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <mutex>

namespace timegraph {

/**
 * @brief Unified data access provider
 * 
 * This class provides a high-level API for accessing signal data,
 * computing statistics, and interpolating values. It handles caching
 * transparently and optimizes batch operations.
 */
class DataProvider {
public:
    // ========================================================================
    // Construction
    // ========================================================================
    
    /**
     * @brief Construct DataProvider with a data source
     * @param source Data source (file, live, or network)
     * @param cache_size_mb Cache memory limit in megabytes
     */
    explicit DataProvider(
        std::shared_ptr<IDataSource> source,
        size_t cache_size_mb = constants::DEFAULT_CACHE_MEMORY_MB
    );
    
    /// Destructor
    ~DataProvider() = default;
    
    // Non-copyable
    DataProvider(const DataProvider&) = delete;
    DataProvider& operator=(const DataProvider&) = delete;
    
    // Movable
    DataProvider(DataProvider&&) = default;
    DataProvider& operator=(DataProvider&&) = default;
    
    // ========================================================================
    // Data Access
    // ========================================================================
    
    /**
     * @brief Get data range for a signal
     * 
     * Retrieves data from cache if available, otherwise loads from source.
     * Uses tile-based caching for efficient access patterns.
     * 
     * @param signal_name Signal identifier
     * @param start_row Starting row index
     * @param row_count Number of rows to retrieve
     * @return DataChunk containing the requested data
     */
    DataChunk get_data(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /**
     * @brief Get data ranges for multiple signals
     * 
     * More efficient than calling get_data() multiple times,
     * as it can parallelize I/O operations.
     * 
     * @param signal_names List of signal identifiers
     * @param start_row Starting row index (same for all)
     * @param row_count Number of rows to retrieve
     * @return Map of signal_name -> DataChunk
     */
    std::map<std::string, DataChunk> get_data_batch(
        const std::vector<std::string>& signal_names,
        uint64_t start_row,
        uint64_t row_count
    );
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    /**
     * @brief Calculate statistics for a signal range
     * 
     * Computes statistics using C++ engine. Results are cached
     * for repeated queries with the same parameters.
     * 
     * @param signal_name Signal identifier
     * @param start_row Starting row index
     * @param row_count Number of rows to analyze
     * @return ColumnStatistics with computed values
     */
    ColumnStatistics get_statistics(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /**
     * @brief Calculate statistics for multiple signals
     * 
     * Parallelizes computation across signals for better performance.
     * 
     * @param signal_names List of signal identifiers
     * @param start_row Starting row index
     * @param row_count Number of rows to analyze
     * @return Map of signal_name -> ColumnStatistics
     */
    std::map<std::string, ColumnStatistics> get_statistics_batch(
        const std::vector<std::string>& signal_names,
        uint64_t start_row,
        uint64_t row_count
    );
    
    // ========================================================================
    // Value Lookup (Cursor Support)
    // ========================================================================
    
    /**
     * @brief Get exact value at row index
     * 
     * Fast lookup for a single value. Uses cache when possible.
     * 
     * @param signal_name Signal identifier
     * @param row_index Row index
     * @return Value at the specified row
     */
    double get_value_at(const std::string& signal_name, uint64_t row_index);
    
    /**
     * @brief Get interpolated value at time
     * 
     * Performs linear interpolation between adjacent samples.
     * Essential for smooth cursor value display.
     * 
     * @param signal_name Signal identifier
     * @param time Time value (seconds)
     * @return Interpolated value
     */
    double interpolate_at(const std::string& signal_name, double time);
    
    /**
     * @brief Get interpolated values at multiple times
     * 
     * Batch interpolation for efficiency.
     * 
     * @param signal_name Signal identifier
     * @param times Vector of time values
     * @return Vector of interpolated values
     */
    std::vector<double> interpolate_batch(
        const std::string& signal_name,
        const std::vector<double>& times
    );
    
    /**
     * @brief Get interpolation result with details
     * 
     * Returns full interpolation information including indices and weight.
     * 
     * @param signal_name Signal identifier
     * @param time Time value
     * @return InterpolationResult with value and metadata
     */
    InterpolationResult interpolate_detailed(
        const std::string& signal_name,
        double time
    );
    
    // ========================================================================
    // Metadata
    // ========================================================================
    
    /**
     * @brief Get list of all signal names
     * @return Vector of signal name strings
     */
    std::vector<std::string> get_signal_names() const;
    
    /**
     * @brief Get metadata for a signal
     * @param name Signal identifier
     * @return SignalMetadata structure
     */
    SignalMetadata get_signal_metadata(const std::string& name) const;
    
    /**
     * @brief Get total number of samples
     * @return Total sample count
     */
    uint64_t get_total_samples() const;
    
    /**
     * @brief Get sample rate
     * @return Sample rate in Hz
     */
    double get_sample_rate() const;
    
    /**
     * @brief Check if signal exists
     * @param name Signal identifier
     * @return true if signal exists
     */
    bool has_signal(const std::string& name) const;
    
    /**
     * @brief Get number of signals
     * @return Signal count
     */
    size_t get_signal_count() const;
    
    // ========================================================================
    // Time Conversion
    // ========================================================================
    
    /**
     * @brief Convert time to row index
     * @param time Time in seconds
     * @return Corresponding row index
     */
    uint64_t time_to_row(double time) const;
    
    /**
     * @brief Convert row index to time
     * @param row Row index
     * @return Corresponding time in seconds
     */
    double row_to_time(uint64_t row) const;
    
    // ========================================================================
    // Cache Management
    // ========================================================================
    
    /**
     * @brief Clear all caches
     */
    void clear_cache();
    
    /**
     * @brief Clear data cache only
     */
    void clear_data_cache();
    
    /**
     * @brief Clear statistics cache only
     */
    void clear_stats_cache();
    
    /**
     * @brief Set cache memory limit
     * @param mb Memory limit in megabytes
     */
    void set_cache_size(size_t mb);
    
    /**
     * @brief Get cache statistics
     * @return CacheStats structure
     */
    CacheStats get_cache_stats() const;
    
    /**
     * @brief Prefetch data around a time point
     * 
     * Loads tiles around the specified time for smooth scrolling.
     * 
     * @param signal_name Signal to prefetch
     * @param center_time Center time for prefetching
     */
    void prefetch(const std::string& signal_name, double center_time);
    
    // ========================================================================
    // Source Information
    // ========================================================================
    
    /**
     * @brief Get source type
     * @return "file", "live", or "network"
     */
    std::string get_source_type() const;
    
    /**
     * @brief Get source path
     * @return File path or connection string
     */
    std::string get_source_path() const;
    
    /**
     * @brief Check if source is live
     * @return true if source provides real-time data
     */
    bool is_live() const;
    
    /**
     * @brief Get the underlying data source
     * @return Shared pointer to data source
     */
    std::shared_ptr<IDataSource> get_source() const { return source_; }
    
private:
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /// Load data from source (internal, handles caching)
    DataChunk load_data_internal(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /// Calculate statistics (internal, handles caching)
    ColumnStatistics calculate_stats_internal(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /// Get tile containing row (uses cache)
    DataChunk get_tile(const std::string& signal_name, uint64_t row);
    
    // ========================================================================
    // Data Members
    // ========================================================================
    
    std::shared_ptr<IDataSource> source_;   ///< Underlying data source
    DataCache data_cache_;                   ///< Data chunk cache
    StatisticsCache stats_cache_;            ///< Statistics cache
    
    // Cached metadata
    mutable std::vector<std::string> signal_names_;
    mutable std::map<std::string, SignalMetadata> metadata_cache_;
    mutable bool metadata_loaded_ = false;
    mutable std::mutex metadata_mutex_;
};

} // namespace timegraph

