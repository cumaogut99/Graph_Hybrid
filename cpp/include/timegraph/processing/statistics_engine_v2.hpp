#pragma once

#include "timegraph/processing/statistics_engine.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <future>

namespace timegraph {

/**
 * Enhanced Statistics Engine (V2)
 * 
 * Improvements over V1:
 * - Batch processing (multiple signals in parallel)
 * - Progress reporting for long operations
 * - Async API for non-blocking UI
 * - Smart caching (LRU cache for computed ranges)
 * - Multi-threaded processing (OpenMP)
 * 
 * Performance targets:
 * - 1M rows: <500ms (multi-threaded)
 * - 10M rows: <3s (with caching)
 * - 50M rows: <15s (streaming + parallel)
 */
class StatisticsEngineV2 {
public:
    /**
     * Calculate statistics for multiple signals in parallel
     * 
     * Example:
     *   auto stats = StatisticsEngineV2::calculate_batch(
     *       reader, {"Signal1", "Signal2", "Signal3"}, 0, 100000
     *   );
     *   // Returns: {"Signal1": stats1, "Signal2": stats2, "Signal3": stats3}
     * 
     * @param reader MPAI reader
     * @param column_names List of signal names
     * @param start_row Starting row index
     * @param row_count Number of rows to process (0 = all)
     * @param max_threads Maximum threads to use (0 = auto-detect)
     * @return Map of signal name -> statistics
     */
    static std::map<std::string, ColumnStatistics> calculate_batch(
        mpai::MpaiReader& reader,
        const std::vector<std::string>& column_names,
        uint64_t start_row = 0,
        uint64_t row_count = 0,
        uint32_t max_threads = 0
    );
    
    /**
     * Calculate statistics with progress callback
     * 
     * Useful for long operations to update UI progress bar
     * 
     * Example:
     *   auto stats = StatisticsEngineV2::calculate_with_progress(
     *       reader, "Signal1", 0, 10000000,
     *       [](float progress) {
     *           std::cout << "Progress: " << progress * 100 << "%" << std::endl;
     *       }
     *   );
     * 
     * @param reader MPAI reader
     * @param column_name Signal name
     * @param start_row Starting row index
     * @param row_count Number of rows
     * @param progress_callback Called periodically with progress (0.0-1.0)
     * @return Statistics result
     */
    static ColumnStatistics calculate_with_progress(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        uint64_t start_row,
        uint64_t row_count,
        std::function<void(float)> progress_callback
    );
    
    /**
     * Async API: Calculate statistics in background thread
     * 
     * Returns immediately with a future object.
     * UI can continue responding while calculation happens in background.
     * 
     * Example:
     *   auto future = StatisticsEngineV2::calculate_async(reader, "Signal1", 0, 1000000);
     *   // ... do other UI work ...
     *   auto stats = future.get();  // Wait for result
     * 
     * @param reader MPAI reader
     * @param column_name Signal name
     * @param start_row Starting row index
     * @param row_count Number of rows
     * @return Future object containing statistics
     */
    static std::future<ColumnStatistics> calculate_async(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /**
     * Multi-threaded range calculation
     * 
     * Splits the range into chunks and processes them in parallel using OpenMP.
     * Automatically combines results from all chunks.
     * 
     * @param reader MPAI reader
     * @param column_name Signal name
     * @param start_row Starting row index
     * @param row_count Number of rows
     * @param max_threads Maximum threads (0 = hardware_concurrency)
     * @return Combined statistics
     */
    static ColumnStatistics calculate_parallel(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        uint64_t start_row,
        uint64_t row_count,
        uint32_t max_threads = 0
    );
    
    /**
     * Cached statistics calculation
     * 
     * Checks LRU cache before computing. If result exists, returns immediately.
     * Otherwise, computes and stores in cache.
     * 
     * Cache key: (signal_name, start_row, row_count)
     * Cache size: Configurable (default 100 entries)
     * 
     * @param reader MPAI reader
     * @param column_name Signal name
     * @param start_row Starting row index
     * @param row_count Number of rows
     * @return Statistics (from cache or newly computed)
     */
    static ColumnStatistics calculate_cached(
        mpai::MpaiReader& reader,
        const std::string& column_name,
        uint64_t start_row,
        uint64_t row_count
    );
    
    /**
     * Clear the statistics cache
     * 
     * Call this when:
     * - File is closed
     * - Filters are applied
     * - Data is modified
     */
    static void clear_cache();
    
    /**
     * Configure cache settings
     * 
     * @param max_entries Maximum cache entries (default 100)
     * @param memory_limit_mb Maximum cache memory in MB (default 100)
     */
    static void configure_cache(size_t max_entries, size_t memory_limit_mb);
    
    /**
     * Get cache statistics (hits, misses, size)
     * 
     * Useful for performance tuning and debugging
     * 
     * @return Map of cache statistics
     */
    static std::map<std::string, size_t> get_cache_stats();

private:
    // Cache implementation (hidden)
    struct CacheImpl;
    static CacheImpl& get_cache();
    
    // Helper: Combine statistics from multiple chunks
    static ColumnStatistics combine_statistics(
        const std::vector<ColumnStatistics>& chunk_stats
    );
};

/**
 * Statistics cache key
 */
struct StatsCacheKey {
    std::string signal_name;
    uint64_t start_row;
    uint64_t row_count;
    
    bool operator==(const StatsCacheKey& other) const {
        return signal_name == other.signal_name &&
               start_row == other.start_row &&
               row_count == other.row_count;
    }
};

} // namespace timegraph

// Hash function for cache key (for std::unordered_map)
namespace std {
    template <>
    struct hash<timegraph::StatsCacheKey> {
        size_t operator()(const timegraph::StatsCacheKey& key) const {
            size_t h1 = std::hash<std::string>{}(key.signal_name);
            size_t h2 = std::hash<uint64_t>{}(key.start_row);
            size_t h3 = std::hash<uint64_t>{}(key.row_count);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

