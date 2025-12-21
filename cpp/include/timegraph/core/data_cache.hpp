#pragma once

/**
 * @file data_cache.hpp
 * @brief LRU cache for signal data chunks
 * 
 * Provides efficient caching of data chunks to minimize disk I/O.
 * Uses LRU (Least Recently Used) eviction policy with memory limits.
 * 
 * Features:
 * - Thread-safe operations
 * - Configurable memory limit
 * - Tile-based caching (aligned chunks)
 * - Prefetching support
 * - Cache statistics
 */

#include "timegraph/core/types.hpp"
#include "timegraph/processing/statistics_engine.hpp"
#include <unordered_map>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <functional>
#include <atomic>

namespace timegraph {

/**
 * @brief Thread-safe LRU cache for DataChunks
 * 
 * This cache stores data chunks with automatic eviction when
 * memory limits are exceeded. It uses a combination of hash map
 * for O(1) lookup and doubly-linked list for LRU ordering.
 * 
 * Thread Safety:
 * - All public methods are thread-safe
 * - Uses shared_mutex for read/write locking
 * - Readers can access concurrently
 * - Writers have exclusive access
 * 
 * Memory Management:
 * - Tracks memory usage of all cached chunks
 * - Automatically evicts LRU entries when limit exceeded
 * - Memory limit is configurable at runtime
 * 
 * Example usage:
 * @code
 *   DataCache cache(512);  // 512 MB limit
 *   
 *   CacheKey key("Signal1", 0, 10000);
 *   auto chunk = cache.get(key);
 *   
 *   if (!chunk) {
 *       // Cache miss - load from source
 *       chunk = source->read_range("Signal1", 0, 10000);
 *       cache.put(key, *chunk);
 *   }
 * @endcode
 */
class DataCache {
public:
    /**
     * @brief Construct cache with memory limit
     * @param memory_limit_mb Maximum memory usage in megabytes
     */
    explicit DataCache(size_t memory_limit_mb = constants::DEFAULT_CACHE_MEMORY_MB);
    
    /// Destructor
    ~DataCache() = default;
    
    // Non-copyable, non-movable (due to mutex)
    DataCache(const DataCache&) = delete;
    DataCache& operator=(const DataCache&) = delete;
    DataCache(DataCache&&) = delete;
    DataCache& operator=(DataCache&&) = delete;
    
    // ========================================================================
    // Cache Operations
    // ========================================================================
    
    /**
     * @brief Get data from cache
     * 
     * Returns the cached data if found, updates LRU order.
     * Thread-safe: multiple readers can access simultaneously.
     * 
     * @param key Cache key (signal name + row range)
     * @return Optional containing DataChunk if found, empty otherwise
     */
    std::optional<DataChunk> get(const CacheKey& key);
    
    /**
     * @brief Put data into cache
     * 
     * Stores the data chunk, evicting old entries if necessary.
     * Thread-safe: exclusive access during write.
     * 
     * @param key Cache key
     * @param data Data chunk to cache
     */
    void put(const CacheKey& key, const DataChunk& data);
    
    /**
     * @brief Put data into cache (move semantics)
     * @param key Cache key
     * @param data Data chunk to cache (moved)
     */
    void put(const CacheKey& key, DataChunk&& data);
    
    /**
     * @brief Check if key exists in cache
     * @param key Cache key
     * @return true if key is cached
     */
    bool contains(const CacheKey& key) const;
    
    /**
     * @brief Remove entry from cache
     * @param key Cache key
     * @return true if entry was removed
     */
    bool remove(const CacheKey& key);
    
    /**
     * @brief Remove all entries for a signal
     * @param signal_name Signal to remove
     * @return Number of entries removed
     */
    size_t remove_signal(const std::string& signal_name);
    
    // ========================================================================
    // Tile-Based Access
    // ========================================================================
    
    /**
     * @brief Get tile containing the given row
     * 
     * Tiles are aligned chunks of TILE_SIZE rows.
     * This method returns the tile that contains the specified row.
     * 
     * @param signal_name Signal name
     * @param row Row index
     * @param loader Function to load tile if not cached
     * @return DataChunk for the tile
     */
    DataChunk get_tile(
        const std::string& signal_name,
        uint64_t row,
        std::function<DataChunk(uint64_t, uint64_t)> loader
    );
    
    /**
     * @brief Calculate tile start row for a given row
     * @param row Row index
     * @return Start row of the containing tile
     */
    static uint64_t get_tile_start(uint64_t row) {
        return (row / constants::DEFAULT_TILE_SIZE) * constants::DEFAULT_TILE_SIZE;
    }
    
    /**
     * @brief Get tile size
     * @return Tile size in rows
     */
    static size_t get_tile_size() {
        return constants::DEFAULT_TILE_SIZE;
    }
    
    // ========================================================================
    // Prefetching
    // ========================================================================
    
    /**
     * @brief Prefetch tiles around a center point
     * 
     * Asynchronously loads tiles adjacent to the specified row.
     * Useful for smooth scrolling/panning.
     * 
     * @param signal_name Signal to prefetch
     * @param center_row Center row for prefetching
     * @param loader Function to load data
     * @param num_tiles Number of tiles to prefetch (before and after)
     */
    void prefetch(
        const std::string& signal_name,
        uint64_t center_row,
        std::function<DataChunk(uint64_t, uint64_t)> loader,
        size_t num_tiles = 2
    );
    
    // ========================================================================
    // Management
    // ========================================================================
    
    /**
     * @brief Clear all cached data
     */
    void clear();
    
    /**
     * @brief Set memory limit
     * @param mb Memory limit in megabytes
     */
    void set_memory_limit(size_t mb);
    
    /**
     * @brief Get current memory limit
     * @return Memory limit in megabytes
     */
    size_t get_memory_limit() const;
    
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes
     */
    size_t get_memory_usage() const;
    
    /**
     * @brief Get number of cached entries
     * @return Entry count
     */
    size_t get_entry_count() const;
    
    /**
     * @brief Get cache statistics
     * @return CacheStats structure
     */
    CacheStats get_stats() const;
    
    /**
     * @brief Reset statistics counters
     */
    void reset_stats();
    
private:
    // ========================================================================
    // Internal Types
    // ========================================================================
    
    struct CacheEntry {
        DataChunk data;
        size_t memory_size;
        
        CacheEntry() : memory_size(0) {}
        CacheEntry(DataChunk d, size_t size) 
            : data(std::move(d)), memory_size(size) {}
    };
    
    // LRU list: front = most recently used, back = least recently used
    using LRUList = std::list<std::pair<CacheKey, CacheEntry>>;
    using LRUIterator = LRUList::iterator;
    using CacheMap = std::unordered_map<CacheKey, LRUIterator>;
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /// Evict entries until we have room for new_size bytes
    void evict_if_needed(size_t new_size);
    
    /// Move entry to front of LRU list (most recently used)
    void move_to_front(LRUIterator it);
    
    /// Estimate memory size of a chunk
    static size_t estimate_chunk_size(const DataChunk& chunk);
    
    // ========================================================================
    // Data Members
    // ========================================================================
    
    CacheMap cache_map_;            ///< Hash map for O(1) lookup
    LRUList lru_list_;              ///< LRU ordering list
    
    size_t memory_limit_;           ///< Memory limit in bytes
    std::atomic<size_t> current_memory_{0};  ///< Current memory usage
    
    // Statistics
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    
    // Thread safety
    mutable std::shared_mutex mutex_;
};

// ============================================================================
// Statistics Cache
// ============================================================================

/**
 * @brief Specialized cache for computed statistics
 * 
 * Caches ColumnStatistics results to avoid recomputation.
 * Smaller and simpler than DataCache since statistics are small.
 */
class StatisticsCache {
public:
    /**
     * @brief Construct statistics cache
     * @param max_entries Maximum number of entries
     */
    explicit StatisticsCache(size_t max_entries = constants::DEFAULT_STATS_CACHE_SIZE);
    
    /**
     * @brief Get cached statistics
     * @param signal_name Signal name
     * @param start_row Start row
     * @param row_count Row count
     * @return Optional containing statistics if cached
     */
    std::optional<ColumnStatistics> get(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    ) const;
    
    /**
     * @brief Put statistics into cache
     * @param signal_name Signal name
     * @param start_row Start row
     * @param row_count Row count
     * @param stats Statistics to cache
     */
    void put(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count,
        const ColumnStatistics& stats
    );
    
    /**
     * @brief Clear all cached statistics
     */
    void clear();
    
    /**
     * @brief Get cache hit rate
     * @return Hit rate (0.0 to 1.0)
     */
    double get_hit_rate() const;
    
private:
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
    
    struct StatsCacheKeyHash {
        size_t operator()(const StatsCacheKey& k) const {
            return std::hash<std::string>()(k.signal_name) ^
                   (std::hash<uint64_t>()(k.start_row) << 1) ^
                   (std::hash<uint64_t>()(k.row_count) << 2);
        }
    };
    
    using StatsMap = std::unordered_map<StatsCacheKey, ColumnStatistics, StatsCacheKeyHash>;
    using StatsLRU = std::list<StatsCacheKey>;
    
    StatsMap cache_;
    StatsLRU lru_;
    std::unordered_map<StatsCacheKey, StatsLRU::iterator, StatsCacheKeyHash> lru_map_;
    
    size_t max_entries_;
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    
    mutable std::shared_mutex mutex_;
};

} // namespace timegraph

