/**
 * @file data_cache.cpp
 * @brief Implementation of LRU data cache
 */

#include "timegraph/core/data_cache.hpp"
#include <algorithm>
#include <thread>
#include <future>

namespace timegraph {

// ============================================================================
// DataCache Implementation
// ============================================================================

DataCache::DataCache(size_t memory_limit_mb)
    : memory_limit_(memory_limit_mb * 1024 * 1024)  // Convert to bytes
{
}

std::optional<DataChunk> DataCache::get(const CacheKey& key) {
    // Try shared lock first (read-only)
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            ++misses_;
            return std::nullopt;
        }
        ++hits_;
    }
    
    // Found - need exclusive lock to update LRU order
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        // Race condition - entry was removed
        return std::nullopt;
    }
    
    // Move to front (most recently used)
    move_to_front(it->second);
    
    return it->second->second.data;
}

void DataCache::put(const CacheKey& key, const DataChunk& data) {
    DataChunk copy = data;
    put(key, std::move(copy));
}

void DataCache::put(const CacheKey& key, DataChunk&& data) {
    size_t chunk_size = estimate_chunk_size(data);
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Check if key already exists
    auto existing = cache_map_.find(key);
    if (existing != cache_map_.end()) {
        // Update existing entry
        size_t old_size = existing->second->second.memory_size;
        current_memory_ -= old_size;
        
        existing->second->second.data = std::move(data);
        existing->second->second.memory_size = chunk_size;
        current_memory_ += chunk_size;
        
        move_to_front(existing->second);
        return;
    }
    
    // Evict if necessary
    evict_if_needed(chunk_size);
    
    // Insert new entry at front
    CacheEntry entry(std::move(data), chunk_size);
    lru_list_.emplace_front(key, std::move(entry));
    cache_map_[key] = lru_list_.begin();
    current_memory_ += chunk_size;
}

bool DataCache::contains(const CacheKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_map_.find(key) != cache_map_.end();
}

bool DataCache::remove(const CacheKey& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        return false;
    }
    
    current_memory_ -= it->second->second.memory_size;
    lru_list_.erase(it->second);
    cache_map_.erase(it);
    
    return true;
}

size_t DataCache::remove_signal(const std::string& signal_name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    size_t removed = 0;
    auto it = lru_list_.begin();
    while (it != lru_list_.end()) {
        if (it->first.signal_name == signal_name) {
            current_memory_ -= it->second.memory_size;
            cache_map_.erase(it->first);
            it = lru_list_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    
    return removed;
}

DataChunk DataCache::get_tile(
    const std::string& signal_name,
    uint64_t row,
    std::function<DataChunk(uint64_t, uint64_t)> loader
) {
    uint64_t tile_start = get_tile_start(row);
    CacheKey key(signal_name, tile_start, constants::DEFAULT_TILE_SIZE);
    
    // Try to get from cache
    auto cached = get(key);
    if (cached) {
        return *cached;
    }
    
    // Cache miss - load tile
    DataChunk chunk = loader(tile_start, constants::DEFAULT_TILE_SIZE);
    put(key, chunk);
    
    return chunk;
}

void DataCache::prefetch(
    const std::string& signal_name,
    uint64_t center_row,
    std::function<DataChunk(uint64_t, uint64_t)> loader,
    size_t num_tiles
) {
    uint64_t center_tile = get_tile_start(center_row);
    size_t tile_size = constants::DEFAULT_TILE_SIZE;
    
    // Launch async prefetch tasks
    std::vector<std::future<void>> futures;
    
    for (size_t i = 1; i <= num_tiles; ++i) {
        // Prefetch tiles before center
        if (center_tile >= i * tile_size) {
            uint64_t tile_start = center_tile - i * tile_size;
            CacheKey key(signal_name, tile_start, tile_size);
            
            if (!contains(key)) {
                futures.push_back(std::async(std::launch::async, [this, key, loader, tile_start, tile_size]() {
                    DataChunk chunk = loader(tile_start, tile_size);
                    put(key, std::move(chunk));
                }));
            }
        }
        
        // Prefetch tiles after center
        uint64_t tile_start = center_tile + i * tile_size;
        CacheKey key(signal_name, tile_start, tile_size);
        
        if (!contains(key)) {
            futures.push_back(std::async(std::launch::async, [this, key, loader, tile_start, tile_size]() {
                DataChunk chunk = loader(tile_start, tile_size);
                put(key, std::move(chunk));
            }));
        }
    }
    
    // Wait for all prefetch tasks (optional - could be fire-and-forget)
    for (auto& f : futures) {
        f.wait();
    }
}

void DataCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_map_.clear();
    lru_list_.clear();
    current_memory_ = 0;
}

void DataCache::set_memory_limit(size_t mb) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    memory_limit_ = mb * 1024 * 1024;
    
    // Evict if over new limit
    while (current_memory_ > memory_limit_ && !lru_list_.empty()) {
        auto& back = lru_list_.back();
        current_memory_ -= back.second.memory_size;
        cache_map_.erase(back.first);
        lru_list_.pop_back();
    }
}

size_t DataCache::get_memory_limit() const {
    return memory_limit_ / (1024 * 1024);
}

size_t DataCache::get_memory_usage() const {
    return current_memory_;
}

size_t DataCache::get_entry_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_map_.size();
}

CacheStats DataCache::get_stats() const {
    CacheStats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.entries = get_entry_count();
    stats.memory_bytes = current_memory_;
    stats.max_memory = memory_limit_;
    stats.update_hit_rate();
    return stats;
}

void DataCache::reset_stats() {
    hits_ = 0;
    misses_ = 0;
}

void DataCache::evict_if_needed(size_t new_size) {
    // Must be called with exclusive lock held
    while (current_memory_ + new_size > memory_limit_ && !lru_list_.empty()) {
        auto& back = lru_list_.back();
        current_memory_ -= back.second.memory_size;
        cache_map_.erase(back.first);
        lru_list_.pop_back();
    }
}

void DataCache::move_to_front(LRUIterator it) {
    // Must be called with exclusive lock held
    if (it != lru_list_.begin()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it);
    }
}

size_t DataCache::estimate_chunk_size(const DataChunk& chunk) {
    // Estimate memory: data + time_data + overhead
    size_t size = chunk.data.size() * sizeof(double);
    size += chunk.time_data.size() * sizeof(double);
    size += chunk.signal_name.size();
    size += sizeof(DataChunk);  // Struct overhead
    size += 64;  // Additional overhead (allocator, etc.)
    return size;
}

// ============================================================================
// StatisticsCache Implementation
// ============================================================================

StatisticsCache::StatisticsCache(size_t max_entries)
    : max_entries_(max_entries)
{
}

std::optional<ColumnStatistics> StatisticsCache::get(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) const {
    StatsCacheKey key{signal_name, start_row, row_count};
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        ++misses_;
        return std::nullopt;
    }
    
    ++hits_;
    return it->second;
}

void StatisticsCache::put(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count,
    const ColumnStatistics& stats
) {
    StatsCacheKey key{signal_name, start_row, row_count};
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Check if already exists
    auto existing = cache_.find(key);
    if (existing != cache_.end()) {
        existing->second = stats;
        // Move to front of LRU
        auto lru_it = lru_map_[key];
        lru_.splice(lru_.begin(), lru_, lru_it);
        return;
    }
    
    // Evict if at capacity
    while (cache_.size() >= max_entries_ && !lru_.empty()) {
        auto& oldest = lru_.back();
        cache_.erase(oldest);
        lru_map_.erase(oldest);
        lru_.pop_back();
    }
    
    // Insert new entry
    cache_[key] = stats;
    lru_.push_front(key);
    lru_map_[key] = lru_.begin();
}

void StatisticsCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_.clear();
    lru_.clear();
    lru_map_.clear();
}

double StatisticsCache::get_hit_rate() const {
    size_t total = hits_ + misses_;
    return total > 0 ? static_cast<double>(hits_) / total : 0.0;
}

} // namespace timegraph

