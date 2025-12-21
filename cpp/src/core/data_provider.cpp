/**
 * @file data_provider.cpp
 * @brief Implementation of unified data provider
 */

#include "timegraph/core/data_provider.hpp"
#include <algorithm>
#include <cmath>
#include <future>
#include <thread>

namespace timegraph {

// ============================================================================
// Construction
// ============================================================================

DataProvider::DataProvider(
    std::shared_ptr<IDataSource> source,
    size_t cache_size_mb
)
    : source_(std::move(source))
    , data_cache_(cache_size_mb)
    , stats_cache_(constants::DEFAULT_STATS_CACHE_SIZE)
{
    if (!source_) {
        throw std::invalid_argument("DataProvider: source cannot be null");
    }
}

// ============================================================================
// Data Access
// ============================================================================

DataChunk DataProvider::get_data(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) {
    return load_data_internal(signal_name, start_row, row_count);
}

std::map<std::string, DataChunk> DataProvider::get_data_batch(
    const std::vector<std::string>& signal_names,
    uint64_t start_row,
    uint64_t row_count
) {
    if (signal_names.empty()) {
        return {};
    }
    
    // For small batches, sequential is fine
    if (signal_names.size() <= 2) {
        std::map<std::string, DataChunk> result;
        for (const auto& name : signal_names) {
            result[name] = get_data(name, start_row, row_count);
        }
        return result;
    }
    
    // For larger batches, parallelize
    std::map<std::string, DataChunk> result;
    std::vector<std::future<std::pair<std::string, DataChunk>>> futures;
    
    for (const auto& name : signal_names) {
        futures.push_back(std::async(std::launch::async, 
            [this, &name, start_row, row_count]() {
                return std::make_pair(name, get_data(name, start_row, row_count));
            }
        ));
    }
    
    for (auto& f : futures) {
        auto pair = f.get();
        result[pair.first] = std::move(pair.second);
    }
    
    return result;
}

// ============================================================================
// Statistics
// ============================================================================

ColumnStatistics DataProvider::get_statistics(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) {
    return calculate_stats_internal(signal_name, start_row, row_count);
}

std::map<std::string, ColumnStatistics> DataProvider::get_statistics_batch(
    const std::vector<std::string>& signal_names,
    uint64_t start_row,
    uint64_t row_count
) {
    if (signal_names.empty()) {
        return {};
    }
    
    // For small batches, sequential
    if (signal_names.size() <= 2) {
        std::map<std::string, ColumnStatistics> result;
        for (const auto& name : signal_names) {
            result[name] = get_statistics(name, start_row, row_count);
        }
        return result;
    }
    
    // Parallelize for larger batches
    std::map<std::string, ColumnStatistics> result;
    std::vector<std::future<std::pair<std::string, ColumnStatistics>>> futures;
    
    for (const auto& name : signal_names) {
        futures.push_back(std::async(std::launch::async,
            [this, &name, start_row, row_count]() {
                return std::make_pair(name, get_statistics(name, start_row, row_count));
            }
        ));
    }
    
    for (auto& f : futures) {
        auto pair = f.get();
        result[pair.first] = std::move(pair.second);
    }
    
    return result;
}

// ============================================================================
// Value Lookup
// ============================================================================

double DataProvider::get_value_at(const std::string& signal_name, uint64_t row_index) {
    // Get tile containing the row
    DataChunk tile = get_tile(signal_name, row_index);
    
    // Calculate offset within tile
    uint64_t offset = row_index - tile.start_row;
    if (offset < tile.data.size()) {
        return tile.data[offset];
    }
    
    // Row not in tile (shouldn't happen normally)
    return std::nan("");
}

double DataProvider::interpolate_at(const std::string& signal_name, double time) {
    auto result = interpolate_detailed(signal_name, time);
    return result.valid ? result.value : std::nan("");
}

std::vector<double> DataProvider::interpolate_batch(
    const std::string& signal_name,
    const std::vector<double>& times
) {
    std::vector<double> results;
    results.reserve(times.size());
    
    for (double t : times) {
        results.push_back(interpolate_at(signal_name, t));
    }
    
    return results;
}

InterpolationResult DataProvider::interpolate_detailed(
    const std::string& signal_name,
    double time
) {
    double sample_rate = get_sample_rate();
    uint64_t total = get_total_samples();
    
    if (total == 0 || sample_rate <= 0) {
        return InterpolationResult::invalid();
    }
    
    // Convert time to row index
    double exact_row = time * sample_rate;
    
    if (exact_row < 0 || exact_row >= total) {
        return InterpolationResult::invalid();
    }
    
    uint64_t left_row = static_cast<uint64_t>(std::floor(exact_row));
    uint64_t right_row = std::min(left_row + 1, total - 1);
    double weight = exact_row - left_row;
    
    // Get values
    double left_value = get_value_at(signal_name, left_row);
    double right_value = get_value_at(signal_name, right_row);
    
    if (std::isnan(left_value) || std::isnan(right_value)) {
        return InterpolationResult::invalid();
    }
    
    // Linear interpolation
    double value = left_value + weight * (right_value - left_value);
    
    return InterpolationResult(value, left_row, right_row, weight);
}

// ============================================================================
// Metadata
// ============================================================================

std::vector<std::string> DataProvider::get_signal_names() const {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    if (!metadata_loaded_) {
        signal_names_ = source_->get_signal_names();
        metadata_loaded_ = true;
    }
    return signal_names_;
}

SignalMetadata DataProvider::get_signal_metadata(const std::string& name) const {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    
    auto it = metadata_cache_.find(name);
    if (it != metadata_cache_.end()) {
        return it->second;
    }
    
    auto metadata = source_->get_signal_metadata(name);
    metadata_cache_[name] = metadata;
    return metadata;
}

uint64_t DataProvider::get_total_samples() const {
    return source_->get_total_samples();
}

double DataProvider::get_sample_rate() const {
    return source_->get_sample_rate();
}

bool DataProvider::has_signal(const std::string& name) const {
    return source_->has_signal(name);
}

size_t DataProvider::get_signal_count() const {
    return source_->get_signal_count();
}

// ============================================================================
// Time Conversion
// ============================================================================

uint64_t DataProvider::time_to_row(double time) const {
    double sample_rate = get_sample_rate();
    if (sample_rate <= 0) {
        return static_cast<uint64_t>(time);  // Assume 1 Hz
    }
    return static_cast<uint64_t>(time * sample_rate);
}

double DataProvider::row_to_time(uint64_t row) const {
    double sample_rate = get_sample_rate();
    if (sample_rate <= 0) {
        return static_cast<double>(row);  // Assume 1 Hz
    }
    return static_cast<double>(row) / sample_rate;
}

// ============================================================================
// Cache Management
// ============================================================================

void DataProvider::clear_cache() {
    data_cache_.clear();
    stats_cache_.clear();
}

void DataProvider::clear_data_cache() {
    data_cache_.clear();
}

void DataProvider::clear_stats_cache() {
    stats_cache_.clear();
}

void DataProvider::set_cache_size(size_t mb) {
    data_cache_.set_memory_limit(mb);
}

CacheStats DataProvider::get_cache_stats() const {
    return data_cache_.get_stats();
}

void DataProvider::prefetch(const std::string& signal_name, double center_time) {
    uint64_t center_row = time_to_row(center_time);
    
    data_cache_.prefetch(signal_name, center_row,
        [this, &signal_name](uint64_t start, uint64_t count) {
            return source_->read_range(signal_name, start, count);
        },
        2  // Prefetch 2 tiles before and after
    );
}

// ============================================================================
// Source Information
// ============================================================================

std::string DataProvider::get_source_type() const {
    return source_->get_source_type();
}

std::string DataProvider::get_source_path() const {
    return source_->get_source_path();
}

bool DataProvider::is_live() const {
    return source_->is_live();
}

// ============================================================================
// Internal Methods
// ============================================================================

DataChunk DataProvider::load_data_internal(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) {
    // For small requests, use tile-based loading
    if (row_count <= constants::DEFAULT_TILE_SIZE) {
        // Check if request spans multiple tiles
        uint64_t start_tile = DataCache::get_tile_start(start_row);
        uint64_t end_tile = DataCache::get_tile_start(start_row + row_count - 1);
        
        if (start_tile == end_tile) {
            // Single tile - use cache
            DataChunk tile = get_tile(signal_name, start_row);
            
            // Extract requested range from tile
            uint64_t offset = start_row - tile.start_row;
            if (offset + row_count <= tile.data.size()) {
                DataChunk result(signal_name);
                result.start_row = start_row;
                result.row_count = row_count;
                result.data.assign(
                    tile.data.begin() + offset,
                    tile.data.begin() + offset + row_count
                );
                if (!tile.time_data.empty()) {
                    result.time_data.assign(
                        tile.time_data.begin() + offset,
                        tile.time_data.begin() + offset + row_count
                    );
                }
                return result;
            }
        }
    }
    
    // Large request or multi-tile - load directly
    return source_->read_range(signal_name, start_row, row_count);
}

ColumnStatistics DataProvider::calculate_stats_internal(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) {
    // Check stats cache first
    auto cached = stats_cache_.get(signal_name, start_row, row_count);
    if (cached) {
        return *cached;
    }
    
    // Load data
    DataChunk chunk = load_data_internal(signal_name, start_row, row_count);
    
    if (chunk.empty()) {
        return ColumnStatistics();
    }
    
    // Calculate statistics manually (since StatisticsEngine methods are private)
    const double* data = chunk.data.data();
    size_t count = chunk.data.size();
    
    ColumnStatistics stats;
    stats.count = count;
    stats.valid_count = count;
    
    // Calculate mean
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    stats.mean = sum / count;
    stats.sum = sum;
    
    // Calculate min, max, std_dev, rms in single pass
    double min_val = data[0];
    double max_val = data[0];
    double sum_sq = 0.0;
    double sum_sq_diff = 0.0;
    
    for (size_t i = 0; i < count; ++i) {
        double val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_sq += val * val;
        double diff = val - stats.mean;
        sum_sq_diff += diff * diff;
    }
    
    stats.min = min_val;
    stats.max = max_val;
    stats.peak_to_peak = max_val - min_val;
    stats.std_dev = std::sqrt(sum_sq_diff / count);
    stats.rms = std::sqrt(sum_sq / count);
    
    // Median (approximate for large datasets)
    if (count <= 100000) {
        std::vector<double> sorted(data, data + count);
        std::nth_element(sorted.begin(), sorted.begin() + count/2, sorted.end());
        stats.median = sorted[count/2];
    } else {
        stats.median = stats.mean;  // Approximate
    }
    
    // Cache the result
    stats_cache_.put(signal_name, start_row, row_count, stats);
    
    return stats;
}

DataChunk DataProvider::get_tile(const std::string& signal_name, uint64_t row) {
    return data_cache_.get_tile(signal_name, row,
        [this, &signal_name](uint64_t start, uint64_t count) {
            return source_->read_range(signal_name, start, count);
        }
    );
}

} // namespace timegraph

