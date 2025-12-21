/**
 * @file mpai_source.cpp
 * @brief Implementation of MPAI data source
 */

#include "timegraph/sources/mpai_source.hpp"
#include <algorithm>
#include <cctype>
#include <future>

namespace timegraph {

// ============================================================================
// Construction
// ============================================================================

MpaiDataSource::MpaiDataSource(const std::string& filepath)
    : filepath_(filepath)
{
    reader_ = std::make_unique<mpai::MpaiReader>(filepath);
    filepath_ = filepath;
    is_open_ = true;
    
    // Detect time column and calculate sample rate
    detect_time_column();
    calculate_sample_rate();
}

// ============================================================================
// IDataSource Implementation
// ============================================================================

std::vector<std::string> MpaiDataSource::get_signal_names() const {
    if (!names_cached_) {
        signal_names_ = reader_->get_column_names();
        names_cached_ = true;
    }
    return signal_names_;
}

SignalMetadata MpaiDataSource::get_signal_metadata(const std::string& name) const {
    SignalMetadata meta(name);
    
    meta.total_samples = reader_->get_row_count();
    meta.sample_rate = sample_rate_;
    meta.start_time = 0.0;
    meta.end_time = meta.total_samples / meta.sample_rate;
    
    // Get min/max from first chunk (approximate)
    // Full min/max would require reading entire column
    try {
        auto chunk = const_cast<mpai::MpaiReader*>(reader_.get())->load_column_slice(name, 0, 10000);
        if (!chunk.empty()) {
            auto minmax = std::minmax_element(chunk.begin(), chunk.end());
            meta.min_value = *minmax.first;
            meta.max_value = *minmax.second;
        }
    } catch (...) {
        meta.min_value = 0.0;
        meta.max_value = 0.0;
    }
    
    return meta;
}

uint64_t MpaiDataSource::get_total_samples() const {
    return reader_->get_row_count();
}

double MpaiDataSource::get_sample_rate() const {
    return sample_rate_;
}

DataChunk MpaiDataSource::read_range(
    const std::string& signal_name,
    uint64_t start_row,
    uint64_t row_count
) {
    DataChunk chunk(signal_name);
    chunk.start_row = start_row;
    
    // Clamp row_count to available data
    uint64_t total = get_total_samples();
    if (start_row >= total) {
        chunk.row_count = 0;
        return chunk;
    }
    
    uint64_t actual_count = std::min(row_count, total - start_row);
    chunk.row_count = actual_count;
    
    // Load data from MPAI reader
    chunk.data = reader_->load_column_slice(signal_name, start_row, actual_count);
    
    // Load time data if available
    if (!time_column_name_.empty() && signal_name != time_column_name_) {
        chunk.time_data = reader_->load_column_slice(time_column_name_, start_row, actual_count);
    }
    
    return chunk;
}

std::map<std::string, DataChunk> MpaiDataSource::read_range_batch(
    const std::vector<std::string>& signal_names,
    uint64_t start_row,
    uint64_t row_count
) {
    std::map<std::string, DataChunk> result;
    
    if (signal_names.size() <= 2) {
        // Sequential for small batches
        for (const auto& name : signal_names) {
            result[name] = read_range(name, start_row, row_count);
        }
    } else {
        // Parallel for larger batches
        std::vector<std::future<std::pair<std::string, DataChunk>>> futures;
        
        for (const auto& name : signal_names) {
            futures.push_back(std::async(std::launch::async,
                [this, &name, start_row, row_count]() {
                    return std::make_pair(name, read_range(name, start_row, row_count));
                }
            ));
        }
        
        for (auto& f : futures) {
            auto pair = f.get();
            result[pair.first] = std::move(pair.second);
        }
    }
    
    return result;
}

std::vector<double> MpaiDataSource::read_time_range(
    uint64_t start_row,
    uint64_t row_count
) {
    if (!time_column_name_.empty()) {
        // Read from time column
        return reader_->load_column_slice(time_column_name_, start_row, row_count);
    }
    
    // Generate time from sample rate
    std::vector<double> time(row_count);
    double dt = 1.0 / sample_rate_;
    for (uint64_t i = 0; i < row_count; ++i) {
        time[i] = (start_row + i) * dt;
    }
    return time;
}

std::string MpaiDataSource::get_source_path() const {
    return filepath_;
}

bool MpaiDataSource::is_open() const {
    return is_open_ && reader_ != nullptr;
}

void MpaiDataSource::close() {
    reader_.reset();
    is_open_ = false;
}

// ============================================================================
// MPAI-Specific Methods
// ============================================================================

const mpai::MpaiHeader& MpaiDataSource::get_header() const {
    return reader_->get_header();
}

double MpaiDataSource::get_compression_ratio() const {
    return reader_->get_compression_ratio();
}

size_t MpaiDataSource::get_memory_usage() const {
    return reader_->get_memory_usage();
}

bool MpaiDataSource::has_time_column() const {
    return !time_column_name_.empty();
}

std::string MpaiDataSource::get_time_column_name() const {
    return time_column_name_;
}

// ============================================================================
// Private Methods
// ============================================================================

void MpaiDataSource::detect_time_column() {
    auto names = get_signal_names();
    
    // Common time column names (case-insensitive check)
    std::vector<std::string> time_names = {
        "time", "Time", "TIME",
        "t", "T",
        "timestamp", "Timestamp", "TIMESTAMP",
        "Zeit", "zeit",  // German
        "temps", "Temps"  // French
    };
    
    for (const auto& name : names) {
        // Convert to lowercase for comparison
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
            [](unsigned char c) { return std::tolower(c); });
        
        for (const auto& time_name : time_names) {
            std::string lower_time = time_name;
            std::transform(lower_time.begin(), lower_time.end(), lower_time.begin(),
                [](unsigned char c) { return std::tolower(c); });
            
            if (lower_name == lower_time) {
                time_column_name_ = name;
                return;
            }
        }
    }
    
    // No time column found
    time_column_name_.clear();
}

void MpaiDataSource::calculate_sample_rate() {
    if (sample_rate_calculated_) {
        return;
    }
    
    sample_rate_calculated_ = true;
    
    if (!time_column_name_.empty()) {
        // Calculate from time column
        try {
            // Read first 1000 samples to estimate sample rate
            auto time_data = reader_->load_column_slice(time_column_name_, 0, 1000);
            
            if (time_data.size() >= 2) {
                // Calculate average time step
                double total_dt = 0.0;
                size_t count = 0;
                
                for (size_t i = 1; i < time_data.size(); ++i) {
                    double dt = time_data[i] - time_data[i-1];
                    if (dt > 0) {
                        total_dt += dt;
                        ++count;
                    }
                }
                
                if (count > 0) {
                    double avg_dt = total_dt / count;
                    sample_rate_ = 1.0 / avg_dt;
                    return;
                }
            }
        } catch (...) {
            // Fall through to default
        }
    }
    
    // Default sample rate
    sample_rate_ = 1.0;
}

} // namespace timegraph

