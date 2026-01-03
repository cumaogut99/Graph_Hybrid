#pragma once

/**
 * @file data_source.hpp
 * @brief Abstract data source interface for TimeGraph
 * 
 * Defines the IDataSource interface that all data sources must implement.
 * This abstraction allows the application to work with different data sources
 * (files, live streams, network) through a unified API.
 * 
 * Supported source types:
 * - File sources: MPAI, CSV, HDF5, TDMS (future)
 * - Live sources: CAN Bus, DAQ, IoT (future)
 * - Network sources: TCP/UDP, MQTT, OPC-UA (future)
 */

#include "timegraph/core/types.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>

namespace timegraph {

// ============================================================================
// Data Source Interface
// ============================================================================

/**
 * @brief Abstract interface for all data sources
 * 
 * This interface provides a unified API for accessing signal data
 * regardless of the underlying source (file, live stream, network).
 * 
 * All data sources must implement:
 * - Metadata access (signal names, sample rate, etc.)
 * - Range-based data reading
 * - Batch operations for multiple signals
 * 
 * Example usage:
 * @code
 *   auto source = std::make_shared<MpaiDataSource>("data.mpai");
 *   auto signals = source->get_signal_names();
 *   auto chunk = source->read_range("Signal1", 0, 10000);
 * @endcode
 */
class IDataSource {
public:
    virtual ~IDataSource() = default;
    
    // ========================================================================
    // Metadata Access
    // ========================================================================
    
    /**
     * @brief Get list of all signal names in the source
     * @return Vector of signal name strings
     */
    virtual std::vector<std::string> get_signal_names() const = 0;
    
    /**
     * @brief Get metadata for a specific signal
     * @param name Signal name
     * @return SignalMetadata structure
     * @throws std::runtime_error if signal not found
     */
    virtual SignalMetadata get_signal_metadata(const std::string& name) const = 0;
    
    /**
     * @brief Get total number of samples (rows) in the source
     * @return Total sample count
     */
    virtual uint64_t get_total_samples() const = 0;
    
    /**
     * @brief Get the sample rate (Hz)
     * @return Sample rate in Hz, or 0 if variable
     */
    virtual double get_sample_rate() const = 0;
    
    /**
     * @brief Check if a signal exists in the source
     * @param name Signal name to check
     * @return true if signal exists
     */
    virtual bool has_signal(const std::string& name) const {
        auto names = get_signal_names();
        return std::find(names.begin(), names.end(), name) != names.end();
    }
    
    /**
     * @brief Get number of signals in the source
     * @return Signal count
     */
    virtual size_t get_signal_count() const {
        return get_signal_names().size();
    }
    
    // ========================================================================
    // Data Access
    // ========================================================================
    
    /**
     * @brief Read a range of data from a signal
     * 
     * This is the primary method for accessing signal data. It reads
     * a contiguous range of samples starting from start_row.
     * 
     * @param signal_name Name of the signal to read
     * @param start_row Starting row index (0-based)
     * @param row_count Number of rows to read (0 = all remaining)
     * @return DataChunk containing the requested data
     * @throws std::runtime_error if signal not found or read fails
     */
    virtual DataChunk read_range(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    ) = 0;
    
    /**
     * @brief Read ranges from multiple signals in parallel
     * 
     * More efficient than calling read_range() multiple times,
     * as it can optimize disk I/O for file sources.
     * 
     * @param signal_names List of signal names to read
     * @param start_row Starting row index (same for all signals)
     * @param row_count Number of rows to read
     * @return Map of signal_name -> DataChunk
     */
    virtual std::map<std::string, DataChunk> read_range_batch(
        const std::vector<std::string>& signal_names,
        uint64_t start_row,
        uint64_t row_count
    ) {
        // Default implementation: sequential reads
        // Subclasses can override for parallel I/O
        std::map<std::string, DataChunk> result;
        for (const auto& name : signal_names) {
            result[name] = read_range(name, start_row, row_count);
        }
        return result;
    }
    
    /**
     * @brief Read time column data
     * 
     * Reads the time/index column for the given range.
     * Returns row indices if no explicit time column exists.
     * 
     * @param start_row Starting row index
     * @param row_count Number of rows
     * @return Vector of time values
     */
    virtual std::vector<double> read_time_range(
        uint64_t start_row,
        uint64_t row_count
    ) {
        // Default: generate row indices as time
        std::vector<double> time(row_count);
        double sample_rate = get_sample_rate();
        double dt = sample_rate > 0 ? 1.0 / sample_rate : 1.0;
        for (uint64_t i = 0; i < row_count; ++i) {
            time[i] = (start_row + i) * dt;
        }
        return time;
    }
    
    // ========================================================================
    // Source Information
    // ========================================================================
    
    /**
     * @brief Get the source type identifier
     * @return "file", "live", or "network"
     */
    virtual std::string get_source_type() const = 0;
    
    /**
     * @brief Get the source path or connection string
     * @return File path, URL, or connection identifier
     */
    virtual std::string get_source_path() const = 0;
    
    /**
     * @brief Check if this is a live (streaming) source
     * @return true if source provides real-time data
     */
    virtual bool is_live() const { return false; }
    
    /**
     * @brief Check if source is currently open and valid
     * @return true if source can be read from
     */
    virtual bool is_open() const = 0;
    
    /**
     * @brief Close the data source and release resources
     */
    virtual void close() = 0;
};

// ============================================================================
// File Data Source Base Class
// ============================================================================

/**
 * @brief Base class for file-based data sources
 * 
 * Provides common functionality for file sources like MPAI, CSV, HDF5.
 */
class FileDataSource : public IDataSource {
public:
    std::string get_source_type() const override { return "file"; }
    bool is_live() const override { return false; }
    
protected:
    std::string filepath_;
    bool is_open_ = false;
};

// ============================================================================
// Live Data Source Interface
// ============================================================================

/**
 * @brief Interface for live (streaming) data sources
 * 
 * Extends IDataSource with methods for real-time data streaming,
 * subscription-based updates, and buffer management.
 * 
 * Used for: CAN Bus, DAQ hardware, IoT sensors, etc.
 */
class ILiveDataSource : public IDataSource {
public:
    std::string get_source_type() const override { return "live"; }
    bool is_live() const override { return true; }
    
    // ========================================================================
    // Live Source Control
    // ========================================================================
    
    /**
     * @brief Start data acquisition
     * @return true if started successfully
     */
    virtual bool start() = 0;
    
    /**
     * @brief Stop data acquisition
     */
    virtual void stop() = 0;
    
    /**
     * @brief Check if acquisition is running
     * @return true if currently acquiring data
     */
    virtual bool is_running() const = 0;
    
    // ========================================================================
    // Subscription Model
    // ========================================================================
    
    /**
     * @brief Subscribe to data updates for a signal
     * 
     * The callback will be called whenever new data is available.
     * 
     * @param signal_name Signal to subscribe to
     * @param callback Function to call with new data
     * @return Subscription ID (for unsubscribe)
     */
    virtual uint64_t subscribe(
        const std::string& signal_name,
        DataCallback callback
    ) = 0;
    
    /**
     * @brief Unsubscribe from data updates
     * @param subscription_id ID returned from subscribe()
     */
    virtual void unsubscribe(uint64_t subscription_id) = 0;
    
    /**
     * @brief Unsubscribe all callbacks for a signal
     * @param signal_name Signal name
     */
    virtual void unsubscribe_all(const std::string& signal_name) = 0;
    
    // ========================================================================
    // Buffer Management
    // ========================================================================
    
    /**
     * @brief Set the internal buffer size
     * @param samples Number of samples to buffer
     */
    virtual void set_buffer_size(size_t samples) = 0;
    
    /**
     * @brief Get current buffer size
     * @return Buffer size in samples
     */
    virtual size_t get_buffer_size() const = 0;
    
    /**
     * @brief Get current number of samples in buffer
     * @return Buffered sample count
     */
    virtual size_t get_buffered_samples() const = 0;
    
    /**
     * @brief Clear all buffered data
     */
    virtual void clear_buffer() = 0;
};

// ============================================================================
// Network Data Source Interface
// ============================================================================

/**
 * @brief Interface for network-based data sources
 * 
 * Extends ILiveDataSource with network-specific functionality.
 * Used for: TCP/UDP streams, MQTT, OPC-UA, etc.
 */
class INetworkDataSource : public ILiveDataSource {
public:
    std::string get_source_type() const override { return "network"; }
    
    /**
     * @brief Connect to the network source
     * @param host Hostname or IP address
     * @param port Port number
     * @return true if connected successfully
     */
    virtual bool connect(const std::string& host, uint16_t port) = 0;
    
    /**
     * @brief Disconnect from the network source
     */
    virtual void disconnect() = 0;
    
    /**
     * @brief Check if connected
     * @return true if connected
     */
    virtual bool is_connected() const = 0;
    
    /**
     * @brief Get connection latency
     * @return Latency in milliseconds
     */
    virtual double get_latency_ms() const = 0;
};

// ============================================================================
// Factory Function Type
// ============================================================================

/// Factory function type for creating data sources
using DataSourceFactory = std::function<std::shared_ptr<IDataSource>(const std::string&)>;

} // namespace timegraph

