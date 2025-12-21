#pragma once

/**
 * @file mpai_source.hpp
 * @brief MPAI file data source implementation
 * 
 * Wraps the existing MpaiReader to implement the IDataSource interface.
 * Provides unified access to MPAI files through the DataProvider API.
 */

#include "timegraph/core/data_source.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include <memory>
#include <string>

namespace timegraph {

/**
 * @brief Data source implementation for MPAI files
 * 
 * This class wraps the MpaiReader to provide IDataSource interface.
 * It handles all MPAI-specific operations and exposes them through
 * the unified data source API.
 * 
 * Features:
 * - Streaming data access (low memory usage)
 * - Chunk-based reading for large files
 * - Metadata extraction from MPAI headers
 * - Sample rate detection
 * 
 * Example:
 * @code
 *   auto source = std::make_shared<MpaiDataSource>("data.mpai");
 *   auto provider = DataProvider(source);
 *   auto stats = provider.get_statistics("Signal1", 0, 100000);
 * @endcode
 */
class MpaiDataSource : public FileDataSource {
public:
    /**
     * @brief Construct MPAI data source from file
     * @param filepath Path to MPAI file
     * @throws std::runtime_error if file cannot be opened
     */
    explicit MpaiDataSource(const std::string& filepath);
    
    /// Destructor
    ~MpaiDataSource() override = default;
    
    // ========================================================================
    // IDataSource Implementation
    // ========================================================================
    
    /**
     * @brief Get list of signal names in the MPAI file
     * @return Vector of signal/column names
     */
    std::vector<std::string> get_signal_names() const override;
    
    /**
     * @brief Get metadata for a signal
     * @param name Signal name
     * @return SignalMetadata with sample rate, total samples, etc.
     */
    SignalMetadata get_signal_metadata(const std::string& name) const override;
    
    /**
     * @brief Get total number of samples (rows) in file
     * @return Total row count
     */
    uint64_t get_total_samples() const override;
    
    /**
     * @brief Get sample rate
     * 
     * Attempts to detect sample rate from time column if present,
     * otherwise returns 1.0 Hz.
     * 
     * @return Sample rate in Hz
     */
    double get_sample_rate() const override;
    
    /**
     * @brief Read data range from signal
     * 
     * Uses MpaiReader's load_column_slice for efficient streaming.
     * 
     * @param signal_name Signal/column name
     * @param start_row Starting row index
     * @param row_count Number of rows to read
     * @return DataChunk with requested data
     */
    DataChunk read_range(
        const std::string& signal_name,
        uint64_t start_row,
        uint64_t row_count
    ) override;
    
    /**
     * @brief Read ranges from multiple signals
     * 
     * Optimized batch reading - can parallelize I/O.
     * 
     * @param signal_names List of signals to read
     * @param start_row Starting row
     * @param row_count Number of rows
     * @return Map of signal_name -> DataChunk
     */
    std::map<std::string, DataChunk> read_range_batch(
        const std::vector<std::string>& signal_names,
        uint64_t start_row,
        uint64_t row_count
    ) override;
    
    /**
     * @brief Read time column data
     * 
     * If MPAI file has a time column, reads it.
     * Otherwise generates time from sample rate.
     * 
     * @param start_row Starting row
     * @param row_count Number of rows
     * @return Vector of time values
     */
    std::vector<double> read_time_range(
        uint64_t start_row,
        uint64_t row_count
    ) override;
    
    /**
     * @brief Get source file path
     * @return MPAI file path
     */
    std::string get_source_path() const override;
    
    /**
     * @brief Check if file is open
     * @return true if file is open and readable
     */
    bool is_open() const override;
    
    /**
     * @brief Close the MPAI file
     */
    void close() override;
    
    // ========================================================================
    // MPAI-Specific Methods
    // ========================================================================
    
    /**
     * @brief Get the underlying MpaiReader
     * @return Reference to MpaiReader
     */
    mpai::MpaiReader& get_reader() { return *reader_; }
    const mpai::MpaiReader& get_reader() const { return *reader_; }
    
    /**
     * @brief Get MPAI file header information
     * @return Header structure
     */
    const mpai::MpaiHeader& get_header() const;
    
    /**
     * @brief Get compression ratio
     * @return Compression ratio (original/compressed)
     */
    double get_compression_ratio() const;
    
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes
     */
    size_t get_memory_usage() const;
    
    /**
     * @brief Check if file has a time column
     * @return true if time column exists
     */
    bool has_time_column() const;
    
    /**
     * @brief Get time column name
     * @return Time column name or empty string
     */
    std::string get_time_column_name() const;
    
private:
    /**
     * @brief Detect time column from signal names
     * 
     * Looks for common time column names:
     * - "Time", "time", "TIME"
     * - "t", "T"
     * - "timestamp", "Timestamp"
     */
    void detect_time_column();
    
    /**
     * @brief Calculate sample rate from time column
     */
    void calculate_sample_rate();
    
    std::unique_ptr<mpai::MpaiReader> reader_;
    std::string filepath_;
    
    // Cached values
    mutable std::vector<std::string> signal_names_;
    mutable bool names_cached_ = false;
    
    std::string time_column_name_;
    double sample_rate_ = 1.0;
    bool sample_rate_calculated_ = false;
};

} // namespace timegraph

