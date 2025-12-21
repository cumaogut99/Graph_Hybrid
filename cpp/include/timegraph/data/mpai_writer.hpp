/**
 * MPAI Writer - Write data and application state to MPAI format
 * 
 * Features:
 * - Streaming write (low memory)
 * - ZSTD compression
 * - Pre-compute statistics
 * - Save application state
 */

#pragma once

#include "mpai_format.hpp"
#include <fstream>
#include <functional>

namespace timegraph {
namespace mpai {

class MpaiWriter {
public:
    using ProgressCallback = std::function<void(const std::string&, int)>;
    
    /**
     * Create MPAI writer
     * 
     * @param filename Output file path
     * @param compression_level ZSTD compression level (1-22, default: 3)
     */
    explicit MpaiWriter(const std::string& filename, int compression_level = 3);
    
    ~MpaiWriter();
    
    // Disable copy
    MpaiWriter(const MpaiWriter&) = delete;
    MpaiWriter& operator=(const MpaiWriter&) = delete;
    
    /**
     * Set progress callback
     */
    void set_progress_callback(ProgressCallback callback);
    
    /**
     * Write header
     * 
     * @param row_count Total number of rows
     * @param column_count Total number of columns
     * @param source_file Original CSV filename
     */
    void write_header(uint64_t row_count, uint32_t column_count,
                     const std::string& source_file = "");
    
    /**
     * Add column metadata
     * 
     * @param metadata Column metadata with pre-computed statistics
     */
    void add_column_metadata(const ColumnMetadata& metadata);
    
    /**
     * Write column chunk
     * 
     * @param column_index Column index
     * @param chunk_id Chunk ID
     * @param data Raw data (will be compressed)
     * @param row_count Number of rows in chunk
     */
    void write_column_chunk(uint32_t column_index, uint32_t chunk_id,
                           const void* data, size_t data_size, uint32_t row_count);
    
    /**
     * Write application state
     * 
     * @param state Application state (graphs, filters, cursors, etc.)
     */
    void write_application_state(const ApplicationState& state);
    
    /**
     * Finalize and close file
     * 
     * Writes checksums and closes file handle.
     */
    void finalize();
    
    /**
     * Get current file size
     */
    uint64_t get_file_size() const;
    
    /**
     * Get compression ratio
     */
    double get_compression_ratio() const;

private:
    std::string filename_;
    std::ofstream file_;
    int compression_level_;
    
    MpaiHeader header_;
    DataMetadata data_metadata_;
    ApplicationState app_state_;
    
    uint64_t current_offset_;
    uint64_t total_uncompressed_;
    uint64_t total_compressed_;
    
    ProgressCallback progress_callback_;
    
    // Helper methods
    void write_data_metadata();
    void write_app_state();
    void update_header();
    uint32_t calculate_crc32(const void* data, size_t size);
    std::vector<uint8_t> compress_data(const void* data, size_t size);
    void emit_progress(const std::string& message, int percentage);
};

} // namespace mpai
} // namespace timegraph

