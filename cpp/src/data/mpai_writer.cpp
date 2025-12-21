/**
 * MPAI Writer Implementation
 */

#include "timegraph/data/mpai_writer.hpp"
#include "timegraph/data/metadata_builder.hpp"
#include <cstring>
#include <ctime>
#include <stdexcept>

// ZSTD compression (conditional)
#ifdef HAVE_ZSTD
#include <zstd.h>
#else
// Fallback: no compression
#pragma message("Warning: ZSTD not available, compression disabled")
#endif

namespace timegraph {
namespace mpai {

MpaiWriter::MpaiWriter(const std::string &filename, int compression_level)
    : filename_(filename), compression_level_(compression_level),
      current_offset_(0), total_uncompressed_(0), total_compressed_(0) {
  // Open file for binary writing
  file_.open(filename_, std::ios::binary | std::ios::out);
  if (!file_.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename_);
  }

  // Reserve space for header (will be updated later)
  current_offset_ = HEADER_SIZE;
  file_.seekp(HEADER_SIZE);
}

MpaiWriter::~MpaiWriter() {
  if (file_.is_open()) {
    finalize();
  }
}

void MpaiWriter::set_progress_callback(ProgressCallback callback) {
  progress_callback_ = callback;
}

void MpaiWriter::write_header(uint64_t row_count, uint32_t column_count,
                              const std::string &source_file) {
  header_.row_count = row_count;
  header_.column_count = column_count;
  header_.creation_time = static_cast<uint64_t>(std::time(nullptr));

  // Copy source filename (truncate if too long)
  std::strncpy(header_.source_file, source_file.c_str(),
               sizeof(header_.source_file) - 1);
  header_.source_file[sizeof(header_.source_file) - 1] = '\0';

  // Set creator
  std::strncpy(header_.creator, "TimeGraph v2.0", sizeof(header_.creator) - 1);

  emit_progress("Header initialized", 0);
}

void MpaiWriter::add_column_metadata(const ColumnMetadata &metadata) {
  data_metadata_.columns.push_back(metadata);
}

void MpaiWriter::write_column_chunk(uint32_t column_index, uint32_t chunk_id,
                                    const void *data, size_t data_size,
                                    uint32_t row_count) {
  if (column_index >= data_metadata_.columns.size()) {
    throw std::out_of_range("Column index out of range");
  }

  // Compress data
  std::vector<uint8_t> compressed = compress_data(data, data_size);

  // Create chunk info
  ChunkInfo chunk_info;
  chunk_info.offset = current_offset_;
  chunk_info.uncompressed_size = static_cast<uint32_t>(data_size);
  chunk_info.compressed_size = static_cast<uint32_t>(compressed.size());
  chunk_info.row_count = row_count;
  chunk_info.crc32 = calculate_crc32(data, data_size);

  // ✅ Calculate pre-aggregated statistics using MetadataBuilder
  // (SIMD-optimized)
  if (data_size >= sizeof(double)) {
    const double *values = static_cast<const double *>(data);
    size_t count = data_size / sizeof(double);

    // Calculate start_row for this chunk
    uint64_t start_row = 0;
    for (const auto &existing_chunk :
         data_metadata_.columns[column_index].chunks) {
      start_row += existing_chunk.row_count;
    }

    // Use MetadataBuilder to calculate all statistics (sum, sum_squares, min,
    // max, count)
    auto chunk_metadata =
        MetadataBuilder::build_chunk_metadata(values, count, start_row);

    // Populate ChunkInfo with pre-aggregated statistics
    chunk_info.min_value = chunk_metadata.min_value;
    chunk_info.max_value = chunk_metadata.max_value;
    chunk_info.sum = chunk_metadata.sum;
    chunk_info.sum_squares = chunk_metadata.sum_squares;
    chunk_info.valid_count = chunk_metadata.count;
  }

  // Add to column metadata
  data_metadata_.columns[column_index].chunks.push_back(chunk_info);

  // Write compressed data
  file_.write(reinterpret_cast<const char *>(compressed.data()),
              compressed.size());

  // Update offset and stats
  current_offset_ += compressed.size();
  total_uncompressed_ += data_size;
  total_compressed_ += compressed.size();

  // Progress
  int progress =
      static_cast<int>((column_index * 100) / data_metadata_.columns.size());
  emit_progress("Writing column " + std::to_string(column_index), progress);
}

void MpaiWriter::write_application_state(const ApplicationState &state) {
  app_state_ = state;
  app_state_.last_modified = static_cast<uint64_t>(std::time(nullptr));
}

void MpaiWriter::finalize() {
  if (!file_.is_open()) {
    return;
  }

  emit_progress("Finalizing file", 90);

  // Save current position (start of metadata section)
  header_.data_metadata_offset = current_offset_;

  // Write data metadata
  write_data_metadata();

  // Write application state
  header_.app_state_offset = current_offset_;
  write_app_state();

  // Update header with final info
  header_.file_size = current_offset_;
  header_.uncompressed_size = static_cast<uint32_t>(total_uncompressed_);
  header_.compressed_size = static_cast<uint32_t>(total_compressed_);

  if (total_uncompressed_ > 0) {
    header_.compression_ratio =
        static_cast<uint32_t>((total_uncompressed_ * 100) / total_compressed_);
  }

  // Calculate checksums
  update_header();

  // Write header at beginning of file
  file_.seekp(0);
  file_.write(reinterpret_cast<const char *>(&header_), sizeof(header_));

  file_.close();

  emit_progress("File saved successfully", 100);
}

uint64_t MpaiWriter::get_file_size() const { return current_offset_; }

double MpaiWriter::get_compression_ratio() const {
  if (total_compressed_ == 0)
    return 0.0;
  return static_cast<double>(total_uncompressed_) /
         static_cast<double>(total_compressed_);
}

// Private helper methods

void MpaiWriter::write_data_metadata() {
  // Write column count
  uint32_t column_count = static_cast<uint32_t>(data_metadata_.columns.size());
  file_.write(reinterpret_cast<const char *>(&column_count),
              sizeof(column_count));
  current_offset_ += sizeof(column_count);

  // Write each column metadata
  for (const auto &col : data_metadata_.columns) {
    // Write column name (length + string)
    uint32_t name_len = static_cast<uint32_t>(col.name.size());
    file_.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    file_.write(col.name.c_str(), name_len);
    current_offset_ += sizeof(name_len) + name_len;

    // Write data type
    file_.write(reinterpret_cast<const char *>(&col.data_type),
                sizeof(col.data_type));
    current_offset_ += sizeof(col.data_type);

    // Write statistics
    file_.write(reinterpret_cast<const char *>(&col.statistics),
                sizeof(col.statistics));
    current_offset_ += sizeof(col.statistics);

    // Write chunk count
    uint32_t chunk_count = static_cast<uint32_t>(col.chunks.size());
    file_.write(reinterpret_cast<const char *>(&chunk_count),
                sizeof(chunk_count));
    current_offset_ += sizeof(chunk_count);

    // Write chunk info array
    for (const auto &chunk : col.chunks) {
      file_.write(reinterpret_cast<const char *>(&chunk), sizeof(chunk));
      current_offset_ += sizeof(chunk);
    }
  }
}

void MpaiWriter::write_app_state() {
  // Write state version
  file_.write(reinterpret_cast<const char *>(&app_state_.state_version),
              sizeof(app_state_.state_version));
  current_offset_ += sizeof(app_state_.state_version);

  // Write graph count
  uint32_t graph_count = static_cast<uint32_t>(app_state_.graphs.size());
  file_.write(reinterpret_cast<const char *>(&graph_count),
              sizeof(graph_count));
  current_offset_ += sizeof(graph_count);

  // Write each graph config
  for (const auto &graph : app_state_.graphs) {
    // Graph ID
    file_.write(reinterpret_cast<const char *>(&graph.graph_id),
                sizeof(graph.graph_id));
    current_offset_ += sizeof(graph.graph_id);

    // Title
    uint32_t title_len = static_cast<uint32_t>(graph.title.size());
    file_.write(reinterpret_cast<const char *>(&title_len), sizeof(title_len));
    file_.write(graph.title.c_str(), title_len);
    current_offset_ += sizeof(title_len) + title_len;

    // Signal count
    uint32_t signal_count = static_cast<uint32_t>(graph.signals.size());
    file_.write(reinterpret_cast<const char *>(&signal_count),
                sizeof(signal_count));
    current_offset_ += sizeof(signal_count);

    // Write each signal
    for (const auto &signal : graph.signals) {
      // Signal name
      uint32_t name_len = static_cast<uint32_t>(signal.signal_name.size());
      file_.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
      file_.write(signal.signal_name.c_str(), name_len);
      current_offset_ += sizeof(name_len) + name_len;

      // Color
      file_.write(reinterpret_cast<const char *>(signal.color),
                  sizeof(signal.color));
      current_offset_ += sizeof(signal.color);

      // Other properties
      file_.write(reinterpret_cast<const char *>(&signal.line_width),
                  sizeof(signal.line_width));
      file_.write(reinterpret_cast<const char *>(&signal.line_style),
                  sizeof(signal.line_style));
      file_.write(reinterpret_cast<const char *>(&signal.visible),
                  sizeof(signal.visible));
      file_.write(reinterpret_cast<const char *>(&signal.normalized),
                  sizeof(signal.normalized));
      file_.write(reinterpret_cast<const char *>(&signal.norm_method),
                  sizeof(signal.norm_method));
      file_.write(reinterpret_cast<const char *>(&signal.y_axis),
                  sizeof(signal.y_axis));
      file_.write(reinterpret_cast<const char *>(&signal.y_min),
                  sizeof(signal.y_min));
      file_.write(reinterpret_cast<const char *>(&signal.y_max),
                  sizeof(signal.y_max));
      file_.write(reinterpret_cast<const char *>(&signal.auto_scale),
                  sizeof(signal.auto_scale));

      current_offset_ += sizeof(signal.line_width) + sizeof(signal.line_style) +
                         sizeof(signal.visible) + sizeof(signal.normalized) +
                         sizeof(signal.norm_method) + sizeof(signal.y_axis) +
                         sizeof(signal.y_min) + sizeof(signal.y_max) +
                         sizeof(signal.auto_scale);
    }

    // Axis settings
    file_.write(reinterpret_cast<const char *>(&graph.x_min),
                sizeof(graph.x_min));
    file_.write(reinterpret_cast<const char *>(&graph.x_max),
                sizeof(graph.x_max));
    file_.write(reinterpret_cast<const char *>(&graph.auto_x_scale),
                sizeof(graph.auto_x_scale));
    current_offset_ +=
        sizeof(graph.x_min) + sizeof(graph.x_max) + sizeof(graph.auto_x_scale);

    // Labels (simplified - just write length + string)
    auto write_string = [this](const std::string &str) {
      uint32_t len = static_cast<uint32_t>(str.size());
      file_.write(reinterpret_cast<const char *>(&len), sizeof(len));
      file_.write(str.c_str(), len);
      current_offset_ += sizeof(len) + len;
    };

    write_string(graph.x_label);
    write_string(graph.y_left_label);
    write_string(graph.y_right_label);

    // Grid & style
    file_.write(reinterpret_cast<const char *>(&graph.show_grid),
                sizeof(graph.show_grid));
    file_.write(reinterpret_cast<const char *>(&graph.show_legend),
                sizeof(graph.show_legend));
    file_.write(reinterpret_cast<const char *>(&graph.legend_position),
                sizeof(graph.legend_position));
    file_.write(reinterpret_cast<const char *>(graph.background_color),
                sizeof(graph.background_color));
    current_offset_ += sizeof(graph.show_grid) + sizeof(graph.show_legend) +
                       sizeof(graph.legend_position) +
                       sizeof(graph.background_color);
  }

  // Write filters, cursors, annotations (similar pattern)
  // TODO: Implement if needed

  // Write layout
  file_.write(reinterpret_cast<const char *>(&app_state_.layout.subplot_count),
              sizeof(app_state_.layout.subplot_count));
  file_.write(reinterpret_cast<const char *>(&app_state_.layout.subplot_layout),
              sizeof(app_state_.layout.subplot_layout));
  current_offset_ += sizeof(app_state_.layout.subplot_count) +
                     sizeof(app_state_.layout.subplot_layout);
}

void MpaiWriter::update_header() {
  // Calculate header CRC32
  header_.header_crc32 = 0; // Exclude checksum field itself
  header_.header_crc32 =
      calculate_crc32(&header_, sizeof(header_) - sizeof(header_.reserved));
}

uint32_t MpaiWriter::calculate_crc32(const void *data, size_t size) {
  // Simple CRC32 implementation
  // TODO: Use proper CRC32 library
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t *bytes = static_cast<const uint8_t *>(data);

  for (size_t i = 0; i < size; ++i) {
    crc ^= bytes[i];
    for (int j = 0; j < 8; ++j) {
      crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }

  return ~crc;
}

std::vector<uint8_t> MpaiWriter::compress_data(const void *data, size_t size) {
#ifdef HAVE_ZSTD
  // Get maximum compressed size
  size_t max_compressed_size = ZSTD_compressBound(size);
  std::vector<uint8_t> compressed(max_compressed_size);

  // Compress
  size_t compressed_size = ZSTD_compress(compressed.data(), max_compressed_size,
                                         data, size, compression_level_);

  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error("ZSTD compression failed: " +
                             std::string(ZSTD_getErrorName(compressed_size)));
  }

  // Resize to actual compressed size
  compressed.resize(compressed_size);
  return compressed;
#else
  // No compression available - just copy data
  std::vector<uint8_t> result(size);
  std::memcpy(result.data(), data, size);
  return result;
#endif
}

void MpaiWriter::emit_progress(const std::string &message, int percentage) {
  if (progress_callback_) {
    progress_callback_(message, percentage);
  }
}

} // namespace mpai
} // namespace timegraph
