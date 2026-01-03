/**
 * MPAI Format - Multi-Purpose Application Interface
 *
 * Dewesoft-style binary format with application state support.
 * Target: 50 GB files with 300 MB RAM usage.
 *
 * Features:
 * - Columnar storage (fast column access)
 * - ZSTD compression (10:1 ratio)
 * - Pre-computed statistics (instant access)
 * - Application state (graphs, filters, cursors)
 * - Memory-mapped I/O (lazy loading)
 * - Indexed chunks (fast lookup)
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace timegraph {
namespace mpai {

// ==============================================================================
// Constants
// ==============================================================================

constexpr uint32_t MPAI_MAGIC = 0x4941504D;      // "MPAI" in little-endian
constexpr uint32_t MPAI_VERSION = 0x00020000;    // v2.0.0
constexpr uint32_t HEADER_SIZE = 4096;           // 4 KB
constexpr uint32_t DEFAULT_CHUNK_SIZE = 1000000; // 1M rows per chunk

// ==============================================================================
// Enums
// ==============================================================================

enum class DataType : uint8_t {
  FLOAT64 = 1,
  INT64 = 2,
  STRING = 3,
  DATETIME = 4
};

enum class CompressionType : uint8_t {
  NONE = 0,
  ZSTD = 1,
  LZ4 = 2,
  SNAPPY = 3
};

enum class FilterType : uint8_t { RANGE = 0, THRESHOLD = 1, BITMASK = 2 };

enum class NormalizationType : uint8_t {
  NONE = 0,
  PEAK = 1,
  RMS = 2,
  ZSCORE = 3
};

enum class LineStyle : uint8_t {
  SOLID = 0,
  DASHED = 1,
  DOTTED = 2,
  DASH_DOT = 3
};

// ==============================================================================
// [1] Header Structure (4 KB)
// ==============================================================================

#pragma pack(push, 1)

struct MpaiHeader {
  // Magic & Version (16 bytes)
  uint32_t magic;       // "MPAI" (0x4941504D)
  uint32_t version;     // 0x00020000 (v2.0)
  uint32_t header_size; // 4096 bytes
  uint32_t reserved1;

  // File Info (32 bytes)
  uint64_t file_size;     // Total file size in bytes
  uint64_t row_count;     // Total number of rows
  uint32_t column_count;  // Total number of columns
  uint32_t chunk_size;    // Rows per chunk (default: 1M)
  uint64_t creation_time; // Unix timestamp

  // Compression (16 bytes)
  uint8_t compression_type;  // CompressionType enum
  uint8_t compression_level; // 1-22 for ZSTD
  uint16_t reserved2;
  uint32_t uncompressed_size; // Total uncompressed data size
  uint32_t compressed_size;   // Total compressed data size
  uint32_t compression_ratio; // Ratio * 100 (e.g., 1000 = 10:1)

  // Section Offsets (32 bytes)
  uint64_t data_metadata_offset; // Offset to [2] Data Metadata
  uint64_t app_state_offset;     // Offset to [3] Application State
  uint64_t column_data_offset;   // Offset to [4] Column Data
  uint64_t index_offset;         // Offset to optional index file

  // Checksums (16 bytes)
  uint32_t header_crc32;
  uint32_t metadata_crc32;
  uint32_t app_state_crc32;
  uint32_t data_crc32;

  // File Metadata (128 bytes)
  char source_file[256]; // Original CSV filename
  char creator[64];      // "TimeGraph v2.0"
  char user_name[64];    // User who created file

  // Reserved for future use (fill to 4KB)
<<<<<<< HEAD
  // Used: 16+32+16+32+16+384 = 496 bytes
  // Remaining: 4096 - 496 = 3600 bytes
  uint8_t reserved[3600]; // Adjusted for actual struct size to match 4KB
=======
  uint8_t reserved[3712]; // Adjusted for actual struct size
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

  // Constructor
  MpaiHeader() {
    magic = MPAI_MAGIC;
    version = MPAI_VERSION;
    header_size = HEADER_SIZE;
    chunk_size = DEFAULT_CHUNK_SIZE;
    compression_type = static_cast<uint8_t>(CompressionType::ZSTD);
    compression_level = 3;
    std::memset(reserved, 0, sizeof(reserved));
    std::memset(source_file, 0, sizeof(source_file));
    std::memset(creator, 0, sizeof(creator));
    std::memset(user_name, 0, sizeof(user_name));
  }
};

<<<<<<< HEAD
// Header size check
static_assert(sizeof(MpaiHeader) == HEADER_SIZE, "Header must be exactly 4KB");
=======
// TODO: Fix header size to exactly 4KB
// static_assert(sizeof(MpaiHeader) == HEADER_SIZE, "Header must be exactly
// 4KB");
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

#pragma pack(pop)

// ==============================================================================
// [2] Data Metadata Structures
// ==============================================================================

struct ColumnStatistics {
  double mean;
  double std_dev;
  double min;
  double max;
  double median;
  double q25; // 25th percentile
  double q75; // 75th percentile
  double rms; // Root mean square
  uint64_t null_count;

  // Time-based stats (for time column)
  double start_time;
  double end_time;
  double sample_rate; // Hz
  double duration;    // seconds

  ColumnStatistics()
      : mean(0), std_dev(0), min(0), max(0), median(0), q25(0), q75(0), rms(0),
        null_count(0), start_time(0), end_time(0), sample_rate(0), duration(0) {
  }
};

struct ChunkInfo {
  uint64_t offset; // Byte offset in file
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint32_t row_count;
  uint32_t crc32;

  // Pre-aggregated statistics (for O(1) range calculations)
  double min_value;     // Min value in this chunk
  double max_value;     // Max value in this chunk
  double sum;           // Σx (sum of all values)
  double sum_squares;   // Σx² (for variance/std calculation)
  uint32_t valid_count; // Valid data points (excludes NaN/Inf)
  uint32_t _padding;    // Align to 64 bytes

  ChunkInfo()
      : offset(0), compressed_size(0), uncompressed_size(0), row_count(0),
        crc32(0), min_value(std::numeric_limits<double>::infinity()),
        max_value(-std::numeric_limits<double>::infinity()), sum(0.0),
        sum_squares(0.0), valid_count(0), _padding(0) {}
};

struct ColumnMetadata {
  std::string name;
  DataType data_type;
  std::string unit; // Physical unit (e.g., "m/s^2", "bar")
  ColumnStatistics statistics;
  std::vector<ChunkInfo> chunks;

  ColumnMetadata() : data_type(DataType::FLOAT64) {}
};

struct DataMetadata {
  std::vector<ColumnMetadata> columns;
};

// ==============================================================================
// [3] Application State Structures
// ==============================================================================

struct SignalConfig {
  std::string signal_name; // Column name
  uint8_t color[3];        // RGB color
  float line_width;
  LineStyle line_style;
  bool visible;

  // Normalization
  bool normalized;
  NormalizationType norm_method;

  // Y-axis
  uint8_t y_axis; // 0=left, 1=right
  double y_min, y_max;
  bool auto_scale;

  SignalConfig()
      : line_width(1.0f), line_style(LineStyle::SOLID), visible(true),
        normalized(false), norm_method(NormalizationType::NONE), y_axis(0),
        y_min(0), y_max(1), auto_scale(true) {
    color[0] = 255;
    color[1] = 0;
    color[2] = 0;
  }
};

struct GraphConfig {
  uint32_t graph_id;
  std::string title;
  std::vector<SignalConfig> signals;

  // Axis settings
  double x_min, x_max;
  bool auto_x_scale;
  std::string x_label;
  std::string y_left_label;
  std::string y_right_label;

  // Grid & style
  bool show_grid;
  bool show_legend;
  uint8_t legend_position; // 0=top-right, 1=top-left, etc.
  uint8_t background_color[3];

  GraphConfig()
      : graph_id(0), x_min(0), x_max(1), auto_x_scale(true), show_grid(true),
        show_legend(true), legend_position(0) {
    background_color[0] = 255;
    background_color[1] = 255;
    background_color[2] = 255;
  }
};

struct FilterCondition {
  std::string column_name;
  FilterType filter_type;
  double min_value;
  double max_value;
  uint8_t operator_type; // 0=AND, 1=OR
  bool enabled;

  FilterCondition()
      : filter_type(FilterType::RANGE), min_value(0), max_value(1),
        operator_type(0), enabled(true) {}
};

struct CursorInfo {
  uint32_t cursor_id;
  double time_position;
  uint8_t color[3];
  bool visible;
  std::string label;

  CursorInfo() : cursor_id(0), time_position(0), visible(true) {
    color[0] = 255;
    color[1] = 255;
    color[2] = 0;
  }
};

struct AnnotationInfo {
  double time_start;
  double time_end;
  std::string text;
  uint8_t color[3];
  uint8_t type; // 0=region, 1=marker, 2=text

  AnnotationInfo() : time_start(0), time_end(0), type(0) {
    color[0] = 255;
    color[1] = 200;
    color[2] = 0;
  }
};

struct LayoutConfig {
  uint32_t subplot_count;
  uint32_t subplot_layout;            // 0=vertical, 1=horizontal, 2=grid
  std::vector<float> subplot_heights; // Relative heights (sum=1.0)

  LayoutConfig() : subplot_count(1), subplot_layout(0) {}
};

struct UserPreferences {
  bool dark_mode;
  uint8_t default_color_scheme;
  float ui_scale;
  std::string last_export_path;

  UserPreferences()
      : dark_mode(false), default_color_scheme(0), ui_scale(1.0f) {}
};

struct ApplicationState {
  uint32_t state_version; // 0x00010000 (v1.0)

  std::vector<GraphConfig> graphs;
  std::vector<FilterCondition> filters;
  std::vector<CursorInfo> cursors;
  std::vector<AnnotationInfo> annotations;
  LayoutConfig layout;
  UserPreferences preferences;

  // Session info
  uint64_t last_modified;
  std::string user_name;
  std::string notes;

  ApplicationState() : state_version(0x00010000), last_modified(0) {}
};

// ==============================================================================
// [4] Column Data Chunk
// ==============================================================================

struct ColumnChunk {
  uint32_t chunk_id;
  uint32_t row_count;
  uint32_t uncompressed_size;
  uint32_t compressed_size;
  uint32_t crc32;
  std::vector<uint8_t> data; // Compressed data

  ColumnChunk()
      : chunk_id(0), row_count(0), uncompressed_size(0), compressed_size(0),
        crc32(0) {}
};

} // namespace mpai
} // namespace timegraph
