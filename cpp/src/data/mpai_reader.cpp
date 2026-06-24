/**
 * MPAI Reader Implementation
 *
 * Memory-mapped I/O for ultra-low memory usage
 * Target: 300 MB RAM for 50 GB files
 */

#include "timegraph/data/mpai_reader.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>


// Platform-specific includes
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

// ZSTD decompression (conditional)
#ifdef HAVE_ZSTD
#include <zstd.h>
#else
#pragma message("Warning: ZSTD not available, decompression disabled")
#endif

namespace timegraph {
namespace mpai {

MpaiReader::MpaiReader(const std::string &filename, bool use_mmap)
    : filename_(filename), use_mmap_(use_mmap), fd_(-1), mmap_data_(nullptr),
      mmap_size_(0), max_cache_size_(10) // Max 10 chunks in cache (~80 MB)
{
  open_file();
  load_header();
  load_data_metadata();
  load_application_state();
  load_application_state();
  build_column_index();
  build_chunk_index();
}

MpaiReader::~MpaiReader() {
#ifdef _WIN32
  if (mmap_data_) {
    UnmapViewOfFile(mmap_data_);
  }
  if (fd_ != -1) {
    CloseHandle(reinterpret_cast<HANDLE>(fd_));
  }
#else
  if (mmap_data_) {
    munmap(mmap_data_, mmap_size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
#endif
}

const MpaiHeader &MpaiReader::get_header() const { return header_; }

const DataMetadata &MpaiReader::get_data_metadata() const {
  return data_metadata_;
}

const ApplicationState &MpaiReader::get_application_state() const {
  return app_state_;
}

const ColumnMetadata *
MpaiReader::get_column_metadata(const std::string &column_name) const {
  auto it = column_index_map_.find(column_name);
  if (it == column_index_map_.end()) {
    return nullptr;
  }
  return &data_metadata_.columns[it->second];
}

const ColumnMetadata &
MpaiReader::get_column_metadata(uint32_t column_index) const {
  if (column_index >= data_metadata_.columns.size()) {
    throw std::out_of_range("Column index out of range");
  }
  return data_metadata_.columns[column_index];
}

const ColumnStatistics &
MpaiReader::get_statistics(const std::string &column_name) const {
  const ColumnMetadata *meta = get_column_metadata(column_name);
  if (!meta) {
    throw std::runtime_error("Column not found: " + column_name);
  }
  return meta->statistics;
}

std::vector<double> MpaiReader::load_column_chunk(uint32_t column_index,
                                                  uint32_t chunk_id) {
  // Check cache first
  const std::vector<double> *cached = get_from_cache(column_index, chunk_id);
  if (cached) {
    return *cached;
  }

  // Get column metadata
  const ColumnMetadata &col_meta = get_column_metadata(column_index);

  if (chunk_id >= col_meta.chunks.size()) {
    throw std::out_of_range("Chunk ID out of range");
  }

  const ChunkInfo &chunk_info = col_meta.chunks[chunk_id];

  // Read compressed data
  std::vector<uint8_t> compressed =
      read_compressed_chunk(chunk_info.offset, chunk_info.compressed_size);

  // Decompress
  std::vector<double> decompressed =
      decompress_chunk(compressed, chunk_info.uncompressed_size);

  // Add to cache
  add_to_cache(column_index, chunk_id, decompressed);

  return decompressed;
}

std::vector<double> MpaiReader::load_column(const std::string &column_name) {
  auto it = column_index_map_.find(column_name);
  if (it == column_index_map_.end()) {
    throw std::runtime_error("Column not found: " + column_name);
  }

  uint32_t column_index = it->second;
  const ColumnMetadata &col_meta = data_metadata_.columns[column_index];

  // Load all chunks
  std::vector<double> result;
  result.reserve(header_.row_count);

  for (size_t chunk_id = 0; chunk_id < col_meta.chunks.size(); ++chunk_id) {
    std::vector<double> chunk_data = load_column_chunk(column_index, chunk_id);
    result.insert(result.end(), chunk_data.begin(), chunk_data.end());
  }

  return result;
}

std::vector<double>
MpaiReader::load_column_slice(const std::string &column_name,
                              uint64_t start_row, uint64_t row_count) {
  auto it = column_index_map_.find(column_name);
  if (it == column_index_map_.end()) {
    throw std::runtime_error("Column not found: " + column_name);
  }

  uint32_t column_index = it->second;
  const ColumnMetadata &col_meta = data_metadata_.columns[column_index];

  std::vector<double> result;
  result.reserve(row_count);

  // Empty request check
  if (row_count == 0)
    return result;

  // Use pre-built chunk index for O(log N) lookup
  const std::vector<uint64_t> &start_rows = chunk_start_rows_[column_index];

  if (start_rows.empty())
    return result;

  // Find first chunk containing start_row using binary search (upper_bound)
  // upper_bound returns iterator to first element > value
  auto it_start =
      std::upper_bound(start_rows.begin(), start_rows.end(), start_row);

  // The chunk containing start_row is the one BEFORE upper_bound
  uint32_t start_chunk_idx = 0;
  if (it_start != start_rows.begin()) {
    start_chunk_idx =
        static_cast<uint32_t>(std::distance(start_rows.begin(), it_start)) - 1;
  }

  // Iterate from start chunk until we have enough data
  uint64_t current_row_offset = start_rows[start_chunk_idx];
  uint64_t rows_collected = 0;
  uint64_t req_end_row = start_row + row_count;

  for (size_t i = start_chunk_idx; i < col_meta.chunks.size(); ++i) {
    // Chunk start row (from cache)
    uint64_t chunk_start = start_rows[i];
    uint64_t chunk_len = col_meta.chunks[i].row_count;
    uint64_t chunk_end = chunk_start + chunk_len;

    // Update current offset just to be safe, though cache should be correct
    current_row_offset = chunk_start;

    if (chunk_end > start_row) {
      // This chunk overlaps with request
      std::vector<double> chunk_data =
          load_column_chunk(column_index, static_cast<uint32_t>(i));

      // Calculate local overlap
      int64_t local_start_signed =
          static_cast<int64_t>(start_row) - static_cast<int64_t>(chunk_start);
      uint64_t local_start = (local_start_signed > 0)
                                 ? static_cast<uint64_t>(local_start_signed)
                                 : 0;

      uint64_t remaining_needed = row_count - rows_collected;
      uint64_t available_in_chunk = chunk_data.size() - local_start;

      uint64_t to_copy = std::min(remaining_needed, available_in_chunk);

      if (to_copy > 0) {
        result.insert(result.end(), chunk_data.begin() + local_start,
                      chunk_data.begin() + local_start + to_copy);

        rows_collected += to_copy;
      }
    }

    if (rows_collected >= row_count)
      break;
  }

  return result;
}

uint64_t MpaiReader::get_row_count() const { return header_.row_count; }

uint32_t MpaiReader::get_column_count() const { return header_.column_count; }

std::vector<std::string> MpaiReader::get_column_names() const {
  std::vector<std::string> names;
  names.reserve(data_metadata_.columns.size());
  for (const auto &col : data_metadata_.columns) {
    names.push_back(col.name);
  }
  return names;
}

bool MpaiReader::has_column(const std::string &column_name) const {
  return column_index_map_.find(column_name) != column_index_map_.end();
}

uint64_t MpaiReader::get_file_size() const { return header_.file_size; }

double MpaiReader::get_compression_ratio() const {
  if (header_.compressed_size == 0)
    return 0.0;
  return static_cast<double>(header_.uncompressed_size) /
         static_cast<double>(header_.compressed_size);
}

size_t MpaiReader::get_memory_usage() const {
  size_t total = 0;

  // Cached chunks
  for (const auto &entry : chunk_cache_) {
    total += entry.data.size() * sizeof(double);
  }

  // Metadata (approximate)
  total += sizeof(header_);
  total += data_metadata_.columns.size() * sizeof(ColumnMetadata);

  return total;
}

std::unordered_map<std::string, double>
MpaiReader::get_batch_values(double timestamp, const std::string &time_col,
                             const std::vector<std::string> &signal_cols) {

  std::unordered_map<std::string, double> result;

  // 1. Get time stats to estimate row
  const ColumnStatistics &time_stats = get_statistics(time_col);
  double t_min = time_stats.min;
  double t_max = time_stats.max;

  // ✅ FIX: If statistics are zero/invalid, use chunk metadata instead
  // This happens when the MPAI file was created without proper statistics
  if (t_min == 0.0 && t_max == 0.0) {
    auto it = column_index_map_.find(time_col);
    if (it != column_index_map_.end()) {
      uint32_t col_idx = it->second;
      const auto &meta = data_metadata_.columns[col_idx];
      if (!meta.chunks.empty()) {
        // Get min from first chunk, max from last chunk
        t_min = meta.chunks.front().min_value;
        t_max = meta.chunks.back().max_value;
      }
    }
  }

  // CLAMP TIMESTAMP to [min, max] range
  // This ensures we always return a value if the cursor is at start/end
  // even if slightly out of bounds due to float precision
  double clamped_timestamp = timestamp;
  if (clamped_timestamp < t_min)
    clamped_timestamp = t_min;
  if (clamped_timestamp > t_max)
    clamped_timestamp = t_max;

  uint64_t total_rows = header_.row_count;
  if (total_rows < 2)
    return result;

  // 2. Estimate row index (assuming roughly linear time)
  double fraction = (clamped_timestamp - t_min) / (t_max - t_min);

  // Safety check for NaN/Inf
  if (!std::isfinite(fraction) || t_max <= t_min) {
    fraction = 0.0;
  }

  // Scan backwards/forwards from this estimate
  uint64_t est_row = static_cast<uint64_t>(fraction * (total_rows - 1));

  // 3. Load small window around estimate to find exact bracket
  // Window size: 200 rows (should catch most non-uniformity)
  const uint64_t WINDOW_SIZE = 200;
  uint64_t start_row =
      (est_row > WINDOW_SIZE / 2) ? (est_row - WINDOW_SIZE / 2) : 0;

  if (start_row + WINDOW_SIZE > total_rows) {
    start_row = (total_rows > WINDOW_SIZE) ? (total_rows - WINDOW_SIZE) : 0;
  }
  uint64_t count = std::min(WINDOW_SIZE, total_rows - start_row);

  // LOAD TIME WINDOW (optimized with binary search inside load_column_slice)
  std::vector<double> time_window =
      load_column_slice(time_col, start_row, count);

  if (time_window.empty())
    return result;

  // 4. local search for t[i] <= timestamp <= t[i+1]
  int found_idx = -1;

  // Refined search in window
  for (size_t i = 0; i < time_window.size() - 1; ++i) {
    if (clamped_timestamp >= time_window[i] &&
        clamped_timestamp <= time_window[i + 1]) {
      found_idx = static_cast<int>(i);
      break;
    }
  }

  // If not found, check if we are at the very ends of the window (start/end of
  // file case) Or if we need to search wider (not implementing wider search for
  // now as 200 is robust enough for linear)
  if (found_idx == -1) {
    if (clamped_timestamp <= time_window.front())
      found_idx = 0;
    else if (clamped_timestamp >= time_window.back())
      found_idx = static_cast<int>(time_window.size()) - 2;
  }

  if (found_idx < 0)
    found_idx = 0;
  if (found_idx >= static_cast<int>(time_window.size()) - 1)
    found_idx = static_cast<int>(time_window.size()) - 2;

  // 5. Calculate interpolation factor
  double t0 = time_window[found_idx];
  double t1 = time_window[found_idx + 1];
  double factor = 0.0;
  if (t1 > t0) {
    factor = (clamped_timestamp - t0) / (t1 - t0);
  }
  // Clamp factor (0-1)
  if (factor < 0.0)
    factor = 0.0;
  if (factor > 1.0)
    factor = 1.0;

  uint64_t row_idx_0 = start_row + found_idx;

  // 6. Batch load values for all signals (just 2 rows!)
  for (const auto &sig_name : signal_cols) {
    try {
      // Optimization: load just 2 rows
      // This uses binary search (log N) now, very fast
      std::vector<double> vals = load_column_slice(sig_name, row_idx_0, 2);

      if (vals.size() >= 2) {
        double v0 = vals[0];
        double v1 = vals[1];
        double val = v0 + factor * (v1 - v0);
        result[sig_name] = val;
      } else if (vals.size() == 1) {
        result[sig_name] = vals[0];
      } else {
        result[sig_name] = 0.0; // Default
      }
    } catch (...) {
      // Ignore missing columns
    }
  }

  return result;
}

// Private helper methods

void MpaiReader::open_file() {
#ifdef _WIN32
  // Windows: CreateFile + CreateFileMapping + MapViewOfFile

  // Convert UTF-8 filename to UTF-16 (Wide Char) for Windows API
  int wlen = MultiByteToWideChar(CP_UTF8, 0, filename_.c_str(), -1, NULL, 0);
  std::wstring wfilename(wlen, 0);
  MultiByteToWideChar(CP_UTF8, 0, filename_.c_str(), -1, &wfilename[0], wlen);

  HANDLE hFile = CreateFileW(wfilename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

  if (hFile == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("Failed to open file: " + filename_);
  }

  fd_ = reinterpret_cast<int>(hFile);

  // Get file size
  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(hFile, &file_size)) {
    CloseHandle(hFile);
    throw std::runtime_error("Failed to get file size");
  }

  mmap_size_ = static_cast<size_t>(file_size.QuadPart);

  if (use_mmap_ && mmap_size_ > 0) {
    // Create file mapping
    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (!hMapping) {
      CloseHandle(hFile);
      throw std::runtime_error("Failed to create file mapping");
    }

    // Map view of file
    mmap_data_ = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0,
                               0 // Map entire file
    );

    CloseHandle(hMapping); // Can close mapping handle after MapViewOfFile

    if (!mmap_data_) {
      CloseHandle(hFile);
      throw std::runtime_error("Failed to map view of file");
    }
  }
#else
  // Unix: open + mmap
  fd_ = open(filename_.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file: " + filename_);
  }

  // Get file size
  struct stat st;
  if (fstat(fd_, &st) == -1) {
    close(fd_);
    throw std::runtime_error("Failed to get file size");
  }

  mmap_size_ = st.st_size;

  if (use_mmap_ && mmap_size_ > 0) {
    mmap_data_ = mmap(NULL, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_data_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("Failed to mmap file");
    }
  }
#endif
}

void MpaiReader::load_header() {
  if (mmap_data_) {
    // Read from memory-mapped region
    std::memcpy(&header_, mmap_data_, sizeof(header_));
  } else {
    // Read from file
    std::ifstream file(filename_, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(&header_), sizeof(header_))) {
      throw std::runtime_error("Failed to read header");
    }
  }

  // Validate magic
  if (header_.magic != MPAI_MAGIC) {
    throw std::runtime_error("Invalid MPAI file: bad magic number");
  }

  // Validate version
  if (header_.version != MPAI_VERSION) {
    throw std::runtime_error("Unsupported MPAI version");
  }
}

void MpaiReader::load_data_metadata() {
  // Open file for reading metadata
  std::ifstream file(filename_, std::ios::binary);

  file.seekg(header_.data_metadata_offset);

  // Read column count
  uint32_t column_count = 0;
  file.read(reinterpret_cast<char *>(&column_count), sizeof(column_count));

  data_metadata_.columns.resize(column_count);

  // Read each column metadata
  for (uint32_t i = 0; i < column_count; ++i) {
    ColumnMetadata &col = data_metadata_.columns[i];

    // Read column name
    uint32_t name_len;
    file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    col.name.resize(name_len);
    file.read(&col.name[0], name_len);

    // Read data type
    file.read(reinterpret_cast<char *>(&col.data_type), sizeof(col.data_type));

    // Read statistics
    file.read(reinterpret_cast<char *>(&col.statistics),
              sizeof(col.statistics));

    // Read chunk count
    uint32_t chunk_count;
    file.read(reinterpret_cast<char *>(&chunk_count), sizeof(chunk_count));
    col.chunks.resize(chunk_count);

    // Read chunk info array
    for (uint32_t j = 0; j < chunk_count; ++j) {
      file.read(reinterpret_cast<char *>(&col.chunks[j]), sizeof(ChunkInfo));
    }
  }
}

void MpaiReader::load_application_state() {
  if (header_.app_state_offset == 0) {
    // No application state
    return;
  }

  std::ifstream file(filename_, std::ios::binary);
  file.seekg(header_.app_state_offset);

  // Read state version
  file.read(reinterpret_cast<char *>(&app_state_.state_version),
            sizeof(app_state_.state_version));

  // Read graph count
  uint32_t graph_count;
  file.read(reinterpret_cast<char *>(&graph_count), sizeof(graph_count));
  app_state_.graphs.resize(graph_count);

  // Read each graph config
  for (uint32_t i = 0; i < graph_count; ++i) {
    GraphConfig &graph = app_state_.graphs[i];

    // Graph ID
    file.read(reinterpret_cast<char *>(&graph.graph_id),
              sizeof(graph.graph_id));

    // Title
    uint32_t title_len;
    file.read(reinterpret_cast<char *>(&title_len), sizeof(title_len));
    graph.title.resize(title_len);
    file.read(&graph.title[0], title_len);

    // Signal count
    uint32_t signal_count;
    file.read(reinterpret_cast<char *>(&signal_count), sizeof(signal_count));
    graph.signals.resize(signal_count);

    // Read each signal
    for (uint32_t j = 0; j < signal_count; ++j) {
      SignalConfig &signal = graph.signals[j];

      // Signal name
      uint32_t name_len;
      file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
      signal.signal_name.resize(name_len);
      file.read(&signal.signal_name[0], name_len);

      // Color
      file.read(reinterpret_cast<char *>(signal.color), sizeof(signal.color));

      // Other properties
      file.read(reinterpret_cast<char *>(&signal.line_width),
                sizeof(signal.line_width));
      file.read(reinterpret_cast<char *>(&signal.line_style),
                sizeof(signal.line_style));
      file.read(reinterpret_cast<char *>(&signal.visible),
                sizeof(signal.visible));
      file.read(reinterpret_cast<char *>(&signal.normalized),
                sizeof(signal.normalized));
      file.read(reinterpret_cast<char *>(&signal.norm_method),
                sizeof(signal.norm_method));
      file.read(reinterpret_cast<char *>(&signal.y_axis),
                sizeof(signal.y_axis));
      file.read(reinterpret_cast<char *>(&signal.y_min), sizeof(signal.y_min));
      file.read(reinterpret_cast<char *>(&signal.y_max), sizeof(signal.y_max));
      file.read(reinterpret_cast<char *>(&signal.auto_scale),
                sizeof(signal.auto_scale));
    }

    // Axis settings
    file.read(reinterpret_cast<char *>(&graph.x_min), sizeof(graph.x_min));
    file.read(reinterpret_cast<char *>(&graph.x_max), sizeof(graph.x_max));
    file.read(reinterpret_cast<char *>(&graph.auto_x_scale),
              sizeof(graph.auto_x_scale));

    // Labels
    auto read_string = [&file](std::string &str) {
      uint32_t len;
      file.read(reinterpret_cast<char *>(&len), sizeof(len));
      str.resize(len);
      file.read(&str[0], len);
    };

    read_string(graph.x_label);
    read_string(graph.y_left_label);
    read_string(graph.y_right_label);

    // Grid & style
    file.read(reinterpret_cast<char *>(&graph.show_grid),
              sizeof(graph.show_grid));
    file.read(reinterpret_cast<char *>(&graph.show_legend),
              sizeof(graph.show_legend));
    file.read(reinterpret_cast<char *>(&graph.legend_position),
              sizeof(graph.legend_position));
    file.read(reinterpret_cast<char *>(graph.background_color),
              sizeof(graph.background_color));
  }

  // Read layout
  file.read(reinterpret_cast<char *>(&app_state_.layout.subplot_count),
            sizeof(app_state_.layout.subplot_count));
  file.read(reinterpret_cast<char *>(&app_state_.layout.subplot_layout),
            sizeof(app_state_.layout.subplot_layout));
}

void MpaiReader::build_column_index() {
  for (size_t i = 0; i < data_metadata_.columns.size(); ++i) {
    column_index_map_[data_metadata_.columns[i].name] =
        static_cast<uint32_t>(i);
  }
}

void MpaiReader::build_chunk_index() {
  chunk_start_rows_.resize(data_metadata_.columns.size());

  for (size_t i = 0; i < data_metadata_.columns.size(); ++i) {
    const auto &col = data_metadata_.columns[i];
    auto &start_rows = chunk_start_rows_[i];
    start_rows.reserve(col.chunks.size());

    uint64_t current_row = 0;
    for (const auto &chunk : col.chunks) {
      start_rows.push_back(current_row);
      current_row += chunk.row_count;
    }
  }
}

std::vector<uint8_t> MpaiReader::read_compressed_chunk(uint64_t offset,
                                                       uint32_t size) {
  std::vector<uint8_t> data(size);

  if (mmap_data_) {
    // Read from memory-mapped region
    const uint8_t *src = static_cast<const uint8_t *>(mmap_data_) + offset;
    std::memcpy(data.data(), src, size);
  } else {
    // Read from file
    std::ifstream file(filename_, std::ios::binary);
    file.seekg(offset);
    file.read(reinterpret_cast<char *>(data.data()), size);
  }

  return data;
}

std::vector<double>
MpaiReader::decompress_chunk(const std::vector<uint8_t> &compressed_data,
                             uint32_t uncompressed_size) {
#ifdef HAVE_ZSTD
  std::vector<double> result(uncompressed_size / sizeof(double));

  size_t decompressed_size =
      ZSTD_decompress(result.data(), uncompressed_size, compressed_data.data(),
                      compressed_data.size());

  if (ZSTD_isError(decompressed_size)) {
    throw std::runtime_error("ZSTD decompression failed: " +
                             std::string(ZSTD_getErrorName(decompressed_size)));
  }

  return result;
#else
  // No compression - just copy data
  std::vector<double> result(compressed_data.size() / sizeof(double));
  std::memcpy(result.data(), compressed_data.data(), compressed_data.size());
  return result;
#endif
}

void MpaiReader::add_to_cache(uint32_t column_index, uint32_t chunk_id,
                              const std::vector<double> &data) {
  // Check if cache is full
  if (chunk_cache_.size() >= max_cache_size_) {
    evict_oldest_cache_entry();
  }

  ChunkCacheEntry entry;
  entry.column_index = column_index;
  entry.chunk_id = chunk_id;
  entry.data = data;
  entry.last_access_time = std::time(nullptr);

  chunk_cache_.push_back(entry);
}

const std::vector<double> *MpaiReader::get_from_cache(uint32_t column_index,
                                                      uint32_t chunk_id) {
  for (auto &entry : chunk_cache_) {
    if (entry.column_index == column_index && entry.chunk_id == chunk_id) {
      entry.last_access_time = std::time(nullptr);
      return &entry.data;
    }
  }
  return nullptr;
}

void MpaiReader::evict_oldest_cache_entry() {
  if (chunk_cache_.empty())
    return;

  // Find oldest entry
  auto oldest = chunk_cache_.begin();
  for (auto it = chunk_cache_.begin(); it != chunk_cache_.end(); ++it) {
    if (it->last_access_time < oldest->last_access_time) {
      oldest = it;
    }
  }

  chunk_cache_.erase(oldest);
}

std::pair<uint32_t, uint32_t>
MpaiReader::get_chunk_range(uint32_t column_index, uint64_t start_row,
                            uint64_t end_row) const {
  if (column_index >= chunk_start_rows_.size()) {
    return {0, 0};
  }
  const auto &start_rows = chunk_start_rows_[column_index];
  if (start_rows.empty()) {
    return {0, 0};
  }

  // Find first chunk that *could* contain start_row
  // upper_bound returns iterator to first element > start_row
  // predecessor contains start_row
  auto it_start =
      std::upper_bound(start_rows.begin(), start_rows.end(), start_row);
  uint32_t start_idx = 0;
  if (it_start != start_rows.begin()) {
    start_idx =
        static_cast<uint32_t>(std::distance(start_rows.begin(), it_start) - 1);
  }

  // Find last chunk that starts *before* end_row
  // lower_bound returns iterator to first element >= end_row
  auto it_end = std::lower_bound(start_rows.begin(), start_rows.end(), end_row);
  uint32_t end_idx = 0;
  if (it_end != start_rows.begin()) {
    end_idx =
        static_cast<uint32_t>(std::distance(start_rows.begin(), it_end) - 1);
  } else {
    end_idx = 0;
  }

  // Safety clamp
  if (start_idx >= start_rows.size()) {
    start_idx = static_cast<uint32_t>(start_rows.size() - 1);
  }
  if (end_idx >= start_rows.size()) {
    end_idx = static_cast<uint32_t>(start_rows.size() - 1);
  }

  if (start_idx > end_idx) {
    if (start_idx > 0)
      end_idx = start_idx;
    else
      start_idx = end_idx;
  }

  return {start_idx, end_idx};
}

uint64_t MpaiReader::get_row_for_value(const std::string &column_name,
                                       double value) const {
  auto it = column_index_map_.find(column_name);
  if (it == column_index_map_.end()) {
    return 0;
  }
  uint32_t col_idx = it->second;
  const auto &meta = data_metadata_.columns[col_idx];

  if (meta.chunks.empty()) {
    return 0;
  }

  // 1. Binary search for chunk based on value (assuming monotonically
  // increasing) Find first chunk where chunk.max_value >= value
  auto chunk_it = std::lower_bound(
      meta.chunks.begin(), meta.chunks.end(), value,
      [](const ChunkInfo &chunk, double val) { return chunk.max_value < val; });

  if (chunk_it == meta.chunks.end()) {
    // Value larger than all chunks -> return total rows
    return get_row_count();
  }

  // Check if value is before this chunk (gap logic)
  if (value < chunk_it->min_value) {
    // Value is in gap before this chunk. Return start of this chunk.
    // (Because previous chunk max < value < current min)
  }

  size_t chunk_idx = std::distance(meta.chunks.begin(), chunk_it);

  // Get cached start row
  uint64_t chunk_start_row = 0;
  if (col_idx < chunk_start_rows_.size() &&
      chunk_idx < chunk_start_rows_[col_idx].size()) {
    chunk_start_row = chunk_start_rows_[col_idx][chunk_idx];
  }

  // 2. Linear interpolate within chunk limits
  double t_min = chunk_it->min_value;
  double t_max = chunk_it->max_value;
  uint64_t rows = chunk_it->row_count;

  if (rows <= 1 || t_max <= t_min) {
    return chunk_start_row;
  }

  double ratio = (value - t_min) / (t_max - t_min);
  ratio = std::max(0.0, std::min(1.0, ratio));

  return chunk_start_row +
         static_cast<uint64_t>(ratio * static_cast<double>(rows - 1));
}

} // namespace mpai
} // namespace timegraph
