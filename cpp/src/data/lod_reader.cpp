/**
 * LOD Reader Implementation
 *
 * Memory-mapped reading of binary LOD files.
 * Zero-copy access for high performance visualization.
 */

#include "timegraph/data/lod_reader.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace timegraph {
namespace lod {

LodReader::~LodReader() { close(); }

LodReader::LodReader(LodReader &&other) noexcept { *this = std::move(other); }

LodReader &LodReader::operator=(LodReader &&other) noexcept {
  if (this != &other) {
    close();

    filepath_ = std::move(other.filepath_);
    file_size_ = other.file_size_;
    mapped_data_ = other.mapped_data_;
    header_ = other.header_;
    metadata_ = std::move(other.metadata_);
    column_index_map_ = std::move(other.column_index_map_);
    bucket_data_ = other.bucket_data_;
    doubles_per_bucket_ = other.doubles_per_bucket_;

#ifdef _WIN32
    file_handle_ = other.file_handle_;
    mapping_handle_ = other.mapping_handle_;
    other.file_handle_ = INVALID_HANDLE_VALUE;
    other.mapping_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif

    other.mapped_data_ = nullptr;
    other.bucket_data_ = nullptr;
    other.file_size_ = 0;
  }
  return *this;
}

bool LodReader::open(const std::string &filepath) {
  close();
  filepath_ = filepath;

#ifdef _WIN32
  // Windows memory mapping
  file_handle_ =
      CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

  if (file_handle_ == INVALID_HANDLE_VALUE) {
    std::cerr << "[LOD] Failed to open file: " << filepath << std::endl;
    return false;
  }

  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(file_handle_, &file_size)) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
  file_size_ = static_cast<size_t>(file_size.QuadPart);

  mapping_handle_ =
      CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);

  if (!mapping_handle_) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }

  mapped_data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);

  if (!mapped_data_) {
    CloseHandle(mapping_handle_);
    CloseHandle(file_handle_);
    mapping_handle_ = nullptr;
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }

#else
  // POSIX memory mapping
  fd_ = ::open(filepath.c_str(), O_RDONLY);
  if (fd_ < 0) {
    std::cerr << "[LOD] Failed to open file: " << filepath << std::endl;
    return false;
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  file_size_ = static_cast<size_t>(st.st_size);

  mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped_data_ == MAP_FAILED) {
    mapped_data_ = nullptr;
    ::close(fd_);
    fd_ = -1;
    return false;
  }
#endif

  // Parse header and metadata
  if (!parse_header()) {
    close();
    return false;
  }

  if (!parse_column_names()) {
    close();
    return false;
  }

  // Set up bucket data pointer
  doubles_per_bucket_ = doubles_per_bucket(header_.column_count);
  bucket_data_ = reinterpret_cast<const double *>(
      static_cast<const uint8_t *>(mapped_data_) + header_.data_offset);

  std::cout << "[LOD] Opened: " << filepath << " (" << header_.bucket_count
            << " buckets, " << header_.column_count << " signals)" << std::endl;

  return true;
}

void LodReader::close() {
  if (!mapped_data_)
    return;

#ifdef _WIN32
  if (mapped_data_) {
    UnmapViewOfFile(mapped_data_);
    mapped_data_ = nullptr;
  }
  if (mapping_handle_) {
    CloseHandle(mapping_handle_);
    mapping_handle_ = nullptr;
  }
  if (file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (mapped_data_) {
    munmap(mapped_data_, file_size_);
    mapped_data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#endif

  bucket_data_ = nullptr;
  file_size_ = 0;
  column_index_map_.clear();
  metadata_.column_names.clear();
}

bool LodReader::parse_header() {
  if (file_size_ < LOD_HEADER_SIZE) {
    std::cerr << "[LOD] File too small for header" << std::endl;
    return false;
  }

  std::memcpy(&header_, mapped_data_, sizeof(LodHeader));

  if (header_.magic != LOD_MAGIC) {
    std::cerr << "[LOD] Invalid magic number" << std::endl;
    return false;
  }

  if ((header_.version >> 16) > 1) {
    std::cerr << "[LOD] Unsupported version" << std::endl;
    return false;
  }

  // Copy to metadata
  metadata_.bucket_size = header_.bucket_size;
  metadata_.bucket_count = header_.bucket_count;
  metadata_.total_row_count = header_.total_row_count;

  return true;
}

bool LodReader::parse_column_names() {
  const uint8_t *names_ptr =
      static_cast<const uint8_t *>(mapped_data_) + header_.names_offset;
  const uint8_t *data_ptr =
      static_cast<const uint8_t *>(mapped_data_) + header_.data_offset;

  // Parse null-terminated strings
  metadata_.column_names.clear();
  column_index_map_.clear();

  const char *current = reinterpret_cast<const char *>(names_ptr);
  const char *end = reinterpret_cast<const char *>(data_ptr);

  uint32_t index = 0;
  while (current < end && index < header_.column_count) {
    std::string name(current);
    if (name.empty())
      break;

    metadata_.column_names.push_back(name);
    column_index_map_[name] = index;

    current += name.length() + 1; // Skip null terminator
    index++;
  }

  if (metadata_.column_names.size() != header_.column_count) {
    std::cerr << "[LOD] Column count mismatch: expected "
              << header_.column_count << ", got "
              << metadata_.column_names.size() << std::endl;
    return false;
  }

  return true;
}

int LodReader::get_column_index(const std::string &name) const {
  auto it = column_index_map_.find(name);
  if (it != column_index_map_.end()) {
    return static_cast<int>(it->second);
  }
  return -1;
}

std::pair<double, double> LodReader::get_time_range(uint64_t start_bucket,
                                                    uint64_t count) const {
  if (!bucket_data_ || start_bucket >= header_.bucket_count) {
    return {0.0, 0.0};
  }

  count = std::min(count, header_.bucket_count - start_bucket);
  if (count == 0)
    return {0.0, 0.0};

  double time_min = std::numeric_limits<double>::max();
  double time_max = std::numeric_limits<double>::lowest();

  for (uint64_t i = 0; i < count; ++i) {
    const double *bucket = get_bucket_ptr(start_bucket + i);
    if (bucket) {
      time_min = std::min(time_min, bucket[0]); // time_min at offset 0
      time_max = std::max(time_max, bucket[1]); // time_max at offset 1
    }
  }

  return {time_min, time_max};
}

std::pair<double, double> LodReader::get_signal_range(uint32_t column_index,
                                                      uint64_t start_bucket,
                                                      uint64_t count) const {
  if (!bucket_data_ || column_index >= header_.column_count) {
    return {0.0, 0.0};
  }

  count = std::min(count, header_.bucket_count - start_bucket);
  if (count == 0)
    return {0.0, 0.0};

  double value_min = std::numeric_limits<double>::max();
  double value_max = std::numeric_limits<double>::lowest();

  // Offset to this signal's min/max in bucket
  // Layout: [time_min, time_max, sig0_min, sig0_max, sig1_min, sig1_max, ...]
  size_t min_offset = 2 + column_index * 2;     // sig_min
  size_t max_offset = 2 + column_index * 2 + 1; // sig_max

  for (uint64_t i = 0; i < count; ++i) {
    const double *bucket = get_bucket_ptr(start_bucket + i);
    if (bucket) {
      value_min = std::min(value_min, bucket[min_offset]);
      value_max = std::max(value_max, bucket[max_offset]);
    }
  }

  return {value_min, value_max};
}

void LodReader::get_render_data(uint32_t column_index, uint64_t start_bucket,
                                uint64_t count, std::vector<double> &out_time,
                                std::vector<double> &out_values) const {
  out_time.clear();
  out_values.clear();

  if (!bucket_data_ || column_index >= header_.column_count) {
    return;
  }

  count = std::min(count, header_.bucket_count - start_bucket);
  if (count == 0)
    return;

  // Reserve space: 2 points per bucket (min and max)
  out_time.reserve(count * 2);
  out_values.reserve(count * 2);

  // Offsets
  size_t min_offset = 2 + column_index * 2;
  size_t max_offset = 2 + column_index * 2 + 1;

  for (uint64_t i = 0; i < count; ++i) {
    const double *bucket = get_bucket_ptr(start_bucket + i);
    if (!bucket)
      continue;

    double t_min = bucket[0];
    double t_max = bucket[1];
    double v_min = bucket[min_offset];
    double v_max = bucket[max_offset];

    // Add min point first, then max point
    // This ensures spikes are visible
    if (v_min <= v_max) {
      out_time.push_back(t_min);
      out_values.push_back(v_min);
      out_time.push_back(t_max);
      out_values.push_back(v_max);
    } else {
      out_time.push_back(t_min);
      out_values.push_back(v_max);
      out_time.push_back(t_max);
      out_values.push_back(v_min);
    }
  }
}

void LodReader::get_render_data(const std::string &column_name,
                                uint64_t start_bucket, uint64_t count,
                                std::vector<double> &out_time,
                                std::vector<double> &out_values) const {
  int idx = get_column_index(column_name);
  if (idx < 0) {
    out_time.clear();
    out_values.clear();
    return;
  }
  get_render_data(static_cast<uint32_t>(idx), start_bucket, count, out_time,
                  out_values);
}

std::pair<uint64_t, uint64_t>
LodReader::find_bucket_range(double time_start, double time_end) const {
  if (!bucket_data_ || header_.bucket_count == 0) {
    return {0, 0};
  }

  // Binary search for start bucket
  uint64_t lo = 0;
  uint64_t hi = header_.bucket_count;

  // Find first bucket where time_max >= time_start
  while (lo < hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    const double *bucket = get_bucket_ptr(mid);
    if (!bucket || bucket[1] < time_start) { // time_max < time_start
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  uint64_t start_bucket = lo;

  // Find last bucket where time_min <= time_end
  lo = start_bucket;
  hi = header_.bucket_count;

  while (lo < hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    const double *bucket = get_bucket_ptr(mid);
    if (!bucket || bucket[0] <= time_end) { // time_min <= time_end
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  uint64_t end_bucket = lo;

  if (start_bucket >= end_bucket) {
    return {0, 0};
  }

  return {start_bucket, end_bucket - start_bucket};
}

} // namespace lod
} // namespace timegraph
