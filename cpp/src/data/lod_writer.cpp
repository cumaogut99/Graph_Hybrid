/**
 * LOD Writer Implementation
 *
 * Generates binary LOD files with streaming bucket writes.
 * Used during data conversion to create pre-computed LOD layers.
 */

#include "timegraph/data/lod_writer.hpp"
#include <cstring>
#include <iostream>

namespace timegraph {
namespace lod {

LodWriter::~LodWriter() {
  if (is_open()) {
    finalize();
  }
}

bool LodWriter::create(const std::string &filepath, uint32_t bucket_size,
                       const std::vector<std::string> &column_names,
                       uint64_t total_row_count) {
  if (is_open()) {
    std::cerr << "[LOD] File already open, call finalize() first" << std::endl;
    return false;
  }

  filepath_ = filepath;
  column_names_ = column_names;
  bucket_count_ = 0;

  // Initialize header
  header_ = LodHeader();
  header_.bucket_size = bucket_size;
  header_.column_count = static_cast<uint32_t>(column_names.size());
  header_.total_row_count = total_row_count;
  header_.names_offset = LOD_HEADER_SIZE;

  // Calculate column names section size
  size_t names_size = 0;
  for (const auto &name : column_names) {
    names_size += name.length() + 1; // Include null terminator
  }
  // Align to 8 bytes
  names_size = (names_size + 7) & ~7;

  header_.data_offset = LOD_HEADER_SIZE + names_size;
  data_offset_ = header_.data_offset;

  // Set up write buffer
  doubles_per_bucket_ = doubles_per_bucket(header_.column_count);
  write_buffer_.resize(doubles_per_bucket_);

  // Open file
  file_.open(filepath, std::ios::binary | std::ios::trunc);
  if (!file_.is_open()) {
    std::cerr << "[LOD] Failed to create file: " << filepath << std::endl;
    return false;
  }

  // Write header placeholder (will be updated in finalize)
  if (!write_header()) {
    file_.close();
    return false;
  }

  // Write column names
  if (!write_column_names()) {
    file_.close();
    return false;
  }

  // Seek to data offset
  file_.seekp(data_offset_);

  std::cout << "[LOD] Creating: " << filepath << " (bucket_size=" << bucket_size
            << ", columns=" << column_names.size() << ")" << std::endl;

  return true;
}

bool LodWriter::write_header() {
  file_.seekp(0);
  file_.write(reinterpret_cast<const char *>(&header_), sizeof(LodHeader));
  return file_.good();
}

bool LodWriter::write_column_names() {
  file_.seekp(header_.names_offset);

  for (const auto &name : column_names_) {
    file_.write(name.c_str(), name.length() + 1); // Include null terminator
  }

  // Pad to alignment
  size_t written =
      file_.tellp() - static_cast<std::streamoff>(header_.names_offset);
  size_t aligned = (written + 7) & ~7;

  while (written < aligned) {
    char zero = 0;
    file_.write(&zero, 1);
    written++;
  }

  return file_.good();
}

bool LodWriter::write_bucket(double time_min, double time_max,
                             const std::vector<double> &signal_min,
                             const std::vector<double> &signal_max) {
  if (signal_min.size() != header_.column_count ||
      signal_max.size() != header_.column_count) {
    std::cerr << "[LOD] Signal count mismatch" << std::endl;
    return false;
  }

  return write_bucket(time_min, time_max, signal_min.data(), signal_max.data(),
                      header_.column_count);
}

bool LodWriter::write_bucket(double time_min, double time_max,
                             const double *signal_min, const double *signal_max,
                             uint32_t count) {
  if (!is_open()) {
    std::cerr << "[LOD] File not open" << std::endl;
    return false;
  }

  if (count != header_.column_count) {
    std::cerr << "[LOD] Column count mismatch" << std::endl;
    return false;
  }

  // Build bucket data
  // Layout: [time_min, time_max, sig0_min, sig0_max, sig1_min, sig1_max, ...]
  write_buffer_[0] = time_min;
  write_buffer_[1] = time_max;

  for (uint32_t i = 0; i < count; ++i) {
    write_buffer_[2 + i * 2] = signal_min[i];
    write_buffer_[2 + i * 2 + 1] = signal_max[i];
  }

  // Write to file
  file_.write(reinterpret_cast<const char *>(write_buffer_.data()),
              doubles_per_bucket_ * sizeof(double));

  if (!file_.good()) {
    std::cerr << "[LOD] Write failed at bucket " << bucket_count_ << std::endl;
    return false;
  }

  bucket_count_++;
  return true;
}

bool LodWriter::finalize() {
  if (!is_open()) {
    return false;
  }

  // Update header with final bucket count
  header_.bucket_count = bucket_count_;

  // TODO: Calculate checksums
  header_.header_crc32 = 0;
  header_.data_crc32 = 0;

  // Write updated header
  if (!write_header()) {
    std::cerr << "[LOD] Failed to update header" << std::endl;
    file_.close();
    return false;
  }

  file_.close();

  std::cout << "[LOD] Finalized: " << filepath_ << " (" << bucket_count_
            << " buckets written)" << std::endl;

  return true;
}

} // namespace lod
} // namespace timegraph
