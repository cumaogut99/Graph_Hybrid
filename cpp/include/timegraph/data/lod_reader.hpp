/**
 * LOD Reader - Memory-Mapped LOD File Reader
 *
 * Zero-copy access to pre-computed LOD data for instant zoom/pan.
 * Uses memory-mapping for minimal RAM usage with large files.
 */

#pragma once

#include "lod_format.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace timegraph {
namespace lod {

class LodReader {
public:
  LodReader() = default;
  ~LodReader();

  // Disable copy
  LodReader(const LodReader &) = delete;
  LodReader &operator=(const LodReader &) = delete;

  // Enable move
  LodReader(LodReader &&other) noexcept;
  LodReader &operator=(LodReader &&other) noexcept;

  /**
   * Open LOD file for reading.
   *
   * @param filepath Path to .tlod file
   * @return true if successful
   */
  bool open(const std::string &filepath);

  /**
   * Close the file and release resources.
   */
  void close();

  /**
   * Check if file is open.
   */
  bool is_open() const { return mapped_data_ != nullptr; }

  // ==================== Metadata Access ====================

  const LodHeader &get_header() const { return header_; }
  const LodMetadata &get_metadata() const { return metadata_; }

  uint32_t get_bucket_size() const { return header_.bucket_size; }
  uint64_t get_bucket_count() const { return header_.bucket_count; }
  uint32_t get_column_count() const { return header_.column_count; }
  uint64_t get_total_row_count() const { return header_.total_row_count; }

  const std::vector<std::string> &get_column_names() const {
    return metadata_.column_names;
  }

  /**
   * Get column index by name.
   * @return column index or -1 if not found
   */
  int get_column_index(const std::string &name) const;

  // ==================== Data Access (Zero-Copy) ====================

  /**
   * Get time range for a bucket range (spike-safe: returns min of mins, max of
   * maxes).
   *
   * @param start_bucket Starting bucket index
   * @param count Number of buckets
   * @return pair of (time_min, time_max) for the entire range
   */
  std::pair<double, double> get_time_range(uint64_t start_bucket,
                                           uint64_t count) const;

  /**
   * Get min/max values for a signal in a bucket range.
   * Spike-Safe: Returns the absolute min and max across all buckets.
   *
   * @param column_index Signal column index
   * @param start_bucket Starting bucket index
   * @param count Number of buckets
   * @return pair of (min, max) values
   */
  std::pair<double, double> get_signal_range(uint32_t column_index,
                                             uint64_t start_bucket,
                                             uint64_t count) const;

  /**
   * Get downsampled data for rendering (spike-safe).
   * Returns 2 points per bucket (min, max) to preserve all spikes.
   *
   * @param column_index Signal column index
   * @param start_bucket Starting bucket index
   * @param count Number of buckets
   * @param out_time Output time values (2 per bucket: time_min, time_max)
   * @param out_values Output signal values (2 per bucket: value at min time,
   * value at max)
   */
  void get_render_data(uint32_t column_index, uint64_t start_bucket,
                       uint64_t count, std::vector<double> &out_time,
                       std::vector<double> &out_values) const;

  /**
   * Get render data by column name.
   */
  void get_render_data(const std::string &column_name, uint64_t start_bucket,
                       uint64_t count, std::vector<double> &out_time,
                       std::vector<double> &out_values) const;

  /**
   * Find bucket range for a time range.
   *
   * @param time_start Start time
   * @param time_end End time
   * @return pair of (start_bucket, bucket_count)
   */
  std::pair<uint64_t, uint64_t> find_bucket_range(double time_start,
                                                  double time_end) const;

private:
  // File mapping
  std::string filepath_;

#ifdef _WIN32
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif

  void *mapped_data_ = nullptr;
  size_t file_size_ = 0;

  // Parsed metadata
  LodHeader header_;
  LodMetadata metadata_;

  // Column name to index mapping
  std::unordered_map<std::string, uint32_t> column_index_map_;

  // Pointer to bucket data (within mapped memory)
  const double *bucket_data_ = nullptr;
  size_t doubles_per_bucket_ = 0;

  // Helper methods
  bool parse_header();
  bool parse_column_names();

  /**
   * Get pointer to specific bucket data.
   */
  const double *get_bucket_ptr(uint64_t bucket_index) const {
    if (!bucket_data_ || bucket_index >= header_.bucket_count) {
      return nullptr;
    }
    return bucket_data_ + (bucket_index * doubles_per_bucket_);
  }
};

} // namespace lod
} // namespace timegraph
