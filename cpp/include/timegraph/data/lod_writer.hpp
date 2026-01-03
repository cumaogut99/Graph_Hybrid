/**
 * LOD Writer - Binary LOD File Writer
 *
 * Generates spike-safe LOD files during data conversion.
 * Supports streaming write for large files.
 */

#pragma once

#include "lod_format.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace timegraph {
namespace lod {

class LodWriter {
public:
  LodWriter() = default;
  ~LodWriter();

  /**
   * Create a new LOD file.
   *
   * @param filepath Output file path
   * @param bucket_size LOD bucket size (100, 10000, or 100000)
   * @param column_names Signal column names (excluding time)
   * @param total_row_count Total rows in original data
   * @return true if successful
   */
  bool create(const std::string &filepath, uint32_t bucket_size,
              const std::vector<std::string> &column_names,
              uint64_t total_row_count);

  /**
   * Write a single bucket.
   *
   * @param time_min Minimum time in bucket
   * @param time_max Maximum time in bucket
   * @param signal_min Min values for each signal (size must match column_count)
   * @param signal_max Max values for each signal (size must match column_count)
   * @return true if successful
   */
  bool write_bucket(double time_min, double time_max,
                    const std::vector<double> &signal_min,
                    const std::vector<double> &signal_max);

  /**
   * Write a single bucket from raw arrays.
   *
   * @param time_min Minimum time in bucket
   * @param time_max Maximum time in bucket
   * @param signal_min Min values array
   * @param signal_max Max values array
   * @param count Number of signals
   * @return true if successful
   */
  bool write_bucket(double time_min, double time_max, const double *signal_min,
                    const double *signal_max, uint32_t count);

  /**
   * Finalize and close the file.
   * Updates header with final bucket count and checksums.
   *
   * @return true if successful
   */
  bool finalize();

  /**
   * Get current bucket count.
   */
  uint64_t get_bucket_count() const { return bucket_count_; }

  /**
   * Check if file is open.
   */
  bool is_open() const { return file_.is_open(); }

private:
  std::ofstream file_;
  std::string filepath_;

  LodHeader header_;
  std::vector<std::string> column_names_;

  uint64_t bucket_count_ = 0;
  uint64_t data_offset_ = 0;

  // Buffer for efficient writes
  std::vector<double> write_buffer_;
  size_t doubles_per_bucket_ = 0;

  // Helper methods
  bool write_header();
  bool write_column_names();
};

} // namespace lod
} // namespace timegraph
