/**
 * @file metadata_builder.hpp
 * @brief Build chunk metadata during data processing
 *
 * Calculates pre-aggregated statistics for chunks using SIMD (AVX2)
 * for maximum performance.
 */

#pragma once

#include "timegraph/data/chunk_metadata.hpp"
#include <algorithm>
#include <cmath>


#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace timegraph {
namespace mpai {

class MetadataBuilder {
public:
  /**
   * Build metadata for a single chunk
   *
   * Uses SIMD (AVX2) when available for 4x speedup
   *
   * @param data Pointer to chunk data
   * @param size Number of elements in chunk
   * @param start_row Global row index of first element
   * @return ChunkMetadata with pre-computed statistics
   */
  static ChunkMetadata build_chunk_metadata(const double *data, size_t size,
                                            uint64_t start_row);

  /**
   * Build metadata for all chunks in a column
   *
   * @param data Pointer to full column data
   * @param total_size Total number of elements
   * @param chunk_size Rows per chunk (default: 4096)
   * @return Vector of chunk metadata
   */
  static std::vector<ChunkMetadata>
  build_column_metadata(const double *data, size_t total_size,
                        size_t chunk_size = 4096);

private:
  /**
   * SIMD-optimized metadata calculation (AVX2)
   */
  static ChunkMetadata build_chunk_metadata_simd(const double *data,
                                                 size_t size,
                                                 uint64_t start_row);

  /**
   * Scalar fallback for non-AVX2 systems
   */
  static ChunkMetadata build_chunk_metadata_scalar(const double *data,
                                                   size_t size,
                                                   uint64_t start_row);

  /**
   * Horizontal sum of AVX2 vector
   */
  static double horizontal_sum(__m256d vec);
};

} // namespace mpai
} // namespace timegraph
