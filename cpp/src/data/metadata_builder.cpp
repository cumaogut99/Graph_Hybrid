/**
 * @file metadata_builder.cpp
 * @brief Implementation of chunk metadata builder with SIMD optimization
 */

#include "timegraph/data/metadata_builder.hpp"
#include <cmath>
#include <limits>

namespace timegraph {
namespace mpai {

ChunkMetadata MetadataBuilder::build_chunk_metadata(const double *data,
                                                    size_t size,
                                                    uint64_t start_row) {
  if (size == 0) {
    return ChunkMetadata();
  }

#ifdef __AVX2__
  return build_chunk_metadata_simd(data, size, start_row);
#else
  return build_chunk_metadata_scalar(data, size, start_row);
#endif
}

#ifdef __AVX2__
double MetadataBuilder::horizontal_sum(__m256d vec) {
  // Horizontal add: [a, b, c, d] -> a+b+c+d
  __m128d low = _mm256_castpd256_pd128(vec);
  __m128d high = _mm256_extractf128_pd(vec, 1);
  __m128d sum128 = _mm_add_pd(low, high);
  __m128d sum64 = _mm_hadd_pd(sum128, sum128);
  return _mm_cvtsd_f64(sum64);
}

ChunkMetadata MetadataBuilder::build_chunk_metadata_simd(const double *data,
                                                         size_t size,
                                                         uint64_t start_row) {
  ChunkMetadata meta;
  meta.start_row = start_row;
  meta.row_count = size;

  // SIMD accumulators
  __m256d sum_vec = _mm256_setzero_pd();
  __m256d sum_sq_vec = _mm256_setzero_pd();
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  uint32_t valid_count = 0;

  // Process 4 doubles at a time with AVX2
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    __m256d vals = _mm256_loadu_pd(&data[i]);

    // Check for NaN/Inf (skip invalid values)
    bool all_valid = true;
    for (int j = 0; j < 4; j++) {
      double val = data[i + j];
      if (std::isnan(val) || std::isinf(val)) {
        all_valid = false;
        break;
      }
    }

    if (all_valid) {
      // Accumulate sum and sum of squares
      sum_vec = _mm256_add_pd(sum_vec, vals);
      __m256d sq = _mm256_mul_pd(vals, vals);
      sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);

      // Update min/max (scalar for simplicity)
      for (int j = 0; j < 4; j++) {
        double val = data[i + j];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }

      valid_count += 4;
    } else {
      // Process individually
      for (int j = 0; j < 4; j++) {
        double val = data[i + j];
        if (!std::isnan(val) && !std::isinf(val)) {
          meta.sum += val;
          meta.sum_squares += val * val;
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
          valid_count++;
        }
      }
    }
  }

  // Horizontal sum of SIMD accumulators
  meta.sum += horizontal_sum(sum_vec);
  meta.sum_squares += horizontal_sum(sum_sq_vec);

  // Process remaining elements (scalar)
  for (; i < size; i++) {
    double val = data[i];
    if (!std::isnan(val) && !std::isinf(val)) {
      meta.sum += val;
      meta.sum_squares += val * val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
      valid_count++;
    }
  }

  meta.min_value = min_val;
  meta.max_value = max_val;
  meta.count = valid_count;

  return meta;
}
#endif

ChunkMetadata MetadataBuilder::build_chunk_metadata_scalar(const double *data,
                                                           size_t size,
                                                           uint64_t start_row) {
  ChunkMetadata meta;
  meta.start_row = start_row;
  meta.row_count = size;

  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  uint32_t valid_count = 0;

  for (size_t i = 0; i < size; i++) {
    double val = data[i];

    // Skip NaN and Inf
    if (std::isnan(val) || std::isinf(val)) {
      continue;
    }

    meta.sum += val;
    meta.sum_squares += val * val;
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
    valid_count++;
  }

  meta.min_value = min_val;
  meta.max_value = max_val;
  meta.count = valid_count;

  return meta;
}

std::vector<ChunkMetadata>
MetadataBuilder::build_column_metadata(const double *data, size_t total_size,
                                       size_t chunk_size) {
  std::vector<ChunkMetadata> metadata;

  size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
  metadata.reserve(num_chunks);

  for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    size_t start = chunk_idx * chunk_size;
    size_t size = std::min(chunk_size, total_size - start);

    ChunkMetadata chunk_meta = build_chunk_metadata(data + start, size, start);

    metadata.push_back(chunk_meta);
  }

  return metadata;
}

} // namespace mpai
} // namespace timegraph
