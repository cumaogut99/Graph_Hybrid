#include "timegraph/processing/filter_engine.hpp"
<<<<<<< HEAD
#include "timegraph/processing/arrow_utils.hpp"
#include "timegraph/processing/simd_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream> // DEBUG: for std::cout
#include <map>
#include <memory>
#include <stdexcept>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif
=======
#include "timegraph/processing/simd_utils.hpp"
#include "timegraph/processing/arrow_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <memory>
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

#ifdef HAVE_ARROW
#include <arrow/api.h>
#include <arrow/compute/api.h>
#endif

namespace timegraph {

<<<<<<< HEAD
bool FilterEngine::check_condition(double value, const FilterCondition &cond) {
  // Handle NaN values
  if (std::isnan(value)) {
    return false;
  }

  switch (cond.type) {
  case FilterType::RANGE:
    return value >= cond.min_value && value <= cond.max_value;

  case FilterType::GREATER:
    // ✅ FIX: Python sends (threshold - epsilon) for >= operator
    // So we need >= comparison, not strict >
    return value >= cond.threshold;

  case FilterType::LESS:
    // ✅ FIX: Python sends (threshold + epsilon) for <= operator
    // So we need <= comparison, not strict <
    return value <= cond.threshold;

  case FilterType::EQUAL:
    return std::abs(value - cond.threshold) < 1e-10;

  case FilterType::NOT_EQUAL:
    return std::abs(value - cond.threshold) >= 1e-10;

  default:
    return false;
  }
}

std::vector<bool>
FilterEngine::apply_single_condition(const DataFrame &df,
                                     const FilterCondition &cond) {
  // Get column data
  if (!df.has_column(cond.column_name)) {
    throw std::runtime_error("Column not found: " + cond.column_name);
  }

  const double *data = df.get_column_ptr_f64(cond.column_name);
  size_t length = df.row_count();

  // Create boolean mask
  std::vector<bool> mask(length);

  // Use SIMD if available and data is large enough
  const size_t simd_threshold = 1000; // Use SIMD for 1000+ elements

  if (length >= simd_threshold && simd::has_avx2()) {
    // Allocate temporary uint8_t mask for SIMD
    std::unique_ptr<uint8_t[]> simd_mask(new uint8_t[length]);

    // Apply filter using SIMD
    switch (cond.type) {
    case FilterType::RANGE:
#ifdef TIMEGRAPH_HAS_AVX2
      simd::filter::apply_range_avx2(data, length, cond.min_value,
                                     cond.max_value, simd_mask.get());
#else
      simd::scalar::apply_range_scalar(data, length, cond.min_value,
                                       cond.max_value, simd_mask.get());
#endif
      break;

    case FilterType::GREATER:
#ifdef TIMEGRAPH_HAS_AVX2
      simd::filter::apply_greater_avx2(data, length, cond.threshold,
                                       simd_mask.get());
#else
      simd::scalar::apply_greater_scalar(data, length, cond.threshold,
                                         simd_mask.get());
#endif
      break;

    case FilterType::LESS:
#ifdef TIMEGRAPH_HAS_AVX2
      simd::filter::apply_less_avx2(data, length, cond.threshold,
                                    simd_mask.get());
#else
      simd::scalar::apply_less_scalar(data, length, cond.threshold,
                                      simd_mask.get());
#endif
      break;

    default:
      // Fall back to scalar for EQUAL/NOT_EQUAL
      for (size_t i = 0; i < length; ++i) {
        mask[i] = check_condition(data[i], cond);
      }
      return mask;
    }

    // Convert uint8_t mask to bool mask
    for (size_t i = 0; i < length; ++i) {
      mask[i] = (simd_mask[i] != 0);
    }
  } else {
    // Scalar fallback for small data or no SIMD
    for (size_t i = 0; i < length; ++i) {
      mask[i] = check_condition(data[i], cond);
    }
  }

  return mask;
}

std::vector<bool> FilterEngine::combine_masks(const std::vector<bool> &mask1,
                                              const std::vector<bool> &mask2,
                                              FilterOperator op) {
  if (mask1.size() != mask2.size()) {
    throw std::runtime_error("Mask size mismatch");
  }

  size_t length = mask1.size();
  std::vector<bool> result(length);

  // Use SIMD for large masks
  const size_t simd_threshold = 1000;

  if (length >= simd_threshold && simd::has_avx2()) {
    // Convert bool masks to uint8_t for SIMD
    std::unique_ptr<uint8_t[]> m1(new uint8_t[length]);
    std::unique_ptr<uint8_t[]> m2(new uint8_t[length]);
    std::unique_ptr<uint8_t[]> res(new uint8_t[length]);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(length); ++i) {
      m1[i] = mask1[i] ? 1 : 0;
      m2[i] = mask2[i] ? 1 : 0;
    }

    // Combine using SIMD
#ifdef TIMEGRAPH_HAS_AVX2
    if (op == FilterOperator::AND) {
      simd::filter::combine_and_avx2(m1.get(), m2.get(), length, res.get());
    } else {
      simd::filter::combine_or_avx2(m1.get(), m2.get(), length, res.get());
    }
#endif

    // Convert back to bool
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(length); ++i) {
      result[i] = (res[i] != 0);
    }
  } else {
    // Scalar fallback with OpenMP parallelization
    switch (op) {
    case FilterOperator::AND:
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) if (length > 10000)
#endif
      for (int64_t i = 0; i < static_cast<int64_t>(length); ++i) {
        result[i] = mask1[i] && mask2[i];
      }
      break;

    case FilterOperator::OR:
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) if (length > 10000)
#endif
      for (int64_t i = 0; i < static_cast<int64_t>(length); ++i) {
        result[i] = mask1[i] || mask2[i];
      }
      break;
    }
  }

  return result;
}

std::vector<TimeSegment>
FilterEngine::mask_to_segments(const std::vector<bool> &mask,
                               const double *time_data, size_t length) {
  std::vector<TimeSegment> segments;

  if (length == 0) {
    return segments;
  }

  // Find continuous regions where mask is true
  bool in_segment = false;
  size_t segment_start = 0;

  for (size_t i = 0; i < length; ++i) {
    if (mask[i] && !in_segment) {
      // Start new segment
      segment_start = i;
      in_segment = true;
    } else if (!mask[i] && in_segment) {
      // End current segment
      segments.emplace_back(time_data[segment_start], time_data[i - 1],
                            segment_start, i - 1);
      in_segment = false;
    }
  }

  // Close last segment if still open
  if (in_segment) {
    segments.emplace_back(time_data[segment_start], time_data[length - 1],
                          segment_start, length - 1);
  }

  return segments;
}

std::vector<bool>
FilterEngine::calculate_mask(const DataFrame &df,
                             const std::vector<FilterCondition> &conditions) {
  if (conditions.empty()) {
    // No conditions = all pass
    return std::vector<bool>(df.row_count(), true);
  }

  // Start with first condition
  std::vector<bool> combined_mask = apply_single_condition(df, conditions[0]);

  // Apply remaining conditions
  for (size_t i = 1; i < conditions.size(); ++i) {
    auto condition_mask = apply_single_condition(df, conditions[i]);
    combined_mask =
        combine_masks(combined_mask, condition_mask, conditions[i].op);
  }

  return combined_mask;
}

std::vector<TimeSegment> FilterEngine::calculate_segments(
    const DataFrame &df, const std::string &time_column,
    const std::vector<FilterCondition> &conditions) {
  // Check time column exists
  if (!df.has_column(time_column)) {
    throw std::runtime_error("Time column not found: " + time_column);
  }

  // Get time data
  const double *time_data = df.get_column_ptr_f64(time_column);
  size_t length = df.row_count();

  // Calculate boolean mask
  auto mask = calculate_mask(df, conditions);

  // Convert mask to segments
  return mask_to_segments(mask, time_data, length);
}

DataFrame
FilterEngine::apply_filter(const DataFrame &df,
                           const std::vector<FilterCondition> &conditions) {
  // Calculate mask
  auto mask = calculate_mask(df, conditions);

  // TODO: Implement DataFrame row filtering
  // For now, throw not implemented
  throw std::runtime_error("DataFrame filtering not yet implemented");
=======
bool FilterEngine::check_condition(double value, const FilterCondition& cond) {
    // Handle NaN values
    if (std::isnan(value)) {
        return false;
    }
    
    switch (cond.type) {
        case FilterType::RANGE:
            return value >= cond.min_value && value <= cond.max_value;
            
        case FilterType::GREATER:
            return value > cond.threshold;
            
        case FilterType::LESS:
            return value < cond.threshold;
            
        case FilterType::EQUAL:
            return std::abs(value - cond.threshold) < 1e-10;
            
        case FilterType::NOT_EQUAL:
            return std::abs(value - cond.threshold) >= 1e-10;
            
        default:
            return false;
    }
}

std::vector<bool> FilterEngine::apply_single_condition(
    const DataFrame& df,
    const FilterCondition& cond
) {
    // Get column data
    if (!df.has_column(cond.column_name)) {
        throw std::runtime_error("Column not found: " + cond.column_name);
    }
    
    const double* data = df.get_column_ptr_f64(cond.column_name);
    size_t length = df.row_count();
    
    // Create boolean mask
    std::vector<bool> mask(length);
    
    // Use SIMD if available and data is large enough
    const size_t simd_threshold = 1000; // Use SIMD for 1000+ elements
    
    if (length >= simd_threshold && simd::has_avx2()) {
        // Allocate temporary uint8_t mask for SIMD
        std::unique_ptr<uint8_t[]> simd_mask(new uint8_t[length]);
        
        // Apply filter using SIMD
        switch (cond.type) {
            case FilterType::RANGE:
#ifdef TIMEGRAPH_HAS_AVX2
                simd::filter::apply_range_avx2(data, length, cond.min_value, cond.max_value, simd_mask.get());
#else
                simd::scalar::apply_range_scalar(data, length, cond.min_value, cond.max_value, simd_mask.get());
#endif
                break;
                
            case FilterType::GREATER:
#ifdef TIMEGRAPH_HAS_AVX2
                simd::filter::apply_greater_avx2(data, length, cond.threshold, simd_mask.get());
#else
                simd::scalar::apply_greater_scalar(data, length, cond.threshold, simd_mask.get());
#endif
                break;
                
            case FilterType::LESS:
#ifdef TIMEGRAPH_HAS_AVX2
                simd::filter::apply_less_avx2(data, length, cond.threshold, simd_mask.get());
#else
                simd::scalar::apply_less_scalar(data, length, cond.threshold, simd_mask.get());
#endif
                break;
                
            default:
                // Fall back to scalar for EQUAL/NOT_EQUAL
                for (size_t i = 0; i < length; ++i) {
                    mask[i] = check_condition(data[i], cond);
                }
                return mask;
        }
        
        // Convert uint8_t mask to bool mask
        for (size_t i = 0; i < length; ++i) {
            mask[i] = (simd_mask[i] != 0);
        }
    } else {
        // Scalar fallback for small data or no SIMD
        for (size_t i = 0; i < length; ++i) {
            mask[i] = check_condition(data[i], cond);
        }
    }
    
    return mask;
}

std::vector<bool> FilterEngine::combine_masks(
    const std::vector<bool>& mask1,
    const std::vector<bool>& mask2,
    FilterOperator op
) {
    if (mask1.size() != mask2.size()) {
        throw std::runtime_error("Mask size mismatch");
    }
    
    size_t length = mask1.size();
    std::vector<bool> result(length);
    
    // Use SIMD for large masks
    const size_t simd_threshold = 1000;
    
    if (length >= simd_threshold && simd::has_avx2()) {
        // Convert bool masks to uint8_t for SIMD
        std::unique_ptr<uint8_t[]> m1(new uint8_t[length]);
        std::unique_ptr<uint8_t[]> m2(new uint8_t[length]);
        std::unique_ptr<uint8_t[]> res(new uint8_t[length]);
        
        for (size_t i = 0; i < length; ++i) {
            m1[i] = mask1[i] ? 1 : 0;
            m2[i] = mask2[i] ? 1 : 0;
        }
        
        // Combine using SIMD
#ifdef TIMEGRAPH_HAS_AVX2
        if (op == FilterOperator::AND) {
            simd::filter::combine_and_avx2(m1.get(), m2.get(), length, res.get());
        } else {
            simd::filter::combine_or_avx2(m1.get(), m2.get(), length, res.get());
        }
#endif
        
        // Convert back to bool
        for (size_t i = 0; i < length; ++i) {
            result[i] = (res[i] != 0);
        }
    } else {
        // Scalar fallback
        switch (op) {
            case FilterOperator::AND:
                for (size_t i = 0; i < length; ++i) {
                    result[i] = mask1[i] && mask2[i];
                }
                break;
                
            case FilterOperator::OR:
                for (size_t i = 0; i < length; ++i) {
                    result[i] = mask1[i] || mask2[i];
                }
                break;
        }
    }
    
    return result;
}

std::vector<TimeSegment> FilterEngine::mask_to_segments(
    const std::vector<bool>& mask,
    const double* time_data,
    size_t length
) {
    std::vector<TimeSegment> segments;
    
    if (length == 0) {
        return segments;
    }
    
    // Find continuous regions where mask is true
    bool in_segment = false;
    size_t segment_start = 0;
    
    for (size_t i = 0; i < length; ++i) {
        if (mask[i] && !in_segment) {
            // Start new segment
            segment_start = i;
            in_segment = true;
        } else if (!mask[i] && in_segment) {
            // End current segment
            segments.emplace_back(
                time_data[segment_start],
                time_data[i - 1],
                segment_start,
                i - 1
            );
            in_segment = false;
        }
    }
    
    // Close last segment if still open
    if (in_segment) {
        segments.emplace_back(
            time_data[segment_start],
            time_data[length - 1],
            segment_start,
            length - 1
        );
    }
    
    return segments;
}

std::vector<bool> FilterEngine::calculate_mask(
    const DataFrame& df,
    const std::vector<FilterCondition>& conditions
) {
    if (conditions.empty()) {
        // No conditions = all pass
        return std::vector<bool>(df.row_count(), true);
    }
    
    // Start with first condition
    std::vector<bool> combined_mask = apply_single_condition(df, conditions[0]);
    
    // Apply remaining conditions
    for (size_t i = 1; i < conditions.size(); ++i) {
        auto condition_mask = apply_single_condition(df, conditions[i]);
        combined_mask = combine_masks(combined_mask, condition_mask, conditions[i].op);
    }
    
    return combined_mask;
}

std::vector<TimeSegment> FilterEngine::calculate_segments(
    const DataFrame& df,
    const std::string& time_column,
    const std::vector<FilterCondition>& conditions
) {
    // Check time column exists
    if (!df.has_column(time_column)) {
        throw std::runtime_error("Time column not found: " + time_column);
    }
    
    // Get time data
    const double* time_data = df.get_column_ptr_f64(time_column);
    size_t length = df.row_count();
    
    // Calculate boolean mask
    auto mask = calculate_mask(df, conditions);
    
    // Convert mask to segments
    return mask_to_segments(mask, time_data, length);
}

DataFrame FilterEngine::apply_filter(
    const DataFrame& df,
    const std::vector<FilterCondition>& conditions
) {
    // Calculate mask
    auto mask = calculate_mask(df, conditions);
    
    // TODO: Implement DataFrame row filtering
    // For now, throw not implemented
    throw std::runtime_error("DataFrame filtering not yet implemented");
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
}

// ========== MPAI STREAMING FILTERING IMPLEMENTATION ==========

std::vector<TimeSegment> FilterEngine::calculate_streaming(
<<<<<<< HEAD
    mpai::MpaiReader &reader, const std::string &time_column,
    const std::vector<FilterCondition> &conditions, uint64_t start_row,
    uint64_t row_count) {
  std::vector<TimeSegment> segments;

  // Get total row count
  const uint64_t total_rows = reader.get_row_count();
  if (total_rows == 0 || conditions.empty()) {
    return segments;
  }

  // DEBUG: Log filter conditions
  std::cout << "[CPP FILTER DEBUG] ====== calculate_streaming START ======"
            << std::endl;
  std::cout << "[CPP FILTER DEBUG] Total rows: " << total_rows << std::endl;
  std::cout << "[CPP FILTER DEBUG] Time column: " << time_column << std::endl;
  std::cout << "[CPP FILTER DEBUG] Conditions count: " << conditions.size()
            << std::endl;

  for (size_t c = 0; c < conditions.size(); ++c) {
    const auto &cond = conditions[c];
    std::cout << "[CPP FILTER DEBUG] Condition " << c << ":" << std::endl;
    std::cout << "  column_name: " << cond.column_name << std::endl;
    std::cout << "  type: " << static_cast<int>(cond.type)
              << " (RANGE=0, GREATER=1, LESS=2, EQUAL=3, NOT_EQUAL=4)"
              << std::endl;
    std::cout << "  threshold: " << cond.threshold << std::endl;
    std::cout << "  min_value: " << cond.min_value << std::endl;
    std::cout << "  max_value: " << cond.max_value << std::endl;
    std::cout << "  op: " << static_cast<int>(cond.op) << " (AND=0, OR=1)"
              << std::endl;
  }

  // Validate start_row
  if (start_row >= total_rows) {
    return segments;
  }

  // Determine actual row count
  uint64_t actual_row_count = row_count;
  if (row_count == 0 || start_row + row_count > total_rows) {
    actual_row_count = total_rows - start_row;
  }

  // Chunk settings
  //
  // NOT: Burada büyük chunk'lar (ör. 100k–1M satır) kullanmak overall
  // performans için çok daha iyi: daha az loop, daha az load_column_slice
  // çağrısı. 1M satır * 8 byte * ~10 kolon ≈ 80 MB => 300–500 MB toplam RAM
  // bütçesi içinde kalıyoruz.
  //
  // MPAI formatındaki DEFAULT_CHUNK_SIZE (1M) ile uyumlu olacak şekilde
  // 1_000_000 satırlık streaming chunk boyutu seçiyoruz.
  const uint64_t chunk_size = 1000000;
  uint64_t rows_processed = 0;

  // State for building segments across chunks
  bool in_segment = false;
  double segment_start_time = 0.0;
  uint64_t segment_start_index = 0;
  double last_time = 0.0;
  bool last_time_valid = false;

  // DEBUG: Sample statistics
  uint64_t total_pass_count = 0;
  uint64_t total_fail_count = 0;
  int sample_log_count = 0;

  while (rows_processed < actual_row_count) {
    const uint64_t current_chunk_size =
        std::min(chunk_size, actual_row_count - rows_processed);

    const uint64_t chunk_start_row = start_row + rows_processed;

    // Load time data for this chunk
    std::vector<double> time_data;
    try {
      time_data =
          reader.load_column_slice(time_column, chunk_start_row,
                                   static_cast<uint64_t>(current_chunk_size));
    } catch (const std::exception &e) {
      // If time column cannot be loaded, abort filtering
      throw std::runtime_error(std::string("Failed to load time column '") +
                               time_column + "': " + e.what());
    }

    if (time_data.empty()) {
      break;
    }

    const size_t local_size = time_data.size();

    // Initialize mask for this chunk (start with all true)
    std::vector<bool> chunk_mask(local_size, true);

    // Apply each condition
    for (const auto &cond : conditions) {
      // Load column data for this condition
      std::vector<double> col_data;
      try {
        col_data = reader.load_column_slice(cond.column_name, chunk_start_row,
                                            static_cast<uint64_t>(local_size));
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to load column '") +
                                 cond.column_name + "': " + e.what());
      }

      if (col_data.size() != local_size) {
        throw std::runtime_error(
            "Column size mismatch during streaming filter calculation");
      }

      // DEBUG: Log first few values being checked (only on first chunk)
      if (rows_processed == 0 && sample_log_count < 5) {
        std::cout << "[CPP FILTER DEBUG] First 10 values of column '"
                  << cond.column_name << "':" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), col_data.size()); ++i) {
          bool pass = check_condition(col_data[i], cond);
          std::cout << "  [" << i << "] value=" << col_data[i] << " -> "
                    << (pass ? "PASS" : "FAIL") << std::endl;
        }
        sample_log_count++;
      }

      // Build mask for this condition and combine
      for (size_t i = 0; i < local_size; ++i) {
        const bool pass = check_condition(col_data[i], cond);

        if (cond.op == FilterOperator::AND) {
          chunk_mask[i] = chunk_mask[i] && pass;
        } else { // OR
          chunk_mask[i] = chunk_mask[i] || pass;
        }
      }
    }

    // Convert mask for this chunk to segments, preserving continuity
    for (size_t i = 0; i < local_size; ++i) {
      const uint64_t global_index = chunk_start_row + i;
      const double t = time_data[i];
      const bool pass = chunk_mask[i];

      // DEBUG: Count pass/fail
      if (pass)
        total_pass_count++;
      else
        total_fail_count++;

      if (pass && !in_segment) {
        // Start new segment
        in_segment = true;
        segment_start_time = t;
        segment_start_index = global_index;
      } else if (!pass && in_segment) {
        // End current segment at previous valid time/index
        const double end_time = last_time_valid ? last_time : t;
        const uint64_t end_index =
            global_index > 0 ? global_index - 1 : global_index;
        segments.emplace_back(segment_start_time, end_time,
                              static_cast<size_t>(segment_start_index),
                              static_cast<size_t>(end_index));
        in_segment = false;
      }

      last_time = t;
      last_time_valid = true;
    }

    rows_processed += current_chunk_size;
  }

  // Close final segment if still open
  if (in_segment && last_time_valid) {
    const uint64_t end_index = start_row + actual_row_count - 1;
    segments.emplace_back(segment_start_time, last_time,
                          static_cast<size_t>(segment_start_index),
                          static_cast<size_t>(end_index));
  }

  // DEBUG: Final summary
  std::cout << "[CPP FILTER DEBUG] ====== calculate_streaming END ======"
            << std::endl;
  std::cout << "[CPP FILTER DEBUG] Total PASS: " << total_pass_count
            << std::endl;
  std::cout << "[CPP FILTER DEBUG] Total FAIL: " << total_fail_count
            << std::endl;
  std::cout << "[CPP FILTER DEBUG] Segments found: " << segments.size()
            << std::endl;

  return segments;
=======
    mpai::MpaiReader& reader,
    const std::string& time_column,
    const std::vector<FilterCondition>& conditions,
    uint64_t start_row,
    uint64_t row_count
) {
    std::vector<TimeSegment> segments;
    
    // Get total row count
    const uint64_t total_rows = reader.get_row_count();
    if (total_rows == 0 || conditions.empty()) {
        return segments;
    }
    
    // Validate start_row
    if (start_row >= total_rows) {
        return segments;
    }
    
    // Determine actual row count
    uint64_t actual_row_count = row_count;
    if (row_count == 0 || start_row + row_count > total_rows) {
        actual_row_count = total_rows - start_row;
    }
    
    // Chunk settings
    //
    // NOT: Burada büyük chunk'lar (ör. 100k–1M satır) kullanmak overall performans
    // için çok daha iyi: daha az loop, daha az load_column_slice çağrısı.
    // 1M satır * 8 byte * ~10 kolon ≈ 80 MB => 300–500 MB toplam RAM bütçesi
    // içinde kalıyoruz.
    //
    // MPAI formatındaki DEFAULT_CHUNK_SIZE (1M) ile uyumlu olacak şekilde
    // 1_000_000 satırlık streaming chunk boyutu seçiyoruz.
    const uint64_t chunk_size = 1000000;
    uint64_t rows_processed = 0;
    
    // State for building segments across chunks
    bool in_segment = false;
    double segment_start_time = 0.0;
    uint64_t segment_start_index = 0;
    double last_time = 0.0;
    bool last_time_valid = false;
    
    while (rows_processed < actual_row_count) {
        const uint64_t current_chunk_size = std::min(
            chunk_size,
            actual_row_count - rows_processed
        );
        
        const uint64_t chunk_start_row = start_row + rows_processed;
        
        // Load time data for this chunk
        std::vector<double> time_data;
        try {
            time_data = reader.load_column_slice(
                time_column,
                chunk_start_row,
                static_cast<uint64_t>(current_chunk_size)
            );
        } catch (const std::exception& e) {
            // If time column cannot be loaded, abort filtering
            throw std::runtime_error(
                std::string("Failed to load time column '") +
                time_column + "': " + e.what()
            );
        }
        
        if (time_data.empty()) {
            break;
        }
        
        const size_t local_size = time_data.size();
        
        // Initialize mask for this chunk (start with all true)
        std::vector<bool> chunk_mask(local_size, true);
        
        // Apply each condition
        for (const auto& cond : conditions) {
            // Load column data for this condition
            std::vector<double> col_data;
            try {
                col_data = reader.load_column_slice(
                    cond.column_name,
                    chunk_start_row,
                    static_cast<uint64_t>(local_size)
                );
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    std::string("Failed to load column '") +
                    cond.column_name + "': " + e.what()
                );
            }
            
            if (col_data.size() != local_size) {
                throw std::runtime_error(
                    "Column size mismatch during streaming filter calculation"
                );
            }
            
            // Build mask for this condition and combine
            for (size_t i = 0; i < local_size; ++i) {
                const bool pass = check_condition(col_data[i], cond);
                
                if (cond.op == FilterOperator::AND) {
                    chunk_mask[i] = chunk_mask[i] && pass;
                } else { // OR
                    chunk_mask[i] = chunk_mask[i] || pass;
                }
            }
        }
        
        // Convert mask for this chunk to segments, preserving continuity
        for (size_t i = 0; i < local_size; ++i) {
            const uint64_t global_index = chunk_start_row + i;
            const double t = time_data[i];
            const bool pass = chunk_mask[i];
            
            if (pass && !in_segment) {
                // Start new segment
                in_segment = true;
                segment_start_time = t;
                segment_start_index = global_index;
            } else if (!pass && in_segment) {
                // End current segment at previous valid time/index
                const double end_time = last_time_valid ? last_time : t;
                const uint64_t end_index = global_index > 0 ? global_index - 1 : global_index;
                segments.emplace_back(
                    segment_start_time,
                    end_time,
                    static_cast<size_t>(segment_start_index),
                    static_cast<size_t>(end_index)
                );
                in_segment = false;
            }
            
            last_time = t;
            last_time_valid = true;
        }
        
        rows_processed += current_chunk_size;
    }
    
    // Close final segment if still open
    if (in_segment && last_time_valid) {
        const uint64_t end_index = start_row + actual_row_count - 1;
        segments.emplace_back(
            segment_start_time,
            last_time,
            static_cast<size_t>(segment_start_index),
            static_cast<size_t>(end_index)
        );
    }
    
    return segments;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
}

// ========== ARROW COMPUTE IMPLEMENTATIONS ==========

<<<<<<< HEAD
std::vector<bool>
FilterEngine::calculate_mask_arrow(const std::vector<double> &data,
                                   const FilterCondition &cond) {
#ifdef HAVE_ARROW
  if (!arrow_utils::is_arrow_available() || data.size() < 1000) {
    // Fallback to native SIMD for small data
    std::vector<bool> mask(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      mask[i] = check_condition(data[i], cond);
    }
    return mask;
  }

  try {
    // Zero-copy wrap as Arrow array
    auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);

    // Build Arrow compute expression
    arrow::compute::ExecContext ctx;
    arrow::Datum result_datum;

    switch (cond.type) {
    case FilterType::RANGE: {
      // (value >= min) AND (value <= max)
      auto ge_result = arrow::compute::CallFunction(
          "greater_equal", {arrow_array, arrow::Datum(cond.min_value)}, &ctx);
      auto le_result = arrow::compute::CallFunction(
          "less_equal", {arrow_array, arrow::Datum(cond.max_value)}, &ctx);

      if (ge_result.ok() && le_result.ok()) {
        result_datum =
            arrow::compute::CallFunction(
                "and", {ge_result.ValueOrDie(), le_result.ValueOrDie()}, &ctx)
                .ValueOrDie();
      }
      break;
    }

    case FilterType::GREATER: {
      // value > threshold
      auto result = arrow::compute::CallFunction(
          "greater", {arrow_array, arrow::Datum(cond.threshold)}, &ctx);
      if (result.ok()) {
        result_datum = result.ValueOrDie();
      }
      break;
    }

    case FilterType::LESS: {
      // value < threshold
      auto result = arrow::compute::CallFunction(
          "less", {arrow_array, arrow::Datum(cond.threshold)}, &ctx);
      if (result.ok()) {
        result_datum = result.ValueOrDie();
      }
      break;
    }

    case FilterType::EQUAL: {
      // value == threshold (with tolerance)
      auto result = arrow::compute::CallFunction(
          "equal", {arrow_array, arrow::Datum(cond.threshold)}, &ctx);
      if (result.ok()) {
        result_datum = result.ValueOrDie();
      }
      break;
    }

    case FilterType::NOT_EQUAL: {
      // value != threshold
      auto result = arrow::compute::CallFunction(
          "not_equal", {arrow_array, arrow::Datum(cond.threshold)}, &ctx);
      if (result.ok()) {
        result_datum = result.ValueOrDie();
      }
      break;
    }
    }

    // Convert Arrow boolean array to std::vector<bool>
    auto bool_array = std::static_pointer_cast<arrow::BooleanArray>(
        result_datum.make_array());

    std::vector<bool> mask(bool_array->length());
    for (int64_t i = 0; i < bool_array->length(); ++i) {
      mask[i] = bool_array->Value(i);
    }

    return mask;

  } catch (const std::exception &e) {
    // Arrow compute failed, fallback to native
    std::vector<bool> mask(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      mask[i] = check_condition(data[i], cond);
    }
    return mask;
  }

#else
  // Arrow not available, use native implementation
  std::vector<bool> mask(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    mask[i] = check_condition(data[i], cond);
  }
  return mask;
=======
std::vector<bool> FilterEngine::calculate_mask_arrow(
    const std::vector<double>& data,
    const FilterCondition& cond
) {
#ifdef HAVE_ARROW
    if (!arrow_utils::is_arrow_available() || data.size() < 1000) {
        // Fallback to native SIMD for small data
        std::vector<bool> mask(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            mask[i] = check_condition(data[i], cond);
        }
        return mask;
    }
    
    try {
        // Zero-copy wrap as Arrow array
        auto arrow_array = arrow_utils::wrap_vector_as_arrow(data);
        
        // Build Arrow compute expression
        arrow::compute::ExecContext ctx;
        arrow::Datum result_datum;
        
        switch (cond.type) {
            case FilterType::RANGE: {
                // (value >= min) AND (value <= max)
                auto ge_result = arrow::compute::CallFunction(
                    "greater_equal",
                    {arrow_array, arrow::Datum(cond.min_value)},
                    &ctx
                );
                auto le_result = arrow::compute::CallFunction(
                    "less_equal",
                    {arrow_array, arrow::Datum(cond.max_value)},
                    &ctx
                );
                
                if (ge_result.ok() && le_result.ok()) {
                    result_datum = arrow::compute::CallFunction(
                        "and",
                        {ge_result.ValueOrDie(), le_result.ValueOrDie()},
                        &ctx
                    ).ValueOrDie();
                }
                break;
            }
            
            case FilterType::GREATER: {
                // value > threshold
                auto result = arrow::compute::CallFunction(
                    "greater",
                    {arrow_array, arrow::Datum(cond.threshold)},
                    &ctx
                );
                if (result.ok()) {
                    result_datum = result.ValueOrDie();
                }
                break;
            }
            
            case FilterType::LESS: {
                // value < threshold
                auto result = arrow::compute::CallFunction(
                    "less",
                    {arrow_array, arrow::Datum(cond.threshold)},
                    &ctx
                );
                if (result.ok()) {
                    result_datum = result.ValueOrDie();
                }
                break;
            }
            
            case FilterType::EQUAL: {
                // value == threshold (with tolerance)
                auto result = arrow::compute::CallFunction(
                    "equal",
                    {arrow_array, arrow::Datum(cond.threshold)},
                    &ctx
                );
                if (result.ok()) {
                    result_datum = result.ValueOrDie();
                }
                break;
            }
            
            case FilterType::NOT_EQUAL: {
                // value != threshold
                auto result = arrow::compute::CallFunction(
                    "not_equal",
                    {arrow_array, arrow::Datum(cond.threshold)},
                    &ctx
                );
                if (result.ok()) {
                    result_datum = result.ValueOrDie();
                }
                break;
            }
        }
        
        // Convert Arrow boolean array to std::vector<bool>
        auto bool_array = std::static_pointer_cast<arrow::BooleanArray>(
            result_datum.make_array()
        );
        
        std::vector<bool> mask(bool_array->length());
        for (int64_t i = 0; i < bool_array->length(); ++i) {
            mask[i] = bool_array->Value(i);
        }
        
        return mask;
        
    } catch (const std::exception& e) {
        // Arrow compute failed, fallback to native
        std::vector<bool> mask(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            mask[i] = check_condition(data[i], cond);
        }
        return mask;
    }
    
#else
    // Arrow not available, use native implementation
    std::vector<bool> mask(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        mask[i] = check_condition(data[i], cond);
    }
    return mask;
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
#endif
}

std::vector<bool> FilterEngine::apply_filter_to_chunk_arrow(
<<<<<<< HEAD
    const std::vector<double> &chunk_data,
    const std::vector<FilterCondition> &conditions) {
  if (conditions.empty()) {
    return std::vector<bool>(chunk_data.size(), true);
  }

  // Start with first condition
  std::vector<bool> combined_mask =
      calculate_mask_arrow(chunk_data, conditions[0]);

  // Combine with remaining conditions
  for (size_t i = 1; i < conditions.size(); ++i) {
    std::vector<bool> mask = calculate_mask_arrow(chunk_data, conditions[i]);

    // Combine based on operator
    if (conditions[i].op == FilterOperator::AND) {
      for (size_t j = 0; j < combined_mask.size(); ++j) {
        combined_mask[j] = combined_mask[j] && mask[j];
      }
    } else { // OR
      for (size_t j = 0; j < combined_mask.size(); ++j) {
        combined_mask[j] = combined_mask[j] || mask[j];
      }
    }
  }

  return combined_mask;
}

// ========== NEW: LOAD FILTERED SEGMENT ==========
std::pair<std::vector<double>, std::vector<double>>
FilterEngine::load_filtered_segment(
    mpai::MpaiReader &reader, const std::string &signal_name,
    const std::string &time_column, double time_start, double time_end,
    const std::vector<FilterCondition> &conditions) {
  std::vector<double> result_time;
  std::vector<double> result_signal;

  const uint64_t total_rows = reader.get_row_count();
  if (total_rows == 0) {
    return {result_time, result_signal};
  }

  // Binary search to find start and end row indices for time range
  // Use first and last time values to estimate row range
  std::vector<double> first_time = reader.load_column_slice(time_column, 0, 1);
  std::vector<double> last_time =
      reader.load_column_slice(time_column, total_rows - 1, 1);

  if (first_time.empty() || last_time.empty()) {
    return {result_time, result_signal};
  }

  double t_min = first_time[0];
  double t_max = last_time[0];
  double duration = t_max - t_min;

  if (duration <= 0) {
    return {result_time, result_signal};
  }

  // Linear interpolation to estimate row indices (O(1) instead of O(log N)
  // binary search) This is accurate for uniformly sampled data
  double sample_rate = (total_rows - 1) / duration;

  uint64_t start_row = 0;
  uint64_t end_row = total_rows;

  if (time_start > t_min) {
    start_row = static_cast<uint64_t>((time_start - t_min) * sample_rate);
    start_row = std::min(start_row, total_rows - 1);
  }

  if (time_end < t_max) {
    end_row = static_cast<uint64_t>((time_end - t_min) * sample_rate) + 1;
    end_row = std::min(end_row, total_rows);
  }

  if (start_row >= end_row) {
    return {result_time, result_signal};
  }

  uint64_t row_count = end_row - start_row;

  // Load time and signal data for this range
  std::vector<double> time_data =
      reader.load_column_slice(time_column, start_row, row_count);
  std::vector<double> signal_data =
      reader.load_column_slice(signal_name, start_row, row_count);

  if (time_data.size() != signal_data.size() || time_data.empty()) {
    return {result_time, result_signal};
  }

  // Pre-load all unique columns needed for filter conditions
  std::map<std::string, std::vector<double>> filter_column_data;
  for (const auto &cond : conditions) {
    if (cond.column_name != signal_name && cond.column_name != time_column) {
      // Need to load this column for filtering
      if (filter_column_data.find(cond.column_name) ==
          filter_column_data.end()) {
        std::vector<double> col_data =
            reader.load_column_slice(cond.column_name, start_row, row_count);
        if (col_data.size() == time_data.size()) {
          filter_column_data[cond.column_name] = std::move(col_data);
        }
      }
    }
  }

  // Reserve approximate capacity to avoid reallocations
  result_time.reserve(time_data.size() / 2);
  result_signal.reserve(signal_data.size() / 2);

  // Apply filter conditions and collect matching points
  for (size_t i = 0; i < time_data.size(); ++i) {
    double t = time_data[i];

    // Check time bounds first (fast path)
    if (t < time_start || t > time_end) {
      continue;
    }

    // Check all filter conditions
    bool passes_all = true;

    for (const auto &cond : conditions) {
      double value_to_check;

      if (cond.column_name == signal_name) {
        // Use already loaded signal data
        value_to_check = signal_data[i];
      } else if (cond.column_name == time_column) {
        // Use time data
        value_to_check = t;
      } else {
        // Use pre-loaded filter column data
        auto it = filter_column_data.find(cond.column_name);
        if (it == filter_column_data.end()) {
          // Column not loaded - skip this condition (shouldn't happen)
          continue;
        }
        value_to_check = it->second[i];
      }

      if (!check_condition(value_to_check, cond)) {
        passes_all = false;
        break;
      }
    }

    if (passes_all) {
      result_time.push_back(t);
      result_signal.push_back(signal_data[i]);
    }
  }

  return {result_time, result_signal};
}

#ifdef HAVE_ARROW
// ========== ARROW BRIDGE IMPLEMENTATION ==========
// This function accepts data that was already converted from Arrow in Python.
// The actual Arrow → vector conversion happens in Python bindings for
// simplicity.

std::vector<TimeSegment> FilterEngine::calculate_segments_from_arrow(
    const std::vector<double> &time_data,
    const std::map<std::string, std::vector<double>> &column_data,
    const std::vector<FilterCondition> &conditions) {

  std::vector<TimeSegment> segments;

  if (time_data.empty() || conditions.empty()) {
    return segments;
  }

  const size_t length = time_data.size();

  // Initialize mask to all true
  std::vector<bool> combined_mask(length, true);

  // Apply each condition
  for (const auto &cond : conditions) {
    // Find the column data for this condition
    auto it = column_data.find(cond.column_name);
    if (it == column_data.end()) {
      std::cerr << "[ARROW FILTER] Column not found: " << cond.column_name
                << std::endl;
      continue;
    }

    const std::vector<double> &col_data = it->second;

    if (col_data.size() != length) {
      std::cerr << "[ARROW FILTER] Column size mismatch: " << cond.column_name
                << " has " << col_data.size() << " rows, expected " << length
                << std::endl;
      continue;
    }

    // Calculate mask for this condition using existing Arrow compute if
    // available
    std::vector<bool> condition_mask = calculate_mask_arrow(col_data, cond);

    // Combine with existing mask
    if (cond.op == FilterOperator::AND) {
      for (size_t i = 0; i < length; ++i) {
        combined_mask[i] = combined_mask[i] && condition_mask[i];
      }
    } else { // OR
      for (size_t i = 0; i < length; ++i) {
        combined_mask[i] = combined_mask[i] || condition_mask[i];
      }
    }
  }

  // Convert mask to segments using existing method
  return mask_to_segments(combined_mask, time_data.data(), length);
}
#endif

=======
    const std::vector<double>& chunk_data,
    const std::vector<FilterCondition>& conditions
) {
    if (conditions.empty()) {
        return std::vector<bool>(chunk_data.size(), true);
    }
    
    // Start with first condition
    std::vector<bool> combined_mask = calculate_mask_arrow(chunk_data, conditions[0]);
    
    // Combine with remaining conditions
    for (size_t i = 1; i < conditions.size(); ++i) {
        std::vector<bool> mask = calculate_mask_arrow(chunk_data, conditions[i]);
        
        // Combine based on operator
        if (conditions[i].op == FilterOperator::AND) {
            for (size_t j = 0; j < combined_mask.size(); ++j) {
                combined_mask[j] = combined_mask[j] && mask[j];
            }
        } else {  // OR
            for (size_t j = 0; j < combined_mask.size(); ++j) {
                combined_mask[j] = combined_mask[j] || mask[j];
            }
        }
    }
    
    return combined_mask;
}

>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
} // namespace timegraph
