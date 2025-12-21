/**
 * Arrow Utilities - Helper functions for MPAI+Arrow hybrid
 * 
 * Zero-copy integration between MPAI chunks and Arrow compute
 */

#pragma once

#include <vector>
#include <memory>
#include <span>

#ifdef HAVE_ARROW
#include <arrow/api.h>
#include <arrow/compute/api.h>
#endif

namespace timegraph {
namespace arrow_utils {

#ifdef HAVE_ARROW

/**
 * Wrap C++ vector as Arrow array (zero-copy!)
 * 
 * CRITICAL: The input vector MUST stay alive while Arrow array is used!
 * 
 * @param data Input vector (will be referenced, not copied)
 * @return Arrow array that references the same memory
 */
inline std::shared_ptr<arrow::DoubleArray> wrap_vector_as_arrow(
    const std::vector<double>& data
) {
    // Create Arrow buffer that wraps existing memory (zero-copy!)
    auto buffer = arrow::Buffer::Wrap(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size() * sizeof(double)
    );
    
    // Create Arrow array from buffer
    auto array_data = arrow::ArrayData::Make(
        arrow::float64(),           // Type: double
        data.size(),                // Length
        {nullptr, buffer},          // Buffers: [null_bitmap, data]
        0                           // null_count
    );
    
    return std::make_shared<arrow::DoubleArray>(array_data);
}

/**
 * Wrap std::span as Arrow array (zero-copy!)
 * 
 * @param data Span (already a view, no ownership)
 * @return Arrow array
 */
inline std::shared_ptr<arrow::DoubleArray> wrap_span_as_arrow(
    std::span<const double> data
) {
    auto buffer = arrow::Buffer::Wrap(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size() * sizeof(double)
    );
    
    auto array_data = arrow::ArrayData::Make(
        arrow::float64(),
        data.size(),
        {nullptr, buffer},
        0
    );
    
    return std::make_shared<arrow::DoubleArray>(array_data);
}

/**
 * Extract data from Arrow array to vector (copy required for ownership)
 * 
 * @param array Arrow array
 * @return Vector with copied data
 */
inline std::vector<double> arrow_to_vector(
    const std::shared_ptr<arrow::DoubleArray>& array
) {
    std::vector<double> result(array->length());
    
    const double* raw_values = array->raw_values();
    std::copy(raw_values, raw_values + array->length(), result.begin());
    
    return result;
}

/**
 * Check if Arrow is available at runtime
 */
inline bool is_arrow_available() {
    return true;
}

#else  // HAVE_ARROW not defined

// Dummy implementations when Arrow is not available

struct DoubleArray {
    size_t length() const { return 0; }
    const double* raw_values() const { return nullptr; }
};

inline std::shared_ptr<DoubleArray> wrap_vector_as_arrow(const std::vector<double>&) {
    return nullptr;
}

inline std::shared_ptr<DoubleArray> wrap_span_as_arrow(std::span<const double>) {
    return nullptr;
}

inline std::vector<double> arrow_to_vector(const std::shared_ptr<DoubleArray>&) {
    return {};
}

inline bool is_arrow_available() {
    return false;
}

#endif  // HAVE_ARROW

}  // namespace arrow_utils
}  // namespace timegraph
