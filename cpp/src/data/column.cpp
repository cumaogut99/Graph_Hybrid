#include "timegraph/data/column.hpp"

namespace timegraph {

// Explicit template instantiations for common types
// This ensures the compiler generates these versions
template class TypedColumn<double>;
template class TypedColumn<int64_t>;
template class TypedColumn<std::string>;

// Helper function for type to string conversion
std::string column_type_to_string(ColumnType type) {
    switch (type) {
        case ColumnType::FLOAT64: return "Float64";
        case ColumnType::INT64: return "Int64";
        case ColumnType::STRING: return "String";
        case ColumnType::DATETIME: return "DateTime";
        default: return "Unknown";
    }
}

} // namespace timegraph
