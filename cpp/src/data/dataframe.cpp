#include "timegraph/data/dataframe.hpp"
#include "timegraph/data/csv_loader.hpp"
#include <stdexcept>

namespace timegraph {

// ===== Column Access =====

const double* DataFrame::get_column_ptr_f64(const std::string& name) const {
    auto it = columns_.find(name);
    if (it == columns_.end()) {
        throw std::runtime_error("Column not found: " + name);
    }
    
    auto* typed_col = dynamic_cast<Float64Column*>(it->second.get());
    if (!typed_col) {
        throw std::runtime_error("Column is not Float64: " + name);
    }
    
    return typed_col->data_ptr();
}

ColumnType DataFrame::column_type(const std::string& name) const {
    auto it = columns_.find(name);
    if (it == columns_.end()) {
        throw std::runtime_error("Column not found: " + name);
    }
    return it->second->type();
}

// ===== Internal Methods =====

void DataFrame::add_column(std::string name, std::shared_ptr<IColumn> column) {
    // Validate row count consistency
    if (column_count() > 0 && column->size() != row_count_) {
        throw std::runtime_error("Column size mismatch");
    }
    
    // Add column
    columns_[name] = std::move(column);
    column_names_.push_back(name);
    
    // Update row count
    if (column_count() == 1) {
        row_count_ = columns_[name]->size();
    }
}

// ===== Factory Methods =====

DataFrame DataFrame::load_csv(const std::string& path, const CsvOptions& opts) {
    return CsvLoader::load(path, opts);
}

DataFrame DataFrame::load_excel(const std::string& path, const ExcelOptions& opts) {
    // TODO: Implement Excel loader in future task
    throw std::runtime_error("Excel loader not yet implemented");
}

} // namespace timegraph

