#pragma once

#include "timegraph/data/column.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <span>
#include <stdexcept>

namespace timegraph {

// Forward declarations
struct CsvOptions;
struct ExcelOptions;

/// Main DataFrame class
/// Column-oriented storage for efficient data processing
class DataFrame {
private:
    /// Column storage (name -> column)
    std::unordered_map<std::string, std::shared_ptr<IColumn>> columns_;
    
    /// Row count
    size_t row_count_ = 0;
    
    /// Column names (preserves insertion order)
    std::vector<std::string> column_names_;
    
public:
    /// Default constructor (empty DataFrame)
    DataFrame() = default;
    
    /// No copy (expensive operation)
    DataFrame(const DataFrame&) = delete;
    DataFrame& operator=(const DataFrame&) = delete;
    
    /// Move semantics (efficient)
    DataFrame(DataFrame&&) = default;
    DataFrame& operator=(DataFrame&&) = default;
    
    // ===== FACTORY METHODS =====
    
    /// Load CSV file (multi-threaded, memory-mapped)
    static DataFrame load_csv(const std::string& path, const CsvOptions& opts);
    
    /// Load Excel file
    static DataFrame load_excel(const std::string& path, const ExcelOptions& opts);
    
    // ===== COLUMN ACCESS =====
    
    /// Get column as typed span (zero-copy, const)
    template<typename T>
    std::span<const T> get_column(const std::string& name) const {
        auto it = columns_.find(name);
        if (it == columns_.end()) {
            throw std::runtime_error("Column not found: " + name);
        }
        
        auto* typed_col = dynamic_cast<TypedColumn<T>*>(it->second.get());
        if (!typed_col) {
            throw std::runtime_error("Type mismatch for column: " + name);
        }
        
        return typed_col->view();
    }
    
    /// Get column pointer for Python bindings (Float64 only)
    const double* get_column_ptr_f64(const std::string& name) const;
    
    /// Check if column exists
    bool has_column(const std::string& name) const {
        return columns_.find(name) != columns_.end();
    }
    
    // ===== METADATA =====
    
    /// Get number of rows
    size_t row_count() const { return row_count_; }
    
    /// Get number of columns
    size_t column_count() const { return columns_.size(); }
    
    /// Get column names
    std::vector<std::string> column_names() const { return column_names_; }
    
    /// Get column type
    ColumnType column_type(const std::string& name) const;
    
    // ===== INTERNAL (for builders) =====
    
    /// Add column (internal use)
    void add_column(std::string name, std::shared_ptr<IColumn> column);
    
    /// Set row count (internal use)
    void set_row_count(size_t count) { row_count_ = count; }
};

/// CSV loading options
struct CsvOptions {
    char delimiter = ',';           ///< Field delimiter
    bool has_header = true;         ///< First row is header
    int skip_rows = 0;              ///< Rows to skip before header
    std::string encoding = "utf-8"; ///< File encoding
    
    // Advanced options
    bool auto_detect_types = true;  ///< Auto-detect column types
    int infer_schema_rows = 1000;   ///< Rows to scan for type inference
};

/// Excel loading options
struct ExcelOptions {
    std::string sheet_name;         ///< Sheet name (empty = first sheet)
    bool has_header = true;         ///< First row is header
    int skip_rows = 0;              ///< Rows to skip
};

} // namespace timegraph

