#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <cstddef>

namespace timegraph {

/// Column data types
enum class ColumnType {
    FLOAT64,    ///< 64-bit floating point
    INT64,      ///< 64-bit integer
    STRING,     ///< String type
    DATETIME    ///< Datetime type (stored as int64 timestamp)
};

/// Abstract column interface
class IColumn {
public:
    virtual ~IColumn() = default;
    
    /// Get column type
    virtual ColumnType type() const = 0;
    
    /// Get number of elements
    virtual size_t size() const = 0;
    
    /// Get column name
    virtual std::string name() const = 0;
};

/// Typed column implementation
/// Stores data in contiguous memory for cache efficiency
template<typename T>
class TypedColumn : public IColumn {
private:
    std::string name_;
    std::vector<T> data_;
    ColumnType type_;
    
public:
    /// Constructor
    TypedColumn(std::string name, std::vector<T> data, ColumnType type)
        : name_(std::move(name))
        , data_(std::move(data))
        , type_(type)
    {}
    
    /// Get read-only view of data (zero-copy)
    std::span<const T> view() const { 
        return std::span<const T>(data_.data(), data_.size()); 
    }
    
    /// Get mutable view of data
    std::span<T> view_mut() { 
        return std::span<T>(data_.data(), data_.size()); 
    }
    
    /// Get raw pointer (for Python bindings)
    const T* data_ptr() const { return data_.data(); }
    
    /// IColumn interface implementation
    ColumnType type() const override { return type_; }
    size_t size() const override { return data_.size(); }
    std::string name() const override { return name_; }
};

/// Type aliases for common columns
using Float64Column = TypedColumn<double>;
using Int64Column = TypedColumn<int64_t>;
using StringColumn = TypedColumn<std::string>;

} // namespace timegraph

