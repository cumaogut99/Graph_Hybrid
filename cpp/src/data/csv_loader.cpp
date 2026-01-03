#include "timegraph/data/csv_loader.hpp"
#include "timegraph/data/column.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace timegraph {

// ===== Public API =====

DataFrame CsvLoader::load(const std::string& path, const CsvOptions& opts) {
    // Open file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    // Get file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read file into memory
    // TODO Sprint 1.3: Replace with memory-mapped I/O (mio)
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    
    // Parse CSV
    return parse_csv(buffer.data(), buffer.size(), opts);
}

// ===== Internal Methods =====

DataFrame CsvLoader::parse_csv(const char* data, size_t size, const CsvOptions& opts) {
    DataFrame df;
    
    // Convert to string for easier processing
    // TODO: Optimize - process char* directly
    std::string content(data, size);
    std::istringstream stream(content);
    std::string line;
    
    // Skip rows
    for (int i = 0; i < opts.skip_rows; ++i) {
        if (!std::getline(stream, line)) {
            throw std::runtime_error("Not enough rows to skip");
        }
    }
    
    // Read header
    std::vector<std::string> column_names;
    if (opts.has_header) {
        if (!std::getline(stream, line)) {
            throw std::runtime_error("Empty file or missing header");
        }
        column_names = split_line(line, opts.delimiter);
        
        // Clean column names
        for (auto& name : column_names) {
            // Trim whitespace
            name.erase(0, name.find_first_not_of(" \t\r\n"));
            name.erase(name.find_last_not_of(" \t\r\n") + 1);
            
            // Remove quotes if present
            if (name.size() >= 2 && name.front() == '"' && name.back() == '"') {
                name = name.substr(1, name.size() - 2);
            }
        }
    }
    
    // Read data rows
    std::vector<std::vector<std::string>> rows;
    while (std::getline(stream, line)) {
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        auto fields = split_line(line, opts.delimiter);
        
        // Validate field count
        if (!rows.empty() && fields.size() != rows[0].size()) {
            // Skip malformed rows
            continue;
        }
        
        rows.push_back(std::move(fields));
    }
    
    if (rows.empty()) {
        throw std::runtime_error("No data rows found");
    }
    
    const size_t num_cols = rows[0].size();
    const size_t num_rows = rows.size();
    
    // Generate column names if no header
    if (!opts.has_header) {
        column_names.clear();
        for (size_t i = 0; i < num_cols; ++i) {
            column_names.push_back("column_" + std::to_string(i));
        }
    }
    
    // Validate column count
    if (column_names.size() != num_cols) {
        throw std::runtime_error("Header column count mismatch");
    }
    
    // Detect column types
    std::vector<ColumnType> types;
    if (opts.auto_detect_types) {
        size_t sample_size = std::min(
            static_cast<size_t>(opts.infer_schema_rows),
            num_rows
        );
        types = detect_types(rows, sample_size);
    } else {
        // Default to Float64
        types = std::vector<ColumnType>(num_cols, ColumnType::FLOAT64);
    }
    
    // Create columns
    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto& col_name = column_names[col_idx];
        const auto col_type = types[col_idx];
        
        // Currently only support Float64
        // TODO: Support other types
        if (col_type == ColumnType::FLOAT64) {
            std::vector<double> values;
            values.reserve(num_rows);
            
            for (const auto& row : rows) {
                double value;
                if (try_parse_double(row[col_idx], value)) {
                    values.push_back(value);
                } else {
                    // Parse failed - use NaN
                    values.push_back(std::nan(""));
                }
            }
            
            auto column = std::make_shared<Float64Column>(col_name, std::move(values), ColumnType::FLOAT64);
            df.add_column(col_name, column);
        } else {
            // Fallback to Float64 for unsupported types
            std::vector<double> values(num_rows, 0.0);
            auto column = std::make_shared<Float64Column>(col_name, std::move(values), ColumnType::FLOAT64);
            df.add_column(col_name, column);
        }
    }
    
    return df;
}

std::vector<ColumnType> CsvLoader::detect_types(
    const std::vector<std::vector<std::string>>& rows,
    size_t sample_size
) {
    if (rows.empty()) {
        return {};
    }
    
    const size_t num_cols = rows[0].size();
    std::vector<ColumnType> types(num_cols, ColumnType::FLOAT64);
    
    // For each column, check sample rows
    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        bool all_int = true;
        bool all_float = true;
        
        size_t samples = std::min(sample_size, rows.size());
        for (size_t row_idx = 0; row_idx < samples; ++row_idx) {
            const auto& value = rows[row_idx][col_idx];
            
            if (value.empty()) {
                continue;
            }
            
            // Try int
            int64_t int_val;
            if (!try_parse_int64(value, int_val)) {
                all_int = false;
            }
            
            // Try float
            double float_val;
            if (!try_parse_double(value, float_val)) {
                all_float = false;
                break;
            }
        }
        
        // Determine type
        if (all_int) {
            types[col_idx] = ColumnType::INT64;
        } else if (all_float) {
            types[col_idx] = ColumnType::FLOAT64;
        } else {
            types[col_idx] = ColumnType::STRING;
        }
    }
    
    return types;
}

bool CsvLoader::try_parse_double(const std::string& str, double& out) {
    if (str.empty()) {
        return false;
    }
    
    try {
        size_t pos;
        out = std::stod(str, &pos);
        // Check if entire string was consumed
        return pos == str.size();
    } catch (...) {
        return false;
    }
}

bool CsvLoader::try_parse_int64(const std::string& str, int64_t& out) {
    if (str.empty()) {
        return false;
    }
    
    try {
        size_t pos;
        out = std::stoll(str, &pos);
        // Check if entire string was consumed
        return pos == str.size();
    } catch (...) {
        return false;
    }
}

std::vector<std::string> CsvLoader::split_line(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
            // Skip quote character
        } else if (c == delimiter && !in_quotes) {
            // Field separator
            fields.push_back(field);
            field.clear();
        } else if (c == '\r' || c == '\n') {
            // End of line
            break;
        } else {
            field += c;
        }
    }
    
    // Add last field
    fields.push_back(field);
    
    return fields;
}

} // namespace timegraph

