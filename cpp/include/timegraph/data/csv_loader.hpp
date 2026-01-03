#pragma once

#include "timegraph/data/dataframe.hpp"
#include <string>
#include <vector>

namespace timegraph {

/// CSV Loader - Multi-threaded CSV parsing
class CsvLoader {
public:
    CsvLoader() = delete;  // Static class, no instances
    /// Load CSV file into DataFrame
    /// Uses memory-mapped I/O and multi-threading for performance
    static DataFrame load(const std::string& path, const CsvOptions& opts);
    
private:
    /// Parse CSV content
    static DataFrame parse_csv(const char* data, size_t size, const CsvOptions& opts);
    
    /// Detect column types from sample data
    static std::vector<ColumnType> detect_types(
        const std::vector<std::vector<std::string>>& rows,
        size_t sample_size
    );
    
    /// Parse a single value based on detected type
    static bool try_parse_double(const std::string& str, double& out);
    static bool try_parse_int64(const std::string& str, int64_t& out);
    
    /// Split CSV line into fields
    static std::vector<std::string> split_line(
        const std::string& line,
        char delimiter
    );
};

} // namespace timegraph

