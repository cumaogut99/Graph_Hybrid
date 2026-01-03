/**
 * @file core_bindings.cpp
 * @brief Python bindings for core data layer (DataProvider, DataCache, etc.)
 * 
 * Exposes the new unified data API to Python:
 * - DataChunk: Data container
 * - SignalMetadata: Signal information
 * - CacheStats: Cache statistics
 * - MpaiDataSource: MPAI file source
 * - DataProvider: Unified data access
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "timegraph/core/types.hpp"
#include "timegraph/core/data_source.hpp"
#include "timegraph/core/data_cache.hpp"
#include "timegraph/core/data_provider.hpp"
#include "timegraph/sources/mpai_source.hpp"
#include "timegraph/processing/statistics_engine.hpp"

namespace py = pybind11;

namespace timegraph {

/**
 * @brief Initialize core data layer bindings
 */
void init_core_bindings(py::module& m) {
    // ========================================================================
    // DataChunk
    // ========================================================================
    py::class_<DataChunk>(m, "DataChunk", 
        "Container for signal data chunk")
        
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("signal_name"))
        
        // Properties
        .def_readonly("signal_name", &DataChunk::signal_name,
            "Signal name/identifier")
        .def_readonly("start_row", &DataChunk::start_row,
            "Starting row index in source")
        .def_readonly("row_count", &DataChunk::row_count,
            "Number of rows in this chunk")
        
        // Data access
        .def_property_readonly("data", 
            [](const DataChunk& c) {
                return py::array_t<double>(c.data.size(), c.data.data());
            },
            "Signal data as NumPy array")
        .def_property_readonly("time_data",
            [](const DataChunk& c) {
                return py::array_t<double>(c.time_data.size(), c.time_data.data());
            },
            "Time data as NumPy array (if available)")
        
        // Methods
        .def("__len__", &DataChunk::size)
        .def("__bool__", [](const DataChunk& c) { return !c.empty(); })
        .def("empty", &DataChunk::empty, "Check if chunk is empty")
        .def("size", &DataChunk::size, "Get number of samples")
        .def("time_at", &DataChunk::time_at, py::arg("index"),
            "Get time value at index");
    
    // ========================================================================
    // SignalMetadata
    // ========================================================================
    py::class_<SignalMetadata>(m, "SignalMetadata",
        "Metadata for a signal/channel")
        
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("name"))
        
        .def_readwrite("name", &SignalMetadata::name)
        .def_readwrite("unit", &SignalMetadata::unit)
        .def_readwrite("description", &SignalMetadata::description)
        .def_readwrite("sample_rate", &SignalMetadata::sample_rate)
        .def_readwrite("total_samples", &SignalMetadata::total_samples)
        .def_readwrite("min_value", &SignalMetadata::min_value)
        .def_readwrite("max_value", &SignalMetadata::max_value)
        .def_readwrite("start_time", &SignalMetadata::start_time)
        .def_readwrite("end_time", &SignalMetadata::end_time)
        
        .def("duration", &SignalMetadata::duration, "Get duration in seconds")
        .def("range", &SignalMetadata::range, "Get value range (max - min)");
    
    // ========================================================================
    // CacheStats
    // ========================================================================
    py::class_<CacheStats>(m, "CacheStats",
        "Statistics about cache performance")
        
        .def(py::init<>())
        
        .def_readonly("hits", &CacheStats::hits, "Number of cache hits")
        .def_readonly("misses", &CacheStats::misses, "Number of cache misses")
        .def_readonly("entries", &CacheStats::entries, "Current entry count")
        .def_readonly("memory_bytes", &CacheStats::memory_bytes, "Memory usage in bytes")
        .def_readonly("max_memory", &CacheStats::max_memory, "Memory limit in bytes")
        .def_readonly("hit_rate", &CacheStats::hit_rate, "Hit rate (0.0 - 1.0)")
        
        .def_property_readonly("memory_mb",
            [](const CacheStats& s) { return s.memory_bytes / (1024.0 * 1024.0); },
            "Memory usage in megabytes");
    
    // NOTE: ColumnStatistics is already defined in processing_bindings.cpp
    // We don't redefine it here to avoid duplicate registration error
    
    // ========================================================================
    // InterpolationResult
    // ========================================================================
    py::class_<InterpolationResult>(m, "InterpolationResult",
        "Result of an interpolation operation")
        
        .def(py::init<>())
        
        .def_readonly("value", &InterpolationResult::value, "Interpolated value")
        .def_readonly("valid", &InterpolationResult::valid, "Whether interpolation succeeded")
        .def_readonly("left_index", &InterpolationResult::left_index, "Index of left neighbor")
        .def_readonly("right_index", &InterpolationResult::right_index, "Index of right neighbor")
        .def_readonly("weight", &InterpolationResult::weight, "Interpolation weight");
    
    // ========================================================================
    // MpaiDataSource
    // ========================================================================
    py::class_<MpaiDataSource, std::shared_ptr<MpaiDataSource>>(m, "MpaiDataSource",
        "Data source for MPAI files")
        
        .def(py::init<const std::string&>(), py::arg("filepath"),
            "Open MPAI file as data source")
        
        // IDataSource methods
        .def("get_signal_names", &MpaiDataSource::get_signal_names,
            "Get list of signal names")
        .def("get_signal_metadata", &MpaiDataSource::get_signal_metadata,
            py::arg("name"), "Get metadata for a signal")
        .def("get_total_samples", &MpaiDataSource::get_total_samples,
            "Get total number of samples")
        .def("get_sample_rate", &MpaiDataSource::get_sample_rate,
            "Get sample rate in Hz")
        .def("has_signal", &MpaiDataSource::has_signal, py::arg("name"),
            "Check if signal exists")
        .def("get_signal_count", &MpaiDataSource::get_signal_count,
            "Get number of signals")
        
        // Data access
        .def("read_range", &MpaiDataSource::read_range,
            py::arg("signal_name"), py::arg("start_row"), py::arg("row_count"),
            "Read data range from signal")
        .def("read_time_range", &MpaiDataSource::read_time_range,
            py::arg("start_row"), py::arg("row_count"),
            "Read time column data")
        
        // Source info
        .def("get_source_path", &MpaiDataSource::get_source_path,
            "Get file path")
        .def("get_source_type", &MpaiDataSource::get_source_type,
            "Get source type ('file')")
        .def("is_open", &MpaiDataSource::is_open,
            "Check if file is open")
        .def("is_live", &MpaiDataSource::is_live,
            "Check if source is live (always False for files)")
        .def("close", &MpaiDataSource::close,
            "Close the file")
        
        // MPAI-specific
        .def("get_compression_ratio", &MpaiDataSource::get_compression_ratio,
            "Get compression ratio")
        .def("get_memory_usage", &MpaiDataSource::get_memory_usage,
            "Get current memory usage in bytes")
        .def("has_time_column", &MpaiDataSource::has_time_column,
            "Check if file has a time column")
        .def("get_time_column_name", &MpaiDataSource::get_time_column_name,
            "Get time column name");
    
    // ========================================================================
    // DataProvider
    // ========================================================================
    py::class_<DataProvider>(m, "DataProvider",
        "Unified data access provider with caching")
        
        .def(py::init<std::shared_ptr<IDataSource>, size_t>(),
            py::arg("source"), py::arg("cache_size_mb") = 512,
            "Create DataProvider with data source and cache size")
        
        // Data access
        .def("get_data", &DataProvider::get_data,
            py::arg("signal_name"), py::arg("start_row"), py::arg("row_count"),
            "Get data range for a signal")
        .def("get_data_batch", &DataProvider::get_data_batch,
            py::arg("signal_names"), py::arg("start_row"), py::arg("row_count"),
            "Get data ranges for multiple signals")
        
        // Statistics
        .def("get_statistics", &DataProvider::get_statistics,
            py::arg("signal_name"), py::arg("start_row"), py::arg("row_count"),
            "Calculate statistics for a signal range")
        .def("get_statistics_batch", &DataProvider::get_statistics_batch,
            py::arg("signal_names"), py::arg("start_row"), py::arg("row_count"),
            "Calculate statistics for multiple signals")
        
        // Value lookup
        .def("get_value_at", &DataProvider::get_value_at,
            py::arg("signal_name"), py::arg("row_index"),
            "Get exact value at row index")
        .def("interpolate_at", &DataProvider::interpolate_at,
            py::arg("signal_name"), py::arg("time"),
            "Get interpolated value at time")
        .def("interpolate_batch", &DataProvider::interpolate_batch,
            py::arg("signal_name"), py::arg("times"),
            "Get interpolated values at multiple times")
        .def("interpolate_detailed", &DataProvider::interpolate_detailed,
            py::arg("signal_name"), py::arg("time"),
            "Get interpolation result with details")
        
        // Metadata
        .def("get_signal_names", &DataProvider::get_signal_names,
            "Get list of signal names")
        .def("get_signal_metadata", &DataProvider::get_signal_metadata,
            py::arg("name"), "Get metadata for a signal")
        .def("get_total_samples", &DataProvider::get_total_samples,
            "Get total number of samples")
        .def("get_sample_rate", &DataProvider::get_sample_rate,
            "Get sample rate in Hz")
        .def("has_signal", &DataProvider::has_signal, py::arg("name"),
            "Check if signal exists")
        .def("get_signal_count", &DataProvider::get_signal_count,
            "Get number of signals")
        
        // Time conversion
        .def("time_to_row", &DataProvider::time_to_row, py::arg("time"),
            "Convert time to row index")
        .def("row_to_time", &DataProvider::row_to_time, py::arg("row"),
            "Convert row index to time")
        
        // Cache management
        .def("clear_cache", &DataProvider::clear_cache,
            "Clear all caches")
        .def("clear_data_cache", &DataProvider::clear_data_cache,
            "Clear data cache only")
        .def("clear_stats_cache", &DataProvider::clear_stats_cache,
            "Clear statistics cache only")
        .def("set_cache_size", &DataProvider::set_cache_size,
            py::arg("mb"), "Set cache memory limit in MB")
        .def("get_cache_stats", &DataProvider::get_cache_stats,
            "Get cache statistics")
        .def("prefetch", &DataProvider::prefetch,
            py::arg("signal_name"), py::arg("center_time"),
            "Prefetch data around a time point")
        
        // Source info
        .def("get_source_type", &DataProvider::get_source_type,
            "Get source type")
        .def("get_source_path", &DataProvider::get_source_path,
            "Get source path")
        .def("is_live", &DataProvider::is_live,
            "Check if source is live");
    
    // ========================================================================
    // Convenience Functions
    // ========================================================================
    
    m.def("open_mpai", [](const std::string& filepath, size_t cache_mb) {
        auto source = std::make_shared<MpaiDataSource>(filepath);
        return std::make_unique<DataProvider>(source, cache_mb);
    }, py::arg("filepath"), py::arg("cache_mb") = 512,
    "Open MPAI file and create DataProvider");
}

} // namespace timegraph

