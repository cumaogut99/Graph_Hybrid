/**
 * @file statistics_bindings.cpp
 * @brief Python bindings for fast statistics calculator
 */

#include "timegraph/data/chunk_metadata.hpp"
#include "timegraph/data/metadata_builder.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/statistics/fast_stats_calculator.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace timegraph::mpai;

void bind_statistics(py::module_ &m) {
  // RangeStatistics structure
  py::class_<RangeStatistics>(m, "RangeStatistics")
      .def(py::init<>())
      .def_readonly("min", &RangeStatistics::min, "Minimum value in range")
      .def_readonly("max", &RangeStatistics::max, "Maximum value in range")
      .def_readonly("mean", &RangeStatistics::mean, "Mean (average) value")
      .def_readonly("variance", &RangeStatistics::variance, "Variance")
      .def_readonly("std_dev", &RangeStatistics::std_dev, "Standard deviation")
      .def_readonly("rms", &RangeStatistics::rms, "Root mean square")
      .def_readonly("count", &RangeStatistics::count,
                    "Number of valid data points")
      .def_readonly("complete_chunks", &RangeStatistics::complete_chunks,
                    "Number of complete chunks (O(1) calculation)")
      .def_readonly("partial_chunks", &RangeStatistics::partial_chunks,
                    "Number of partial chunks (loaded from disk)")
      .def_readonly("rows_loaded", &RangeStatistics::rows_loaded,
                    "Actual rows loaded from disk")
      .def("__repr__", [](const RangeStatistics &s) {
        return "<RangeStatistics min=" + std::to_string(s.min) +
               " max=" + std::to_string(s.max) +
               " mean=" + std::to_string(s.mean) +
               " std=" + std::to_string(s.std_dev) +
               " count=" + std::to_string(s.count) + ">";
      });

  // FastStatsCalculator class
  py::class_<FastStatsCalculator>(m, "FastStatsCalculator")
      .def_static("calculate_range_statistics",
                  &FastStatsCalculator::calculate_range_statistics,
                  py::arg("reader"), py::arg("column_name"),
                  py::arg("start_row"), py::arg("end_row"),
                  R"pbdoc(
                Calculate statistics for a row range using pre-aggregated metadata.
                
                This method uses O(1) aggregation for complete chunks and only loads
                edge data from disk, making it extremely fast even for large ranges.
                
                Args:
                    reader: MpaiReader instance
                    column_name: Name of column to calculate statistics for
                    start_row: Start row index (inclusive)
                    end_row: End row index (exclusive)
                
                Returns:
                    RangeStatistics object with min, max, mean, std, rms, etc.
                
                Performance:
                    - 100M points, full range: < 1ms (all complete chunks)
                    - 100M points, partial range: < 10ms (mostly complete chunks + edges)
                
                Example:
                    >>> reader = MpaiReader("data.mpai")
                    >>> stats = FastStatsCalculator.calculate_range_statistics(
                    ...     reader, "signal1", 0, 10_000_000
                    ... )
                    >>> print(f"Mean: {stats.mean:.2f}, Std: {stats.std_dev:.2f}")
                    Mean: 1.23, Std: 0.45
            )pbdoc")
      .def_static("calculate_time_range_statistics",
                  &FastStatsCalculator::calculate_time_range_statistics,
                  py::arg("reader"), py::arg("column_name"),
                  py::arg("start_time"), py::arg("end_time"),
                  py::arg("time_column") = "time",
                  R"pbdoc(
                Calculate statistics for a time range.
                
                Converts time values to row indices and then calculates statistics.
                
                Args:
                    reader: MpaiReader instance
                    column_name: Name of column to calculate statistics for
                    start_time: Start time value
                    end_time: End time value
                    time_column: Name of time column (default: "time")
                
                Returns:
                    RangeStatistics object
                
                Example:
                    >>> stats = FastStatsCalculator.calculate_time_range_statistics(
                    ...     reader, "signal1", 10.0, 20.0
                    ... )
            )pbdoc");

  // MetadataBuilder class
  py::class_<MetadataBuilder>(m, "MetadataBuilder")
      .def_static("build_chunk_metadata",
                  &MetadataBuilder::build_chunk_metadata, py::arg("data"),
                  py::arg("size"), py::arg("start_row"),
                  R"pbdoc(
                Build metadata for a single chunk (internal use).
                
                Uses SIMD (AVX2) when available for 4x speedup.
            )pbdoc")
      .def_static(
          "build_column_metadata",
          [](py::array_t<double> data, size_t chunk_size) {
            auto buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            size_t size = buf.size;
            return MetadataBuilder::build_column_metadata(ptr, size,
                                                          chunk_size);
          },
          py::arg("data"), py::arg("chunk_size") = 4096,
          R"pbdoc(
                Build metadata for all chunks in a column.
                
                Args:
                    data: NumPy array of float64 values
                    chunk_size: Rows per chunk (default: 4096)
                
                Returns:
                    List of ChunkMetadata objects
            )pbdoc");

  // ChunkMetadata structure
  py::class_<ChunkMetadata>(m, "ChunkMetadata")
      .def(py::init<>())
      .def_readonly("start_row", &ChunkMetadata::start_row)
      .def_readonly("row_count", &ChunkMetadata::row_count)
      .def_readonly("sum", &ChunkMetadata::sum)
      .def_readonly("sum_squares", &ChunkMetadata::sum_squares)
      .def_readonly("min_value", &ChunkMetadata::min_value)
      .def_readonly("max_value", &ChunkMetadata::max_value)
      .def_readonly("count", &ChunkMetadata::count);
}
