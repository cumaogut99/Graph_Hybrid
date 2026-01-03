<<<<<<< HEAD
#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/processing/arrow_utils.hpp"
#include "timegraph/processing/critical_points.hpp"
#include "timegraph/processing/downsample.hpp"
#include "timegraph/processing/filter_engine.hpp"
#include "timegraph/processing/limit_violation_engine.hpp"
#include "timegraph/processing/smart_downsampler.hpp"
#include "timegraph/processing/statistics_engine.hpp"
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
=======
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include "timegraph/processing/filter_engine.hpp"
#include "timegraph/processing/statistics_engine.hpp"
#include "timegraph/processing/limit_violation_engine.hpp"
#include "timegraph/processing/downsample.hpp"
#include "timegraph/processing/critical_points.hpp"
#include "timegraph/processing/smart_downsampler.hpp"
#include "timegraph/processing/arrow_utils.hpp"
#include "timegraph/data/mpai_reader.hpp"
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

namespace py = pybind11;
using namespace timegraph;

<<<<<<< HEAD
void bind_processing(py::module_ &m) {
  // ===== FILTER ENGINE =====

  // FilterType enum
  py::enum_<FilterType>(m, "FilterType")
      .value("RANGE", FilterType::RANGE)
      .value("GREATER", FilterType::GREATER)
      .value("LESS", FilterType::LESS)
      .value("EQUAL", FilterType::EQUAL)
      .value("NOT_EQUAL", FilterType::NOT_EQUAL)
      .export_values();

  // FilterOperator enum
  py::enum_<FilterOperator>(m, "FilterOperator")
      .value("AND", FilterOperator::AND)
      .value("OR", FilterOperator::OR)
      .export_values();

  // FilterCondition struct
  py::class_<FilterCondition>(m, "FilterCondition")
      .def(py::init<>())
      .def_readwrite("column_name", &FilterCondition::column_name)
      .def_readwrite("type", &FilterCondition::type)
      .def_readwrite("min_value", &FilterCondition::min_value)
      .def_readwrite("max_value", &FilterCondition::max_value)
      .def_readwrite("threshold", &FilterCondition::threshold)
      .def_readwrite("op", &FilterCondition::op)
      .def("__repr__", [](const FilterCondition &c) {
        return "<FilterCondition column='" + c.column_name + "'>";
      });

  // TimeSegment struct
  py::class_<TimeSegment>(m, "TimeSegment")
      .def(py::init<double, double, size_t, size_t>(), py::arg("start_time"),
           py::arg("end_time"), py::arg("start_index"), py::arg("end_index"))
      .def_readonly("start_time", &TimeSegment::start_time)
      .def_readonly("end_time", &TimeSegment::end_time)
      .def_readonly("start_index", &TimeSegment::start_index)
      .def_readonly("end_index", &TimeSegment::end_index)
      .def("__repr__", [](const TimeSegment &s) {
        return "<TimeSegment [" + std::to_string(s.start_time) + ", " +
               std::to_string(s.end_time) + "]>";
      });

  // FilterEngine class
  py::class_<FilterEngine>(m, "FilterEngine")
      .def(py::init<>())
      .def("calculate_segments", &FilterEngine::calculate_segments,
           py::arg("df"), py::arg("time_column"), py::arg("conditions"),
           "Calculate time segments that satisfy filter conditions")
      .def("calculate_mask", &FilterEngine::calculate_mask, py::arg("df"),
           py::arg("conditions"),
           "Calculate boolean mask for filter conditions")
      .def("calculate_streaming", &FilterEngine::calculate_streaming,
           py::arg("reader"), py::arg("time_column"), py::arg("conditions"),
           py::arg("start_row") = 0, py::arg("row_count") = 0,
           "Calculate time segments from MPAI file using streaming (low RAM)")
      .def(
          "calculate_mask_arrow",
          [](FilterEngine &self, py::array_t<double> data,
             const FilterCondition &cond) {
            // Convert NumPy array to std::vector (zero-copy view preferred)
            py::buffer_info buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);

            // Call Arrow compute function
            return self.calculate_mask_arrow(vec, cond);
          },
          py::arg("data"), py::arg("condition"),
          "Calculate mask using Arrow Compute (SIMD-optimized, 8-15x faster!)\n"
          "Falls back to native if Arrow not available.")
      .def(
          "apply_filter_to_chunk_arrow",
          [](FilterEngine &self, py::array_t<double> chunk_data,
             const std::vector<FilterCondition> &conditions) {
            py::buffer_info buf = chunk_data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);

            return self.apply_filter_to_chunk_arrow(vec, conditions);
          },
          py::arg("chunk_data"), py::arg("conditions"),
          "Apply filter to MPAI chunk using Arrow Compute\n"
          "Wraps chunk as Arrow array (zero-copy) for SIMD filtering.")
      .def_static("check_condition", &FilterEngine::check_condition,
                  py::arg("value"), py::arg("condition"),
                  "Check if a value passes a filter condition")
      .def("load_filtered_segment", &FilterEngine::load_filtered_segment,
           py::arg("reader"), py::arg("signal_name"), py::arg("time_column"),
           py::arg("time_start"), py::arg("time_end"), py::arg("conditions"),
           "Load filtered segment data from MPAI file.\n"
           "Returns only (x, y) pairs that match ALL filter conditions.\n\n"
           "Args:\n"
           "    reader: MpaiReader instance\n"
           "    signal_name: Signal column to load\n"
           "    time_column: Time column name\n"
           "    time_start: Segment start time\n"
           "    time_end: Segment end time\n"
           "    conditions: List of FilterCondition objects\n\n"
           "Returns:\n"
           "    Tuple of (time_values, signal_values) for matching points")
#ifdef HAVE_ARROW
      .def(
          "calculate_segments_from_arrow",
          [](FilterEngine &self, py::array_t<double> time_data,
             py::dict column_data_dict,
             const std::vector<FilterCondition> &conditions) {
            // Convert time array to vector
            py::buffer_info time_buf = time_data.request();
            const double *time_ptr = static_cast<const double *>(time_buf.ptr);
            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);

            // Convert dict of arrays to map of vectors
            std::map<std::string, std::vector<double>> column_map;
            for (auto item : column_data_dict) {
              std::string col_name = py::cast<std::string>(item.first);
              py::array_t<double> col_array =
                  py::cast<py::array_t<double>>(item.second);
              py::buffer_info col_buf = col_array.request();
              const double *col_ptr = static_cast<const double *>(col_buf.ptr);
              column_map[col_name] =
                  std::vector<double>(col_ptr, col_ptr + col_buf.size);
            }

            return self.calculate_segments_from_arrow(time_vec, column_map,
                                                      conditions);
          },
          py::arg("time_data"), py::arg("column_data"), py::arg("conditions"),
          "Calculate filter segments from Arrow arrays (zero-copy from Python "
          "mmap).\n\n"
          "This is the main entry point for Arrow Bridge filtering.\n\n"
          "Args:\n"
          "    time_data: NumPy array of time values (from Arrow/mmap)\n"
          "    column_data: Dict mapping column names to NumPy arrays\n"
          "    conditions: List of FilterCondition objects\n\n"
          "Returns:\n"
          "    List of TimeSegment objects matching the filter")
#endif
      ;

  // ===== STATISTICS ENGINE =====

  // ColumnStatistics struct
  py::class_<ColumnStatistics>(m, "ColumnStatistics")
      .def(py::init<>())
      .def_readonly("mean", &ColumnStatistics::mean)
      .def_readonly("std_dev", &ColumnStatistics::std_dev)
      .def_readonly("min", &ColumnStatistics::min)
      .def_readonly("max", &ColumnStatistics::max)
      .def_readonly("median", &ColumnStatistics::median)
      .def_readonly("sum", &ColumnStatistics::sum)
      .def_readonly("rms", &ColumnStatistics::rms)
      .def_readonly("peak_to_peak", &ColumnStatistics::peak_to_peak)
      .def_readonly("count", &ColumnStatistics::count)
      .def_readonly("valid_count", &ColumnStatistics::valid_count)
      .def("__repr__", [](const ColumnStatistics &s) {
        return "<ColumnStatistics mean=" + std::to_string(s.mean) +
               " std=" + std::to_string(s.std_dev) +
               " min=" + std::to_string(s.min) +
               " max=" + std::to_string(s.max) +
               " rms=" + std::to_string(s.rms) + ">";
      });

  // ThresholdStatistics struct
  py::class_<ThresholdStatistics, ColumnStatistics>(m, "ThresholdStatistics")
      .def(py::init<>())
      .def_readonly("threshold", &ThresholdStatistics::threshold)
      .def_readonly("above_count", &ThresholdStatistics::above_count)
      .def_readonly("below_count", &ThresholdStatistics::below_count)
      .def_readonly("above_percentage", &ThresholdStatistics::above_percentage)
      .def_readonly("below_percentage", &ThresholdStatistics::below_percentage)
      .def_readonly("time_above", &ThresholdStatistics::time_above)
      .def_readonly("time_below", &ThresholdStatistics::time_below)
      .def("__repr__", [](const ThresholdStatistics &s) {
        return "<ThresholdStatistics threshold=" + std::to_string(s.threshold) +
               " above=" + std::to_string(s.above_percentage) + "%>";
      });

  // StatisticsEngine class
  py::class_<StatisticsEngine>(m, "StatisticsEngine")
      .def(py::init<>())
      .def_static("calculate", &StatisticsEngine::calculate, py::arg("df"),
                  py::arg("column_name"),
                  "Calculate basic statistics for a column")
      .def_static("calculate_range", &StatisticsEngine::calculate_range,
                  py::arg("df"), py::arg("column_name"), py::arg("start_index"),
                  py::arg("end_index"),
                  "Calculate statistics for a column range")
      .def_static("calculate_with_threshold",
                  &StatisticsEngine::calculate_with_threshold, py::arg("df"),
                  py::arg("column_name"), py::arg("time_column"),
                  py::arg("threshold"),
                  "Calculate statistics with threshold analysis")
      .def_static("calculate_rolling", &StatisticsEngine::calculate_rolling,
                  py::arg("df"), py::arg("column_name"), py::arg("window_size"),
                  "Calculate rolling statistics")
      .def_static("percentile", &StatisticsEngine::percentile, py::arg("df"),
                  py::arg("column_name"), py::arg("percentile"),
                  "Calculate percentile value")
      .def_static("histogram", &StatisticsEngine::histogram, py::arg("df"),
                  py::arg("column_name"), py::arg("num_bins"),
                  "Calculate histogram bins")
      // NEW: MPAI Streaming Statistics
      .def_static("calculate_streaming", &StatisticsEngine::calculate_streaming,
                  py::arg("reader"), py::arg("column_name"),
                  py::arg("start_row") = 0, py::arg("row_count") = 0,
                  "Calculate statistics from MPAI file (streaming, low RAM)")
      .def_static("calculate_time_range_streaming",
                  &StatisticsEngine::calculate_time_range_streaming,
                  py::arg("reader"), py::arg("column_name"),
                  py::arg("time_column"), py::arg("start_time"),
                  py::arg("end_time"),
                  "Calculate statistics for time range (MPAI streaming)")
      // NEW: Arrow Compute Statistics (20-30x faster!)
      .def_static(
          "calculate_arrow",
          [](py::array_t<double> data) {
            py::buffer_info buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            return StatisticsEngine::calculate_arrow(vec);
          },
          py::arg("data"),
          "Calculate statistics using Arrow Compute (SIMD-optimized, 20-30x "
          "faster!)\n"
          "Falls back to native SIMD if Arrow not available.")
      .def_static(
          "calculate_chunk_arrow",
          [](py::array_t<double> chunk_data) {
            py::buffer_info buf = chunk_data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            return StatisticsEngine::calculate_chunk_arrow(vec);
          },
          py::arg("chunk_data"),
          "Calculate statistics for MPAI chunk using Arrow Compute")
      .def_static(
          "mean_arrow",
          [](py::array_t<double> data) {
            py::buffer_info buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            return StatisticsEngine::mean_arrow(vec);
          },
          py::arg("data"), "Fast mean calculation using Arrow Compute")
      .def_static(
          "stddev_arrow",
          [](py::array_t<double> data) {
            py::buffer_info buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            return StatisticsEngine::stddev_arrow(vec);
          },
          py::arg("data"), "Fast standard deviation using Arrow Compute")
      .def_static(
          "minmax_arrow",
          [](py::array_t<double> data) {
            py::buffer_info buf = data.request();
            const double *ptr = static_cast<const double *>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            double min, max;
            StatisticsEngine::minmax_arrow(vec, min, max);
            return py::make_tuple(min, max);
          },
          py::arg("data"),
          "Fast min/max using Arrow Compute\nReturns: (min, max) tuple");

  // ===== DOWNSAMPLE (STREAMING) =====
  m.def(
      "downsample_minmax_streaming",
      [](timegraph::mpai::MpaiReader &reader, const std::string &time_col,
         const std::string &signal_col, uint64_t max_points, double warning_min,
         double warning_max) {
        auto result = timegraph::downsample_minmax_streaming(
            reader, time_col, signal_col, max_points, warning_min, warning_max);
        py::dict d;
        d["time"] = std::move(result.time);
        d["value"] = std::move(result.value);
        return d;
      },
      py::arg("reader"), py::arg("time_column"), py::arg("signal_column"),
      py::arg("max_points") = 100000,
      py::arg("warning_min") = std::numeric_limits<double>::quiet_NaN(),
      py::arg("warning_max") = std::numeric_limits<double>::quiet_NaN(),
      "Downsample MPAI column with first/min/max (+optional threshold) per "
      "bucket, streaming/low RAM");

  // ===== LIMIT VIOLATION ENGINE =====

  // ViolationSegment struct
  py::class_<ViolationSegment>(m, "ViolationSegment")
      .def(py::init<uint64_t, uint64_t, double, double, double, double>(),
           py::arg("start_index"), py::arg("end_index"), py::arg("start_time"),
           py::arg("end_time"), py::arg("min_value"), py::arg("max_value"))
      .def_readonly("start_index", &ViolationSegment::start_index)
      .def_readonly("end_index", &ViolationSegment::end_index)
      .def_readonly("start_time", &ViolationSegment::start_time)
      .def_readonly("end_time", &ViolationSegment::end_time)
      .def_readonly("min_value", &ViolationSegment::min_value)
      .def_readonly("max_value", &ViolationSegment::max_value)
      .def("__repr__", [](const ViolationSegment &s) {
        return "<ViolationSegment [" + std::to_string(s.start_time) + ", " +
               std::to_string(s.end_time) + "] range=[" +
               std::to_string(s.min_value) + ", " +
               std::to_string(s.max_value) + "]>";
      });

  // LimitViolationResult struct
  py::class_<LimitViolationResult>(m, "LimitViolationResult")
      .def(py::init<>())
      .def_readonly("violations", &LimitViolationResult::violations)
      .def_readonly("total_violation_points",
                    &LimitViolationResult::total_violation_points)
      .def_readonly("total_data_points",
                    &LimitViolationResult::total_data_points)
      .def_readonly("min_violation_value",
                    &LimitViolationResult::min_violation_value)
      .def_readonly("max_violation_value",
                    &LimitViolationResult::max_violation_value)
      .def("__repr__", [](const LimitViolationResult &r) {
        return "<LimitViolationResult violations=" +
               std::to_string(r.violations.size()) +
               " total_points=" + std::to_string(r.total_data_points) +
               " violation_points=" + std::to_string(r.total_violation_points) +
               ">";
      });

  // LimitViolationEngine class
  py::class_<LimitViolationEngine>(m, "LimitViolationEngine")
      .def(py::init<>())
      .def(
          "calculate_violations_streaming",
          [](LimitViolationEngine &engine, timegraph::mpai::MpaiReader &reader,
             const std::string &signal_name, const std::string &time_column,
             double warning_min, double warning_max, uint64_t start_row,
             uint64_t row_count) {
            try {
              return engine.calculate_violations_streaming(
                  reader, signal_name, time_column, warning_min, warning_max,
                  start_row, row_count);
            } catch (const std::exception &e) {
              // Convert C++ exception to Python RuntimeError
              throw std::runtime_error(
                  std::string("C++ Limit Violation Error: ") + e.what());
            }
          },
          py::arg("reader"), py::arg("signal_name"), py::arg("time_column"),
          py::arg("warning_min"), py::arg("warning_max"),
          py::arg("start_row") = 0, py::arg("row_count") = 0,
          "Find all limit violations in MPAI data using streaming (FAST, "
          "checks ALL data)")
      .def(
          "calculate_violations_arrays",
          [](LimitViolationEngine &engine, py::array_t<double> signal_data,
             py::array_t<double> time_data, double warning_min,
             double warning_max) {
            // Get numpy array pointers
            py::buffer_info signal_buf = signal_data.request();
            py::buffer_info time_buf = time_data.request();

            if (signal_buf.ndim != 1 || time_buf.ndim != 1) {
              throw std::runtime_error("Arrays must be 1-dimensional");
            }

            if (signal_buf.size != time_buf.size) {
              throw std::runtime_error(
                  "Signal and time arrays must have same length");
            }

            const double *signal_ptr = static_cast<double *>(signal_buf.ptr);
            const double *time_ptr = static_cast<double *>(time_buf.ptr);

            return engine.calculate_violations_arrays(signal_ptr, time_ptr,
                                                      signal_buf.size,
                                                      warning_min, warning_max);
          },
          py::arg("signal_data"), py::arg("time_data"), py::arg("warning_min"),
          py::arg("warning_max"),
          "Find all limit violations in NumPy arrays (for small datasets)");

  // ===== ARROW INTEGRATION UTILITIES =====

  // Check if Arrow is available
  m.def("is_arrow_available", &arrow_utils::is_arrow_available,
        "Check if Arrow Compute is available for SIMD acceleration\n"
        "Returns True if Arrow was compiled in, False otherwise.");

  // Version info
  m.def(
      "get_arrow_info",
      []() {
=======
void bind_processing(py::module_& m) {
    // ===== FILTER ENGINE =====
    
    // FilterType enum
    py::enum_<FilterType>(m, "FilterType")
        .value("RANGE", FilterType::RANGE)
        .value("GREATER", FilterType::GREATER)
        .value("LESS", FilterType::LESS)
        .value("EQUAL", FilterType::EQUAL)
        .value("NOT_EQUAL", FilterType::NOT_EQUAL)
        .export_values();
    
    // FilterOperator enum
    py::enum_<FilterOperator>(m, "FilterOperator")
        .value("AND", FilterOperator::AND)
        .value("OR", FilterOperator::OR)
        .export_values();
    
    // FilterCondition struct
    py::class_<FilterCondition>(m, "FilterCondition")
        .def(py::init<>())
        .def_readwrite("column_name", &FilterCondition::column_name)
        .def_readwrite("type", &FilterCondition::type)
        .def_readwrite("min_value", &FilterCondition::min_value)
        .def_readwrite("max_value", &FilterCondition::max_value)
        .def_readwrite("threshold", &FilterCondition::threshold)
        .def_readwrite("op", &FilterCondition::op)
        .def("__repr__", [](const FilterCondition& c) {
            return "<FilterCondition column='" + c.column_name + "'>";
        });
    
    // TimeSegment struct
    py::class_<TimeSegment>(m, "TimeSegment")
        .def(py::init<double, double, size_t, size_t>(),
             py::arg("start_time"), py::arg("end_time"),
             py::arg("start_index"), py::arg("end_index"))
        .def_readonly("start_time", &TimeSegment::start_time)
        .def_readonly("end_time", &TimeSegment::end_time)
        .def_readonly("start_index", &TimeSegment::start_index)
        .def_readonly("end_index", &TimeSegment::end_index)
        .def("__repr__", [](const TimeSegment& s) {
            return "<TimeSegment [" + std::to_string(s.start_time) + 
                   ", " + std::to_string(s.end_time) + "]>";
        });
    
    // FilterEngine class
    py::class_<FilterEngine>(m, "FilterEngine")
        .def(py::init<>())
        .def("calculate_segments", &FilterEngine::calculate_segments,
             py::arg("df"), py::arg("time_column"), py::arg("conditions"),
             "Calculate time segments that satisfy filter conditions")
        .def("calculate_mask", &FilterEngine::calculate_mask,
             py::arg("df"), py::arg("conditions"),
             "Calculate boolean mask for filter conditions")
        .def("calculate_streaming", &FilterEngine::calculate_streaming,
             py::arg("reader"), py::arg("time_column"), py::arg("conditions"),
             py::arg("start_row") = 0, py::arg("row_count") = 0,
             "Calculate time segments from MPAI file using streaming (low RAM)")
        .def("calculate_mask_arrow", 
             [](FilterEngine& self, py::array_t<double> data, const FilterCondition& cond) {
                 // Convert NumPy array to std::vector (zero-copy view preferred)
                 py::buffer_info buf = data.request();
                 const double* ptr = static_cast<const double*>(buf.ptr);
                 std::vector<double> vec(ptr, ptr + buf.size);
                 
                 // Call Arrow compute function
                 return self.calculate_mask_arrow(vec, cond);
             },
             py::arg("data"), py::arg("condition"),
             "Calculate mask using Arrow Compute (SIMD-optimized, 8-15x faster!)\n"
             "Falls back to native if Arrow not available.")
        .def("apply_filter_to_chunk_arrow",
             [](FilterEngine& self, py::array_t<double> chunk_data, 
                const std::vector<FilterCondition>& conditions) {
                 py::buffer_info buf = chunk_data.request();
                 const double* ptr = static_cast<const double*>(buf.ptr);
                 std::vector<double> vec(ptr, ptr + buf.size);
                 
                 return self.apply_filter_to_chunk_arrow(vec, conditions);
             },
             py::arg("chunk_data"), py::arg("conditions"),
             "Apply filter to MPAI chunk using Arrow Compute\n"
             "Wraps chunk as Arrow array (zero-copy) for SIMD filtering.")
        .def_static("check_condition", &FilterEngine::check_condition,
                   py::arg("value"), py::arg("condition"),
                   "Check if a value passes a filter condition");
    
    // ===== STATISTICS ENGINE =====
    
    // ColumnStatistics struct
    py::class_<ColumnStatistics>(m, "ColumnStatistics")
        .def(py::init<>())
        .def_readonly("mean", &ColumnStatistics::mean)
        .def_readonly("std_dev", &ColumnStatistics::std_dev)
        .def_readonly("min", &ColumnStatistics::min)
        .def_readonly("max", &ColumnStatistics::max)
        .def_readonly("median", &ColumnStatistics::median)
        .def_readonly("sum", &ColumnStatistics::sum)
        .def_readonly("rms", &ColumnStatistics::rms)
        .def_readonly("peak_to_peak", &ColumnStatistics::peak_to_peak)
        .def_readonly("count", &ColumnStatistics::count)
        .def_readonly("valid_count", &ColumnStatistics::valid_count)
        .def("__repr__", [](const ColumnStatistics& s) {
            return "<ColumnStatistics mean=" + std::to_string(s.mean) + 
                   " std=" + std::to_string(s.std_dev) + 
                   " min=" + std::to_string(s.min) +
                   " max=" + std::to_string(s.max) + 
                   " rms=" + std::to_string(s.rms) + ">";
        });
    
    // ThresholdStatistics struct
    py::class_<ThresholdStatistics, ColumnStatistics>(m, "ThresholdStatistics")
        .def(py::init<>())
        .def_readonly("threshold", &ThresholdStatistics::threshold)
        .def_readonly("above_count", &ThresholdStatistics::above_count)
        .def_readonly("below_count", &ThresholdStatistics::below_count)
        .def_readonly("above_percentage", &ThresholdStatistics::above_percentage)
        .def_readonly("below_percentage", &ThresholdStatistics::below_percentage)
        .def_readonly("time_above", &ThresholdStatistics::time_above)
        .def_readonly("time_below", &ThresholdStatistics::time_below)
        .def("__repr__", [](const ThresholdStatistics& s) {
            return "<ThresholdStatistics threshold=" + std::to_string(s.threshold) + 
                   " above=" + std::to_string(s.above_percentage) + "%>";
        });
    
    // StatisticsEngine class
    py::class_<StatisticsEngine>(m, "StatisticsEngine")
        .def(py::init<>())
        .def_static("calculate", &StatisticsEngine::calculate,
                   py::arg("df"), py::arg("column_name"),
                   "Calculate basic statistics for a column")
        .def_static("calculate_range", &StatisticsEngine::calculate_range,
                   py::arg("df"), py::arg("column_name"),
                   py::arg("start_index"), py::arg("end_index"),
                   "Calculate statistics for a column range")
        .def_static("calculate_with_threshold", &StatisticsEngine::calculate_with_threshold,
                   py::arg("df"), py::arg("column_name"),
                   py::arg("time_column"), py::arg("threshold"),
                   "Calculate statistics with threshold analysis")
        .def_static("calculate_rolling", &StatisticsEngine::calculate_rolling,
                   py::arg("df"), py::arg("column_name"), py::arg("window_size"),
                   "Calculate rolling statistics")
        .def_static("percentile", &StatisticsEngine::percentile,
                   py::arg("df"), py::arg("column_name"), py::arg("percentile"),
                   "Calculate percentile value")
        .def_static("histogram", &StatisticsEngine::histogram,
                   py::arg("df"), py::arg("column_name"), py::arg("num_bins"),
                   "Calculate histogram bins")
        // NEW: MPAI Streaming Statistics
        .def_static("calculate_streaming", &StatisticsEngine::calculate_streaming,
                   py::arg("reader"), py::arg("column_name"),
                   py::arg("start_row") = 0, py::arg("row_count") = 0,
                   "Calculate statistics from MPAI file (streaming, low RAM)")
        .def_static("calculate_time_range_streaming", &StatisticsEngine::calculate_time_range_streaming,
                   py::arg("reader"), py::arg("column_name"),
                   py::arg("time_column"), py::arg("start_time"), py::arg("end_time"),
                   "Calculate statistics for time range (MPAI streaming)")
        // NEW: Arrow Compute Statistics (20-30x faster!)
        .def_static("calculate_arrow",
                   [](py::array_t<double> data) {
                       py::buffer_info buf = data.request();
                       const double* ptr = static_cast<const double*>(buf.ptr);
                       std::vector<double> vec(ptr, ptr + buf.size);
                       return StatisticsEngine::calculate_arrow(vec);
                   },
                   py::arg("data"),
                   "Calculate statistics using Arrow Compute (SIMD-optimized, 20-30x faster!)\n"
                   "Falls back to native SIMD if Arrow not available.")
        .def_static("calculate_chunk_arrow",
                   [](py::array_t<double> chunk_data) {
                       py::buffer_info buf = chunk_data.request();
                       const double* ptr = static_cast<const double*>(buf.ptr);
                       std::vector<double> vec(ptr, ptr + buf.size);
                       return StatisticsEngine::calculate_chunk_arrow(vec);
                   },
                   py::arg("chunk_data"),
                   "Calculate statistics for MPAI chunk using Arrow Compute")
        .def_static("mean_arrow",
                   [](py::array_t<double> data) {
                       py::buffer_info buf = data.request();
                       const double* ptr = static_cast<const double*>(buf.ptr);
                       std::vector<double> vec(ptr, ptr + buf.size);
                       return StatisticsEngine::mean_arrow(vec);
                   },
                   py::arg("data"),
                   "Fast mean calculation using Arrow Compute")
        .def_static("stddev_arrow",
                   [](py::array_t<double> data) {
                       py::buffer_info buf = data.request();
                       const double* ptr = static_cast<const double*>(buf.ptr);
                       std::vector<double> vec(ptr, ptr + buf.size);
                       return StatisticsEngine::stddev_arrow(vec);
                   },
                   py::arg("data"),
                   "Fast standard deviation using Arrow Compute")
        .def_static("minmax_arrow",
                   [](py::array_t<double> data) {
                       py::buffer_info buf = data.request();
                       const double* ptr = static_cast<const double*>(buf.ptr);
                       std::vector<double> vec(ptr, ptr + buf.size);
                       double min, max;
                       StatisticsEngine::minmax_arrow(vec, min, max);
                       return py::make_tuple(min, max);
                   },
                   py::arg("data"),
                   "Fast min/max using Arrow Compute\nReturns: (min, max) tuple");

    // ===== DOWNSAMPLE (STREAMING) =====
    m.def("downsample_minmax_streaming",
        [](timegraph::mpai::MpaiReader& reader,
           const std::string& time_col,
           const std::string& signal_col,
           uint64_t max_points,
           double warning_min,
           double warning_max) {
            auto result = timegraph::downsample_minmax_streaming(
                reader, time_col, signal_col, max_points, warning_min, warning_max);
            py::dict d;
            d["time"] = std::move(result.time);
            d["value"] = std::move(result.value);
            return d;
        },
        py::arg("reader"),
        py::arg("time_column"),
        py::arg("signal_column"),
        py::arg("max_points") = 100000,
        py::arg("warning_min") = std::numeric_limits<double>::quiet_NaN(),
        py::arg("warning_max") = std::numeric_limits<double>::quiet_NaN(),
        "Downsample MPAI column with first/min/max (+optional threshold) per bucket, streaming/low RAM");
    
    // ===== LIMIT VIOLATION ENGINE =====
    
    // ViolationSegment struct
    py::class_<ViolationSegment>(m, "ViolationSegment")
        .def(py::init<uint64_t, uint64_t, double, double, double, double>(),
             py::arg("start_index"), py::arg("end_index"),
             py::arg("start_time"), py::arg("end_time"),
             py::arg("min_value"), py::arg("max_value"))
        .def_readonly("start_index", &ViolationSegment::start_index)
        .def_readonly("end_index", &ViolationSegment::end_index)
        .def_readonly("start_time", &ViolationSegment::start_time)
        .def_readonly("end_time", &ViolationSegment::end_time)
        .def_readonly("min_value", &ViolationSegment::min_value)
        .def_readonly("max_value", &ViolationSegment::max_value)
        .def("__repr__", [](const ViolationSegment& s) {
            return "<ViolationSegment [" + std::to_string(s.start_time) + 
                   ", " + std::to_string(s.end_time) + 
                   "] range=[" + std::to_string(s.min_value) + 
                   ", " + std::to_string(s.max_value) + "]>";
        });
    
    // LimitViolationResult struct
    py::class_<LimitViolationResult>(m, "LimitViolationResult")
        .def(py::init<>())
        .def_readonly("violations", &LimitViolationResult::violations)
        .def_readonly("total_violation_points", &LimitViolationResult::total_violation_points)
        .def_readonly("total_data_points", &LimitViolationResult::total_data_points)
        .def_readonly("min_violation_value", &LimitViolationResult::min_violation_value)
        .def_readonly("max_violation_value", &LimitViolationResult::max_violation_value)
        .def("__repr__", [](const LimitViolationResult& r) {
            return "<LimitViolationResult violations=" + 
                   std::to_string(r.violations.size()) + 
                   " total_points=" + std::to_string(r.total_data_points) +
                   " violation_points=" + std::to_string(r.total_violation_points) + ">";
        });
    
    // LimitViolationEngine class
    py::class_<LimitViolationEngine>(m, "LimitViolationEngine")
        .def(py::init<>())
        .def("calculate_violations_streaming",
             [](LimitViolationEngine& engine,
                timegraph::mpai::MpaiReader& reader,
                const std::string& signal_name,
                const std::string& time_column,
                double warning_min,
                double warning_max,
                uint64_t start_row,
                uint64_t row_count) {
                 try {
                     return engine.calculate_violations_streaming(
                         reader, signal_name, time_column,
                         warning_min, warning_max,
                         start_row, row_count
                     );
                 } catch (const std::exception& e) {
                     // Convert C++ exception to Python RuntimeError
                     throw std::runtime_error(
                         std::string("C++ Limit Violation Error: ") + e.what()
                     );
                 }
             },
             py::arg("reader"), py::arg("signal_name"), py::arg("time_column"),
             py::arg("warning_min"), py::arg("warning_max"),
             py::arg("start_row") = 0, py::arg("row_count") = 0,
             "Find all limit violations in MPAI data using streaming (FAST, checks ALL data)")
        .def("calculate_violations_arrays", 
             [](LimitViolationEngine& engine,
                py::array_t<double> signal_data,
                py::array_t<double> time_data,
                double warning_min,
                double warning_max) {
                 // Get numpy array pointers
                 py::buffer_info signal_buf = signal_data.request();
                 py::buffer_info time_buf = time_data.request();
                 
                 if (signal_buf.ndim != 1 || time_buf.ndim != 1) {
                     throw std::runtime_error("Arrays must be 1-dimensional");
                 }
                 
                 if (signal_buf.size != time_buf.size) {
                     throw std::runtime_error("Signal and time arrays must have same length");
                 }
                 
                 const double* signal_ptr = static_cast<double*>(signal_buf.ptr);
                 const double* time_ptr = static_cast<double*>(time_buf.ptr);
                 
                 return engine.calculate_violations_arrays(
                     signal_ptr,
                     time_ptr,
                     signal_buf.size,
                     warning_min,
                     warning_max
                 );
             },
             py::arg("signal_data"), py::arg("time_data"),
             py::arg("warning_min"), py::arg("warning_max"),
             "Find all limit violations in NumPy arrays (for small datasets)");
    
    // ===== ARROW INTEGRATION UTILITIES =====
    
    // Check if Arrow is available
    m.def("is_arrow_available", &arrow_utils::is_arrow_available,
          "Check if Arrow Compute is available for SIMD acceleration\n"
          "Returns True if Arrow was compiled in, False otherwise.");
    
    // Version info
    m.def("get_arrow_info", []() {
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
        py::dict info;
#ifdef HAVE_ARROW
        info["available"] = true;
        info["version"] = "PyArrow runtime";
        info["features"] = py::list();
        py::list features = info["features"];
        features.append("compute");
        features.append("zero_copy");
        features.append("simd");
#else
        info["available"] = false;
        info["version"] = "not compiled";
        info["features"] = py::list();
#endif
        return info;
<<<<<<< HEAD
      },
      "Get Arrow integration information");

  // ===== CRITICAL POINTS DETECTION =====

  // CriticalPoint::Type enum
  py::enum_<CriticalPoint::Type>(m, "CriticalPointType")
      .value("LOCAL_MAX", CriticalPoint::Type::LOCAL_MAX)
      .value("LOCAL_MIN", CriticalPoint::Type::LOCAL_MIN)
      .value("SUDDEN_CHANGE", CriticalPoint::Type::SUDDEN_CHANGE)
      .value("LIMIT_VIOLATION", CriticalPoint::Type::LIMIT_VIOLATION)
      .value("INFLECTION", CriticalPoint::Type::INFLECTION)
      .export_values();

  // CriticalPoint struct
  py::class_<CriticalPoint>(m, "CriticalPoint")
      .def(py::init<>())
      .def_readwrite("index", &CriticalPoint::index)
      .def_readwrite("time", &CriticalPoint::time)
      .def_readwrite("value", &CriticalPoint::value)
      .def_readwrite("type", &CriticalPoint::type)
      .def_readwrite("significance", &CriticalPoint::significance)
      .def("__repr__", [](const CriticalPoint &cp) {
        std::string type_str;
        switch (cp.type) {
        case CriticalPoint::Type::LOCAL_MAX:
          type_str = "MAX";
          break;
        case CriticalPoint::Type::LOCAL_MIN:
          type_str = "MIN";
          break;
        case CriticalPoint::Type::SUDDEN_CHANGE:
          type_str = "CHANGE";
          break;
        case CriticalPoint::Type::LIMIT_VIOLATION:
          type_str = "VIOLATION";
          break;
        case CriticalPoint::Type::INFLECTION:
          type_str = "INFLECTION";
          break;
        }
        return "<CriticalPoint " + type_str + " @" + std::to_string(cp.time) +
               " val=" + std::to_string(cp.value) +
               " sig=" + std::to_string(cp.significance) + ">";
      });

  // CriticalPointsConfig struct
  py::class_<CriticalPointsConfig>(m, "CriticalPointsConfig")
      .def(py::init<>())
      .def_readwrite("detect_peaks", &CriticalPointsConfig::detect_peaks)
      .def_readwrite("detect_valleys", &CriticalPointsConfig::detect_valleys)
      .def_readwrite("window_size", &CriticalPointsConfig::window_size)
      .def_readwrite("min_prominence", &CriticalPointsConfig::min_prominence)
      .def_readwrite("detect_sudden_changes",
                     &CriticalPointsConfig::detect_sudden_changes)
      .def_readwrite("change_threshold",
                     &CriticalPointsConfig::change_threshold)
      .def_readwrite("detect_limit_violations",
                     &CriticalPointsConfig::detect_limit_violations)
      .def_readwrite("warning_limits", &CriticalPointsConfig::warning_limits)
      .def_readwrite("error_limits", &CriticalPointsConfig::error_limits)
      .def_readwrite("detect_inflections",
                     &CriticalPointsConfig::detect_inflections)
      .def_readwrite("max_points", &CriticalPointsConfig::max_points)
      .def_readwrite("min_significance",
                     &CriticalPointsConfig::min_significance);

  // CriticalPointsDetector class
  py::class_<CriticalPointsDetector>(m, "CriticalPointsDetector")
      .def(py::init<>())
      .def_static(
          "detect",
          [](py::array_t<double> time_data, py::array_t<double> signal_data,
             const CriticalPointsConfig &config) {
            py::buffer_info time_buf = time_data.request();
            py::buffer_info signal_buf = signal_data.request();

            const double *time_ptr = static_cast<const double *>(time_buf.ptr);
            const double *signal_ptr =
                static_cast<const double *>(signal_buf.ptr);

            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
            std::vector<double> signal_vec(signal_ptr,
                                           signal_ptr + signal_buf.size);

            return CriticalPointsDetector::detect(time_vec, signal_vec, config);
          },
          py::arg("time_data"), py::arg("signal_data"),
          py::arg("config") = CriticalPointsConfig(),
          "Detect critical points (peaks, valleys, sudden changes, limit "
          "violations)")
      .def_static(
          "detect_local_extrema",
          [](py::array_t<double> time_data, py::array_t<double> signal_data,
             size_t window_size, double min_prominence) {
            py::buffer_info time_buf = time_data.request();
            py::buffer_info signal_buf = signal_data.request();

            const double *time_ptr = static_cast<const double *>(time_buf.ptr);
            const double *signal_ptr =
                static_cast<const double *>(signal_buf.ptr);

            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
            std::vector<double> signal_vec(signal_ptr,
                                           signal_ptr + signal_buf.size);

            return CriticalPointsDetector::detect_local_extrema(
                time_vec, signal_vec, window_size, min_prominence);
          },
          py::arg("time_data"), py::arg("signal_data"),
          py::arg("window_size") = 10, py::arg("min_prominence") = 0.0,
          "Detect local extrema (peaks and valleys)");

  // ===== SMART DOWNSAMPLING =====

  // DownsampleResult struct (already exists, add indices field)
  py::class_<DownsampleResult>(m, "DownsampleResult")
      .def(py::init<>())
      .def_readwrite("time", &DownsampleResult::time)
      .def_readwrite("value", &DownsampleResult::value)
      .def_readwrite("indices", &DownsampleResult::indices)
      .def_readwrite("critical_count", &DownsampleResult::critical_count)
      .def("__repr__", [](const DownsampleResult &r) {
        return "<DownsampleResult points=" + std::to_string(r.time.size()) +
               " critical=" + std::to_string(r.critical_count) + ">";
      });

  // LTTB functions
  m.def(
      "downsample_lttb",
      [](py::array_t<double> time_data, py::array_t<double> signal_data,
         size_t max_points) {
        py::buffer_info time_buf = time_data.request();
        py::buffer_info signal_buf = signal_data.request();

        const double *time_ptr = static_cast<const double *>(time_buf.ptr);
        const double *signal_ptr = static_cast<const double *>(signal_buf.ptr);

        std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
        std::vector<double> signal_vec(signal_ptr,
                                       signal_ptr + signal_buf.size);

        return downsample_lttb(time_vec, signal_vec, max_points);
      },
      py::arg("time_data"), py::arg("signal_data"), py::arg("max_points"),
      "LTTB (Largest Triangle Three Buckets) downsampling\n"
      "Preserves visual characteristics while reducing point count");

  m.def(
      "downsample_lttb_with_critical",
      [](py::array_t<double> time_data, py::array_t<double> signal_data,
         size_t max_points, const CriticalPointsConfig &config) {
        py::buffer_info time_buf = time_data.request();
        py::buffer_info signal_buf = signal_data.request();

        const double *time_ptr = static_cast<const double *>(time_buf.ptr);
        const double *signal_ptr = static_cast<const double *>(signal_buf.ptr);

        std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
        std::vector<double> signal_vec(signal_ptr,
                                       signal_ptr + signal_buf.size);

        return downsample_lttb_with_critical(time_vec, signal_vec, max_points,
                                             config);
      },
      py::arg("time_data"), py::arg("signal_data"), py::arg("max_points"),
      py::arg("config") = CriticalPointsConfig(),
      "Smart LTTB downsampling with critical points preservation\n"
      "Ensures no data loss for peaks, valleys, and limit violations");

  m.def(
      "downsample_auto",
      [](py::array_t<double> time_data, py::array_t<double> signal_data,
         size_t screen_width, bool has_limits) {
        py::buffer_info time_buf = time_data.request();
        py::buffer_info signal_buf = signal_data.request();

        const double *time_ptr = static_cast<const double *>(time_buf.ptr);
        const double *signal_ptr = static_cast<const double *>(signal_buf.ptr);

        std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
        std::vector<double> signal_vec(signal_ptr,
                                       signal_ptr + signal_buf.size);

        return downsample_auto(time_vec, signal_vec, screen_width, has_limits);
      },
      py::arg("time_data"), py::arg("signal_data"),
      py::arg("screen_width") = 1920, py::arg("has_limits") = false,
      "Auto-adaptive downsampling strategy\n"
      "Chooses between LTTB, LTTB+Critical, or no downsampling");

  // ===== SMART DOWNSAMPLER (HYBRID LTTB + CRITICAL POINTS) =====

  // SmartDownsampleConfig struct
  py::class_<SmartDownsampleConfig>(m, "SmartDownsampleConfig")
      .def(py::init<>())
      .def_readwrite("target_points", &SmartDownsampleConfig::target_points)
      .def_readwrite("spike_threshold_high",
                     &SmartDownsampleConfig::spike_threshold_high)
      .def_readwrite("spike_threshold_low",
                     &SmartDownsampleConfig::spike_threshold_low)
      .def_readwrite("auto_threshold_sigma",
                     &SmartDownsampleConfig::auto_threshold_sigma)
      .def_readwrite("use_auto_threshold",
                     &SmartDownsampleConfig::use_auto_threshold)
      .def_readwrite("detect_local_extrema",
                     &SmartDownsampleConfig::detect_local_extrema)
      .def_readwrite("extrema_window", &SmartDownsampleConfig::extrema_window)
      .def_readwrite("min_prominence", &SmartDownsampleConfig::min_prominence)
      .def_readwrite("detect_sudden_changes",
                     &SmartDownsampleConfig::detect_sudden_changes)
      .def_readwrite("change_sigma", &SmartDownsampleConfig::change_sigma)
      .def_readwrite("use_lttb", &SmartDownsampleConfig::use_lttb)
      .def_readwrite("lttb_ratio", &SmartDownsampleConfig::lttb_ratio)
      .def_readwrite("simd_chunk_size", &SmartDownsampleConfig::simd_chunk_size)
      .def_readwrite("use_simd", &SmartDownsampleConfig::use_simd)
      .def_readwrite("parallel", &SmartDownsampleConfig::parallel)
      .def_readwrite("dedup_distance", &SmartDownsampleConfig::dedup_distance)
      .def("__repr__", [](const SmartDownsampleConfig &c) {
        return "<SmartDownsampleConfig target=" +
               std::to_string(c.target_points) + " thresh_high=" +
               (std::isnan(c.spike_threshold_high)
                    ? "auto"
                    : std::to_string(c.spike_threshold_high)) +
               " lttb=" + (c.use_lttb ? "true" : "false") + ">";
      });

  // SmartDownsampleResult struct
  py::class_<SmartDownsampleResult>(m, "SmartDownsampleResult")
      .def(py::init<>())
      .def_readonly("x", &SmartDownsampleResult::x)
      .def_readonly("y", &SmartDownsampleResult::y)
      .def_readonly("original_indices",
                    &SmartDownsampleResult::original_indices)
      .def_readonly("input_size", &SmartDownsampleResult::input_size)
      .def_readonly("output_size", &SmartDownsampleResult::output_size)
      .def_readonly("critical_points_count",
                    &SmartDownsampleResult::critical_points_count)
      .def_readonly("lttb_points_count",
                    &SmartDownsampleResult::lttb_points_count)
      .def_readonly("spike_count", &SmartDownsampleResult::spike_count)
      .def_readonly("peak_count", &SmartDownsampleResult::peak_count)
      .def_readonly("valley_count", &SmartDownsampleResult::valley_count)
      .def_readonly("change_count", &SmartDownsampleResult::change_count)
      .def("compression_ratio", &SmartDownsampleResult::compression_ratio)
      .def("is_valid", &SmartDownsampleResult::is_valid)
      .def(
          "to_numpy",
          [](const SmartDownsampleResult &r) {
            // Convert to NumPy arrays for easy Python usage
            py::array_t<double> x_arr(r.x.size());
            py::array_t<double> y_arr(r.y.size());

            auto x_buf = x_arr.request();
            auto y_buf = y_arr.request();

            std::memcpy(x_buf.ptr, r.x.data(), r.x.size() * sizeof(double));
            std::memcpy(y_buf.ptr, r.y.data(), r.y.size() * sizeof(double));

            return py::make_tuple(x_arr, y_arr);
          },
          "Convert result to NumPy arrays (x, y)")
      .def("__repr__", [](const SmartDownsampleResult &r) {
        return "<SmartDownsampleResult " + std::to_string(r.input_size) +
               " -> " + std::to_string(r.output_size) +
               " points (ratio=" + std::to_string(r.compression_ratio()) +
               ", critical=" + std::to_string(r.critical_points_count) +
               ", spikes=" + std::to_string(r.spike_count) + ")>";
      });

  // SmartDownsampler class
  py::class_<SmartDownsampler>(m, "SmartDownsampler")
      .def(py::init<>())
      .def(
          "downsample",
          [](SmartDownsampler &self, py::array_t<double> x_data,
             py::array_t<double> y_data, const SmartDownsampleConfig &config) {
            py::buffer_info x_buf = x_data.request();
            py::buffer_info y_buf = y_data.request();

            if (x_buf.ndim != 1 || y_buf.ndim != 1) {
              throw std::runtime_error("Arrays must be 1-dimensional");
            }
            if (x_buf.size != y_buf.size) {
              throw std::runtime_error("X and Y arrays must have same length");
            }

            const double *x_ptr = static_cast<const double *>(x_buf.ptr);
            const double *y_ptr = static_cast<const double *>(y_buf.ptr);

            return self.downsample(x_ptr, y_ptr, x_buf.size, config);
          },
          py::arg("x_data"), py::arg("y_data"),
          py::arg("config") = SmartDownsampleConfig(),
          "Downsample data with smart hybrid algorithm\n\n"
          "Args:\n"
          "    x_data: Time/X values (NumPy array)\n"
          "    y_data: Signal/Y values (NumPy array)\n"
          "    config: SmartDownsampleConfig (optional)\n\n"
          "Returns:\n"
          "    SmartDownsampleResult with downsampled x, y and statistics")
      .def(
          "downsample_streaming",
          [](SmartDownsampler &self, timegraph::mpai::MpaiReader &reader,
             const std::string &time_column, const std::string &signal_column,
             const SmartDownsampleConfig &config) {
            return self.downsample_streaming(reader, time_column, signal_column,
                                             config);
          },
          py::arg("reader"), py::arg("time_column"), py::arg("signal_column"),
          py::arg("config") = SmartDownsampleConfig(),
          "Streaming downsample from MPAI file (low memory)\n\n"
          "Args:\n"
          "    reader: MpaiReader instance\n"
          "    time_column: Name of time column\n"
          "    signal_column: Name of signal column\n"
          "    config: SmartDownsampleConfig (optional)\n\n"
          "Returns:\n"
          "    SmartDownsampleResult")
      .def_static(
          "quick_downsample",
          [](py::array_t<double> x_data, py::array_t<double> y_data,
             size_t target_points, py::object threshold) {
            py::buffer_info x_buf = x_data.request();
            py::buffer_info y_buf = y_data.request();

            const double *x_ptr = static_cast<const double *>(x_buf.ptr);
            const double *y_ptr = static_cast<const double *>(y_buf.ptr);

            std::vector<double> x_vec(x_ptr, x_ptr + x_buf.size);
            std::vector<double> y_vec(y_ptr, y_ptr + y_buf.size);

            std::optional<double> thresh_opt;
            if (!threshold.is_none()) {
              thresh_opt = threshold.cast<double>();
            }

            return SmartDownsampler::quick_downsample(
                x_vec, y_vec, target_points, thresh_opt);
          },
          py::arg("x_data"), py::arg("y_data"), py::arg("target_points") = 4000,
          py::arg("threshold") = py::none(),
          "Quick downsample with sensible defaults\n\n"
          "Args:\n"
          "    x_data: Time values (NumPy array)\n"
          "    y_data: Signal values (NumPy array)\n"
          "    target_points: Target output size (default: 4000)\n"
          "    threshold: Optional spike threshold (None = auto)\n\n"
          "Returns:\n"
          "    SmartDownsampleResult");

  // Convenience function - Python-friendly interface
  m.def(
      "smart_downsample",
      [](py::array_t<double> x_data, py::array_t<double> y_data,
         size_t target_points, py::object threshold_high,
         py::object threshold_low) {
        py::buffer_info x_buf = x_data.request();
        py::buffer_info y_buf = y_data.request();

        if (x_buf.ndim != 1 || y_buf.ndim != 1) {
          throw std::runtime_error("Arrays must be 1-dimensional");
        }
        if (x_buf.size != y_buf.size) {
          throw std::runtime_error("X and Y arrays must have same length");
        }

        const double *x_ptr = static_cast<const double *>(x_buf.ptr);
        const double *y_ptr = static_cast<const double *>(y_buf.ptr);

        std::vector<double> x_vec(x_ptr, x_ptr + x_buf.size);
        std::vector<double> y_vec(y_ptr, y_ptr + y_buf.size);

        double th_high = threshold_high.is_none()
                             ? std::numeric_limits<double>::quiet_NaN()
                             : threshold_high.cast<double>();
        double th_low = threshold_low.is_none()
                            ? std::numeric_limits<double>::quiet_NaN()
                            : threshold_low.cast<double>();

        return smart_downsample(x_vec, y_vec, target_points, th_high, th_low);
      },
      py::arg("x_data"), py::arg("y_data"), py::arg("target_points") = 4000,
      py::arg("threshold_high") = py::none(),
      py::arg("threshold_low") = py::none(),
      R"doc(
=======
    }, "Get Arrow integration information");
    
    // ===== CRITICAL POINTS DETECTION =====
    
    // CriticalPoint::Type enum
    py::enum_<CriticalPoint::Type>(m, "CriticalPointType")
        .value("LOCAL_MAX", CriticalPoint::Type::LOCAL_MAX)
        .value("LOCAL_MIN", CriticalPoint::Type::LOCAL_MIN)
        .value("SUDDEN_CHANGE", CriticalPoint::Type::SUDDEN_CHANGE)
        .value("LIMIT_VIOLATION", CriticalPoint::Type::LIMIT_VIOLATION)
        .value("INFLECTION", CriticalPoint::Type::INFLECTION)
        .export_values();
    
    // CriticalPoint struct
    py::class_<CriticalPoint>(m, "CriticalPoint")
        .def(py::init<>())
        .def_readwrite("index", &CriticalPoint::index)
        .def_readwrite("time", &CriticalPoint::time)
        .def_readwrite("value", &CriticalPoint::value)
        .def_readwrite("type", &CriticalPoint::type)
        .def_readwrite("significance", &CriticalPoint::significance)
        .def("__repr__", [](const CriticalPoint& cp) {
            std::string type_str;
            switch (cp.type) {
                case CriticalPoint::Type::LOCAL_MAX: type_str = "MAX"; break;
                case CriticalPoint::Type::LOCAL_MIN: type_str = "MIN"; break;
                case CriticalPoint::Type::SUDDEN_CHANGE: type_str = "CHANGE"; break;
                case CriticalPoint::Type::LIMIT_VIOLATION: type_str = "VIOLATION"; break;
                case CriticalPoint::Type::INFLECTION: type_str = "INFLECTION"; break;
            }
            return "<CriticalPoint " + type_str + " @" + 
                   std::to_string(cp.time) + " val=" + 
                   std::to_string(cp.value) + " sig=" + 
                   std::to_string(cp.significance) + ">";
        });
    
    // CriticalPointsConfig struct
    py::class_<CriticalPointsConfig>(m, "CriticalPointsConfig")
        .def(py::init<>())
        .def_readwrite("detect_peaks", &CriticalPointsConfig::detect_peaks)
        .def_readwrite("detect_valleys", &CriticalPointsConfig::detect_valleys)
        .def_readwrite("window_size", &CriticalPointsConfig::window_size)
        .def_readwrite("min_prominence", &CriticalPointsConfig::min_prominence)
        .def_readwrite("detect_sudden_changes", &CriticalPointsConfig::detect_sudden_changes)
        .def_readwrite("change_threshold", &CriticalPointsConfig::change_threshold)
        .def_readwrite("detect_limit_violations", &CriticalPointsConfig::detect_limit_violations)
        .def_readwrite("warning_limits", &CriticalPointsConfig::warning_limits)
        .def_readwrite("error_limits", &CriticalPointsConfig::error_limits)
        .def_readwrite("detect_inflections", &CriticalPointsConfig::detect_inflections)
        .def_readwrite("max_points", &CriticalPointsConfig::max_points)
        .def_readwrite("min_significance", &CriticalPointsConfig::min_significance);
    
    // CriticalPointsDetector class
    py::class_<CriticalPointsDetector>(m, "CriticalPointsDetector")
        .def(py::init<>())
        .def_static("detect",
                   [](py::array_t<double> time_data,
                      py::array_t<double> signal_data,
                      const CriticalPointsConfig& config) {
                       py::buffer_info time_buf = time_data.request();
                       py::buffer_info signal_buf = signal_data.request();
                       
                       const double* time_ptr = static_cast<const double*>(time_buf.ptr);
                       const double* signal_ptr = static_cast<const double*>(signal_buf.ptr);
                       
                       std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
                       std::vector<double> signal_vec(signal_ptr, signal_ptr + signal_buf.size);
                       
                       return CriticalPointsDetector::detect(time_vec, signal_vec, config);
                   },
                   py::arg("time_data"), py::arg("signal_data"),
                   py::arg("config") = CriticalPointsConfig(),
                   "Detect critical points (peaks, valleys, sudden changes, limit violations)")
        .def_static("detect_local_extrema",
                   [](py::array_t<double> time_data,
                      py::array_t<double> signal_data,
                      size_t window_size,
                      double min_prominence) {
                       py::buffer_info time_buf = time_data.request();
                       py::buffer_info signal_buf = signal_data.request();
                       
                       const double* time_ptr = static_cast<const double*>(time_buf.ptr);
                       const double* signal_ptr = static_cast<const double*>(signal_buf.ptr);
                       
                       std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
                       std::vector<double> signal_vec(signal_ptr, signal_ptr + signal_buf.size);
                       
                       return CriticalPointsDetector::detect_local_extrema(
                           time_vec, signal_vec, window_size, min_prominence
                       );
                   },
                   py::arg("time_data"), py::arg("signal_data"),
                   py::arg("window_size") = 10, py::arg("min_prominence") = 0.0,
                   "Detect local extrema (peaks and valleys)");
    
    // ===== SMART DOWNSAMPLING =====
    
    // DownsampleResult struct (already exists, add indices field)
    py::class_<DownsampleResult>(m, "DownsampleResult")
        .def(py::init<>())
        .def_readwrite("time", &DownsampleResult::time)
        .def_readwrite("value", &DownsampleResult::value)
        .def_readwrite("indices", &DownsampleResult::indices)
        .def_readwrite("critical_count", &DownsampleResult::critical_count)
        .def("__repr__", [](const DownsampleResult& r) {
            return "<DownsampleResult points=" + std::to_string(r.time.size()) +
                   " critical=" + std::to_string(r.critical_count) + ">";
        });
    
    // LTTB functions
    m.def("downsample_lttb",
        [](py::array_t<double> time_data,
           py::array_t<double> signal_data,
           size_t max_points) {
            py::buffer_info time_buf = time_data.request();
            py::buffer_info signal_buf = signal_data.request();
            
            const double* time_ptr = static_cast<const double*>(time_buf.ptr);
            const double* signal_ptr = static_cast<const double*>(signal_buf.ptr);
            
            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
            std::vector<double> signal_vec(signal_ptr, signal_ptr + signal_buf.size);
            
            return downsample_lttb(time_vec, signal_vec, max_points);
        },
        py::arg("time_data"), py::arg("signal_data"), py::arg("max_points"),
        "LTTB (Largest Triangle Three Buckets) downsampling\n"
        "Preserves visual characteristics while reducing point count");
    
    m.def("downsample_lttb_with_critical",
        [](py::array_t<double> time_data,
           py::array_t<double> signal_data,
           size_t max_points,
           const CriticalPointsConfig& config) {
            py::buffer_info time_buf = time_data.request();
            py::buffer_info signal_buf = signal_data.request();
            
            const double* time_ptr = static_cast<const double*>(time_buf.ptr);
            const double* signal_ptr = static_cast<const double*>(signal_buf.ptr);
            
            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
            std::vector<double> signal_vec(signal_ptr, signal_ptr + signal_buf.size);
            
            return downsample_lttb_with_critical(time_vec, signal_vec, max_points, config);
        },
        py::arg("time_data"), py::arg("signal_data"), py::arg("max_points"),
        py::arg("config") = CriticalPointsConfig(),
        "Smart LTTB downsampling with critical points preservation\n"
        "Ensures no data loss for peaks, valleys, and limit violations");
    
    m.def("downsample_auto",
        [](py::array_t<double> time_data,
           py::array_t<double> signal_data,
           size_t screen_width,
           bool has_limits) {
            py::buffer_info time_buf = time_data.request();
            py::buffer_info signal_buf = signal_data.request();
            
            const double* time_ptr = static_cast<const double*>(time_buf.ptr);
            const double* signal_ptr = static_cast<const double*>(signal_buf.ptr);
            
            std::vector<double> time_vec(time_ptr, time_ptr + time_buf.size);
            std::vector<double> signal_vec(signal_ptr, signal_ptr + signal_buf.size);
            
            return downsample_auto(time_vec, signal_vec, screen_width, has_limits);
        },
        py::arg("time_data"), py::arg("signal_data"),
        py::arg("screen_width") = 1920, py::arg("has_limits") = false,
        "Auto-adaptive downsampling strategy\n"
        "Chooses between LTTB, LTTB+Critical, or no downsampling");
    
    // ===== SMART DOWNSAMPLER (HYBRID LTTB + CRITICAL POINTS) =====
    
    // SmartDownsampleConfig struct
    py::class_<SmartDownsampleConfig>(m, "SmartDownsampleConfig")
        .def(py::init<>())
        .def_readwrite("target_points", &SmartDownsampleConfig::target_points)
        .def_readwrite("spike_threshold_high", &SmartDownsampleConfig::spike_threshold_high)
        .def_readwrite("spike_threshold_low", &SmartDownsampleConfig::spike_threshold_low)
        .def_readwrite("auto_threshold_sigma", &SmartDownsampleConfig::auto_threshold_sigma)
        .def_readwrite("use_auto_threshold", &SmartDownsampleConfig::use_auto_threshold)
        .def_readwrite("detect_local_extrema", &SmartDownsampleConfig::detect_local_extrema)
        .def_readwrite("extrema_window", &SmartDownsampleConfig::extrema_window)
        .def_readwrite("min_prominence", &SmartDownsampleConfig::min_prominence)
        .def_readwrite("detect_sudden_changes", &SmartDownsampleConfig::detect_sudden_changes)
        .def_readwrite("change_sigma", &SmartDownsampleConfig::change_sigma)
        .def_readwrite("use_lttb", &SmartDownsampleConfig::use_lttb)
        .def_readwrite("lttb_ratio", &SmartDownsampleConfig::lttb_ratio)
        .def_readwrite("simd_chunk_size", &SmartDownsampleConfig::simd_chunk_size)
        .def_readwrite("use_simd", &SmartDownsampleConfig::use_simd)
        .def_readwrite("parallel", &SmartDownsampleConfig::parallel)
        .def_readwrite("dedup_distance", &SmartDownsampleConfig::dedup_distance)
        .def("__repr__", [](const SmartDownsampleConfig& c) {
            return "<SmartDownsampleConfig target=" + std::to_string(c.target_points) +
                   " thresh_high=" + (std::isnan(c.spike_threshold_high) ? "auto" : std::to_string(c.spike_threshold_high)) +
                   " lttb=" + (c.use_lttb ? "true" : "false") + ">";
        });
    
    // SmartDownsampleResult struct
    py::class_<SmartDownsampleResult>(m, "SmartDownsampleResult")
        .def(py::init<>())
        .def_readonly("x", &SmartDownsampleResult::x)
        .def_readonly("y", &SmartDownsampleResult::y)
        .def_readonly("original_indices", &SmartDownsampleResult::original_indices)
        .def_readonly("input_size", &SmartDownsampleResult::input_size)
        .def_readonly("output_size", &SmartDownsampleResult::output_size)
        .def_readonly("critical_points_count", &SmartDownsampleResult::critical_points_count)
        .def_readonly("lttb_points_count", &SmartDownsampleResult::lttb_points_count)
        .def_readonly("spike_count", &SmartDownsampleResult::spike_count)
        .def_readonly("peak_count", &SmartDownsampleResult::peak_count)
        .def_readonly("valley_count", &SmartDownsampleResult::valley_count)
        .def_readonly("change_count", &SmartDownsampleResult::change_count)
        .def("compression_ratio", &SmartDownsampleResult::compression_ratio)
        .def("is_valid", &SmartDownsampleResult::is_valid)
        .def("to_numpy", [](const SmartDownsampleResult& r) {
            // Convert to NumPy arrays for easy Python usage
            py::array_t<double> x_arr(r.x.size());
            py::array_t<double> y_arr(r.y.size());
            
            auto x_buf = x_arr.request();
            auto y_buf = y_arr.request();
            
            std::memcpy(x_buf.ptr, r.x.data(), r.x.size() * sizeof(double));
            std::memcpy(y_buf.ptr, r.y.data(), r.y.size() * sizeof(double));
            
            return py::make_tuple(x_arr, y_arr);
        }, "Convert result to NumPy arrays (x, y)")
        .def("__repr__", [](const SmartDownsampleResult& r) {
            return "<SmartDownsampleResult " + std::to_string(r.input_size) + 
                   " -> " + std::to_string(r.output_size) + 
                   " points (ratio=" + std::to_string(r.compression_ratio()) +
                   ", critical=" + std::to_string(r.critical_points_count) +
                   ", spikes=" + std::to_string(r.spike_count) + ")>";
        });
    
    // SmartDownsampler class
    py::class_<SmartDownsampler>(m, "SmartDownsampler")
        .def(py::init<>())
        .def("downsample",
            [](SmartDownsampler& self,
               py::array_t<double> x_data,
               py::array_t<double> y_data,
               const SmartDownsampleConfig& config) {
                py::buffer_info x_buf = x_data.request();
                py::buffer_info y_buf = y_data.request();
                
                if (x_buf.ndim != 1 || y_buf.ndim != 1) {
                    throw std::runtime_error("Arrays must be 1-dimensional");
                }
                if (x_buf.size != y_buf.size) {
                    throw std::runtime_error("X and Y arrays must have same length");
                }
                
                const double* x_ptr = static_cast<const double*>(x_buf.ptr);
                const double* y_ptr = static_cast<const double*>(y_buf.ptr);
                
                return self.downsample(x_ptr, y_ptr, x_buf.size, config);
            },
            py::arg("x_data"), py::arg("y_data"),
            py::arg("config") = SmartDownsampleConfig(),
            "Downsample data with smart hybrid algorithm\n\n"
            "Args:\n"
            "    x_data: Time/X values (NumPy array)\n"
            "    y_data: Signal/Y values (NumPy array)\n"
            "    config: SmartDownsampleConfig (optional)\n\n"
            "Returns:\n"
            "    SmartDownsampleResult with downsampled x, y and statistics")
        .def("downsample_streaming",
            [](SmartDownsampler& self,
               timegraph::mpai::MpaiReader& reader,
               const std::string& time_column,
               const std::string& signal_column,
               const SmartDownsampleConfig& config) {
                return self.downsample_streaming(reader, time_column, signal_column, config);
            },
            py::arg("reader"), py::arg("time_column"), py::arg("signal_column"),
            py::arg("config") = SmartDownsampleConfig(),
            "Streaming downsample from MPAI file (low memory)\n\n"
            "Args:\n"
            "    reader: MpaiReader instance\n"
            "    time_column: Name of time column\n"
            "    signal_column: Name of signal column\n"
            "    config: SmartDownsampleConfig (optional)\n\n"
            "Returns:\n"
            "    SmartDownsampleResult")
        .def_static("quick_downsample",
            [](py::array_t<double> x_data,
               py::array_t<double> y_data,
               size_t target_points,
               py::object threshold) {
                py::buffer_info x_buf = x_data.request();
                py::buffer_info y_buf = y_data.request();
                
                const double* x_ptr = static_cast<const double*>(x_buf.ptr);
                const double* y_ptr = static_cast<const double*>(y_buf.ptr);
                
                std::vector<double> x_vec(x_ptr, x_ptr + x_buf.size);
                std::vector<double> y_vec(y_ptr, y_ptr + y_buf.size);
                
                std::optional<double> thresh_opt;
                if (!threshold.is_none()) {
                    thresh_opt = threshold.cast<double>();
                }
                
                return SmartDownsampler::quick_downsample(x_vec, y_vec, target_points, thresh_opt);
            },
            py::arg("x_data"), py::arg("y_data"),
            py::arg("target_points") = 4000,
            py::arg("threshold") = py::none(),
            "Quick downsample with sensible defaults\n\n"
            "Args:\n"
            "    x_data: Time values (NumPy array)\n"
            "    y_data: Signal values (NumPy array)\n"
            "    target_points: Target output size (default: 4000)\n"
            "    threshold: Optional spike threshold (None = auto)\n\n"
            "Returns:\n"
            "    SmartDownsampleResult");
    
    // Convenience function - Python-friendly interface
    m.def("smart_downsample",
        [](py::array_t<double> x_data,
           py::array_t<double> y_data,
           size_t target_points,
           py::object threshold_high,
           py::object threshold_low) {
            py::buffer_info x_buf = x_data.request();
            py::buffer_info y_buf = y_data.request();
            
            if (x_buf.ndim != 1 || y_buf.ndim != 1) {
                throw std::runtime_error("Arrays must be 1-dimensional");
            }
            if (x_buf.size != y_buf.size) {
                throw std::runtime_error("X and Y arrays must have same length");
            }
            
            const double* x_ptr = static_cast<const double*>(x_buf.ptr);
            const double* y_ptr = static_cast<const double*>(y_buf.ptr);
            
            std::vector<double> x_vec(x_ptr, x_ptr + x_buf.size);
            std::vector<double> y_vec(y_ptr, y_ptr + y_buf.size);
            
            double th_high = threshold_high.is_none() ? 
                std::numeric_limits<double>::quiet_NaN() : threshold_high.cast<double>();
            double th_low = threshold_low.is_none() ? 
                std::numeric_limits<double>::quiet_NaN() : threshold_low.cast<double>();
            
            return smart_downsample(x_vec, y_vec, target_points, th_high, th_low);
        },
        py::arg("x_data"),
        py::arg("y_data"),
        py::arg("target_points") = 4000,
        py::arg("threshold_high") = py::none(),
        py::arg("threshold_low") = py::none(),
        R"doc(
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
Smart downsampling with hybrid LTTB + Critical Points algorithm.

This function reduces large time series data to a target size while:
1. Preserving ALL spikes above/below thresholds (motor fault detection)
2. Keeping local peaks and valleys for visual accuracy
3. Using LTTB for smooth trend preservation

Args:
    x_data: Time/X values (NumPy array, must be sorted)
    y_data: Signal/Y values (NumPy array)
    target_points: Target output size (default: 4000)
    threshold_high: Upper spike threshold (None = auto-calculate from 3σ)
    threshold_low: Lower spike threshold (None = auto-calculate from -3σ)

Returns:
    SmartDownsampleResult containing:
        - x, y: Downsampled data
        - original_indices: Indices in original data
        - Statistics: spike_count, peak_count, etc.

Example:
    >>> import numpy as np
    >>> import time_graph_cpp as tg
    >>> 
    >>> # Generate 1M data points with spikes
    >>> x = np.arange(1_000_000, dtype=np.float64)
    >>> y = np.sin(x / 10000) + np.random.randn(1_000_000) * 0.1
    >>> y[500000] = 100.0  # Add spike
    >>> 
    >>> # Downsample to 4000 points, preserving spike
    >>> result = tg.smart_downsample(x, y, 4000, threshold_high=50.0)
    >>> print(f"Reduced {result.input_size} -> {result.output_size}")
    >>> print(f"Spikes preserved: {result.spike_count}")
    >>> 
    >>> # Get NumPy arrays
    >>> x_ds, y_ds = result.to_numpy()
)doc");
}
<<<<<<< HEAD
=======


>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
