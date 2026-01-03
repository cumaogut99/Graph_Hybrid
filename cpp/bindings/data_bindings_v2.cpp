#include "timegraph/data/column.hpp"
#include "timegraph/data/dataframe.hpp"
<<<<<<< HEAD
#include "timegraph/data/lod_format.hpp"
#include "timegraph/data/lod_reader.hpp"
#include "timegraph/data/lod_writer.hpp"
#include "timegraph/data/mpai_format.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/data/mpai_writer.hpp"
#include "timegraph/processing/expression_engine.hpp"
=======
#include "timegraph/data/mpai_format.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/data/mpai_writer.hpp"
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace timegraph;
using namespace timegraph::mpai;
<<<<<<< HEAD
// Note: NOT using timegraph::lod namespace to avoid std:: conflicts
namespace expr = timegraph::expr; // Alias for expression engine
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

/// Helper: Convert C++ column to NumPy array (zero-copy)
py::array_t<double> get_column_as_numpy(const DataFrame &df,
                                        const std::string &name) {
  // Get const pointer from DataFrame
  const double *data_ptr = df.get_column_ptr_f64(name);
  size_t size = df.row_count();

  // Create NumPy array that references C++ memory
  // CRITICAL: py::cast(df) keeps DataFrame alive as long as array exists
  return py::array_t<double>(
      size,        // Shape
      data_ptr,    // Data pointer
      py::cast(df) // Keep-alive: prevent DataFrame destruction
  );
}

void init_data_bindings(py::module &m) {
  // ==========================================
  // Options
  // ==========================================

  // CsvOptions
  py::class_<CsvOptions>(m, "CsvOptions", "CSV loading options")
      .def(py::init<>(), "Default constructor")
      .def_readwrite("delimiter", &CsvOptions::delimiter)
      .def_readwrite("has_header", &CsvOptions::has_header)
      .def_readwrite("skip_rows", &CsvOptions::skip_rows)
      .def_readwrite("encoding", &CsvOptions::encoding)
      .def_readwrite("auto_detect_types", &CsvOptions::auto_detect_types)
      .def_readwrite("infer_schema_rows", &CsvOptions::infer_schema_rows);

  // ExcelOptions
  py::class_<ExcelOptions>(m, "ExcelOptions", "Excel loading options")
      .def(py::init<>(), "Default constructor")
      .def_readwrite("sheet_name", &ExcelOptions::sheet_name)
      .def_readwrite("has_header", &ExcelOptions::has_header)
      .def_readwrite("skip_rows", &ExcelOptions::skip_rows);

  // ==========================================
  // DataFrame
  // ==========================================
  py::class_<DataFrame>(m, "DataFrame", "Column-oriented data container")
      .def(py::init<>(), "Create empty DataFrame")

      // Factory methods
      .def_static("load_csv", &DataFrame::load_csv,
                  "Load DataFrame from CSV file", py::arg("path"),
                  py::arg("options"), py::return_value_policy::move)
      .def_static("load_excel", &DataFrame::load_excel,
                  "Load DataFrame from Excel file", py::arg("path"),
                  py::arg("options"), py::return_value_policy::move)

      // Column access (zero-copy via NumPy)
      .def("get_column_f64", &get_column_as_numpy,
           "Get Float64 column as NumPy array (zero-copy)", py::arg("name"),
           py::keep_alive<0, 1>()) // Keep DataFrame alive while array exists

      // Metadata
      .def("row_count", &DataFrame::row_count)
      .def("column_count", &DataFrame::column_count)
      .def("column_names", &DataFrame::column_names)
      .def("column_type", &DataFrame::column_type, py::arg("name"))
      .def("has_column", &DataFrame::has_column, py::arg("name"))

      // String representation
      .def("__repr__",
           [](const DataFrame &df) {
             return "<DataFrame rows=" + std::to_string(df.row_count()) +
                    " cols=" + std::to_string(df.column_count()) + ">";
           })
      .def("__len__", &DataFrame::row_count);

  // ==========================================
  // Enums
  // ==========================================
  py::enum_<DataType>(m, "DataType")
      .value("FLOAT64", DataType::FLOAT64)
      .value("INT64", DataType::INT64)
      .value("STRING", DataType::STRING)
      .value("DATETIME", DataType::DATETIME)
      .export_values();

  py::enum_<CompressionType>(m, "CompressionType")
      .value("NONE", CompressionType::NONE)
      .value("ZSTD", CompressionType::ZSTD)
      .value("LZ4", CompressionType::LZ4)
      .value("SNAPPY", CompressionType::SNAPPY)
      .export_values();

  py::enum_<FilterType>(m, "MpaiFilterType")
      .value("RANGE", FilterType::RANGE)
      .value("THRESHOLD", FilterType::THRESHOLD)
      .value("BITMASK", FilterType::BITMASK)
      .export_values();

  py::enum_<NormalizationType>(m, "NormalizationType")
      .value("NONE", NormalizationType::NONE)
      .value("PEAK", NormalizationType::PEAK)
      .value("RMS", NormalizationType::RMS)
      .value("ZSCORE", NormalizationType::ZSCORE)
      .export_values();

  py::enum_<LineStyle>(m, "LineStyle")
      .value("SOLID", LineStyle::SOLID)
      .value("DASHED", LineStyle::DASHED)
      .value("DOTTED", LineStyle::DOTTED)
      .value("DASH_DOT", LineStyle::DASH_DOT)
      .export_values();

  // ==========================================
  // Data Structures
  // ==========================================

  // ColumnStatistics
  py::class_<ColumnStatistics>(m, "MpaiColumnStatistics")
      .def(py::init<>())
      .def_readwrite("mean", &ColumnStatistics::mean)
      .def_readwrite("std_dev", &ColumnStatistics::std_dev)
      .def_readwrite("min", &ColumnStatistics::min)
      .def_readwrite("max", &ColumnStatistics::max)
      .def_readwrite("median", &ColumnStatistics::median)
      .def_readwrite("q25", &ColumnStatistics::q25)
      .def_readwrite("q75", &ColumnStatistics::q75)
      .def_readwrite("rms", &ColumnStatistics::rms)
      .def_readwrite("null_count", &ColumnStatistics::null_count)
      .def_readwrite("start_time", &ColumnStatistics::start_time)
      .def_readwrite("end_time", &ColumnStatistics::end_time)
      .def_readwrite("sample_rate", &ColumnStatistics::sample_rate)
      .def_readwrite("duration", &ColumnStatistics::duration);

  // ChunkInfo
  py::class_<ChunkInfo>(m, "ChunkInfo")
      .def(py::init<>())
      .def_readwrite("offset", &ChunkInfo::offset)
      .def_readwrite("compressed_size", &ChunkInfo::compressed_size)
      .def_readwrite("uncompressed_size", &ChunkInfo::uncompressed_size)
      .def_readwrite("row_count", &ChunkInfo::row_count)
      .def_readwrite("crc32", &ChunkInfo::crc32)
      .def_readwrite("min_value", &ChunkInfo::min_value)
      .def_readwrite("max_value", &ChunkInfo::max_value);

  // ColumnMetadata
  py::class_<ColumnMetadata>(m, "ColumnMetadata")
      .def(py::init<>())
      .def_readwrite("name", &ColumnMetadata::name)
      .def_readwrite("data_type", &ColumnMetadata::data_type)
      .def_readwrite("unit", &ColumnMetadata::unit)
      .def_readwrite("statistics", &ColumnMetadata::statistics)
      .def_readwrite("chunks", &ColumnMetadata::chunks);

  // DataMetadata
  py::class_<DataMetadata>(m, "DataMetadata")
      .def(py::init<>())
      .def_readwrite("columns", &DataMetadata::columns);

  // ==========================================
  // Application State Structures
  // ==========================================

  // SignalConfig
  py::class_<SignalConfig>(m, "SignalConfig")
      .def(py::init<>())
      .def_readwrite("signal_name", &SignalConfig::signal_name)
      // .def_readwrite("color", &SignalConfig::color) // Array binding needs
      // special handling if needed
      .def_readwrite("line_width", &SignalConfig::line_width)
      .def_readwrite("line_style", &SignalConfig::line_style)
      .def_readwrite("visible", &SignalConfig::visible)
      .def_readwrite("normalized", &SignalConfig::normalized)
      .def_readwrite("norm_method", &SignalConfig::norm_method)
      .def_readwrite("y_axis", &SignalConfig::y_axis)
      .def_readwrite("y_min", &SignalConfig::y_min)
      .def_readwrite("y_max", &SignalConfig::y_max)
      .def_readwrite("auto_scale", &SignalConfig::auto_scale);

  // GraphConfig
  py::class_<GraphConfig>(m, "GraphConfig")
      .def(py::init<>())
      .def_readwrite("graph_id", &GraphConfig::graph_id)
      .def_readwrite("title", &GraphConfig::title)
      .def_readwrite("signals", &GraphConfig::signals)
      .def_readwrite("x_min", &GraphConfig::x_min)
      .def_readwrite("x_max", &GraphConfig::x_max)
      .def_readwrite("auto_x_scale", &GraphConfig::auto_x_scale)
      .def_readwrite("x_label", &GraphConfig::x_label)
      .def_readwrite("y_left_label", &GraphConfig::y_left_label)
      .def_readwrite("y_right_label", &GraphConfig::y_right_label)
      .def_readwrite("show_grid", &GraphConfig::show_grid)
      .def_readwrite("show_legend", &GraphConfig::show_legend)
      .def_readwrite("legend_position", &GraphConfig::legend_position);
  // .def_readwrite("background_color", &GraphConfig::background_color);

  // FilterCondition
  py::class_<FilterCondition>(m, "MpaiFilterCondition")
      .def(py::init<>())
      .def_readwrite("column_name", &FilterCondition::column_name)
      .def_readwrite("filter_type", &FilterCondition::filter_type)
      .def_readwrite("min_value", &FilterCondition::min_value)
      .def_readwrite("max_value", &FilterCondition::max_value)
      .def_readwrite("operator_type", &FilterCondition::operator_type)
      .def_readwrite("enabled", &FilterCondition::enabled);

  // CursorInfo
  py::class_<CursorInfo>(m, "CursorInfo")
      .def(py::init<>())
      .def_readwrite("cursor_id", &CursorInfo::cursor_id)
      .def_readwrite("time_position", &CursorInfo::time_position)
      .def_readwrite("visible", &CursorInfo::visible)
      .def_readwrite("label", &CursorInfo::label);

  // AnnotationInfo
  py::class_<AnnotationInfo>(m, "AnnotationInfo")
      .def(py::init<>())
      .def_readwrite("time_start", &AnnotationInfo::time_start)
      .def_readwrite("time_end", &AnnotationInfo::time_end)
      .def_readwrite("text", &AnnotationInfo::text)
      .def_readwrite("type", &AnnotationInfo::type);

  // LayoutConfig
  py::class_<LayoutConfig>(m, "LayoutConfig")
      .def(py::init<>())
      .def_readwrite("subplot_count", &LayoutConfig::subplot_count)
      .def_readwrite("subplot_layout", &LayoutConfig::subplot_layout)
      .def_readwrite("subplot_heights", &LayoutConfig::subplot_heights);

  // UserPreferences
  py::class_<UserPreferences>(m, "UserPreferences")
      .def(py::init<>())
      .def_readwrite("dark_mode", &UserPreferences::dark_mode)
      .def_readwrite("default_color_scheme",
                     &UserPreferences::default_color_scheme)
      .def_readwrite("ui_scale", &UserPreferences::ui_scale)
      .def_readwrite("last_export_path", &UserPreferences::last_export_path);

  // ApplicationState
  py::class_<ApplicationState>(m, "ApplicationState")
      .def(py::init<>())
      .def_readwrite("state_version", &ApplicationState::state_version)
      .def_readwrite("graphs", &ApplicationState::graphs)
      .def_readwrite("filters", &ApplicationState::filters)
      .def_readwrite("cursors", &ApplicationState::cursors)
      .def_readwrite("annotations", &ApplicationState::annotations)
      .def_readwrite("layout", &ApplicationState::layout)
      .def_readwrite("preferences", &ApplicationState::preferences)
      .def_readwrite("last_modified", &ApplicationState::last_modified)
      .def_readwrite("user_name", &ApplicationState::user_name)
      .def_readwrite("notes", &ApplicationState::notes);

  // ==========================================
  // MpaiReader
  // ==========================================
  py::class_<MpaiReader>(m, "MpaiReader")
      .def(py::init<const std::string &, bool>(), py::arg("filename"),
           py::arg("use_mmap") = true)

      // Metadata Accessors
      .def("get_header", &MpaiReader::get_header,
           py::return_value_policy::reference)
      .def("get_data_metadata", &MpaiReader::get_data_metadata,
           py::return_value_policy::reference)
      .def("get_application_state", &MpaiReader::get_application_state,
           py::return_value_policy::reference)

      // Column Metadata
      .def("get_column_metadata",
           static_cast<const ColumnMetadata &(MpaiReader::*)(uint32_t) const>(
               &MpaiReader::get_column_metadata),
           py::return_value_policy::reference)

      // Statistics
      .def("get_statistics", &MpaiReader::get_statistics,
           py::return_value_policy::reference)

      // Data Loading (Heavy)
      .def("load_column_chunk", &MpaiReader::load_column_chunk,
           "Load specific chunk (decompressed)", py::arg("column_index"),
           py::arg("chunk_id"))

      .def("load_column", &MpaiReader::load_column,
           "Load entire column (WARNING: High Memory Usage)",
           py::arg("column_name"))

      .def("load_column_slice", &MpaiReader::load_column_slice,
           "Load partial column data", py::arg("column_name"),
           py::arg("start_row"), py::arg("row_count"))

      // Info
      .def("get_row_count", &MpaiReader::get_row_count)
      .def("get_column_count", &MpaiReader::get_column_count)
      .def("get_column_names", &MpaiReader::get_column_names)
      .def("has_column", &MpaiReader::has_column)
      .def("get_file_size", &MpaiReader::get_file_size)
      .def("get_compression_ratio", &MpaiReader::get_compression_ratio)
      .def("get_memory_usage", &MpaiReader::get_memory_usage)

      .def("get_batch_values", &MpaiReader::get_batch_values,
           "Get interpolated values for multiple signals at a timestamp",
           py::arg("timestamp"), py::arg("time_col"), py::arg("signal_cols"));

  // ==========================================
  // MpaiWriter
  // ==========================================
  py::class_<MpaiWriter>(m, "MpaiWriter")
      .def(py::init<const std::string &, int>(), "Create MPAI writer",
           py::arg("filename"), py::arg("compression_level") = 3)

      .def("write_header", &MpaiWriter::write_header, "Write MPAI header",
           py::arg("row_count"), py::arg("column_count"),
           py::arg("source_file") = "")

      .def("add_column_metadata", &MpaiWriter::add_column_metadata,
           "Add column metadata", py::arg("metadata"))

      .def(
          "write_column_chunk",
          [](MpaiWriter &writer, uint32_t col_idx, uint32_t chunk_id,
             py::bytes data, size_t size, size_t row_count) {
            std::string data_str = data;
            writer.write_column_chunk(col_idx, chunk_id, data_str.data(), size,
                                      row_count);
          },
          "Write column chunk", py::arg("column_index"), py::arg("chunk_id"),
          py::arg("data"), py::arg("size"), py::arg("row_count"))

      .def("write_application_state", &MpaiWriter::write_application_state,
           "Write application state", py::arg("state"))

      .def("finalize", &MpaiWriter::finalize, "Finalize MPAI file");
<<<<<<< HEAD

  // ==========================================
  // LodReader (Spike-Safe LOD Reading)
  // ==========================================
  py::class_<lod::LodReader>(m, "LodReader",
                             "Memory-mapped LOD file reader (spike-safe)")
      .def(py::init<>(), "Create empty LOD reader")

      .def("open", &lod::LodReader::open, "Open LOD file", py::arg("filepath"))
      .def("close", &lod::LodReader::close, "Close file and release resources")
      .def("is_open", &lod::LodReader::is_open, "Check if file is open")

      // Metadata
      .def("get_bucket_size", &lod::LodReader::get_bucket_size)
      .def("get_bucket_count", &lod::LodReader::get_bucket_count)
      .def("get_column_count", &lod::LodReader::get_column_count)
      .def("get_total_row_count", &lod::LodReader::get_total_row_count)
      .def("get_column_names", &lod::LodReader::get_column_names)
      .def("get_column_index", &lod::LodReader::get_column_index,
           py::arg("name"))

      // Time range
      .def("get_time_range", &lod::LodReader::get_time_range,
           "Get time range for buckets (spike-safe)", py::arg("start_bucket"),
           py::arg("count"))

      // Signal range
      .def("get_signal_range", &lod::LodReader::get_signal_range,
           "Get min/max range for signal (spike-safe)", py::arg("column_index"),
           py::arg("start_bucket"), py::arg("count"))

      // Render data (spike-safe: returns 2 points per bucket as tuple)
      .def(
          "get_render_data",
          [](const lod::LodReader &self, uint32_t column_index,
             uint64_t start_bucket, uint64_t count) {
            std::vector<double> out_time;
            std::vector<double> out_values;
            self.get_render_data(column_index, start_bucket, count, out_time,
                                 out_values);
            return std::make_tuple(out_time, out_values);
          },
          "Get downsampled data for rendering (spike-safe)",
          py::arg("column_index"), py::arg("start_bucket"), py::arg("count"))

      // Find bucket range
      .def("find_bucket_range", &lod::LodReader::find_bucket_range,
           "Find bucket range for time range", py::arg("time_start"),
           py::arg("time_end"));

  // ==========================================
  // LodWriter (Streaming LOD Writing)
  // ==========================================
  py::class_<lod::LodWriter>(m, "LodWriter", "Streaming LOD file writer")
      .def(py::init<>(), "Create LOD writer")

      .def("create", &lod::LodWriter::create, "Create new LOD file",
           py::arg("filepath"), py::arg("bucket_size"), py::arg("column_names"),
           py::arg("total_row_count"))

      .def("write_bucket",
           static_cast<bool (lod::LodWriter::*)(
               double, double, const std::vector<double> &,
               const std::vector<double> &)>(&lod::LodWriter::write_bucket),
           "Write a single bucket (spike-safe min/max)", py::arg("time_min"),
           py::arg("time_max"), py::arg("signal_min"), py::arg("signal_max"))

      .def("finalize", &lod::LodWriter::finalize, "Finalize and close file")
      .def("get_bucket_count", &lod::LodWriter::get_bucket_count)
      .def("is_open", &lod::LodWriter::is_open);

  // ==========================================
  // LOD Helper Functions
  // ==========================================
  m.def("get_lod_bucket_size", &lod::get_lod_bucket_size,
        "Get recommended LOD bucket size for visible sample count",
        py::arg("visible_samples"));

  m.def("get_lod_filename", &lod::get_lod_filename,
        "Get LOD filename for bucket size", py::arg("bucket_size"));

  // ==========================================
  // Expression Engine (Calculated Parameters)
  // ==========================================

  // Factory functions for building expression trees from Python
  m.def("expr_column", &expr::make_column, "Create column reference node",
        py::arg("name"));

  m.def("expr_constant", &expr::make_constant, "Create constant value node",
        py::arg("value"));

  m.def("expr_binary", &expr::make_binary,
        "Create binary operation node (+, -, *, /, **, %)", py::arg("left"),
        py::arg("right"), py::arg("op"));

  m.def("expr_unary", &expr::make_unary,
        "Create unary operation node (-, abs, sqrt, sin, cos, ...)",
        py::arg("operand"), py::arg("op"));

  // EvaluationContext
  py::class_<expr::EvaluationContext>(m, "EvaluationContext",
                                      "Context for expression evaluation")
      .def(py::init<>())
      .def("set_column_data", &expr::EvaluationContext::set_column_data,
           "Set column data for evaluation", py::arg("name"), py::arg("data"));

  // ExpressionNode (abstract base - just for type hints)
  py::class_<expr::ExpressionNode, expr::ExprPtr>(m, "ExpressionNode",
                                                  "Base expression node")
      .def("to_string", &expr::ExpressionNode::to_string)
      .def("get_dependencies", &expr::ExpressionNode::get_dependencies);

  // ExpressionEngine
  py::class_<expr::ExpressionEngine>(
      m, "ExpressionEngine",
      "Streaming expression evaluator for calculated parameters")
      .def(py::init<>())
      .def("register_expression", &expr::ExpressionEngine::register_expression,
           "Register expression with name", py::arg("name"), py::arg("expr"))
      .def("evaluate", &expr::ExpressionEngine::evaluate, "Evaluate expression",
           py::arg("name"), py::arg("context"), py::arg("length"))
      .def("get_dependencies", &expr::ExpressionEngine::get_dependencies,
           "Get dependencies for expression", py::arg("name"))
      .def("has_expression", &expr::ExpressionEngine::has_expression)
      .def("get_expression_names",
           &expr::ExpressionEngine::get_expression_names);
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
}
