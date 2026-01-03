#include "timegraph/data/column.hpp"
#include "timegraph/data/dataframe.hpp"
#include "timegraph/data/mpai_format.hpp"
#include "timegraph/data/mpai_reader.hpp"
#include "timegraph/data/mpai_writer.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace timegraph;
using namespace timegraph::mpai;

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
      py::cast(df) // Keep-alive: prevent DataFrame deletion
  );
}

/// Initialize data-related Python bindings
void init_data_bindings(py::module &m) {

  // ===== ColumnType Enum =====
  py::enum_<ColumnType>(m, "ColumnType", "Column data types")
      .value("FLOAT64", ColumnType::FLOAT64, "64-bit floating point")
      .value("INT64", ColumnType::INT64, "64-bit integer")
      .value("STRING", ColumnType::STRING, "String type")
      .value("DATETIME", ColumnType::DATETIME, "Datetime (int64 timestamp)")
      .export_values();

  // ===== CsvOptions =====
  py::class_<CsvOptions>(m, "CsvOptions", "CSV loading options")
      .def(py::init<>(), "Default constructor")
      .def_readwrite("delimiter", &CsvOptions::delimiter,
                     "Field delimiter (default: ',')")
      .def_readwrite("has_header", &CsvOptions::has_header,
                     "First row is header (default: True)")
      .def_readwrite("skip_rows", &CsvOptions::skip_rows,
                     "Number of rows to skip (default: 0)")
      .def_readwrite("encoding", &CsvOptions::encoding,
                     "File encoding (default: 'utf-8')")
      .def_readwrite("auto_detect_types", &CsvOptions::auto_detect_types,
                     "Auto-detect column types (default: True)")
      .def_readwrite("infer_schema_rows", &CsvOptions::infer_schema_rows,
                     "Rows to scan for type inference (default: 1000)")
      .def("__repr__", [](const CsvOptions &opts) {
        return "<CsvOptions delimiter='" + std::string(1, opts.delimiter) +
               "' has_header=" + (opts.has_header ? "True" : "False") + ">";
      });

  // ===== ExcelOptions =====
  py::class_<ExcelOptions>(m, "ExcelOptions", "Excel loading options")
      .def(py::init<>(), "Default constructor")
      .def_readwrite("sheet_name", &ExcelOptions::sheet_name,
                     "Sheet name (empty = first sheet)")
      .def_readwrite("has_header", &ExcelOptions::has_header,
                     "First row is header (default: True)")
      .def_readwrite("skip_rows", &ExcelOptions::skip_rows,
                     "Number of rows to skip (default: 0)");

  // ===== DataFrame =====
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
      .def("row_count", &DataFrame::row_count, "Get number of rows")
      .def("column_count", &DataFrame::column_count, "Get number of columns")
      .def("column_names", &DataFrame::column_names, "Get list of column names")
      .def("column_type", &DataFrame::column_type, "Get column type",
           py::arg("name"))
      .def("has_column", &DataFrame::has_column, "Check if column exists",
           py::arg("name"))

      // String representation
      .def("__repr__",
           [](const DataFrame &df) {
             return "<DataFrame rows=" + std::to_string(df.row_count()) +
                    " cols=" + std::to_string(df.column_count()) + ">";
           })
      .def("__len__", &DataFrame::row_count, "Number of rows (len(df))");

  // ===== MPAI Format =====

  // DataType enum
  py::enum_<DataType>(m, "DataType", "MPAI data types")
      .value("FLOAT64", DataType::FLOAT64, "64-bit floating point")
      .value("INT64", DataType::INT64, "64-bit signed integer")
      .value("STRING", DataType::STRING, "String (UTF-8)")
      .value("DATETIME", DataType::DATETIME, "Datetime (int64 microseconds)")
      .export_values();

  // CompressionType enum
  py::enum_<CompressionType>(m, "CompressionType", "MPAI compression types")
      .value("NONE", CompressionType::NONE, "No compression")
      .value("ZSTD", CompressionType::ZSTD, "Zstandard compression")
      .value("LZ4", CompressionType::LZ4, "LZ4 compression")
      .export_values();

  // ColumnStatistics struct
  py::class_<ColumnStatistics>(m, "ColumnStatistics",
                               "Pre-computed column statistics")
      .def(py::init<>())
      .def_readwrite("mean", &ColumnStatistics::mean)
      .def_readwrite("std_dev", &ColumnStatistics::std_dev)
      .def_readwrite("min", &ColumnStatistics::min)
      .def_readwrite("max", &ColumnStatistics::max)
      .def_readwrite("median", &ColumnStatistics::median)
      .def_readwrite("q25", &ColumnStatistics::q25)
      .def_readwrite("q75", &ColumnStatistics::q75)
      .def_readwrite("rms", &ColumnStatistics::rms)
      .def("__repr__", [](const ColumnStatistics &stats) {
        return "<ColumnStatistics mean=" + std::to_string(stats.mean) +
               " std=" + std::to_string(stats.std_dev) + ">";
      });

  // ColumnMetadata struct
  py::class_<ColumnMetadata>(m, "ColumnMetadata", "MPAI column metadata")
      .def(py::init<>())
      .def_readwrite("name", &ColumnMetadata::name)
      .def_readwrite("unit", &ColumnMetadata::unit)
      .def_readwrite("data_type", &ColumnMetadata::data_type)
      .def_readwrite("compression", &ColumnMetadata::compression)
      .def_readwrite("chunk_count", &ColumnMetadata::chunk_count)
      .def_readwrite("total_size_bytes", &ColumnMetadata::total_size_bytes)
      .def_readwrite("compressed_size_bytes",
                     &ColumnMetadata::compressed_size_bytes)
      .def_readwrite("statistics", &ColumnMetadata::statistics)
      .def("__repr__", [](const ColumnMetadata &meta) {
        return "<ColumnMetadata name='" + std::string(meta.name) +
               "' chunks=" + std::to_string(meta.chunk_count) + ">";
      });

  // GraphConfig struct
  py::class_<GraphConfig>(m, "GraphConfig", "Graph configuration")
      .def(py::init<>())
      .def_readwrite("graph_id", &GraphConfig::graph_id)
      .def_readwrite("title", &GraphConfig::title)
      .def_readwrite("x_column", &GraphConfig::x_column)
      .def_readwrite("y_column", &GraphConfig::y_column)
      .def_readwrite("x_min", &GraphConfig::x_min)
      .def_readwrite("x_max", &GraphConfig::x_max)
      .def_readwrite("y_min", &GraphConfig::y_min)
      .def_readwrite("y_max", &GraphConfig::y_max)
      .def_readwrite("line_color", &GraphConfig::line_color)
      .def_readwrite("line_width", &GraphConfig::line_width)
      .def_readwrite("visible", &GraphConfig::visible);

  // FilterCondition struct
  py::class_<FilterCondition>(m, "FilterCondition", "Filter condition")
      .def(py::init<>())
      .def_readwrite("column_name", &FilterCondition::column_name)
      .def_readwrite("min_value", &FilterCondition::min_value)
      .def_readwrite("max_value", &FilterCondition::max_value)
      .def_readwrite("enabled", &FilterCondition::enabled);

  // CursorInfo struct
  py::class_<CursorInfo>(m, "CursorInfo", "Cursor information")
      .def(py::init<>())
      .def_readwrite("cursor_id", &CursorInfo::cursor_id)
      .def_readwrite("x_position", &CursorInfo::x_position)
      .def_readwrite("y_position", &CursorInfo::y_position)
      .def_readwrite("label", &CursorInfo::label)
      .def_readwrite("color", &CursorInfo::color)
      .def_readwrite("visible", &CursorInfo::visible);

  // AnnotationInfo struct
  py::class_<AnnotationInfo>(m, "AnnotationInfo", "Annotation information")
      .def(py::init<>())
      .def_readwrite("annotation_id", &AnnotationInfo::annotation_id)
      .def_readwrite("x_position", &AnnotationInfo::x_position)
      .def_readwrite("y_position", &AnnotationInfo::y_position)
      .def_readwrite("text", &AnnotationInfo::text)
      .def_readwrite("color", &AnnotationInfo::color)
      .def_readwrite("font_size", &AnnotationInfo::font_size);

  // LayoutConfig struct
  py::class_<LayoutConfig>(m, "LayoutConfig", "UI layout configuration")
      .def(py::init<>())
      .def_readwrite("window_width", &LayoutConfig::window_width)
      .def_readwrite("window_height", &LayoutConfig::window_height)
      .def_readwrite("splitter_sizes", &LayoutConfig::splitter_sizes);

  // UserPreferences struct
  py::class_<UserPreferences>(m, "UserPreferences", "User preferences")
      .def(py::init<>())
      .def_readwrite("theme", &UserPreferences::theme)
      .def_readwrite("font_size", &UserPreferences::font_size)
      .def_readwrite("auto_save", &UserPreferences::auto_save)
      .def_readwrite("show_grid", &UserPreferences::show_grid)
      .def_readwrite("show_legend", &UserPreferences::show_legend);

  // ApplicationState struct
  py::class_<ApplicationState>(m, "ApplicationState", "Application state")
      .def(py::init<>())
      .def_readwrite("graph_count", &ApplicationState::graph_count)
      .def_readwrite("filter_count", &ApplicationState::filter_count)
      .def_readwrite("cursor_count", &ApplicationState::cursor_count)
      .def_readwrite("annotation_count", &ApplicationState::annotation_count)
      .def_readwrite("graphs", &ApplicationState::graphs)
      .def_readwrite("filters", &ApplicationState::filters)
      .def_readwrite("cursors", &ApplicationState::cursors)
      .def_readwrite("annotations", &ApplicationState::annotations)
      .def_readwrite("layout", &ApplicationState::layout)
      .def_readwrite("preferences", &ApplicationState::preferences);

  // ===== MpaiWriter =====
  py::class_<MpaiWriter>(m, "MpaiWriter", "MPAI file writer")
      .def(py::init<const std::string &, int>(), "Create MPAI writer",
           py::arg("path"), py::arg("compression_level") = 3)

      .def("write_header", &MpaiWriter::write_header, "Write MPAI header",
           py::arg("row_count"), py::arg("column_count"),
           py::arg("source_file"))

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

  // ===== MpaiReader =====
  py::class_<MpaiReader>(m, "MpaiReader", "MPAI file reader")
      .def(py::init<const std::string &>(), "Create MPAI reader",
           py::arg("path"))

      .def("get_row_count", &MpaiReader::get_row_count, "Get total row count")

      .def("get_column_count", &MpaiReader::get_column_count,
           "Get total column count")

      .def("get_column_names", &MpaiReader::get_column_names,
           "Get list of column names")

      .def("get_column_metadata", &MpaiReader::get_column_metadata,
           "Get column metadata", py::arg("column_name"))

      .def("get_statistics", &MpaiReader::get_statistics,
           "Get pre-computed statistics", py::arg("column_name"))

      .def(
          "load_column_chunk",
          [](MpaiReader &reader, uint32_t col_idx,
             uint32_t chunk_id) -> py::array_t<double> {
            std::vector<double> data =
                reader.load_column_chunk(col_idx, chunk_id);
            return py::array_t<double>(data.size(), data.data());
          },
          "Load column chunk as NumPy array", py::arg("column_index"),
          py::arg("chunk_id"))

      .def(
          "load_column_slice",
          [](MpaiReader &reader, const std::string &col_name,
             uint64_t start_row, uint64_t row_count) -> py::array_t<double> {
            std::vector<double> data =
                reader.load_column_slice(col_name, start_row, row_count);
            return py::array_t<double>(data.size(), data.data());
          },
          "Load column slice as NumPy array", py::arg("column_name"),
          py::arg("start_row"), py::arg("row_count"))

      .def("get_application_state", &MpaiReader::get_application_state,
           "Get application state")

      .def("get_cache_stats", &MpaiReader::get_cache_stats,
           "Get cache statistics (hits, misses, size)")

      .def("get_interpolated_values", &MpaiReader::get_interpolated_values,
           "Get interpolated values for multiple signals at a timestamp",
           py::arg("timestamp"), py::arg("time_col"), py::arg("signal_cols"));
}
