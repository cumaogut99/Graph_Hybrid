#include "timegraph/core/version.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for binding functions
void init_data_bindings(py::module &m);
void bind_processing(py::module &m);
void bind_statistics(py::module &m); // Fast statistics calculator

// Core data layer bindings (Sprint 8)
namespace timegraph {
void init_core_bindings(py::module &m);
}

/// Main Python module definition
PYBIND11_MODULE(time_graph_cpp, m) {
  m.doc() = "Time Graph C++ Core Library - High-performance data processing "
            "and visualization";

  // Version information
  m.attr("__version__") = timegraph::Version::get_version_string();
  m.def("get_version", &timegraph::Version::get_version_string,
        "Get library version string");

  // Data bindings (DataFrame, CsvOptions, etc.) - Legacy
  init_data_bindings(m);

  // Processing bindings (Filter, Statistics)
  bind_processing(m);

  // Fast statistics bindings (Pre-aggregated chunks)
  bind_statistics(m);

  // Core data layer bindings (Sprint 8 - New unified API)
  // DataProvider, DataChunk, MpaiDataSource, etc.
  timegraph::init_core_bindings(m);
}
