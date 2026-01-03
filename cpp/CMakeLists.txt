cmake_minimum_required(VERSION 3.18)
project(time_graph_cpp VERSION 2.0.0 LANGUAGES CXX)

# ==============================================================================
# Build Configuration
# ==============================================================================

# C++20 standard (required for std::span)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ==============================================================================
# Compiler Flags
# ==============================================================================

# Common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # MSVC: Maximum optimization, AVX2, inline functions
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /arch:AVX2")
        # Enable parallel compilation
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    else()
        # GCC/Clang: Aggressive optimization, native arch, AVX2
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mavx2")
        # Link-time optimization
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    endif()
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    endif()
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# Qt5 (required for QCustomPlot - optional for now)
find_package(Qt5 COMPONENTS Core Widgets PrintSupport)
if(Qt5_FOUND)
    message(STATUS "Qt5 found: ${Qt5_VERSION}")
    message(STATUS "Qt5 dir: ${Qt5_DIR}")
    set(HAVE_QT5 1)
else()
    message(WARNING "Qt5 not found - plotting features will be disabled")
    message(WARNING "To enable Qt5: set Qt5_DIR to Qt5Config.cmake directory")
    set(HAVE_QT5 0)
endif()

# OpenMP (optional, for parallel processing)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(HAVE_OPENMP 1)
else()
    message(WARNING "OpenMP not found - parallel processing will be disabled")
    set(HAVE_OPENMP 0)
endif()

# Pybind11 (for Python bindings)
# Try to find system pybind11 first, if not found use bundled
find_package(pybind11 2.10 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "System pybind11 not found, using bundled version")
    add_subdirectory(thirdparty/pybind11)
else()
    message(STATUS "Using system pybind11: ${pybind11_VERSION}")
endif()

<<<<<<< HEAD
# ZSTD (DISABLED - using raw binary format like DeweSoft)
# DeweSoft does not compress signal data in real-time.
# We follow the same approach: binary format is compact enough (10x smaller than CSV)
set(ZSTD_FOUND FALSE)
message(STATUS "ZSTD: DISABLED (using raw binary format)")
=======
# ZSTD (for MPAI compression)
find_package(ZSTD QUIET)
if(NOT ZSTD_FOUND)
    message(STATUS "ZSTD not found, will try to find library manually")
    find_library(ZSTD_LIBRARY NAMES zstd libzstd)
    if(ZSTD_LIBRARY)
        message(STATUS "Found ZSTD library: ${ZSTD_LIBRARY}")
        set(ZSTD_FOUND TRUE)
    else()
        message(WARNING "ZSTD not found - MPAI format will be disabled")
        message(WARNING "Install ZSTD: pip install zstandard (includes C library)")
        set(ZSTD_FOUND FALSE)
    endif()
endif()
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

# Apache Arrow (for Hybrid MPAI+Arrow compute)
find_package(Arrow QUIET)
if(Arrow_FOUND)
    message(STATUS "Arrow found: ${Arrow_VERSION}")
    message(STATUS "Arrow include: ${Arrow_INCLUDE_DIRS}")
    set(HAVE_ARROW 1)
else()
    message(STATUS "Arrow not found - trying manual detection")
    # Try to find via Python's pyarrow
    execute_process(
        COMMAND python -c "import pyarrow; print(pyarrow.get_include())"
        OUTPUT_VARIABLE ARROW_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND python -c "import pyarrow; print(pyarrow.get_library_dirs()[0] if pyarrow.get_library_dirs() else '')"
        OUTPUT_VARIABLE ARROW_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(ARROW_INCLUDE_DIR AND ARROW_LIB_DIR)
        message(STATUS "Found Arrow via PyArrow: ${ARROW_INCLUDE_DIR}")
        set(Arrow_INCLUDE_DIRS ${ARROW_INCLUDE_DIR})
        set(Arrow_LIB_DIRS ${ARROW_LIB_DIR})
        set(HAVE_ARROW 1)
    else()
        message(WARNING "Arrow not found - Arrow Compute features will be disabled")
        message(WARNING "Install Arrow: pip install pyarrow")
        set(HAVE_ARROW 0)
    endif()
endif()

# ==============================================================================
# Include Directories
# ==============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

# Arrow include directories
if(HAVE_ARROW)
    include_directories(${Arrow_INCLUDE_DIRS})
    link_directories(${Arrow_LIB_DIRS})
endif()

# ==============================================================================
# Source Files
# ==============================================================================

# Data module
set(DATA_SOURCES
    src/data/column.cpp
    src/data/dataframe.cpp
    src/data/csv_loader.cpp
    src/data/mpai_writer.cpp
    src/data/mpai_reader.cpp
    src/data/metadata_builder.cpp
<<<<<<< HEAD
    src/data/lod_reader.cpp
    src/data/lod_writer.cpp
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
)

# Processing module
set(PROCESSING_SOURCES
    src/processing/filter_engine.cpp
    src/processing/statistics_engine.cpp
    src/processing/limit_violation_engine.cpp
    src/processing/simd_filter.cpp
    src/processing/simd_stats.cpp
    src/processing/critical_points.cpp
    src/processing/downsample.cpp
    src/processing/smart_downsampler.cpp
)

# Statistics module (Fast stats with pre-aggregated chunks)
set(STATISTICS_SOURCES
    src/statistics/fast_stats_calculator.cpp
)

# Core module (Sprint 8 - New unified data layer)
set(CORE_SOURCES
    src/core/version.cpp
    src/core/data_cache.cpp
    src/core/data_provider.cpp
)

# Sources module (Data source implementations)
set(SOURCES_MODULE
    src/sources/mpai_source.cpp
)

# All sources
set(SOURCES
    ${DATA_SOURCES}
    ${PROCESSING_SOURCES}
    ${STATISTICS_SOURCES}
    ${CORE_SOURCES}
    ${SOURCES_MODULE}
)

# ==============================================================================
# Python Module (Pybind11)
# ==============================================================================

pybind11_add_module(time_graph_cpp
    ${SOURCES}
    bindings/module.cpp
    bindings/data_bindings_v2.cpp
    bindings/processing_bindings.cpp
    bindings/statistics_bindings.cpp
    bindings/core_bindings.cpp
)

# Link libraries
if(Qt5_FOUND)
    target_link_libraries(time_graph_cpp PRIVATE
        Qt5::Core
        Qt5::Widgets
        Qt5::PrintSupport
    )
endif()

# OpenMP linking
if(OpenMP_CXX_FOUND)
    target_link_libraries(time_graph_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(time_graph_cpp PRIVATE HAVE_OPENMP=1)
endif()

<<<<<<< HEAD
# ZSTD linking - DISABLED
# No compression used, following DeweSoft approach
=======
# ZSTD linking
if(ZSTD_FOUND)
    if(TARGET zstd::libzstd_static OR TARGET zstd::libzstd_shared)
        target_link_libraries(time_graph_cpp PRIVATE zstd::libzstd_static)
    else()
        target_link_libraries(time_graph_cpp PRIVATE ${ZSTD_LIBRARY})
    endif()
    target_compile_definitions(time_graph_cpp PRIVATE HAVE_ZSTD=1)
endif()
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b

# Arrow linking
if(HAVE_ARROW)
    if(TARGET Arrow::arrow_shared)
        target_link_libraries(time_graph_cpp PRIVATE Arrow::arrow_shared)
    else()
        # Manual linking
        find_library(ARROW_LIB NAMES arrow PATHS ${Arrow_LIB_DIRS})
        if(ARROW_LIB)
            target_link_libraries(time_graph_cpp PRIVATE ${ARROW_LIB})
        endif()
    endif()
    target_compile_definitions(time_graph_cpp PRIVATE HAVE_ARROW=1)
endif()

# Compiler definitions
target_compile_definitions(time_graph_cpp PRIVATE
    TG_VERSION_MAJOR=2
    TG_VERSION_MINOR=0
    TG_VERSION_PATCH=0
)

# ==============================================================================
# Installation
# ==============================================================================

# Install Python module to project root (for easy import)
install(TARGETS time_graph_cpp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/..
    RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/..
)

# ==============================================================================
# Testing (Optional)
# ==============================================================================

option(BUILD_TESTS "Build C++ unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    # Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, using bundled version")
        add_subdirectory(thirdparty/googletest)
    endif()
    
    add_subdirectory(tests)
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "======================================")
message(STATUS "Time Graph C++ Configuration Summary")
message(STATUS "======================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Qt5 version:       ${Qt5_VERSION}")
message(STATUS "OpenMP:            ${OpenMP_CXX_FOUND}")
message(STATUS "Pybind11:          ${pybind11_VERSION}")
message(STATUS "ZSTD:              ${ZSTD_FOUND}")
message(STATUS "Arrow:             ${HAVE_ARROW}")
message(STATUS "Build tests:       ${BUILD_TESTS}")
message(STATUS "Install directory: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "======================================")
message(STATUS "")

