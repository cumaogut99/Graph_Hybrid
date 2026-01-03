/**
 * LOD Format - Level of Detail Binary Format
 *
 * DeweSoft-style pre-computed downsampled data for instant zoom/pan.
 * Format is compatible with memory-mapped I/O for zero-copy access.
 *
 * Spike-Safe Guarantee:
 * - Every bucket stores both MIN and MAX values
 * - This ensures no data peak is ever lost during visualization
 *
 * Structure:
 * [LodHeader 64 bytes]
 * [Column Names: null-terminated strings]
 * [LodBucket × bucket_count]
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace timegraph {
namespace lod {

// ==============================================================================
// Constants
// ==============================================================================

constexpr uint32_t LOD_MAGIC = 0x444F4C54;     // "TLOD" in little-endian
constexpr uint32_t LOD_VERSION = 0x00010000;   // v1.0.0
constexpr uint32_t LOD_HEADER_SIZE = 64;

// Standard bucket sizes (matching DeweSoft approach)
constexpr uint32_t LOD_BUCKET_100 = 100;       // Level 1: every 100 samples
constexpr uint32_t LOD_BUCKET_10K = 10000;     // Level 2: every 10K samples
constexpr uint32_t LOD_BUCKET_100K = 100000;   // Level 3: every 100K samples

// ==============================================================================
// Header Structure (64 bytes, fixed size)
// ==============================================================================

#pragma pack(push, 1)

struct LodHeader {
    // Identification (8 bytes)
    uint32_t magic;              // "TLOD" (0x444F4C54)
    uint32_t version;            // 0x00010000 (v1.0)
    
    // LOD Configuration (16 bytes)
    uint32_t bucket_size;        // 100, 10000, or 100000
    uint64_t bucket_count;       // Total number of buckets
    uint32_t column_count;       // Number of signal columns (excluding time)
    
    // Data Layout (24 bytes)
    uint64_t names_offset;       // Offset to column names section
    uint64_t data_offset;        // Offset to bucket data
    uint64_t total_row_count;    // Original row count
    
    // Checksums (8 bytes)
    uint32_t header_crc32;
    uint32_t data_crc32;
    
    // Reserved (8 bytes)
    uint8_t reserved[8];
    
    LodHeader() {
        magic = LOD_MAGIC;
        version = LOD_VERSION;
        bucket_size = LOD_BUCKET_100;
        bucket_count = 0;
        column_count = 0;
        names_offset = LOD_HEADER_SIZE;
        data_offset = 0;
        total_row_count = 0;
        header_crc32 = 0;
        data_crc32 = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(LodHeader) == LOD_HEADER_SIZE, "LodHeader must be exactly 64 bytes");

#pragma pack(pop)

// ==============================================================================
// Runtime Structures (not packed, for convenience)
// ==============================================================================

/**
 * Single LOD bucket containing min/max pairs.
 * 
 * Memory Layout per bucket:
 * [time_min][time_max][sig0_min][sig0_max][sig1_min][sig1_max]...
 * 
 * Total doubles per bucket = 2 + 2 * column_count
 */
struct LodBucket {
    double time_min;
    double time_max;
    std::vector<double> signal_min;  // Min value for each signal
    std::vector<double> signal_max;  // Max value for each signal
    
    LodBucket() : time_min(0), time_max(0) {}
    
    explicit LodBucket(uint32_t column_count) 
        : time_min(0), time_max(0),
          signal_min(column_count, 0.0),
          signal_max(column_count, 0.0) {}
};

/**
 * LOD Metadata (in-memory representation)
 */
struct LodMetadata {
    uint32_t bucket_size;
    uint64_t bucket_count;
    uint64_t total_row_count;
    std::vector<std::string> column_names;
    
    LodMetadata() : bucket_size(0), bucket_count(0), total_row_count(0) {}
};

// ==============================================================================
// Helper Functions
// ==============================================================================

/**
 * Get the appropriate LOD level for a given visible sample count.
 * 
 * @param visible_samples Number of samples in the current view
 * @return Recommended bucket size (0 means use raw data)
 */
inline uint32_t get_lod_bucket_size(uint64_t visible_samples) {
    if (visible_samples < 20000) {
        return 0;  // Use raw data
    } else if (visible_samples < 2000000) {
        return LOD_BUCKET_100;  // LOD1
    } else if (visible_samples < 20000000) {
        return LOD_BUCKET_10K;  // LOD2
    } else {
        return LOD_BUCKET_100K; // LOD3
    }
}

/**
 * Calculate doubles per bucket based on column count.
 * Layout: [time_min, time_max, sig0_min, sig0_max, sig1_min, sig1_max, ...]
 */
inline size_t doubles_per_bucket(uint32_t column_count) {
    return 2 + 2 * column_count;  // 2 for time + 2 per signal
}

/**
 * Get LOD filename for a given bucket size.
 */
inline std::string get_lod_filename(uint32_t bucket_size) {
    switch (bucket_size) {
        case LOD_BUCKET_100:   return "lod_100.tlod";
        case LOD_BUCKET_10K:   return "lod_10k.tlod";
        case LOD_BUCKET_100K:  return "lod_100k.tlod";
        default:               return "lod.tlod";
    }
}

} // namespace lod
} // namespace timegraph
