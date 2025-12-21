#pragma once

namespace timegraph {

/// Version information
struct Version {
    static constexpr int MAJOR = TG_VERSION_MAJOR;
    static constexpr int MINOR = TG_VERSION_MINOR;
    static constexpr int PATCH = TG_VERSION_PATCH;
    
    static const char* get_version_string();
};

} // namespace timegraph

