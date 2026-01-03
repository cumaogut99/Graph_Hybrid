#include "timegraph/core/version.hpp"
#include <sstream>

namespace timegraph {

const char* Version::get_version_string() {
    static std::string version;
    if (version.empty()) {
        std::ostringstream oss;
        oss << MAJOR << "." << MINOR << "." << PATCH;
        version = oss.str();
    }
    return version.c_str();
}

} // namespace timegraph

