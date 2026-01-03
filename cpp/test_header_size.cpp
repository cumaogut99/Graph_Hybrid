#include <iostream>
#include "timegraph/data/mpai_format.hpp"

int main() {
    using namespace timegraph::mpai;
    
    std::cout << "MpaiHeader size: " << sizeof(MpaiHeader) << " bytes" << std::endl;
    std::cout << "Target size: " << HEADER_SIZE << " bytes" << std::endl;
    std::cout << "Difference: " << (sizeof(MpaiHeader) - HEADER_SIZE) << " bytes" << std::endl;
    
    // Calculate individual field sizes
    std::cout << "\nField sizes:" << std::endl;
    std::cout << "magic: " << sizeof(uint32_t) << std::endl;
    std::cout << "version: " << sizeof(uint32_t) << std::endl;
    std::cout << "header_size: " << sizeof(uint32_t) << std::endl;
    std::cout << "reserved1: " << sizeof(uint32_t) << std::endl;
    std::cout << "file_size: " << sizeof(uint64_t) << std::endl;
    std::cout << "row_count: " << sizeof(uint64_t) << std::endl;
    std::cout << "column_count: " << sizeof(uint32_t) << std::endl;
    std::cout << "chunk_size: " << sizeof(uint32_t) << std::endl;
    std::cout << "creation_time: " << sizeof(uint64_t) << std::endl;
    std::cout << "source_file: 256" << std::endl;
    std::cout << "creator: 64" << std::endl;
    std::cout << "user_name: 64" << std::endl;
    
    int calculated = 4*4 + 3*8 + 2*4 + 8 + 1 + 1 + 2 + 4*4 + 4*8 + 4*4 + 256 + 64 + 64 + 3712;
    std::cout << "\nCalculated total: " << calculated << std::endl;
    
    return 0;
}

