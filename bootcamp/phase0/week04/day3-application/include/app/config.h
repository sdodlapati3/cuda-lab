#pragma once
/**
 * config.h - Application configuration
 */

#include <string>

namespace app {

struct Config {
    // Input/output
    std::string input_file;
    std::string output_file;
    
    // Processing parameters
    int problem_size = 1000000;
    int block_size = 256;
    int iterations = 1;
    
    // Flags
    bool verbose = false;
    bool verify = true;
    int device_id = 0;
    
    // Parse from command line
    static Config parse(int argc, char** argv);
    
    // Print configuration
    void print() const;
};

void print_usage(const char* program_name);

}  // namespace app
