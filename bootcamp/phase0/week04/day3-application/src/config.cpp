/**
 * config.cpp - Configuration implementation
 */

#include "app/config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace app {

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -n, --size N      Problem size (default: 1000000)\n");
    printf("  -b, --block N     CUDA block size (default: 256)\n");
    printf("  -i, --iter N      Number of iterations (default: 1)\n");
    printf("  -d, --device N    CUDA device ID (default: 0)\n");
    printf("  -v, --verbose     Verbose output\n");
    printf("  --no-verify       Skip verification\n");
    printf("  -h, --help        Show this help\n");
}

Config Config::parse(int argc, char** argv) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--size") == 0) && i + 1 < argc) {
            config.problem_size = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block") == 0) && i + 1 < argc) {
            config.block_size = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iter") == 0) && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) && i + 1 < argc) {
            config.device_id = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        }
        else if (strcmp(argv[i], "--no-verify") == 0) {
            config.verify = false;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

void Config::print() const {
    printf("Configuration:\n");
    printf("  Problem size: %d\n", problem_size);
    printf("  Block size: %d\n", block_size);
    printf("  Iterations: %d\n", iterations);
    printf("  Device ID: %d\n", device_id);
    printf("  Verify: %s\n", verify ? "yes" : "no");
    printf("\n");
}

}  // namespace app
