/**
 * main.cpp - Test runner
 */

#include "test/test_framework.h"
#include "test/cuda_test.cuh"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    printf("CUDA Test Framework\n");
    printf("===================\n\n");
    
    std::string filter = "";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            filter = argv[++i];
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -f FILTER   Run only tests matching filter\n");
            printf("  -h          Show this help\n");
            return 0;
        }
    }
    
    // Initialize CUDA
    test::cuda::init();
    
    // Run tests
    int failures = test::TestRegistry::instance().run_all(filter);
    
    // Cleanup
    test::cuda::cleanup();
    
    return failures > 0 ? 1 : 0;
}
