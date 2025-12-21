/**
 * main.cpp - Application entry point
 */

#include "myproject/myproject.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n SIZE     Problem size (default: 1M)\n");
    printf("  -d DEVICE   CUDA device ID (default: 0)\n");
    printf("  -h          Show help\n");
}

int main(int argc, char** argv) {
    printf("MyProject Application v%d.%d.%d\n",
           myproject::VERSION_MAJOR,
           myproject::VERSION_MINOR,
           myproject::VERSION_PATCH);
    printf("================================\n\n");
    
    int n = 1 << 20;
    int device = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Initialize
    if (!myproject::initialize(device)) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    
    printf("Device: %s\n", myproject::get_device_info());
    printf("Problem size: %d elements\n\n", n);
    
    // Vector addition demo
    printf("Running vector addition...\n");
    std::vector<float> a(n), b(n), c(n);
    
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(n - i);
    }
    
    myproject::vector_add(c.data(), a.data(), b.data(), n);
    
    // Verify
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != static_cast<float>(n)) {
            passed = false;
            break;
        }
    }
    printf("Vector add: %s\n", passed ? "PASS" : "FAIL");
    
    // Reduce demo
    printf("\nRunning reduction...\n");
    std::vector<float> data(n, 1.0f);
    float sum = myproject::reduce_sum(data.data(), n);
    printf("Sum of %d ones: %.0f (%s)\n", n, sum, 
           (sum == static_cast<float>(n)) ? "PASS" : "FAIL");
    
    // Cleanup
    myproject::cleanup();
    printf("\nDone!\n");
    
    return 0;
}
