/**
 * main.cpp - Application entry point
 */

#include "app/config.h"
#include "app/cuda_ops.cuh"
#include "app/utils.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    printf("CUDA Application Template\n");
    printf("=========================\n\n");
    
    // Parse configuration
    app::Config config = app::Config::parse(argc, argv);
    
    if (config.verbose) {
        config.print();
    }
    
    // Initialize CUDA
    if (!app::cuda::init(config.device_id)) {
        fprintf(stderr, "Failed to initialize CUDA device %d\n", config.device_id);
        return 1;
    }
    
    printf("Device: %s\n\n", app::cuda::get_device_info());
    
    // Allocate and initialize data
    printf("Allocating %d elements (%.1f MB)...\n", 
           config.problem_size,
           config.problem_size * sizeof(float) / 1e6);
    
    std::vector<float> data(config.problem_size);
    for (int i = 0; i < config.problem_size; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Process on GPU
    printf("Processing (%d iterations)...\n", config.iterations);
    
    float gpu_time = app::cuda::process(data.data(), data.size(), config.iterations);
    
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Throughput: %.2f GB/s\n", 
           2.0 * config.problem_size * sizeof(float) * config.iterations / gpu_time / 1e6);
    
    // Verify results
    if (config.verify) {
        printf("\nVerifying results...\n");
        if (app::cuda::verify(data.data(), data.size())) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            app::cuda::cleanup();
            return 1;
        }
    }
    
    // Cleanup
    app::cuda::cleanup();
    printf("\nDone!\n");
    
    return 0;
}
