/**
 * main.cpp - Benchmark entry point
 */

#include "benchmark/benchmark.h"
#include "benchmark/implementations.cuh"
#include <cstdio>
#include <cstdlib>

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n SIZE     Problem size (default: 1M)\n");
    printf("  -r RUNS     Number of runs (default: 10)\n");
    printf("  -w WARMUP   Warmup runs (default: 3)\n");
    printf("  -o FILE     Output CSV file\n");
    printf("  -j FILE     Output JSON file\n");
}

int main(int argc, char** argv) {
    printf("CUDA Benchmark: Reduction Implementations\n");
    printf("==========================================\n\n");
    
    size_t n = 1 << 20;  // 1M elements
    int runs = 10;
    int warmup = 3;
    const char* csv_file = nullptr;
    const char* json_file = nullptr;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atol(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            json_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    printf("Configuration:\n");
    printf("  Problem size: %zu elements (%.1f MB)\n", n, n * sizeof(float) / 1e6);
    printf("  Runs: %d (warmup: %d)\n", runs, warmup);
    printf("\n");
    
    // Setup
    impl::setup(n);
    
    // Create benchmark
    bench::Benchmark bm("Reduction", warmup, runs);
    
    // Run benchmarks
    auto r0 = bm.run("V0: Global Atomics", impl::reduce_v0_global);
    r0.throughput_gbps = n * sizeof(float) / r0.mean_ms / 1e6;
    bm.add_result(r0);
    
    auto r1 = bm.run("V1: Shared Memory", impl::reduce_v1_shared);
    r1.throughput_gbps = n * sizeof(float) / r1.mean_ms / 1e6;
    bm.add_result(r1);
    
    auto r2 = bm.run("V2: Warp Shuffle", impl::reduce_v2_warp);
    r2.throughput_gbps = n * sizeof(float) / r2.mean_ms / 1e6;
    bm.add_result(r2);
    
    auto r3 = bm.run("V3: Vectorized", impl::reduce_v3_vector);
    r3.throughput_gbps = n * sizeof(float) / r3.mean_ms / 1e6;
    bm.add_result(r3);
    
    // Print results
    printf("\n");
    bm.print_results();
    
    // Export if requested
    if (csv_file) {
        bm.export_csv(csv_file);
        printf("\nResults saved to: %s\n", csv_file);
    }
    if (json_file) {
        bm.export_json(json_file);
        printf("Results saved to: %s\n", json_file);
    }
    
    // Cleanup
    impl::cleanup();
    
    return 0;
}
