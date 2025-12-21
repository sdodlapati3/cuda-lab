/**
 * bench_main.cpp - Benchmark runner
 */

#include "myproject/myproject.h"
#include "myproject/timer.h"
#include <cstdio>
#include <vector>

void benchmark_vector_add(int n, int runs) {
    printf("Benchmarking vector_add (n=%d, runs=%d)\n", n, runs);
    
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(n - i);
    }
    
    // Warmup
    myproject::vector_add(c.data(), a.data(), b.data(), n);
    
    myproject::CpuTimer timer;
    timer.start();
    for (int i = 0; i < runs; i++) {
        myproject::vector_add(c.data(), a.data(), b.data(), n);
    }
    timer.stop();
    
    double avg_ms = timer.elapsed_ms() / runs;
    double bw = 3.0 * n * sizeof(float) / avg_ms / 1e6;  // Read a, b; write c
    
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Bandwidth: %.2f GB/s\n\n", bw);
}

void benchmark_reduce(int n, int runs) {
    printf("Benchmarking reduce_sum (n=%d, runs=%d)\n", n, runs);
    
    std::vector<float> data(n, 1.0f);
    
    // Warmup
    myproject::reduce_sum(data.data(), n);
    
    myproject::CpuTimer timer;
    timer.start();
    for (int i = 0; i < runs; i++) {
        myproject::reduce_sum(data.data(), n);
    }
    timer.stop();
    
    double avg_ms = timer.elapsed_ms() / runs;
    double bw = n * sizeof(float) / avg_ms / 1e6;
    
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Bandwidth: %.2f GB/s\n\n", bw);
}

void benchmark_matmul(int size, int runs) {
    printf("Benchmarking matmul (%dx%d, runs=%d)\n", size, size, runs);
    
    int n = size * size;
    std::vector<float> A(n), B(n), C(n);
    for (int i = 0; i < n; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    
    // Warmup
    myproject::matmul(C.data(), A.data(), B.data(), size, size, size);
    
    myproject::CpuTimer timer;
    timer.start();
    for (int i = 0; i < runs; i++) {
        myproject::matmul(C.data(), A.data(), B.data(), size, size, size);
    }
    timer.stop();
    
    double avg_ms = timer.elapsed_ms() / runs;
    double gflops = 2.0 * size * size * size / avg_ms / 1e6;
    
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Performance: %.2f GFLOPS\n\n", gflops);
}

int main() {
    printf("MyProject Benchmarks\n");
    printf("====================\n\n");
    
    if (!myproject::initialize(0)) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    
    printf("Device: %s\n\n", myproject::get_device_info());
    
    // Run benchmarks
    benchmark_vector_add(1 << 20, 100);
    benchmark_vector_add(1 << 24, 10);
    
    benchmark_reduce(1 << 20, 100);
    benchmark_reduce(1 << 24, 10);
    
    benchmark_matmul(512, 10);
    benchmark_matmul(1024, 5);
    
    myproject::cleanup();
    return 0;
}
