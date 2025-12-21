/**
 * launch_config.cu - Explore launch configurations
 * 
 * Learning objectives:
 * - Benchmark different block sizes
 * - Understand occupancy impact
 * - Find optimal configuration
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple compute kernel for benchmarking
__global__ void saxpy(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// More compute-intensive kernel (uses more registers)
__global__ void compute_heavy(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float val = data[i];
        for (int iter = 0; iter < iterations; iter++) {
            val = sinf(val) + cosf(val);
            val = sqrtf(fabsf(val) + 1.0f);
        }
        data[i] = val;
    }
}

float benchmark_kernel(int block_size, int num_blocks, float* d_y, const float* d_x, 
                       float a, int n, int warmup, int trials) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        saxpy<<<num_blocks, block_size>>>(d_y, d_x, a, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < trials; i++) {
        saxpy<<<num_blocks, block_size>>>(d_y, d_x, a, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / trials;
}

int main() {
    printf("=== Launch Configuration Benchmark ===\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 100;
    
    float* d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_x, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("=== Varying Block Size (fixed grid = 2×SMs) ===\n");
    printf("Array: %d elements\n\n", N);
    
    int num_blocks = prop.multiProcessorCount * 2;
    
    printf("%10s %12s %12s %15s\n", "Block Size", "Warps/Block", "Time (ms)", "Bandwidth (GB/s)");
    printf("-------------------------------------------------------------\n");
    
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    float best_time = 1e9f;
    int best_block_size = 128;
    
    for (int block_size : block_sizes) {
        float ms = benchmark_kernel(block_size, num_blocks, d_y, d_x, 2.0f, N, 10, TRIALS);
        float gb_s = 3.0f * N * sizeof(float) / ms / 1e6;  // 2 reads + 1 write
        
        printf("%10d %12d %12.4f %15.1f", 
               block_size, block_size / 32, ms, gb_s);
        
        if (ms < best_time) {
            best_time = ms;
            best_block_size = block_size;
            printf(" *");
        }
        printf("\n");
    }
    
    printf("\nBest block size: %d\n", best_block_size);
    
    // Vary grid size
    printf("\n=== Varying Grid Size (block size = 256) ===\n\n");
    
    int block_size = 256;
    int grid_sizes[] = {1, 10, 100, prop.multiProcessorCount, 
                        prop.multiProcessorCount * 2,
                        prop.multiProcessorCount * 4,
                        (N + block_size - 1) / block_size};
    
    printf("%10s %12s %15s\n", "Grid Size", "Time (ms)", "Bandwidth (GB/s)");
    printf("------------------------------------------\n");
    
    for (int grid_size : grid_sizes) {
        float ms = benchmark_kernel(block_size, grid_size, d_y, d_x, 2.0f, N, 10, TRIALS);
        float gb_s = 3.0f * N * sizeof(float) / ms / 1e6;
        
        const char* note = "";
        if (grid_size == prop.multiProcessorCount) note = " (SMs)";
        if (grid_size == prop.multiProcessorCount * 2) note = " (2×SMs)";
        if (grid_size == (N + block_size - 1) / block_size) note = " (max)";
        
        printf("%10d %12.4f %15.1f%s\n", grid_size, ms, gb_s, note);
    }
    
    delete[] h_data;
    cudaFree(d_x);
    cudaFree(d_y);
    
    printf("\n=== Guidelines ===\n");
    printf("1. Block size: 128-256 usually optimal\n");
    printf("2. Grid size: 2-4× SMs for grid-stride loops\n");
    printf("3. Profile your specific kernel - results vary!\n");
    printf("4. Use occupancy API for automatic tuning\n");
    
    return 0;
}
