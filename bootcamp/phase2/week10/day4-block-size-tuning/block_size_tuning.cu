/**
 * block_size_tuning.cu - Finding optimal block sizes
 * 
 * Learning objectives:
 * - Benchmark different block sizes
 * - Use occupancy API for suggestions
 * - Understand when block size matters
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Test kernel: SAXPY
__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a * x[idx] + y[idx];
    }
}

// Reduction kernel - block size matters more here
template<int BLOCK_SIZE>
__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE * 2 + tid;
    
    float sum = 0;
    if (idx < n) sum = in[idx];
    if (idx + BLOCK_SIZE < n) sum += in[idx + BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Stencil kernel - uses shared memory
template<int BLOCK_SIZE>
__global__ void stencil_1d(const float* in, float* out, int n) {
    __shared__ float smem[BLOCK_SIZE + 2];  // +2 for halos
    
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load center
    if (idx < n) smem[tid + 1] = in[idx];
    
    // Load halos
    if (tid == 0 && idx > 0) smem[0] = in[idx - 1];
    if (tid == BLOCK_SIZE - 1 && idx < n - 1) smem[BLOCK_SIZE + 1] = in[idx + 1];
    __syncthreads();
    
    if (idx > 0 && idx < n - 1) {
        out[idx] = 0.25f * smem[tid] + 0.5f * smem[tid + 1] + 0.25f * smem[tid + 2];
    }
}

void benchmark_block_sizes(const char* kernel_name, 
                           void (*run_kernel)(float*, float*, float*, int, int),
                           float* d_a, float* d_b, float* d_out, int n) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    printf("\n=== %s Block Size Benchmark ===\n\n", kernel_name);
    printf("%-12s %12s %12s %12s\n", "BlockSize", "Time(ms)", "Speedup", "Occupancy");
    printf("------------------------------------------------\n");
    
    float baseline_time = 0;
    const int TRIALS = 100;
    
    for (int i = 0; i < num_sizes; i++) {
        int block_size = block_sizes[i];
        
        // Warmup
        run_kernel(d_a, d_b, d_out, n, block_size);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            run_kernel(d_a, d_b, d_out, n, block_size);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float time_per = ms / TRIALS;
        
        if (i == 0) baseline_time = time_per;
        float speedup = baseline_time / time_per;
        
        // Get occupancy
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, saxpy, block_size, 0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        float occ = 100.0f * max_blocks * block_size / prop.maxThreadsPerMultiProcessor;
        
        printf("%-12d %12.4f %12.2fx %11.0f%%\n", block_size, time_per, speedup, occ);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_saxpy(float* d_a, float* d_b, float* d_out, int n, int block_size) {
    int blocks = (n + block_size - 1) / block_size;
    saxpy<<<blocks, block_size>>>(2.0f, d_a, d_b, d_out, n);
}

int main() {
    printf("=== Block Size Tuning ===\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n\n", prop.warpSize);
    
    // Use occupancy API
    int suggested_block, min_grid;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &suggested_block, saxpy, 0, 0);
    printf("API suggested block size for SAXPY: %d\n", suggested_block);
    printf("API suggested min grid size: %d\n", min_grid);
    
    // Allocate
    const int N = 1 << 24;  // 16M
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark SAXPY
    benchmark_block_sizes("SAXPY", run_saxpy, d_a, d_b, d_out, N);
    
    // Reduction benchmarks
    printf("\n=== Reduction Block Size Benchmark ===\n\n");
    printf("%-12s %12s\n", "BlockSize", "Time(ms)");
    printf("---------------------------\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float* d_partial;
    cudaMalloc(&d_partial, (N / 64) * sizeof(float));
    
    const int TRIALS = 50;
    
    #define BENCH_REDUCE(BS) { \
        cudaEventRecord(start); \
        for (int t = 0; t < TRIALS; t++) { \
            reduce_sum<BS><<<N / (BS * 2), BS>>>(d_a, d_partial, N); \
        } \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        float ms; \
        cudaEventElapsedTime(&ms, start, stop); \
        printf("%-12d %12.4f\n", BS, ms / TRIALS); \
    }
    
    BENCH_REDUCE(32);
    BENCH_REDUCE(64);
    BENCH_REDUCE(128);
    BENCH_REDUCE(256);
    BENCH_REDUCE(512);
    BENCH_REDUCE(1024);
    
    printf("\n=== Block Size Selection Guidelines ===\n\n");
    printf("1. Always use multiples of 32 (warp size)\n");
    printf("2. 256 is a good default for most kernels\n");
    printf("3. Memory-bound: larger blocks rarely help much\n");
    printf("4. Compute-bound: may benefit from larger blocks\n");
    printf("5. Reduction/shared-mem: test multiple sizes\n");
    printf("6. Use cudaOccupancyMaxPotentialBlockSize() as starting point\n");
    printf("7. Always benchmark YOUR specific kernel\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_partial);
    delete[] h_data;
    
    return 0;
}
