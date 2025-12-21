/**
 * bank_conflicts.cu - Demonstrate shared memory bank conflicts
 * 
 * Learning objectives:
 * - See performance impact of bank conflicts
 * - Understand conflict patterns
 * - Apply fixes
 */

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_SIZE 32

// No bank conflict: stride-1 access
__global__ void no_conflict(float* output, int iterations) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    
    // Initialize
    smem[tid] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Stride-1 access: thread i reads smem[i]
        // Each thread accesses different bank → NO CONFLICT
        sum += smem[tid];
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

// 2-way bank conflict: stride-2 access
__global__ void two_way_conflict(float* output, int iterations) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Stride-2: threads 0,16 → bank 0, threads 1,17 → bank 2, etc.
        // 2 threads per bank → 2-way conflict
        sum += smem[(tid * 2) % 1024];
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

// 32-way bank conflict: stride-32 access (WORST CASE)
__global__ void worst_conflict(float* output, int iterations) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Stride-32: ALL threads in warp access bank 0!
        // 32-way conflict → serialized
        sum += smem[(tid * 32) % 1024];
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

// Broadcast: all threads read SAME address (special case - no conflict!)
__global__ void broadcast(float* output, int iterations) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // All threads read same address → BROADCAST (hardware optimized)
        sum += smem[0];
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

// Column access (common in matrix operations)
__global__ void column_conflict(float* output, int iterations) {
    __shared__ float matrix[32][32];  // 32-way conflict on column access!
    
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    
    matrix[row][col] = (float)tid;
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Column access: threads access matrix[0-31][same_col]
        // All 32 threads hit same bank!
        sum += matrix[tid % 32][0];  // Accessing column 0
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

// Column access with padding (FIXED)
__global__ void column_no_conflict(float* output, int iterations) {
    __shared__ float matrix[32][33];  // Padding: 33 instead of 32!
    
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    
    if (row < 32 && col < 32) {
        matrix[row][col] = (float)tid;
    }
    __syncthreads();
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // With padding, column access goes to different banks
        sum += matrix[tid % 32][0];
    }
    
    if (tid == 0) output[blockIdx.x] = sum;
}

int main() {
    printf("=== Shared Memory Bank Conflicts ===\n\n");
    
    printf("32 banks, each 4 bytes wide\n");
    printf("Address i → Bank (i/4) %% 32\n\n");
    
    const int ITERATIONS = 10000;
    const int BLOCKS = 256;
    const int THREADS = 256;
    
    float* d_output;
    cudaMalloc(&d_output, BLOCKS * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Warmup
    no_conflict<<<BLOCKS, THREADS>>>(d_output, 10);
    cudaDeviceSynchronize();
    
    printf("=== Access Pattern Benchmark ===\n");
    printf("%-25s %10s %10s\n", "Pattern", "Time (ms)", "Slowdown");
    printf("--------------------------------------------------\n");
    
    // No conflict baseline
    cudaEventRecord(start);
    no_conflict<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float baseline = ms;
    printf("%-25s %10.3f %10.1fx\n", "No conflict (stride 1)", ms, 1.0f);
    
    // Broadcast
    cudaEventRecord(start);
    broadcast<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s %10.3f %10.1fx\n", "Broadcast (same addr)", ms, ms / baseline);
    
    // 2-way conflict
    cudaEventRecord(start);
    two_way_conflict<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s %10.3f %10.1fx\n", "2-way conflict (stride 2)", ms, ms / baseline);
    
    // 32-way conflict
    cudaEventRecord(start);
    worst_conflict<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s %10.3f %10.1fx\n", "32-way conflict (stride 32)", ms, ms / baseline);
    
    // Column access
    cudaEventRecord(start);
    column_conflict<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s %10.3f %10.1fx\n", "Column access (conflict)", ms, ms / baseline);
    
    // Column access fixed
    cudaEventRecord(start);
    column_no_conflict<<<BLOCKS, THREADS>>>(d_output, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s %10.3f %10.1fx\n", "Column access (padded)", ms, ms / baseline);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    
    printf("\n=== Bank Calculation ===\n");
    printf("Example: float smem[N]\n");
    printf("Address of smem[i] in bytes: i * 4\n");
    printf("Bank of smem[i]: (i * 4 / 4) %% 32 = i %% 32\n");
    printf("\n");
    printf("smem[0]  → Bank 0\n");
    printf("smem[1]  → Bank 1\n");
    printf("smem[32] → Bank 0 (wraps around!)\n");
    printf("smem[33] → Bank 1\n");
    
    printf("\n=== Fix Techniques ===\n");
    printf("1. Padding: __shared__ float m[32][33] instead of [32][32]\n");
    printf("2. Change access pattern if possible\n");
    printf("3. Broadcast is free - exploit if many threads need same data\n");
    
    return 0;
}
