/**
 * cub_reduction.cu - Compare hand-written reduction vs CUB library
 * 
 * Key lesson: Library-first development!
 * CUB is highly optimized - know when to use it vs custom kernels.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Our best hand-written reduction (warp shuffle + first add)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void reduce_handwritten(const float* input, float* output, int n) {
    __shared__ float warp_sums[8];  // 256/32 = 8 warps
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // First add during load
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Store warp results
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

// CUB block-level reduction for comparison
__global__ void reduce_cub_block(const float* input, float* output, int n) {
    // Use CUB's BlockReduce
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;
    
    float block_sum = BlockReduce(temp_storage).Sum(val);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

int main() {
    printf("=== Library-First: CUB vs Hand-Written Reduction ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 100;
    
    printf("Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    
    // Allocate
    float* h_input = new float[N];
    double expected = 0.0;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple test: sum = N
        expected += 1.0;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Get device info for bandwidth calculation
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_bw = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    
    printf("Device: %s\n", prop.name);
    printf("Peak memory bandwidth: %.0f GB/s\n", peak_bw);
    printf("Expected sum: %.0f\n\n", expected);
    
    printf("%-25s %-12s %-12s %-12s %-10s\n", 
           "Method", "Time(us)", "GB/s", "% Peak", "Result");
    printf("--------------------------------------------------------------------\n");
    
    int blocks, threads;
    float result;
    float bw;
    
    // ============================================
    // 1. Hand-written reduction (our best version)
    // ============================================
    blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    threads = BLOCK_SIZE;
    
    float zero = 0.0f;
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_handwritten<<<blocks, threads>>>(d_input, d_output, N);
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
        reduce_handwritten<<<blocks, threads>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    bw = (N * sizeof(float)) / (ms / TRIALS / 1000) / 1e9;
    printf("%-25s %-12.1f %-12.1f %-12.1f %-10.0f\n", 
           "Hand-written (warp shfl)", ms / TRIALS * 1000, bw, bw / peak_bw * 100, result);
    
    // ============================================
    // 2. CUB BlockReduce (block-level primitive)
    // ============================================
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_cub_block<<<blocks, threads>>>(d_input, d_output, N);
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
        reduce_cub_block<<<blocks, threads>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    bw = (N * sizeof(float)) / (ms / TRIALS / 1000) / 1e9;
    printf("%-25s %-12.1f %-12.1f %-12.1f %-10.0f\n", 
           "CUB BlockReduce", ms / TRIALS * 1000, bw, bw / peak_bw * 100, result);
    
    // ============================================
    // 3. CUB DeviceReduce (full device-wide primitive)
    // ============================================
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Get temp storage size
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run once to verify
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    bw = (N * sizeof(float)) / (ms / TRIALS / 1000) / 1e9;
    printf("%-25s %-12.1f %-12.1f %-12.1f %-10.0f\n", 
           "CUB DeviceReduce::Sum", ms / TRIALS * 1000, bw, bw / peak_bw * 100, result);
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. CUB DeviceReduce is typically fastest - uses optimal algorithms\n");
    printf("2. CUB handles edge cases, arbitrary sizes, and different data types\n");
    printf("3. Hand-written kernels are educational but rarely beat libraries\n");
    printf("4. Use CUB when: simple reduction, standard operation\n");
    printf("5. Write custom when: fusion opportunities, non-standard ops\n");
    printf("\n");
    printf("Temp storage for CUB: %zu bytes\n", temp_storage_bytes);
    printf("\n");
    printf("Library-First Rule:\n");
    printf("  CUB::DeviceReduce should be your DEFAULT choice.\n");
    printf("  Only write custom if you need fusion or custom operators.\n");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);
    delete[] h_input;
    
    return 0;
}
