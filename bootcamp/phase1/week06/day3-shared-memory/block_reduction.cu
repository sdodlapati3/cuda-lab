/**
 * block_reduction.cu - Parallel reduction using shared memory
 * 
 * Learning objectives:
 * - Implement parallel reduction pattern
 * - Use shared memory for inter-thread communication
 * - Understand reduction tree pattern
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple sequential reduction (for comparison)
__global__ void reduce_sequential(const float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    *output = sum;
}

// Parallel reduction using shared memory (basic version)
__global__ void reduce_shared_v1(const float* input, float* output, int n) {
    __shared__ float smem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load to shared memory
    smem[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// Optimized reduction: avoid warp divergence
__global__ void reduce_shared_v2(const float* input, float* output, int n) {
    __shared__ float smem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    smem[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Contiguous threads do the work (no divergence within warp)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// Warp-level reduction (uses warp shuffle for last steps)
__global__ void reduce_shared_v3(const float* input, float* output, int n) {
    __shared__ float smem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid / 32;
    
    // Load to shared
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // First reduce within warp using shuffle (no shared memory needed!)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    
    // First thread of each warp writes to shared
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces the warp results
    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

int main() {
    printf("=== Block Reduction with Shared Memory ===\n\n");
    
    const int N = 1024 * 1024;  // 1M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Array size: %d elements\n", N);
    printf("Block size: %d, Num blocks: %d\n\n", BLOCK_SIZE, NUM_BLOCKS);
    
    // Allocate and initialize
    float* h_input = new float[N];
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple: sum should equal N
        expected += h_input[i];
    }
    
    float* d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("Expected sum: %.0f\n\n", expected);
    
    // Test each version
    printf("=== Version 1: Basic (with divergence) ===\n");
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_shared_v1<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.0f (%.3f ms)\n\n", result, ms);
    
    printf("=== Version 2: Contiguous (no divergence) ===\n");
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_shared_v2<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.0f (%.3f ms)\n\n", result, ms);
    
    printf("=== Version 3: Warp Shuffle ===\n");
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_shared_v3<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.0f (%.3f ms)\n\n", result, ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    printf("=== Reduction Pattern Summary ===\n");
    printf("1. Load data to shared memory\n");
    printf("2. Reduce in tree pattern (stride halving)\n");
    printf("3. Use contiguous threads to avoid divergence\n");
    printf("4. Use warp shuffle for last 5 steps (lane 0-31)\n");
    printf("5. atomicAdd for multi-block reduction\n");
    
    return 0;
}
