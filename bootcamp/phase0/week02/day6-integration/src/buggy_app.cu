/**
 * Day 6: Buggy Application
 * 
 * This application has MULTIPLE bugs. Use the debug tools
 * from this week to find and fix them all.
 * 
 * Bugs to find:
 * 1. Race condition
 * 2. Memory error (out of bounds)
 * 3. Missing synchronization
 * 4. Memory leak
 * 
 * Tools to use:
 * - compute-sanitizer --tool racecheck
 * - compute-sanitizer --tool memcheck
 * - cuda-gdb
 */

#include <cstdio>
#include <cuda_runtime.h>

#define N 1000
#define BLOCK_SIZE 256

// ============================================================================
// Bug 1: Race condition in reduction
// ============================================================================
__global__ void buggy_sum(int* data, int* result) {
    __shared__ int sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < N) ? data[idx] : 0;
    // BUG: Missing __syncthreads() here!
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // BUG: Missing __syncthreads() here!
    }
    
    // BUG: Race condition - multiple blocks write to same result
    if (tid == 0) {
        *result += sdata[0];  // RACE!
    }
}

// ============================================================================
// Bug 2: Out of bounds access
// ============================================================================
__global__ void buggy_transform(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BUG: No bounds check!
    output[idx] = input[idx] * 2.0f;
    
    // BUG: Accessing out of bounds
    output[idx] += input[idx + 1];  // When idx == n-1, this is OOB!
}

// ============================================================================
// Main with bugs
// ============================================================================
int main() {
    printf("Buggy Application - Find the bugs!\n");
    printf("===================================\n\n");
    
    // Setup
    int* h_data = (int*)malloc(N * sizeof(int));
    int* d_data;
    int* d_result;
    
    for (int i = 0; i < N; i++) h_data[i] = 1;
    
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));
    
    // Run buggy sum
    printf("Running buggy_sum...\n");
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buggy_sum<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum result: %d (expected: %d)\n", result, N);
    
    // Run buggy transform
    printf("\nRunning buggy_transform...\n");
    float* h_float = (float*)malloc(N * sizeof(float));
    float* d_input;
    float* d_output;
    
    for (int i = 0; i < N; i++) h_float[i] = 1.0f;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_float, N * sizeof(float), cudaMemcpyHostToDevice);
    
    buggy_transform<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_float, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Transform complete (check for memory errors)\n");
    
    // BUG: Memory leaks! Not freeing memory
    // cudaFree(d_data);
    // cudaFree(d_result);
    // cudaFree(d_input);
    // cudaFree(d_output);
    // free(h_data);
    // free(h_float);
    
    printf("\nDone. Use debug tools to find the bugs!\n");
    printf("\nCommands to try:\n");
    printf("  compute-sanitizer --tool racecheck ./build/buggy_app\n");
    printf("  compute-sanitizer --tool memcheck ./build/buggy_app\n");
    printf("  compute-sanitizer --leak-check full ./build/buggy_app\n");
    
    return 0;
}
