/**
 * Day 3: cuda-gdb Example
 * 
 * Simple kernels for practicing breakpoints and inspection.
 * 
 * Run: cuda-gdb ./debug_example
 */

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel for basic debugging
__global__ void simple_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Good place for breakpoint - every thread runs this
    if (idx < n) {
        data[idx] = idx * 2;
    }
}

// Kernel with shared memory
__global__ void shared_mem_kernel(int* data, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();  // Good breakpoint after sync
    
    // Modify in shared memory
    sdata[tid] *= 2;
    __syncthreads();
    
    // Write back
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Kernel with a bug to find
__global__ void buggy_kernel(int* data, int* result, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    // Reduction (intentional bug for debugging practice)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // BUG: Missing __syncthreads() here!
        // Use cuda-gdb to step through and see the problem
    }
    
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    printf("cuda-gdb Debug Examples\n");
    printf("=======================\n");
    printf("Run with: cuda-gdb ./debug_example\n");
    printf("\nSuggested breakpoints:\n");
    printf("  break simple_kernel\n");
    printf("  break shared_mem_kernel\n");
    printf("  break buggy_kernel\n\n");
    
    const int N = 256;
    int *d_data, *d_result;
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    // Initialize
    int* h_data = new int[N];
    for (int i = 0; i < N; i++) h_data[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Running simple_kernel...\n");
    simple_kernel<<<1, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Running shared_mem_kernel...\n");
    shared_mem_kernel<<<1, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Running buggy_kernel...\n");
    buggy_kernel<<<1, 256>>>(d_data, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check result
    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Reduction result: %d (expected: %d)\n", result, N);
    
    if (result != N) {
        printf("BUG DETECTED! Use cuda-gdb to find it.\n");
    }
    
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_result);
    
    return 0;
}
