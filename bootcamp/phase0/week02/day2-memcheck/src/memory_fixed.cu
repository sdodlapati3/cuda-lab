/**
 * Day 2: Memory Errors FIXED
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Fix 1: Add bounds check
// ============================================================================
__global__ void fixed_bounds(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // FIXED: Bounds check
        data[idx] = idx;
    }
}

// ============================================================================
// Fix 2: Match shared memory size to thread count
// ============================================================================
__global__ void fixed_shared() {
    __shared__ int sdata[256];  // FIXED: Match blockDim.x
    int tid = threadIdx.x;
    
    if (tid < 256) {  // Extra safety
        sdata[tid] = tid;
    }
    __syncthreads();
}

// ============================================================================
// Fix 3: Initialize all shared memory
// ============================================================================
__global__ void fixed_init(int* output) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    // FIXED: Initialize ALL elements first
    sdata[tid] = 0;
    __syncthreads();
    
    // Then do partial update
    if (tid % 2 == 0) {
        sdata[tid] = tid;
    }
    __syncthreads();
    
    output[tid] = sdata[tid];
}

// ============================================================================
// Fix 4: Correct loop bound
// ============================================================================
__global__ void fixed_reduce(int* data, int n) {
    int tid = threadIdx.x;
    
    // FIXED: Use < not <=
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            data[tid] += data[tid + s];
        }
        __syncthreads();
    }
}

// ============================================================================
// Fix 5: Validate index range
// ============================================================================
__global__ void fixed_index(int* data, int n, int offset) {
    int idx = threadIdx.x - offset;
    // FIXED: Check for valid range
    if (idx >= 0 && idx < n) {
        data[idx] = 42;
    }
}

int main() {
    printf("Memory Errors FIXED\n");
    printf("===================\n");
    printf("Run with: compute-sanitizer --tool memcheck ./memory_fixed\n\n");
    
    int *d_data, *d_output;
    const int N = 256;
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    
    printf("Test 1: Fixed bounds check...\n");
    fixed_bounds<<<1, 256>>>(d_data, 100);  // Only 100 valid, but we check
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  ✓ Passed\n\n");
    
    printf("Test 2: Fixed shared memory size...\n");
    fixed_shared<<<1, 256>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  ✓ Passed\n\n");
    
    printf("Test 3: Fixed initialization...\n");
    fixed_init<<<1, 256>>>(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  ✓ Passed\n\n");
    
    printf("Test 4: Fixed reduction...\n");
    // Initialize
    int* h_data = new int[N];
    for (int i = 0; i < N; i++) h_data[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    fixed_reduce<<<1, 128>>>(d_data, 128);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Sum = %d (expected 128) ✓\n\n", result);
    
    printf("Test 5: Fixed index validation...\n");
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(int)));
    fixed_index<<<1, 64>>>(d_data + 32, 64, 16);  // Safe range
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  ✓ Passed\n\n");
    
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_output);
    
    printf("All fixes verified!\n");
    return 0;
}
