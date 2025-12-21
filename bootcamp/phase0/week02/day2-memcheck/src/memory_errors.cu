/**
 * Day 2: Memory Errors
 * 
 * Intentional memory errors for learning.
 * Run: compute-sanitizer --tool memcheck ./memory_errors
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
// Error 1: Out-of-bounds global memory access
// ============================================================================
__global__ void oob_global(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: No bounds check! With 256 threads and n=100, this overflows
    data[idx] = idx;
}

// ============================================================================
// Error 2: Out-of-bounds shared memory access
// ============================================================================
__global__ void oob_shared() {
    __shared__ int sdata[64];  // Only 64 elements!
    int tid = threadIdx.x;
    
    // BUG: If blockDim.x > 64, this overflows
    sdata[tid] = tid;
    __syncthreads();
}

// ============================================================================
// Error 3: Reading uninitialized shared memory
// ============================================================================
__global__ void uninit_shared(int* output) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    // Only even threads initialize
    if (tid % 2 == 0) {
        sdata[tid] = tid;
    }
    __syncthreads();
    
    // BUG: Odd indices are uninitialized!
    output[tid] = sdata[tid];
}

// ============================================================================
// Error 4: Off-by-one error in reduction
// ============================================================================
__global__ void off_by_one_reduce(int* data, int n) {
    int tid = threadIdx.x;
    
    // BUG: Should be tid + 1, but we use tid + s which can overflow
    for (int s = 1; s <= blockDim.x; s *= 2) {  // BUG: <= instead of <
        if (tid % (2 * s) == 0) {
            if (tid + s < n) {  // This check is good
                data[tid] += data[tid + s];
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Error 5: Negative index access
// ============================================================================
__global__ void negative_index(int* data, int offset) {
    int idx = threadIdx.x - offset;  // Can be negative!
    // BUG: Negative index wraps to huge positive in unsigned arithmetic
    data[idx] = 42;  // Out of bounds!
}

int main() {
    printf("Memory Error Examples\n");
    printf("=====================\n");
    printf("Run with: compute-sanitizer --tool memcheck ./memory_errors\n\n");
    
    int *d_data;
    const int N = 100;  // Intentionally smaller than thread count
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    
    printf("Test 1: Out-of-bounds global access...\n");
    oob_global<<<1, 256>>>(d_data, N);  // 256 threads, only 100 elements!
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer)\n\n");
    
    printf("Test 2: Out-of-bounds shared access...\n");
    oob_shared<<<1, 256>>>();  // 256 threads, only 64 shared elements!
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer)\n\n");
    
    printf("Test 3: Uninitialized shared memory...\n");
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, 256 * sizeof(int)));
    uninit_shared<<<1, 256>>>(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (try --tool initcheck)\n\n");
    
    printf("Test 4: Off-by-one in reduction...\n");
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(int)));
    off_by_one_reduce<<<1, 128>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer)\n\n");
    
    printf("Test 5: Negative index...\n");
    negative_index<<<1, 32>>>(d_data + 16, 32);  // offset=32, so idx goes negative
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer)\n\n");
    
    cudaFree(d_data);
    cudaFree(d_output);
    
    printf("All tests completed.\n");
    return 0;
}
