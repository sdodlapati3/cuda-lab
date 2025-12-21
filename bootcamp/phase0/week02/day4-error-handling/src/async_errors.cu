/**
 * Day 4: Async Error Demonstrations
 * 
 * Shows how async errors work and when they're detected.
 */

#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// Simple error check macro
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("  CUDA error at %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// ============================================================================
// Kernel that causes an illegal memory access
// ============================================================================
__global__ void illegal_access(int* data) {
    // Access a null pointer - causes illegal memory access
    int* bad_ptr = nullptr;
    *bad_ptr = 42;  // BOOM!
}

// ============================================================================
// Kernel with out-of-bounds access
// ============================================================================
__global__ void oob_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No bounds check - will access beyond allocation
    data[idx + 10000] = idx;  // OOB!
}

// ============================================================================
// Demonstration: Launch error vs execution error
// ============================================================================
void demo_launch_error() {
    printf("\n=== Launch Error Demo ===\n");
    
    // Invalid launch configuration
    printf("Launching with invalid config (too many threads)...\n");
    
    int* d_data;
    cudaMalloc(&d_data, sizeof(int));
    
    // 2048 threads per block is too many (max is usually 1024)
    dim3 block(2048, 1, 1);  // INVALID!
    oob_kernel<<<1, block>>>(d_data, 100);
    
    // Launch error is synchronous - detected immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  Launch error detected: %s\n", cudaGetErrorString(err));
    } else {
        printf("  No launch error (unexpected)\n");
    }
    
    cudaFree(d_data);
}

void demo_execution_error() {
    printf("\n=== Execution Error Demo ===\n");
    
    int* d_data;
    cudaMalloc(&d_data, 100 * sizeof(int));
    
    printf("Launching kernel with OOB access...\n");
    oob_kernel<<<1, 256>>>(d_data, 100);
    
    // Check immediately after launch
    cudaError_t launch_err = cudaGetLastError();
    printf("  After launch: %s\n", cudaGetErrorString(launch_err));
    
    // Now synchronize
    printf("Synchronizing...\n");
    cudaError_t exec_err = cudaDeviceSynchronize();
    printf("  After sync: %s\n", cudaGetErrorString(exec_err));
    
    // Check again
    cudaError_t final_err = cudaGetLastError();
    printf("  Final check: %s\n", cudaGetErrorString(final_err));
    
    cudaFree(d_data);
    
    // Clear the error state
    cudaGetLastError();
}

void demo_sticky_errors() {
    printf("\n=== Sticky Error Demo ===\n");
    
    // Cause an error
    int* d_data;
    cudaMalloc(&d_data, 100 * sizeof(int));
    oob_kernel<<<1, 256>>>(d_data, 100);
    cudaDeviceSynchronize();
    
    printf("After causing error:\n");
    
    // Error persists until cleared
    for (int i = 0; i < 3; i++) {
        cudaError_t err = cudaPeekAtLastError();  // Peek (don't clear)
        printf("  Peek %d: %s\n", i + 1, cudaGetErrorString(err));
    }
    
    // Clear the error
    cudaError_t err = cudaGetLastError();  // This clears it
    printf("  After cudaGetLastError: %s\n", cudaGetErrorString(err));
    
    // Now it's clear
    err = cudaGetLastError();
    printf("  Next check: %s\n", cudaGetErrorString(err));
    
    cudaFree(d_data);
}

void demo_error_after_multiple_kernels() {
    printf("\n=== Multiple Kernels Demo ===\n");
    
    int* d_data;
    cudaMalloc(&d_data, 100 * sizeof(int));
    
    // Launch multiple kernels - error happens in middle
    printf("Launching 3 kernels (error in 2nd)...\n");
    
    oob_kernel<<<1, 32>>>(d_data, 100);  // OK (32 < 100)
    oob_kernel<<<1, 256>>>(d_data, 100); // BAD (256 + 10000 > 100)
    oob_kernel<<<1, 32>>>(d_data, 100);  // Never runs (previous error)
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    
    printf("  Error after sync: %s\n", cudaGetErrorString(err));
    printf("  Note: Can't tell WHICH kernel failed without events!\n");
    
    cudaFree(d_data);
}

int main() {
    printf("CUDA Async Error Demonstrations\n");
    printf("================================\n");
    
    demo_launch_error();
    demo_execution_error();
    demo_sticky_errors();
    demo_error_after_multiple_kernels();
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. Launch errors are synchronous (detected at cudaGetLastError)\n");
    printf("2. Execution errors are async (detected after sync)\n");
    printf("3. Errors are sticky - persist until cudaGetLastError clears them\n");
    printf("4. Multiple kernels: can't identify which failed without events\n");
    
    return 0;
}
