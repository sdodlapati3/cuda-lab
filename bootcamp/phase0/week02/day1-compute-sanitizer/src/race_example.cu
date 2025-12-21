/**
 * Day 1: Race Condition Examples
 * 
 * This file contains intentional race conditions for learning.
 * Use compute-sanitizer to detect them!
 * 
 * Run: compute-sanitizer --tool racecheck ./race_example
 */

#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// Example 1: Obvious race - all threads write to same location
// ============================================================================
__global__ void race_global_counter(int* counter) {
    // RACE: All threads try to read-modify-write the same location
    *counter = *counter + 1;
}

// ============================================================================
// Example 2: Shared memory race - missing __syncthreads
// ============================================================================
__global__ void race_shared_memory(const int* input, int* output) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = input[blockIdx.x * blockDim.x + tid];
    
    // RACE: Missing __syncthreads() here!
    // Some threads may read before others have written
    
    // Try to compute sum of neighbors
    if (tid > 0 && tid < blockDim.x - 1) {
        // This reads sdata[tid-1] and sdata[tid+1] which might not be ready!
        output[blockIdx.x * blockDim.x + tid] = sdata[tid-1] + sdata[tid] + sdata[tid+1];
    }
}

// ============================================================================
// Example 3: Subtle race in reduction
// ============================================================================
__global__ void race_reduction(const int* input, int* output, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // RACE: Not all threads participate in sync!
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // RACE: __syncthreads inside conditional!
        // Only threads with tid < s execute the sync
        if (tid < s) {
            __syncthreads();  // BUG: This is inside the conditional!
        }
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Example 4: Write-after-write race
// ============================================================================
__global__ void race_waw(int* output) {
    int tid = threadIdx.x;
    
    // RACE: Multiple threads write to the same location
    // The "winner" is non-deterministic
    if (tid % 32 == 0) {  // Warp leaders
        output[0] = tid;  // WAW race!
    }
}

// ============================================================================
// Helper to check CUDA errors
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("Race Condition Examples\n");
    printf("========================\n");
    printf("Run with: compute-sanitizer --tool racecheck ./race_example\n\n");
    
    const int N = 256;
    int *d_input, *d_output, *d_counter;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    
    // Initialize input
    int* h_input = new int[N];
    for (int i = 0; i < N; i++) h_input[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Test 1: Global counter race...\n");
    race_global_counter<<<1, 256>>>(d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int counter;
    CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Counter = %d (expected 256, got %d due to race)\n\n", counter, counter);
    
    printf("Test 2: Shared memory race...\n");
    race_shared_memory<<<1, 256>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer output)\n\n");
    
    printf("Test 3: Reduction race...\n");
    race_reduction<<<1, 256>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer output)\n\n");
    
    printf("Test 4: Write-after-write race...\n");
    race_waw<<<1, 256>>>(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed (check sanitizer output)\n\n");
    
    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counter);
    
    printf("All tests completed.\n");
    printf("If compute-sanitizer found 0 errors, you might be lucky!\n");
    printf("Race conditions are non-deterministic.\n");
    
    return 0;
}
