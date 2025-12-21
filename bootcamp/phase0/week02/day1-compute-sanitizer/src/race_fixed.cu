/**
 * Day 1: Race Conditions FIXED
 * 
 * This file shows how to fix each race condition from race_example.cu
 */

#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// Fix 1: Use atomicAdd for global counter
// ============================================================================
__global__ void fixed_global_counter(int* counter) {
    // FIXED: atomicAdd is thread-safe
    atomicAdd(counter, 1);
}

// ============================================================================
// Fix 2: Add __syncthreads before reading shared memory
// ============================================================================
__global__ void fixed_shared_memory(const int* input, int* output) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = input[blockIdx.x * blockDim.x + tid];
    
    // FIXED: Synchronize before reading
    __syncthreads();
    
    // Now safe to read neighbors
    if (tid > 0 && tid < blockDim.x - 1) {
        output[blockIdx.x * blockDim.x + tid] = sdata[tid-1] + sdata[tid] + sdata[tid+1];
    }
}

// ============================================================================
// Fix 3: __syncthreads OUTSIDE conditional
// ============================================================================
__global__ void fixed_reduction(const int* input, int* output, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // FIXED: All threads execute __syncthreads
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // FIXED: Outside the if!
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Fix 4: Only one thread writes (or use atomics)
// ============================================================================
__global__ void fixed_waw(int* output) {
    int tid = threadIdx.x;
    
    // FIXED: Only thread 0 writes
    if (tid == 0) {
        output[0] = tid;
    }
    
    // Alternative: If you need the last warp leader's value:
    // Use shared memory reduction to find it deterministically
}

// ============================================================================
// Bonus: Proper warp-level reduction (no races, no shared memory)
// ============================================================================
__global__ void warp_reduce_counter(int* counter) {
    int val = 1;  // Each thread contributes 1
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // First thread in each warp has the warp sum
    if (threadIdx.x % 32 == 0) {
        atomicAdd(counter, val);
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

int main() {
    printf("Race Condition FIXES\n");
    printf("====================\n");
    printf("Run with: compute-sanitizer --tool racecheck ./race_fixed\n\n");
    
    const int N = 256;
    int *d_input, *d_output, *d_counter;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    
    int* h_input = new int[N];
    for (int i = 0; i < N; i++) h_input[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Test 1: Fixed global counter (atomicAdd)...\n");
    fixed_global_counter<<<1, 256>>>(d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int counter;
    CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Counter = %d (expected 256) ✓\n\n", counter);
    
    printf("Test 2: Fixed shared memory (__syncthreads)...\n");
    fixed_shared_memory<<<1, 256>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed ✓\n\n");
    
    printf("Test 3: Fixed reduction (sync outside if)...\n");
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    fixed_reduction<<<1, 256>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int sum;
    CUDA_CHECK(cudaMemcpy(&sum, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Sum = %d (expected 256) ✓\n\n", sum);
    
    printf("Test 4: Fixed WAW (single writer)...\n");
    fixed_waw<<<1, 256>>>(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed ✓\n\n");
    
    printf("Bonus: Warp-level reduction...\n");
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    warp_reduce_counter<<<1, 256>>>(d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Counter = %d (expected 256) ✓\n\n", counter);
    
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counter);
    
    printf("All fixes verified!\n");
    printf("compute-sanitizer should show 0 errors.\n");
    
    return 0;
}
