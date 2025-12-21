/**
 * sync_demo.cu - Understanding __syncthreads()
 * 
 * Learning objectives:
 * - When synchronization is needed
 * - What happens without it
 * - The cost of sync
 */

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

// Buggy version: Missing __syncthreads()
__global__ void shift_left_buggy(float* data, int n) {
    __shared__ float sdata[BLOCK_SIZE + 1];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < n) {
        sdata[tid] = data[idx];
    }
    if (tid == blockDim.x - 1 && idx + 1 < n) {
        sdata[tid + 1] = data[idx + 1];
    }
    
    // BUG: No __syncthreads()!
    // Other threads may not have finished writing sdata yet!
    
    // Read neighbor - may get stale data
    if (idx < n - 1) {
        data[idx] = sdata[tid + 1];
    }
}

// Correct version: With __syncthreads()
__global__ void shift_left_correct(float* data, int n) {
    __shared__ float sdata[BLOCK_SIZE + 1];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < n) {
        sdata[tid] = data[idx];
    }
    if (tid == blockDim.x - 1 && idx + 1 < n) {
        sdata[tid + 1] = data[idx + 1];
    }
    
    __syncthreads();  // CORRECT: Wait for all loads
    
    // Now safe to read neighbor
    if (idx < n - 1) {
        data[idx] = sdata[tid + 1];
    }
}

// Demonstrate reduction sync pattern
__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();  // Sync after load
    
    // Reduction with sync at each step
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // MUST sync each iteration!
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Show sync cost
__global__ void sync_cost_test(float* data, int iterations) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    sdata[tid] = data[tid];
    
    // Multiple syncs to measure overhead
    for (int i = 0; i < iterations; i++) {
        __syncthreads();
        sdata[tid] += 1.0f;
    }
    
    data[tid] = sdata[tid];
}

int main() {
    printf("=== Understanding __syncthreads() ===\n\n");
    
    const int N = 1024;
    float* h_data = new float[N];
    float* h_result = new float[N];
    
    // Initialize: 0, 1, 2, 3, ...
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    printf("=== Test 1: Shift Left (Race Condition Demo) ===\n");
    printf("Expected: [1, 2, 3, 4, ...] (each element replaced by next)\n\n");
    
    // Buggy version
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    shift_left_buggy<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, N);
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Buggy (no sync): ");
    int buggy_errors = 0;
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_result[i]);
        if (i < N - 1 && h_result[i] != i + 1) buggy_errors++;
    }
    printf("...\n");
    
    // Correct version
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    shift_left_correct<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, N);
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Correct (with sync): ");
    int correct_errors = 0;
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_result[i]);
        if (i < N - 1 && h_result[i] != i + 1) correct_errors++;
    }
    printf("...\n");
    
    printf("\nNote: Buggy version may appear to work sometimes!\n");
    printf("Race conditions are non-deterministic - they depend on timing.\n");
    printf("This makes them very hard to debug.\n");
    
    printf("\n=== Test 2: Sync Cost ===\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    cudaMemcpy(d_data, h_data, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    sync_cost_test<<<1, BLOCK_SIZE>>>(d_data, 1);
    cudaDeviceSynchronize();
    
    int iterations[] = {1, 10, 100, 1000};
    printf("%-15s %-15s %-15s\n", "Iterations", "Time (us)", "Time per sync (ns)");
    printf("------------------------------------------------\n");
    
    for (int iter : iterations) {
        cudaEventRecord(start);
        sync_cost_test<<<1, BLOCK_SIZE>>>(d_data, iter);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        
        float ns_per_sync = ms * 1e6 / iter;
        printf("%-15d %-15.2f %-15.1f\n", iter, ms * 1000, ns_per_sync);
    }
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. __syncthreads() is a BARRIER - all threads must reach it\n");
    printf("2. Use AFTER writes to shared memory, BEFORE reads\n");
    printf("3. NEVER place in divergent code (if/else where threads split)\n");
    printf("4. Cost is small but not zero - minimize when possible\n");
    printf("5. Missing sync = race condition = non-deterministic bugs!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    delete[] h_data;
    delete[] h_result;
    
    return 0;
}
