/**
 * atomic_patterns.cu - Atomic optimization patterns
 * 
 * Learning objectives:
 * - Reducing atomic contention
 * - Privatization pattern
 * - Warp aggregation
 */

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Version 1: All threads atomicAdd to one location (maximum contention)
__global__ void count_single_atomic(const int* data, int* count, int n, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        if (data[i] == target) {
            atomicAdd(count, 1);  // All threads compete!
        }
    }
}

// Version 2: Reduce within block first, then one atomic per block
__global__ void count_block_reduce(const int* data, int* count, int n, int target) {
    __shared__ int block_count;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (tid == 0) block_count = 0;
    __syncthreads();
    
    // Each thread counts locally
    int local_count = 0;
    for (int i = idx; i < n; i += stride) {
        if (data[i] == target) {
            local_count++;
        }
    }
    
    // Atomic to shared (faster than global)
    atomicAdd(&block_count, local_count);
    __syncthreads();
    
    // One thread does global atomic
    if (tid == 0) {
        atomicAdd(count, block_count);
    }
}

// Warp reduce for count
__device__ int warp_reduce_sum(int val) {
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

// Version 3: Warp reduce then atomic
__global__ void count_warp_reduce(const int* data, int* count, int n, int target) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Count locally
    int local_count = 0;
    for (int i = idx; i < n; i += stride) {
        if (data[i] == target) {
            local_count++;
        }
    }
    
    // Warp-level reduction
    local_count = warp_reduce_sum(local_count);
    
    // Lane 0 of each warp stores to shared
    __shared__ int warp_counts[8];  // 256 / 32 = 8 warps
    if (lane == 0) {
        warp_counts[warp_id] = local_count;
    }
    __syncthreads();
    
    // First warp reduces warp counts
    if (warp_id == 0) {
        int warp_sum = (lane < 8) ? warp_counts[lane] : 0;
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0) {
            atomicAdd(count, warp_sum);
        }
    }
}

// Test atomic contention
__global__ void contention_test(int* counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(counter, 1);
    }
}

int main() {
    printf("=== Atomic Optimization Patterns ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);
    
    // Test data
    const int N = 1 << 20;  // 1M elements
    const int TRIALS = 10;
    
    int* h_data = new int[N];
    int target = 42;
    int expected_count = 0;
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 100;
        if (h_data[i] == target) expected_count++;
    }
    printf("Data: %d elements, counting occurrences of %d (expected: %d)\n\n", 
           N, target, expected_count);
    
    int* d_data;
    int* d_count;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    int blocks = 256;
    
    printf("%-30s %-12s %-12s %-10s\n", "Version", "Time(us)", "Speedup", "Result");
    printf("----------------------------------------------------------------\n");
    
    float baseline_time;
    
    auto benchmark = [&](const char* name, auto kernel_fn) {
        int result;
        
        // Warmup
        cudaMemset(d_count, 0, sizeof(int));
        kernel_fn<<<blocks, BLOCK_SIZE>>>(d_data, d_count, N, target);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            cudaMemset(d_count, 0, sizeof(int));
            kernel_fn<<<blocks, BLOCK_SIZE>>>(d_data, d_count, N, target);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms = ms / TRIALS * 1000;  // Convert to microseconds
        
        cudaMemcpy(&result, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        static bool first = true;
        float speedup;
        if (first) {
            baseline_time = ms;
            speedup = 1.0f;
            first = false;
        } else {
            speedup = baseline_time / ms;
        }
        
        printf("%-30s %-12.1f %-12.2fx %-10d\n", name, ms, speedup, result);
    };
    
    benchmark("V1: Single atomic", count_single_atomic);
    benchmark("V2: Block reduce", count_block_reduce);
    benchmark("V3: Warp reduce", count_warp_reduce);
    
    // Contention scaling test
    printf("\n=== Contention Scaling Test ===\n");
    printf("All threads atomic to single location\n\n");
    
    printf("%-15s %-15s %-15s\n", "Blocks", "Time(us)", "Atomics/sec");
    printf("----------------------------------------------\n");
    
    int iterations = 1000;
    for (int test_blocks = 1; test_blocks <= 256; test_blocks *= 4) {
        cudaMemset(d_count, 0, sizeof(int));
        
        cudaEventRecord(start);
        contention_test<<<test_blocks, 256>>>(d_count, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        
        long long total_atomics = (long long)test_blocks * 256 * iterations;
        float atomics_per_sec = total_atomics / (ms / 1000) / 1e9;
        
        printf("%-15d %-15.1f %-15.2f B\n", test_blocks, ms * 1000, atomics_per_sec);
    }
    
    printf("\n=== Key Insights ===\n");
    printf("1. Single-location atomics = maximum contention\n");
    printf("2. Privatization: each block/warp accumulates locally\n");
    printf("3. Hierarchy: thread-local → warp → block → global\n");
    printf("4. More contention = less scaling with thread count\n");
    printf("5. Shared memory atomics are faster than global\n");
    printf("6. Warp-level reduction eliminates most atomics\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_count);
    delete[] h_data;
    
    return 0;
}
