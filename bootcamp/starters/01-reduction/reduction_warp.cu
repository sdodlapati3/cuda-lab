/**
 * Warp-Shuffle Reduction: The Foundation Pattern
 * 
 * This kernel demonstrates THE most important optimization in CUDA:
 * using warp shuffle instructions to avoid shared memory for intra-warp communication.
 * 
 * Master this pattern → Apply to: scan, histogram, softmax, attention, everything.
 * 
 * Progression:
 *   V1: Naive (shared memory, many syncs)      → ~15% of peak bandwidth
 *   V2: Warp shuffle (no shared for warp)      → ~45% of peak bandwidth  
 *   V3: Warp shuffle + unrolling               → ~75% of peak bandwidth
 *   V4: Multiple elements per thread           → ~85% of peak bandwidth
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Error checking macro
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// V1: Naive Reduction (Baseline)
// 
// Problem: Uses shared memory for ALL communication, many __syncthreads()
// Why it's slow: Shared memory has ~30 cycle latency, syncs serialize warps
// ============================================================================
__global__ void reduce_naive(const float* __restrict__ input, 
                             float* __restrict__ output, 
                             int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduce in shared memory (sequential addressing)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // EXPENSIVE: Every iteration!
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================================
// V2: Warp Shuffle Reduction
// 
// Key insight: Threads in a warp execute in lockstep (SIMT)
// → No sync needed within a warp!
// → Use __shfl_down_sync instead of shared memory for last 32 elements
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // All 32 threads in warp participate
    // __shfl_down_sync: Get value from thread (lane_id + offset)
    //
    // Visual (8 threads shown, actual is 32):
    //   Offset 16: [0+16, 1+17, 2+18, 3+19, 4+20, 5+21, 6+22, 7+23]
    //   Offset 8:  [0+8,  1+9,  2+10, 3+11, 4,    5,    6,    7   ]
    //   Offset 4:  [0+4,  1+5,  2+6,  3+7,  ...]
    //   ...until offset 1
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Only lane 0 has the final sum
}

__global__ void reduce_warp_shuffle(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Load
    float val = (gid < n) ? input[gid] : 0.0f;
    
    // Warp-level reduction (no shared memory, no sync!)
    val = warp_reduce_sum(val);
    
    // First thread of each warp writes to shared memory
    if (lane == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction: first warp reduces all warp results
    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum(val);
        
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// ============================================================================
// V3: Multiple Elements Per Thread
// 
// Key insight: Increase arithmetic intensity by having each thread process
// multiple elements BEFORE the reduction phase.
// 
// This hides memory latency and reduces total reduction operations.
// ============================================================================
template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void reduce_multi_element(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n) {
    __shared__ float sdata[BLOCK_SIZE / 32];  // One slot per warp
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Each thread processes ELEMENTS_PER_THREAD elements
    // Grid-stride loop pattern for arbitrary sizes
    float thread_sum = 0.0f;
    
    int grid_size = BLOCK_SIZE * gridDim.x * ELEMENTS_PER_THREAD;
    int start = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + tid;
    
    // Strided access for coalescing
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = start + i * BLOCK_SIZE;
        if (idx < n) {
            thread_sum += input[idx];
        }
    }
    
    // Warp reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results
    if (lane == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (tid < BLOCK_SIZE / 32) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum(val);
        
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// ============================================================================
// Benchmark harness
// ============================================================================
void benchmark_reduction(const char* name, 
                         void (*kernel)(const float*, float*, int),
                         const float* d_input, float* d_output, int n,
                         int block_size, int shared_mem_size,
                         int warmup_runs, int timed_runs) {
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        int grid_size = (n + block_size - 1) / block_size;
        kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        int grid_size = (n + block_size - 1) / block_size;
        kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;
    
    // Calculate bandwidth (read input once)
    double bytes = (double)n * sizeof(float);
    double gb_per_s = (bytes / 1e9) / (avg_ms / 1e3);
    
    // Get peak bandwidth
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    double efficiency = 100.0 * gb_per_s / peak_bw;
    
    printf("%-25s: %8.2f μs | %7.1f GB/s | %5.1f%% peak\n", 
           name, avg_ms * 1000, gb_per_s, efficiency);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // 16M elements default
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║         WARP-SHUFFLE REDUCTION BENCHMARK                       ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    printf("║ Device: %-54s ║\n", prop.name);
    printf("║ Peak Bandwidth: %7.1f GB/s                                   ║\n", peak_bw);
    printf("║ Elements: %d (%d MB)                                   ║\n", 
           n, (int)(n * sizeof(float) / (1024 * 1024)));
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    // Allocate
    float *h_input = (float*)malloc(n * sizeof(float));
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    // Initialize with random values
    float expected_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
        expected_sum += h_input[i];
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int warmup = 5;
    int timed = 20;
    
    // Benchmark each version
    benchmark_reduction("V1: Naive (baseline)", 
                        reduce_naive, d_input, d_output, n,
                        block_size, block_size * sizeof(float), warmup, timed);
    
    benchmark_reduction("V2: Warp shuffle", 
                        reduce_warp_shuffle, d_input, d_output, n,
                        block_size, (block_size / 32) * sizeof(float), warmup, timed);
    
    // V3 with template - need wrapper
    auto v3_wrapper = [](const float* in, float* out, int n) {
        constexpr int BS = 256;
        constexpr int EPT = 8;
        int grid = (n + BS * EPT - 1) / (BS * EPT);
        reduce_multi_element<BS, EPT><<<grid, BS>>>(in, out, n);
    };
    
    // Manual benchmark for V3
    {
        constexpr int BS = 256;
        constexpr int EPT = 8;
        int grid = (n + BS * EPT - 1) / (BS * EPT);
        
        for (int i = 0; i < warmup; i++) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
            reduce_multi_element<BS, EPT><<<grid, BS>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < timed; i++) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
            reduce_multi_element<BS, EPT><<<grid, BS>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / timed;
        double bytes = (double)n * sizeof(float);
        double gb_per_s = (bytes / 1e9) / (avg_ms / 1e3);
        double efficiency = 100.0 * gb_per_s / peak_bw;
        
        printf("%-25s: %8.2f μs | %7.1f GB/s | %5.1f%% peak\n", 
               "V3: Multi-element (8x)", avg_ms * 1000, gb_per_s, efficiency);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Verify correctness
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    {
        constexpr int BS = 256;
        constexpr int EPT = 8;
        int grid = (n + BS * EPT - 1) / (BS * EPT);
        reduce_multi_element<BS, EPT><<<grid, BS>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Verification: Expected=%.2f, Got=%.2f, Diff=%.4f%%            \n", 
           expected_sum, gpu_sum, 100.0f * fabsf(expected_sum - gpu_sum) / expected_sum);
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    // Cleanup
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

/*
 * YOUR TURN: Exercises
 * 
 * 1. Add V4: Use vectorized loads (float4) for even better bandwidth
 * 
 * 2. Implement warp_reduce_max() and warp_reduce_min()
 *    Hint: Replace + with fmaxf/fminf
 * 
 * 3. Implement a "find index of max" kernel using warp shuffles
 *    Hint: Track both value and index, shuffle both
 * 
 * 4. Profile with Nsight Compute and identify:
 *    - Memory throughput achieved
 *    - Stall reasons (if any)
 *    - Why V3 is faster than V2
 * 
 * 5. Implement two-pass reduction for VERY large arrays:
 *    - Pass 1: Reduce to one value per block
 *    - Pass 2: Reduce block results to final answer
 */
