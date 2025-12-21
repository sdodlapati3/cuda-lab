/**
 * Day 6: Complete Optimization Workflow
 * 
 * Demonstrates iterative optimization process with measurements.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Version 0: Naive baseline
// ============================================================================
__global__ void reduce_v0(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Terrible: atomic add from every thread
    if (idx < n) {
        atomicAdd(out, in[idx]);
    }
}

// ============================================================================
// Version 1: Shared memory reduction
// ============================================================================
__global__ void reduce_v1(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // Naive reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ============================================================================
// Version 2: Sequential addressing (no divergence)
// ============================================================================
__global__ void reduce_v2(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - no divergent branches
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ============================================================================
// Version 3: First add during load
// ============================================================================
__global__ void reduce_v3(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Add two elements during load
    float sum = 0.0f;
    if (idx < n) sum += in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ============================================================================
// Version 4: Warp-level reduction (no sync needed in warp)
// ============================================================================
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) sum += in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Unroll last iterations
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ============================================================================
// Version 5: Shuffle reduction (modern GPUs)
// ============================================================================
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v5(const float* in, float* out, int n) {
    float sum = 0.0f;
    
    // Grid-stride loop for arbitrary sizes
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        sum += in[idx];
    }
    
    // Warp reduction
    sum = warpReduceSum(sum);
    
    // Reduce across warps
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warp == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        
        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}

// ============================================================================
// Benchmark function
// ============================================================================
struct Result {
    std::string name;
    float time_ms;
    float result;
    float speedup;
};

template<typename Func>
Result benchmark(const char* name, Func kernel, const float* d_in, float* d_out,
                 int n, int blockSize, float baseline_time = 0.0f) {
    
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    
    // Calculate grid size based on kernel version
    int numBlocks = (n + blockSize - 1) / blockSize;
    int sharedMem = blockSize * sizeof(float);
    
    // Warmup
    kernel<<<numBlocks, blockSize, sharedMem>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        kernel<<<numBlocks, blockSize, sharedMem>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    float speedup = (baseline_time > 0) ? baseline_time / ms : 1.0f;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return {name, ms, result, speedup};
}

int main() {
    printf("Optimization Workflow: Parallel Reduction\n");
    printf("==========================================\n\n");
    
    const int N = 1 << 24;  // 16M elements
    int blockSize = 256;
    
    // Allocate
    float* h_in = (float*)malloc(N * sizeof(float));
    float* d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    
    // Initialize
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;  // Easy to verify: sum = N
        expected += h_in[i];
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Problem: Sum of %d elements (expected: %.0f)\n\n", N, expected);
    
    // Benchmark each version
    std::vector<Result> results;
    
    printf("Running optimizations...\n\n");
    
    results.push_back(benchmark("V0: Naive atomics", reduce_v0, d_in, d_out, N, blockSize));
    float baseline = results[0].time_ms;
    
    results.push_back(benchmark("V1: Shared memory", reduce_v1, d_in, d_out, N, blockSize, baseline));
    results.push_back(benchmark("V2: Sequential addressing", reduce_v2, d_in, d_out, N, blockSize, baseline));
    results.push_back(benchmark("V3: First add during load", reduce_v3, d_in, d_out, N, blockSize, baseline));
    results.push_back(benchmark("V4: Warp unrolling", reduce_v4, d_in, d_out, N, blockSize, baseline));
    results.push_back(benchmark("V5: Shuffle reduction", reduce_v5, d_in, d_out, N, blockSize, baseline));
    
    // Print results
    printf("%-30s %10s %10s %10s\n", "Version", "Time(ms)", "Speedup", "Correct");
    printf("%-30s %10s %10s %10s\n", "-------", "--------", "-------", "-------");
    
    for (const auto& r : results) {
        bool correct = (fabs(r.result - expected) / expected < 0.001f);
        printf("%-30s %10.3f %10.1fx %10s\n", 
               r.name.c_str(), r.time_ms, r.speedup, correct ? "YES" : "NO");
    }
    
    printf("\n=== Optimization Journey ===\n");
    printf("V0 → V1: Replace global atomics with shared memory reduction\n");
    printf("V1 → V2: Fix warp divergence with sequential addressing\n");
    printf("V2 → V3: Increase work per thread (2 elements on load)\n");
    printf("V3 → V4: Unroll last warp (no __syncthreads needed)\n");
    printf("V4 → V5: Use shuffle instructions (fastest warp reduction)\n");
    
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    
    return 0;
}
