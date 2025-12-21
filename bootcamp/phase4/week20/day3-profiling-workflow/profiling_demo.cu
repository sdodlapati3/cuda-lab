/**
 * Profiling Demo: Before and After Optimization
 * 
 * This example shows common performance issues and their fixes.
 * Profile with: nsys profile ./profiling_demo
 *              ncu --set full ./profiling_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16384
#define BLOCK_SIZE 256

// ============================================================================
// Example 1: Non-coalesced vs Coalesced Memory Access
// ============================================================================

// BAD: Column-major access (strided)
__global__ void matrixCopy_bad(const float* src, float* dst, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        // Non-coalesced: threads access different rows
        dst[col * height + row] = src[col * height + row];
    }
}

// GOOD: Row-major access (coalesced)
__global__ void matrixCopy_good(const float* src, float* dst, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        // Coalesced: consecutive threads access consecutive memory
        dst[row * width + col] = src[row * width + col];
    }
}

// ============================================================================
// Example 2: Bank Conflicts in Shared Memory
// ============================================================================

// BAD: Strided access causes bank conflicts
__global__ void reduceSum_bad(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    sdata[tid + blockDim.x] = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Reduction with strided access (bad)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// GOOD: Sequential access avoids bank conflicts
__global__ void reduceSum_good(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    sdata[tid] += (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Reduction with sequential access (good)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// ============================================================================
// Example 3: Warp Divergence
// ============================================================================

// BAD: Significant warp divergence
__global__ void process_bad(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    
    // Different threads in same warp take different paths
    if (idx % 7 == 0) {
        val = sinf(val) * cosf(val);
    } else if (idx % 7 == 1) {
        val = sqrtf(fabsf(val));
    } else if (idx % 7 == 2) {
        val = logf(fabsf(val) + 1.0f);
    } else if (idx % 7 == 3) {
        val = expf(-val * val);
    } else {
        val = val * val;
    }
    
    data[idx] = val;
}

// GOOD: Minimal divergence (all threads same path)
__global__ void process_good(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    
    // All threads execute same operations
    val = sinf(val);
    val = val * val;
    val = sqrtf(fabsf(val) + 0.001f);
    
    data[idx] = val;
}

// ============================================================================
// Benchmark function
// ============================================================================

template<typename KernelFunc>
float benchmark(const char* name, KernelFunc kernel, dim3 grid, dim3 block, 
                size_t sharedMem, int iterations) {
    // Warm-up
    kernel<<<grid, block, sharedMem>>>();
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block, sharedMem>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iterations;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("  %-25s: %.3f ms\n", name, ms);
    return ms;
}

int main() {
    printf("=== Profiling Demo: Performance Comparison ===\n\n");
    
    int iterations = 100;
    
    // ========== Example 1: Memory Coalescing ==========
    printf("Example 1: Memory Coalescing\n");
    
    int width = 4096, height = 4096;
    float *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * sizeof(float));
    cudaMalloc(&d_dst, width * height * sizeof(float));
    
    dim3 block2d(16, 16);
    dim3 grid2d((width + 15) / 16, (height + 15) / 16);
    
    auto copy_bad = [=]() { matrixCopy_bad<<<grid2d, block2d>>>(d_src, d_dst, width, height); };
    auto copy_good = [=]() { matrixCopy_good<<<grid2d, block2d>>>(d_src, d_dst, width, height); };
    
    float bad1 = benchmark("Non-coalesced", copy_bad, grid2d, block2d, 0, iterations);
    float good1 = benchmark("Coalesced", copy_good, grid2d, block2d, 0, iterations);
    printf("  Speedup: %.2fx\n\n", bad1 / good1);
    
    cudaFree(d_src);
    cudaFree(d_dst);
    
    // ========== Example 2: Bank Conflicts ==========
    printf("Example 2: Shared Memory Bank Conflicts\n");
    
    int n = 16 * 1024 * 1024;
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    dim3 block1d(BLOCK_SIZE);
    dim3 grid1d((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    size_t sharedSize = 2 * BLOCK_SIZE * sizeof(float);
    
    auto reduce_bad = [=]() { 
        cudaMemset(d_output, 0, sizeof(float));
        reduceSum_bad<<<grid1d, block1d, sharedSize>>>(d_input, d_output, n); 
    };
    auto reduce_good = [=]() { 
        cudaMemset(d_output, 0, sizeof(float));
        reduceSum_good<<<grid1d, block1d, sharedSize>>>(d_input, d_output, n); 
    };
    
    float bad2 = benchmark("Strided reduction", reduce_bad, grid1d, block1d, sharedSize, iterations);
    float good2 = benchmark("Sequential reduction", reduce_good, grid1d, block1d, sharedSize, iterations);
    printf("  Speedup: %.2fx\n\n", bad2 / good2);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    // ========== Example 3: Warp Divergence ==========
    printf("Example 3: Warp Divergence\n");
    
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    auto process_v1 = [=]() { process_bad<<<grid1d, block1d>>>(d_data, n); };
    auto process_v2 = [=]() { process_good<<<grid1d, block1d>>>(d_data, n); };
    
    float bad3 = benchmark("With divergence", process_v1, grid1d, block1d, 0, iterations);
    float good3 = benchmark("No divergence", process_v2, grid1d, block1d, 0, iterations);
    printf("  Speedup: %.2fx\n\n", bad3 / good3);
    
    cudaFree(d_data);
    
    // ========== Summary ==========
    printf("=== Profiling Tips ===\n");
    printf("1. Run: nsys profile ./profiling_demo\n");
    printf("2. Run: ncu --set full ./profiling_demo\n");
    printf("3. Look for:\n");
    printf("   - Memory throughput (should be high)\n");
    printf("   - Shared memory efficiency (should be 100%%)\n");
    printf("   - Warp execution efficiency (should be >80%%)\n");
    
    return 0;
}
