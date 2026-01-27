// compute_metrics.cu - Kernels for analyzing compute throughput and occupancy
// Profile with: ncu --section ComputeWorkloadAnalysis ./compute_metrics
//
// Learning objectives:
// - Understand occupancy and its impact on performance
// - Measure compute throughput (FLOPS)
// - Analyze warp execution efficiency
// - Identify compute-bound vs latency-bound kernels

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 22) // 4M elements
#define BLOCK_SIZE_LOW 64
#define BLOCK_SIZE_MED 256
#define BLOCK_SIZE_HIGH 512

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// =============================================================================
// Kernel 1: Low arithmetic intensity (memory-bound)
// Simple element-wise operation - 1 FLOP per 2 memory ops
// =============================================================================
__global__ void low_intensity(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = src[idx] + 1.0f; // 1 ADD
    }
}

// =============================================================================
// Kernel 2: Medium arithmetic intensity
// Multiple operations per element
// =============================================================================
__global__ void medium_intensity(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];
        // 10 FLOPS per element
        val = val * val + val;        // 2 ops
        val = val * 2.0f - 1.0f;      // 2 ops
        val = val * val + val * 0.5f; // 3 ops
        val = val + val * val;        // 2 ops
        dst[idx] = val + 1.0f;        // 1 op
    }
}

// =============================================================================
// Kernel 3: High arithmetic intensity (compute-bound)
// Many operations per memory access
// =============================================================================
__global__ void high_intensity(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];

// ~50 FLOPS per element
#pragma unroll
        for (int i = 0; i < 10; i++)
        {
            val = val * val + val;         // 2 ops
            val = sqrtf(val * val + 1.0f); // 3 ops (approx)
        }
        dst[idx] = val;
    }
}

// =============================================================================
// Kernel 4: Divergent branches (low warp efficiency)
// Different threads take different paths
// =============================================================================
__global__ void divergent_kernel(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];

        // Divergent branch based on thread ID within warp
        if (threadIdx.x % 2 == 0)
        {
            val = val * val + val;
            val = val * 2.0f;
        }
        else
        {
            val = val + val + val;
            val = val * 0.5f;
            val = val + 1.0f;
        }
        dst[idx] = val;
    }
}

// =============================================================================
// Kernel 5: Uniform branches (high warp efficiency)
// All threads in a warp take the same path
// =============================================================================
__global__ void uniform_kernel(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];

        // Uniform branch - based on warp ID, not thread ID within warp
        int warp_id = idx / 32;
        if (warp_id % 2 == 0)
        {
            val = val * val + val;
            val = val * 2.0f;
        }
        else
        {
            val = val + val + val;
            val = val * 0.5f;
            val = val + 1.0f;
        }
        dst[idx] = val;
    }
}

// =============================================================================
// Kernel 6: Register pressure test
// Uses many registers per thread - limits occupancy
// =============================================================================
__global__ void high_register_pressure(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Declare many variables to increase register pressure
        float r0 = src[idx];
        float r1 = r0 * 1.1f;
        float r2 = r1 * 1.2f;
        float r3 = r2 * 1.3f;
        float r4 = r3 * 1.4f;
        float r5 = r4 * 1.5f;
        float r6 = r5 * 1.6f;
        float r7 = r6 * 1.7f;
        float r8 = r7 * 1.8f;
        float r9 = r8 * 1.9f;
        float r10 = r9 * 2.0f;
        float r11 = r10 * 2.1f;
        float r12 = r11 * 2.2f;
        float r13 = r12 * 2.3f;
        float r14 = r13 * 2.4f;
        float r15 = r14 * 2.5f;

        // Use all registers to prevent optimization
        dst[idx] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                   r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    }
}

// =============================================================================
// Kernel 7: Shared memory intensive - limits occupancy differently
// =============================================================================
__global__ void shared_memory_intensive(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    // Large shared memory allocation limits blocks per SM
    __shared__ float smem[4096]; // 16KB per block

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Fill shared memory
    for (int i = tid; i < 4096; i += blockDim.x)
    {
        smem[i] = (float)i;
    }
    __syncthreads();

    if (idx < n)
    {
        float val = src[idx];
        val += smem[tid % 4096];
        dst[idx] = val;
    }
}

void benchmark_kernel(const char *name, dim3 grid, dim3 block,
                      void (*kernel)(float *, const float *, int),
                      float *d_dst, float *d_src, int n)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel<<<grid, block>>>(d_dst, d_src, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<grid, block>>>(d_dst, d_src, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("%-25s [%4d threads]: %8.3f ms avg\n", name, block.x, ms / iterations);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
    printf("=== Compute Metrics Analysis ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", N, N * sizeof(float) / 1e6);

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

    // Allocate memory
    float *h_src = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        h_src[i] = 1.0f + (float)(i % 100) / 100.0f;
    }

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("=== Arithmetic Intensity Tests ===\n");
    int numBlocks = (N + BLOCK_SIZE_MED - 1) / BLOCK_SIZE_MED;
    benchmark_kernel("Low intensity", numBlocks, BLOCK_SIZE_MED, low_intensity, d_dst, d_src, N);
    benchmark_kernel("Medium intensity", numBlocks, BLOCK_SIZE_MED, medium_intensity, d_dst, d_src, N);
    benchmark_kernel("High intensity", numBlocks, BLOCK_SIZE_MED, high_intensity, d_dst, d_src, N);

    printf("\n=== Warp Efficiency Tests ===\n");
    benchmark_kernel("Divergent branches", numBlocks, BLOCK_SIZE_MED, divergent_kernel, d_dst, d_src, N);
    benchmark_kernel("Uniform branches", numBlocks, BLOCK_SIZE_MED, uniform_kernel, d_dst, d_src, N);

    printf("\n=== Occupancy Limiters ===\n");
    benchmark_kernel("High register pressure", numBlocks, BLOCK_SIZE_MED, high_register_pressure, d_dst, d_src, N);
    benchmark_kernel("Shared mem intensive", (N + 256 - 1) / 256, 256, shared_memory_intensive, d_dst, d_src, N);

    printf("\n=== Block Size Impact on Occupancy ===\n");
    printf("Testing medium_intensity kernel with different block sizes:\n");
    for (int bs : {32, 64, 128, 256, 512, 1024})
    {
        int nb = (N + bs - 1) / bs;
        benchmark_kernel("Medium intensity", nb, bs, medium_intensity, d_dst, d_src, N);
    }

    printf("\n=== Profile Commands ===\n");
    printf("ncu --section ComputeWorkloadAnalysis ./compute_metrics\n");
    printf("ncu --section Occupancy ./compute_metrics\n");
    printf("ncu --section WarpStateStatistics ./compute_metrics\n");
    printf("ncu --kernel-name divergent_kernel --section WarpStateStatistics ./compute_metrics\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);

    return 0;
}
