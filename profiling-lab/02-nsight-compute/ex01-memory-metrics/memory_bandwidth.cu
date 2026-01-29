// memory_bandwidth.cu - Kernels demonstrating different memory access patterns
// Profile with: ncu --section MemoryWorkloadAnalysis ./memory_bandwidth
//
// Learning objectives:
// - Measure achieved vs theoretical memory bandwidth
// - Understand coalesced vs uncoalesced access
// - Analyze L1/L2 cache hit rates

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 24) // 16M elements
#define BLOCK_SIZE 256

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
// Kernel 1: Coalesced memory access (GOOD)
// Adjacent threads access adjacent memory locations
// Expected: High memory bandwidth utilization
// =============================================================================
__global__ void coalesced_copy(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// Kernel 2: Strided memory access (BAD)
// Threads access memory with a stride - causes multiple memory transactions
// Expected: Low memory bandwidth utilization
// =============================================================================
__global__ void strided_copy(float *__restrict__ dst, const float *__restrict__ src, int n, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (idx < n)
    {
        dst[idx] = src[strided_idx];
    }
}

// =============================================================================
// Kernel 3: Random access pattern (WORST)
// Each thread accesses a pseudo-random location
// Expected: Very low bandwidth, poor cache utilization
// =============================================================================
__global__ void random_copy(float *__restrict__ dst, const float *__restrict__ src,
                            const int *__restrict__ indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = src[indices[idx]];
    }
}

// =============================================================================
// Kernel 4: Reduction - demonstrates memory read patterns
// Good for measuring memory bandwidth in reduction operations
// =============================================================================
__global__ void naive_reduce(const float *__restrict__ input, float *__restrict__ output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global memory to shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory (naive - has bank conflicts)
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Kernel 5: Optimized reduction - better memory access pattern
// Sequential addressing eliminates divergent warps
// =============================================================================
__global__ void optimized_reduce(const float *__restrict__ input, float *__restrict__ output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Sequential addressing - no divergent branches
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Helper function to generate random indices
void generate_random_indices(int *indices, int n)
{
    for (int i = 0; i < n; i++)
    {
        indices[i] = rand() % n;
    }
}

void run_and_time(const char *name, void (*kernel_launcher)(), int iterations)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel_launcher();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        kernel_launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Calculate bandwidth: 2 * N * sizeof(float) for copy (read + write)
    double bytes = 2.0 * N * sizeof(float) * iterations;
    double gb = bytes / 1e9;
    double seconds = ms / 1000.0;
    double bandwidth = gb / seconds;

    printf("%-25s: %8.2f ms, %8.2f GB/s\n", name, ms / iterations, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Global pointers for kernel launchers
float *d_src, *d_dst;
int *d_indices;
float *d_reduce_out;
int numBlocks;

void launch_coalesced()
{
    coalesced_copy<<<numBlocks, BLOCK_SIZE>>>(d_dst, d_src, N);
}

void launch_strided()
{
    strided_copy<<<numBlocks, BLOCK_SIZE>>>(d_dst, d_src, N, 32);
}

void launch_random()
{
    random_copy<<<numBlocks, BLOCK_SIZE>>>(d_dst, d_src, d_indices, N);
}

void launch_naive_reduce()
{
    naive_reduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_src, d_reduce_out, N);
}

void launch_optimized_reduce()
{
    optimized_reduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_src, d_reduce_out, N);
}

int main(int argc, char **argv)
{
    printf("=== Memory Bandwidth Analysis ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", N, N * sizeof(float) / 1e6);

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    // Note: memoryClockRate deprecated in CUDA 12+, just print bus width
    printf("Memory bus width: %d bits\n\n", prop.memoryBusWidth);

    // Allocate host memory
    float *h_src = (float *)malloc(N * sizeof(float));
    int *h_indices = (int *)malloc(N * sizeof(int));

    // Initialize data
    for (int i = 0; i < N; i++)
    {
        h_src[i] = 1.0f;
    }
    generate_random_indices(h_indices, N);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));

    numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_reduce_out, numBlocks * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("Running memory access pattern benchmarks...\n");
    printf("%-25s  %10s  %10s\n", "Kernel", "Time", "Bandwidth");
    printf("-----------------------------------------------\n");

    int iterations = 10;
    run_and_time("Coalesced copy", launch_coalesced, iterations);
    run_and_time("Strided copy (stride=32)", launch_strided, iterations);
    run_and_time("Random access", launch_random, iterations);
    run_and_time("Naive reduction", launch_naive_reduce, iterations);
    run_and_time("Optimized reduction", launch_optimized_reduce, iterations);

    printf("\n=== Profile Individual Kernels with Nsight Compute ===\n");
    printf("ncu --kernel-name coalesced_copy ./memory_bandwidth\n");
    printf("ncu --kernel-name strided_copy ./memory_bandwidth\n");
    printf("ncu --kernel-name random_copy ./memory_bandwidth\n");
    printf("ncu --section MemoryWorkloadAnalysis ./memory_bandwidth\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_reduce_out));
    free(h_src);
    free(h_indices);

    return 0;
}
