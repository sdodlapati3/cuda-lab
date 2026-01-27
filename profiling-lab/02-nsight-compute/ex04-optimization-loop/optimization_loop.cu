// optimization_loop.cu - Complete profile-optimize-reprofile workflow example
// This file contains multiple versions of a kernel, from naive to optimized
//
// Learning objectives:
// - Practice the iterative optimization workflow
// - Use Nsight Compute to guide optimization decisions
// - Measure improvement at each step
// - Understand which optimizations help which bottlenecks

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4096
#define BLOCK_SIZE 16

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
// Problem: Matrix Transpose
// We'll optimize this kernel through multiple iterations
// =============================================================================

// =============================================================================
// Version 0: Naive transpose - BASELINE
// Issues:
// - Uncoalesced global memory writes
// - No use of shared memory
// =============================================================================
__global__ void transpose_v0_naive(float *__restrict__ out, const float *__restrict__ in,
                                   int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Read is coalesced (adjacent threads read adjacent elements)
        // Write is NOT coalesced (adjacent threads write to different rows)
        out[x * height + y] = in[y * width + x];
    }
}

// =============================================================================
// Version 1: Shared memory transpose
// Improvement: Use shared memory to enable coalesced writes
// Issues:
// - Bank conflicts in shared memory
// =============================================================================
__global__ void transpose_v1_shared(float *__restrict__ out, const float *__restrict__ in,
                                    int width, int height)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Coalesced read from global memory to shared memory
    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Calculate transposed position
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Coalesced write from shared memory to global memory
    // BUT: Bank conflicts when reading tile[threadIdx.x][threadIdx.y]
    if (x < height && y < width)
    {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// =============================================================================
// Version 2: Shared memory with padding (no bank conflicts)
// Improvement: Add padding to avoid bank conflicts
// =============================================================================
__global__ void transpose_v2_padded(float *__restrict__ out, const float *__restrict__ in,
                                    int width, int height)
{
    // +1 padding eliminates bank conflicts
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x < height && y < width)
    {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// =============================================================================
// Version 3: Optimized with multiple elements per thread
// Improvement: Better instruction-level parallelism
// =============================================================================
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_v3_coarsened(float *__restrict__ out, const float *__restrict__ in,
                                       int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

// Each thread loads multiple elements
#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < width && (y + j) < height)
        {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < height && (y + j) < width)
        {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// =============================================================================
// Version 4: Final optimized version with all techniques
// =============================================================================
__global__ void transpose_v4_final(float *__restrict__ out, const float *__restrict__ in,
                                   int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    // Prefetch multiple elements
    float tmp[TILE_DIM / BLOCK_ROWS];

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x_in < width && (y_in + j) < height)
        {
            tmp[j / BLOCK_ROWS] = in[(y_in + j) * width + x_in];
        }
    }

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = tmp[j / BLOCK_ROWS];
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x_out < height && (y_out + j) < width)
        {
            out[(y_out + j) * height + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Verification function
void verify_transpose(float *h_in, float *h_out, int width, int height)
{
    int errors = 0;
    for (int y = 0; y < height && errors < 10; y++)
    {
        for (int x = 0; x < width && errors < 10; x++)
        {
            float expected = h_in[y * width + x];
            float actual = h_out[x * height + y];
            if (fabsf(expected - actual) > 1e-5f)
            {
                printf("Mismatch at (%d,%d): expected %f, got %f\n", x, y, expected, actual);
                errors++;
            }
        }
    }
    if (errors == 0)
    {
        printf("Verification PASSED\n");
    }
    else
    {
        printf("Verification FAILED (%d errors)\n", errors);
    }
}

typedef void (*transpose_fn)(float *, const float *, int, int);

void benchmark_transpose(const char *name, transpose_fn kernel,
                         dim3 grid, dim3 block,
                         float *d_out, float *d_in, int width, int height,
                         float *h_in, float *h_out)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel<<<grid, block>>>(d_out, d_in, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_out, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n%-25s: ", name);
    verify_transpose(h_in, h_out, width, height);

    // Benchmark
    int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<grid, block>>>(d_out, d_in, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float time_per_iter = ms / iterations;

    // Calculate effective bandwidth (2x because read + write)
    double bytes = 2.0 * width * height * sizeof(float);
    double gb = bytes / 1e9;
    double bandwidth = gb / (time_per_iter / 1000.0);

    printf("%-25s: %8.4f ms, %8.2f GB/s\n", name, time_per_iter, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
    int width = N;
    int height = N;

    printf("=== Matrix Transpose Optimization Loop ===\n");
    printf("Matrix size: %d x %d (%.2f MB)\n\n", width, height,
           width * height * sizeof(float) / 1e6);

    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    printf("Theoretical peak bandwidth: %.2f GB/s\n\n", peak_bw);

    // Allocate memory
    size_t size = width * height * sizeof(float);
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    // Initialize input
    for (int i = 0; i < width * height; i++)
    {
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    printf("=== Optimization Progression ===\n");

    // V0: Naive
    dim3 block0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid0((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    benchmark_transpose("V0: Naive", transpose_v0_naive, grid0, block0,
                        d_out, d_in, width, height, h_in, h_out);

    // V1: Shared memory
    benchmark_transpose("V1: Shared memory", transpose_v1_shared, grid0, block0,
                        d_out, d_in, width, height, h_in, h_out);

    // V2: Padded shared memory
    benchmark_transpose("V2: Padded (no bank conf)", transpose_v2_padded, grid0, block0,
                        d_out, d_in, width, height, h_in, h_out);

    // V3: Coarsened
    dim3 block3(TILE_DIM, BLOCK_ROWS);
    dim3 grid3((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    benchmark_transpose("V3: Coarsened", transpose_v3_coarsened, grid3, block3,
                        d_out, d_in, width, height, h_in, h_out);

    // V4: Final
    benchmark_transpose("V4: Final optimized", transpose_v4_final, grid3, block3,
                        d_out, d_in, width, height, h_in, h_out);

    printf("\n=== Profile Each Version ===\n");
    printf("Compare versions with Nsight Compute:\n\n");
    printf("# Profile V0 (baseline):\n");
    printf("ncu --kernel-name transpose_v0_naive -o v0_report ./optimization_loop\n\n");
    printf("# Profile V2 (with bank conflict fix):\n");
    printf("ncu --kernel-name transpose_v2_padded -o v2_report ./optimization_loop\n\n");
    printf("# Compare reports:\n");
    printf("ncu --import v0_report.ncu-rep --import v2_report.ncu-rep --page details\n\n");
    printf("# Key metrics to compare:\n");
    printf("  - Memory Throughput (should improve V0 -> V2)\n");
    printf("  - L1/TEX Hit Rate\n");
    printf("  - Shared Memory Bank Conflicts (V1 vs V2)\n");
    printf("  - Global Store Efficiency\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
