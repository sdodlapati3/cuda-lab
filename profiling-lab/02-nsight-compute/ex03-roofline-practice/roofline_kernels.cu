// roofline_kernels.cu - Kernels designed to demonstrate roofline analysis
// Profile with: ncu --set roofline ./roofline_kernels
//
// Learning objectives:
// - Understand the roofline model
// - Plot kernels on a roofline diagram
// - Identify memory-bound vs compute-bound kernels
// - Calculate arithmetic intensity

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
// Roofline Model Basics:
//
// Performance = min(Peak_FLOPS, Peak_Bandwidth × Arithmetic_Intensity)
//
// Arithmetic Intensity (AI) = FLOPS / Bytes_Moved
//
// If AI < Ridge Point: Memory-bound
// If AI > Ridge Point: Compute-bound
//
// Ridge Point = Peak_FLOPS / Peak_Bandwidth
// =============================================================================

// =============================================================================
// Kernel 1: STREAM Copy - AI ≈ 0 FLOP/Byte (pure memory bound)
// Just copies data - no arithmetic
// =============================================================================
__global__ void stream_copy(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = src[idx];
    }
}
// AI = 0 FLOP / 8 Bytes = 0 FLOP/Byte

// =============================================================================
// Kernel 2: STREAM Scale - AI = 0.125 FLOP/Byte (memory bound)
// One multiply per element
// =============================================================================
__global__ void stream_scale(float *__restrict__ dst, const float *__restrict__ src,
                             float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = scalar * src[idx];
    }
}
// AI = 1 FLOP / 8 Bytes = 0.125 FLOP/Byte

// =============================================================================
// Kernel 3: STREAM Add - AI = 0.125 FLOP/Byte (memory bound)
// One add per two reads, one write
// =============================================================================
__global__ void stream_add(float *__restrict__ c, const float *__restrict__ a,
                           const float *__restrict__ b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
// AI = 1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte

// =============================================================================
// Kernel 4: STREAM Triad - AI = 0.167 FLOP/Byte (memory bound)
// a[i] = b[i] + scalar * c[i]
// =============================================================================
__global__ void stream_triad(float *__restrict__ a, const float *__restrict__ b,
                             const float *__restrict__ c, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] = b[idx] + scalar * c[idx];
    }
}
// AI = 2 FLOPS / 12 Bytes ≈ 0.167 FLOP/Byte

// =============================================================================
// Kernel 5: SAXPY with high reuse - AI ≈ 1 FLOP/Byte (near ridge point)
// Multiple operations reusing loaded data
// =============================================================================
__global__ void saxpy_extended(float *__restrict__ y, const float *__restrict__ x,
                               float a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float xi = x[idx];
        float yi = y[idx];
        // 8 FLOPS: 4 multiplies + 4 adds
        yi = a * xi + yi;
        yi = a * xi + yi;
        yi = a * xi + yi;
        yi = a * xi + yi;
        y[idx] = yi;
    }
}
// AI = 8 FLOPS / 8 Bytes = 1 FLOP/Byte

// =============================================================================
// Kernel 6: Medium compute intensity - AI ≈ 5 FLOP/Byte
// =============================================================================
__global__ void medium_compute(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];
// ~40 FLOPS
#pragma unroll
        for (int i = 0; i < 10; i++)
        {
            val = val * val + val;     // 2 FLOPS
            val = val * 0.99f + 0.01f; // 2 FLOPS
        }
        dst[idx] = val;
    }
}
// AI = 40 FLOPS / 8 Bytes = 5 FLOP/Byte

// =============================================================================
// Kernel 7: High compute intensity - AI ≈ 25 FLOP/Byte (compute bound)
// Heavy computation with minimal memory access
// =============================================================================
__global__ void high_compute(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];
// ~200 FLOPS
#pragma unroll
        for (int i = 0; i < 50; i++)
        {
            val = val * val + val;       // 2 FLOPS
            val = val * 0.999f + 0.001f; // 2 FLOPS
        }
        dst[idx] = val;
    }
}
// AI = 200 FLOPS / 8 Bytes = 25 FLOP/Byte

// =============================================================================
// Kernel 8: Very high compute intensity - AI ≈ 100 FLOP/Byte (deep compute bound)
// =============================================================================
__global__ void very_high_compute(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = src[idx];
// ~800 FLOPS
#pragma unroll
        for (int i = 0; i < 100; i++)
        {
            val = fmaf(val, val, val); // FMA = 2 FLOPS
            val = fmaf(val, 0.9999f, 0.0001f);
            val = fmaf(val, val, val);
            val = fmaf(val, 0.9999f, 0.0001f);
        }
        dst[idx] = val;
    }
}
// AI = 800 FLOPS / 8 Bytes = 100 FLOP/Byte

// =============================================================================
// Kernel 9: Matrix-like access pattern (for comparison)
// Simulates matrix row access - different memory pattern
// =============================================================================
#define TILE_SIZE 32
__global__ void matrix_row_sum(float *__restrict__ row_sums, const float *__restrict__ matrix,
                               int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++)
        {
            sum += matrix[row * cols + col];
        }
        row_sums[row] = sum;
    }
}

void run_roofline_test(const char *name, float ai,
                       void (*launcher)(float *, const float *, float *, int),
                       float *d_dst, float *d_src, float *d_aux, int n, int block_size)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int numBlocks = (n + block_size - 1) / block_size;

    // Warmup
    launcher(d_dst, d_src, d_aux, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int iterations = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        launcher(d_dst, d_src, d_aux, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float time_per_iter = ms / iterations;

    // Calculate metrics
    // Assuming 8 bytes per element (1 read + 1 write of float)
    double bytes = 2.0 * n * sizeof(float);
    double gb = bytes / 1e9;
    double seconds = time_per_iter / 1000.0;
    double bandwidth = gb / seconds;
    double gflops = (ai * bytes) / 1e9 / seconds;

    printf("%-20s | AI: %6.2f | %8.3f ms | %8.2f GB/s | %8.2f GFLOP/s\n",
           name, ai, time_per_iter, bandwidth, gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Launcher wrappers
float *g_dst, *g_src, *g_aux;
int g_n, g_block_size;

void launch_copy(float *dst, const float *src, float *aux, int n)
{
    stream_copy<<<(n + 256 - 1) / 256, 256>>>(dst, src, n);
}

void launch_scale(float *dst, const float *src, float *aux, int n)
{
    stream_scale<<<(n + 256 - 1) / 256, 256>>>(dst, src, 2.0f, n);
}

void launch_add(float *dst, const float *src, float *aux, int n)
{
    stream_add<<<(n + 256 - 1) / 256, 256>>>(dst, src, aux, n);
}

void launch_triad(float *dst, const float *src, float *aux, int n)
{
    stream_triad<<<(n + 256 - 1) / 256, 256>>>(dst, src, aux, 2.0f, n);
}

void launch_saxpy_ext(float *dst, const float *src, float *aux, int n)
{
    CUDA_CHECK(cudaMemcpy(dst, aux, n * sizeof(float), cudaMemcpyDeviceToDevice));
    saxpy_extended<<<(n + 256 - 1) / 256, 256>>>(dst, src, 2.0f, n);
}

void launch_medium(float *dst, const float *src, float *aux, int n)
{
    medium_compute<<<(n + 256 - 1) / 256, 256>>>(dst, src, n);
}

void launch_high(float *dst, const float *src, float *aux, int n)
{
    high_compute<<<(n + 256 - 1) / 256, 256>>>(dst, src, n);
}

void launch_very_high(float *dst, const float *src, float *aux, int n)
{
    very_high_compute<<<(n + 256 - 1) / 256, 256>>>(dst, src, n);
}

int main(int argc, char **argv)
{
    int n = 1 << 24; // 16M elements

    printf("=== Roofline Analysis Kernels ===\n\n");

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);

    // Calculate theoretical peaks
    float peak_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6; // GB/s
    float peak_flops = prop.multiProcessorCount * prop.clockRate * 2.0f / 1e6;            // Rough estimate
    float ridge_point = peak_flops / peak_bandwidth;

    printf("Estimated Peak Memory Bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Estimated Peak Compute: %.2f GFLOP/s (FP32)\n", peak_flops);
    printf("Ridge Point: %.2f FLOP/Byte\n\n", ridge_point);

    printf("NOTE: Use 'ncu --set roofline' for accurate roofline measurements!\n\n");

    // Allocate memory
    float *h_src = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        h_src[i] = 1.0f + (float)(i % 1000) / 10000.0f;
    }

    float *d_src, *d_dst, *d_aux;
    CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aux, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_src, n * sizeof(float), cudaMemcpyHostToDevice));

    printf("%-20s | %-10s | %10s | %12s | %12s\n",
           "Kernel", "AI (F/B)", "Time", "Bandwidth", "Compute");
    printf("-------------------------------------------------------------------------\n");

    // Run tests with different arithmetic intensities
    run_roofline_test("STREAM Copy", 0.0f, launch_copy, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("STREAM Scale", 0.125f, launch_scale, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("STREAM Add", 0.083f, launch_add, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("STREAM Triad", 0.167f, launch_triad, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("SAXPY Extended", 1.0f, launch_saxpy_ext, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("Medium Compute", 5.0f, launch_medium, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("High Compute", 25.0f, launch_high, d_dst, d_src, d_aux, n, 256);
    run_roofline_test("Very High Compute", 100.0f, launch_very_high, d_dst, d_src, d_aux, n, 256);

    printf("\n=== Roofline Interpretation ===\n");
    printf("Memory-bound kernels: Copy, Scale, Add, Triad (AI < %.1f)\n", ridge_point);
    printf("Compute-bound kernels: High Compute, Very High Compute (AI > %.1f)\n", ridge_point);
    printf("Transition zone: Medium Compute (near ridge point)\n");

    printf("\n=== Nsight Compute Commands ===\n");
    printf("# Generate roofline chart:\n");
    printf("ncu --set roofline -o roofline_report ./roofline_kernels\n\n");
    printf("# View in GUI:\n");
    printf("ncu-ui roofline_report.ncu-rep\n\n");
    printf("# Specific kernel roofline:\n");
    printf("ncu --kernel-name stream_copy --set roofline ./roofline_kernels\n");
    printf("ncu --kernel-name high_compute --set roofline ./roofline_kernels\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_aux));
    free(h_src);

    return 0;
}
