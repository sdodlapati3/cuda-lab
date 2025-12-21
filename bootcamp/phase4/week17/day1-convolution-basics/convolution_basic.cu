/*
 * Day 1: Convolution Basics
 * 
 * Demonstrates naive 2D convolution with different border handling modes.
 * This baseline shows the memory inefficiency we'll fix with tiling.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Border handling modes
enum BorderMode {
    BORDER_ZERO,    // Out-of-bounds = 0
    BORDER_CLAMP,   // Repeat edge pixels
    BORDER_MIRROR   // Reflect at boundary
};

// Device function for border-aware pixel access
__device__ float getPixel(const float* image, int x, int y, 
                          int width, int height, BorderMode mode) {
    switch (mode) {
        case BORDER_ZERO:
            if (x < 0 || x >= width || y < 0 || y >= height)
                return 0.0f;
            break;
        case BORDER_CLAMP:
            x = max(0, min(width - 1, x));
            y = max(0, min(height - 1, y));
            break;
        case BORDER_MIRROR:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = max(0, min(width - 1, x));
            y = max(0, min(height - 1, y));
            break;
    }
    return image[y * width + x];
}

// Naive 2D convolution kernel
template<int FILTER_SIZE, BorderMode MODE>
__global__ void convolution2D_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int half = FILTER_SIZE / 2;
    float sum = 0.0f;
    
    // Apply filter
    #pragma unroll
    for (int fy = 0; fy < FILTER_SIZE; fy++) {
        #pragma unroll
        for (int fx = 0; fx < FILTER_SIZE; fx++) {
            int ix = x + fx - half;
            int iy = y + fy - half;
            float pixel = getPixel(input, ix, iy, width, height, MODE);
            sum += pixel * filter[fy * FILTER_SIZE + fx];
        }
    }
    
    output[y * width + x] = sum;
}

// Create box blur filter (uniform averaging)
void createBoxFilter(float* filter, int size) {
    float value = 1.0f / (size * size);
    for (int i = 0; i < size * size; i++) {
        filter[i] = value;
    }
}

// Create Gaussian filter
void createGaussianFilter(float* filter, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - half;
            float dy = y - half;
            float value = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            filter[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

// CPU reference implementation
void convolution2D_cpu(const float* input, float* output, const float* filter,
                       int width, int height, int filterSize) {
    int half = filterSize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int fy = 0; fy < filterSize; fy++) {
                for (int fx = 0; fx < filterSize; fx++) {
                    int ix = x + fx - half;
                    int iy = y + fy - half;
                    // Zero padding
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += input[iy * width + ix] * filter[fy * filterSize + fx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// Benchmark convolution
template<int FILTER_SIZE, BorderMode MODE>
float benchmarkConvolution(const float* d_input, float* d_output, 
                           const float* d_filter, int width, int height,
                           int iterations) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Warmup
    convolution2D_naive<FILTER_SIZE, MODE><<<grid, block>>>(
        d_input, d_output, d_filter, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        convolution2D_naive<FILTER_SIZE, MODE><<<grid, block>>>(
            d_input, d_output, d_filter, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / iterations;
}

int main() {
    printf("=== Day 1: Convolution Basics ===\n\n");
    
    // Test image size (Full HD)
    const int width = 1920;
    const int height = 1080;
    const size_t imageSize = width * height * sizeof(float);
    const int iterations = 100;
    
    printf("Image size: %d x %d\n", width, height);
    printf("Image bytes: %.2f MB\n\n", imageSize / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float* h_input = new float[width * height];
    float* h_output = new float[width * height];
    float* h_output_cpu = new float[width * height];
    float h_filter3x3[9];
    float h_filter5x5[25];
    float h_filter7x7[49];
    
    // Initialize input with test pattern
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Create filters
    createBoxFilter(h_filter3x3, 3);
    createGaussianFilter(h_filter5x5, 5, 1.0f);
    createGaussianFilter(h_filter7x7, 7, 1.5f);
    
    // Allocate device memory
    float *d_input, *d_output;
    float *d_filter3x3, *d_filter5x5, *d_filter7x7;
    
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA(cudaMalloc(&d_filter3x3, 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter5x5, 25 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter7x7, 49 * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter3x3, h_filter3x3, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter5x5, h_filter5x5, 25 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter7x7, h_filter7x7, 49 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Verify correctness with CPU
    printf("--- Correctness Check (3x3 Box Blur) ---\n");
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    convolution2D_naive<3, BORDER_ZERO><<<grid, block>>>(
        d_input, d_output, d_filter3x3, width, height);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    convolution2D_cpu(h_input, h_output_cpu, h_filter3x3, width, height, 3);
    
    float maxError = 0.0f;
    for (int i = 0; i < width * height; i++) {
        maxError = fmaxf(maxError, fabsf(h_output[i] - h_output_cpu[i]));
    }
    printf("Max error vs CPU: %e\n\n", maxError);
    
    // Benchmark different filter sizes
    printf("--- Performance (Border: Zero Padding) ---\n");
    printf("%-15s %10s %12s %12s\n", "Filter Size", "Time (ms)", "Throughput", "Efficiency");
    printf("%-15s %10s %12s %12s\n", "-----------", "---------", "----------", "----------");
    
    // 3x3
    float ms3 = benchmarkConvolution<3, BORDER_ZERO>(
        d_input, d_output, d_filter3x3, width, height, iterations);
    float throughput3 = (2.0f * imageSize / (1024.0f * 1024.0f * 1024.0f)) / (ms3 / 1000.0f);
    printf("%-15s %10.3f %10.1f GB/s\n", "3x3 Box", ms3, throughput3);
    
    // 5x5
    float ms5 = benchmarkConvolution<5, BORDER_ZERO>(
        d_input, d_output, d_filter5x5, width, height, iterations);
    float throughput5 = (2.0f * imageSize / (1024.0f * 1024.0f * 1024.0f)) / (ms5 / 1000.0f);
    printf("%-15s %10.3f %10.1f GB/s\n", "5x5 Gaussian", ms5, throughput5);
    
    // 7x7
    float ms7 = benchmarkConvolution<7, BORDER_ZERO>(
        d_input, d_output, d_filter7x7, width, height, iterations);
    float throughput7 = (2.0f * imageSize / (1024.0f * 1024.0f * 1024.0f)) / (ms7 / 1000.0f);
    printf("%-15s %10.3f %10.1f GB/s\n", "7x7 Gaussian", ms7, throughput7);
    
    // Benchmark different border modes
    printf("\n--- Border Mode Comparison (3x3) ---\n");
    printf("%-15s %10s\n", "Border Mode", "Time (ms)");
    printf("%-15s %10s\n", "-----------", "---------");
    
    float ms_zero = benchmarkConvolution<3, BORDER_ZERO>(
        d_input, d_output, d_filter3x3, width, height, iterations);
    printf("%-15s %10.3f\n", "Zero", ms_zero);
    
    float ms_clamp = benchmarkConvolution<3, BORDER_CLAMP>(
        d_input, d_output, d_filter3x3, width, height, iterations);
    printf("%-15s %10.3f\n", "Clamp", ms_clamp);
    
    float ms_mirror = benchmarkConvolution<3, BORDER_MIRROR>(
        d_input, d_output, d_filter3x3, width, height, iterations);
    printf("%-15s %10.3f\n", "Mirror", ms_mirror);
    
    // Analysis
    printf("\n--- Analysis ---\n");
    printf("This naive implementation:\n");
    printf("- Each thread reads %d pixels for 3x3 filter\n", 9);
    printf("- Neighboring threads have 6/9 = 67%% overlap\n");
    printf("- Memory reads = %.1f MB per kernel\n", 
           (float)(width * height * 9 * sizeof(float)) / (1024.0f * 1024.0f));
    printf("- Actual data = %.1f MB\n", imageSize / (1024.0f * 1024.0f));
    printf("- Read amplification: ~%.1fx\n", 9.0f);
    printf("\nTomorrow: Tiled convolution reduces this to ~1x!\n");
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filter3x3));
    CHECK_CUDA(cudaFree(d_filter5x5));
    CHECK_CUDA(cudaFree(d_filter7x7));
    
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_cpu;
    
    printf("\n=== Day 1 Complete ===\n");
    return 0;
}
