/*
 * Day 3: Separable Filters
 * 
 * Demonstrates decomposing 2D convolution into two 1D passes.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 128

// Horizontal 1D convolution (coalesced access)
template<int FILTER_SIZE>
__global__ void convolution1D_horizontal(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width,
    int height)
{
    constexpr int RADIUS = FILTER_SIZE / 2;
    __shared__ float tile[TILE_SIZE + 2 * RADIUS];
    
    const int y = blockIdx.y;
    const int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Load tile with halo
    int loadX = blockIdx.x * TILE_SIZE - RADIUS + threadIdx.x;
    if (threadIdx.x < TILE_SIZE + 2 * RADIUS) {
        int clampedX = max(0, min(width - 1, loadX));
        tile[threadIdx.x] = input[y * width + clampedX];
    }
    
    // Load extra elements if needed
    if (threadIdx.x < 2 * RADIUS) {
        int extraIdx = threadIdx.x + TILE_SIZE;
        int loadExtraX = blockIdx.x * TILE_SIZE - RADIUS + extraIdx;
        int clampedX = max(0, min(width - 1, loadExtraX));
        if (extraIdx < TILE_SIZE + 2 * RADIUS) {
            tile[extraIdx] = input[y * width + clampedX];
        }
    }
    
    __syncthreads();
    
    if (x >= width) return;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < FILTER_SIZE; i++) {
        sum += tile[threadIdx.x + i] * filter[i];
    }
    
    output[y * width + x] = sum;
}

// Vertical 1D convolution (strided access)
template<int FILTER_SIZE>
__global__ void convolution1D_vertical(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width,
    int height)
{
    constexpr int RADIUS = FILTER_SIZE / 2;
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < FILTER_SIZE; i++) {
        int iy = y + i - RADIUS;
        iy = max(0, min(height - 1, iy));
        sum += input[iy * width + x] * filter[i];
    }
    
    output[y * width + x] = sum;
}

// 2D convolution for comparison
template<int FILTER_SIZE>
__global__ void convolution2D(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = FILTER_SIZE / 2;
    float sum = 0.0f;
    
    for (int fy = 0; fy < FILTER_SIZE; fy++) {
        for (int fx = 0; fx < FILTER_SIZE; fx++) {
            int ix = max(0, min(width - 1, x + fx - radius));
            int iy = max(0, min(height - 1, y + fy - radius));
            sum += input[iy * width + ix] * filter[fy * FILTER_SIZE + fx];
        }
    }
    
    output[y * width + x] = sum;
}

// Create 1D Gaussian filter
void createGaussian1D(float* filter, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float x = i - half;
        filter[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += filter[i];
    }
    
    for (int i = 0; i < size; i++) {
        filter[i] /= sum;
    }
}

// Create 2D Gaussian filter (separable)
void createGaussian2D(float* filter, int size, float sigma) {
    float* filter1D = new float[size];
    createGaussian1D(filter1D, size, sigma);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            filter[y * size + x] = filter1D[y] * filter1D[x];
        }
    }
    
    delete[] filter1D;
}

int main() {
    printf("=== Day 3: Separable Filters ===\n\n");
    
    const int width = 1920;
    const int height = 1080;
    const size_t imageSize = width * height * sizeof(float);
    const int iterations = 100;
    
    printf("Image size: %d x %d\n\n", width, height);
    
    // Host memory
    float* h_input = new float[width * height];
    float* h_output_2d = new float[width * height];
    float* h_output_sep = new float[width * height];
    
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Create filters
    const float sigma = 1.5f;
    float h_filter1D_5[5], h_filter1D_7[7], h_filter1D_9[9];
    float h_filter2D_5[25], h_filter2D_7[49], h_filter2D_9[81];
    
    createGaussian1D(h_filter1D_5, 5, sigma);
    createGaussian1D(h_filter1D_7, 7, sigma);
    createGaussian1D(h_filter1D_9, 9, sigma);
    createGaussian2D(h_filter2D_5, 5, sigma);
    createGaussian2D(h_filter2D_7, 7, sigma);
    createGaussian2D(h_filter2D_9, 9, sigma);
    
    // Device memory
    float *d_input, *d_temp, *d_output;
    float *d_filter1D_5, *d_filter1D_7, *d_filter1D_9;
    float *d_filter2D_5, *d_filter2D_7, *d_filter2D_9;
    
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_temp, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA(cudaMalloc(&d_filter1D_5, 5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter1D_7, 7 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter1D_9, 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter2D_5, 25 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter2D_7, 49 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter2D_9, 81 * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter1D_5, h_filter1D_5, 5 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter1D_7, h_filter1D_7, 7 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter1D_9, h_filter1D_9, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter2D_5, h_filter2D_5, 25 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter2D_7, h_filter2D_7, 49 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter2D_9, h_filter2D_9, 81 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Verify correctness (5x5 Gaussian)
    printf("--- Correctness Check (5x5 Gaussian) ---\n");
    
    dim3 block2d(16, 16);
    dim3 grid2d((width + 15) / 16, (height + 15) / 16);
    dim3 gridH((width + TILE_SIZE - 1) / TILE_SIZE, height);
    dim3 blockH(TILE_SIZE);
    dim3 gridV((width + 15) / 16, (height + 15) / 16);
    dim3 blockV(16, 16);
    
    convolution2D<5><<<grid2d, block2d>>>(d_input, d_output, d_filter2D_5, width, height);
    CHECK_CUDA(cudaMemcpy(h_output_2d, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    convolution1D_horizontal<5><<<gridH, blockH>>>(d_input, d_temp, d_filter1D_5, width, height);
    convolution1D_vertical<5><<<gridV, blockV>>>(d_temp, d_output, d_filter1D_5, width, height);
    CHECK_CUDA(cudaMemcpy(h_output_sep, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    float maxError = 0.0f;
    for (int i = 0; i < width * height; i++) {
        maxError = fmaxf(maxError, fabsf(h_output_2d[i] - h_output_sep[i]));
    }
    printf("Max error (separable vs 2D): %e\n\n", maxError);
    
    // Benchmark function
    auto benchmark = [&](int filterSize, float* d_filter1D, float* d_filter2D) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        // 2D convolution
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            if (filterSize == 5) {
                convolution2D<5><<<grid2d, block2d>>>(d_input, d_output, d_filter2D, width, height);
            } else if (filterSize == 7) {
                convolution2D<7><<<grid2d, block2d>>>(d_input, d_output, d_filter2D, width, height);
            } else {
                convolution2D<9><<<grid2d, block2d>>>(d_input, d_output, d_filter2D, width, height);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms2d;
        CHECK_CUDA(cudaEventElapsedTime(&ms2d, start, stop));
        
        // Separable
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            if (filterSize == 5) {
                convolution1D_horizontal<5><<<gridH, blockH>>>(d_input, d_temp, d_filter1D, width, height);
                convolution1D_vertical<5><<<gridV, blockV>>>(d_temp, d_output, d_filter1D, width, height);
            } else if (filterSize == 7) {
                convolution1D_horizontal<7><<<gridH, blockH>>>(d_input, d_temp, d_filter1D, width, height);
                convolution1D_vertical<7><<<gridV, blockV>>>(d_temp, d_output, d_filter1D, width, height);
            } else {
                convolution1D_horizontal<9><<<gridH, blockH>>>(d_input, d_temp, d_filter1D, width, height);
                convolution1D_vertical<9><<<gridV, blockV>>>(d_temp, d_output, d_filter1D, width, height);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float msSep;
        CHECK_CUDA(cudaEventElapsedTime(&msSep, start, stop));
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        return std::make_pair(ms2d / iterations, msSep / iterations);
    };
    
    // Performance comparison
    printf("--- Performance: 2D vs Separable ---\n");
    printf("%-10s %12s %12s %10s %15s\n", "Filter", "2D (ms)", "Sep (ms)", "Speedup", "Ops Reduction");
    printf("%-10s %12s %12s %10s %15s\n", "------", "-------", "--------", "-------", "-------------");
    
    auto [ms2d_5, msSep_5] = benchmark(5, d_filter1D_5, d_filter2D_5);
    printf("%-10s %12.3f %12.3f %9.2fx %14.1fx\n", "5x5", ms2d_5, msSep_5, ms2d_5/msSep_5, 25.0f/10.0f);
    
    auto [ms2d_7, msSep_7] = benchmark(7, d_filter1D_7, d_filter2D_7);
    printf("%-10s %12.3f %12.3f %9.2fx %14.1fx\n", "7x7", ms2d_7, msSep_7, ms2d_7/msSep_7, 49.0f/14.0f);
    
    auto [ms2d_9, msSep_9] = benchmark(9, d_filter1D_9, d_filter2D_9);
    printf("%-10s %12.3f %12.3f %9.2fx %14.1fx\n", "9x9", ms2d_9, msSep_9, ms2d_9/msSep_9, 81.0f/18.0f);
    
    printf("\n--- Key Insight ---\n");
    printf("Separable filters trade 2 kernel launches for N²→2N operation reduction.\n");
    printf("Larger filters benefit more from separability.\n");
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filter1D_5));
    CHECK_CUDA(cudaFree(d_filter1D_7));
    CHECK_CUDA(cudaFree(d_filter1D_9));
    CHECK_CUDA(cudaFree(d_filter2D_5));
    CHECK_CUDA(cudaFree(d_filter2D_7));
    CHECK_CUDA(cudaFree(d_filter2D_9));
    
    delete[] h_input;
    delete[] h_output_2d;
    delete[] h_output_sep;
    
    printf("\n=== Day 3 Complete ===\n");
    return 0;
}
