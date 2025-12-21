/*
 * Day 2: Tiled Convolution
 * 
 * Uses shared memory to cache tile + halo region for efficient convolution.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Tile dimensions (output pixels per block)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Tiled convolution with shared memory
template<int FILTER_SIZE>
__global__ void convolution2D_tiled(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width,
    int height)
{
    constexpr int RADIUS = FILTER_SIZE / 2;
    constexpr int SHARED_WIDTH = TILE_WIDTH + 2 * RADIUS;
    constexpr int SHARED_HEIGHT = TILE_HEIGHT + 2 * RADIUS;
    
    __shared__ float tile[SHARED_HEIGHT][SHARED_WIDTH];
    
    // Output pixel coordinates
    const int outX = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int outY = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    // Load tile + halo into shared memory
    // Each thread loads multiple pixels to cover the halo
    const int loadX = blockIdx.x * TILE_WIDTH - RADIUS + threadIdx.x;
    const int loadY = blockIdx.y * TILE_HEIGHT - RADIUS + threadIdx.y;
    
    // Load primary position
    for (int dy = 0; dy < SHARED_HEIGHT; dy += TILE_HEIGHT) {
        for (int dx = 0; dx < SHARED_WIDTH; dx += TILE_WIDTH) {
            int sx = threadIdx.x + dx;
            int sy = threadIdx.y + dy;
            if (sx < SHARED_WIDTH && sy < SHARED_HEIGHT) {
                int gx = loadX + dx;
                int gy = loadY + dy;
                // Clamp boundary
                gx = max(0, min(width - 1, gx));
                gy = max(0, min(height - 1, gy));
                tile[sy][sx] = input[gy * width + gx];
            }
        }
    }
    
    __syncthreads();
    
    if (outX >= width || outY >= height) return;
    
    // Apply filter using shared memory
    float sum = 0.0f;
    
    #pragma unroll
    for (int fy = 0; fy < FILTER_SIZE; fy++) {
        #pragma unroll
        for (int fx = 0; fx < FILTER_SIZE; fx++) {
            sum += tile[threadIdx.y + fy][threadIdx.x + fx] * filter[fy * FILTER_SIZE + fx];
        }
    }
    
    output[outY * width + outX] = sum;
}

// Naive convolution for comparison
template<int FILTER_SIZE>
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
    
    const int radius = FILTER_SIZE / 2;
    float sum = 0.0f;
    
    #pragma unroll
    for (int fy = 0; fy < FILTER_SIZE; fy++) {
        #pragma unroll
        for (int fx = 0; fx < FILTER_SIZE; fx++) {
            int ix = x + fx - radius;
            int iy = y + fy - radius;
            ix = max(0, min(width - 1, ix));
            iy = max(0, min(height - 1, iy));
            sum += input[iy * width + ix] * filter[fy * FILTER_SIZE + fx];
        }
    }
    
    output[y * width + x] = sum;
}

void createBoxFilter(float* filter, int size) {
    float value = 1.0f / (size * size);
    for (int i = 0; i < size * size; i++) {
        filter[i] = value;
    }
}

template<int FILTER_SIZE>
float benchmarkTiled(const float* d_input, float* d_output, 
                     const float* d_filter, int width, int height, int iterations) {
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, 
              (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    // Warmup
    convolution2D_tiled<FILTER_SIZE><<<grid, block>>>(d_input, d_output, d_filter, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        convolution2D_tiled<FILTER_SIZE><<<grid, block>>>(d_input, d_output, d_filter, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / iterations;
}

template<int FILTER_SIZE>
float benchmarkNaive(const float* d_input, float* d_output, 
                     const float* d_filter, int width, int height, int iterations) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    convolution2D_naive<FILTER_SIZE><<<grid, block>>>(d_input, d_output, d_filter, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        convolution2D_naive<FILTER_SIZE><<<grid, block>>>(d_input, d_output, d_filter, width, height);
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
    printf("=== Day 2: Tiled Convolution ===\n\n");
    
    const int width = 1920;
    const int height = 1080;
    const size_t imageSize = width * height * sizeof(float);
    const int iterations = 100;
    
    printf("Image size: %d x %d\n", width, height);
    printf("Tile size: %d x %d\n\n", TILE_WIDTH, TILE_HEIGHT);
    
    // Host memory
    float* h_input = new float[width * height];
    float* h_output_naive = new float[width * height];
    float* h_output_tiled = new float[width * height];
    float h_filter3x3[9], h_filter5x5[25], h_filter7x7[49];
    
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    createBoxFilter(h_filter3x3, 3);
    createBoxFilter(h_filter5x5, 5);
    createBoxFilter(h_filter7x7, 7);
    
    // Device memory
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
    
    // Verify correctness
    printf("--- Correctness Check ---\n");
    dim3 block_naive(16, 16);
    dim3 grid_naive((width + 15) / 16, (height + 15) / 16);
    dim3 block_tiled(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_tiled((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    convolution2D_naive<3><<<grid_naive, block_naive>>>(d_input, d_output, d_filter3x3, width, height);
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    convolution2D_tiled<3><<<grid_tiled, block_tiled>>>(d_input, d_output, d_filter3x3, width, height);
    CHECK_CUDA(cudaMemcpy(h_output_tiled, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    float maxError = 0.0f;
    for (int i = 0; i < width * height; i++) {
        maxError = fmaxf(maxError, fabsf(h_output_naive[i] - h_output_tiled[i]));
    }
    printf("Max error (tiled vs naive): %e\n\n", maxError);
    
    // Performance comparison
    printf("--- Performance Comparison ---\n");
    printf("%-10s %12s %12s %10s\n", "Filter", "Naive (ms)", "Tiled (ms)", "Speedup");
    printf("%-10s %12s %12s %10s\n", "------", "----------", "----------", "-------");
    
    float naive3 = benchmarkNaive<3>(d_input, d_output, d_filter3x3, width, height, iterations);
    float tiled3 = benchmarkTiled<3>(d_input, d_output, d_filter3x3, width, height, iterations);
    printf("%-10s %12.3f %12.3f %9.2fx\n", "3x3", naive3, tiled3, naive3/tiled3);
    
    float naive5 = benchmarkNaive<5>(d_input, d_output, d_filter5x5, width, height, iterations);
    float tiled5 = benchmarkTiled<5>(d_input, d_output, d_filter5x5, width, height, iterations);
    printf("%-10s %12.3f %12.3f %9.2fx\n", "5x5", naive5, tiled5, naive5/tiled5);
    
    float naive7 = benchmarkNaive<7>(d_input, d_output, d_filter7x7, width, height, iterations);
    float tiled7 = benchmarkTiled<7>(d_input, d_output, d_filter7x7, width, height, iterations);
    printf("%-10s %12.3f %12.3f %9.2fx\n", "7x7", naive7, tiled7, naive7/tiled7);
    
    // Bandwidth analysis
    printf("\n--- Bandwidth Analysis (3x3) ---\n");
    float bandwidth_naive = (2.0f * imageSize / (1024.0f * 1024.0f * 1024.0f)) / (naive3 / 1000.0f);
    float bandwidth_tiled = (2.0f * imageSize / (1024.0f * 1024.0f * 1024.0f)) / (tiled3 / 1000.0f);
    printf("Naive effective bandwidth:  %.1f GB/s\n", bandwidth_naive);
    printf("Tiled effective bandwidth:  %.1f GB/s\n", bandwidth_tiled);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filter3x3));
    CHECK_CUDA(cudaFree(d_filter5x5));
    CHECK_CUDA(cudaFree(d_filter7x7));
    
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_tiled;
    
    printf("\n=== Day 2 Complete ===\n");
    return 0;
}
