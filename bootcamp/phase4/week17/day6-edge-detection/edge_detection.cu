/*
 * Day 6: Edge Detection
 * 
 * Sobel edge detection with gradient magnitude computation.
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

#define TILE_SIZE 16

// Naive Sobel (two separate convolutions)
__global__ void sobel_naive(
    const float* __restrict__ input,
    float* __restrict__ gradX,
    float* __restrict__ gradY,
    float* __restrict__ magnitude,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel kernels
    const float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    float gx = 0.0f, gy = 0.0f;
    
    for (int fy = -1; fy <= 1; fy++) {
        for (int fx = -1; fx <= 1; fx++) {
            int ix = max(0, min(width - 1, x + fx));
            int iy = max(0, min(height - 1, y + fy));
            float pixel = input[iy * width + ix];
            
            int kidx = (fy + 1) * 3 + (fx + 1);
            gx += pixel * sobelX[kidx];
            gy += pixel * sobelY[kidx];
        }
    }
    
    int idx = y * width + x;
    gradX[idx] = gx;
    gradY[idx] = gy;
    magnitude[idx] = sqrtf(gx * gx + gy * gy);
}

// Tiled Sobel with shared memory
__global__ void sobel_tiled(
    const float* __restrict__ input,
    float* __restrict__ magnitude,
    int width, int height)
{
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    // Load tile with halo
    int loadX = blockIdx.x * TILE_SIZE - 1 + tx;
    int loadY = blockIdx.y * TILE_SIZE - 1 + ty;
    
    // Main tile
    if (tx < TILE_SIZE + 2 && ty < TILE_SIZE + 2) {
        int ix = max(0, min(width - 1, loadX));
        int iy = max(0, min(height - 1, loadY));
        tile[ty][tx] = input[iy * width + ix];
    }
    
    // Extra columns
    if (tx < 2 && ty < TILE_SIZE + 2) {
        int extraX = loadX + TILE_SIZE;
        int ix = max(0, min(width - 1, extraX));
        int iy = max(0, min(height - 1, loadY));
        if (tx + TILE_SIZE < TILE_SIZE + 2) {
            tile[ty][tx + TILE_SIZE] = input[iy * width + ix];
        }
    }
    
    // Extra rows
    if (ty < 2 && tx < TILE_SIZE + 2) {
        int extraY = loadY + TILE_SIZE;
        int ix = max(0, min(width - 1, loadX));
        int iy = max(0, min(height - 1, extraY));
        if (ty + TILE_SIZE < TILE_SIZE + 2) {
            tile[ty + TILE_SIZE][tx] = input[iy * width + ix];
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height || tx >= TILE_SIZE || ty >= TILE_SIZE) return;
    
    // Apply Sobel using shared memory
    float gx = -tile[ty][tx] + tile[ty][tx + 2]
              -2.0f * tile[ty + 1][tx] + 2.0f * tile[ty + 1][tx + 2]
              -tile[ty + 2][tx] + tile[ty + 2][tx + 2];
    
    float gy = -tile[ty][tx] - 2.0f * tile[ty][tx + 1] - tile[ty][tx + 2]
              +tile[ty + 2][tx] + 2.0f * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];
    
    magnitude[y * width + x] = sqrtf(gx * gx + gy * gy);
}

// Fused Sobel with threshold (binary edge map)
__global__ void sobel_threshold(
    const float* __restrict__ input,
    unsigned char* __restrict__ edges,
    float threshold,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float gx = 0.0f, gy = 0.0f;
    
    // Unrolled Sobel
    for (int fy = -1; fy <= 1; fy++) {
        int iy = max(0, min(height - 1, y + fy));
        
        float p0 = input[iy * width + max(0, x - 1)];
        float p1 = input[iy * width + x];
        float p2 = input[iy * width + min(width - 1, x + 1)];
        
        float wx = (fy == 0) ? 2.0f : 1.0f;
        float wy = (fy == 0) ? 0.0f : (fy == -1 ? -1.0f : 1.0f);
        
        gx += wx * (p2 - p0);
        gy += wy * (p0 + 2.0f * p1 + p2);
    }
    
    float mag = sqrtf(gx * gx + gy * gy);
    edges[y * width + x] = (mag > threshold) ? 255 : 0;
}

int main() {
    printf("=== Day 6: Edge Detection ===\n\n");
    
    const int width = 1920;
    const int height = 1080;
    const size_t imageSize = width * height * sizeof(float);
    const int iterations = 100;
    
    printf("Image size: %d x %d\n\n", width, height);
    
    // Host memory
    float* h_input = new float[width * height];
    float* h_magnitude = new float[width * height];
    
    // Create test pattern with edges
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Checkerboard pattern for strong edges
            int cx = x / 64;
            int cy = y / 64;
            h_input[y * width + x] = ((cx + cy) % 2) ? 1.0f : 0.0f;
        }
    }
    
    // Device memory
    float *d_input, *d_gradX, *d_gradY, *d_magnitude;
    unsigned char *d_edges;
    
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_gradX, imageSize));
    CHECK_CUDA(cudaMalloc(&d_gradY, imageSize));
    CHECK_CUDA(cudaMalloc(&d_magnitude, imageSize));
    CHECK_CUDA(cudaMalloc(&d_edges, width * height));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    dim3 blockTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridTiled((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    printf("--- Performance Comparison ---\n");
    printf("%-25s %12s %12s\n", "Method", "Time (ms)", "Mpix/s");
    printf("%-25s %12s %12s\n", "------", "---------", "------");
    
    // Naive Sobel
    sobel_naive<<<grid, block>>>(d_input, d_gradX, d_gradY, d_magnitude, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sobel_naive<<<grid, block>>>(d_input, d_gradX, d_gradY, d_magnitude, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msNaive;
    CHECK_CUDA(cudaEventElapsedTime(&msNaive, start, stop));
    msNaive /= iterations;
    float mpixNaive = (width * height / 1e6f) / (msNaive / 1000.0f);
    printf("%-25s %12.3f %12.1f\n", "Naive (Gx,Gy,Mag)", msNaive, mpixNaive);
    
    // Tiled Sobel
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sobel_tiled<<<gridTiled, blockTiled>>>(d_input, d_magnitude, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msTiled;
    CHECK_CUDA(cudaEventElapsedTime(&msTiled, start, stop));
    msTiled /= iterations;
    float mpixTiled = (width * height / 1e6f) / (msTiled / 1000.0f);
    printf("%-25s %12.3f %12.1f\n", "Tiled (Mag only)", msTiled, mpixTiled);
    
    // With threshold
    float threshold = 0.5f;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sobel_threshold<<<grid, block>>>(d_input, d_edges, threshold, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msThreshold;
    CHECK_CUDA(cudaEventElapsedTime(&msThreshold, start, stop));
    msThreshold /= iterations;
    float mpixThreshold = (width * height / 1e6f) / (msThreshold / 1000.0f);
    printf("%-25s %12.3f %12.1f\n", "Fused + Threshold", msThreshold, mpixThreshold);
    
    // Verify output
    CHECK_CUDA(cudaMemcpy(h_magnitude, d_magnitude, imageSize, cudaMemcpyDeviceToHost));
    
    float maxMag = 0.0f;
    int edgeCount = 0;
    for (int i = 0; i < width * height; i++) {
        maxMag = fmaxf(maxMag, h_magnitude[i]);
        if (h_magnitude[i] > 0.5f) edgeCount++;
    }
    
    printf("\n--- Results ---\n");
    printf("Max gradient magnitude: %.2f\n", maxMag);
    printf("Edge pixels (threshold 0.5): %d (%.2f%%)\n", 
           edgeCount, 100.0f * edgeCount / (width * height));
    
    printf("\n--- Week 17 Summary ---\n");
    printf("Completed image processing fundamentals:\n");
    printf("  - 2D convolution (naive, tiled, separable)\n");
    printf("  - Histogram computation with atomics\n");
    printf("  - Image resizing with interpolation\n");
    printf("  - Edge detection with Sobel\n");
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_gradX));
    CHECK_CUDA(cudaFree(d_gradY));
    CHECK_CUDA(cudaFree(d_magnitude));
    CHECK_CUDA(cudaFree(d_edges));
    
    delete[] h_input;
    delete[] h_magnitude;
    
    printf("\n=== Day 6 Complete ===\n");
    return 0;
}
