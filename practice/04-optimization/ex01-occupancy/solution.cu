#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void kernel_high_registers(float* out, const float* in, int n) {
    float a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0;
    float i = 0, j = 0, k = 0, l = 0, m = 0, n2 = 0, o = 0, p = 0;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a = in[idx]; b = a * 2; c = b * 3; d = c * 4;
        e = d * 5; f = e * 6; g = f * 7; h = g * 8;
        i = h * 9; j = i * 10; k = j * 11; l = k * 12;
        m = l * 13; n2 = m * 14; o = n2 * 15; p = o * 16;
        out[idx] = a + b + c + d + e + f + g + h + i + j + k + l + m + n2 + o + p;
    }
}

__launch_bounds__(256, 4)
__global__ void kernel_limited_registers(float* out, const float* in, int n) {
    float a = 0, b = 0, c = 0, d = 0;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a = in[idx]; b = a * 2; c = b * 3; d = c * 4;
        out[idx] = a + b + c + d;
    }
}

// SOLUTION: Occupancy analysis
void analyze_occupancy(void* kernel, const char* name, int blockSize) {
    int numBlocks;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, 0));
    
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    int activeWarps = numBlocks * (blockSize / 32);
    float occupancy = (float)activeWarps / maxWarpsPerSM * 100;
    
    printf("%s (block=%d): %d blocks/SM, %d active warps, %.1f%% occupancy\n", 
           name, blockSize, numBlocks, activeWarps, occupancy);
}

int main() {
    printf("Occupancy Analysis - SOLUTION\n");
    printf("==============================\n\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu bytes\n\n", prop.sharedMemPerMultiprocessor);
    
    printf("Kernel Analysis:\n");
    printf("----------------\n");
    analyze_occupancy((void*)kernel_high_registers, "High Registers", 256);
    analyze_occupancy((void*)kernel_limited_registers, "Limited Registers", 256);
    
    // Try different block sizes
    printf("\nBlock Size Sweep (limited_registers):\n");
    for (int bs = 64; bs <= 1024; bs *= 2) {
        analyze_occupancy((void*)kernel_limited_registers, "  ", bs);
    }
    
    // Find optimal block size
    int minGridSize, optimalBlockSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                                   kernel_limited_registers, 0, 0));
    printf("\nOptimal block size: %d (min grid: %d)\n", optimalBlockSize, minGridSize);
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
