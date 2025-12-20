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

// =============================================================================
// Kernel with many registers (low occupancy)
// =============================================================================
__global__ void kernel_high_registers(float* out, const float* in, int n) {
    // Using many local variables forces high register usage
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

// =============================================================================
// Kernel with register limit (higher occupancy)
// =============================================================================
__launch_bounds__(256, 4)  // Max 256 threads, at least 4 blocks per SM
__global__ void kernel_limited_registers(float* out, const float* in, int n) {
    float a = 0, b = 0, c = 0, d = 0;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a = in[idx]; b = a * 2; c = b * 3; d = c * 4;
        out[idx] = a + b + c + d;
    }
}

// =============================================================================
// TODO: Use occupancy API to analyze kernels
// =============================================================================
void analyze_occupancy(void* kernel, const char* name, int blockSize) {
    int numBlocks;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    // TODO: Get max active blocks per SM
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, 0);
    
    // TODO: Calculate occupancy percentage
    // int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    // int activeWarps = numBlocks * (blockSize / 32);
    // float occupancy = (float)activeWarps / maxWarpsPerSM * 100;
    
    // printf("%s: %d blocks/SM, %.1f%% occupancy\n", name, numBlocks, occupancy);
    printf("%s: (implement occupancy analysis)\n", name);
}

int main() {
    printf("Occupancy Analysis Exercise\n");
    printf("===========================\n\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu bytes\n\n", prop.sharedMemPerMultiprocessor);
    
    // Analyze different kernels
    analyze_occupancy((void*)kernel_high_registers, "High Registers", 256);
    analyze_occupancy((void*)kernel_limited_registers, "Limited Registers", 256);
    
    // TODO: Find optimal block size
    int minGridSize, optimalBlockSize;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
    //                                    kernel_limited_registers, 0, 0);
    // printf("\nOptimal block size for limited_registers: %d\n", optimalBlockSize);
    
    printf("\n✓ Exercise complete!\n");
    return 0;
}
