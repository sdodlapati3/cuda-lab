/**
 * cuda_ops.cu - CUDA operations implementation
 */

#include "app/cuda_ops.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace app {
namespace cuda {

static char g_device_info[256] = "";

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// Kernel: scale and add
__global__ void process_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

bool init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    snprintf(g_device_info, sizeof(g_device_info),
             "%s (SM %d.%d, %.1f GB)",
             prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / 1e9);
    
    return true;
}

void cleanup() {
    cudaDeviceReset();
}

const char* get_device_info() {
    return g_device_info;
}

float process(float* data, size_t n, int iterations) {
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    process_kernel<<<numBlocks, blockSize>>>(d_data, n);
    cudaDeviceSynchronize();
    
    // Reset
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        process_kernel<<<numBlocks, blockSize>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Copy back
    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return ms;
}

bool verify(const float* data, size_t n) {
    // After 'iterations' passes: data[i] = ((i * 2 + 1) * 2 + 1) * ...
    // For 1 iteration: data[i] = i * 2 + 1
    for (size_t i = 0; i < n; i++) {
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        if (fabs(data[i] - expected) > 1e-3f) {
            fprintf(stderr, "Mismatch at %zu: got %f, expected %f\n",
                    i, data[i], expected);
            return false;
        }
    }
    return true;
}

}  // namespace cuda
}  // namespace app
