/**
 * implementations.cu - Various reduction implementations
 */

#include "benchmark/implementations.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace impl {

static float* d_data = nullptr;
static float* d_result = nullptr;
static size_t g_n = 0;
static float h_result = 0.0f;

// V0: Global atomics (naive)
__global__ void reduce_v0(const float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, data[idx]);
    }
}

// V1: Shared memory reduction
__global__ void reduce_v1(const float* data, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// V2: Warp shuffle reduction
__global__ void reduce_v2(const float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? data[idx] : 0.0f;
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // First lane of each warp adds to result
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(result, val);
    }
}

// V3: Vectorized load (float4)
__global__ void reduce_v3(const float4* data, float* result, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (idx < n4) {
        float4 v = data[idx];
        val = v.x + v.y + v.z + v.w;
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(result, val);
    }
}

void setup(size_t n) {
    g_n = n;
    
    // Allocate
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    // Initialize data on host
    std::vector<float> h_data(n);
    for (size_t i = 0; i < n; i++) {
        h_data[i] = 1.0f;  // Sum should be n
    }
    
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Setup complete. Expected result: %.0f\n", (float)n);
}

void cleanup() {
    cudaFree(d_data);
    cudaFree(d_result);
    d_data = nullptr;
    d_result = nullptr;
}

float reduce_v0_global() {
    cudaMemset(d_result, 0, sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (g_n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    reduce_v0<<<numBlocks, blockSize>>>(d_data, d_result, g_n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

float reduce_v1_shared() {
    cudaMemset(d_result, 0, sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (g_n + blockSize - 1) / blockSize;
    size_t sharedMem = blockSize * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    reduce_v1<<<numBlocks, blockSize, sharedMem>>>(d_data, d_result, g_n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

float reduce_v2_warp() {
    cudaMemset(d_result, 0, sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (g_n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    reduce_v2<<<numBlocks, blockSize>>>(d_data, d_result, g_n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

float reduce_v3_vector() {
    cudaMemset(d_result, 0, sizeof(float));
    
    int n4 = g_n / 4;
    int blockSize = 256;
    int numBlocks = (n4 + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    reduce_v3<<<numBlocks, blockSize>>>((float4*)d_data, d_result, n4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

float get_result() { return h_result; }
size_t get_data_size() { return g_n; }

}  // namespace impl
