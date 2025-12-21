/**
 * cuda_test.cu - CUDA test implementations
 */

#include "test/cuda_test.cuh"
#include <cuda_runtime.h>

namespace test {
namespace cuda {

// Vector add kernel
__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector scale kernel
__global__ void scale_kernel(float* out, const float* in, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scalar;
    }
}

// Reduction kernel
__global__ void reduce_kernel(float* result, const float* data, int n) {
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

void init() {
    cudaSetDevice(0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n\n", prop.name);
}

void cleanup() {
    cudaDeviceReset();
}

void vector_add(float* c, const float* a, const float* b, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add_kernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b, n);
    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void vector_scale(float* out, const float* in, float scalar, int n) {
    float *d_in, *d_out;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    scale_kernel<<<numBlocks, blockSize>>>(d_out, d_in, scalar, n);
    
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
}

float reduce_sum(const float* data, int n) {
    float *d_data, *d_result;
    float result = 0.0f;
    
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    reduce_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_result, d_data, n);
    
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return result;
}

}  // namespace cuda
}  // namespace test
