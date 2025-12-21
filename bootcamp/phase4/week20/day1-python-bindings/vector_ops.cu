/**
 * CUDA Vector Operations Library
 * 
 * Simple operations that can be wrapped for Python access.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// External C interface for easy binding
extern "C" {

// Vector addition kernel
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector multiplication kernel
__global__ void vectorMulKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Scalar multiply kernel
__global__ void scalarMulKernel(float* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

// Reduction sum kernel (simple version)
__global__ void sumReduceKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Host-callable wrapper functions
void cuda_vector_add(const float* h_a, const float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAddKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void cuda_vector_mul(const float* h_a, const float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorMulKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

float cuda_sum(const float* h_input, int n) {
    float *d_input, *d_output;
    float result = 0.0f;
    
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sumReduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);
    
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// Get CUDA device info
int cuda_get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

const char* cuda_get_device_name(int device) {
    static cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.name;
}

}  // extern "C"
