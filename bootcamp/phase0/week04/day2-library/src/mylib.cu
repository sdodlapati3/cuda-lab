/**
 * mylib.cu - Library implementation
 */

#include "mylib/mylib.h"
#include "mylib/kernels.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace mylib {

static char g_error_msg[256] = "";
static bool g_initialized = false;

static void set_error(const char* msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
}

static void set_cuda_error(cudaError_t err) {
    set_error(cudaGetErrorString(err));
}

bool initialize() {
    cudaError_t err = cudaFree(0);  // Initialize CUDA context
    if (err != cudaSuccess) {
        set_cuda_error(err);
        return false;
    }
    g_initialized = true;
    g_error_msg[0] = '\0';
    return true;
}

void cleanup() {
    cudaDeviceReset();
    g_initialized = false;
}

const char* get_last_error() {
    return g_error_msg;
}

void vector_add(const float* a, const float* b, float* c, size_t n, bool on_device) {
    const float *d_a, *d_b;
    float* d_c;
    
    if (on_device) {
        d_a = a; d_b = b; d_c = c;
    } else {
        cudaMalloc((void**)&d_a, n * sizeof(float));
        cudaMalloc((void**)&d_b, n * sizeof(float));
        cudaMalloc((void**)&d_c, n * sizeof(float));
        cudaMemcpy((void*)d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernels::add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    if (!on_device) {
        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_a);
        cudaFree((void*)d_b);
        cudaFree((void*)d_c);
    }
}

void vector_scale(const float* a, float* b, float alpha, size_t n, bool on_device) {
    const float* d_a;
    float* d_b;
    
    if (on_device) {
        d_a = a; d_b = b;
    } else {
        cudaMalloc((void**)&d_a, n * sizeof(float));
        cudaMalloc((void**)&d_b, n * sizeof(float));
        cudaMemcpy((void*)d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernels::scale_kernel<<<numBlocks, blockSize>>>(d_a, d_b, alpha, n);
    cudaDeviceSynchronize();
    
    if (!on_device) {
        cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree((void*)d_a);
        cudaFree((void*)d_b);
    }
}

float dot_product(const float* a, const float* b, size_t n, bool on_device) {
    const float *d_a, *d_b;
    float* d_partial;
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    if (on_device) {
        d_a = a; d_b = b;
    } else {
        cudaMalloc((void**)&d_a, n * sizeof(float));
        cudaMalloc((void**)&d_b, n * sizeof(float));
        cudaMemcpy((void*)d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    
    kernels::dot_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_a, d_b, d_partial, n);
    cudaDeviceSynchronize();
    
    // Reduce on CPU
    float* h_partial = new float[numBlocks];
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_partial);
    
    if (!on_device) {
        cudaFree((void*)d_a);
        cudaFree((void*)d_b);
    }
    
    return result;
}

}  // namespace mylib
