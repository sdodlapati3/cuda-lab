/**
 * myproject.cu - Library implementation
 */

#include "myproject/myproject.h"
#include "myproject/cuda_utils.cuh"
#include <cuda_runtime.h>

namespace myproject {

static char g_device_info[256] = "";
static bool g_initialized = false;

// Kernels
__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void scale_kernel(float* out, const float* in, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scalar;
    }
}

__global__ void reduce_kernel(float* result, const float* data, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void matmul_kernel(float* C, const float* A, const float* B,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool initialize(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return false;
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return false;
    
    snprintf(g_device_info, sizeof(g_device_info),
             "%s (SM %d.%d, %.1f GB, %d SMs)",
             prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / 1e9,
             prop.multiProcessorCount);
    
    g_initialized = true;
    return true;
}

void cleanup() {
    if (g_initialized) {
        cudaDeviceReset();
        g_initialized = false;
    }
}

const char* get_device_info() {
    return g_device_info;
}

void vector_add(float* c, const float* a, const float* b, size_t n) {
    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
    
    d_a.copyFrom(a);
    d_b.copyFrom(b);
    
    int blockSize = 256;
    int numBlocks = divUp(n, blockSize);
    add_kernel<<<numBlocks, blockSize>>>(d_c.get(), d_a.get(), d_b.get(), n);
    
    d_c.copyTo(c);
}

void vector_scale(float* out, const float* in, float scalar, size_t n) {
    DeviceBuffer<float> d_in(n), d_out(n);
    
    d_in.copyFrom(in);
    
    int blockSize = 256;
    int numBlocks = divUp(n, blockSize);
    scale_kernel<<<numBlocks, blockSize>>>(d_out.get(), d_in.get(), scalar, n);
    
    d_out.copyTo(out);
}

float reduce_sum(const float* data, size_t n) {
    DeviceBuffer<float> d_data(n);
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    
    d_data.copyFrom(data);
    
    int blockSize = 256;
    int numBlocks = divUp(n, blockSize);
    reduce_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_result, d_data.get(), n);
    
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return result;
}

void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    DeviceBuffer<float> d_A(M * K), d_B(K * N), d_C(M * N);
    
    d_A.copyFrom(A);
    d_B.copyFrom(B);
    
    dim3 blockDim(16, 16);
    dim3 gridDim(divUp(N, blockDim.x), divUp(M, blockDim.y));
    matmul_kernel<<<gridDim, blockDim>>>(d_C.get(), d_A.get(), d_B.get(), M, N, K);
    
    d_C.copyTo(C);
}

}  // namespace myproject
