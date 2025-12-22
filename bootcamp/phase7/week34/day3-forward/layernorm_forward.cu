/**
 * Week 34, Day 3: LayerNorm Forward Optimization
 * 
 * Vectorized loads, block-level reduction, fused operations.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized forward kernel
__global__ void layerNormForwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_out,    // Save for backward
    float* __restrict__ invstd_out,  // Save for backward
    int hidden, float eps
) {
    __shared__ float s_mean, s_invstd;
    
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden;
    float* y = output + batch_idx * hidden;
    
    // Pass 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += x[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) s_mean = sum / hidden;
    __syncthreads();
    float mean = s_mean;
    
    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSum(var_sum);
    if (threadIdx.x == 0) s_invstd = rsqrtf(var_sum / hidden + eps);
    __syncthreads();
    float invstd = s_invstd;
    
    // Save for backward pass
    if (threadIdx.x == 0 && mean_out && invstd_out) {
        mean_out[batch_idx] = mean;
        invstd_out[batch_idx] = invstd;
    }
    
    // Pass 3: Normalize with gamma/beta
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        y[i] = gamma[i] * (x[i] - mean) * invstd + beta[i];
    }
}

// Vectorized version using float4
__global__ void layerNormForwardVec4Kernel(
    const float4* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float4* __restrict__ output,
    int hidden, float eps
) {
    __shared__ float s_mean, s_invstd;
    
    int batch_idx = blockIdx.x;
    int hidden4 = hidden / 4;
    const float4* x = input + batch_idx * hidden4;
    float4* y = output + batch_idx * hidden4;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden4; i += blockDim.x) {
        float4 v = x[i];
        sum += v.x + v.y + v.z + v.w;
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) s_mean = sum / hidden;
    __syncthreads();
    float mean = s_mean;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden4; i += blockDim.x) {
        float4 v = x[i];
        float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
        var_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    var_sum = blockReduceSum(var_sum);
    if (threadIdx.x == 0) s_invstd = rsqrtf(var_sum / hidden + eps);
    __syncthreads();
    float invstd = s_invstd;
    
    // Normalize
    for (int i = threadIdx.x; i < hidden4; i += blockDim.x) {
        float4 v = x[i];
        int base = i * 4;
        float4 out;
        out.x = gamma[base+0] * (v.x - mean) * invstd + beta[base+0];
        out.y = gamma[base+1] * (v.y - mean) * invstd + beta[base+1];
        out.z = gamma[base+2] * (v.z - mean) * invstd + beta[base+2];
        out.w = gamma[base+3] * (v.w - mean) * invstd + beta[base+3];
        y[i] = out;
    }
}

int main() {
    printf("Week 34 Day 3: LayerNorm Forward Optimization\n\n");
    
    const int batch = 128, hidden = 1024;
    
    float *h_input = new float[batch * hidden];
    float *h_gamma = new float[hidden];
    float *h_beta = new float[hidden];
    float *h_output = new float[batch * hidden];
    
    for (int i = 0; i < batch * hidden; i++) h_input[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < hidden; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }
    
    float *d_input, *d_gamma, *d_beta, *d_output, *d_mean, *d_invstd;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_beta, hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    cudaMalloc(&d_mean, batch * sizeof(float));
    cudaMalloc(&d_invstd, batch * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Scalar version
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        layerNormForwardKernel<<<batch, BLOCK_SIZE>>>(
            d_input, d_gamma, d_beta, d_output, d_mean, d_invstd, hidden, 1e-5f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_scalar;
    cudaEventElapsedTime(&ms_scalar, start, stop);
    
    // Vec4 version
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        layerNormForwardVec4Kernel<<<batch, BLOCK_SIZE>>>(
            (float4*)d_input, d_gamma, d_beta, (float4*)d_output, hidden, 1e-5f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_vec4;
    cudaEventElapsedTime(&ms_vec4, start, stop);
    
    printf("Performance (%d×%d, 1000 iters):\n", batch, hidden);
    printf("  Scalar: %.2f ms (%.2f us/call)\n", ms_scalar, ms_scalar);
    printf("  Vec4:   %.2f ms (%.2f us/call)\n", ms_vec4, ms_vec4);
    printf("  Speedup: %.2fx\n", ms_scalar / ms_vec4);
    
    delete[] h_input; delete[] h_gamma; delete[] h_beta; delete[] h_output;
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_beta); 
    cudaFree(d_output); cudaFree(d_mean); cudaFree(d_invstd);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
