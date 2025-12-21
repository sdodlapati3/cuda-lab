/**
 * Week 30, Day 4: Fused Operations
 * Fused LayerNorm + GELU example.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Separate operations (baseline)
__global__ void layerNormKernel(const float* input, float* output,
                                 const float* gamma, const float* beta,
                                 int batch, int hidden) {
    int b = blockIdx.x;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += input[b * hidden + i];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    float mean = s_sum / hidden;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float diff = input[b * hidden + i] - mean;
        var_sum += diff * diff;
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var_sum);
    __syncthreads();
    float rstd = rsqrtf(s_var / hidden + 1e-5f);
    
    // Normalize
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float x = input[b * hidden + i];
        output[b * hidden + i] = gamma[i] * (x - mean) * rstd + beta[i];
    }
}

// Fused LayerNorm + GELU
__global__ void fusedLayerNormGELU(const float* input, float* output,
                                    const float* gamma, const float* beta,
                                    int batch, int hidden) {
    int b = blockIdx.x;
    
    // Same LayerNorm computation...
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += input[b * hidden + i];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    float mean = s_sum / hidden;
    
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float diff = input[b * hidden + i] - mean;
        var_sum += diff * diff;
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var_sum);
    __syncthreads();
    float rstd = rsqrtf(s_var / hidden + 1e-5f);
    
    // Fused: Normalize + GELU in one pass
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float x = input[b * hidden + i];
        float normed = gamma[i] * (x - mean) * rstd + beta[i];
        // GELU
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * 
                   (normed + 0.044715f * normed * normed * normed)));
        output[b * hidden + i] = normed * cdf;
    }
}

int main() {
    printf("Week 30 Day 4: Fused Operations\n\n");
    
    const int batch = 64, hidden = 768;  // BERT-base
    
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_beta, hidden * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    
    // Benchmark fused
    fusedLayerNormGELU<<<batch, 256>>>(d_input, d_output, d_gamma, d_beta, batch, hidden);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fusedLayerNormGELU<<<batch, 256>>>(d_input, d_output, d_gamma, d_beta, batch, hidden);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fusedMs;
    cudaEventElapsedTime(&fusedMs, start, stop);
    fusedMs /= iterations;
    
    printf("Fused LayerNorm+GELU Performance:\n");
    printf("  Batch: %d, Hidden: %d\n", batch, hidden);
    printf("  Fused kernel: %.3f ms\n", fusedMs);
    printf("\nFusion Benefits:\n");
    printf("  - Single memory read/write (vs 2 separate)\n");
    printf("  - Reduced kernel launch overhead\n");
    printf("  - Better cache utilization\n");
    printf("  - Typical speedup: 1.5-2×\n");
    
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_gamma); cudaFree(d_beta);
    
    return 0;
}
