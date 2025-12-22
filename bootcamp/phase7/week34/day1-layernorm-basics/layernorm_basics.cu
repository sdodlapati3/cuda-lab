/**
 * Week 34, Day 1: LayerNorm Basics
 * 
 * LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
 * 
 * Normalizes across the feature dimension (last axis).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

void layerNormCPU(const float* input, const float* gamma, const float* beta,
                  float* output, int batch, int hidden, float eps) {
    for (int b = 0; b < batch; b++) {
        const float* x = input + b * hidden;
        float* y = output + b * hidden;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x[i];
        mean /= hidden;
        
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= hidden;
        
        // Normalize
        float invStd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < hidden; i++) {
            y[i] = gamma[i] * (x[i] - mean) * invStd + beta[i];
        }
    }
}

__global__ void layerNormNaiveKernel(const float* input, const float* gamma,
                                      const float* beta, float* output,
                                      int hidden, float eps) {
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden;
    float* y = output + batch_idx * hidden;
    
    // Single thread computes everything (naive!)
    if (threadIdx.x == 0) {
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x[i];
        mean /= hidden;
        
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= hidden;
        
        float invStd = rsqrtf(var + eps);
        for (int i = 0; i < hidden; i++) {
            y[i] = gamma[i] * (x[i] - mean) * invStd + beta[i];
        }
    }
}

int main() {
    printf("Week 34 Day 1: LayerNorm Basics\n\n");
    
    printf("LayerNorm Formula:\n");
    printf("  y = gamma * (x - mean) / sqrt(var + eps) + beta\n\n");
    
    printf("Where:\n");
    printf("  mean = (1/n) * sum(x)\n");
    printf("  var  = (1/n) * sum((x - mean)^2)\n\n");
    
    const int batch = 4;
    const int hidden = 8;
    const float eps = 1e-5f;
    
    float h_input[batch * hidden];
    float h_gamma[hidden], h_beta[hidden];
    float h_output_cpu[batch * hidden];
    float h_output_gpu[batch * hidden];
    
    // Initialize
    for (int i = 0; i < batch * hidden; i++) h_input[i] = (float)(i % 10);
    for (int i = 0; i < hidden; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }
    
    // CPU reference
    layerNormCPU(h_input, h_gamma, h_beta, h_output_cpu, batch, hidden, eps);
    
    // GPU
    float *d_input, *d_gamma, *d_beta, *d_output;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_beta, hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batch * hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden * sizeof(float), cudaMemcpyHostToDevice);
    
    layerNormNaiveKernel<<<batch, 1>>>(d_input, d_gamma, d_beta, d_output, hidden, eps);
    cudaMemcpy(h_output_gpu, d_output, batch * hidden * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Input (batch 0): ");
    for (int i = 0; i < hidden; i++) printf("%.1f ", h_input[i]);
    printf("\n");
    
    printf("Output (batch 0): ");
    for (int i = 0; i < hidden; i++) printf("%.3f ", h_output_cpu[i]);
    printf("\n\n");
    
    // Verify normalized
    float mean = 0, var = 0;
    for (int i = 0; i < hidden; i++) mean += h_output_cpu[i];
    mean /= hidden;
    for (int i = 0; i < hidden; i++) var += (h_output_cpu[i] - mean) * (h_output_cpu[i] - mean);
    var /= hidden;
    
    printf("After normalization: mean=%.6f, var=%.6f\n", mean, var);
    printf("(Should be ~0 and ~1)\n");
    
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_output);
    return 0;
}
