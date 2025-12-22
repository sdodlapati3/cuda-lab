/**
 * Week 33, Day 2: Stable Softmax
 * Max subtraction prevents overflow.
 * 
 * softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * 
 * Three-pass algorithm:
 * 1. Find max
 * 2. Compute exp(x - max) and sum
 * 3. Normalize
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// CPU reference (stable)
void softmaxCPU_stable(const float* input, float* output, int n) {
    // Pass 1: Find max
    float maxVal = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    
    // Pass 2: exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - maxVal);
        sum += output[i];
    }
    
    // Pass 3: Normalize
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Stable GPU kernel - three passes
__global__ void softmaxStableKernel(const float* input, float* output, int n) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    // Pass 1: Find max (parallel reduction)
    float localMax = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        localMax = fmaxf(localMax, input[i]);
    }
    
    // Warp reduction for max
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }
    
    if (threadIdx.x == 0) s_max = localMax;
    __syncthreads();
    float maxVal = s_max;
    
    // Pass 2: exp(x - max) and sum
    float localSum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = expf(input[i] - maxVal);
        output[i] = val;
        localSum += val;
    }
    
    // Warp reduction for sum
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    
    if (threadIdx.x == 0) s_sum = localSum;
    __syncthreads();
    float sum = s_sum;
    
    // Pass 3: Normalize
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] /= sum;
    }
}

int main() {
    printf("Week 33 Day 2: Stable Softmax\n\n");
    
    printf("Stable Softmax Formula:\n");
    printf("  softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)\n\n");
    
    printf("Why it works:\n");
    printf("  exp(x - max) ≤ 1 for all x (prevents overflow)\n");
    printf("  Result is mathematically identical\n\n");
    
    // Test with large values
    const int N = 8;
    float h_input[N] = {1.0f, 2.0f, 3.0f, 100.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float h_output_cpu[N], h_output_gpu[N];
    
    printf("Input (includes 100.0 which would overflow naive):\n  ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_input[i]);
    printf("\n\n");
    
    // CPU stable softmax
    softmaxCPU_stable(h_input, h_output_cpu, N);
    printf("Stable Softmax (CPU):\n  ");
    for (int i = 0; i < N; i++) printf("%.6e ", h_output_cpu[i]);
    printf("\n");
    
    // GPU stable softmax
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    softmaxStableKernel<<<1, 32>>>(d_input, d_output, N);
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Stable Softmax (GPU):\n  ");
    for (int i = 0; i < N; i++) printf("%.6e ", h_output_gpu[i]);
    printf("\n\n");
    
    // Note: exp(100 - 100) = exp(0) = 1, others are tiny
    printf("Observation:\n");
    printf("  exp(100 - 100) = 1.0\n");
    printf("  exp(1 - 100) = exp(-99) ≈ 0\n");
    printf("  Result: element 3 (value 100) dominates\n\n");
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output_cpu[i];
    printf("Sum of probabilities: %.6f ✓\n\n", sum);
    
    printf("Three-Pass Algorithm:\n");
    printf("  Pass 1: max = max(x)           [O(n) reads]\n");
    printf("  Pass 2: exp(x-max), sum        [O(n) reads + writes]\n");
    printf("  Pass 3: normalize              [O(n) reads + writes]\n");
    printf("  Total: 3 passes over data = 3× memory traffic\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
