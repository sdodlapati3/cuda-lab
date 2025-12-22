/**
 * Week 33, Day 1: Naive Softmax
 * Understanding the softmax function and numerical issues.
 * 
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * 
 * Problem: exp(x) overflows for x > 88 (FP32)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// CPU reference (naive - will overflow!)
void softmaxCPU_naive(const float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Naive GPU kernel - demonstrates overflow problem
__global__ void softmaxNaiveKernel(const float* input, float* output, int n) {
    // Pass 1: Compute exp and sum
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = expf(input[i]);
        output[i] = val;
        atomicAdd(&s_sum, val);
    }
    __syncthreads();
    
    // Pass 2: Normalize
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] /= s_sum;
    }
}

int main() {
    printf("Week 33 Day 1: Naive Softmax\n\n");
    
    printf("Softmax Function:\n");
    printf("  softmax(x_i) = exp(x_i) / Σ exp(x_j)\n\n");
    
    printf("Numerical Issues:\n");
    printf("  - exp(89) = inf (overflow in FP32)\n");
    printf("  - exp(-100) = 0 (underflow)\n");
    printf("  - inf/inf = NaN\n\n");
    
    // Demonstrate overflow
    const int N = 8;
    float h_input[N] = {1.0f, 2.0f, 3.0f, 100.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float h_output[N];
    
    printf("Input with large value (100.0):\n  ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_input[i]);
    printf("\n\n");
    
    softmaxCPU_naive(h_input, h_output, N);
    printf("Naive Softmax Output (CPU):\n  ");
    for (int i = 0; i < N; i++) printf("%.6f ", h_output[i]);
    printf("\n");
    printf("  Note: exp(100) = %.2e (overflow!)\n\n", expf(100.0f));
    
    // Safe input
    float h_safe[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    printf("Safe input (no overflow):\n  ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_safe[i]);
    printf("\n");
    
    softmaxCPU_naive(h_safe, h_output, N);
    printf("Naive Softmax Output:\n  ");
    for (int i = 0; i < N; i++) printf("%.6f ", h_output[i]);
    printf("\n\n");
    
    // Verify probabilities sum to 1
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    printf("Sum of probabilities: %.6f (should be 1.0)\n\n", sum);
    
    printf("Solution Preview:\n");
    printf("  Stable softmax subtracts max(x) before exp:\n");
    printf("  softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)\n");
    printf("  This prevents overflow while preserving the result.\n");
    
    return 0;
}
