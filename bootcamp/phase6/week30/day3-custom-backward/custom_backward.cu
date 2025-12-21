/**
 * Week 30, Day 3: Custom Backward Pass
 * GELU backward (gradient computation).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// GELU backward:
// d_GELU/d_x = 0.5 * (1 + tanh(a)) + 0.5 * x * sech²(a) * a'
// where a = sqrt(2/π) * (x + 0.044715 * x³)
// Simplified: we compute it directly

__global__ void geluBackwardKernel(const float* grad_output, const float* input,
                                    float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        
        // Constants
        const float sqrt_2_pi = 0.7978845608f;
        const float c = 0.044715f;
        
        // Forward values needed for gradient
        float inner = sqrt_2_pi * (x + c * x3);
        float tanh_inner = tanhf(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        
        // Derivative of inner w.r.t. x
        float d_inner = sqrt_2_pi * (1.0f + 3.0f * c * x * x);
        
        // Full gradient
        float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
        
        grad_input[idx] = grad_output[idx] * grad;
    }
}

/*
 * PyTorch Autograd Function Pattern:
 *
 * class GELUFunction(torch.autograd.Function):
 *     @staticmethod
 *     def forward(ctx, input):
 *         ctx.save_for_backward(input)
 *         return gelu_cuda.forward(input)
 *     
 *     @staticmethod
 *     def backward(ctx, grad_output):
 *         input, = ctx.saved_tensors
 *         return gelu_cuda.backward(grad_output, input)
 */

int main() {
    printf("Week 30 Day 3: Custom Backward Pass\n\n");
    
    printf("Autograd Integration:\n");
    printf("  1. Save tensors needed for backward in forward()\n");
    printf("  2. Implement backward() to compute gradients\n");
    printf("  3. Return gradients w.r.t. each input\n\n");
    
    const int N = 1 << 20;
    
    float *d_grad_out, *d_input, *d_grad_in;
    cudaMalloc(&d_grad_out, N * sizeof(float));
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_grad_in, N * sizeof(float));
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark backward
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    geluBackwardKernel<<<blocks, threads>>>(d_grad_out, d_input, d_grad_in, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        geluBackwardKernel<<<blocks, threads>>>(d_grad_out, d_input, d_grad_in, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;
    
    printf("GELU Backward Performance (%d elements):\n", N);
    printf("  Time: %.3f ms\n", ms);
    printf("  Bandwidth: %.2f GB/s\n", 3.0f * N * sizeof(float) / ms / 1e6);
    
    cudaFree(d_grad_out); cudaFree(d_input); cudaFree(d_grad_in);
    delete[] h_data;
    
    return 0;
}
