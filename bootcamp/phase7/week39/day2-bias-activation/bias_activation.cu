/**
 * Week 39, Day 2: Bias + Activation Fusion
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__device__ __forceinline__ float relu(float x) { return fmaxf(0.0f, x); }

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float swiglu(float x, float gate) {
    float silu = x / (1.0f + expf(-x));
    return silu * gate;
}

// Fused matmul output + bias + activation
__global__ void fusedBiasActivation(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int rows, int cols,
    int act_type  // 0=relu, 1=gelu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float v = out[idx] + bias[col];
        
        if (act_type == 0) v = relu(v);
        else if (act_type == 1) v = gelu(v);
        
        out[idx] = v;
    }
}

// SwiGLU: fused gate computation (LLaMA FFN)
__global__ void fusedSwiGLU(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = swiglu(x[idx], gate[idx]);
    }
}

int main() {
    printf("Week 39 Day 2: Bias + Activation Fusion\n\n");
    
    printf("Common Fusion Patterns in Transformers:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. MatMul + Bias + ReLU       (Classic MLP)                       ║\n");
    printf("║ 2. MatMul + Bias + GELU       (BERT/GPT FFN)                      ║\n");
    printf("║ 3. MatMul + SwiGLU            (LLaMA FFN)                         ║\n");
    printf("║ 4. Attention + Dropout        (Training)                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    const int n = 4096, d = 4096;
    
    float *d_out, *d_bias;
    cudaMalloc(&d_out, n * d * sizeof(float));
    cudaMalloc(&d_bias, d * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n * d + 255) / 256);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("Benchmarking Fused Bias+GELU (%dx%d):\n", n, d);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        fusedBiasActivation<<<grid, block>>>(d_out, d_bias, n, d, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float gb = n * d * sizeof(float) * 2 / 1e9;  // read + write
    float bw = gb / (ms / 1000.0f) * 1000;
    printf("  Time: %.3f ms/1000 iters\n", ms);
    printf("  Bandwidth: %.1f GB/s\n\n", bw);
    
    printf("Memory Efficiency:\n");
    printf("┌──────────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Pattern          │ Unfused Traffic │ Fused Traffic   │\n");
    printf("├──────────────────┼─────────────────┼─────────────────┤\n");
    printf("│ MatMul+Bias+GELU │ 4× N            │ 2× N            │\n");
    printf("│ (matmul output,  │ (read, write,   │ (read, write)   │\n");
    printf("│  add, activate)  │  read, write)   │                 │\n");
    printf("└──────────────────┴─────────────────┴─────────────────┘\n");
    
    cudaFree(d_out); cudaFree(d_bias);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
