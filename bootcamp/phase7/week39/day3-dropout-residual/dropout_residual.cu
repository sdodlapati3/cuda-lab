/**
 * Week 39, Day 3: Dropout + Residual Fusion
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

// Fused dropout + residual add (training pattern)
__global__ void fusedDropoutResidual(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float p,           // dropout probability
    unsigned long seed,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        float val = x[idx];
        float keep = curand_uniform(&state) > p ? 1.0f : 0.0f;
        val = val * keep / (1.0f - p);  // Scale to maintain expectation
        
        out[idx] = val + residual[idx];
    }
}

// Fused bias + dropout + residual (common in attention)
__global__ void fusedBiasDropoutResidual(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float p,
    unsigned long seed,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        float val = x[idx] + bias[col];
        float keep = curand_uniform(&state) > p ? 1.0f : 0.0f;
        val = val * keep / (1.0f - p);
        
        out[idx] = val + residual[idx];
    }
}

int main() {
    printf("Week 39 Day 3: Dropout + Residual Fusion\n\n");
    
    printf("Transformer Block Pattern:\n");
    printf("╔═════════════════════════════════════════════════════════════════════╗\n");
    printf("║ x = residual + Dropout(Attention(LN(x)))                            ║\n");
    printf("║ x = residual + Dropout(FFN(LN(x)))                                  ║\n");
    printf("║                                                                     ║\n");
    printf("║ Fusion opportunity: Dropout + Residual Add                          ║\n");
    printf("║   • Saves one memory read/write cycle                               ║\n");
    printf("║   • Common in training (dropout disabled at inference)              ║\n");
    printf("╚═════════════════════════════════════════════════════════════════════╝\n\n");
    
    const int n = 4096 * 4096;
    const float p = 0.1f;
    
    float *d_x, *d_res, *d_out;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_res, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        fusedDropoutResidual<<<grid, block>>>(d_out, d_x, d_res, p, i, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Fused Dropout+Residual: %.2f ms / 100 iters\n", ms);
    printf("Size: %d elements (%.1f MB)\n\n", n, n * sizeof(float) / 1e6);
    
    printf("Memory Traffic Comparison:\n");
    printf("┌──────────────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Operation            │ Unfused         │ Fused           │\n");
    printf("├──────────────────────┼─────────────────┼─────────────────┤\n");
    printf("│ Dropout + Residual   │ 4× N read/write │ 3× N read/write │\n");
    printf("│ Bias+Drop+Residual   │ 5× N read/write │ 4× N read/write │\n");
    printf("└──────────────────────┴─────────────────┴─────────────────┘\n");
    
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
