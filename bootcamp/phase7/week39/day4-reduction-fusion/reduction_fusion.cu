/**
 * Week 39, Day 4: Reduction Fusion
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Fused LayerNorm: single kernel for mean, var, normalize
__global__ void fusedLayerNorm(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int rows, int cols
) {
    extern __shared__ float smem[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row_x = x + row * cols;
    float* row_out = out + row * cols;
    
    // Welford's online algorithm for mean and variance
    float mean = 0.0f, M2 = 0.0f;
    int count = 0;
    
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_x[i];
        count++;
        float delta = val - mean;
        mean += delta / count;
        M2 += delta * (val - mean);
    }
    
    // Store in shared memory for reduction
    smem[tid] = mean;
    smem[tid + blockDim.x] = M2;
    smem[tid + 2 * blockDim.x] = (float)count;
    __syncthreads();
    
    // Parallel reduction to combine Welford statistics
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float n_a = smem[tid + 2 * blockDim.x];
            float n_b = smem[tid + s + 2 * blockDim.x];
            float n = n_a + n_b;
            
            if (n > 0) {
                float delta = smem[tid + s] - smem[tid];
                smem[tid] = (smem[tid] * n_a + smem[tid + s] * n_b) / n;
                smem[tid + blockDim.x] += smem[tid + s + blockDim.x] + 
                    delta * delta * n_a * n_b / n;
                smem[tid + 2 * blockDim.x] = n;
            }
        }
        __syncthreads();
    }
    
    float final_mean = smem[0];
    float var = smem[blockDim.x] / cols;
    float rstd = rsqrtf(var + 1e-5f);
    
    // Normalize in same kernel
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_x[i];
        row_out[i] = gamma[i] * (val - final_mean) * rstd + beta[i];
    }
}

int main() {
    printf("Week 39 Day 4: Reduction Fusion\n\n");
    
    printf("Fused Reduction Patterns:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ LayerNorm: mean + var + normalize in single kernel                ║\n");
    printf("║ Softmax: max + sum + normalize in single kernel                   ║\n");
    printf("║                                                                   ║\n");
    printf("║ Key: Keep intermediate values in registers/shared memory          ║\n");
    printf("║      Never write reduction results to global memory               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    const int rows = 4096, cols = 1024;
    
    float *d_x, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_x, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    
    int blockSize = 256;
    size_t smemSize = 3 * blockSize * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        fusedLayerNorm<<<rows, blockSize, smemSize>>>(
            d_out, d_x, d_gamma, d_beta, rows, cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Fused LayerNorm (%dx%d): %.2f ms / 1000 iters\n", rows, cols, ms);
    printf("Per-call: %.3f µs\n\n", ms);
    
    printf("Fusion Benefit for LayerNorm:\n");
    printf("┌───────────────┬───────────────────────────────────────┐\n");
    printf("│ Unfused       │ 3 kernel launches, 4× memory traffic  │\n");
    printf("│ Fused         │ 1 kernel launch, 2× memory traffic    │\n");
    printf("│ Speedup       │ ~2× (memory bound)                    │\n");
    printf("└───────────────┴───────────────────────────────────────┘\n");
    
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
