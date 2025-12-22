/**
 * Week 34, Day 6: LayerNorm Summary
 * 
 * Complete comparison of normalization techniques.
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

// 1. LayerNorm (GPT, BERT)
__global__ void layerNormKernel(const float* x, const float* gamma, const float* beta,
                                float* y, int hidden, float eps) {
    __shared__ float s_mean, s_var_inv;
    int batch = blockIdx.x;
    const float* in = x + batch * hidden;
    float* out = y + batch * hidden;
    
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = in[i];
        sum += val;
        sum_sq += val * val;
    }
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    if (threadIdx.x == 0) {
        float mean = sum / hidden;
        float var = sum_sq / hidden - mean * mean;
        s_mean = mean;
        s_var_inv = rsqrtf(var + eps);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out[i] = (in[i] - s_mean) * s_var_inv * gamma[i] + beta[i];
    }
}

// 2. RMSNorm (LLaMA)
__global__ void rmsNormKernel(const float* x, const float* gamma, float* y, int hidden, float eps) {
    __shared__ float s_rms_inv;
    int batch = blockIdx.x;
    const float* in = x + batch * hidden;
    float* out = y + batch * hidden;
    
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        sum_sq += in[i] * in[i];
    sum_sq = blockReduceSum(sum_sq);
    
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(sum_sq / hidden + eps);
    __syncthreads();
    
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[i] = in[i] * s_rms_inv * gamma[i];
}

// 3. BatchNorm (for comparison, CNNs)
__global__ void batchNormKernel(const float* x, const float* gamma, const float* beta,
                                 float* y, int batch, int channels, int spatial, float eps) {
    int c = blockIdx.x;
    const int total = batch * spatial;
    
    __shared__ float s_mean, s_var_inv;
    
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        float val = x[(b * channels + c) * spatial + s];
        sum += val;
        sum_sq += val * val;
    }
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    if (threadIdx.x == 0) {
        float mean = sum / total;
        float var = sum_sq / total - mean * mean;
        s_mean = mean;
        s_var_inv = rsqrtf(var + eps);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        int idx = (b * channels + c) * spatial + s;
        y[idx] = (x[idx] - s_mean) * s_var_inv * gamma[c] + beta[c];
    }
}

int main() {
    printf("Week 34 Day 6: LayerNorm Summary\n\n");
    
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║              NORMALIZATION TECHNIQUES COMPARISON                  ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                   ║\n");
    printf("║  BatchNorm: Normalize across batch (CNNs, needs running stats)   ║\n");
    printf("║    Stats over: [B, H, W] for each channel C                      ║\n");
    printf("║    Params: γ(C), β(C)                                            ║\n");
    printf("║                                                                   ║\n");
    printf("║  LayerNorm: Normalize across features (Transformers)             ║\n");
    printf("║    Stats over: [hidden] for each token                           ║\n");
    printf("║    Params: γ(hidden), β(hidden)                                  ║\n");
    printf("║                                                                   ║\n");
    printf("║  RMSNorm: Simplified LayerNorm (Modern LLMs)                     ║\n");
    printf("║    No mean subtraction, just scale by RMS                        ║\n");
    printf("║    Params: γ(hidden) only                                        ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Week 34 Learning Journey:\n");
    printf("┌─────┬─────────────────────────────────────────────────────────────┐\n");
    printf("│ Day │ Topic                                                       │\n");
    printf("├─────┼─────────────────────────────────────────────────────────────┤\n");
    printf("│  1  │ LayerNorm basics: mean, variance, normalize                 │\n");
    printf("│  2  │ Welford's algorithm: numerically stable variance            │\n");
    printf("│  3  │ Forward pass: vec4 loads, fused operations                  │\n");
    printf("│  4  │ Backward pass: dx, dgamma, dbeta gradients                  │\n");
    printf("│  5  │ RMSNorm: simpler, ~15%% faster                               │\n");
    printf("│  6  │ Summary and comparison                                      │\n");
    printf("└─────┴─────────────────────────────────────────────────────────────┘\n\n");
    
    // Benchmark all variants
    const int batch = 64, hidden = 4096;
    const float eps = 1e-6f;
    
    float *d_x, *d_gamma, *d_beta, *d_y;
    cudaMalloc(&d_x, batch * hidden * sizeof(float));
    cudaMalloc(&d_gamma, hidden * sizeof(float));
    cudaMalloc(&d_beta, hidden * sizeof(float));
    cudaMalloc(&d_y, batch * hidden * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iters = 1000;
    float ms;
    
    // LayerNorm timing
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        layerNormKernel<<<batch, BLOCK_SIZE>>>(d_x, d_gamma, d_beta, d_y, hidden, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float ln_time = ms / iters;
    
    // RMSNorm timing  
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        rmsNormKernel<<<batch, BLOCK_SIZE>>>(d_x, d_gamma, d_y, hidden, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float rms_time = ms / iters;
    
    printf("Performance Comparison (%d×%d):\n", batch, hidden);
    printf("┌─────────────┬───────────────┬────────────────┐\n");
    printf("│ Method      │ Time (µs)     │ Relative       │\n");
    printf("├─────────────┼───────────────┼────────────────┤\n");
    printf("│ LayerNorm   │ %7.1f       │ 1.00×          │\n", ln_time * 1000);
    printf("│ RMSNorm     │ %7.1f       │ %.2f× faster   │\n", rms_time * 1000, ln_time / rms_time);
    printf("└─────────────┴───────────────┴────────────────┘\n\n");
    
    printf("Key Optimizations Learned:\n");
    printf("  1. Welford's algorithm for numerical stability\n");
    printf("  2. Warp shuffle + shared memory reductions\n");
    printf("  3. Vectorized memory access (float4)\n");
    printf("  4. Fused operations to reduce memory traffic\n");
    printf("  5. Online algorithms for single-pass computation\n\n");
    
    printf("Next Week Preview: Attention Building Blocks\n");
    printf("  - QK^T computation\n");
    printf("  - Causal and padding masks\n");
    printf("  - Softmax over attention scores\n");
    printf("  - Output projection (PV)\n");
    
    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
