/**
 * Week 35, Day 6: PV Output (Attention Output Projection)
 * 
 * Final step: Output = P × V
 * Where P = softmax(QK^T / sqrt(d_k))
 * 
 * Shape: [batch, heads, seq_q, seq_k] × [batch, heads, seq_k, d_v]
 *      = [batch, heads, seq_q, d_v]
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16

// PV matmul: P[seq_q, seq_k] × V[seq_k, d_v] = O[seq_q, d_v]
__global__ void pvOutputKernel(
    const float* __restrict__ P,   // [seq_q, seq_k] attention weights
    const float* __restrict__ V,   // [seq_k, d_v] value vectors
    float* __restrict__ O,         // [seq_q, d_v] output
    int seq_q, int seq_k, int d_v
) {
    __shared__ float Ps[TILE_SIZE][TILE_SIZE];
    __shared__ float Vs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // query position
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // output dim
    
    float sum = 0.0f;
    
    for (int t = 0; t < (seq_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_offset = t * TILE_SIZE;
        
        // Load P tile
        if (row < seq_q && (k_offset + threadIdx.x) < seq_k)
            Ps[threadIdx.y][threadIdx.x] = P[row * seq_k + k_offset + threadIdx.x];
        else
            Ps[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load V tile
        if ((k_offset + threadIdx.y) < seq_k && col < d_v)
            Vs[threadIdx.y][threadIdx.x] = V[(k_offset + threadIdx.y) * d_v + col];
        else
            Vs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += Ps[threadIdx.y][i] * Vs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < seq_q && col < d_v) {
        O[row * d_v + col] = sum;
    }
}

// Full attention in one kernel (for small sequences)
__global__ void fullAttentionKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq, int d_k, int d_v, float scale
) {
    // Simplified: each thread handles one output position
    int q_idx = blockIdx.x;
    int d_idx = threadIdx.x;
    
    if (q_idx >= seq || d_idx >= d_v) return;
    
    // Compute attention scores for this query
    float scores[256];  // Assume seq <= 256
    float max_score = -INFINITY;
    
    for (int k = 0; k < seq; k++) {
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            score += Q[q_idx * d_k + d] * K[k * d_k + d];
        }
        scores[k] = score * scale;
        max_score = fmaxf(max_score, scores[k]);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int k = 0; k < seq; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum += scores[k];
    }
    for (int k = 0; k < seq; k++) {
        scores[k] /= sum;
    }
    
    // Weighted sum of values
    float out = 0.0f;
    for (int k = 0; k < seq; k++) {
        out += scores[k] * V[k * d_v + d_idx];
    }
    
    O[q_idx * d_v + d_idx] = out;
}

int main() {
    printf("Week 35 Day 6: PV Output\n\n");
    
    printf("Complete Attention:\n");
    printf("  1. QK^T: Compute attention scores\n");
    printf("  2. Scale: Divide by sqrt(d_k)\n");
    printf("  3. Mask: Apply causal/padding mask\n");
    printf("  4. Softmax: Normalize to probabilities\n");
    printf("  5. PV: Weighted sum of values\n\n");
    
    printf("Week 35 Summary - Attention Building Blocks:\n");
    printf("┌─────┬─────────────────────────────────────────────────┐\n");
    printf("│ Day │ Topic                                           │\n");
    printf("├─────┼─────────────────────────────────────────────────┤\n");
    printf("│  1  │ QK^T basics: naive and tiled matmul             │\n");
    printf("│  2  │ Batched QK^T for multi-head attention           │\n");
    printf("│  3  │ Causal masking for autoregressive               │\n");
    printf("│  4  │ Padding mask for variable lengths               │\n");
    printf("│  5  │ Row-wise softmax with online algorithm          │\n");
    printf("│  6  │ PV output and full attention                    │\n");
    printf("└─────┴─────────────────────────────────────────────────┘\n\n");
    
    const int seq = 128, d_k = 64, d_v = 64;
    const float scale = 1.0f / sqrtf((float)d_k);
    
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, seq * d_k * sizeof(float));
    cudaMalloc(&d_K, seq * d_k * sizeof(float));
    cudaMalloc(&d_V, seq * d_v * sizeof(float));
    cudaMalloc(&d_O, seq * d_v * sizeof(float));
    
    // Initialize
    float *h_Q = new float[seq * d_k];
    for (int i = 0; i < seq * d_k; i++) h_Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    cudaMemcpy(d_Q, h_Q, seq * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, seq * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q, seq * d_v * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run full attention
    fullAttentionKernel<<<seq, d_v>>>(d_Q, d_K, d_V, d_O, seq, d_k, d_v, scale);
    cudaDeviceSynchronize();
    
    float *h_O = new float[seq * d_v];
    cudaMemcpy(h_O, d_O, seq * d_v * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Output sample (first query position):\n  [");
    for (int i = 0; i < 8; i++) printf("%.3f ", h_O[i]);
    printf("...]\n\n");
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        fullAttentionKernel<<<seq, d_v>>>(d_Q, d_K, d_V, d_O, seq, d_k, d_v, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Full attention (seq=%d): %.2f us/call\n\n", seq, ms);
    
    printf("Memory Analysis:\n");
    printf("  Standard attention stores full [seq × seq] matrix\n");
    printf("  Memory: O(seq²) - problematic for long sequences\n\n");
    
    printf("Next Week Preview: FlashAttention\n");
    printf("  Avoid storing full attention matrix\n");
    printf("  Online softmax + tiling = O(seq) memory\n");
    printf("  IO-aware algorithm for HBM efficiency\n");
    
    delete[] h_Q; delete[] h_O;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
