/**
 * Week 35, Day 2: Batched QK^T
 * 
 * Multi-head attention requires batched matmul over:
 *   [batch * num_heads] independent QK^T operations
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16

// Batched QK^T: each block handles one (batch, head) pair
__global__ void batchedQktKernel(
    const float* __restrict__ Q,      // [batch, heads, seq_q, d_k]
    const float* __restrict__ K,      // [batch, heads, seq_k, d_k]
    float* __restrict__ scores,       // [batch, heads, seq_q, seq_k]
    int batch, int heads, int seq_q, int seq_k, int d_k, float scale
) {
    // Determine which batch and head this block handles
    int batch_head_idx = blockIdx.z;
    int b = batch_head_idx / heads;
    int h = batch_head_idx % heads;
    
    // Pointers to this head's Q, K, and output
    int head_size = seq_q * d_k;
    int k_head_size = seq_k * d_k;
    int out_size = seq_q * seq_k;
    
    const float* Q_head = Q + (b * heads + h) * head_size;
    const float* K_head = K + (b * heads + h) * k_head_size;
    float* out_head = scores + (b * heads + h) * out_size;
    
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int d_offset = t * TILE_SIZE;
        
        if (row < seq_q && (d_offset + threadIdx.x) < d_k)
            Qs[threadIdx.y][threadIdx.x] = Q_head[row * d_k + d_offset + threadIdx.x];
        else
            Qs[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < seq_k && (d_offset + threadIdx.y) < d_k)
            Ks[threadIdx.y][threadIdx.x] = K_head[col * d_k + d_offset + threadIdx.y];
        else
            Ks[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++)
            sum += Qs[threadIdx.y][i] * Ks[i][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < seq_q && col < seq_k) {
        out_head[row * seq_k + col] = sum * scale;
    }
}

int main() {
    printf("Week 35 Day 2: Batched QK^T\n\n");
    
    printf("Multi-Head Attention Layout:\n");
    printf("  Q: [batch, heads, seq_q, d_k]\n");
    printf("  K: [batch, heads, seq_k, d_k]\n");
    printf("  Each head computes independent QK^T\n\n");
    
    const int batch = 4, heads = 12, seq_q = 256, seq_k = 256, d_k = 64;
    const float scale = 1.0f / sqrtf((float)d_k);
    
    printf("Config: batch=%d, heads=%d, seq=%d, d_k=%d\n", batch, heads, seq_q, d_k);
    printf("Total independent matmuls: %d\n\n", batch * heads);
    
    size_t Q_size = batch * heads * seq_q * d_k * sizeof(float);
    size_t K_size = batch * heads * seq_k * d_k * sizeof(float);
    size_t out_size = batch * heads * seq_q * seq_k * sizeof(float);
    
    printf("Memory:\n");
    printf("  Q: %.2f MB\n", Q_size / (1024.0f * 1024.0f));
    printf("  K: %.2f MB\n", K_size / (1024.0f * 1024.0f));
    printf("  Scores: %.2f MB\n\n", out_size / (1024.0f * 1024.0f));
    
    float *d_Q, *d_K, *d_scores;
    cudaMalloc(&d_Q, Q_size);
    cudaMalloc(&d_K, K_size);
    cudaMalloc(&d_scores, out_size);
    
    // Initialize with random data
    float *h_Q = new float[batch * heads * seq_q * d_k];
    for (size_t i = 0; i < batch * heads * seq_q * d_k; i++)
        h_Q[i] = (float)(rand() % 100) / 100.0f;
    cudaMemcpy(d_Q, h_Q, Q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, K_size, cudaMemcpyHostToDevice);  // Just copy same data
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (seq_k + TILE_SIZE - 1) / TILE_SIZE,
        (seq_q + TILE_SIZE - 1) / TILE_SIZE,
        batch * heads
    );
    
    // Warmup
    batchedQktKernel<<<grid, block>>>(d_Q, d_K, d_scores, batch, heads, seq_q, seq_k, d_k, scale);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        batchedQktKernel<<<grid, block>>>(d_Q, d_K, d_scores, batch, heads, seq_q, seq_k, d_k, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Compute FLOPS
    long long flops = (long long)batch * heads * seq_q * seq_k * (2 * d_k - 1);
    float gflops = (flops * 100) / (ms / 1000.0f) / 1e9;
    
    printf("Performance:\n");
    printf("  Time: %.2f us/call\n", ms * 10);
    printf("  Throughput: %.1f GFLOPS\n\n", gflops);
    
    printf("Grid configuration:\n");
    printf("  Block: (%d, %d)\n", block.x, block.y);
    printf("  Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("  Total thread blocks: %d\n", grid.x * grid.y * grid.z);
    
    delete[] h_Q;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_scores);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
