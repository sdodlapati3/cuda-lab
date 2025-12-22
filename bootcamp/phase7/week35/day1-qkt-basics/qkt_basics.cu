/**
 * Week 35, Day 1: QK^T Basics
 * 
 * Attention scores = Q × K^T / sqrt(d_k)
 * 
 * Shape analysis for multi-head attention:
 *   Q: [batch, heads, seq_q, d_k]
 *   K: [batch, heads, seq_k, d_k]
 *   QK^T: [batch, heads, seq_q, seq_k]
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Simple QK^T kernel for one head
__global__ void qktKernel(
    const float* __restrict__ Q,  // [seq_q, d_k]
    const float* __restrict__ K,  // [seq_k, d_k]
    float* __restrict__ scores,   // [seq_q, seq_k]
    int seq_q, int seq_k, int d_k, float scale
) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx < seq_q && k_idx < seq_k) {
        float sum = 0.0f;
        for (int d = 0; d < d_k; d++) {
            sum += Q[q_idx * d_k + d] * K[k_idx * d_k + d];
        }
        scores[q_idx * seq_k + k_idx] = sum * scale;
    }
}

// Tiled QK^T with shared memory
#define TILE_SIZE 16

__global__ void qktTiledKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int seq_q, int seq_k, int d_k, float scale
) {
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int d_offset = t * TILE_SIZE;
        
        // Load Q tile
        if (row < seq_q && (d_offset + threadIdx.x) < d_k)
            Qs[threadIdx.y][threadIdx.x] = Q[row * d_k + d_offset + threadIdx.x];
        else
            Qs[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load K tile (transposed access)
        if (col < seq_k && (d_offset + threadIdx.y) < d_k)
            Ks[threadIdx.y][threadIdx.x] = K[col * d_k + d_offset + threadIdx.y];
        else
            Ks[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += Qs[threadIdx.y][i] * Ks[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < seq_q && col < seq_k) {
        scores[row * seq_k + col] = sum * scale;
    }
}

// CPU reference
void qktCPU(const float* Q, const float* K, float* scores,
            int seq_q, int seq_k, int d_k, float scale) {
    for (int q = 0; q < seq_q; q++) {
        for (int k = 0; k < seq_k; k++) {
            float sum = 0.0f;
            for (int d = 0; d < d_k; d++) {
                sum += Q[q * d_k + d] * K[k * d_k + d];
            }
            scores[q * seq_k + k] = sum * scale;
        }
    }
}

int main() {
    printf("Week 35 Day 1: QK^T Basics\n\n");
    
    printf("Attention Score Computation:\n");
    printf("  scores = Q × K^T / sqrt(d_k)\n\n");
    
    printf("Shape Analysis:\n");
    printf("┌──────────┬────────────────────────────────────┐\n");
    printf("│ Input    │ Shape                              │\n");
    printf("├──────────┼────────────────────────────────────┤\n");
    printf("│ Q        │ [batch, heads, seq_q, d_k]         │\n");
    printf("│ K        │ [batch, heads, seq_k, d_k]         │\n");
    printf("│ QK^T     │ [batch, heads, seq_q, seq_k]       │\n");
    printf("└──────────┴────────────────────────────────────┘\n\n");
    
    const int seq_q = 512, seq_k = 512, d_k = 64;
    const float scale = 1.0f / sqrtf((float)d_k);
    
    printf("Test: seq_q=%d, seq_k=%d, d_k=%d, scale=%.4f\n\n", seq_q, seq_k, d_k, scale);
    
    float *h_Q = new float[seq_q * d_k];
    float *h_K = new float[seq_k * d_k];
    float *h_scores_cpu = new float[seq_q * seq_k];
    float *h_scores_gpu = new float[seq_q * seq_k];
    
    for (int i = 0; i < seq_q * d_k; i++) h_Q[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < seq_k * d_k; i++) h_K[i] = (float)(rand() % 100) / 100.0f;
    
    // CPU reference
    qktCPU(h_Q, h_K, h_scores_cpu, seq_q, seq_k, d_k, scale);
    
    // GPU
    float *d_Q, *d_K, *d_scores;
    cudaMalloc(&d_Q, seq_q * d_k * sizeof(float));
    cudaMalloc(&d_K, seq_k * d_k * sizeof(float));
    cudaMalloc(&d_scores, seq_q * seq_k * sizeof(float));
    
    cudaMemcpy(d_Q, h_Q, seq_q * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq_k * d_k * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((seq_k + TILE_SIZE - 1) / TILE_SIZE, (seq_q + TILE_SIZE - 1) / TILE_SIZE);
    
    qktTiledKernel<<<grid, block>>>(d_Q, d_K, d_scores, seq_q, seq_k, d_k, scale);
    cudaMemcpy(h_scores_gpu, d_scores, seq_q * seq_k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    float max_diff = 0.0f;
    for (int i = 0; i < seq_q * seq_k; i++)
        max_diff = fmaxf(max_diff, fabsf(h_scores_cpu[i] - h_scores_gpu[i]));
    printf("CPU vs GPU max diff: %.2e\n\n", max_diff);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        qktTiledKernel<<<grid, block>>>(d_Q, d_K, d_scores, seq_q, seq_k, d_k, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tiled QK^T: %.2f us/call\n", ms);
    
    printf("\nWhy scale by 1/sqrt(d_k)?\n");
    printf("  - Dot products grow with d_k\n");
    printf("  - Large values → softmax saturates\n");
    printf("  - Scaling keeps values in reasonable range\n");
    
    delete[] h_Q; delete[] h_K; delete[] h_scores_cpu; delete[] h_scores_gpu;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_scores);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
