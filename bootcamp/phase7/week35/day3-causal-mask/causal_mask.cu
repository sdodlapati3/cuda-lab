/**
 * Week 35, Day 3: Causal Masking
 * 
 * For autoregressive models (GPT), each position can only
 * attend to itself and previous positions.
 * 
 * Mask: upper triangle of attention matrix → -inf
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <limits>

// Apply causal mask after computing scores
__global__ void applyCausalMask(
    float* __restrict__ scores,  // [seq, seq]
    int seq
) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx < seq && k_idx < seq) {
        // Causal: q can only attend to k where k <= q
        if (k_idx > q_idx) {
            scores[q_idx * seq + k_idx] = -INFINITY;
        }
    }
}

// Fused QK^T with causal mask (more efficient)
__global__ void qktWithCausalMask(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int seq, int d_k, float scale
) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx < seq && k_idx < seq) {
        if (k_idx > q_idx) {
            // Future position: set to -inf immediately
            scores[q_idx * seq + k_idx] = -INFINITY;
        } else {
            // Valid position: compute dot product
            float sum = 0.0f;
            for (int d = 0; d < d_k; d++) {
                sum += Q[q_idx * d_k + d] * K[k_idx * d_k + d];
            }
            scores[q_idx * seq + k_idx] = sum * scale;
        }
    }
}

// Print attention pattern
void printAttentionPattern(int seq) {
    printf("Causal Attention Pattern (seq=%d):\n", seq);
    for (int q = 0; q < seq; q++) {
        printf("  q=%d: [", q);
        for (int k = 0; k < seq; k++) {
            if (k <= q) printf(" ✓");
            else printf(" ×");
        }
        printf(" ]\n");
    }
}

int main() {
    printf("Week 35 Day 3: Causal Masking\n\n");
    
    printf("Causal Attention:\n");
    printf("  Position i can attend to positions 0..i only\n");
    printf("  Used in: GPT, LLaMA, autoregressive models\n\n");
    
    printf("Implementation Options:\n");
    printf("  1. Post-hoc masking: compute all, then mask\n");
    printf("  2. Fused masking: skip computation for masked positions\n");
    printf("  3. Block-sparse: skip entire blocks (FlashAttention)\n\n");
    
    printAttentionPattern(6);
    printf("\n");
    
    const int seq = 512, d_k = 64;
    const float scale = 1.0f / sqrtf((float)d_k);
    
    float *d_Q, *d_K, *d_scores;
    cudaMalloc(&d_Q, seq * d_k * sizeof(float));
    cudaMalloc(&d_K, seq * d_k * sizeof(float));
    cudaMalloc(&d_scores, seq * seq * sizeof(float));
    
    // Initialize
    float *h_Q = new float[seq * d_k];
    for (int i = 0; i < seq * d_k; i++) h_Q[i] = (float)(rand() % 100) / 100.0f;
    cudaMemcpy(d_Q, h_Q, seq * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, seq * d_k * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((seq + 15) / 16, (seq + 15) / 16);
    
    // Benchmark both approaches
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("Benchmark (seq=%d):\n", seq);
    
    // Fused approach
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        qktWithCausalMask<<<grid, block>>>(d_Q, d_K, d_scores, seq, d_k, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  Fused QK^T + causal mask: %.2f us/call\n", ms);
    
    // Verify mask was applied correctly
    float *h_scores = new float[seq * seq];
    cudaMemcpy(h_scores, d_scores, seq * seq * sizeof(float), cudaMemcpyDeviceToHost);
    
    int masked_count = 0;
    int expected_masked = seq * (seq - 1) / 2;  // Upper triangle
    for (int q = 0; q < seq; q++) {
        for (int k = 0; k < seq; k++) {
            if (k > q && isinf(h_scores[q * seq + k]) && h_scores[q * seq + k] < 0) {
                masked_count++;
            }
        }
    }
    printf("\nMask verification: %d/%d positions masked (%s)\n",
           masked_count, expected_masked,
           masked_count == expected_masked ? "✓" : "✗");
    
    printf("\nAfter softmax, masked positions → 0\n");
    printf("This ensures no information leakage from future\n");
    
    delete[] h_Q; delete[] h_scores;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_scores);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
