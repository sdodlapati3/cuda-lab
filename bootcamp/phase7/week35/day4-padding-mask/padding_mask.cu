/**
 * Week 35, Day 4: Padding Mask
 * 
 * In batched processing, sequences have different lengths.
 * Pad to max length, then mask out padding tokens.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Apply padding mask based on actual sequence lengths
__global__ void applyPaddingMask(
    float* __restrict__ scores,      // [batch, seq_q, seq_k]
    const int* __restrict__ seq_lens, // [batch] actual lengths
    int batch, int seq_q, int seq_k
) {
    int b = blockIdx.z;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx < seq_q && k_idx < seq_k) {
        int actual_len = seq_lens[b];
        
        // Mask if:
        // 1. Query position is padding (q >= actual_len)
        // 2. Key position is padding (k >= actual_len)
        if (q_idx >= actual_len || k_idx >= actual_len) {
            scores[b * seq_q * seq_k + q_idx * seq_k + k_idx] = -INFINITY;
        }
    }
}

// Combined causal + padding mask
__global__ void applyCombinedMask(
    float* __restrict__ scores,
    const int* __restrict__ seq_lens,
    int batch, int seq_q, int seq_k, bool causal
) {
    int b = blockIdx.z;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx < seq_q && k_idx < seq_k) {
        int actual_len = seq_lens[b];
        bool mask = false;
        
        // Padding mask
        if (q_idx >= actual_len || k_idx >= actual_len) {
            mask = true;
        }
        
        // Causal mask
        if (causal && k_idx > q_idx) {
            mask = true;
        }
        
        if (mask) {
            scores[b * seq_q * seq_k + q_idx * seq_k + k_idx] = -INFINITY;
        }
    }
}

void printBatchedPattern(const int* lens, int batch, int max_seq) {
    printf("Batched Attention with Variable Lengths:\n\n");
    for (int b = 0; b < batch; b++) {
        printf("Batch %d (len=%d, max=%d):\n", b, lens[b], max_seq);
        for (int q = 0; q < max_seq && q < 6; q++) {
            printf("  q=%d: [", q);
            for (int k = 0; k < max_seq && k < 8; k++) {
                bool valid = q < lens[b] && k < lens[b];
                printf(" %s", valid ? "✓" : "×");
            }
            if (max_seq > 8) printf(" ...");
            printf(" ]\n");
        }
        if (max_seq > 6) printf("  ...\n");
        printf("\n");
    }
}

int main() {
    printf("Week 35 Day 4: Padding Mask\n\n");
    
    printf("Variable Length Sequences:\n");
    printf("  Batch: ['Hello world', 'Hi', 'How are you today']\n");
    printf("  Lengths: [2, 1, 4]\n");
    printf("  Max length: 4 (pad shorter sequences)\n\n");
    
    const int batch = 4, max_seq = 8;
    int h_seq_lens[] = {8, 5, 3, 6};  // Variable lengths
    
    printBatchedPattern(h_seq_lens, batch, max_seq);
    
    float *d_scores;
    int *d_seq_lens;
    cudaMalloc(&d_scores, batch * max_seq * max_seq * sizeof(float));
    cudaMalloc(&d_seq_lens, batch * sizeof(int));
    
    // Initialize all scores to 1.0
    float *h_scores = new float[batch * max_seq * max_seq];
    for (int i = 0; i < batch * max_seq * max_seq; i++) h_scores[i] = 1.0f;
    cudaMemcpy(d_scores, h_scores, batch * max_seq * max_seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_lens, h_seq_lens, batch * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 block(8, 8);
    dim3 grid((max_seq + 7) / 8, (max_seq + 7) / 8, batch);
    
    // Apply padding mask
    applyPaddingMask<<<grid, block>>>(d_scores, d_seq_lens, batch, max_seq, max_seq);
    cudaMemcpy(h_scores, d_scores, batch * max_seq * max_seq * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Count masked positions per batch
    printf("Mask Statistics:\n");
    printf("┌───────┬────────┬─────────┬──────────────────┐\n");
    printf("│ Batch │ Length │ Valid   │ Masked           │\n");
    printf("├───────┼────────┼─────────┼──────────────────┤\n");
    
    for (int b = 0; b < batch; b++) {
        int valid = 0, masked = 0;
        for (int q = 0; q < max_seq; q++) {
            for (int k = 0; k < max_seq; k++) {
                float v = h_scores[b * max_seq * max_seq + q * max_seq + k];
                if (isinf(v)) masked++;
                else valid++;
            }
        }
        printf("│   %d   │   %d    │   %2d    │   %2d             │\n",
               b, h_seq_lens[b], valid, masked);
    }
    printf("└───────┴────────┴─────────┴──────────────────┘\n\n");
    
    printf("Key Insight:\n");
    printf("  Padding mask prevents attending to/from pad tokens\n");
    printf("  Combined with causal mask for decoder-only models\n\n");
    
    printf("Memory Efficiency:\n");
    printf("  Naive: Always compute full max_seq × max_seq\n");
    printf("  Better: Variable-length attention (FlashAttention v2)\n");
    printf("  Skip entire blocks of computation for short sequences\n");
    
    delete[] h_scores;
    cudaFree(d_scores);
    cudaFree(d_seq_lens);
    
    return 0;
}
