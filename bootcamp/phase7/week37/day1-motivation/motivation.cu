/**
 * Week 37, Day 1: FlashAttention Motivation
 * 
 * FlashAttention is not about reducing FLOPs.
 * It's about reducing HBM reads/writes (IO-aware).
 */
#include <cstdio>
#include <cmath>

void analyzeIOComplexity(int seq, int d) {
    // Standard attention
    long long std_read_qk = 2 * seq * d;            // Read Q, K
    long long std_write_s = seq * (long long)seq;   // Write S to HBM
    long long std_read_softmax = seq * (long long)seq;   // Read S
    long long std_write_softmax = seq * (long long)seq;  // Write S
    long long std_read_pv = seq * (long long)seq + seq * d;  // Read S, V
    long long std_write_o = seq * d;                // Write O
    long long std_total = (std_read_qk + std_write_s + std_read_softmax + 
                           std_write_softmax + std_read_pv + std_write_o);
    
    // FlashAttention
    long long flash_read = 3 * seq * d;   // Read Q, K, V (once each)
    long long flash_write = seq * d;      // Write O (once)
    long long flash_total = flash_read + flash_write;
    
    printf("Sequence length: %d, head dim: %d\n", seq, d);
    printf("Standard Attention IO:\n");
    printf("  Total elements: %lld (%.2f MB)\n", std_total, std_total * 4.0 / (1024*1024));
    printf("  S matrix: %lld elements (%.2f MB)\n", (long long)seq * seq, seq * seq * 4.0 / (1024*1024));
    printf("FlashAttention IO:\n");
    printf("  Total elements: %lld (%.2f MB)\n", flash_total, flash_total * 4.0 / (1024*1024));
    printf("  No S matrix stored!\n");
    printf("IO Reduction: %.1f×\n\n", (double)std_total / flash_total);
}

int main() {
    printf("Week 37 Day 1: FlashAttention Motivation\n\n");
    
    printf("The Core Problem:\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║ Standard Attention stores O(N²) intermediate matrix S        ║\n");
    printf("║                                                               ║\n");
    printf("║   S is written to HBM after QK^T                              ║\n");
    printf("║   S is read from HBM for softmax                              ║\n");
    printf("║   S is read again from HBM for PV                             ║\n");
    printf("║                                                               ║\n");
    printf("║   → 3× read/write of O(N²) data through slow HBM!             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("FlashAttention Insight:\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║ Never materialize the full S matrix                          ║\n");
    printf("║                                                               ║\n");
    printf("║   Compute S in small tiles that fit in SRAM                   ║\n");
    printf("║   Apply softmax online (incrementally)                        ║\n");
    printf("║   Accumulate output O incrementally                           ║\n");
    printf("║                                                               ║\n");
    printf("║   → O(N) HBM access instead of O(N²)!                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("IO Complexity Comparison:\n");
    printf("════════════════════════════════════════════════════════════════\n");
    analyzeIOComplexity(512, 64);
    analyzeIOComplexity(2048, 64);
    analyzeIOComplexity(8192, 128);
    
    printf("Key Observations:\n");
    printf("  • Standard attention IO grows as O(N²)\n");
    printf("  • FlashAttention IO grows as O(N)\n");
    printf("  • Speedup increases with sequence length!\n\n");
    
    printf("FlashAttention Tradeoff:\n");
    printf("  MORE compute (recompute S in backward pass)\n");
    printf("  LESS memory bandwidth (no S to HBM)\n");
    printf("  \n");
    printf("  Since attention is memory-bound, this is a WIN!\n\n");
    
    printf("Historical Context:\n");
    printf("  • FlashAttention v1 (2022): Dao et al.\n");
    printf("  • FlashAttention v2 (2023): 2× faster, better parallelism\n");
    printf("  • FlashAttention v3 (2024): H100 optimizations, FP8\n");
    printf("  • Now standard in all major LLM frameworks\n");
    
    return 0;
}
