/**
 * Week 37, Day 5: Causal FlashAttention
 * 
 * For autoregressive (decoder) models:
 * - Position i can only attend to positions 0..i
 * - Skip entire tiles that are fully masked
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

void demonstrateCausalTiling() {
    printf("Causal Masking with Tiling:\n\n");
    
    printf("Full attention matrix with causal mask:\n");
    printf("  ┌───┬───┬───┬───┬───┬───┬───┬───┐\n");
    printf("  │ ✓ │ × │ × │ × │ × │ × │ × │ × │  q=0\n");
    printf("  │ ✓ │ ✓ │ × │ × │ × │ × │ × │ × │  q=1\n");
    printf("  │ ✓ │ ✓ │ ✓ │ × │ × │ × │ × │ × │  q=2\n");
    printf("  │ ✓ │ ✓ │ ✓ │ ✓ │ × │ × │ × │ × │  q=3\n");
    printf("  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ × │ × │ × │  q=4\n");
    printf("  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ × │ × │  q=5\n");
    printf("  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ × │  q=6\n");
    printf("  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │  q=7\n");
    printf("  └───┴───┴───┴───┴───┴───┴───┴───┘\n");
    printf("    k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7\n\n");
    
    printf("With 4×4 tiles:\n");
    printf("  ┌───────────┬───────────┐\n");
    printf("  │ PARTIAL   │ SKIP      │  Q block 0 (q=0-3)\n");
    printf("  │ (compute  │ (all ×)   │\n");
    printf("  │  with mask│           │\n");
    printf("  ├───────────┼───────────┤\n");
    printf("  │ FULL      │ PARTIAL   │  Q block 1 (q=4-7)\n");
    printf("  │ (all ✓)   │ (compute  │\n");
    printf("  │           │  with mask│\n");
    printf("  └───────────┴───────────┘\n");
    printf("    K block 0   K block 1\n\n");
    
    printf("Optimization:\n");
    printf("  • FULL tiles: compute without mask check\n");
    printf("  • PARTIAL tiles: apply mask per-element\n");
    printf("  • SKIP tiles: don't compute at all!\n");
}

// Check if a tile is fully masked (can skip)
__device__ bool shouldSkipTile(int q_start, int q_end, int k_start, int k_end) {
    // Causal: q can attend to k where k <= q
    // Skip if smallest k > largest q (entire tile masked)
    return k_start > q_end - 1;
}

// Check if a tile needs per-element masking
__device__ bool needsMasking(int q_start, int q_end, int k_start, int k_end) {
    // Need masking if tile is on the diagonal or partially masked
    // Full tile (no masking) if largest k <= smallest q
    return !(k_end - 1 <= q_start);
}

int main() {
    printf("Week 37 Day 5: Causal FlashAttention\n\n");
    
    demonstrateCausalTiling();
    
    printf("Causal Mask in FlashAttention:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ // Modified inner loop for causal attention                     │\n");
    printf("│ for k_start in range(0, seq, Bc):                               │\n");
    printf("│     k_end = min(k_start + Bc, seq)                              │\n");
    printf("│                                                                 │\n");
    printf("│     // Early exit: skip fully masked tiles                      │\n");
    printf("│     if k_start > q_end - 1:                                     │\n");
    printf("│         break  // All remaining K tiles are masked              │\n");
    printf("│                                                                 │\n");
    printf("│     // Load K, V tiles...                                       │\n");
    printf("│     // Compute S_ij = Q_i @ K_j^T                               │\n");
    printf("│                                                                 │\n");
    printf("│     // Apply causal mask to S_ij                                │\n");
    printf("│     for qi in range(Br):                                        │\n");
    printf("│         global_q = q_start + qi                                 │\n");
    printf("│         for kj in range(Bc):                                    │\n");
    printf("│             global_k = k_start + kj                             │\n");
    printf("│             if global_k > global_q:                             │\n");
    printf("│                 S_ij[qi, kj] = -inf                             │\n");
    printf("│                                                                 │\n");
    printf("│     // Continue with online softmax...                          │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");
    
    // Compute savings
    const int seq = 2048;
    const int Br = 64, Bc = 64;
    
    int full_tiles = 0, partial_tiles = 0, skip_tiles = 0;
    
    for (int q_start = 0; q_start < seq; q_start += Br) {
        int q_end = q_start + Br;
        for (int k_start = 0; k_start < seq; k_start += Bc) {
            int k_end = k_start + Bc;
            
            if (k_start > q_end - 1) {
                skip_tiles++;
            } else if (k_end - 1 <= q_start) {
                full_tiles++;
            } else {
                partial_tiles++;
            }
        }
    }
    
    int total_tiles = (seq / Br) * (seq / Bc);
    printf("Tile Statistics (seq=%d, tile=%d×%d):\n", seq, Br, Bc);
    printf("  Total tiles: %d\n", total_tiles);
    printf("  Full tiles (no mask check): %d (%.1f%%)\n", 
           full_tiles, 100.0f * full_tiles / total_tiles);
    printf("  Partial tiles (need mask): %d (%.1f%%)\n",
           partial_tiles, 100.0f * partial_tiles / total_tiles);
    printf("  Skip tiles (all masked): %d (%.1f%%)\n",
           skip_tiles, 100.0f * skip_tiles / total_tiles);
    printf("\nCompute saved by skipping: %.1f%%\n", 100.0f * skip_tiles / total_tiles);
    
    return 0;
}
