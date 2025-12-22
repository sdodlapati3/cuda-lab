/**
 * Week 38, Day 1: FlashAttention Backward Pass
 * 
 * Key insight: Recompute S during backward pass instead of storing.
 * Extra FLOPs but massive memory savings.
 */
#include <cstdio>

int main() {
    printf("Week 38 Day 1: FlashAttention Backward\n\n");
    
    printf("Standard Backward Pass:\n");
    printf("  • Store S matrix from forward: O(N²) memory\n");
    printf("  • Read S in backward for gradient computation\n\n");
    
    printf("FlashAttention Backward:\n");
    printf("  • Don't store S matrix\n");
    printf("  • Recompute S tile-by-tile in backward\n");
    printf("  • Only store O, L (logsumexp) from forward: O(N·d)\n\n");
    
    printf("Backward Algorithm:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ Inputs: Q, K, V, O, L, dO (gradient of output)                  │\n");
    printf("│ Outputs: dQ, dK, dV                                             │\n");
    printf("│                                                                 │\n");
    printf("│ for each KV tile j:                                             │\n");
    printf("│   Load K_j, V_j from HBM                                        │\n");
    printf("│   for each Q tile i:                                            │\n");
    printf("│     Load Q_i, O_i, dO_i, L_i from HBM                           │\n");
    printf("│                                                                 │\n");
    printf("│     // RECOMPUTE attention scores                               │\n");
    printf("│     S_ij = Q_i @ K_j^T / sqrt(d)                                │\n");
    printf("│     P_ij = softmax(S_ij, using stored L_i)                      │\n");
    printf("│                                                                 │\n");
    printf("│     // Compute gradients                                        │\n");
    printf("│     dV_j += P_ij^T @ dO_i                                       │\n");
    printf("│     dP_ij = dO_i @ V_j^T                                        │\n");
    printf("│     dS_ij = P_ij * (dP_ij - sum(dP_ij * P_ij, dim=-1))          │\n");
    printf("│     dQ_i += dS_ij @ K_j                                         │\n");
    printf("│     dK_j += dS_ij^T @ Q_i                                       │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Memory Comparison (N=4096, d=64):\n");
    printf("┌─────────────────────┬───────────────┬───────────────┐\n");
    printf("│ What's Stored       │ Standard      │ FlashAttention│\n");
    printf("├─────────────────────┼───────────────┼───────────────┤\n");
    printf("│ Q, K, V, O          │ 4 × N × d     │ 4 × N × d     │\n");
    printf("│ Attention matrix S  │ N × N         │ 0             │\n");
    printf("│ Logsumexp L         │ 0             │ N             │\n");
    printf("├─────────────────────┼───────────────┼───────────────┤\n");
    printf("│ Total (N=4096,d=64) │ ~67 MB        │ ~4 MB         │\n");
    printf("│ Memory Savings      │ -             │ 94%%           │\n");
    printf("└─────────────────────┴───────────────┴───────────────┘\n\n");
    
    printf("Recomputation Cost:\n");
    printf("  • Extra 2× QK^T computation in backward\n");
    printf("  • But attention is memory-bound, not compute-bound\n");
    printf("  • Net result: ~2× training speedup despite recomputation!\n");
    
    return 0;
}
