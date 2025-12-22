/**
 * Week 37, Day 6: FlashAttention Concepts Summary
 */
#include <cstdio>

int main() {
    printf("Week 37 Day 6: FlashAttention Summary\n\n");
    
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║             FLASHATTENTION KEY CONCEPTS                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                   ║\n");
    printf("║  1. IO-AWARE ALGORITHM                                            ║\n");
    printf("║     • Focus on HBM ↔ SRAM data movement, not FLOPs                ║\n");
    printf("║     • Memory bandwidth is the bottleneck                          ║\n");
    printf("║     • Trade extra compute for less memory traffic                 ║\n");
    printf("║                                                                   ║\n");
    printf("║  2. TILING                                                        ║\n");
    printf("║     • Process attention in SRAM-sized blocks                      ║\n");
    printf("║     • Br × Bc tile sizes fit Q, K, V, S, O tiles                  ║\n");
    printf("║     • Never materialize full N×N attention matrix                 ║\n");
    printf("║                                                                   ║\n");
    printf("║  3. ONLINE SOFTMAX                                                ║\n");
    printf("║     • Incrementally update max (m) and sum (l)                    ║\n");
    printf("║     • Rescale previous output when max increases                  ║\n");
    printf("║     • Mathematically equivalent to standard softmax               ║\n");
    printf("║                                                                   ║\n");
    printf("║  4. KERNEL FUSION                                                 ║\n");
    printf("║     • Single kernel: QK^T, mask, softmax, PV                      ║\n");
    printf("║     • No intermediate writes to HBM                               ║\n");
    printf("║     • Each element of Q, K, V read once from HBM                  ║\n");
    printf("║                                                                   ║\n");
    printf("║  5. CAUSAL OPTIMIZATION                                           ║\n");
    printf("║     • Skip fully masked tiles entirely                            ║\n");
    printf("║     • ~50%% compute savings for autoregressive                     ║\n");
    printf("║     • Early exit from K tile loop                                 ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Week 37 Learning Journey:\n");
    printf("┌─────┬──────────────────────────────────────────────────────────────┐\n");
    printf("│ Day │ Topic                                                        │\n");
    printf("├─────┼──────────────────────────────────────────────────────────────┤\n");
    printf("│  1  │ Motivation: IO-awareness and memory hierarchy                │\n");
    printf("│  2  │ Tiling strategy: SRAM-sized blocks                           │\n");
    printf("│  3  │ Online softmax in attention context                          │\n");
    printf("│  4  │ Forward pass implementation                                  │\n");
    printf("│  5  │ Causal masking with tile skipping                            │\n");
    printf("│  6  │ Summary and review                                           │\n");
    printf("└─────┴──────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Complexity Comparison:\n");
    printf("┌─────────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Algorithm       │ HBM Reads/Writes│ FLOPs           │\n");
    printf("├─────────────────┼─────────────────┼─────────────────┤\n");
    printf("│ Standard        │ O(N²d + N²)     │ O(N²d)          │\n");
    printf("│ FlashAttention  │ O(N·d)          │ O(N²d)          │\n");
    printf("├─────────────────┼─────────────────┼─────────────────┤\n");
    printf("│ Improvement     │ O(N/d) less IO  │ Same compute    │\n");
    printf("└─────────────────┴─────────────────┴─────────────────┘\n\n");
    
    printf("Next Week (38): FlashAttention Implementation Details\n");
    printf("  • Backward pass and recomputation\n");
    printf("  • FlashAttention-2 parallelization\n");
    printf("  • Integration with frameworks\n");
    printf("  • Benchmarking and comparison\n");
    
    return 0;
}
