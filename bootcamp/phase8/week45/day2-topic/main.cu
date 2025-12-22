/**
 * Week 45, Day 2: Ring and Tree Algorithms
 */
#include <cstdio>

int main() {
    printf("Week 45 Day 2: Ring and Tree AllReduce\n\n");
    
    printf("Ring AllReduce:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ GPU0 → GPU1 → GPU2 → GPU3 → GPU0 (ring topology)                  ║\n");
    printf("║                                                                   ║\n");
    printf("║ Phase 1 (Reduce-Scatter): N-1 steps, each GPU gets 1/N of result  ║\n");
    printf("║ Phase 2 (AllGather): N-1 steps, broadcast partial results         ║\n");
    printf("║                                                                   ║\n");
    printf("║ Total: 2(N-1) steps, 2(N-1)/N × data transferred per GPU          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Tree AllReduce:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║       GPU0 (root)                                                 ║\n");
    printf("║      /    \\                                                       ║\n");
    printf("║   GPU1    GPU2                                                    ║\n");
    printf("║    |        |                                                     ║\n");
    printf("║  GPU3    GPU4                                                     ║\n");
    printf("║                                                                   ║\n");
    printf("║ Reduce up tree, broadcast down                                    ║\n");
    printf("║ Lower latency for small messages                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("When to Use Each:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Ring:  Large messages (gradients) - bandwidth optimal              │\n");
    printf("│ Tree:  Small messages - latency optimal                            │\n");
    printf("│ NCCL:  Automatically chooses based on message size!                │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Bandwidth Analysis (Ring):\n");
    printf("  N GPUs, each with B GB/s interconnect bandwidth\n");
    printf("  Message size: M bytes\n");
    printf("  Time ≈ 2 × (N-1)/N × M / B\n");
    printf("  As N→∞, time → 2M/B (scales linearly with message, not GPU count!)\n");
    
    return 0;
}
