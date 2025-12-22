/**
 * Week 47, Day 1: Data Parallelism Deep Dive
 */
#include <cstdio>

int main() {
    printf("Week 47 Day 1: Data Parallelism\n\n");
    
    printf("Data Parallelism Concept:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Same model on all GPUs                                          ║\n");
    printf("║ • Different data batches on each GPU                              ║\n");
    printf("║ • Gradients synchronized after each backward pass                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Visual:\n");
    printf("  GPU0: Model + Batch0 → Grad0 ─┐\n");
    printf("  GPU1: Model + Batch1 → Grad1 ─┼→ AllReduce → Update All\n");
    printf("  GPU2: Model + Batch2 → Grad2 ─┤\n");
    printf("  GPU3: Model + Batch3 → Grad3 ─┘\n\n");
    
    printf("DDP vs DataParallel:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ DataParallel (DP):                                                 │\n");
    printf("│   • Single-process, multi-thread                                   │\n");
    printf("│   • GIL bottleneck, master GPU overhead                            │\n");
    printf("│   • DON'T USE (legacy)                                             │\n");
    printf("│                                                                    │\n");
    printf("│ DistributedDataParallel (DDP):                                     │\n");
    printf("│   • Multi-process (one per GPU)                                    │\n");
    printf("│   • No GIL, better scaling                                         │\n");
    printf("│   • USE THIS                                                       │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Scaling Efficiency:\n");
    printf("```\n");
    printf("Ideal: N GPUs → N× throughput\n");
    printf("Real:  N GPUs → ~0.9N× throughput (communication overhead)\n");
    printf("```\n");
    
    return 0;
}
