/**
 * Week 47, Day 6: Week Summary
 */
#include <cstdio>

int main() {
    printf("Week 47 Summary: Multi-GPU Patterns\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Day 1: Data Parallelism\n");
    printf("  • Same model, different data\n");
    printf("  • DDP > DataParallel\n\n");
    
    printf("Day 2: Tensor Parallelism\n");
    printf("  • Split layers across GPUs\n");
    printf("  • Column/Row parallel patterns\n\n");
    
    printf("Day 3: Pipeline Parallelism\n");
    printf("  • Split layers sequentially\n");
    printf("  • GPipe, 1F1B schedules\n\n");
    
    printf("Day 4: FSDP\n");
    printf("  • Shard params + grads + optimizer\n");
    printf("  • 4× memory reduction\n\n");
    
    printf("Day 5: 3D Parallelism\n");
    printf("  • Combine DP + TP + PP\n");
    printf("  • Scale to 1000s of GPUs\n\n");
    
    printf("When to Use What:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Model fits in GPU:          DDP                                   ║\n");
    printf("║ Model too large:            FSDP or ZeRO                          ║\n");
    printf("║ Very large (100B+):         3D parallelism                        ║\n");
    printf("║ Inference large model:      Tensor parallel                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
