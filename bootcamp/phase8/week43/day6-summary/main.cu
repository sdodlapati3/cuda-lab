/**
 * Week 43, Day 6: Week Summary
 */
#include <cstdio>

int main() {
    printf("Week 43 Summary: Triton Programming\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Day 1: Triton Basics\n");
    printf("  • @triton.jit, tl.program_id, tl.arange\n");
    printf("  • Block-level programming model\n\n");
    
    printf("Day 2: Memory Model\n");
    printf("  • Automatic coalescing and shared memory\n");
    printf("  • Block pointers for multi-dimensional access\n\n");
    
    printf("Day 3: Softmax\n");
    printf("  • tl.max, tl.sum for reductions\n");
    printf("  • ~20 lines vs ~100 in CUDA\n\n");
    
    printf("Day 4: MatMul\n");
    printf("  • tl.dot for tensor core operations\n");
    printf("  • 80-90%% of cuBLAS performance\n\n");
    
    printf("Day 5: Kernel Fusion\n");
    printf("  • Easy to add operations in same kernel\n");
    printf("  • Fused LayerNorm, BiasReluDropout examples\n\n");
    
    printf("When to Use Triton vs CUDA:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Triton:                                                           ║\n");
    printf("║   ✓ Rapid prototyping                                             ║\n");
    printf("║   ✓ Fused element-wise + reduction ops                            ║\n");
    printf("║   ✓ Need 80-90%% of peak quickly                                   ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║ CUDA:                                                             ║\n");
    printf("║   ✓ Need absolute peak performance                                ║\n");
    printf("║   ✓ Complex inter-thread communication                            ║\n");
    printf("║   ✓ Low-level hardware control                                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
