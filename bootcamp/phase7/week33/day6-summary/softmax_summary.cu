/**
 * Week 33, Day 6: Softmax Summary
 */
#include <cstdio>

int main() {
    printf("Week 33 Summary: Softmax Fundamentals\n\n");
    
    printf("Day 1: Naive Softmax\n");
    printf("  - exp(x) overflows for x > 88 (FP32)\n");
    printf("  - Direct implementation causes NaN\n\n");
    
    printf("Day 2: Stable Softmax\n");
    printf("  - Subtract max before exp\n");
    printf("  - Three passes: max → exp+sum → normalize\n\n");
    
    printf("Day 3: Online Softmax\n");
    printf("  - Compute max and sum in single pass\n");
    printf("  - Key: rescale sum when max changes\n");
    printf("  - Foundation for FlashAttention\n\n");
    
    printf("Day 4: Warp Softmax\n");
    printf("  - For seq ≤ 32: shuffle-only\n");
    printf("  - No shared memory needed\n");
    printf("  - Perfect for attention heads\n\n");
    
    printf("Day 5: Block Softmax\n");
    printf("  - Block-level reduction with shared memory\n");
    printf("  - Handles arbitrary sequence lengths\n");
    printf("  - Fused version reduces memory traffic\n\n");
    
    printf("Softmax Implementation Comparison:\n");
    printf("┌─────────────────┬─────────────┬────────────────────┐\n");
    printf("│ Approach        │ Passes      │ Best For           │\n");
    printf("├─────────────────┼─────────────┼────────────────────┤\n");
    printf("│ Naive           │ 2           │ Never (overflows!) │\n");
    printf("│ Stable          │ 3           │ Simple cases       │\n");
    printf("│ Online          │ 2           │ Memory-bound cases │\n");
    printf("│ Warp            │ 2           │ seq ≤ 32           │\n");
    printf("│ Block           │ 2-3         │ seq > 32           │\n");
    printf("└─────────────────┴─────────────┴────────────────────┘\n\n");
    
    printf("Memory Traffic Analysis (N elements):\n");
    printf("  Three-pass: 3×N reads + 2×N writes = 5N\n");
    printf("  Two-pass:   2×N reads + N writes = 3N\n");
    printf("  Savings:    ~40%% reduction\n\n");
    
    printf("Production Notes:\n");
    printf("  - cuDNN softmax is highly optimized\n");
    printf("  - Custom softmax mainly for fusion\n");
    printf("  - Online softmax essential for FlashAttention\n\n");
    
    printf("Next Week: LayerNorm (similar patterns!)\n");
    
    return 0;
}
