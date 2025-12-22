/**
 * Week 38, Day 2: FlashAttention-2 Improvements
 */
#include <cstdio>

int main() {
    printf("Week 38 Day 2: FlashAttention-2\n\n");
    
    printf("FlashAttention-2 Improvements over v1:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. BETTER PARALLELIZATION                                         ║\n");
    printf("║    v1: Parallelize over batch × heads                             ║\n");
    printf("║    v2: Also parallelize over sequence dimension                   ║\n");
    printf("║    → Better GPU utilization for long sequences                    ║\n");
    printf("║                                                                   ║\n");
    printf("║ 2. REDUCED NON-MATMUL FLOPS                                       ║\n");
    printf("║    v1: Online softmax updates mixed with matmul                   ║\n");
    printf("║    v2: Restructured to minimize non-tensor-core ops               ║\n");
    printf("║    → Better utilization of tensor cores                           ║\n");
    printf("║                                                                   ║\n");
    printf("║ 3. LOOP ORDER CHANGE                                              ║\n");
    printf("║    v1: Outer loop over KV, inner loop over Q                      ║\n");
    printf("║    v2: Outer loop over Q (parallelized), inner over KV            ║\n");
    printf("║    → Each thread block handles one Q tile completely              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Speedup (A100, causal):\n");
    printf("┌─────────────┬───────────┬───────────┬───────────┐\n");
    printf("│ Seq Length  │ Standard  │ Flash v1  │ Flash v2  │\n");
    printf("├─────────────┼───────────┼───────────┼───────────┤\n");
    printf("│ 512         │ 1.0×      │ 2.1×      │ 2.8×      │\n");
    printf("│ 1024        │ 1.0×      │ 2.4×      │ 3.2×      │\n");
    printf("│ 2048        │ 1.0×      │ 2.7×      │ 3.8×      │\n");
    printf("│ 4096        │ OOM       │ 3.0×      │ 4.2×      │\n");
    printf("│ 8192        │ OOM       │ 3.4×      │ 4.8×      │\n");
    printf("└─────────────┴───────────┴───────────┴───────────┘\n\n");
    
    printf("Usage in Practice:\n");
    printf("  PyTorch: torch.nn.functional.scaled_dot_product_attention()\n");
    printf("  xFormers: xformers.ops.memory_efficient_attention()\n");
    printf("  Direct:   flash_attn.flash_attn_func()\n");
    
    return 0;
}
