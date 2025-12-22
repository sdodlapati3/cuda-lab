/**
 * Week 42, Day 5: Memory Optimization
 */
#include <cstdio>

int main() {
    printf("Week 42 Day 5: Memory Optimization for Training\n\n");
    
    printf("Memory Trade-offs:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Standard: Save activations for backward (high memory)             ║\n");
    printf("║ Recomputation: Recompute in backward (lower memory, more compute) ║\n");
    printf("║ Checkpointing: Hybrid approach (save some, recompute some)        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("torch.utils.checkpoint:\n");
    printf("```python\n");
    printf("from torch.utils.checkpoint import checkpoint\n");
    printf("\n");
    printf("class TransformerBlock(nn.Module):\n");
    printf("    def forward(self, x):\n");
    printf("        # Checkpoint: don't save attention activations\n");
    printf("        x = x + checkpoint(self.attention, x)\n");
    printf("        x = x + checkpoint(self.ffn, x)\n");
    printf("        return x\n");
    printf("```\n\n");
    
    printf("Custom Recomputation (like FlashAttention):\n");
    printf("```python\n");
    printf("class FlashAttentionFunc(torch.autograd.Function):\n");
    printf("    @staticmethod\n");
    printf("    def forward(ctx, q, k, v):\n");
    printf("        # Only save q, k, v and logsumexp\n");
    printf("        # Don't save attention matrix (N² memory)\n");
    printf("        ctx.save_for_backward(q, k, v, logsumexp)\n");
    printf("        return output\n");
    printf("    \n");
    printf("    @staticmethod\n");
    printf("    def backward(ctx, grad_output):\n");
    printf("        # Recompute attention on-the-fly\n");
    printf("        # Memory: O(N) instead of O(N²)\n");
    printf("        ...\n");
    printf("```\n\n");
    
    printf("Memory Savings:\n");
    printf("┌─────────────────────┬───────────────┬───────────────┐\n");
    printf("│ Approach            │ Memory        │ Compute       │\n");
    printf("├─────────────────────┼───────────────┼───────────────┤\n");
    printf("│ Standard            │ O(N²)         │ 1×            │\n");
    printf("│ Checkpointing       │ O(N)          │ ~1.3×         │\n");
    printf("│ FlashAttention      │ O(N)          │ 1× (IO-aware) │\n");
    printf("└─────────────────────┴───────────────┴───────────────┘\n");
    
    return 0;
}
