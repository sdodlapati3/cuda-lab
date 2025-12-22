/**
 * Week 38, Day 4: Framework Integration
 */
#include <cstdio>

int main() {
    printf("Week 38 Day 4: Framework Integration\n\n");
    
    printf("Using FlashAttention in PyTorch:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ # PyTorch 2.0+ native support                                   │\n");
    printf("│ import torch.nn.functional as F                                 │\n");
    printf("│                                                                 │\n");
    printf("│ # Automatic backend selection                                   │\n");
    printf("│ out = F.scaled_dot_product_attention(                           │\n");
    printf("│     query, key, value,                                          │\n");
    printf("│     attn_mask=None,                                             │\n");
    printf("│     dropout_p=0.0,                                              │\n");
    printf("│     is_causal=True  # Use causal attention                      │\n");
    printf("│ )                                                               │\n");
    printf("│                                                                 │\n");
    printf("│ # Force specific backend                                        │\n");
    printf("│ with torch.backends.cuda.sdp_kernel(                            │\n");
    printf("│     enable_flash=True,                                          │\n");
    printf("│     enable_math=False,                                          │\n");
    printf("│     enable_mem_efficient=False                                  │\n");
    printf("│ ):                                                              │\n");
    printf("│     out = F.scaled_dot_product_attention(q, k, v)               │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Using flash-attn library directly:\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ from flash_attn import flash_attn_func                          │\n");
    printf("│                                                                 │\n");
    printf("│ # Q, K, V shape: [batch, seqlen, heads, head_dim]               │\n");
    printf("│ out = flash_attn_func(q, k, v, causal=True)                     │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
