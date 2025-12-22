/**
 * Week 38, Day 3: Standard vs FlashAttention Comparison
 */
#include <cstdio>

int main() {
    printf("Week 38 Day 3: Implementation Comparison\n\n");
    
    printf("Standard Attention Implementation:\n");
    printf("  Kernel 1: QK^T matmul → write S to HBM\n");
    printf("  Kernel 2: Softmax → read S, write S\n");
    printf("  Kernel 3: PV matmul → read S, write O\n");
    printf("  Total: 3 kernels, 3× S memory traffic\n\n");
    
    printf("FlashAttention Implementation:\n");
    printf("  Single fused kernel:\n");
    printf("    - Tile loop over K, V\n");
    printf("    - Online softmax update\n");
    printf("    - Incremental output accumulation\n");
    printf("  Total: 1 kernel, 0× S memory traffic\n\n");
    
    printf("Memory Traffic (N=2048, d=64, batch×heads=48):\n");
    printf("┌────────────────────┬────────────────┬────────────────┐\n");
    printf("│ Component          │ Standard (MB)  │ Flash (MB)     │\n");
    printf("├────────────────────┼────────────────┼────────────────┤\n");
    printf("│ Q, K, V read       │ 24.0           │ 24.0           │\n");
    printf("│ S write            │ 768.0          │ 0.0            │\n");
    printf("│ S read (softmax)   │ 768.0          │ 0.0            │\n");
    printf("│ S write (softmax)  │ 768.0          │ 0.0            │\n");
    printf("│ S read (PV)        │ 768.0          │ 0.0            │\n");
    printf("│ O write            │ 24.0           │ 24.0           │\n");
    printf("├────────────────────┼────────────────┼────────────────┤\n");
    printf("│ TOTAL              │ 3120.0 MB      │ 48.0 MB        │\n");
    printf("│ Reduction          │ -              │ 65×            │\n");
    printf("└────────────────────┴────────────────┴────────────────┘\n");
    
    return 0;
}
