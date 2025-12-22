/**
 * Week 40, Day 2: Triton Overview
 * 
 * This file explains Triton concepts. Actual Triton code is Python.
 */
#include <cstdio>

int main() {
    printf("Week 40 Day 2: Triton for Custom Kernels\n\n");
    
    printf("Triton: Python DSL for GPU Programming\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Write kernels in Python, compile to PTX                         ║\n");
    printf("║ • Block-level programming (not thread-level like CUDA)            ║\n");
    printf("║ • Automatic memory coalescing and shared memory management        ║\n");
    printf("║ • Used in PyTorch 2.0's torch.compile backend                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Triton vs CUDA Comparison:\n");
    printf("┌──────────────────────┬───────────────────┬───────────────────┐\n");
    printf("│ Aspect               │ CUDA              │ Triton            │\n");
    printf("├──────────────────────┼───────────────────┼───────────────────┤\n");
    printf("│ Language             │ C++               │ Python            │\n");
    printf("│ Abstraction          │ Thread            │ Block/Tile        │\n");
    printf("│ Shared Memory        │ Manual            │ Automatic         │\n");
    printf("│ Memory Coalescing    │ Manual            │ Automatic         │\n");
    printf("│ Performance Ceiling  │ Higher            │ Good (80-90%%)     │\n");
    printf("│ Development Speed    │ Slower            │ Much Faster       │\n");
    printf("└──────────────────────┴───────────────────┴───────────────────┘\n\n");
    
    printf("Example: Triton Fused Softmax\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):\n");
    printf("    row_idx = tl.program_id(0)\n");
    printf("    col_offsets = tl.arange(0, BLOCK_SIZE)\n");
    printf("    \n");
    printf("    # Load entire row (automatic memory coalescing)\n");
    printf("    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=col_offsets < n_cols)\n");
    printf("    \n");
    printf("    # Compute softmax (automatic shared memory for reductions)\n");
    printf("    row_max = tl.max(row, axis=0)\n");
    printf("    numerator = tl.exp(row - row_max)\n");
    printf("    denominator = tl.sum(numerator, axis=0)\n");
    printf("    softmax_out = numerator / denominator\n");
    printf("    \n");
    printf("    tl.store(output_ptr + row_idx * n_cols + col_offsets, softmax_out, mask=col_offsets < n_cols)\n");
    printf("```\n\n");
    
    printf("When to Use Triton:\n");
    printf("  ✓ Rapid prototyping of fused kernels\n");
    printf("  ✓ Element-wise and reduction operations\n");
    printf("  ✓ Need 80-90%% of peak performance quickly\n");
    printf("  ✗ Need absolute peak performance (use CUDA)\n");
    printf("  ✗ Complex inter-thread communication\n");
    
    return 0;
}
