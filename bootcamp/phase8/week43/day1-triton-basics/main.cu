/**
 * Week 43, Day 1: Triton Basics
 */
#include <cstdio>

int main() {
    printf("Week 43 Day 1: Triton Programming Basics\n\n");
    
    printf("What is Triton?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Python DSL for GPU programming                                  ║\n");
    printf("║ • Block-level abstraction (not thread-level like CUDA)            ║\n");
    printf("║ • Automatic memory coalescing and shared memory management        ║\n");
    printf("║ • Compiles to PTX via MLIR                                        ║\n");
    printf("║ • Powers torch.compile's TorchInductor backend                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Hello World in Triton:\n");
    printf("```python\n");
    printf("import triton\n");
    printf("import triton.language as tl\n");
    printf("\n");
    printf("@triton.jit\n");
    printf("def add_kernel(x_ptr, y_ptr, out_ptr, n,\n");
    printf("               BLOCK_SIZE: tl.constexpr):\n");
    printf("    # Program ID = which block\n");
    printf("    pid = tl.program_id(0)\n");
    printf("    \n");
    printf("    # Compute offsets for this block\n");
    printf("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n");
    printf("    mask = offsets < n\n");
    printf("    \n");
    printf("    # Load, compute, store\n");
    printf("    x = tl.load(x_ptr + offsets, mask=mask)\n");
    printf("    y = tl.load(y_ptr + offsets, mask=mask)\n");
    printf("    tl.store(out_ptr + offsets, x + y, mask=mask)\n");
    printf("\n");
    printf("# Launch\n");
    printf("grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n");
    printf("add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)\n");
    printf("```\n\n");
    
    printf("Key Concepts:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ tl.program_id(axis)  - Block index (like blockIdx)                  │\n");
    printf("│ tl.arange(0, N)      - Vector of indices [0, 1, ..., N-1]           │\n");
    printf("│ tl.load/store        - Memory access with automatic coalescing      │\n");
    printf("│ mask                 - Handles bounds checking                      │\n");
    printf("│ tl.constexpr         - Compile-time constant (like template param)  │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
