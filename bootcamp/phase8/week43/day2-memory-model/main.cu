/**
 * Week 43, Day 2: Triton Memory Model
 */
#include <cstdio>

int main() {
    printf("Week 43 Day 2: Triton Memory Model\n\n");
    
    printf("Triton vs CUDA Memory:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ CUDA: Explicit thread → warp → block → grid hierarchy             ║\n");
    printf("║ Triton: You think in blocks, compiler handles threads             ║\n");
    printf("║                                                                   ║\n");
    printf("║ CUDA: Manual shared memory allocation and synchronization         ║\n");
    printf("║ Triton: Automatic (compiler decides what goes in SRAM)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Block Pointers (Triton 2.0+):\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,\n");
    printf("                  stride_am, stride_ak,\n");
    printf("                  stride_bk, stride_bn,\n");
    printf("                  stride_cm, stride_cn,\n");
    printf("                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,\n");
    printf("                  BLOCK_K: tl.constexpr):\n");
    printf("    # 2D block indexing\n");
    printf("    pid_m = tl.program_id(0)\n");
    printf("    pid_n = tl.program_id(1)\n");
    printf("    \n");
    printf("    # Create block pointers\n");
    printf("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n");
    printf("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n");
    printf("    offs_k = tl.arange(0, BLOCK_K)\n");
    printf("    \n");
    printf("    # A: [BLOCK_M, BLOCK_K], B: [BLOCK_K, BLOCK_N]\n");
    printf("    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n");
    printf("    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n");
    printf("```\n\n");
    
    printf("Automatic Optimizations:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ • Memory coalescing (reorders loads for efficiency)                 │\n");
    printf("│ • Shared memory tiling (caches frequently accessed data)            │\n");
    printf("│ • Software pipelining (overlaps loads with compute)                 │\n");
    printf("│ • Register allocation (minimizes spilling)                          │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    
    return 0;
}
