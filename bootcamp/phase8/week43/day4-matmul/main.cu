/**
 * Week 43, Day 4: MatMul in Triton
 */
#include <cstdio>

int main() {
    printf("Week 43 Day 4: MatMul in Triton\n\n");
    
    printf("Tiled MatMul:\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,\n");
    printf("                  stride_am, stride_ak, stride_bk, stride_bn,\n");
    printf("                  stride_cm, stride_cn,\n");
    printf("                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,\n");
    printf("                  BLOCK_K: tl.constexpr):\n");
    printf("    pid = tl.program_id(0)\n");
    printf("    num_pid_n = tl.cdiv(N, BLOCK_N)\n");
    printf("    pid_m = pid // num_pid_n\n");
    printf("    pid_n = pid %% num_pid_n\n");
    printf("    \n");
    printf("    # Block starting positions\n");
    printf("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n");
    printf("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n");
    printf("    offs_k = tl.arange(0, BLOCK_K)\n");
    printf("    \n");
    printf("    # Pointers for first tiles\n");
    printf("    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n");
    printf("    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n");
    printf("    \n");
    printf("    # Accumulator\n");
    printf("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n");
    printf("    \n");
    printf("    # K-dimension loop\n");
    printf("    for k in range(0, K, BLOCK_K):\n");
    printf("        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)\n");
    printf("        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)\n");
    printf("        acc += tl.dot(a, b)  # Uses tensor cores if available!\n");
    printf("        a_ptrs += BLOCK_K * stride_ak\n");
    printf("        b_ptrs += BLOCK_K * stride_bk\n");
    printf("    \n");
    printf("    # Store result\n");
    printf("    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn\n");
    printf("    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))\n");
    printf("```\n\n");
    
    printf("Performance Notes:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • tl.dot() automatically uses Tensor Cores on Ampere+             ║\n");
    printf("║ • Typical: 80-90%% of cuBLAS for standard shapes                   ║\n");
    printf("║ • Advantage: Easy to add epilogues (bias, activation)             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
