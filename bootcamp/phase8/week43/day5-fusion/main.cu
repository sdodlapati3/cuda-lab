/**
 * Week 43, Day 5: Kernel Fusion in Triton
 */
#include <cstdio>

int main() {
    printf("Week 43 Day 5: Kernel Fusion in Triton\n\n");
    
    printf("Why Triton Excels at Fusion:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Easy to add operations in the same kernel                       ║\n");
    printf("║ • Compiler handles register allocation                            ║\n");
    printf("║ • No manual shared memory management                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Fused Bias + ReLU + Dropout:\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def fused_bias_relu_dropout_kernel(\n");
    printf("    x_ptr, bias_ptr, out_ptr, n_rows, n_cols, p,\n");
    printf("    seed, BLOCK_SIZE: tl.constexpr\n");
    printf("):\n");
    printf("    row_idx = tl.program_id(0)\n");
    printf("    col_offsets = tl.arange(0, BLOCK_SIZE)\n");
    printf("    mask = col_offsets < n_cols\n");
    printf("    \n");
    printf("    # Load\n");
    printf("    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask)\n");
    printf("    bias = tl.load(bias_ptr + col_offsets, mask=mask)\n");
    printf("    \n");
    printf("    # Fused: bias + relu + dropout\n");
    printf("    x = x + bias\n");
    printf("    x = tl.where(x > 0, x, 0.0)  # ReLU\n");
    printf("    \n");
    printf("    # Dropout\n");
    printf("    random = tl.rand(seed, row_idx * n_cols + col_offsets)\n");
    printf("    keep_mask = random > p\n");
    printf("    x = tl.where(keep_mask, x / (1 - p), 0.0)\n");
    printf("    \n");
    printf("    tl.store(out_ptr + row_idx * n_cols + col_offsets, x, mask=mask)\n");
    printf("```\n\n");
    
    printf("Fused LayerNorm:\n");
    printf("```python\n");
    printf("@triton.jit\n");
    printf("def layernorm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr,\n");
    printf("                     n_cols, eps, BLOCK_SIZE: tl.constexpr):\n");
    printf("    row = tl.program_id(0)\n");
    printf("    cols = tl.arange(0, BLOCK_SIZE)\n");
    printf("    mask = cols < n_cols\n");
    printf("    \n");
    printf("    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)\n");
    printf("    \n");
    printf("    # Mean and variance in one pass\n");
    printf("    mean = tl.sum(x, axis=0) / n_cols\n");
    printf("    var = tl.sum((x - mean) ** 2, axis=0) / n_cols\n");
    printf("    rstd = 1.0 / tl.sqrt(var + eps)\n");
    printf("    \n");
    printf("    # Normalize\n");
    printf("    gamma = tl.load(gamma_ptr + cols, mask=mask)\n");
    printf("    beta = tl.load(beta_ptr + cols, mask=mask)\n");
    printf("    y = gamma * (x - mean) * rstd + beta\n");
    printf("    \n");
    printf("    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)\n");
    printf("```\n");
    
    return 0;
}
