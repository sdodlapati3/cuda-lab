/**
 * Week 42, Day 2: Forward and Backward Kernels
 */
#include <cstdio>

int main() {
    printf("Week 42 Day 2: Forward and Backward Kernels\n\n");
    
    printf("Example: Fused LayerNorm Forward + Backward\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Forward Pass:\n");
    printf("```cpp\n");
    printf("// layernorm_forward_cuda(x, gamma, beta) -> (y, mean, rstd)\n");
    printf("__global__ void layernorm_forward_kernel(\n");
    printf("    const float* x, const float* gamma, const float* beta,\n");
    printf("    float* y, float* mean, float* rstd,\n");
    printf("    int N, int D\n");
    printf(") {\n");
    printf("    // Each block handles one row\n");
    printf("    int row = blockIdx.x;\n");
    printf("    // ... compute mean, variance, normalize\n");
    printf("    // Save mean and rstd for backward pass!\n");
    printf("}\n");
    printf("```\n\n");
    
    printf("Backward Pass:\n");
    printf("```cpp\n");
    printf("// layernorm_backward_cuda(dy, x, mean, rstd, gamma) -> (dx, dgamma, dbeta)\n");
    printf("__global__ void layernorm_backward_kernel(\n");
    printf("    const float* dy, const float* x,\n");
    printf("    const float* mean, const float* rstd, const float* gamma,\n");
    printf("    float* dx, float* dgamma, float* dbeta,\n");
    printf("    int N, int D\n");
    printf(") {\n");
    printf("    // Gradient w.r.t. input: complex due to normalization\n");
    printf("    // dx = gamma * rstd * (dy - mean(dy) - x_hat * mean(dy * x_hat))\n");
    printf("}\n");
    printf("```\n\n");
    
    printf("Key Insight:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Forward saves intermediates (mean, rstd) that backward needs.     ║\n");
    printf("║ Trade-off: memory vs recomputation (like FlashAttention)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
