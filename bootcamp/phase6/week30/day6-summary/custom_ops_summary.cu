/**
 * Week 30, Day 6: Custom Operators Summary
 */
#include <cstdio>

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 30 Summary: Custom CUDA Operators               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Key Techniques:\n");
    printf("  1. PyTorch extensions via torch.utils.cpp_extension\n");
    printf("  2. Custom autograd functions for backward pass\n");
    printf("  3. Fused operations for memory efficiency\n");
    printf("  4. TensorFlow custom ops via REGISTER_OP\n\n");
    
    printf("When to Write Custom Ops:\n");
    printf("  ✓ Operation not in framework\n");
    printf("  ✓ Fusion opportunity (multiple ops → one kernel)\n");
    printf("  ✓ Specialized hardware features (Tensor Cores)\n");
    printf("  ✓ Memory-bound operations with fusion potential\n\n");
    
    printf("Performance Tips:\n");
    printf("  - Fuse elementwise ops (LayerNorm+GELU)\n");
    printf("  - Minimize global memory round-trips\n");
    printf("  - Use shared memory for reductions\n");
    printf("  - Consider occupancy vs register usage\n\n");
    
    printf("Common Fusions in Transformers:\n");
    printf("  - LayerNorm + Activation\n");
    printf("  - Attention (QKV projection + softmax + V)\n");
    printf("  - Bias + Activation + Dropout\n");
    printf("  - GEMM + Bias + Activation (epilogue fusion)\n\n");
    
    printf("Next: Week 31 - Inference Optimization (TensorRT)\n");
    
    return 0;
}
