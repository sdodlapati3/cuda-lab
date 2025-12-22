/**
 * Week 39, Day 5: Fusion Trade-offs
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

int main() {
    printf("Week 39 Day 5: When NOT to Fuse\n\n");
    
    printf("Fusion Trade-offs:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ FUSE when:                                                        ║\n");
    printf("║   • Operations are element-wise                                   ║\n");
    printf("║   • Kernel is memory-bound                                        ║\n");
    printf("║   • Intermediate results don't need to be saved                   ║\n");
    printf("║   • Combined kernel still fits in registers                       ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║ DON'T FUSE when:                                                  ║\n");
    printf("║   • Individual kernels are compute-bound (MatMul)                 ║\n");
    printf("║   • Library kernels are already highly optimized                  ║\n");
    printf("║   • Fusion increases register pressure significantly              ║\n");
    printf("║   • Code complexity outweighs performance gain                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Example: MatMul + Bias + ReLU\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Option 1: cuBLAS GEMM + Custom BiasReLU                             │\n");
    printf("│   • GEMM: Highly optimized, near peak FLOPS                         │\n");
    printf("│   • BiasReLU: Simple fused kernel                                   │\n");
    printf("│   • Total: GEMM dominates, BiasReLU is fast                         │\n");
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ Option 2: Custom fused MatMulBiasReLU                               │\n");
    printf("│   • Your MatMul likely 2-5× slower than cuBLAS                      │\n");
    printf("│   • Not worth it unless you're an expert                            │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Practical Guidelines:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ 1. Profile first: Is the operation memory or compute bound?        │\n");
    printf("│ 2. Start with library: cuBLAS, cuDNN, CUTLASS                       │\n");
    printf("│ 3. Fuse element-wise ops following heavy compute                    │\n");
    printf("│ 4. Measure after fusion: Did it actually help?                      │\n");
    printf("│ 5. Consider maintenance: Is the complexity worth it?                │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Modern Approaches:\n");
    printf("  • Triton: Python DSL for custom fused kernels\n");
    printf("  • torch.compile: Automatic fusion via TorchInductor\n");
    printf("  • CUTLASS epilogues: Fuse bias/activation with GEMM\n");
    printf("  • XLA/TensorRT: Graph-level fusion optimization\n");
    
    return 0;
}
