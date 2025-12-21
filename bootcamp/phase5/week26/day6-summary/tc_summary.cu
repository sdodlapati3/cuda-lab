/**
 * Week 26, Day 6: Tensor Core Summary
 */
#include <cstdio>

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 26 Summary: Tensor Core Basics                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("WMMA API Essentials:\n");
    printf("  1. fragment<matrix_a/b/accumulator, M, N, K, type, layout>\n");
    printf("  2. fill_fragment(frag, value)\n");
    printf("  3. load_matrix_sync(frag, ptr, stride)\n");
    printf("  4. mma_sync(c, a, b, c) - D = A×B + C\n");
    printf("  5. store_matrix_sync(ptr, frag, stride, layout)\n\n");
    
    printf("Key Points:\n");
    printf("  - One warp executes one MMA operation\n");
    printf("  - 16×16×16 tile: 4096 FMAs in ~4 cycles\n");
    printf("  - FP16 input, FP32 accumulation for accuracy\n");
    printf("  - Layout matters for performance\n\n");
    
    printf("A100 Tensor Core Peak:\n");
    printf("  - FP16: 312 TFLOPS\n");
    printf("  - TF32: 156 TFLOPS\n");
    printf("  - FP64: 19.5 TFLOPS\n\n");
    
    printf("Next: Week 27 - Advanced Tensor Core Optimizations\n");
    
    return 0;
}
