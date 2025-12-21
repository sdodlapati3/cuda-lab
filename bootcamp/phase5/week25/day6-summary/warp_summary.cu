/**
 * Week 25, Day 6: Warp-Level Summary
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 25 Summary: Warp-Level Optimizations            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Techniques Covered:\n");
    printf("  Day 1: Warp tiling - organizing work by warps\n");
    printf("  Day 2: Warp shuffle - register-to-register comm\n");
    printf("  Day 3: Shuffle GEMM - broadcast A values\n");
    printf("  Day 4: Warp reduction - parallel sums\n");
    printf("  Day 5: Warp specialization - load vs compute\n\n");
    
    printf("Key Benefits:\n");
    printf("  - No __syncthreads() needed within warp\n");
    printf("  - Fast register-to-register communication\n");
    printf("  - Reduced shared memory pressure\n");
    printf("  - Foundation for Tensor Core programming\n\n");
    
    printf("Next: Week 26 - Tensor Core Basics (WMMA API)\n");
    
    return 0;
}
