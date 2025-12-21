/**
 * Week 27, Day 5: Tensor Core Occupancy Analysis
 */
#include <cuda_runtime.h>
#include <cstdio>

template<int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__(256, 2)  // 256 threads, min 2 blocks/SM
tcKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[STAGES][BK][BM];
    __shared__ float Bs[STAGES][BK][BN];
    // Kernel body...
}

int main() {
    printf("Week 27 Day 5: Tensor Core Occupancy Analysis\n\n");
    
    printf("A100 Resources:\n");
    printf("  Registers/SM: 65536\n");
    printf("  Shared mem/SM: 164 KB (configurable)\n");
    printf("  Max threads/SM: 2048\n");
    printf("  Max blocks/SM: 32\n\n");
    
    printf("Tensor Core GEMM Resource Usage:\n");
    printf("┌────────────────┬────────────┬────────────┬───────────┐\n");
    printf("│ Configuration  │ Shared Mem │ Registers  │ Occupancy │\n");
    printf("├────────────────┼────────────┼────────────┼───────────┤\n");
    printf("│ 64×64, 2-stage │   ~16 KB   │   ~64/thr  │    50%%    │\n");
    printf("│ 128×128, 2-stg │   ~64 KB   │   ~96/thr  │    25%%    │\n");
    printf("│ 128×128, 3-stg │   ~96 KB   │  ~128/thr  │    12%%    │\n");
    printf("│ 256×128, 3-stg │  ~144 KB   │  ~160/thr  │    12%%    │\n");
    printf("└────────────────┴────────────┴────────────┴───────────┘\n\n");
    
    printf("Tradeoffs:\n");
    printf("  - Larger tiles = more compute per load = better arithmetic intensity\n");
    printf("  - More stages = better latency hiding = more shared memory\n");
    printf("  - Lower occupancy OK if compute-bound\n\n");
    
    printf("Rule of thumb: For TC GEMM, 25-50%% occupancy often optimal\n");
    
    return 0;
}
