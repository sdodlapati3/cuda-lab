/**
 * Week 26, Day 4: Layout Optimization
 */
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

int main() {
    printf("Week 26 Day 4: Layout Optimization\n\n");
    
    printf("WMMA Layout Options:\n");
    printf("  Matrix A: row_major or col_major\n");
    printf("  Matrix B: row_major or col_major\n");
    printf("  Output C: row_major via store\n\n");
    
    printf("Optimal for Coalescing:\n");
    printf("  A: row_major (threads read consecutive K elements)\n");
    printf("  B: row_major (threads read consecutive N elements)\n\n");
    
    printf("Fragment Layout (16x16x16):\n");
    printf("  a_frag: 16x16 half = 256 elements per warp\n");
    printf("  b_frag: 16x16 half = 256 elements per warp\n");
    printf("  c_frag: 16x16 float = 256 elements (8 per thread)\n");
    
    return 0;
}
