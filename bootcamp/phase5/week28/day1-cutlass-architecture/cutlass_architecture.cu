/**
 * Week 28, Day 1: CUTLASS Architecture
 * Understanding the template structure.
 */
#include <cstdio>

/*
 * CUTLASS Hierarchy (conceptual):
 *
 * Device Level:
 *   cutlass::gemm::device::Gemm<...>
 *   - Orchestrates grid launch
 *   - Manages problem size
 *
 * Kernel Level:
 *   cutlass::gemm::kernel::Gemm<...>
 *   - Block-level coordination
 *   - Shared memory management
 *
 * Threadblock Level:
 *   cutlass::gemm::threadblock::Mma<...>
 *   - Tile iterators
 *   - Warp-level MMAs
 *
 * Warp Level:
 *   cutlass::gemm::warp::MmaTensorOp<...>
 *   - WMMA/MMA operations
 *   - Fragment management
 *
 * Instruction Level:
 *   PTX mma.sync instructions
 */

int main() {
    printf("Week 28 Day 1: CUTLASS Architecture\n\n");
    
    printf("CUTLASS Template Hierarchy:\n");
    printf("┌─────────────────────────────────────────────────────┐\n");
    printf("│ Device:     cutlass::gemm::device::Gemm             │\n");
    printf("│   └─ Manages problem size, grid launch              │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│ Kernel:     cutlass::gemm::kernel::Gemm             │\n");
    printf("│   └─ Block coordination, shared memory              │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│ Threadblock: cutlass::gemm::threadblock::Mma        │\n");
    printf("│   └─ Tile iterators, warp coordination              │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│ Warp:       cutlass::gemm::warp::MmaTensorOp        │\n");
    printf("│   └─ WMMA operations, fragment management           │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│ Instruction: PTX mma.sync                           │\n");
    printf("│   └─ Hardware Tensor Core operations                │\n");
    printf("└─────────────────────────────────────────────────────┘\n\n");
    
    printf("Key CUTLASS Types:\n");
    printf("  - Element types: cutlass::half_t, cutlass::bfloat16_t\n");
    printf("  - Layouts: cutlass::layout::RowMajor, ColumnMajor\n");
    printf("  - Tile shapes: cutlass::gemm::GemmShape<M, N, K>\n");
    printf("  - Epilogues: LinearCombination, ReLU, GELU, etc.\n\n");
    
    printf("Installation:\n");
    printf("  git clone https://github.com/NVIDIA/cutlass.git\n");
    printf("  # Header-only library, just include the headers\n");
    
    return 0;
}
