/**
 * Week 28, Day 3: Custom Epilogues
 * Fusing operations after GEMM for efficiency.
 */
#include <cstdio>

/*
 * CUTLASS Epilogue Options:
 *
 * 1. LinearCombination: D = alpha * A*B + beta * C
 *    cutlass::epilogue::thread::LinearCombination<...>
 *
 * 2. LinearCombinationRelu: D = ReLU(alpha * A*B + beta * C)
 *    cutlass::epilogue::thread::LinearCombinationRelu<...>
 *
 * 3. LinearCombinationGELU: D = GELU(alpha * A*B + beta * C)
 *    Custom GELU epilogue
 *
 * 4. LinearCombinationBias: D = alpha * A*B + beta * C + bias
 *    With per-column or per-row bias vector
 *
 * Benefits of fused epilogues:
 * - Avoid extra global memory round-trip
 * - Leverage register data before writing
 * - Single kernel instead of GEMM + element-wise
 */

int main() {
    printf("Week 28 Day 3: Custom Epilogues\n\n");
    
    printf("Standard Epilogue (Linear Combination):\n");
    printf("  D = alpha * (A × B) + beta * C\n\n");
    
    printf("Available Fused Epilogues:\n");
    printf("┌────────────────────────┬────────────────────────────────┐\n");
    printf("│ Epilogue               │ Operation                      │\n");
    printf("├────────────────────────┼────────────────────────────────┤\n");
    printf("│ LinearCombination      │ D = α×(A×B) + β×C              │\n");
    printf("│ LinearCombinationRelu  │ D = ReLU(α×(A×B) + β×C)        │\n");
    printf("│ LinearCombinationClamp │ D = clamp(α×(A×B) + β×C)       │\n");
    printf("│ LinearCombinationBias  │ D = α×(A×B) + β×C + bias       │\n");
    printf("│ Custom                 │ Any user-defined functor       │\n");
    printf("└────────────────────────┴────────────────────────────────┘\n\n");
    
    printf("Performance Impact:\n");
    printf("  Without fusion: GEMM kernel + ReLU kernel = 2 passes\n");
    printf("  With fusion:    GEMM+ReLU kernel = 1 pass\n");
    printf("  Speedup: ~20-30%% for memory-bound epilogues\n\n");
    
    printf("Custom Epilogue Template:\n");
    printf("  struct MyEpilogue {\n");
    printf("    __device__ OutputType operator()(AccumType accum) {\n");
    printf("      return my_function(accum);\n");
    printf("    }\n");
    printf("  };\n");
    
    return 0;
}
