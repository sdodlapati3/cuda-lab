/**
 * Week 28, Day 5: Mixed Precision CUTLASS
 * Different precision configurations.
 */
#include <cstdio>

int main() {
    printf("Week 28 Day 5: Mixed Precision CUTLASS\n\n");
    
    printf("A100 Tensor Core Precisions:\n");
    printf("┌───────────┬────────────┬────────────┬──────────────┐\n");
    printf("│ Type      │ Input      │ Accumulate │ Peak TFLOPS  │\n");
    printf("├───────────┼────────────┼────────────┼──────────────┤\n");
    printf("│ FP16      │ FP16       │ FP16/FP32  │ 312          │\n");
    printf("│ BF16      │ BF16       │ FP32       │ 312          │\n");
    printf("│ TF32      │ FP32*      │ FP32       │ 156          │\n");
    printf("│ FP64      │ FP64       │ FP64       │ 19.5         │\n");
    printf("│ INT8      │ INT8       │ INT32      │ 624          │\n");
    printf("│ INT4      │ INT4       │ INT32      │ 1248         │\n");
    printf("└───────────┴────────────┴────────────┴──────────────┘\n");
    printf("* TF32 uses FP32 input with truncated mantissa\n\n");
    
    printf("CUTLASS Type Configuration:\n");
    printf("  FP16:  cutlass::half_t\n");
    printf("  BF16:  cutlass::bfloat16_t\n");
    printf("  TF32:  cutlass::tfloat32_t\n");
    printf("  FP64:  double\n");
    printf("  INT8:  int8_t\n\n");
    
    printf("Precision Comparison (Same Workload):\n");
    printf("┌───────────┬──────────────┬──────────────┐\n");
    printf("│ Precision │ Relative Perf│ Accuracy     │\n");
    printf("├───────────┼──────────────┼──────────────┤\n");
    printf("│ FP64      │ 1.0×         │ ~15 digits   │\n");
    printf("│ FP32      │ 2.0×         │ ~7 digits    │\n");
    printf("│ TF32      │ 8.0×         │ ~3 digits    │\n");
    printf("│ FP16/BF16 │ 16.0×        │ ~3 digits    │\n");
    printf("│ INT8      │ 32.0×        │ Quantized    │\n");
    printf("└───────────┴──────────────┴──────────────┘\n\n");
    
    printf("Use Cases:\n");
    printf("  FP16:  Deep learning training/inference\n");
    printf("  BF16:  Training (larger dynamic range)\n");
    printf("  TF32:  Drop-in FP32 replacement for DL\n");
    printf("  INT8:  Inference acceleration\n");
    
    return 0;
}
