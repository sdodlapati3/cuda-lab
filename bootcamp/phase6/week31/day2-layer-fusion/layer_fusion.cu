/**
 * Week 31, Day 2: Layer Fusion
 * Understanding TensorRT automatic fusion.
 */
#include <cstdio>

int main() {
    printf("Week 31 Day 2: Layer Fusion\n\n");
    
    printf("TensorRT Automatic Fusion Patterns:\n\n");
    
    printf("1. Convolution Fusions:\n");
    printf("   Conv + BatchNorm + ReLU → Single kernel\n");
    printf("   Conv + Add + ReLU (residual) → Fused\n\n");
    
    printf("2. Pointwise Fusions:\n");
    printf("   Multiple elementwise ops → One kernel\n");
    printf("   Example: x * scale + bias → (GELU) → dropout\n\n");
    
    printf("3. GEMM Fusions:\n");
    printf("   MatMul + Bias + Activation → cuBLAS epilogue\n");
    printf("   Attention: Q×K^T + mask + softmax\n\n");
    
    printf("4. Transformer-specific:\n");
    printf("   Multi-Head Attention → fMHA kernel\n");
    printf("   LayerNorm → Single pass kernel\n\n");
    
    printf("Fusion Impact Example (ResNet-50):\n");
    printf("┌────────────────────┬───────────┬───────────┐\n");
    printf("│ Metric             │ Unfused   │ Fused     │\n");
    printf("├────────────────────┼───────────┼───────────┤\n");
    printf("│ Layers             │ 152       │ 57        │\n");
    printf("│ Kernels launched   │ ~200      │ ~60       │\n");
    printf("│ Memory traffic     │ High      │ 3× lower  │\n");
    printf("│ Latency            │ 5.2 ms    │ 1.8 ms    │\n");
    printf("└────────────────────┴───────────┴───────────┘\n\n");
    
    printf("Viewing Fusion with trtexec:\n");
    printf("  trtexec --onnx=model.onnx --verbose 2>&1 | grep Fusion\n");
    
    return 0;
}
