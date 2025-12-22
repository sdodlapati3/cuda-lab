/**
 * Week 36, Day 4: Roofline Model for Attention
 * 
 * Arithmetic Intensity = FLOPs / Bytes
 * Determines if kernel is compute or memory bound.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

struct RooflineAnalysis {
    double flops;
    double bytes;
    double arithmetic_intensity;
    double achieved_tflops;
    bool memory_bound;
};

// A100 specs
const double A100_PEAK_TFLOPS = 19.5;  // FP32
const double A100_HBM_BW_TBS = 2.0;    // TB/s

RooflineAnalysis analyzeKernel(double flops, double bytes, double time_ms) {
    RooflineAnalysis r;
    r.flops = flops;
    r.bytes = bytes;
    r.arithmetic_intensity = flops / bytes;
    r.achieved_tflops = (flops / 1e12) / (time_ms / 1000.0);
    
    // Ridge point: where compute and memory rooflines meet
    double ridge_point = A100_PEAK_TFLOPS / A100_HBM_BW_TBS;  // ~10 FLOP/Byte
    r.memory_bound = r.arithmetic_intensity < ridge_point;
    
    return r;
}

void printRoofline(const char* name, RooflineAnalysis& r) {
    double memory_roof = r.arithmetic_intensity * A100_HBM_BW_TBS;
    double effective_roof = fmin(A100_PEAK_TFLOPS, memory_roof);
    
    printf("%s:\n", name);
    printf("  FLOPs: %.2e\n", r.flops);
    printf("  Bytes: %.2e\n", r.bytes);
    printf("  Arithmetic Intensity: %.2f FLOP/Byte\n", r.arithmetic_intensity);
    printf("  Achieved: %.2f TFLOPS\n", r.achieved_tflops);
    printf("  Effective Roof: %.2f TFLOPS\n", effective_roof);
    printf("  Efficiency: %.1f%%\n", 100.0 * r.achieved_tflops / effective_roof);
    printf("  Status: %s\n\n", r.memory_bound ? "🔴 MEMORY BOUND" : "🟢 COMPUTE BOUND");
}

int main() {
    printf("Week 36 Day 4: Roofline Model\n\n");
    
    printf("A100 GPU Specs:\n");
    printf("  Peak Compute: %.1f TFLOPS (FP32)\n", A100_PEAK_TFLOPS);
    printf("  HBM Bandwidth: %.1f TB/s\n", A100_HBM_BW_TBS);
    printf("  Ridge Point: %.1f FLOP/Byte\n\n", A100_PEAK_TFLOPS / A100_HBM_BW_TBS);
    
    printf("Roofline Model:\n");
    printf("  ┌─────────────────────────────────────────────────────┐\n");
    printf("  │                                                     │\n");
    printf("  │  TFLOPS │      _______________Compute Roof          │\n");
    printf("  │         │     /                                     │\n");
    printf("  │         │    /                                      │\n");
    printf("  │         │   / Memory Roof                           │\n");
    printf("  │         │  /                                        │\n");
    printf("  │         │ /        * Kernel                         │\n");
    printf("  │         │/                                          │\n");
    printf("  │         └───────────────────────────────────────    │\n");
    printf("  │              Arithmetic Intensity (FLOP/Byte)       │\n");
    printf("  └─────────────────────────────────────────────────────┘\n\n");
    
    // Example: GEMM vs Attention
    int M = 4096, N = 4096, K = 4096;
    
    // GEMM: 2MNK FLOPs, (M*K + K*N + M*N) * 4 bytes
    double gemm_flops = 2.0 * M * N * K;
    double gemm_bytes = (M*K + K*N + M*N) * 4.0;
    
    printf("Example Kernels:\n\n");
    
    printf("1. GEMM (%dx%dx%d):\n", M, N, K);
    printf("   FLOPs: %.2e\n", gemm_flops);
    printf("   Bytes: %.2e\n", gemm_bytes);
    printf("   AI: %.1f FLOP/Byte → COMPUTE BOUND ✓\n\n", gemm_flops/gemm_bytes);
    
    // Standard Attention: low AI due to reading S matrix
    int seq = 2048, d = 64, bh = 48;
    double attn_flops = bh * (2.0 * seq * seq * d + seq * seq + 2.0 * seq * seq * d);
    double attn_bytes = bh * (3.0 * seq * d + 3.0 * seq * seq + seq * d) * 4.0;
    
    printf("2. Standard Attention (seq=%d, d=%d):\n", seq, d);
    printf("   FLOPs: %.2e\n", attn_flops);
    printf("   Bytes: %.2e (dominated by seq² matrix)\n", attn_bytes);
    printf("   AI: %.1f FLOP/Byte → MEMORY BOUND ✗\n\n", attn_flops/attn_bytes);
    
    // FlashAttention: higher AI
    double flash_flops = attn_flops;  // Same compute
    double flash_bytes = bh * (3.0 * seq * d + seq * d) * 4.0;  // No seq² term!
    
    printf("3. FlashAttention (seq=%d, d=%d):\n", seq, d);
    printf("   FLOPs: %.2e\n", flash_flops);
    printf("   Bytes: %.2e (no seq² storage)\n", flash_bytes);
    printf("   AI: %.1f FLOP/Byte → COMPUTE BOUND ✓\n\n", flash_flops/flash_bytes);
    
    printf("Key Takeaway:\n");
    printf("  FlashAttention transforms attention from\n");
    printf("  memory-bound to compute-bound by eliminating\n");
    printf("  the O(N²) intermediate storage.\n\n");
    
    printf("  Standard AI: ~%.0f FLOP/Byte (below ridge)\n", attn_flops/attn_bytes);
    printf("  Flash AI:    ~%.0f FLOP/Byte (above ridge)\n", flash_flops/flash_bytes);
    printf("  Ridge Point: ~%.0f FLOP/Byte\n", A100_PEAK_TFLOPS / A100_HBM_BW_TBS);
    
    return 0;
}
