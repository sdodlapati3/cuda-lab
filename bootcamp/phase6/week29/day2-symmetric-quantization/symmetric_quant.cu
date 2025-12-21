/**
 * Week 29, Day 2: Symmetric Quantization
 * Per-tensor and per-channel symmetric quantization.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Per-tensor symmetric: one scale for entire tensor
__global__ void perTensorQuantize(const float* input, int8_t* output, 
                                   float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t q = __float2int_rn(input[idx] / scale);
        output[idx] = static_cast<int8_t>(max(-128, min(127, q)));
    }
}

// Per-channel symmetric: different scale per output channel
__global__ void perChannelQuantize(const float* input, int8_t* output,
                                    const float* scales, int channels, int hw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * hw;
    if (idx < total) {
        int c = idx / hw;  // Channel index
        float scale = scales[c];
        int32_t q = __float2int_rn(input[idx] / scale);
        output[idx] = static_cast<int8_t>(max(-128, min(127, q)));
    }
}

int main() {
    printf("Week 29 Day 2: Symmetric Quantization\n\n");
    
    printf("Symmetric Quantization Properties:\n");
    printf("  - Zero point = 0 (centered at origin)\n");
    printf("  - Range: [-127, 127] × scale\n");
    printf("  - Best for: Weights (typically centered)\n\n");
    
    printf("Per-Tensor vs Per-Channel:\n");
    printf("┌─────────────────┬──────────────────┬───────────────────┐\n");
    printf("│ Method          │ Scales           │ Accuracy          │\n");
    printf("├─────────────────┼──────────────────┼───────────────────┤\n");
    printf("│ Per-tensor      │ 1 per tensor     │ Lower             │\n");
    printf("│ Per-channel     │ 1 per channel    │ Higher            │\n");
    printf("│ Per-group       │ Groups of values │ Highest           │\n");
    printf("└─────────────────┴──────────────────┴───────────────────┘\n\n");
    
    // Demo: Compare per-tensor vs per-channel
    const int C = 4, HW = 256;  // 4 channels, 256 elements each
    const int N = C * HW;
    
    float* h_weights = new float[N];
    float* h_scales = new float[C];
    
    // Different distributions per channel
    for (int c = 0; c < C; c++) {
        float range = 1.0f + c * 0.5f;  // Different range per channel
        for (int i = 0; i < HW; i++) {
            h_weights[c * HW + i] = range * (2.0f * rand() / RAND_MAX - 1.0f);
        }
        h_scales[c] = range / 127.0f;
    }
    
    // Per-tensor scale (max across all)
    float globalMax = 0.0f;
    for (int i = 0; i < N; i++) {
        globalMax = fmaxf(globalMax, fabsf(h_weights[i]));
    }
    float perTensorScale = globalMax / 127.0f;
    
    printf("Scale Comparison:\n");
    printf("  Per-tensor scale: %.6f\n", perTensorScale);
    for (int c = 0; c < C; c++) {
        printf("  Channel %d scale: %.6f\n", c, h_scales[c]);
    }
    
    printf("\nPer-channel allows finer granularity for diverse distributions.\n");
    
    delete[] h_weights;
    delete[] h_scales;
    
    return 0;
}
