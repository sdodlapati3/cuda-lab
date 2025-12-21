/**
 * Week 29, Day 3: Asymmetric Quantization
 * Full range utilization with zero point.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Asymmetric: q = round(x / scale) + zero_point
// x = (q - zero_point) * scale
// scale = (max - min) / 255
// zero_point = round(-min / scale)

__global__ void asymmetricQuantize(const float* input, uint8_t* output,
                                    float scale, uint8_t zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scaled = input[idx] / scale + zero_point;
        int32_t q = __float2int_rn(scaled);
        output[idx] = static_cast<uint8_t>(max(0, min(255, q)));
    }
}

__global__ void asymmetricDequantize(const uint8_t* input, float* output,
                                      float scale, uint8_t zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (static_cast<float>(input[idx]) - zero_point) * scale;
    }
}

int main() {
    printf("Week 29 Day 3: Asymmetric Quantization\n\n");
    
    printf("Asymmetric vs Symmetric:\n");
    printf("┌─────────────────┬───────────────┬───────────────────┐\n");
    printf("│ Property        │ Symmetric     │ Asymmetric        │\n");
    printf("├─────────────────┼───────────────┼───────────────────┤\n");
    printf("│ Range           │ [-127,127]    │ [0, 255]          │\n");
    printf("│ Zero point      │ Always 0      │ Calculated        │\n");
    printf("│ Best for        │ Weights       │ Activations(ReLU) │\n");
    printf("│ Computation     │ Simpler       │ Extra subtract    │\n");
    printf("└─────────────────┴───────────────┴───────────────────┘\n\n");
    
    const int N = 1024;
    
    // Generate ReLU-like activations (non-negative)
    float* h_input = new float[N];
    float* h_output = new float[N];
    
    float minVal = 0.0f, maxVal = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 6.0f * rand() / RAND_MAX;  // ReLU6-like: [0, 6]
        minVal = fminf(minVal, h_input[i]);
        maxVal = fmaxf(maxVal, h_input[i]);
    }
    
    // Asymmetric quantization parameters
    float scale = (maxVal - minVal) / 255.0f;
    uint8_t zero_point = static_cast<uint8_t>(roundf(-minVal / scale));
    
    printf("ReLU Activation Quantization:\n");
    printf("  Data range: [%.2f, %.2f]\n", minVal, maxVal);
    printf("  Scale: %.6f\n", scale);
    printf("  Zero point: %u\n\n", zero_point);
    
    // Compare with symmetric
    float symScale = maxVal / 127.0f;  // Must cover [0, max]
    printf("Comparison:\n");
    printf("  Asymmetric uses full [0,255] range\n");
    printf("  Symmetric wastes [-127,0] for ReLU outputs\n");
    printf("  Asymmetric scale: %.6f (finer resolution)\n", scale);
    printf("  Symmetric scale: %.6f (coarser)\n", symScale);
    printf("  Resolution improvement: %.1f%%\n", 100.0f * (symScale/scale - 1.0f));
    
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
