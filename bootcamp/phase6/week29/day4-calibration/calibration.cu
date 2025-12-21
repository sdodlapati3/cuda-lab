/**
 * Week 29, Day 4: Calibration
 * Finding optimal scale factors for quantization.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

// Calibration methods:
// 1. MinMax: Use observed min/max
// 2. Percentile: Use Nth percentile to clip outliers
// 3. Entropy (KL divergence): Minimize information loss
// 4. MSE: Minimize reconstruction error

float computeMSE(const float* original, const float* reconstructed, int n) {
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    return mse / n;
}

void quantizeWithScale(const float* input, float* output, int n, float scale) {
    for (int i = 0; i < n; i++) {
        int8_t q = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, 
                   roundf(input[i] / scale))));
        output[i] = q * scale;  // Dequantize
    }
}

int main() {
    printf("Week 29 Day 4: Calibration\n\n");
    
    printf("Calibration Methods:\n");
    printf("┌─────────────────┬────────────────────────────────────┐\n");
    printf("│ Method          │ Description                        │\n");
    printf("├─────────────────┼────────────────────────────────────┤\n");
    printf("│ MinMax          │ Use full observed range            │\n");
    printf("│ Percentile      │ Clip to Nth percentile             │\n");
    printf("│ Entropy (KL)    │ Minimize distribution divergence   │\n");
    printf("│ MSE             │ Minimize reconstruction error      │\n");
    printf("└─────────────────┴────────────────────────────────────┘\n\n");
    
    const int N = 10000;
    std::vector<float> data(N);
    std::vector<float> reconstructed(N);
    
    // Generate data with outliers (typical activation distribution)
    for (int i = 0; i < N; i++) {
        float u1 = rand() / (float)RAND_MAX;
        float u2 = rand() / (float)RAND_MAX;
        // Box-Muller for normal distribution
        data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
    }
    // Add some outliers
    data[100] = 10.0f;
    data[200] = -12.0f;
    
    // Method 1: MinMax
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());
    float minmaxScale = std::max(fabsf(minVal), fabsf(maxVal)) / 127.0f;
    
    // Method 2: Percentile (99.9%)
    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    float p001 = sorted[static_cast<int>(N * 0.001)];
    float p999 = sorted[static_cast<int>(N * 0.999)];
    float percentileScale = std::max(fabsf(p001), fabsf(p999)) / 127.0f;
    
    // Method 3: MSE search (simplified)
    float bestMSE = 1e10f, bestScale = minmaxScale;
    for (float s = percentileScale * 0.5f; s <= minmaxScale * 1.1f; s += minmaxScale * 0.01f) {
        quantizeWithScale(data.data(), reconstructed.data(), N, s);
        float mse = computeMSE(data.data(), reconstructed.data(), N);
        if (mse < bestMSE) {
            bestMSE = mse;
            bestScale = s;
        }
    }
    
    printf("Data Statistics:\n");
    printf("  N = %d samples\n", N);
    printf("  Min: %.2f, Max: %.2f\n", minVal, maxVal);
    printf("  99.9%% range: [%.2f, %.2f]\n\n", p001, p999);
    
    printf("Calibration Results:\n");
    quantizeWithScale(data.data(), reconstructed.data(), N, minmaxScale);
    printf("  MinMax:     scale=%.4f, MSE=%.6f\n", minmaxScale, 
           computeMSE(data.data(), reconstructed.data(), N));
    
    quantizeWithScale(data.data(), reconstructed.data(), N, percentileScale);
    printf("  Percentile: scale=%.4f, MSE=%.6f\n", percentileScale,
           computeMSE(data.data(), reconstructed.data(), N));
    
    printf("  MSE-opt:    scale=%.4f, MSE=%.6f\n", bestScale, bestMSE);
    
    printf("\nPercentile often best: clips outliers, preserves majority.\n");
    
    return 0;
}
