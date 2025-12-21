/**
 * Week 29, Day 1: Quantization Basics
 * Understanding INT8 representation and quantization.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Quantization: FP32 → INT8
// q = round(x / scale) + zero_point
// x = (q - zero_point) * scale

__global__ void quantizeKernel(const float* input, int8_t* output, 
                                float scale, int8_t zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scaled = input[idx] / scale;
        int32_t quantized = __float2int_rn(scaled) + zero_point;
        // Clamp to INT8 range
        quantized = max(-128, min(127, quantized));
        output[idx] = static_cast<int8_t>(quantized);
    }
}

__global__ void dequantizeKernel(const int8_t* input, float* output,
                                  float scale, int8_t zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (static_cast<float>(input[idx]) - zero_point) * scale;
    }
}

void printQuantizationError(float* original, float* reconstructed, int n) {
    float maxError = 0.0f, sumError = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabsf(original[i] - reconstructed[i]);
        maxError = fmaxf(maxError, error);
        sumError += error;
    }
    printf("  Max error: %.6f\n", maxError);
    printf("  Mean error: %.6f\n", sumError / n);
}

int main() {
    printf("Week 29 Day 1: Quantization Basics\n\n");
    
    const int N = 1024;
    
    // Host data
    float* h_input = new float[N];
    float* h_output = new float[N];
    int8_t* h_quantized = new int8_t[N];
    
    // Generate test data: uniform in [-1, 1]
    for (int i = 0; i < N; i++) {
        h_input[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
    
    // Find min/max for scale calculation
    float minVal = h_input[0], maxVal = h_input[0];
    for (int i = 1; i < N; i++) {
        minVal = fminf(minVal, h_input[i]);
        maxVal = fmaxf(maxVal, h_input[i]);
    }
    
    // Symmetric quantization: scale = max(|min|, |max|) / 127
    float absMax = fmaxf(fabsf(minVal), fabsf(maxVal));
    float scale = absMax / 127.0f;
    int8_t zero_point = 0;  // Symmetric
    
    printf("Data Statistics:\n");
    printf("  Min: %.4f, Max: %.4f\n", minVal, maxVal);
    printf("  Scale: %.6f\n", scale);
    printf("  Zero point: %d\n\n", zero_point);
    
    // Device memory
    float *d_input, *d_output;
    int8_t *d_quantized;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_quantized, N * sizeof(int8_t));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Quantize
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    quantizeKernel<<<gridSize, blockSize>>>(d_input, d_quantized, scale, zero_point, N);
    
    // Dequantize
    dequantizeKernel<<<gridSize, blockSize>>>(d_quantized, d_output, scale, zero_point, N);
    
    // Copy back
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_quantized, d_quantized, N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    
    printf("Quantization Error (Symmetric):\n");
    printQuantizationError(h_input, h_output, N);
    
    printf("\nSample Values:\n");
    printf("  Original → Quantized → Reconstructed\n");
    for (int i = 0; i < 5; i++) {
        printf("  %.4f → %4d → %.4f\n", h_input[i], h_quantized[i], h_output[i]);
    }
    
    printf("\nMemory Savings:\n");
    printf("  FP32: %d bytes\n", N * 4);
    printf("  INT8: %d bytes\n", N * 1);
    printf("  Compression: 4×\n");
    
    // Cleanup
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_quantized);
    delete[] h_input; delete[] h_output; delete[] h_quantized;
    
    return 0;
}
