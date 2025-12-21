/*
 * Day 3: Quantization Basics
 * 
 * FP32 to FP16/INT8 conversion and performance comparison.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// FP32 to FP16 conversion kernel
__global__ void fp32_to_fp16(const float* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// FP16 to FP32 conversion kernel
__global__ void fp16_to_fp32(const half* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// Quantize FP32 to INT8
__global__ void quantize_int8(const float* input, int8_t* output,
                               float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale;
        val = fmaxf(-128.0f, fminf(127.0f, roundf(val)));
        output[idx] = (int8_t)val;
    }
}

// Dequantize INT8 to FP32
__global__ void dequantize_int8(const int8_t* input, float* output,
                                 float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (float)input[idx] * scale;
    }
}

// Vector add in FP32
__global__ void vadd_fp32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Vector add in FP16
__global__ void vadd_fp16(const half* a, const half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = __hadd(a[idx], b[idx]);
}

int main() {
    printf("=== Day 3: Quantization Basics ===\n\n");
    
    const int n = 10 * 1024 * 1024;  // 10M elements
    const int iterations = 100;
    
    // Allocate memory
    float *d_fp32_a, *d_fp32_b, *d_fp32_c, *d_fp32_restored;
    half *d_fp16_a, *d_fp16_b, *d_fp16_c;
    int8_t *d_int8;
    
    CHECK_CUDA(cudaMalloc(&d_fp32_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp32_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp32_c, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp32_restored, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp16_a, n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_fp16_b, n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_fp16_c, n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_int8, n * sizeof(int8_t)));
    
    // Initialize with random data
    float* h_data = new float[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_fp32_a, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp32_b, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // Convert to FP16
    fp32_to_fp16<<<grid, block>>>(d_fp32_a, d_fp16_a, n);
    fp32_to_fp16<<<grid, block>>>(d_fp32_b, d_fp16_b, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    printf("--- Memory Comparison ---\n");
    printf("FP32: %.2f MB\n", n * sizeof(float) / (1024.0f * 1024.0f));
    printf("FP16: %.2f MB (2x compression)\n", n * sizeof(half) / (1024.0f * 1024.0f));
    printf("INT8: %.2f MB (4x compression)\n\n", n * sizeof(int8_t) / (1024.0f * 1024.0f));
    
    // Benchmark FP32 vector add
    vadd_fp32<<<grid, block>>>(d_fp32_a, d_fp32_b, d_fp32_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        vadd_fp32<<<grid, block>>>(d_fp32_a, d_fp32_b, d_fp32_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));
    ms_fp32 /= iterations;
    
    // Benchmark FP16 vector add
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        vadd_fp16<<<grid, block>>>(d_fp16_a, d_fp16_b, d_fp16_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp16;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16, start, stop));
    ms_fp16 /= iterations;
    
    printf("--- Vector Add Performance ---\n");
    printf("FP32: %.3f ms (%.1f GB/s)\n", ms_fp32, 
           3.0f * n * sizeof(float) / (ms_fp32 * 1e6f));
    printf("FP16: %.3f ms (%.1f GB/s) - %.2fx faster\n", ms_fp16,
           3.0f * n * sizeof(half) / (ms_fp16 * 1e6f), ms_fp32 / ms_fp16);
    
    // Quantization accuracy test
    printf("\n--- Quantization Accuracy ---\n");
    
    // Find scale for INT8
    float maxVal = 2.0f;  // Known from data generation
    float scale = maxVal / 127.0f;
    
    quantize_int8<<<grid, block>>>(d_fp32_a, d_int8, scale, n);
    dequantize_int8<<<grid, block>>>(d_int8, d_fp32_restored, scale, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Calculate error
    float* h_original = new float[n];
    float* h_restored = new float[n];
    CHECK_CUDA(cudaMemcpy(h_original, d_fp32_a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_restored, d_fp32_restored, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxError = 0.0f, sumError = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(h_original[i] - h_restored[i]);
        maxError = fmaxf(maxError, err);
        sumError += err;
    }
    
    printf("INT8 quantization (scale=%.6f):\n", scale);
    printf("  Max error: %.6f\n", maxError);
    printf("  Mean error: %.6f\n", sumError / n);
    printf("  Relative error: %.4f%%\n", 100.0f * sumError / n / maxVal);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_fp32_a));
    CHECK_CUDA(cudaFree(d_fp32_b));
    CHECK_CUDA(cudaFree(d_fp32_c));
    CHECK_CUDA(cudaFree(d_fp32_restored));
    CHECK_CUDA(cudaFree(d_fp16_a));
    CHECK_CUDA(cudaFree(d_fp16_b));
    CHECK_CUDA(cudaFree(d_fp16_c));
    CHECK_CUDA(cudaFree(d_int8));
    delete[] h_data;
    delete[] h_original;
    delete[] h_restored;
    
    printf("\n=== Day 3 Complete ===\n");
    return 0;
}
