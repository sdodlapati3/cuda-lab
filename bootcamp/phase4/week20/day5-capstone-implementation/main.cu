/**
 * Capstone Project: [Your Project Name]
 * 
 * [Brief description of what this project does]
 * 
 * Author: [Your Name]
 * Date: [Date]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// ============================================================================
// Error handling macro
// ============================================================================
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Configuration
// ============================================================================
#define BLOCK_SIZE 256

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel 1: [Name]
 * [Description of what this kernel does]
 */
__global__ void kernel1(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement your kernel logic
        output[idx] = input[idx];
    }
}

/**
 * Kernel 2: [Name] (if applicable)
 */
__global__ void kernel2(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement your kernel logic
        data[idx] *= 2.0f;
    }
}

// ============================================================================
// CPU Baseline (for comparison)
// ============================================================================
void cpuBaseline(float* output, const float* input, int n) {
    for (int i = 0; i < n; i++) {
        // TODO: Implement CPU version of your algorithm
        output[i] = input[i];
    }
}

// ============================================================================
// Verification
// ============================================================================
bool verify(const float* gpu, const float* cpu, int n, float tolerance = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu[i] - cpu[i]) > tolerance) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("=== Capstone Project: [Your Project Name] ===\n\n");
    
    // Parse arguments (if needed)
    int n = 1000000;  // Default problem size
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    printf("Problem size: %d elements\n\n", n);
    
    // Allocate host memory
    size_t size = n * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    
    // Initialize input data
    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Warm-up
    kernel1<<<grid, block>>>(d_output, d_input, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark GPU
    printf("Running GPU benchmark...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel1<<<grid, block>>>(d_output, d_input, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    gpuMs /= iterations;
    
    printf("  GPU time: %.3f ms\n", gpuMs);
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    // Benchmark CPU
    printf("Running CPU benchmark...\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuBaseline(h_output_cpu, h_input, n);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuMs = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();
    
    printf("  CPU time: %.3f ms\n", cpuMs);
    
    // Verify results
    printf("\nVerifying results...\n");
    bool correct = verify(h_output_gpu, h_output_cpu, n);
    printf("  Result: %s\n", correct ? "PASS" : "FAIL");
    
    // Print summary
    printf("\n=== Performance Summary ===\n");
    printf("GPU time:  %.3f ms\n", gpuMs);
    printf("CPU time:  %.3f ms\n", cpuMs);
    printf("Speedup:   %.2fx\n", cpuMs / gpuMs);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    
    return correct ? 0 : 1;
}
