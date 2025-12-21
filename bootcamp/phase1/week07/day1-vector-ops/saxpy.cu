/**
 * saxpy.cu - BLAS Level 1: Y = a*X + Y
 * 
 * Learning objectives:
 * - Implement standard BLAS operation
 * - Compare with cuBLAS
 * - Understand performance limits
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// Simple SAXPY kernel
__global__ void saxpy_kernel(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// Grid-stride SAXPY
__global__ void saxpy_stride(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

void verify(const float* x_orig, const float* y_orig, const float* y_result, 
            float a, int n) {
    int errors = 0;
    for (int i = 0; i < n && errors < 10; i++) {
        float expected = a * x_orig[i] + y_orig[i];
        if (fabsf(y_result[i] - expected) > 1e-4 * fabsf(expected)) {
            printf("Mismatch at %d: %.6f != %.6f\n", i, y_result[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification PASSED\n");
    }
}

int main() {
    printf("=== SAXPY: Y = a*X + Y ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bandwidth = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Theoretical peak bandwidth: %.0f GB/s\n\n", peak_bandwidth);
    
    const int N = 1 << 24;  // 16M elements
    const float a = 2.0f;
    const int TRIALS = 100;
    
    size_t bytes = N * sizeof(float);
    printf("Array size: %d elements (%.1f MB each)\n\n", N, bytes / 1e6);
    
    // Allocate host memory
    float* h_x = new float[N];
    float* h_y_orig = new float[N];
    float* h_y = new float[N];
    
    for (int i = 0; i < N; i++) {
        h_x[i] = rand() / (float)RAND_MAX;
        h_y_orig[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Benchmark Results ===\n\n");
    printf("%-20s %-12s %-15s %-12s\n", "Implementation", "Time (ms)", "Bandwidth (GB/s)", "Efficiency");
    printf("----------------------------------------------------------------------\n");
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    // My SAXPY - simple
    cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
    saxpy_kernel<<<num_blocks, block_size>>>(d_y, d_x, a, N);  // Warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
        saxpy_kernel<<<num_blocks, block_size>>>(d_y, d_x, a, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    // Bandwidth: 2 reads (X, Y) + 1 write (Y) = 3N floats
    float bandwidth = 3.0f * bytes / ms / 1e6;
    printf("%-20s %-12.3f %-15.1f %-12.1f%%\n", "My SAXPY (simple)", 
           ms, bandwidth, 100.0f * bandwidth / peak_bandwidth);
    
    // Verify
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    verify(h_x, h_y_orig, h_y, a, N);
    
    // My SAXPY - grid stride
    int sm_count = prop.multiProcessorCount;
    int stride_blocks = sm_count * 2;
    
    cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
    saxpy_stride<<<stride_blocks, block_size>>>(d_y, d_x, a, N);  // Warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
        saxpy_stride<<<stride_blocks, block_size>>>(d_y, d_x, a, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    bandwidth = 3.0f * bytes / ms / 1e6;
    printf("%-20s %-12.3f %-15.1f %-12.1f%%\n", "My SAXPY (stride)", 
           ms, bandwidth, 100.0f * bandwidth / peak_bandwidth);
    
    // cuBLAS SAXPY
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
    cublasSaxpy(handle, N, &a, d_x, 1, d_y, 1);  // Warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_y, h_y_orig, bytes, cudaMemcpyHostToDevice);
        cublasSaxpy(handle, N, &a, d_x, 1, d_y, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    bandwidth = 3.0f * bytes / ms / 1e6;
    printf("%-20s %-12.3f %-15.1f %-12.1f%%\n", "cuBLAS SAXPY", 
           ms, bandwidth, 100.0f * bandwidth / peak_bandwidth);
    
    // Verify cuBLAS
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    verify(h_x, h_y_orig, h_y, a, N);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y_orig;
    delete[] h_y;
    
    printf("\n=== Analysis ===\n");
    printf("SAXPY is memory-bound:\n");
    printf("  - 2 FLOPs per element (multiply, add)\n");
    printf("  - 12 bytes per element (2 reads + 1 write)\n");
    printf("  - Arithmetic intensity: 2/12 = 0.17 FLOPs/byte\n");
    printf("  - Peak: 2 TB/s × 0.17 = 340 GFLOPS (vs 19+ TFLOPS theoretical)\n");
    printf("\nWe're limited by memory bandwidth, not compute!\n");
    
    return 0;
}
