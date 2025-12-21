/**
 * fusion_demo.cu - Kernel fusion benefits
 * 
 * Learning objectives:
 * - Demonstrate kernel fusion
 * - Measure memory traffic reduction
 * - Understand when to fuse
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Unfused: Two separate kernels
__global__ void scale_kernel(float* out, const float* in, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = s * in[idx];
    }
}

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Fused: SAXPY in one kernel
__global__ void fused_saxpy(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// More complex fusion: z = sigmoid(a * x + b * y)
__global__ void unfused_linear(float* temp1, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp1[idx] = a * x[idx];
    }
}

__global__ void unfused_add_linear(float* temp2, const float* y, const float* temp1, 
                                    float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp2[idx] = temp1[idx] + b * y[idx];
    }
}

__global__ void unfused_sigmoid(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void fused_linear_sigmoid(float* out, const float* x, const float* y,
                                     float a, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a * x[idx] + b * y[idx];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

int main() {
    printf("=== Kernel Fusion Demo ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bw = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Peak bandwidth: %.0f GB/s\n\n", peak_bw);
    
    const int N = 1 << 24;
    const int TRIALS = 100;
    const float a = 2.0f, b = 3.0f;
    size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);
    
    float *d_x, *d_y, *d_out, *d_temp1, *d_temp2;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_temp1, bytes);
    cudaMalloc(&d_temp2, bytes);
    
    // Initialize with random data
    float *h_x = new float[N];
    float *h_y = new float[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = (rand() / (float)RAND_MAX) * 2 - 1;
        h_y[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid = (N + block - 1) / block;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Test 1: SAXPY (y = a*x + y) ===\n\n");
    printf("%-25s %-12s %-15s %-12s\n", 
           "Method", "Time(ms)", "Bandwidth(GB/s)", "Speedup");
    printf("----------------------------------------------------------------\n");
    
    // Unfused SAXPY: temp = a*x, then y = temp + y
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    scale_kernel<<<grid, block>>>(d_temp1, d_x, a, N);
    add_kernel<<<grid, block>>>(d_y, d_temp1, d_y, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
        scale_kernel<<<grid, block>>>(d_temp1, d_x, a, N);
        add_kernel<<<grid, block>>>(d_y, d_temp1, d_y, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float unfused_saxpy_ms = ms / TRIALS;
    
    // Memory: scale reads X, writes temp (2N), add reads temp+Y, writes Y (3N) = 5N
    float unfused_bw = 5.0f * bytes / unfused_saxpy_ms / 1e6;
    printf("%-25s %-12.3f %-15.1f %-12s\n", 
           "Unfused (2 kernels)", unfused_saxpy_ms, unfused_bw, "1.00x");
    
    // Fused SAXPY
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    fused_saxpy<<<grid, block>>>(d_y, d_x, a, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
        fused_saxpy<<<grid, block>>>(d_y, d_x, a, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float fused_saxpy_ms = ms / TRIALS;
    
    // Memory: reads X+Y, writes Y = 3N
    float fused_bw = 3.0f * bytes / fused_saxpy_ms / 1e6;
    printf("%-25s %-12.3f %-15.1f %-12.2fx\n", 
           "Fused (1 kernel)", fused_saxpy_ms, fused_bw, 
           unfused_saxpy_ms / fused_saxpy_ms);
    
    printf("\n=== Test 2: z = sigmoid(a*x + b*y) ===\n\n");
    printf("%-25s %-12s %-15s %-12s\n", 
           "Method", "Time(ms)", "Bandwidth(GB/s)", "Speedup");
    printf("----------------------------------------------------------------\n");
    
    // Unfused: 3 kernels
    unfused_linear<<<grid, block>>>(d_temp1, d_x, a, N);
    unfused_add_linear<<<grid, block>>>(d_temp2, d_y, d_temp1, b, N);
    unfused_sigmoid<<<grid, block>>>(d_out, d_temp2, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        unfused_linear<<<grid, block>>>(d_temp1, d_x, a, N);
        unfused_add_linear<<<grid, block>>>(d_temp2, d_y, d_temp1, b, N);
        unfused_sigmoid<<<grid, block>>>(d_out, d_temp2, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float unfused_ms = ms / TRIALS;
    
    // Memory: kernel1(2N) + kernel2(3N) + kernel3(2N) = 7N
    float unfused_complex_bw = 7.0f * bytes / unfused_ms / 1e6;
    printf("%-25s %-12.3f %-15.1f %-12s\n", 
           "Unfused (3 kernels)", unfused_ms, unfused_complex_bw, "1.00x");
    
    // Fused: 1 kernel
    fused_linear_sigmoid<<<grid, block>>>(d_out, d_x, d_y, a, b, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        fused_linear_sigmoid<<<grid, block>>>(d_out, d_x, d_y, a, b, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float fused_ms = ms / TRIALS;
    
    // Memory: reads X+Y, writes out = 3N
    float fused_complex_bw = 3.0f * bytes / fused_ms / 1e6;
    printf("%-25s %-12.3f %-15.1f %-12.2fx\n", 
           "Fused (1 kernel)", fused_ms, fused_complex_bw, 
           unfused_ms / fused_ms);
    
    printf("\n=== Analysis ===\n");
    printf("SAXPY fusion:\n");
    printf("  - Unfused: 5N bytes (temp storage = 2N extra)\n");
    printf("  - Fused: 3N bytes (40%% reduction)\n");
    printf("\nLinear+Sigmoid fusion:\n");
    printf("  - Unfused: 7N bytes (two temps = 4N extra)\n");
    printf("  - Fused: 3N bytes (57%% reduction)\n");
    printf("\nFusion benefits:\n");
    printf("  1. Fewer kernel launches (less overhead)\n");
    printf("  2. No intermediate storage (saves memory)\n");
    printf("  3. Reduced memory traffic (data reuse in registers)\n");
    printf("  4. Better cache utilization\n");
    printf("\nWhen NOT to fuse:\n");
    printf("  1. Intermediate results needed elsewhere\n");
    printf("  2. Operations have different parallelism requirements\n");
    printf("  3. Fused kernel becomes too complex (register pressure)\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    delete[] h_x;
    delete[] h_y;
    
    return 0;
}
