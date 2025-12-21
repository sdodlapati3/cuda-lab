/**
 * bandwidth_test.cu - Measure GPU memory bandwidth
 * 
 * Learning objectives:
 * - Measure theoretical vs achieved bandwidth
 * - Compare different measurement methods
 * - Establish your GPU's bandwidth ceiling
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// Simple copy kernel (read + write)
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Vector copy for better bandwidth
__global__ void copy_kernel_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Read-only kernel (measures read bandwidth)
__global__ void read_kernel(const float* __restrict__ in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += in[i];
    }
    
    // Write once to prevent optimization
    if (idx == 0) {
        *out = sum;
    }
}

// Write-only kernel (measures write bandwidth)
__global__ void write_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f;
    }
}

void print_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device: %s ===\n", prop.name);
    
    // Calculate theoretical peak
    float mem_clock_ghz = prop.memoryClockRate / 1e6;
    float bus_width_bytes = prop.memoryBusWidth / 8.0;
    float peak_bw = 2.0 * mem_clock_ghz * bus_width_bytes;
    
    printf("Memory Clock: %.2f GHz\n", mem_clock_ghz);
    printf("Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Theoretical Peak: %.0f GB/s\n", peak_bw);
    printf("Expected Achieved: %.0f-%.0f GB/s (75-85%%)\n", 
           peak_bw * 0.75, peak_bw * 0.85);
    printf("\n");
}

int main() {
    print_device_info();
    
    const size_t sizes[] = {
        1 << 20,   // 1 MB
        1 << 22,   // 4 MB
        1 << 24,   // 16 MB
        1 << 26,   // 64 MB
        1 << 28,   // 256 MB
    };
    const int TRIALS = 20;
    const int WARMUP = 5;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("=== Memory Bandwidth Measurements ===\n\n");
    printf("%-12s %-15s %-15s %-15s %-15s\n", 
           "Size (MB)", "cudaMemcpy", "Copy Kernel", "Vec4 Copy", "Read Only");
    printf("--------------------------------------------------------------------------\n");
    
    for (size_t size : sizes) {
        float *d_src, *d_dst;
        cudaMalloc(&d_src, size);
        cudaMalloc(&d_dst, size);
        cudaMemset(d_src, 0, size);
        
        int n = size / sizeof(float);
        int n4 = size / sizeof(float4);
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        int blocks4 = (n4 + threads - 1) / threads;
        
        float ms;
        float bw_memcpy, bw_copy, bw_vec4, bw_read;
        
        // 1. cudaMemcpy D2D
        for (int i = 0; i < WARMUP; i++) {
            cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(start);
        for (int i = 0; i < TRIALS; i++) {
            cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        bw_memcpy = (2.0 * size * TRIALS) / (ms / 1000) / 1e9;  // Read + Write
        
        // 2. Copy kernel
        for (int i = 0; i < WARMUP; i++) {
            copy_kernel<<<blocks, threads>>>(d_src, d_dst, n);
        }
        cudaEventRecord(start);
        for (int i = 0; i < TRIALS; i++) {
            copy_kernel<<<blocks, threads>>>(d_src, d_dst, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        bw_copy = (2.0 * size * TRIALS) / (ms / 1000) / 1e9;
        
        // 3. Vectorized copy kernel
        for (int i = 0; i < WARMUP; i++) {
            copy_kernel_vec4<<<blocks4, threads>>>((float4*)d_src, (float4*)d_dst, n4);
        }
        cudaEventRecord(start);
        for (int i = 0; i < TRIALS; i++) {
            copy_kernel_vec4<<<blocks4, threads>>>((float4*)d_src, (float4*)d_dst, n4);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        bw_vec4 = (2.0 * size * TRIALS) / (ms / 1000) / 1e9;
        
        // 4. Read-only kernel
        for (int i = 0; i < WARMUP; i++) {
            read_kernel<<<blocks, threads>>>(d_src, d_dst, n);
        }
        cudaEventRecord(start);
        for (int i = 0; i < TRIALS; i++) {
            read_kernel<<<blocks, threads>>>(d_src, d_dst, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        bw_read = (1.0 * size * TRIALS) / (ms / 1000) / 1e9;  // Read only
        
        printf("%-12.0f %-15.1f %-15.1f %-15.1f %-15.1f\n", 
               size / 1e6, bw_memcpy, bw_copy, bw_vec4, bw_read);
        
        cudaFree(d_src);
        cudaFree(d_dst);
    }
    
    printf("\n=== Key Observations ===\n");
    printf("1. cudaMemcpy is highly optimized - use as baseline\n");
    printf("2. Vec4 (float4) often matches or beats scalar copy\n");
    printf("3. Achieved bandwidth is 75-85%% of theoretical peak\n");
    printf("4. Larger sizes show stable bandwidth (GPU saturated)\n");
    printf("5. Small sizes may show lower bandwidth (latency-bound)\n");
    printf("\n");
    printf("Your practical bandwidth ceiling is the max achieved value.\n");
    printf("Use this to evaluate your kernel's bandwidth efficiency.\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
