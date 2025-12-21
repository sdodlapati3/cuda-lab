/**
 * zero_copy.cu - Mapped memory and unified addressing
 * 
 * Learning objectives:
 * - Use cudaHostAllocMapped for zero-copy
 * - Compare zero-copy vs explicit transfer
 * - Understand access pattern impact
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple read-once kernel
__global__ void read_once_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Kernel that reads same data multiple times
__global__ void read_multiple_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 10; i++) {
            sum += in[idx];
        }
        out[idx] = sum;
    }
}

// Random access pattern
__global__ void random_access_kernel(const float* in, const int* indices, 
                                      float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];
    }
}

int main() {
    printf("=== Zero-Copy Memory Demo ===\n\n");
    
    // Check if UVA is supported
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (!prop.unifiedAddressing) {
        printf("UVA not supported on this device.\n");
        return 1;
    }
    
    printf("Device: %s\n", prop.name);
    printf("Unified Virtual Addressing: %s\n\n", 
           prop.unifiedAddressing ? "Yes" : "No");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Zero-Copy vs Explicit Transfer (Single Read)
    // ========================================================================
    {
        printf("1. Single Read: Zero-Copy vs Explicit Transfer\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;  // 1M elements
        const size_t size = N * sizeof(float);
        
        // Allocate mapped memory
        float* h_mapped;
        cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
        for (int i = 0; i < N; i++) h_mapped[i] = 1.0f;
        
        // Device pointer for mapped memory (with UVA, same as h_mapped)
        float* d_mapped;
        cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
        
        // Regular device memory
        float* d_in, *d_out;
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);
        
        // Warmup
        read_once_kernel<<<(N+255)/256, 256>>>(d_mapped, d_out, N);
        cudaDeviceSynchronize();
        
        // Zero-copy approach
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            read_once_kernel<<<(N+255)/256, 256>>>(d_mapped, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float zerocopy_ms;
        cudaEventElapsedTime(&zerocopy_ms, start, stop);
        zerocopy_ms /= 10;
        
        // Explicit transfer approach
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            cudaMemcpy(d_in, h_mapped, size, cudaMemcpyHostToDevice);
            read_once_kernel<<<(N+255)/256, 256>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float transfer_ms;
        cudaEventElapsedTime(&transfer_ms, start, stop);
        transfer_ms /= 10;
        
        printf("   %d elements (single read per element)\n", N);
        printf("   Zero-copy:  %.3f ms\n", zerocopy_ms);
        printf("   Explicit:   %.3f ms\n", transfer_ms);
        printf("   Winner: %s\n\n", 
               zerocopy_ms < transfer_ms ? "Zero-copy" : "Explicit");
        
        cudaFreeHost(h_mapped);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // ========================================================================
    // Part 2: Multiple Reads (Zero-Copy Loses)
    // ========================================================================
    {
        printf("2. Multiple Reads: Zero-Copy vs Explicit\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;
        const size_t size = N * sizeof(float);
        
        float* h_mapped;
        cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
        for (int i = 0; i < N; i++) h_mapped[i] = 1.0f;
        
        float* d_mapped;
        cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
        
        float *d_in, *d_out;
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);
        
        // Zero-copy
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            read_multiple_kernel<<<(N+255)/256, 256>>>(d_mapped, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float zerocopy_ms;
        cudaEventElapsedTime(&zerocopy_ms, start, stop);
        zerocopy_ms /= 10;
        
        // Explicit
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            cudaMemcpy(d_in, h_mapped, size, cudaMemcpyHostToDevice);
            read_multiple_kernel<<<(N+255)/256, 256>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float transfer_ms;
        cudaEventElapsedTime(&transfer_ms, start, stop);
        transfer_ms /= 10;
        
        printf("   %d elements (10 reads per element)\n", N);
        printf("   Zero-copy:  %.3f ms\n", zerocopy_ms);
        printf("   Explicit:   %.3f ms\n", transfer_ms);
        printf("   Winner: %s (PCIe becomes bottleneck)\n\n",
               zerocopy_ms < transfer_ms ? "Zero-copy" : "Explicit");
        
        cudaFreeHost(h_mapped);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // ========================================================================
    // Part 3: Varying Data Sizes
    // ========================================================================
    {
        printf("3. Size Comparison: When Zero-Copy Wins\n");
        printf("─────────────────────────────────────────\n");
        
        size_t sizes[] = {1 << 12, 1 << 16, 1 << 20, 1 << 24};  // 4KB to 16MB
        
        for (auto size : sizes) {
            int n = size / sizeof(float);
            
            float* h_mapped;
            cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
            
            float* d_mapped;
            cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
            
            float *d_in, *d_out;
            cudaMalloc(&d_in, size);
            cudaMalloc(&d_out, size);
            
            // Zero-copy
            cudaEventRecord(start);
            for (int i = 0; i < 100; i++) {
                read_once_kernel<<<(n+255)/256, 256>>>(d_mapped, d_out, n);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float zc_ms;
            cudaEventElapsedTime(&zc_ms, start, stop);
            
            // Explicit
            cudaEventRecord(start);
            for (int i = 0; i < 100; i++) {
                cudaMemcpy(d_in, h_mapped, size, cudaMemcpyHostToDevice);
                read_once_kernel<<<(n+255)/256, 256>>>(d_in, d_out, n);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ex_ms;
            cudaEventElapsedTime(&ex_ms, start, stop);
            
            printf("   %8zu bytes: ZC=%.2fms, Explicit=%.2fms → %s\n",
                   size, zc_ms, ex_ms, zc_ms < ex_ms ? "ZeroCopy" : "Explicit");
            
            cudaFreeHost(h_mapped);
            cudaFree(d_in);
            cudaFree(d_out);
        }
        printf("\n");
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Zero-copy good for single-read, small data\n");
    printf("2. Explicit transfer wins for multiple reads\n");
    printf("3. Larger data favors explicit transfer\n");
    printf("4. Integrated GPUs benefit more from zero-copy\n");
    printf("5. Use for data larger than GPU memory\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
