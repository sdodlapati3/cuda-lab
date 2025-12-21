/**
 * timeline_demo.cu - Demonstrate nsys timeline patterns
 * 
 * Profile this with: nsys profile -o timeline ./build/timeline_demo
 * View with: nsight-sys timeline.nsys-rep
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

// Simple kernels for timeline demo
__global__ void kernel_short(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

__global__ void kernel_long(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

// NVTX ranges for custom annotations (if available)
void annotate_section(const char* name) {
    // In real code, use nvtxRangePush(name) and nvtxRangePop()
    printf("=== %s ===\n", name);
}

int main() {
    printf("Timeline Demo for Nsight Systems\n");
    printf("Profile with: nsys profile -o timeline ./build/timeline_demo\n\n");
    
    const int N = 1 << 22;  // 4M
    const size_t bytes = N * sizeof(float);
    
    float *h_data, *d_data;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // ========== Section 1: Basic kernel launches ==========
    annotate_section("Section 1: Basic kernels (look for gaps)");
    
    cudaMalloc(&d_data, bytes);
    h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    
    // Multiple kernel launches - look for gaps
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    for (int i = 0; i < 5; i++) {
        kernel_short<<<blocks, threads>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    
    // ========== Section 2: Long kernels ==========
    annotate_section("Section 2: Long kernels (high GPU utilization)");
    
    for (int i = 0; i < 3; i++) {
        kernel_long<<<blocks, threads>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    
    // ========== Section 3: Memory transfer patterns ==========
    annotate_section("Section 3: Sync transfers (blocking)");
    
    // Synchronous transfers - blocks GPU
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        kernel_short<<<blocks, threads>>>(d_data, N);
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    
    // ========== Section 4: Async transfers with streams ==========
    annotate_section("Section 4: Async transfers with pinned memory");
    
    float *h_pinned, *d_data2;
    cudaMallocHost(&h_pinned, bytes);  // Pinned memory
    cudaMalloc(&d_data2, bytes);
    for (int i = 0; i < N; i++) h_pinned[i] = 1.0f;
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Overlapped operations
    int half_n = N / 2;
    int half_blocks = (half_n + threads - 1) / threads;
    
    cudaMemcpyAsync(d_data, h_pinned, bytes / 2, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data + half_n, h_pinned + half_n, bytes / 2, cudaMemcpyHostToDevice, stream2);
    
    kernel_short<<<half_blocks, threads, 0, stream1>>>(d_data, half_n);
    kernel_short<<<half_blocks, threads, 0, stream2>>>(d_data + half_n, half_n);
    
    cudaMemcpyAsync(h_pinned, d_data, bytes / 2, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_pinned + half_n, d_data + half_n, bytes / 2, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // ========== Section 5: Many small kernels ==========
    annotate_section("Section 5: Many small kernels (launch overhead)");
    
    const int SMALL_N = 1024;
    for (int i = 0; i < 100; i++) {
        kernel_short<<<4, 256>>>(d_data, SMALL_N);
    }
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_pinned);
    cudaFree(d_data);
    cudaFree(d_data2);
    free(h_data);
    
    printf("\n=== Timeline Analysis Tips ===\n\n");
    printf("1. Gaps between kernels? → Kernel fusion or streams\n");
    printf("2. Long H2D/D2H? → Pinned memory, async transfers\n");
    printf("3. Many tiny kernels? → Kernel fusion\n");
    printf("4. GPU idle while CPU works? → Overlap with streams\n");
    printf("5. Look for 'CUDA API' row to see blocking calls\n");
    
    return 0;
}
