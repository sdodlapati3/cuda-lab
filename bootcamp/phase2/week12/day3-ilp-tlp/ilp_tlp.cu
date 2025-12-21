/**
 * ilp_tlp.cu - Demonstrate ILP vs TLP tradeoffs
 * 
 * Learning objectives:
 * - See how ILP improves throughput
 * - Understand TLP alternatives
 * - Find the sweet spot
 */

#include <cuda_runtime.h>
#include <cstdio>

// Version 1: Low ILP - one element per thread
__global__ void low_ilp_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

// Version 2: ILP=2 - two elements per thread
__global__ void ilp2_kernel(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx + 1 < n) {
        float a = in[idx];
        float b = in[idx + 1];
        
        // Independent computations - can execute in parallel
        float ra = a * 2.0f + 1.0f;
        float rb = b * 2.0f + 1.0f;
        
        out[idx] = ra;
        out[idx + 1] = rb;
    }
}

// Version 3: ILP=4 - four elements per thread
__global__ void ilp4_kernel(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // Load all at once
        float a = in[idx];
        float b = in[idx + 1];
        float c = in[idx + 2];
        float d = in[idx + 3];
        
        // All independent!
        float ra = a * 2.0f + 1.0f;
        float rb = b * 2.0f + 1.0f;
        float rc = c * 2.0f + 1.0f;
        float rd = d * 2.0f + 1.0f;
        
        out[idx] = ra;
        out[idx + 1] = rb;
        out[idx + 2] = rc;
        out[idx + 3] = rd;
    }
}

// Version 4: ILP=8 - eight elements per thread
__global__ void ilp8_kernel(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < n) {
        float a = in[idx], b = in[idx+1], c = in[idx+2], d = in[idx+3];
        float e = in[idx+4], f = in[idx+5], g = in[idx+6], h = in[idx+7];
        
        a = a * 2.0f + 1.0f; b = b * 2.0f + 1.0f;
        c = c * 2.0f + 1.0f; d = d * 2.0f + 1.0f;
        e = e * 2.0f + 1.0f; f = f * 2.0f + 1.0f;
        g = g * 2.0f + 1.0f; h = h * 2.0f + 1.0f;
        
        out[idx] = a; out[idx+1] = b; out[idx+2] = c; out[idx+3] = d;
        out[idx+4] = e; out[idx+5] = f; out[idx+6] = g; out[idx+7] = h;
    }
}

// Vectorized ILP - using float4
__global__ void vectorized_kernel(const float4* in, float4* out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = in[idx];
        val.x = val.x * 2.0f + 1.0f;
        val.y = val.y * 2.0f + 1.0f;
        val.z = val.z * 2.0f + 1.0f;
        val.w = val.w * 2.0f + 1.0f;
        out[idx] = val;
    }
}

template<typename Func>
float benchmark(Func kernel_launch, int iters = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel_launch();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        kernel_launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iters;
}

int main() {
    printf("=== ILP vs TLP Demo ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const int bytes = N * sizeof(float);
    const int block_size = 256;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    
    printf("Data size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);
    printf("%-25s | %10s | %12s | %12s\n", 
           "Version", "Time (ms)", "GB/s", "Speedup");
    printf("-------------------------------------------------------------------\n");
    
    // Test each version
    auto test_version = [&](const char* name, auto kernel, int elements_per_thread) {
        int threads_needed = N / elements_per_thread;
        int blocks = (threads_needed + block_size - 1) / block_size;
        
        float ms = benchmark([&](){ kernel<<<blocks, block_size>>>(d_in, d_out, N); });
        float gbps = (2.0 * bytes / 1e9) / (ms / 1000);  // Read + Write
        return ms;
    };
    
    int blocks1 = (N + block_size - 1) / block_size;
    float t1 = benchmark([&](){ low_ilp_kernel<<<blocks1, block_size>>>(d_in, d_out, N); });
    printf("%-25s | %10.4f | %12.2f | %11.2fx\n", "Low ILP (1 elem/thread)", t1, 2*bytes/1e9/(t1/1000), 1.0);
    
    int blocks2 = (N/2 + block_size - 1) / block_size;
    float t2 = benchmark([&](){ ilp2_kernel<<<blocks2, block_size>>>(d_in, d_out, N); });
    printf("%-25s | %10.4f | %12.2f | %11.2fx\n", "ILP=2 (2 elem/thread)", t2, 2*bytes/1e9/(t2/1000), t1/t2);
    
    int blocks4 = (N/4 + block_size - 1) / block_size;
    float t4 = benchmark([&](){ ilp4_kernel<<<blocks4, block_size>>>(d_in, d_out, N); });
    printf("%-25s | %10.4f | %12.2f | %11.2fx\n", "ILP=4 (4 elem/thread)", t4, 2*bytes/1e9/(t4/1000), t1/t4);
    
    int blocks8 = (N/8 + block_size - 1) / block_size;
    float t8 = benchmark([&](){ ilp8_kernel<<<blocks8, block_size>>>(d_in, d_out, N); });
    printf("%-25s | %10.4f | %12.2f | %11.2fx\n", "ILP=8 (8 elem/thread)", t8, 2*bytes/1e9/(t8/1000), t1/t8);
    
    int blocks_vec = (N/4 + block_size - 1) / block_size;
    float tv = benchmark([&](){ vectorized_kernel<<<blocks_vec, block_size>>>((float4*)d_in, (float4*)d_out, N/4); });
    printf("%-25s | %10.4f | %12.2f | %11.2fx\n", "Vectorized (float4)", tv, 2*bytes/1e9/(tv/1000), t1/tv);
    
    printf("\n=== Analysis ===\n\n");
    printf("Why does ILP help?\n");
    printf("1. More independent operations in flight\n");
    printf("2. Compiler can schedule them in parallel\n");
    printf("3. Fewer threads means less launch overhead\n\n");
    
    printf("Why can too much ILP hurt?\n");
    printf("1. Uses more registers per thread\n");
    printf("2. Can reduce occupancy\n");
    printf("3. Boundary handling gets complex\n\n");
    
    printf("=== TLP Effect: Block Size Sweep ===\n\n");
    printf("%-15s | %10s | %12s\n", "Block Size", "Time (ms)", "Rel. Perf");
    printf("--------------------------------------------\n");
    
    float baseline = 0;
    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        int blocks = (N + bs - 1) / bs;
        float ms = benchmark([&](){ low_ilp_kernel<<<blocks, bs>>>(d_in, d_out, N); }, 50);
        if (baseline == 0) baseline = ms;
        printf("%-15d | %10.4f | %11.2fx\n", bs, ms, baseline/ms);
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
